"""Final-day S6E10 blends from saved OOFs: hill-climb over every model, logistic stacks, and a weight grid
around the blend34 core. Writes candidates to submissions/final_*.csv and a CSV of OOF AUCs."""

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import OOF, SUB, TARGET, folds, load  # noqa: E402

train, test, y = load()
names = sorted(p.name[:-8] for p in OOF.glob("*_oof.npy"))
P = {n: np.load(OOF / f"{n}_oof.npy") for n in names}
T = {n: np.load(OOF / f"{n}_test.npy") for n in names}
names = [n for n in names if P[n].shape == (len(y),) and T[n].shape == (len(test),)]
R = {n: rankdata(P[n]) / len(y) for n in names}
RT = {n: rankdata(T[n]) / len(test) for n in names}
rows, outs = [], {}


def add(tag, desc, oof, tst):
    auc = roc_auc_score(y, oof)
    rows.append((tag, round(auc, 6), desc))
    outs[tag] = tst
    print(f"{tag:<28} {auc:.6f}  {desc}", flush=True)


def mix(w):
    ks = [k for k, v in w.items() if v]
    return sum(w[k] * R[k] for k in ks), sum(w[k] * RT[k] for k in ks)


def fmt(w):
    return ", ".join(f"{k} {v:g}" for k, v in w.items() if v)


# 1. greedy hill-climb with replacement over all models
counts = dict.fromkeys(names, 0)
cur, best = np.zeros(len(y)), -1.0
for _ in range(60):
    sc = {n: roc_auc_score(y, cur + R[n]) for n in names}
    n = max(sc, key=sc.get)
    if sc[n] <= best + 1e-7:
        break
    best, counts[n], cur = sc[n], counts[n] + 1, cur + R[n]
w_hc = {k: v for k, v in counts.items() if v}
add("final_hc_all", "hill-climb rank blend over all models: " + fmt(w_hc), *mix(w_hc))

# 2. logistic stacks over all models (OOF via inner CV)
def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


Z = np.column_stack([logit(P[n]) for n in names])
ZT = np.column_stack([logit(T[n]) for n in names])
for C in (1.0, 0.01):
    oof = np.zeros(len(y))
    for tr, va in folds(y):
        oof[va] = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
    full = LogisticRegression(C=C, max_iter=2000).fit(Z, y)
    add(f"final_stack_all_C{C:g}", f"logistic stack of all {len(names)} models, C={C:g}", oof, full.predict_proba(ZT)[:, 1])

# 3. weight grid around the blend34 core
core = {"realmlp_v7": 2, "realmlp_v6": 1, "realmlp_v5": 1, "cat_knn_route": 1, "lgbm_orig2_te": 1, "xgb_knn_route": 1}
grid = {"realmlp_v7": [2, 3], "realmlp_orig_bag": [0, 0.5, 1], "tabm_v2": [0, 0.25, 0.5],
        "cat_orig2_route": [0, 0.5], "lgbm_knn_route_initv7": [0, 0.5, 1], "tabr_v1": [0, 0.25]}
for vals in itertools.product(*grid.values()):
    w = dict(core, **dict(zip(grid, vals)))
    o, t = mix(w)
    add("grid_" + hashlib.md5(fmt(w).encode()).hexdigest()[:6], "rank mix: " + fmt(w), o, t)

df = pd.DataFrame(rows, columns=["tag", "oof_auc", "desc"]).sort_values("oof_auc", ascending=False)
df.to_csv(ROOT / "logs_final_mix.csv", index=False)
for tag in df.tag.head(15):
    pd.DataFrame({"id": test["id"], TARGET: outs[tag]}).to_csv(SUB / f"{tag}.csv", index=False, float_format="%.6f")
print(df.head(15).to_string())
