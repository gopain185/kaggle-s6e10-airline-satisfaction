"""Builds s6e10-realmlp-v2.ipynb: the public RealMLP recipe plus original-dataset target statistics.

Preprocessing and hyper-parameters follow "PS|S6|E10: RealMLP · PyTabKit" by Vladimir Demidov (yekenot),
credited in the notebook. My addition is the block of original-dataset statistics from ../src/common.py.
"""

import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import common  # noqa: E402

intro = """# S6E10 · RealMLP + original-dataset statistics

**Credit:** preprocessing and RealMLP hyper-parameters follow
[PS|S6|E10: RealMLP · PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit) by Vladimir Demidov:
every numeric column is also used as a categorical, count encoding, four interaction keys with in-fold target
encoding, and a tuned RealMLP configuration.

**What I add:** satisfaction rates looked up in the public CC0 dataset this competition was synthesised from
(per column value, and per exact rating vector / service profile). Those rows are disjoint from the competition
data, so the lookup cannot leak competition labels. Same 5-fold `StratifiedKFold(seed=42)` as my tree models,
so the OOF predictions can be blended with them.
"""

setup = '''!pip install -q pytabkit
import glob, os, time, warnings
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from pytabkit import RealMLP_TD_Classifier
warnings.filterwarnings("ignore")

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]

train = pd.read_csv(find("train.csv")); test = pd.read_csv(find("test.csv"))
orig = pd.read_csv(find("data.csv"))
TARGET = "satisfaction"
y = train.pop(TARGET).astype(int).to_numpy()
orig[TARGET] = orig[TARGET].astype(int)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(train.shape, test.shape, orig.shape, DEVICE)
'''

orig_block = (
    f"CATEGORICAL = {common.CATEGORICAL!r}\nRATINGS = {common.RATINGS!r}\n"
    "RATING_KEY = RATINGS\nSERVICE_KEY = [\"Customer Type\", \"Type of Travel\", \"Class\"] + RATINGS\n\n"
    + inspect.getsource(common.original_stats)
    + '''
O_tr, O_te = original_stats(train, orig), original_stats(test, orig)
O_tr, O_te = O_tr.drop(columns=["orig_key_full_mean", "orig_key_full_n"]), O_te.drop(columns=["orig_key_full_mean", "orig_key_full_n"])
O_med = O_tr.median()
O_tr, O_te = O_tr.fillna(O_med).astype("float32"), O_te.fillna(O_med).astype("float32")
print(O_tr.shape)
'''
)

prep = '''X, X_test = train.drop(columns=["id"]), test.drop(columns=["id"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
COMBOS = [("Class", "Type of Travel", "Gender"), ("Class", "Type of Travel", "Age"),
          ("Food and drink", "Cleanliness"), ("Flight Distance", "Departure/Arrival time convenient")]
combo_names = ["_".join(c) + "_" for c in COMBOS]

def engineer(df, ref):
    """ref is the training frame: category codes and counts are learned from it only."""
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].fillna("missing")
    for c in num_cols:
        out[c] = out[c].fillna(0.0)
    for c in num_cols:
        uniques = pd.Index(ref[c].fillna(0.0).unique())
        out[f"{c}_cat_"] = pd.Categorical(uniques.get_indexer(out[c]))
    for c in cat_cols + ["Leg room service"]:
        out[f"_{c}_count"] = out[c].map(ref[c].fillna("missing").value_counts()).fillna(0).astype("int32")
    for cols, name in zip(COMBOS, combo_names):
        key = lambda d: d[list(cols)].fillna(0).astype(str).agg("_".join, axis=1)
        uniques = pd.Index(key(ref).unique())
        out[name] = pd.Categorical(uniques.get_indexer(key(out)))
    return out

Xe, Xe_test = engineer(X, X), engineer(X_test, X)
Xe, Xe_test = pd.concat([Xe, O_tr], axis=1), pd.concat([Xe_test, O_te], axis=1)
print(Xe.shape, Xe_test.shape)
'''

params = '''params = dict(
    random_state=42, verbosity=0, val_metric_name="1-auc_ovr", device=DEVICE,
    n_ens=8, n_epochs=3, batch_size=288, use_early_stopping=False,
    lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15", first_layer_lr_factor=0.25,
    embedding_size=5, max_one_hot_cat_size=2, hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="pow5",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
)
'''

train_cell = '''oof, test_pred = np.zeros(len(y)), np.zeros(len(Xe_test))
drop_after_te = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xe, y)):
    t = time.time()
    X_tr, X_va, X_te = Xe.iloc[tr].copy(), Xe.iloc[va].copy(), Xe_test.copy()
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=42)  # training fold only
    names = [f"_{c}TE" for c in combo_names]
    X_tr[names] = enc.fit_transform(X_tr[combo_names].astype(str), y[tr])
    X_va[names] = enc.transform(X_va[combo_names].astype(str))
    X_te[names] = enc.transform(X_te[combo_names].astype(str))
    X_tr, X_va, X_te = (d.drop(columns=drop_after_te) for d in (X_tr, X_va, X_te))
    model = RealMLP_TD_Classifier(**params)
    model.fit(X_tr, y[tr], X_va, y[va])
    oof[va] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_te)[:, 1] / 5
    print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}  ({time.time()-t:.0f}s)", flush=True)
    torch.cuda.empty_cache()
print(f"OOF AUC {roc_auc_score(y, oof):.6f}")
np.save("realmlp_v2_oof.npy", oof); np.save("realmlp_v2_test.npy", test_pred)
pd.DataFrame({"id": test["id"], TARGET: test_pred}).to_csv("submission.csv", index=False, float_format="%.6f")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    return {**c, "outputs": [], "execution_count": None} if kind == "code" else c


def build() -> None:
    cells = [cell("markdown", intro), cell("code", setup),
             cell("markdown", "## Original-dataset statistics (my addition)"), cell("code", orig_block),
             cell("markdown", "## Preprocessing (public recipe)"), cell("code", prep),
             cell("markdown", "## RealMLP configuration (public recipe)"), cell("code", params),
             cell("markdown", "## 5-fold CV with in-fold target encoding"), cell("code", train_cell)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE.parent / "kernels" / "realmlp_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s6e10-realmlp-v2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out / "s6e10-realmlp-v2.ipynb")


if __name__ == "__main__":
    build()
