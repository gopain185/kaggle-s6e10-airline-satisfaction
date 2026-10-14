"""Builds s6e10-mlp.ipynb: a GPU MLP with categorical embeddings, run on Kaggle.

Feature code is copied from ../src/common.py and ../src/knn_features.py at build time so the notebook is
self-contained and uses exactly the same features and folds as the local tree models.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"


def _strip_header(code: str) -> str:
    """Drop the local path constants; the notebook defines its own."""
    lines = code.splitlines()
    keep = [l for l in lines if not l.startswith(("ROOT =", "DATA =", "OOF =", "SUB =", "from pathlib", "from __future__"))]
    return "\n".join(keep)


md_intro = """# S6E10 · Embedding MLP on original-data features

A neural network to complement gradient-boosted trees. It uses the same feature set as my tree models:

1. **Original-dataset target statistics.** The competition data is synthesised from a public CC0 survey
   dataset (129,880 rows). For every column, pair of key columns and full rating vector, I look up the
   satisfaction rate among *original* passengers with the same values. The original rows are disjoint
   from the competition data, so this cannot leak competition labels.
2. **Nearest original neighbours.** Synthetic rows are perturbed copies of original rows, so the labels of
   the closest original passengers stay informative where no exact match exists (one feature alone has AUC ≈ 0.95).
3. **Frequency encoding** of numeric values and of the full rating vector (label-free).

The MLP embeds every low-cardinality column (the 0–5 ratings and the 4 categoricals) and feeds standardised
numeric features alongside. Same 5-fold `StratifiedKFold(seed=42)` as my local models, so the OOF predictions
can be blended with them.
"""

setup = '''import glob, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]

COMP = Path(find("sample_submission.csv")).parent
ORIG_CSV = find("data.csv")
DATA = COMP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(COMP, ORIG_CSV, DEVICE)
'''


def build() -> None:
    common = _strip_header((SRC / "common.py").read_text(encoding="utf-8"))
    common = common.replace('pd.read_csv(DATA / "original" / "data.csv")', "pd.read_csv(ORIG_CSV)")
    common = common.replace('pd.read_parquet(DATA / f"knn_{split}.parquet")', "KNN[split]")

    knn = (SRC / "knn_features.py").read_text(encoding="utf-8")
    knn_body = knn.split("KS = (5, 25, 100)", 1)[1].split('if __name__ == "__main__":')[0]
    knn_body = "KS = (5, 25, 100)" + knn_body.replace(
        'pd.DataFrame(out).to_parquet(DATA / f"knn_{name}.parquet")', "KNN[name] = pd.DataFrame(out)"
    )

    features_cell = common + "\n\nKNN = {}\n" + knn_body + "\n\nt = time.time(); build(); print(f'knn {time.time()-t:.0f}s')\n"
    features_cell += '''
train, test, y = load()
X, X_test = features(train, "knn"), features(test, "knn")
print(X.shape, X_test.shape)
'''

    prep = '''EMB_COLS = CATEGORICAL + RATINGS
NUM_COLS = [c for c in X.columns if c not in EMB_COLS]

def encode(X, X_test):
    codes, cards = [], []
    for c in EMB_COLS:
        cats = pd.Categorical(pd.concat([X[c].astype(str), X_test[c].astype(str)])).categories
        codes.append((pd.Categorical(X[c].astype(str), cats).codes, pd.Categorical(X_test[c].astype(str), cats).codes))
        cards.append(len(cats))
    C = np.stack([a for a, _ in codes], 1).astype(np.int64)
    CT = np.stack([b for _, b in codes], 1).astype(np.int64)
    N, NT = X[NUM_COLS].astype(float).to_numpy(), X_test[NUM_COLS].astype(float).to_numpy()
    skewed = [i for i, c in enumerate(NUM_COLS) if c in ("Age", "Flight Distance", "Departure Delay in Minutes",
              "Arrival Delay in Minutes", "delay_total") or c.startswith("freq_") or c.endswith("_n")]
    N[:, skewed], NT[:, skewed] = np.log1p(np.nan_to_num(N[:, skewed])), np.log1p(np.nan_to_num(NT[:, skewed]))
    usable = ~np.isnan(N).all(0) & (np.nanstd(N, 0) > 0)  # drop all-missing / constant columns
    N, NT = N[:, usable], NT[:, usable]
    med = np.nanmedian(N, 0)
    N, NT = np.where(np.isnan(N), med, N), np.where(np.isnan(NT), med, NT)
    mu, sd = N.mean(0), N.std(0) + 1e-6
    return C, CT, ((N - mu) / sd).astype(np.float32), ((NT - mu) / sd).astype(np.float32), cards

C, CT, N, NT, CARDS = encode(X, X_test)
print(C.shape, N.shape, CARDS)
'''

    model = '''class Net(nn.Module):
    def __init__(self, cards, n_num, emb=8, hidden=(512, 256, 128), drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, emb) for c in cards])
        layers, d = [], emb * len(cards) + n_num
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, c, x):
        e = torch.cat([m(c[:, i]) for i, m in enumerate(self.embs)], 1)
        return self.mlp(torch.cat([e, x], 1)).squeeze(1)


def predict(net, c, x, bs=16384):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(c), bs):
            out.append(torch.sigmoid(net(c[i:i+bs], x[i:i+bs])).float().cpu().numpy())
    return np.concatenate(out)


def fit_fold(tr, va, epochs=12, bs=2048, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    net = Net(CARDS, N.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    c_tr, x_tr = torch.tensor(C[tr], device=DEVICE), torch.tensor(N[tr], device=DEVICE)
    y_tr = torch.tensor(y[tr], dtype=torch.float32, device=DEVICE)
    c_va, x_va = torch.tensor(C[va], device=DEVICE), torch.tensor(N[va], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=DEVICE)
        for i in range(0, len(tr), bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss_fn(net(c_tr[b], x_tr[b]), y_tr[b]).backward()
            opt.step(); sched.step()
        auc = roc_auc_score(y[va], predict(net, c_va, x_va))
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best
'''

    train_cell = '''SEEDS = [0, 1, 2]
oof, test_pred = np.zeros(len(y)), np.zeros(len(CT))
c_te, x_te = torch.tensor(CT, device=DEVICE), torch.tensor(NT, device=DEVICE)
folds = StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
for k, (tr, va) in enumerate(folds):
    t = time.time()
    for s in SEEDS:
        net, best = fit_fold(tr, va, seed=s)
        oof[va] += predict(net, torch.tensor(C[va], device=DEVICE), torch.tensor(N[va], device=DEVICE)) / len(SEEDS)
        test_pred += predict(net, c_te, x_te) / (5 * len(SEEDS))
    print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}  ({time.time()-t:.0f}s)")
print(f"OOF AUC {roc_auc_score(y, oof):.6f}")
'''

    save_cell = '''np.save("mlp_knn_oof.npy", oof)
np.save("mlp_knn_test.npy", test_pred)
pd.DataFrame({"id": test["id"], "satisfaction": test_pred}).to_csv("submission.csv", index=False, float_format="%.6f")
print(os.listdir("."))
'''

    cells = [("markdown", md_intro), ("code", setup), ("markdown", "## Features (shared with my tree models)"),
             ("code", features_cell), ("markdown", "## Encoding for the network"), ("code", prep),
             ("markdown", "## Model: embeddings + MLP, one-cycle schedule, best epoch by validation AUC"),
             ("code", model), ("markdown", "## 5-fold CV × 3 seeds"), ("code", train_cell), ("code", save_cell)]
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (HERE / "s6e10-mlp.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", HERE / "s6e10-mlp.ipynb")


if __name__ == "__main__":
    build()
