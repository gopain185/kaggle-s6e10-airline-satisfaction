"""Logistic regression on one-hot features and their pairwise crosses.

A linear model is a weak learner here on its own, but it makes very different errors from the tree and
neural models, which is what a blend needs. Every column is treated as categorical (numerics are
quantile-binned), and all pairs of columns are crossed, so the model sees second-order interactions
the way a factorisation machine would.
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from common import CATEGORICAL, RATINGS, folds, load, save

BINNED = {"Age": 16, "Flight Distance": 16, "Departure Delay in Minutes": 8, "Arrival Delay in Minutes": 8}


def tokens(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every column as a string token; bin edges come from the training frame only."""
    tr, te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for c in CATEGORICAL + RATINGS:
        tr[c], te[c] = train[c].astype(str), test[c].astype(str)
    for c, q in BINNED.items():
        edges = np.unique(np.nanquantile(train[c], np.linspace(0, 1, q + 1)[1:-1]))
        tr[c] = np.digitize(train[c].fillna(-1), edges).astype(str)
        te[c] = np.digitize(test[c].fillna(-1), edges).astype(str)
    for a, b in combinations(list(tr.columns[: len(CATEGORICAL + RATINGS) + len(BINNED)]), 2):
        tr[f"{a}*{b}"] = tr[a] + "|" + tr[b]
        te[f"{a}*{b}"] = te[a] + "|" + te[b]
    return tr, te


def main(name: str, C: float) -> None:
    train, test, y = load()
    T, T_test = tokens(train, test)
    print("token columns:", T.shape[1], flush=True)
    oof, test_pred = np.zeros(len(y)), np.zeros(len(test))
    for k, (tr, va) in enumerate(folds(y)):
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32)
        X_tr = enc.fit_transform(T.iloc[tr])  # vocabulary from the training fold only
        X_va, X_te = enc.transform(T.iloc[va]), enc.transform(T_test)
        model = LogisticRegression(C=C, max_iter=400, solver="lbfgs")
        model.fit(sparse.csr_matrix(X_tr), y[tr])
        oof[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / 5
        print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}  ({X_tr.shape[1]} features)", flush=True)
    save(name, oof, test_pred, test["id"], y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="logreg_cross")
    parser.add_argument("--C", type=float, default=0.1)
    args = parser.parse_args()
    main(args.name, args.C)
