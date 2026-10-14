"""Blend saved OOF predictions.

Two blends are compared, both scored out-of-fold so the reported CV is honest:
- rank average with weights from a non-negative logistic stack fitted inside the CV folds
- a plain hill-climb on rank-averaged weights (Caruana-style selection with replacement)
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OOF, SUB, TARGET, folds, load


def _rank(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def _logit(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(P / (1 - P))


def stack_cv(P: np.ndarray, T: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions of a logistic stacker on logit-transformed model outputs, plus test predictions."""
    Z, ZT = _logit(P), _logit(T)
    oof = np.zeros(len(y))
    for tr, va in folds(y):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], y[tr])
        oof[va] = m.predict_proba(Z[va])[:, 1]
    full = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
    return oof, full.predict_proba(ZT)[:, 1], full.coef_.ravel()


def hill_climb(R: np.ndarray, y: np.ndarray, steps: int = 40) -> np.ndarray:
    counts = np.zeros(R.shape[1])
    best = -1.0
    for _ in range(steps):
        scores = [roc_auc_score(y, R @ (counts + np.eye(len(counts))[j])) for j in range(len(counts))]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-7:
            break
        best, counts[j] = scores[j], counts[j] + 1
    return counts / counts.sum()


def main(names: list[str], out: str) -> None:
    train, test, y = load()
    P = np.column_stack([np.load(OOF / f"{n}_oof.npy") for n in names])
    T = np.column_stack([np.load(OOF / f"{n}_test.npy") for n in names])
    for n, col in zip(names, P.T):
        print(f"{n:<22} {roc_auc_score(y, col):.6f}")

    stack_oof, stack_test, coef = stack_cv(P, T, y)
    print(f"{'logistic stack (OOF)':<22} {roc_auc_score(y, stack_oof):.6f}  coef {np.round(coef, 3)}")

    R, RT = np.column_stack([_rank(c) for c in P.T]), np.column_stack([_rank(c) for c in T.T])
    w = hill_climb(R, y)
    print(f"{'hill-climb rank blend':<22} {roc_auc_score(y, R @ w):.6f}  weights {np.round(w, 3)}")

    SUB.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], TARGET: RT @ w}).to_csv(SUB / f"{out}.csv", index=False, float_format="%.6f")
    pd.DataFrame({"id": test["id"], TARGET: stack_test}).to_csv(SUB / f"{out}_stack.csv", index=False, float_format="%.6f")
    print(f"wrote submissions/{out}.csv (hill-climb) and submissions/{out}_stack.csv (logistic stack)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--out", default="blend")
    args = parser.parse_args()
    main(args.models, args.out)
