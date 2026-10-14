"""Nearest-neighbour features against the original dataset, cached to data/knn_*.parquet.

Synthetic rows are perturbed copies of original rows, so the labels of the closest original passengers
are informative even where no exact key match exists. Original rows are disjoint from the competition
data, so computing this once outside the CV loop does not leak competition labels.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import CATEGORICAL, DATA, RATINGS, TARGET, load, load_original

KS = (5, 25, 100)


def _matrix(df: pd.DataFrame) -> np.ndarray:
    parts = [df[RATINGS].to_numpy(float)]
    for col in CATEGORICAL:
        parts.append(pd.get_dummies(df[col]).reindex(columns=sorted(df[col].unique()), fill_value=0).to_numpy(float) * 2.0)
    parts.append(df[["Age"]].to_numpy(float) / 10.0)
    parts.append(np.log1p(df[["Flight Distance"]].to_numpy(float)))
    return np.hstack(parts)


def build() -> None:
    train, test, _ = load()
    orig = load_original()
    cats = {c: sorted(orig[c].unique()) for c in CATEGORICAL}
    for df in (train, test, orig):
        for c in CATEGORICAL:
            df[c] = pd.Categorical(df[c], categories=cats[c])
    X_orig = _matrix(orig)
    y_orig = orig[TARGET].to_numpy()
    nn = NearestNeighbors(n_neighbors=max(KS), n_jobs=-1).fit(X_orig)
    for name, df in (("train", train), ("test", test)):
        dist, idx = nn.kneighbors(_matrix(df))
        out = {"knn_dist1": dist[:, 0], f"knn_dist{KS[1]}": dist[:, KS[1] - 1]}
        for k in KS:
            out[f"knn_mean{k}"] = y_orig[idx[:, :k]].mean(axis=1)
        # distance-weighted vote over the closest 25
        w = 1.0 / (dist[:, :KS[1]] + 0.5)
        out["knn_wmean25"] = (y_orig[idx[:, :KS[1]]] * w).sum(axis=1) / w.sum(axis=1)
        pd.DataFrame(out).to_parquet(DATA / f"knn_{name}.parquet")
        print(name, "done", flush=True)


if __name__ == "__main__":
    build()
