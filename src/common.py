"""Shared loading, feature and cross-validation helpers for Playground S6E10 (airline satisfaction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OOF = ROOT / "oof"
SUB = ROOT / "submissions"
TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42

CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
    "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
    "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
]


def load() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train.pop(TARGET).astype(int).to_numpy()
    return train, test, y


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns with categoricals as pandas category dtype (GBM libraries handle them natively)."""
    out = df.drop(columns=["id"]).copy()
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out


def features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Feature views compared in experiments.

    basic   : raw columns, only the four text columns categorical
    cat     : also treat every 0-5 rating as categorical; synthetic generators often break the ordinal
              relationship, so per-level splits can beat threshold splits
    allcat  : additionally bin Age and Flight Distance as categories, plus summary statistics of ratings
    orig    : basic + target statistics looked up from the original dataset (see original_stats)
    """
    out = basic_features(df)
    if mode == "basic":
        return out
    if mode == "orig":
        return pd.concat([out, original_stats(df, load_original())], axis=1)
    if mode in ("orig2", "knn"):
        orig = load_original()
        parts = [out, original_stats(df, orig), pair_stats(df, orig), frequency(df)]
        if mode == "knn":
            split = "train" if df["id"].iloc[0] == 0 else "test"
            parts.append(pd.read_parquet(DATA / f"knn_{split}.parquet").set_axis(df.index))
        return pd.concat(parts, axis=1)
    ratings = df[RATINGS]
    out["rating_mean"] = ratings.mean(axis=1)
    out["rating_min"] = ratings.min(axis=1)
    out["n_zero_ratings"] = (ratings == 0).sum(axis=1)
    out["delay_total"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"].fillna(0)
    out["arrival_missing"] = df["Arrival Delay in Minutes"].isna().astype(int)
    for col in RATINGS:
        out[col] = df[col].astype("category")
    if mode == "allcat":
        out["Age_cat"] = df["Age"].astype("category")
        out["Distance_cat"] = (df["Flight Distance"] // 50).astype("category")
    return out


def load_original() -> pd.DataFrame:
    """The public CC0 dataset the competition data was synthesised from (129,880 survey rows)."""
    orig = pd.read_csv(DATA / "original" / "data.csv")
    orig[TARGET] = orig[TARGET].astype(int)
    return orig


RATING_KEY = RATINGS
SERVICE_KEY = ["Customer Type", "Type of Travel", "Class"] + RATINGS


def original_stats(df: pd.DataFrame, orig: pd.DataFrame, smoothing: float = 20.0) -> pd.DataFrame:
    """Target statistics from the original dataset, looked up for each competition row.

    The original rows are disjoint from the competition data, so these features cannot leak the
    competition labels and are safe to compute once, outside the CV loop.
    """
    prior = orig[TARGET].mean()
    out = pd.DataFrame(index=df.index)
    raw_cols = CATEGORICAL + RATINGS + ["Age", "Flight Distance", "Departure Delay in Minutes"]
    for col in raw_cols:
        g = orig.groupby(col)[TARGET].agg(["sum", "count"])
        te = (g["sum"] + smoothing * prior) / (g["count"] + smoothing)
        out[f"orig_te_{col}"] = df[col].map(te).astype(float).fillna(prior)
    for name, key in [("ratings", RATING_KEY), ("service", SERVICE_KEY), ("full", [c for c in orig.columns if c != TARGET])]:
        g = orig.groupby(key, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
        g = g.rename(columns={"mean": f"orig_key_{name}_mean", "count": f"orig_key_{name}_n"})
        merged = df[key].merge(g, on=key, how="left")
        out[f"orig_key_{name}_mean"] = merged[f"orig_key_{name}_mean"].to_numpy()
        out[f"orig_key_{name}_n"] = merged[f"orig_key_{name}_n"].fillna(0).to_numpy()
    return out


PAIR_COLS = [
    "Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
    "Inflight entertainment", "Seat comfort", "Leg room service",
]


def pair_stats(df: pd.DataFrame, orig: pd.DataFrame, smoothing: float = 20.0) -> pd.DataFrame:
    """Original-dataset target mean for every pair of the most important columns."""
    prior = orig[TARGET].mean()
    out = {}
    for i, a in enumerate(PAIR_COLS):
        for b in PAIR_COLS[i + 1:]:
            g = orig.groupby([a, b])[TARGET].agg(["sum", "count"])
            te = ((g["sum"] + smoothing * prior) / (g["count"] + smoothing)).rename("te").reset_index()
            out[f"orig_pair_{a}__{b}"] = df[[a, b]].merge(te, on=[a, b], how="left")["te"].fillna(prior).to_numpy()
    return pd.DataFrame(out, index=df.index)


_FREQ_CACHE: dict[str, pd.Series] = {}


def frequency(df: pd.DataFrame) -> pd.DataFrame:
    """How often each value occurs across train + test (label-free, so safe outside the CV loop)."""
    if not _FREQ_CACHE:
        both = pd.concat([pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")])
        for col in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
            _FREQ_CACHE[col] = both[col].value_counts()
        _FREQ_CACHE["ratings_key"] = both[RATINGS].astype(str).agg("".join, axis=1).value_counts()
    out = pd.DataFrame(index=df.index)
    for col, counts in _FREQ_CACHE.items():
        values = df[RATINGS].astype(str).agg("".join, axis=1) if col == "ratings_key" else df[col]
        out[f"freq_{col}"] = values.map(counts).fillna(0).to_numpy()
    return out


# Combinations whose in-fold target encoding helps; idea credited to the public notebook
# "PS|S6|E10: RealMLP · PyTabKit" by Vladimir Demidov (yekenot).
TE_COMBOS = [
    ("Class", "Type of Travel", "Gender"),
    ("Class", "Type of Travel", "Age"),
    ("Food and drink", "Cleanliness"),
    ("Flight Distance", "Departure/Arrival time convenient"),
]


def combo_keys(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"_".join(c): df[list(c)].astype(str).agg("_".join, axis=1) for c in TE_COMBOS}, index=df.index)


def fold_target_encoding(keys_tr: pd.DataFrame, y_tr: np.ndarray, others: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Target-encode combo keys using only the training fold.

    sklearn's TargetEncoder cross-fits on the training rows themselves, so the encoded training column never
    sees its own label; validation and test rows are encoded with the encoder fitted on the full training fold.
    """
    from sklearn.preprocessing import TargetEncoder

    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
    names = [f"te_{c}" for c in keys_tr.columns]
    out = [pd.DataFrame(enc.fit_transform(keys_tr, y_tr), columns=names, index=keys_tr.index)]
    out += [pd.DataFrame(enc.transform(k), columns=names, index=k.index) for k in others]
    return out


def folds(y: np.ndarray):
    return StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)


def save(name: str, oof: np.ndarray, test_pred: np.ndarray, test_ids: pd.Series, y: np.ndarray) -> float:
    OOF.mkdir(exist_ok=True)
    SUB.mkdir(exist_ok=True)
    auc = roc_auc_score(y, oof)
    np.save(OOF / f"{name}_oof.npy", oof)
    np.save(OOF / f"{name}_test.npy", test_pred)
    pd.DataFrame({"id": test_ids, TARGET: test_pred}).to_csv(SUB / f"{name}.csv", index=False)
    print(f"{name}: OOF AUC {auc:.6f}")
    return auc
