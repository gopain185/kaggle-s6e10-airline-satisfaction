"""LightGBM with native categoricals, 5-fold stratified CV, early stopping on each fold's validation split."""

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import pandas as pd

from common import RATINGS, combo_keys, features, fold_target_encoding, folds, load, save

_ENV = __import__("os").environ
PARAMS = dict(
    objective="binary", learning_rate=float(_ENV.get("LR", "0.05")), num_leaves=int(_ENV.get("LEAVES", "63")),
    min_child_samples=int(_ENV.get("MIN_CHILD", "50")), feature_fraction=float(_ENV.get("FF", "0.7")), bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_estimators=5000, verbose=-1, random_state=42,
)


def route_profile(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flight Distance behaves like a route ID (idea from the S6E10 discussion "Flight Distance behaves like
    a route ID"). Describe each route by label-free statistics of its passengers, pooled over train and test."""
    both = pd.concat([train, test])
    g = both.groupby("Flight Distance")
    prof = pd.DataFrame({
        "route_n": g.size(),
        "route_business": g["Class"].apply(lambda s: (s == "Business").mean()),
        "route_business_travel": g["Type of Travel"].apply(lambda s: (s == "Business travel").mean()),
        "route_loyal": g["Customer Type"].apply(lambda s: (s == "Loyal Customer").mean()),
        "route_age": g["Age"].mean(),
        "route_delay": g["Departure Delay in Minutes"].mean(),
        **{f"route_{c}": g[c].mean() for c in RATINGS[:6]},
    })
    return (prof.reindex(df["Flight Distance"]).set_axis(df.index) for df in (train, test))


def main(name: str, mode: str, te: bool, route: bool = False) -> None:
    train, test, y = load()
    X, X_test = features(train, mode), features(test, mode)
    keys, keys_test = combo_keys(train), combo_keys(test)
    if route:  # in-fold target encoding of the route alone, plus its passenger profile
        keys["route"], keys_test["route"] = train["Flight Distance"].astype(str), test["Flight Distance"].astype(str)
        P, P_test = route_profile(train, test)
        X, X_test = pd.concat([X, P], axis=1), pd.concat([X_test, P_test], axis=1)
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds(y)):
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test
        if te:  # fitted on the training fold only
            t_tr, t_va, t_te = fold_target_encoding(keys.iloc[tr], y[tr], [keys.iloc[va], keys_test])
            X_tr, X_va, X_te = (pd.concat([a, b], axis=1) for a, b in ((X_tr, t_tr), (X_va, t_va), (X_te, t_te)))
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X_tr, y[tr], eval_X=(X_va,), eval_y=(y[va],), eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / 5
        print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f} at {model.best_iteration_} trees")
    save(name, oof, test_pred, test["id"], y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="basic", choices=["basic", "cat", "allcat", "orig", "orig2", "knn"])
    parser.add_argument("--te", action="store_true", help="add in-fold target encoding of TE_COMBOS")
    parser.add_argument("--route", action="store_true", help="add route (Flight Distance) TE and profile; needs --te")
    parser.add_argument("--name")
    args = parser.parse_args()
    main(args.name or f"lgbm_{args.mode}{'_te' if args.te else ''}", args.mode, args.te, args.route)
