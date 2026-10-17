"""XGBoost (hist, native categoricals) on the same folds and feature views as the other models."""

import argparse

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import pandas as pd

from common import combo_keys, features, fold_target_encoding, folds, load, save

PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist", enable_categorical=True,
    learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
    reg_lambda=2.0, max_cat_to_onehot=1, n_estimators=6000, early_stopping_rounds=300,
    n_jobs=8, random_state=42,
)


def main(name: str, mode: str, te: bool, route: bool = False) -> None:
    train, test, y = load()
    X, X_test = features(train, mode), features(test, mode)
    keys, keys_test = combo_keys(train), combo_keys(test)
    if route:  # same route features as LightGBM: in-fold TE of the route plus its passenger profile
        from train_lgbm import route_profile
        keys["route"], keys_test["route"] = train["Flight Distance"].astype(str), test["Flight Distance"].astype(str)
        P, P_test = route_profile(train, test)
        X, X_test = pd.concat([X, P], axis=1), pd.concat([X_test, P_test], axis=1)
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds(y)):
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test
        if te:  # fitted on the training fold only
            t_tr, t_va, t_te = fold_target_encoding(keys.iloc[tr], y[tr], [keys.iloc[va], keys_test])
            X_tr, X_va, X_te = (pd.concat([a, b], axis=1) for a, b in ((X_tr, t_tr), (X_va, t_va), (X_te, t_te)))
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
        oof[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / 5
        print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f} at {model.best_iteration} trees", flush=True)
    save(name, oof, test_pred, test["id"], y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="orig", choices=["basic", "cat", "allcat", "orig", "orig2", "knn"])
    parser.add_argument("--te", action="store_true", help="add in-fold target encoding of TE_COMBOS")
    parser.add_argument("--route", action="store_true", help="add route (Flight Distance) TE and profile; needs --te")
    parser.add_argument("--name")
    args = parser.parse_args()
    main(args.name or f"xgb_{args.mode}{'_te' if args.te else ''}", args.mode, args.te, args.route)
