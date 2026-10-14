"""CatBoost with ordered target statistics on categoricals; same folds as the other models."""

import argparse

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from common import CATEGORICAL, RATINGS, features, folds, load, save


def main(name: str, mode: str, depth: int) -> None:
    train, test, y = load()
    X, X_test = features(train, mode), features(test, mode)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c in cat_cols:  # CatBoost wants plain strings/ints for categoricals
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    test_pool = Pool(X_test, cat_features=cat_cols)
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds(y)):
        model = CatBoostClassifier(
            iterations=4000, learning_rate=float(__import__("os").environ.get("LR", "0.08")), depth=depth, l2_leaf_reg=3, eval_metric="AUC",
            od_type="Iter", od_wait=300, random_seed=42, thread_count=12, verbose=0, border_count=128,
        )
        model.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_cols), eval_set=Pool(X.iloc[va], y[va], cat_features=cat_cols))
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(test_pool)[:, 1] / 5
        print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f} at {model.get_best_iteration()} trees", flush=True)
    save(name, oof, test_pred, test["id"], y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="basic", choices=["basic", "cat", "allcat", "orig", "orig2", "knn"])
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--name")
    args = parser.parse_args()
    main(args.name or f"cat_{args.mode}_d{args.depth}", args.mode, args.depth)
