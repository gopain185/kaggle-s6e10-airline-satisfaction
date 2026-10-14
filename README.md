# Kaggle Playground S6E10: Predicting Airline Satisfaction

CV-first pipeline for the [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
competition (metric: ROC AUC). Best public LB so far: **0.96112**.

## What is here

- `src/common.py`: data loading, 5-fold stratified CV (seed 42), feature modes, original-dataset statistics,
  in-fold combo target encoding
- `src/train_lgbm.py`, `src/train_xgb.py`, `src/train_catboost.py`: GBMs that save out-of-fold and test predictions
- `src/knn_features.py`: kNN label mean against the original (CC0) dataset
- `src/ensemble.py`: logistic stack and hill-climb rank blend, both scored out-of-fold
- `notebooks/`, `kernels/`: builders and Kaggle notebooks for the GPU models (RealMLP)

## What mattered (OOF AUC)

| model | OOF AUC |
| --- | --- |
| LightGBM + original-dataset target statistics | 0.95992 |
| XGBoost + kNN feature + in-fold combo target encoding | 0.96035 |
| LightGBM + original statistics v2 + in-fold combo target encoding | 0.96044 |
| RealMLP (public recipe, credited in the notebook) + original statistics | 0.96117 |
| hill-climb blend of RealMLP, LightGBM and XGBoost | 0.96139 |

Models are chosen by CV, not by the public leaderboard (the public LB moves about ±0.0005 on noise).

## Reproduce

Download the competition data to `data/` and the original dataset to `data/original/data.csv`, then run e.g.
`python src/train_lgbm.py --mode orig2 --te` and `python src/ensemble.py lgbm_orig2_te xgb_knn_te --out blend`.
Data and predictions are not committed (competition rules).
