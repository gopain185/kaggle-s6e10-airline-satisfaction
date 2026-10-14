# How well does 5-fold CV track the public LB here? Ten submissions, one table

I logged the out-of-fold AUC of every submission I made on day one next to its public LB score, to see how much
of a CV gain actually shows up on the leaderboard in this competition.

**Setup:** 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the full training set. All models,
encoders and blends are fitted inside the folds, and blend weights are chosen on out-of-fold predictions.

| # | submission | OOF AUC | public LB | Δ OOF vs previous | Δ LB vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | LightGBM + original-dataset target statistics | 0.95984 | 0.95947 | | |
| 2 | + pair target encoding, frequency encoding, lr 0.02 | 0.96018 | 0.95946 | +0.00034 | −0.00001 |
| 3 | hill-climb rank blend of 3 GBMs | 0.96027 | 0.95958 | +0.00009 | +0.00012 |
| 4 | blend of 4 GBMs incl. a kNN-to-original feature | 0.96042 | 0.95973 | +0.00015 | +0.00015 |
| 5 | different 4-GBM blend | 0.96042 | 0.95976 | 0.00000 | +0.00003 |
| 6 | + a deep LightGBM | 0.96045 | 0.95980 | +0.00003 | +0.00004 |
| 7 | logistic stack: RealMLP + 6 GBMs | 0.96134 | 0.96107 | +0.00089 | +0.00127 |
| 8 | + LightGBM with in-fold combo target encoding | 0.96137 | 0.96110 | +0.00003 | +0.00003 |
| 9 | + second RealMLP (kNN feature, other seed) | 0.96139 | 0.96112 | +0.00002 | +0.00002 |
| 10 | hill-climb blend, positive weights only | 0.96139 | 0.96110 | 0.00000 | −0.00002 |

## What I take from it

- **The ranking transfers.** Across the ten submissions, OOF and LB move in the same direction almost every
  time, and the one large CV jump (adding RealMLP, +0.0009 OOF) produced the one large LB jump (+0.0013).
- **Small CV gains do not.** Submission 2 gained +0.00034 OOF and nothing on the LB. On a public split of this
  size, differences below a few 1e-4 are inside the noise, so I would not choose between two models on the
  public LB when their OOF scores are that close.
- **The LB sits about 0.0003 below OOF** for every submission, a stable offset rather than a sign of overfitting
  the folds.
- **Diversity beat tuning.** Six GBM variants together added about +0.0006 over a single LightGBM. One
  neural model with a different inductive bias added more than all of that.

## What helped, in order of impact

1. **A tabular neural net next to the GBMs.** RealMLP from `pytabkit`, following the recipe in
   [yekenot's public notebook](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit) (numerics also
   fed as categoricals, count encoding), plus target statistics from the original dataset.
2. **The original dataset** ([Airline Satisfaction, CC0](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k)):
   per-value satisfaction rates and a k-nearest-neighbour label mean against the original rows. The kNN
   feature alone scores about 0.95 AUC.
3. **In-fold target encoding of column combinations** (idea from the same notebook). It has to be fitted
   inside each fold; fitting it on the full training set leaks the target and inflates CV.

## What did not help

- Treating the 0–5 survey ratings as categoricals in LightGBM.
- An MLP trained only on my engineered features. It reached 0.9586 OOF and added nothing to the blend.

I plan to keep selecting final submissions by OOF AUC rather than by public LB. Happy to hear if your CV and
LB disagree more than this.
