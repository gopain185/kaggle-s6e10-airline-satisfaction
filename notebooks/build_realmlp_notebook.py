"""Builds s6e10-realmlp.ipynb from the feature cells of s6e10-mlp.ipynb plus a RealMLP (pytabkit) model."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

intro = """# S6E10 · RealMLP on original-data + kNN features

[RealMLP](https://github.com/dholzmueller/pytabkit) is a tuned-by-default tabular MLP (robust scaling,
periodic-linear numeric embeddings, careful learning-rate schedules). Here it gets the same features and the
same 5-fold `StratifiedKFold(seed=42)` as my gradient-boosted models, so its OOF predictions can be blended:

1. **Original-dataset target statistics**: satisfaction rate among *original* (CC0) passengers sharing a
   column value, a pair of key values, or the full rating vector. Original rows are disjoint from the
   competition data, so this cannot leak competition labels.
2. **Nearest original neighbours**: label mean of the closest original passengers.
3. **Frequency encoding** (label-free).

Each fold early-stops on its own validation split (best epoch by validation AUC).
"""

model = '''from pytabkit import RealMLP_TD_Classifier

CAT_COLS = CATEGORICAL + RATINGS
num = X.columns.difference(CAT_COLS, sort=False)
usable = [c for c in num if X[c].notna().any() and X[c].nunique() > 1]
med = X[usable].median()
Xm = pd.concat([X[CAT_COLS].astype(str).astype("category"), X[usable].fillna(med).astype("float32")], axis=1)
Xt = pd.concat([X_test[CAT_COLS].astype(str).astype("category"), X_test[usable].fillna(med).astype("float32")], axis=1)
for c in CAT_COLS:  # identical category sets in train and test
    cats = Xm[c].cat.categories.union(Xt[c].cat.categories)
    Xm[c], Xt[c] = Xm[c].cat.set_categories(cats), Xt[c].cat.set_categories(cats)
print(Xm.shape, Xt.shape)
'''

train_cell = '''oof, test_pred = np.zeros(len(y)), np.zeros(len(Xt))
folds = StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
for k, (tr, va) in enumerate(folds):
    t = time.time()
    clf = RealMLP_TD_Classifier(
        device=DEVICE, random_state=k, n_epochs=N_EPOCHS, batch_size=1024,
        val_metric_name="1-auc_ovr", verbosity=0,
    )
    clf.fit(Xm.iloc[tr], y[tr], X_val=Xm.iloc[va], y_val=y[va], cat_col_names=CAT_COLS)
    oof[va] = clf.predict_proba(Xm.iloc[va])[:, 1]
    test_pred += clf.predict_proba(Xt)[:, 1] / 5
    print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}  ({time.time()-t:.0f}s)", flush=True)
print(f"OOF AUC {roc_auc_score(y, oof):.6f}")
'''

save_cell = '''np.save("realmlp_knn_oof.npy", oof)
np.save("realmlp_knn_test.npy", test_pred)
pd.DataFrame({"id": test["id"], "satisfaction": test_pred}).to_csv("submission.csv", index=False, float_format="%.6f")
print(os.listdir("."))
'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True), "outputs": [], "execution_count": None}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build() -> None:
    base = json.loads((HERE / "s6e10-mlp.ipynb").read_text(encoding="utf-8"))
    setup, features_md, features = base["cells"][1], base["cells"][2], base["cells"][3]
    cells = [md(intro), code("!pip install -q pytabkit\nN_EPOCHS = 64"), setup, features_md, features,
             md("## RealMLP: same folds, early stopping on validation AUC"), code(model), code(train_cell), code(save_cell)]
    base["cells"] = cells
    (HERE / "s6e10-realmlp.ipynb").write_text(json.dumps(base, indent=1), encoding="utf-8")
    print("wrote", HERE / "s6e10-realmlp.ipynb")


if __name__ == "__main__":
    build()
