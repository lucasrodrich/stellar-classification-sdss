"""
Tier-2 experiment: augment training with the ORIGINAL real SDSS17 dataset to
counter the concept drift diagnosed in docs/diagnostics.md.

Hypothesis: the competition data is synthetic (a generator's imperfect copy of a
real dataset). The real SDSS17 rows carry the clean, undistorted label rule the
synthetic TEST set approximates, so mixing them into training anchors the decision
boundary to the truth and should generalise to the test set better.

Important: the test set is synthetic, so adding real data may LOWER the
synthetic-CV yet RAISE the leaderboard. This experiment is therefore
**leaderboard-gated**, not CV-gated — we submit and let the real test judge.

External data: fedesoriano/stellar-classification-dataset-sdss17 (data/external/).
It lacks the two synthetic-only categoricals (added as missing) and uses -9999 as a
missing-value sentinel (cleaned to NaN).

Run:  python src/train_external.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

import ensemble as E

BASELINE_CV = 0.96805
BANDS = ["u", "g", "r", "i", "z"]


def load_external():
    ext = pd.read_csv("data/external/star_classification.csv")
    ext = ext[["alpha", "delta", *BANDS, "redshift", "class"]].copy()
    for c in BANDS:                       # SDSS missing-value sentinel -> NaN
        ext[c] = ext[c].replace(-9999.0, np.nan)
    ext["spectral_type"] = np.nan         # synthetic-only cols: mark as missing
    ext["galaxy_population"] = np.nan
    return ext


def main():
    train = E.add_features(pd.read_csv("data/train.csv"))
    test = E.add_features(pd.read_csv("data/test.csv"))
    ext = E.add_features(load_external())

    y = train["class"].map(E.CLASS_TO_INT).values
    y_ext = ext["class"].map(E.CLASS_TO_INT).values
    print(f"synthetic train: {len(train):,} rows | external: {len(ext):,} rows")

    # consistent categorical dtype across all three (categories from competition train)
    for c in E.CAT_COLS:
        cats = train[c].astype("category").cat.categories
        train[c] = pd.Categorical(train[c], categories=cats)
        test[c] = pd.Categorical(test[c], categories=cats)
        ext[c] = pd.Categorical(ext[c], categories=cats)  # all-missing

    feats = [c for c in train.columns if c not in ["id", "class"]]
    X, Xt, Xe = train[feats], test[feats], ext[feats]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(Xt), 3))

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        # each fold trains on synthetic-train-fold + ALL real data;
        # validates on the held-out SYNTHETIC fold (representative of the test).
        X_tr = pd.concat([X.iloc[tr], Xe], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr], y_ext])
        model = lgb.LGBMClassifier(**E.LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(E.EARLY_STOP, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(Xt) / 5
        print(f"  fold {fold}: synthetic-val acc = "
              f"{accuracy_score(y[va], oof[va].argmax(1)):.5f}")

    cv = accuracy_score(y, oof.argmax(1))
    print("=" * 60)
    print(f"CV with external data (synthetic-val): {cv:.5f}")
    print(f"Baseline (synthetic only)            : {BASELINE_CV:.5f}")
    print(f"Delta (note: LB is the real judge)   : {cv - BASELINE_CV:+.5f}")
    print("=" * 60)

    pd.DataFrame({
        "id": pd.read_csv("data/test.csv")["id"],
        "class": [E.INT_TO_CLASS[i] for i in test_pred.argmax(1)],
    }).to_csv("submission_external.csv", index=False)
    print("Saved submission_external.csv")


if __name__ == "__main__":
    main()
