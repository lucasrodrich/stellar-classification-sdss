"""
Kaggle Playground S6E6 - Stellar Class: 3-MODEL ENSEMBLE
(LightGBM + XGBoost + CatBoost, blended)

This is the next step after src/stellar.py. The idea in one sentence:
    "Run the baseline three times with three DIFFERENT model families,
     then average their probabilities."

Why it helps: LightGBM, XGBoost and CatBoost are all gradient-boosted trees,
but they grow trees differently and therefore make DIFFERENT mistakes. When you
average three estimators that err in different directions, the errors partly
cancel out and the blended answer is steadier and usually more accurate than any
single model.

What it does (same four jobs as the baseline, just x3):
  1. LOAD train.csv / test.csv.
  2. ENGINEER the exact same color-index features as the baseline (so the
     comparison is apples-to-apples).
  3. TRAIN each of the 3 models with 5-fold cross-validation, collecting
     out-of-fold (OOF) probabilities so we can score each model honestly.
  4. BLEND the three by averaging, then write submission_ensemble.csv.

Requirements:  pip install pandas numpy scikit-learn lightgbm xgboost catboost
Run from the project root:  python src/ensemble.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ----------------------------------------------------------------------
# Constants (named once, reused everywhere -> no "magic" values scattered around)
# ----------------------------------------------------------------------
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
CAT_COLS = ["spectral_type", "galaxy_population"]
N_SPLITS = 5
SEED = 42
EARLY_STOP = 80          # stop a model when its watch-score stalls for this many rounds
N_CLASSES = len(CLASSES)


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (identical to src/stellar.py on purpose)
# ----------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with color-index and redshift features added."""
    df = df.copy()
    bands = ["u", "g", "r", "i", "z"]

    # adjacent color indices (the classic, most useful ones)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    # wider color indices (skip a band)
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]

    # simple summaries across the 5 bands
    df["mag_mean"] = df[bands].mean(axis=1)
    df["mag_std"] = df[bands].std(axis=1)
    df["mag_range"] = df[bands].max(axis=1) - df[bands].min(axis=1)

    # redshift interactions (redshift is the strongest physical signal)
    df["z_x_ug"] = df["redshift"] * df["u_g"]
    df["z_x_gr"] = df["redshift"] * df["g_r"]
    df["z_x_ri"] = df["redshift"] * df["r_i"]
    df["redshift_log"] = np.log1p(df["redshift"].clip(lower=0))
    return df


def prepare_data():
    """Load both CSVs, engineer features, and return X, y, Xt, feature list.

    Categorical columns are left as the pandas 'category' dtype here, which
    LightGBM and XGBoost both accept natively. CatBoost needs them as plain
    strings, so we convert just-in-time inside its fit function.
    """
    train = add_features(pd.read_csv("data/train.csv"))
    test = add_features(pd.read_csv("data/test.csv"))

    y = train["class"].map(CLASS_TO_INT).values

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        # force test to use the SAME category list/order as train
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    features = [c for c in train.columns if c not in ["id", "class"]]
    X = train[features]
    Xt = test[features]
    print(f"Prepared {len(features)} features on {len(X):,} training rows\n")
    return X, y, Xt, features


# ----------------------------------------------------------------------
# Model definitions: each "fit_*" function builds ONE model and trains it on a
# single fold. They differ only in each library's API for categoricals + early
# stopping. Everything else (the CV loop) is shared in cross_validate().
# ----------------------------------------------------------------------
LGB_PARAMS = dict(
    objective="multiclass", num_class=N_CLASSES,
    learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=2.0, reg_alpha=0.5, min_child_samples=40,
    n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1,
)

XGB_PARAMS = dict(
    objective="multi:softprob", num_class=N_CLASSES,
    learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.7,
    reg_lambda=2.0, reg_alpha=0.5,
    n_estimators=3000, tree_method="hist", enable_categorical=True,
    eval_metric="mlogloss", early_stopping_rounds=EARLY_STOP,
    random_state=SEED, n_jobs=-1,
)

CAT_PARAMS = dict(
    loss_function="MultiClass", learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, iterations=3000, early_stopping_rounds=EARLY_STOP,
    random_seed=SEED, thread_count=-1, verbose=False,
)


def fit_lgb(X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return model


def fit_xgb(X_tr, y_tr, X_va, y_va):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def fit_cat(X_tr, y_tr, X_va, y_va):
    # CatBoost wants categorical columns as strings, not the 'category' dtype.
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)
    model = CatBoostClassifier(**CAT_PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=CAT_COLS, verbose=False)
    return model


def cat_predict_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost needs string categoricals at predict time too."""
    X = X.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


# ----------------------------------------------------------------------
# 3. The shared cross-validation engine (DRY: written once, used 3x)
# ----------------------------------------------------------------------
def cross_validate(name, fit_one, X, y, Xt, predict_prep=None):
    """Train `fit_one` over N_SPLITS folds. Return (oof_probs, test_probs).

    name         : label for printing (e.g. "LightGBM")
    fit_one      : a fit_* function that trains one model on one fold
    predict_prep : optional transform applied to a frame before predict_proba
                   (CatBoost uses it to stringify categoricals)
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), N_CLASSES))
    test_probs = np.zeros((len(Xt), N_CLASSES))
    prep = predict_prep if predict_prep is not None else (lambda f: f)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = fit_one(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx])
        oof[va_idx] = model.predict_proba(prep(X.iloc[va_idx]))
        test_probs += model.predict_proba(prep(Xt)) / N_SPLITS
        acc = accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        print(f"  [{name}] fold {fold}: accuracy = {acc:.5f}")

    cv = accuracy_score(y, oof.argmax(1))
    print(f"  [{name}] overall CV accuracy = {cv:.5f}\n")
    return oof, test_probs


# ----------------------------------------------------------------------
# Weight search: find convex blend weights (summing to 1) that maximize the
# OOF accuracy. Equal weights are just one point in this search; a weaker model
# (here CatBoost) should get less weight, not an automatic 1/3.
# ----------------------------------------------------------------------
def search_blend_weights(oofs, y, step=0.05):
    """Grid-search non-negative weights summing to 1 over the model OOF probs.

    Returns (best_weights, best_accuracy). With 3 models there are only 2 free
    weights, so a coarse grid is plenty and stays interpretable.
    """
    best_w, best_acc = None, -1.0
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w_lgb in grid:
        for w_xgb in grid:
            w_cat = 1.0 - w_lgb - w_xgb
            if w_cat < -1e-9:
                continue  # weights must stay non-negative and sum to 1
            blend = w_lgb * oofs[0] + w_xgb * oofs[1] + w_cat * oofs[2]
            acc = accuracy_score(y, blend.argmax(1))
            if acc > best_acc:
                best_acc = acc
                best_w = (round(w_lgb, 3), round(w_xgb, 3), round(w_cat, 3))
    return best_w, best_acc


# ----------------------------------------------------------------------
# 4. Blend the three models and write the submission
# ----------------------------------------------------------------------
def main():
    X, y, Xt, _ = prepare_data()

    print("Training LightGBM ...")
    oof_lgb, test_lgb = cross_validate("LightGBM", fit_lgb, X, y, Xt)

    print("Training XGBoost ...")
    oof_xgb, test_xgb = cross_validate("XGBoost", fit_xgb, X, y, Xt)

    print("Training CatBoost ...")
    oof_cat, test_cat = cross_validate(
        "CatBoost", fit_cat, X, y, Xt, predict_prep=cat_predict_frame
    )

    # Save OOF + test probabilities so blend weights can be re-tuned later WITHOUT
    # retraining (10 min -> milliseconds). These .npy files are gitignored.
    for name, arr in [
        ("oof_lgb", oof_lgb), ("oof_xgb", oof_xgb), ("oof_cat", oof_cat),
        ("test_lgb", test_lgb), ("test_xgb", test_xgb), ("test_cat", test_cat),
    ]:
        np.save(f"{name}.npy", arr)

    oofs = [oof_lgb, oof_xgb, oof_cat]
    tests = [test_lgb, test_xgb, test_cat]
    singles = [accuracy_score(y, o.argmax(1)) for o in oofs]
    equal_blend_acc = accuracy_score(y, (oof_lgb + oof_xgb + oof_cat).argmax(1))

    # Find the best weighted blend on the OOF predictions.
    best_w, best_acc = search_blend_weights(oofs, y)

    print("=" * 60)
    print("Cross-validated accuracy:")
    print(f"  LightGBM        : {singles[0]:.5f}")
    print(f"  XGBoost         : {singles[1]:.5f}")
    print(f"  CatBoost        : {singles[2]:.5f}")
    print(f"  Equal blend     : {equal_blend_acc:.5f}")
    print(f"  Weighted blend  : {best_acc:.5f}   weights (lgb, xgb, cat) = {best_w}")
    print("=" * 60)

    # Submit whichever scored best on OOF: the weighted blend, or the best single
    # model if no blend beat it.
    if best_acc >= max(singles):
        test_final = best_w[0] * tests[0] + best_w[1] * tests[1] + best_w[2] * tests[2]
        chosen = f"weighted blend {best_w}"
    else:
        winner = int(np.argmax(singles))
        test_final = tests[winner]
        chosen = ["LightGBM", "XGBoost", "CatBoost"][winner]
    print(f"Using {chosen} for the submission.")

    final = pd.DataFrame({
        "id": pd.read_csv("data/test.csv")["id"],
        "class": [INT_TO_CLASS[i] for i in test_final.argmax(1)],
    })
    final.to_csv("submission_ensemble.csv", index=False)
    print("\nSaved submission_ensemble.csv")
    print(final["class"].value_counts())


if __name__ == "__main__":
    main()
