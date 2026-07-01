"""
Experiment campaign: try to beat the 0.96805 LightGBM baseline, honestly.

Each variant is scored with the SAME 5-fold out-of-fold protocol as the baseline,
so deltas are comparable. Target encoding is done strictly out-of-fold (encoding
computed on the training folds only) to avoid leaking the label into CV.

Verdict rule (learned the hard way — see docs/diagnostics.md): CV overstates the
leaderboard by ~0.011, and CV differences below ~0.003 are noise that can even
invert on the real test set. So a variant is only "worth a leaderboard check" if
it beats baseline by >= 0.003.

Run:  python src/experiments.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

import ensemble as E

BASELINE_CV = 0.96805
N_CLASSES = len(E.CLASSES)
SMOOTHING = 100.0


def base_frame():
    """Engineered features (27) with categoricals as category dtype."""
    df = E.add_features(pd.read_csv("data/train.csv"))
    y = df["class"].map(E.CLASS_TO_INT).values
    for c in E.CAT_COLS:
        df[c] = df[c].astype("category")
    feats = [c for c in df.columns if c not in ["id", "class"]]
    return df[feats], y


def te_columns(cat_tr, y_tr, cat_apply):
    """Out-of-fold target encoding: P(class | category), smoothed to the prior.

    Encoding is estimated from (cat_tr, y_tr) only, then applied to cat_apply.
    Returns an array of shape (len(cat_apply), N_CLASSES).
    """
    prior = np.bincount(y_tr, minlength=N_CLASSES) / len(y_tr)
    dfm = pd.DataFrame({"c": np.asarray(cat_tr), "y": y_tr})
    enc = {}
    for cat, grp in dfm.groupby("c", observed=True):
        cnt = np.bincount(grp["y"].to_numpy(), minlength=N_CLASSES)
        enc[cat] = (cnt + SMOOTHING * prior) / (cnt.sum() + SMOOTHING)
    return np.vstack([enc.get(v, prior) for v in np.asarray(cat_apply)])


def cv_accuracy(X, y, use_target_encoding=False, drop_cols=(), params=None):
    """5-fold OOF accuracy for a variant, matching the baseline protocol."""
    params = params or E.LGB_PARAMS
    cols = [c for c in X.columns if c not in drop_cols]
    Xv = X[cols]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(Xv), N_CLASSES))

    for tr, va in skf.split(Xv, y):
        X_tr, X_va = Xv.iloc[tr].copy(), Xv.iloc[va].copy()
        if use_target_encoding:
            for c in E.CAT_COLS:
                if c in cols:
                    te_tr = te_columns(X.iloc[tr][c], y[tr], X.iloc[tr][c])
                    te_va = te_columns(X.iloc[tr][c], y[tr], X.iloc[va][c])
                    for k in range(N_CLASSES):
                        X_tr[f"te_{c}_{k}"] = te_tr[:, k]
                        X_va[f"te_{c}_{k}"] = te_va[:, k]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])],
                  callbacks=[lgb.early_stopping(E.EARLY_STOP, verbose=False)])
        oof[va] = model.predict_proba(X_va)
    return accuracy_score(y, oof.argmax(1))


def main():
    X, y = base_frame()
    print(f"Base: {X.shape[1]} features on {len(X):,} rows\n")

    tuned = dict(E.LGB_PARAMS)
    tuned.update(num_leaves=255, learning_rate=0.02, reg_lambda=5.0)

    variants = [
        ("baseline (27 feat)",          dict()),
        ("+ target encoding",           dict(use_target_encoding=True)),
        ("drop alpha/delta (position)", dict(drop_cols=("alpha", "delta"))),
        ("target enc + drop position",  dict(use_target_encoding=True,
                                             drop_cols=("alpha", "delta"))),
        ("tuned (leaves255,lr.02,l2.5)", dict(params=tuned)),
    ]

    results = []
    for name, kw in variants:
        acc = cv_accuracy(X, y, **kw)
        delta = acc - BASELINE_CV
        verdict = "WORTH LB CHECK" if delta >= 0.003 else "noise floor - hold"
        results.append((name, acc, delta, verdict))
        print(f"{name:32s} CV={acc:.5f}  delta={delta:+.5f}  {verdict}")

    print("\n" + "=" * 68)
    print("SUMMARY (vs baseline 0.96805):")
    for name, acc, delta, verdict in sorted(results, key=lambda r: -r[1]):
        print(f"  {name:32s} {acc:.5f}  ({delta:+.5f})  {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    main()
