"""
Fast iteration harness: LightGBM + XGBoost only (the two quick models), with an
EXPANDED, physics-motivated feature set. CatBoost is intentionally dropped (it was
the slowest and weakest member; the weight search always zeroed it out).

Goal of this script: test whether new features move cross-validated accuracy by
more than the ~0.003 noise floor. Anything smaller is not trustworthy on this
competition (CV overstates the leaderboard by ~0.011 — see docs/diagnostics.md),
so only a clear CV gain justifies spending a leaderboard submission to confirm.

Run from the project root:  python src/train_fast.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import ensemble as E  # reuse the base features, params, fit_* and CV engine (DRY)

BASELINE_CV = 0.96805  # pure LightGBM, 27 features — the number to beat


def add_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Base color/redshift features + new physics-motivated ones."""
    df = E.add_features(df)  # the original 27-feature engineering

    # spectral-shape ("color curvature"): how the color gradient itself bends
    df["ug_gr"] = df["u_g"] - df["g_r"]
    df["gr_ri"] = df["g_r"] - df["r_i"]
    df["ri_iz"] = df["r_i"] - df["i_z"]

    # extra redshift interactions (redshift is the strongest signal)
    df["z_x_iz"] = df["redshift"] * df["i_z"]
    df["z_x_uz"] = df["redshift"] * df["u_z"]
    df["z_x_mag"] = df["redshift"] * df["mag_mean"]
    df["redshift_sq"] = df["redshift"] ** 2

    # stars sit at redshift ~ 0; an explicit flag targets the STAR<->GALAXY mix-up
    df["low_z_flag"] = (df["redshift"] < 0.0025).astype(int)
    return df


def prepare():
    train = add_features_v2(pd.read_csv("data/train.csv"))
    test = add_features_v2(pd.read_csv("data/test.csv"))
    y = train["class"].map(E.CLASS_TO_INT).values
    for c in E.CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    features = [c for c in train.columns if c not in ["id", "class"]]
    print(f"Prepared {len(features)} features on {len(train):,} rows "
          f"(+{len(features) - 27} new)\n")
    return train[features], y, test[features]


def best_two_way_blend(oof_a, oof_b, y, step=0.05):
    best_w, best = 0.0, -1.0
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        acc = accuracy_score(y, (w * oof_a + (1 - w) * oof_b).argmax(1))
        if acc > best:
            best, best_w = acc, round(float(w), 3)
    return best_w, best


def main():
    X, y, Xt = prepare()

    print("Training LightGBM ...")
    oof_lgb, test_lgb = E.cross_validate("LightGBM", E.fit_lgb, X, y, Xt)
    print("Training XGBoost ...")
    oof_xgb, test_xgb = E.cross_validate("XGBoost", E.fit_xgb, X, y, Xt)

    w, blend_cv = best_two_way_blend(oof_lgb, oof_xgb, y)
    lgb_cv = accuracy_score(y, oof_lgb.argmax(1))
    xgb_cv = accuracy_score(y, oof_xgb.argmax(1))

    print("=" * 60)
    print(f"LightGBM CV        : {lgb_cv:.5f}")
    print(f"XGBoost  CV        : {xgb_cv:.5f}")
    print(f"Best blend CV      : {blend_cv:.5f}   (w_lgb={w}, w_xgb={round(1 - w, 3)})")
    print(f"Baseline (27 feat) : {BASELINE_CV:.5f}")
    delta = blend_cv - BASELINE_CV
    verdict = ("MEANINGFUL — worth an LB check" if delta >= 0.003
               else "below the ~0.003 noise floor — do NOT trust it")
    print(f"Delta vs baseline  : {delta:+.5f}   -> {verdict}")
    print("=" * 60)

    test_final = w * test_lgb + (1 - w) * test_xgb
    pd.DataFrame({
        "id": pd.read_csv("data/test.csv")["id"],
        "class": [E.INT_TO_CLASS[i] for i in test_final.argmax(1)],
    }).to_csv("submission_fast.csv", index=False)
    print("Saved submission_fast.csv")


if __name__ == "__main__":
    main()
