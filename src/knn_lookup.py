"""
Hypothesis: the synthetic rows are close to real SDSS objects, so the nearest real
neighbour's label may recover the truth better than the model in some regions.

This script evaluates that OFFLINE on the labelled training set (free — no
submission), then reports whether a "trust the real label when the match is very
close, else trust the model" hybrid beats the model alone. Only a positive offline
result justifies a leaderboard submission.

Run:  python src/knn_lookup.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

import ensemble as E

# shared numeric features (real data lacks the two synthetic-only categoricals)
KNN_FEATS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def load_real():
    ext = pd.read_csv("data/external/star_classification.csv")[KNN_FEATS + ["class"]].copy()
    for c in ["u", "g", "r", "i", "z"]:
        ext[c] = ext[c].replace(-9999.0, np.nan)
    ext = ext.dropna(subset=KNN_FEATS).reset_index(drop=True)
    return ext


def main():
    train = pd.read_csv("data/train.csv")
    y = train["class"].map(E.CLASS_TO_INT).values
    real = load_real()
    y_real = real["class"].map(E.CLASS_TO_INT).values

    # standardise on train stats, apply to both (features are on very different scales)
    scaler = StandardScaler().fit(train[KNN_FEATS])
    Xtr = scaler.transform(train[KNN_FEATS])
    Xrl = scaler.transform(real[KNN_FEATS])

    nn = NearestNeighbors(n_neighbors=1).fit(Xrl)
    dist, idx = nn.kneighbors(Xtr)
    dist, idx = dist.ravel(), idx.ravel()
    knn_label = y_real[idx]

    # the model's train predictions (out-of-fold, saved earlier)
    model_pred = np.load("oof_lgb.npy").argmax(1)

    print(f"real reference rows: {len(real):,}")
    print(f"nearest-neighbour distance:  median={np.median(dist):.4f}  "
          f"p10={np.percentile(dist,10):.4f}  min={dist.min():.4f}")
    print()
    print(f"model (OOF) accuracy      : {accuracy_score(y, model_pred):.5f}")
    print(f"kNN-label accuracy (all)  : {accuracy_score(y, knn_label):.5f}")
    print()

    # Hybrid: use the real label only where the match is closer than a threshold.
    print("Hybrid — use kNN label when distance < t, else the model:")
    best = (None, accuracy_score(y, model_pred))
    for q in [1, 2, 5, 10, 20, 40]:
        t = np.percentile(dist, q)
        use = dist < t
        hyb = np.where(use, knn_label, model_pred)
        acc = accuracy_score(y, hyb)
        flag = "  <-- beats model" if acc > accuracy_score(y, model_pred) + 1e-9 else ""
        print(f"  t=p{q:<2d} ({t:.3f}): overrides {use.mean()*100:5.2f}%  acc={acc:.5f}{flag}")
        if acc > best[1]:
            best = (q, acc)

    print()
    if best[0] is None:
        print("VERDICT: no threshold beats the model offline -> do NOT submit.")
    else:
        print(f"VERDICT: best hybrid at p{best[0]} -> acc {best[1]:.5f} "
              f"(+{best[1]-accuracy_score(y, model_pred):.5f}). "
              f"Worth a submission only if this clears the noise floor.")


if __name__ == "__main__":
    main()
