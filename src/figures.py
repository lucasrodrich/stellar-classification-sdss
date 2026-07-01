"""
Generate the figures used in the README from the real data.

Produces (into assets/):
  - confusion_matrix.png  : where the model confuses the three classes
  - feature_importance.png: which engineered features drive the predictions

Reproducible: anyone with data/train.csv can run `python src/figures.py`.
Requirements:  pip install -r requirements.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import lightgbm as lgb

from ensemble import prepare_data, LGB_PARAMS, CLASSES, EARLY_STOP

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)


def main():
    X, y, _, _ = prepare_data()

    # One honest 80/20 holdout is enough for illustrative figures (fast + reproducible).
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    pred = model.predict(X_va)
    print(f"holdout accuracy: {accuracy_score(y_va, pred):.5f}")

    # --- Figure 1: confusion matrix (row-normalized) ------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        y_va, pred, display_labels=CLASSES, normalize="true",
        cmap="Blues", values_format=".3f", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (holdout, row-normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    # --- Figure 2: top feature importances (GAIN = real predictive value) ---
    # NB: LightGBM's default importance is "split" count, which over-credits
    # high-cardinality continuous columns. "gain" reflects actual contribution.
    imp = model.booster_.feature_importance(importance_type="gain")
    names = np.array(model.booster_.feature_name())
    order = np.argsort(imp)[::-1][:15][::-1]  # top 15, ascending for barh
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.barh(names[order], imp[order], color="#3b6fb0")
    ax.set_title("Top 15 feature importances (LightGBM, gain)")
    ax.set_xlabel("total gain")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "feature_importance.png"), dpi=130)
    plt.close(fig)

    print(f"Saved figures to {ASSETS}")


if __name__ == "__main__":
    main()
