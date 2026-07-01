# Stellar Classification (SDSS) — an honest end-to-end ML case study

Classifying astronomical objects as **GALAXY / QSO / STAR** from Sloan Digital Sky Survey
photometry — Kaggle Playground **S6E6**. The model is a solid gradient-boosting pipeline;
the *real* work is everything around it: catching a lying validation score, diagnosing it
scientifically, and running a disciplined campaign to beat it.

> **What this repo actually demonstrates:** not a leaderboard trophy, but the judgment to
> tell **signal from noise**, debug by **falsifiable hypotheses** (including discarding my
> own), and stay **honest about a ceiling** instead of chasing a number I couldn't explain.
> That's the harder, rarer half of machine learning — and it's the whole point here.

---

## Skills demonstrated

| Area | In this project |
|---|---|
| **Feature engineering** | Physics-motivated color indices + redshift interactions; validated by gain-based importances |
| **Rigorous validation** | StratifiedKFold, out-of-fold scoring, leak-free CV, noise-floor discipline |
| **Model diagnostics** | Adversarial validation, concept-drift detection, error analysis, confidence calibration |
| **Scientific debugging** | 7 falsified hypotheses, one variable at a time — evidence over intuition |
| **Ensembling & stacking** | Weighted blends, out-of-fold stacking with a meta-learner, a **TabPFN foundation model** on GPU |
| **Honest evaluation** | Reporting the real **0.957**, not the flattering 0.968 — and explaining exactly why they differ |
| **Reproducibility & tooling** | Pinned env, scriptable pipeline, Kaggle API, a GPU notebook, clean git history |

## The problem

[Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6):
3-class classification, **~577k** labelled train rows, **~247k** test rows, scored on
**accuracy**. Classes: `GALAXY`, `QSO` (quasar), `STAR`.

## Approach & result

**Feature engineering.** The five photometric bands `u, g, r, i, z` are brightnesses through
different filters; in astronomy the *difference* between two bands (an object's "color")
separates these types far better than raw brightness. The pipeline builds adjacent and wide
color indices, magnitude summaries, and redshift interactions — **27 features**. Importances
confirm the thesis: redshift and engineered colors dominate.

**Model.** LightGBM with 5-fold cross-validation and early stopping. A tuned single model
turned out to beat every fancier thing built on top of it.

| Feature importance (gain) | Confusion matrix (holdout) |
|---|---|
| ![feature importance](assets/feature_importance.png) | ![confusion matrix](assets/confusion_matrix.png) |

*Engineered colors + redshift are the top predictors; `STAR` is the hardest class, most often
confused with `GALAXY`.*

## Results — the simple model won

| Model | CV (5-fold) | Public LB | Private LB |
|-------|:-----------:|:---------:|:---------:|
| **LightGBM baseline** ⭐ | 0.96805 | **0.95657** | **0.95736** |
| LightGBM + XGBoost blend | 0.96809 | 0.95644 | 0.95717 |
| + external real SDSS data | 0.96812 | 0.95663 | 0.95732 |
| 5-model stack + TabPFN (GPU) | — | 0.95410 | 0.95483 |

Every attempt to beat the baseline — blending, external data, kNN lookup, and diverse GPU
stacking — was measured and **came in at or below it**. The blend's +0.00004 CV "edge" was
noise; the leaderboard ranked it *below* the baseline. **Simplicity, honestly verified, won.**

## The headline story: my cross-validation was lying

Cross-validation promised **0.968**; the leaderboard delivered **0.956** — a gap ~10× larger
than sampling noise. Instead of accepting it, I ran a **hypothesis-elimination investigation**
and falsified **seven** explanations with evidence — adversarial validation, a leak-free
out-of-fold recomputation, prediction-agreement analysis — *including my own leading theory*.

The survivor: **concept drift** — identical feature distributions but a subtly different label
rule between train and test. The pipeline was clean; the discrepancy was in the data. That
diagnosis reshaped how I worked: **treat CV as a relative compass, respect the noise floor,
and never trust complexity you can't verify.**

📄 **Full investigation:** [`docs/diagnostics.md`](docs/diagnostics.md)
📄 **The improvement campaign (every attempt, all honest negatives):** [`docs/experiments.md`](docs/experiments.md)
📄 **From-scratch teaching walkthrough (incl. the full journey):** [`LEARNING.md`](LEARNING.md)

## Repository layout

```
stellar-classification-sdss/
├── src/
│   ├── stellar.py         # LightGBM 5-fold baseline → submission.csv (best model)
│   ├── ensemble.py        # LGB + XGB + CatBoost, OOF blend + weight search
│   ├── experiments.py     # controlled ablations (target encoding, tuning, ...)
│   ├── train_external.py  # augment with real SDSS17 data (concept-drift bridge)
│   ├── knn_lookup.py      # nearest-neighbour label lookup (falsified offline)
│   ├── stack.py / stack_tabpfn.py  # diverse stacking + TabPFN foundation model
│   └── figures.py         # regenerates the README figures
├── notebooks/
│   └── gpu_stack.ipynb    # GPU-ready stacking notebook (LGB+XGB+TabPFN)
├── docs/
│   ├── diagnostics.md     # the CV-vs-leaderboard investigation  ← start here
│   └── experiments.md     # the full improvement campaign
├── assets/                # generated figures
├── LEARNING.md            # from-scratch explainer + the full project journey
├── requirements.txt
└── data/                  # train.csv / test.csv — gitignored, add your own
```

## Reproduce

```bash
python -m venv venv
# Activate — Windows PowerShell: venv\Scripts\Activate.ps1 | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Get the data (not committed — Kaggle redistribution rules):
kaggle competitions download -c playground-series-s6e6 -p data/ && unzip data/*.zip -d data/

python src/stellar.py     # baseline → submission.csv  (the best model)
python src/ensemble.py    # blend + weight search
python src/figures.py     # regenerate the figures
```

## Development note

Built **AI-assisted**, using [Claude Code](https://www.anthropic.com/claude-code) as a pair
programmer for scaffolding, implementation, and running experiments. **The direction and
judgment are mine:** the feature strategy, reading the CV-vs-leaderboard gap as a red flag,
choosing which hypotheses to test, deciding when to stop — and *correcting the AI* when its
early-stopping hypothesis was falsified by the evidence. Directing and verifying an AI
collaborator, rather than trusting it blindly, is the workflow I'd bring to a team.
