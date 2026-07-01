# Experiment Campaign: Chasing the Last +0.001 (and Documenting Every Dead End)

> After the [diagnostic investigation](diagnostics.md) showed the pipeline was clean
> and the ~0.011 CV→LB gap was concept drift, I ran a controlled campaign to try to
> beat the baseline anyway. **Nothing beat it.** This is the honest log of what was
> tried, what it scored, and why each was kept or discarded — because a rigorous
> negative result is worth more than a lucky number.

## Ground rules (learned from the diagnostics)

- **Every variant is one change against a fixed baseline** (LightGBM, 27 features) so
  each result is attributable (an *ablation*).
- **Noise floor:** CV differences below ~0.003, and LB differences below ~±0.001, are
  indistinguishable from luck and are **discarded**.
- **CV-gated vs LB-gated:** most variants are judged on CV. The external-data variant
  is judged on the **leaderboard**, because the test set is synthetic and CV
  structurally cannot see the effect (see Tier 2).

## Reference scores

| Submission | CV | Public LB | Private LB |
|---|---|---|---|
| **LightGBM baseline (best)** | 0.96805 | 0.95657 | **0.95736** |
| LightGBM + XGBoost blend | 0.96809 | 0.95644 | 0.95717 |

## Tier 1 — CV-screened variants (`src/experiments.py`, `src/train_fast.py`)

| Variant | CV | Δ vs baseline | Decision |
|---|---|---|---|
| baseline (27 features) | 0.96805 | — | reference |
| + 8 engineered features (curvatures, interactions, `low_z` flag) | 0.96797 | −0.00008 | **discard** |
| + out-of-fold target encoding | 0.96805 | +0.00000 | **discard** |
| drop `alpha` / `delta` (sky position) | 0.95422 | **−0.01383** | **discard** |
| target encoding + drop position | 0.95421 | −0.01384 | **discard** |
| tuned (num_leaves 255, lr 0.02, λ 5) | 0.96806 | +0.00001 | **discard** |

**Nothing cleared +0.003.** Two findings worth keeping:

- **More engineered features didn't help** (they even hurt slightly). They were
  deterministic recombinations of existing columns — and gradient-boosted trees can
  already represent those internally, so pre-computing them adds nothing. (For *tree*
  models, hand-made interactions of existing inputs rarely help — unlike linear models.)
- **Sky position is critical, not noise.** Dropping `alpha`/`delta` cost **0.014** — a
  huge amount — even though the gain-based importance chart ranked them *low*. This is
  **"importance ≠ necessity":** a feature can rank modestly yet capture signal nothing
  else substitutes for, so only *ablation* reveals what is safe to remove. (Position
  likely encodes SDSS survey-selection structure, which correlates with class.)

## Tier 2 — external real data (`src/train_external.py`)

**Hypothesis:** the competition data is synthetic (a generator's imperfect copy of a
real dataset). Mixing in the original **real SDSS17** rows should anchor the decision
boundary to the true label rule the synthetic test approximates — bridging the concept
drift. 100,000 real rows were cleaned (SDSS `-9999` sentinels → NaN; the two
synthetic-only categoricals marked missing) and added to every training fold.

**This is leaderboard-gated:** the test set is synthetic, so real data can *lower*
synthetic-CV while *raising* the LB. CV cannot judge it — only a submission can.

| | CV (synthetic-val) | Public LB | Private LB |
|---|---|---|---|
| baseline | 0.96805 | 0.95657 | 0.95736 |
| + 100k real SDSS rows | 0.96812 | 0.95663 | 0.95732 |
| Δ | +0.00007 | **+0.00006** | **−0.00004** |

**Null result.** The LB moved by +0.00006 (public) / −0.00004 (private) — both inside
the ±0.001 noise floor, and **opposite in sign**, the classic signature of pure noise.
Only 0.37% of predictions changed. **Discard.** The concept drift here is not the kind
of correctable generator label-noise that real data fixes.

## Conclusion

Across engineered features, target encoding, feature pruning, hyperparameter tuning,
model blending, and real-data augmentation, **nothing beat the plain LightGBM
baseline** (0.95736 private). That is the honest result: on this competition, with this
approach, ~0.957 is the ceiling, and the ~0.011 gap to a hypothetical 0.968 is concept
drift that none of these levers could close.

**What this campaign demonstrates** is not a winning score — it's the *method*:
controlled one-variable experiments, a noise-floor discipline that refuses to chase
luck, a principled CV-vs-LB distinction, and the willingness to run the highest-upside
idea (external data) and **report that it failed**. Knowing what *doesn't* work, and
why, is the harder and more valuable half of the job.

### If the hunt continued

The most likely path to the ~0.970 leader cluster is **not** more modelling — it's
matching synthetic test rows back to their nearest real SDSS neighbours to recover
cleaner labels (a lookup, not a fit). That is a genuinely different technique and a
separate experiment; it is noted here as the next hypothesis rather than a claim.
