# Diagnostic Investigation: When Cross-Validation Lied by 0.011

> A short write-up of the most instructive part of this project: catching a
> cross-validation score that systematically disagreed with the real test set,
> and running a hypothesis-elimination process to find out why.

## TL;DR

My 5-fold cross-validation reported **0.968** accuracy. The Kaggle leaderboard —
the real, held-out test set — returned **0.956**. That is a **~0.011 gap**, and on
the ~49,000-row public split (noise ≈ ±0.001) it is roughly **10× too large to be
chance**. The CV wasn't noisy; it was *biased*.

I treated this as a bug hunt, not a shrug. I formed one hypothesis at a time and
tried to kill each with evidence. Seven pipeline explanations were falsified —
including my own leading theory. The surviving explanation is **concept drift**:
the train and test sets share an identical *feature* distribution but a slightly
different *label* rule, which no training-set analysis can detect and only a
submission can reveal. **The pipeline is clean; the gap is a property of the data.**

## The anomaly

| Submission | CV (5-fold) | Public LB | Private LB |
|---|---|---|---|
| Pure LightGBM baseline | 0.96805 | 0.95657 | 0.95736 |
| Weighted blend (0.7 LGB / 0.3 XGB) | 0.96809 | 0.95644 | 0.95717 |

Two independent models land at ~0.956 with an almost identical CV→LB gap (0.01148
vs 0.01165). A consistent, model-agnostic offset is the signature of a *data*
effect, not a coding bug in any one model.

## Hypothesis elimination

| # | Hypothesis | How I tested it | Result |
|---|---|---|---|
| 1 | Test has unseen categorical values → NaN | count test rows with categories absent from train | ❌ 0 rows |
| 2 | Train/test distribution shift | compare per-feature min / mean / max | ❌ identical |
| 3 | Missing values mishandled | `isna()` on both sets | ❌ none |
| 4 | Categoricals secretly encode the label | majority-vote accuracy from the 2 categoricals alone | ❌ only 0.767 |
| 5 | Hidden multivariate covariate shift | **adversarial validation** (classify train-vs-test) | ❌ AUC 0.5006 |
| 6 | XGBoost test-path bug (feature misalignment) | LGB↔XGB prediction agreement, blend-vs-baseline delta | ❌ agree 0.9961 on test; blend differs from pure LGB by ~0.1% |
| 7 | Early-stopping leak on the validation fold | recompute a **leak-free OOF** (early-stop on an inner split) | ❌ 0.96805 → 0.96759 (only −0.0005) |

Notes on the two most interesting checks:

- **Adversarial validation (#5):** train a classifier to tell train rows from test
  rows. AUC ≈ 0.5 means it *can't* — the feature distributions are statistically
  identical, ruling out covariate shift.
- **Leak-free OOF (#7):** my leading theory was that early stopping peeks at the
  fold it then scores, inflating CV. I predicted the leak-free CV would fall toward
  0.956. It fell by 0.0005. **My own hypothesis was falsified** — which is the point
  of testing it.

## The clincher: confidence profiles

If the *inputs* were unfamiliar, the model would be less confident on test. If the
*label rule* changed, the model would be **just as confident but wrong more often.**

| | mean max-probability | fraction > 0.9 |
|---|---|---|
| OOF (train) | 0.9692 | 0.911 |
| Test | 0.9690 | 0.911 |

Identical. The model cannot tell it is doing worse on test — exactly what concept
drift looks like, and inconsistent with any input-side problem (already ruled out
by #1–#5).

## Conclusion

`P(features)` is provably identical between train and test, yet an honest,
leak-free CV of 0.9676 still overshoots the 0.956 test score. Logically the
difference can only live in `P(class | features)` — **concept drift**. It cannot be
*proven* without the hidden test labels, but it is the single explanation
consistent with all seven falsified hypotheses plus the confidence signature. It is
also common in synthetic tabular competitions, where train and test labels are
generated with subtly different characteristics.

## What this changed about how I work

1. **CV here is a *relative* compass, not an absolute score.** The trustworthy
   number is the leaderboard (~0.956).
2. **Only act on CV moves bigger than the noise floor (~0.003).** The blend was
   chosen over the baseline for a **+0.00004** CV edge; the leaderboard then ranked
   them the *other way* (baseline +0.00013). Chasing a sub-noise "gain" added
   complexity for a net loss — *complexity you can't verify is a liability*.
3. **The simpler model won.** The pure LightGBM baseline beats the ensemble on both
   public and private leaderboards, so it is the submission of record.
4. **Real gains must come from features**, not pipeline plumbing — a change large
   enough for CV to actually track.

## Reproduce

Every number above comes from `data/train.csv` / `data/test.csv` plus the scripts
in `src/`. The adversarial-validation, leak-free-OOF, and confidence checks were
run as standalone diagnostics against the models in `src/ensemble.py`.
