# LEARNING.md — How This Project Was Built, and How You Could Build One Yourself

This file is a teaching companion to the code. It explains **two different things**:

1. **The thinking** — what goes through an engineer's head *before and during* writing
   each part. This is the part most tutorials skip, and it's the part that actually
   lets you build things on your own.
2. **The code** — what every meaningful line in `src/stellar.py` does, in plain
   language.

Read it top to bottom once. Then keep it open beside `src/stellar.py` as a reference.

> **Important context added after the fact:** the "~0.968" cross-validated accuracy
> celebrated below turned out to be **optimistic** — the real leaderboard score is
> **~0.956**. That gap became the most valuable lesson in the whole project. Wherever
> this file says 0.968, read it as "the CV estimate," and see
> [`docs/diagnostics.md`](docs/diagnostics.md) for why CV and reality diverged and how
> I tracked it down.

---

## Part 0 — The mental model (read this first)

Before any code, you need the right picture in your head. **A machine-learning
classification project is a guessing machine that you train by example.**

Imagine you want to teach someone who has never seen the night sky to tell apart
three kinds of objects: a **galaxy**, a **quasar (QSO)**, and a **star**. You can't
explain it in words, but you *can* show them 577,347 labeled examples — "this set of
measurements is a GALAXY, this one is a STAR" — and let them figure out the pattern.
Then you hand them 247,435 new objects with the labels hidden and ask them to guess.

That is exactly what this project does:

- **`train.csv`** = the 577,347 examples *with* the answer (`class`).
- **`test.csv`** = the 247,435 new objects *without* the answer. We must guess.
- **The model** (LightGBM) = the "student" that learns the pattern.
- **`submission.csv`** = our guesses, in the format Kaggle wants.
- **Accuracy** = the fraction of guesses that are correct. That's our grade.

Everything in the code serves one of four jobs, in this order:

```
   LOAD  →  ENGINEER FEATURES  →  TRAIN & CHECK  →  PREDICT & SAVE
   (1)            (2)                 (3)                (4)
```

If you ever feel lost in the code, ask "which of these four jobs is this line doing?"
The script is even divided into four numbered sections matching this exact flow.

---

## Part 1 — How an engineer plans *before* writing code

Senior engineers do not open a blank file and start typing. They answer a handful of
questions first. Here are the questions, and the answers we settled on for this project.

**Q1. What kind of problem is this?**
Three possible answers (GALAXY / QSO / STAR), and each object is exactly one of them.
That makes it a **multi-class classification** problem (not regression, which predicts
a number; not binary classification, which has only two answers). This single decision
determines almost everything else — the model settings, the loss function, the metric.

**Q2. How will I know if I'm doing well *before* I submit?**
This is the most important question and the one beginners skip. Kaggle only lets you
submit a few times a day, and the public leaderboard can lie to you. So we need our
**own** honest scoreboard. The answer is **cross-validation** (explained in detail in
Part 5). Decide your validation strategy *before* you model — it is your steering wheel.

**Q3. What's the simplest thing that could work?**
Not the fanciest. The simplest. One good model (LightGBM) with sensible features. You
earn the right to build a 3-model ensemble *after* the simple version runs and gives a
trustworthy score. Complexity you add before you have a baseline is complexity you
can't measure.

**Q4. What do I know about the data that the model doesn't?**
This is where domain knowledge becomes features. We know from astronomy that the
*difference* between two brightness bands (a "color") separates these objects better
than raw brightness. The model can't know that on its own from a small number of trees,
so we hand it those differences directly. This is **feature engineering**, and it's
usually where Kaggle competitions are won or lost.

**Q5. What format must the final answer be in?**
Kaggle requires a CSV with exactly an `id` column and a `class` column. If we get the
format wrong, a perfect model scores zero. So we design the last step backward from
that requirement.

Notice: **four of these five questions have nothing to do with syntax.** That's the
lesson. The thinking is the job; the typing is the easy part.

---

## Part 2 — The domain knowledge, explained simply

You don't need an astronomy degree, but a little context makes the feature engineering
obvious instead of magical.

- **Photometric bands `u, g, r, i, z`** — these are five color filters, from ultraviolet
  (`u`) through to near-infrared (`z`). Each number is how bright the object looks
  through that filter. Think of measuring a paint sample's brightness through five
  differently tinted pieces of glass.
- **"Color" = a difference between two bands** — e.g. `u - g`. A red object is brighter
  in red filters than blue ones; a blue object is the reverse. The *difference* captures
  this regardless of how far away (and therefore how dim overall) the object is. That's
  why color separates object *types* better than raw brightness: brightness mostly tells
  you distance, color tells you what kind of thing it is.
- **`redshift`** — how much the object's light has been stretched toward red by the
  expansion of the universe. Stars in our galaxy have tiny redshift. Galaxies have
  moderate redshift. Quasars are extremely far away and have huge redshift. This single
  number is the **strongest** clue for telling the three classes apart — which is why the
  code builds extra features by multiplying redshift with the colors.
- **`spectral_type`, `galaxy_population`** — text labels (categories), not numbers. The
  model has to be told to treat them as categories, not accidentally as math.
- **`alpha`, `delta`** — the object's position in the sky (like latitude/longitude).
  Usually weak signal, but harmless to include.

That's the entire astronomy lesson. Everything in Section 2 of the code is just turning
these ideas into columns of numbers.

---

## Part 3 — Setting up the workshop (environment & dependencies)

Before the code runs, the tools have to exist. Two concepts:

- **Virtual environment (`venv`)** — a private, isolated copy of Python *for this project
  only*. Why? So the libraries this project needs (and their exact versions) don't
  collide with other projects on your machine. You create it once:
  ```bash
  python -m venv venv
  ```
  This makes a `venv/` folder. It is deliberately **gitignored** — it's big, machine-
  specific, and anyone can recreate it from `requirements.txt`. You never commit it.

- **`requirements.txt`** — the shopping list of libraries. Each line is one library, and
  `==` pins an exact version so the project behaves identically on any machine next year.
  ```
  pandas            # tables of data (like a programmable spreadsheet)
  numpy             # fast math on arrays of numbers
  scikit-learn      # the "glue" toolkit: splitting data, scoring, utilities
  lightgbm==4.6.0   # the gradient-boosting model that does the actual learning
  xgboost==3.3.0    # a second model family (for the future ensemble)
  catboost==1.2.10  # a third model family (for the future ensemble)
  ```
  You install everything in one command:
  ```bash
  pip install -r requirements.txt
  ```

**The thinking here:** reproducibility. "It works on my machine" is not good enough. By
pinning versions and isolating the environment, *future you* (or a teammate) can rebuild
the exact setup from two files. We installed `xgboost` and `catboost` now even though the
baseline doesn't use them yet — because we already know the next step is an ensemble, and
adding them to the list now costs nothing.

---

## Part 4 — Line-by-line walkthrough of `src/stellar.py`

Now the code itself. I'll quote each block and explain it underneath. Open the real file
beside this so you can see the blocks in context.

### The imports

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
```

- `import pandas as pd` — load the tables library and give it the short nickname `pd`
  (universal convention; everyone writes `pd`).
- `import numpy as np` — fast numeric arrays; nickname `np`.
- `from sklearn.model_selection import StratifiedKFold` — pull in just the one tool we
  need for splitting the data into folds (more in Part 5).
- `from sklearn.metrics import accuracy_score` — the function that compares guesses to
  truth and returns the fraction correct.
- `import lightgbm as lgb` — the model library.

**Thinking:** import only what you use, and use the conventional nicknames so other people
(and future you) instantly recognize the code.

### Section 1 — Load the data

```python
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
```

- `pd.read_csv(...)` reads a CSV file from disk into a **DataFrame** — pandas' name for a
  table with named columns and rows. `train` now holds all 577,347 labeled rows; `test`
  holds the 247,435 rows we must predict.
- The path is `"data/train.csv"` — **relative to where you run the script from**. That's
  why you run `python src/stellar.py` from the project root, not from inside `src/`. If you
  `cd` into `src` and run it, `data/` won't be found. (A more robust version would build an
  absolute path; we kept it simple on purpose.)

### Section 2 — Feature engineering

```python
def add_features(df):
    df = df.copy()
    bands = ["u", "g", "r", "i", "z"]
```

- `def add_features(df):` defines a reusable **function**. We define it once and call it on
  *both* `train` and `test`, guaranteeing they get the *identical* treatment. (If you
  hand-edited each table separately, they'd drift apart and the model would break.) This is
  the DRY principle — Don't Repeat Yourself.
- `df = df.copy()` makes a **copy** so we don't accidentally modify the caller's original
  table. This is the "immutability" habit: build a new thing rather than mutating the input.
  It prevents a whole category of confusing bugs.
- `bands = [...]` just names our five brightness columns so we can reuse the list below.

```python
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
```

- Each line creates a **new column** holding a "color" (a band difference). `df["u_g"] =
  df["u"] - df["g"]` subtracts the whole `g` column from the whole `u` column, row by row,
  in one shot. (Pandas does this "vectorized" — no manual loop needed, and it's fast.)
- These four are the *adjacent* colors — neighboring filters. They're the classic, most
  informative ones.

```python
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]
```

- Same idea, but *wider* colors that skip a band. They capture broader brightness trends
  across the spectrum that the narrow adjacent colors might miss. We give the model many
  views and let it decide which matter.

```python
    df["mag_mean"]  = df[bands].mean(axis=1)
    df["mag_std"]   = df[bands].std(axis=1)
    df["mag_range"] = df[bands].max(axis=1) - df[bands].min(axis=1)
```

- `df[bands]` selects just the five band columns. `.mean(axis=1)` means "average across the
  columns, for each row" (`axis=1` = horizontally, across columns; `axis=0` would be down a
  column). So `mag_mean` is each object's average brightness.
- `mag_std` is the spread of its five brightnesses (are they all similar, or very different?).
- `mag_range` is brightest-minus-dimmest band. These three are cheap **summary features** —
  compact descriptions of the whole brightness profile.

```python
    df["z_x_ug"] = df["redshift"] * df["u_g"]
    df["z_x_gr"] = df["redshift"] * df["g_r"]
    df["z_x_ri"] = df["redshift"] * df["r_i"]
    df["redshift_log"] = np.log1p(df["redshift"].clip(lower=0))
    return df
```

- The first three are **interaction features**: redshift multiplied by a color. The idea is
  "how a color behaves *depends on* the redshift," and multiplying lets the model use the two
  signals together rather than separately. Since redshift is the strongest signal, combining
  it with colors is high-value.
- `redshift_log` compresses the redshift scale. `np.log1p(x)` computes `log(1 + x)`, which
  squashes huge values and spreads out tiny ones, so the model isn't dominated by a few
  extreme quasars. `.clip(lower=0)` first forces any negative redshift to 0 — because `log`
  of a negative number is undefined and would crash. This is defensive coding: protect the
  math from bad inputs.
- `return df` hands back the enriched table.

```python
train = add_features(train)
test  = add_features(test)
```

- Apply the *same* function to both tables. After this, both have the original columns plus
  all the new engineered ones (the script reports 27 features total at runtime).

```python
classes = ["GALAXY", "QSO", "STAR"]
class_to_int = {c: i for i, c in enumerate(classes)}
y = train["class"].map(class_to_int).values
```

- Models do math, and math needs numbers, not the words "GALAXY"/"QSO"/"STAR". So we map
  each class name to an integer.
- `class_to_int` becomes `{"GALAXY": 0, "QSO": 1, "STAR": 2}`. (`enumerate` pairs each item
  with its position; the `{c: i for ...}` is a "dict comprehension" — a compact way to build
  a dictionary.)
- `y = train["class"].map(class_to_int).values` translates the whole answer column into those
  integers. `y` is now our **target** — the thing we're trying to predict. By strong
  convention, the target is named `y` and the inputs are named `X`.

```python
cat_cols = ["spectral_type", "galaxy_population"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
```

- These two columns are text categories. `astype("category")` tells pandas/LightGBM "treat
  this as a label, not a number" — LightGBM can handle categories natively and split on them
  intelligently.
- The second line is subtle and important: it forces the **test** table to use the *exact same
  category list* as the **train** table. If test happened to contain a category train never
  saw (or in a different internal order), the numbers wouldn't line up and predictions would
  be garbage. We pin them together on purpose. (This is a classic real-world ML bug, handled
  here in one line.)

```python
features = [c for c in train.columns if c not in ["id", "class"]]
X  = train[features]
Xt = test[features]
print(f"Using {len(features)} features on {len(X):,} training rows")
```

- `features` is the list of every column *except* `id` (a meaningless row number — feeding it
  in would let the model "cheat" or just add noise) and `class` (that's the answer; including
  it would be giving the student the test answers). This filtering is a "list comprehension."
- `X` = the training inputs (all feature columns of train). `Xt` = the test inputs. So now we
  have the classic trio: `X` (train inputs), `y` (train answers), `Xt` (test inputs).
- The `print` is a sanity check. The `f"...{len(X):,}..."` is an **f-string**; `:,` adds
  thousands separators so it prints `577,347` instead of `577347`. Always print a checkpoint
  so you can confirm reality matches your expectation.

### Section 3 — Train with cross-validation

```python
params = dict(
    objective="multiclass", num_class=3,
    learning_rate=0.03,
    num_leaves=127,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=2.0, reg_alpha=0.5,
    min_child_samples=40,
    n_estimators=3000,
    random_state=42, n_jobs=-1, verbose=-1,
)
```

These are the model's **settings (hyperparameters)** — the dials you choose, as opposed to
the patterns the model learns by itself. In plain terms:

- `objective="multiclass", num_class=3` — "this is a 3-way classification." This must match
  Q1 from the planning stage.
- `learning_rate=0.03` — how big a correction each new tree makes. Small = careful and
  accurate but needs more trees. Like taking small, steady steps instead of giant leaps.
- `num_leaves=127` — how complex each individual tree may get. Bigger = can capture finer
  patterns but risks **overfitting** (memorizing the training data instead of learning the
  general rule).
- `subsample=0.8, subsample_freq=1` — each tree trains on a random 80% of the rows. Showing
  each tree a slightly different slice keeps them from all making the same mistakes.
- `colsample_bytree=0.7` — each tree only sees a random 70% of the features. Same anti-
  overfitting idea, applied to columns.
- `reg_lambda=2.0, reg_alpha=0.5` — "regularization" penalties that push the model toward
  simpler solutions. Simpler models generalize better to unseen data.
- `min_child_samples=40` — don't make a decision based on fewer than 40 examples. Prevents
  the model from inventing rules from tiny, possibly-coincidental groups.
- `n_estimators=3000` — the *maximum* number of trees. We rarely use all 3000 because of
  early stopping (below).
- `random_state=42` — fixes the randomness so every run is **reproducible**. Same input →
  same output, every time. (42 is a traditional arbitrary choice.)
- `n_jobs=-1` — use all CPU cores. `verbose=-1` — stay quiet, don't flood the screen.

**Thinking:** almost every one of these dials is a different way of saying "don't let the
model memorize; force it to learn the general pattern." Overfitting is the central enemy in
machine learning, and most of the settings exist to fight it.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred  = np.zeros((len(X), 3))
test_pred = np.zeros((len(Xt), 3))
```

- `StratifiedKFold(n_splits=5, ...)` sets up the 5-way split (Part 5 explains it fully).
  "Stratified" means each fold keeps the same GALAXY/QSO/STAR proportions as the whole
  dataset — important because the classes are imbalanced (lots of galaxies, fewer stars).
- `oof_pred` ("out-of-fold predictions") is an empty results table, one row per training
  object, three columns (one probability per class). We'll fill it in as we go and use it to
  compute our honest score.
- `test_pred` is the same shape for the real test set; we'll accumulate predictions into it.
- `np.zeros((rows, 3))` just creates a table of that size filled with zeros, ready to fill.

```python
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx], y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
```

This loop runs **5 times**, once per fold. Each time:

- `skf.split(X, y)` hands back two sets of row positions: `tr_idx` (the ~80% to train on this
  round) and `va_idx` (the held-out ~20% to validate on). `enumerate` also gives `fold` = 0,1,2,3,4.
- `model = lgb.LGBMClassifier(**params)` creates a fresh model using our settings. `**params`
  "unpacks" the dictionary into named arguments — a tidy way to pass many settings.
- `model.fit(...)` is where the actual **learning** happens. We give it the training slice
  (`X.iloc[tr_idx]`, `y[tr_idx]`) — `.iloc[...]` means "select these row positions."
- `eval_set=[(X.iloc[va_idx], y[va_idx])]` shows the model the held-out slice *to watch its
  progress on*, not to learn from. The model checks itself against this after every tree.
- `callbacks=[lgb.early_stopping(80, ...)]` = **early stopping**: "if the score on the watch
  set hasn't improved for 80 trees in a row, stop adding trees." This is why we set the max to
  3000 but actually use ~510–580 (the run reported those numbers). It automatically finds the
  right amount of training and avoids overfitting. Elegant and important.

```python
    oof_pred[va_idx] = model.predict_proba(X.iloc[va_idx])
    test_pred += model.predict_proba(Xt) / 5
    acc = accuracy_score(y[va_idx], oof_pred[va_idx].argmax(1))
    print(f"  fold {fold}: accuracy = {acc:.5f}  (best tree count: {model.best_iteration_})")
```

- `model.predict_proba(...)` outputs **probabilities** — e.g. "70% GALAXY, 20% QSO, 10% STAR"
  — not a single hard label. We store the held-out slice's probabilities into `oof_pred` at
  exactly the rows that were held out this fold. After all 5 folds, *every* training row has a
  prediction that was made by a model which never saw it — an honest dress rehearsal.
- `test_pred += model.predict_proba(Xt) / 5` predicts the **real test set** with this fold's
  model and adds one-fifth of it to the running total. After 5 folds, `test_pred` is the
  *average* of all 5 models — averaging cancels out each model's individual quirks and gives a
  steadier answer. This is a mini-ensemble already.
- `accuracy_score(truth, guesses)` grades this fold. `oof_pred[va_idx].argmax(1)` converts the
  three probabilities into a single guess by picking the **arg**ument (position) with the
  **max** probability (`argmax`), per row (`axis=1`). So "70/20/10" → position 0 → GALAXY.
- The `print` shows the fold's score and how many trees it actually used.

```python
cv_acc = accuracy_score(y, oof_pred.argmax(1))
print(f"\nOverall cross-validated accuracy: {cv_acc:.5f}")
```

- After the loop, score **all** the out-of-fold predictions at once. This single number —
  `0.96805` on our run — is the project's honest self-grade. It's the most trustworthy estimate
  of how we'll do on truly unseen data, and it's what you watch when trying to improve. If a
  change raises this number, it's probably a real improvement; if it doesn't, ignore the change
  no matter how clever it felt.

### Section 4 — Write the submission

```python
int_to_class = {i: c for c, i in class_to_int.items()}
final = pd.DataFrame({
    "id": test["id"],
    "class": [int_to_class[i] for i in test_pred.argmax(1)],
})
final.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(final["class"].value_counts())
```

- `int_to_class` is the reverse dictionary: `{0: "GALAXY", 1: "QSO", 2: "STAR"}`. We turned
  words into numbers to train; now we turn numbers back into words to submit.
- `test_pred.argmax(1)` converts the averaged test probabilities into one integer guess per
  object; the list comprehension maps each integer back to its class name.
- `pd.DataFrame({...})` builds the final two-column table: the `id` from the test set, and our
  predicted `class`. **The format must exactly match what Kaggle expects** — this is Q5 from
  planning, designed backward from the requirement.
- `final.to_csv("submission.csv", index=False)` writes it to disk. `index=False` stops pandas
  from adding its own row-number column, which would break the format.
- The final `print(... value_counts())` shows how many of each class we predicted — a last
  sanity check. If it predicted 100% STAR, you'd instantly know something was wrong. (Ours
  came out ~162k GALAXY / 50k QSO / 35k STAR, which is believable.)

---

## Part 5 — The single most important idea: cross-validation

If you take one concept away from this project, make it this one.

**The problem:** if you train a model and then test it on the *same* data it learned from,
of course it scores well — it has effectively seen the answers. That score is a lie. It tells
you nothing about how the model handles data it's never seen, which is the only thing that
matters on Kaggle.

**The fix — k-fold cross-validation:**

1. Shuffle the training data and cut it into 5 equal piles (the "folds").
2. **Round 1:** train on piles 1–4, then test on pile 5 (which the model never saw). Record
   the score.
3. **Round 2:** train on piles 1,2,3,5, test on pile 4. Record.
4. ...and so on, until each of the 5 piles has had exactly one turn as the unseen test pile.
5. The average of the 5 scores is your honest estimate.

```
Fold 0:  [TEST ][train][train][train][train]
Fold 1:  [train][TEST ][train][train][train]
Fold 2:  [train][train][TEST ][train][train]
Fold 3:  [train][train][train][TEST ][train]
Fold 4:  [train][train][train][train][TEST ]
            every pile is the unseen test exactly once
```

**Why "stratified"?** Our classes are imbalanced. Stratified splitting makes sure each pile has
the same GALAXY/QSO/STAR mix as the whole, so no fold is accidentally easy or hard.

**The payoff:** the `oof_pred` table collects each row's prediction from the one round where it
was the unseen pile. Scoring that whole table gives one trustworthy number — your private,
honest leaderboard that you can check as often as you like, without burning Kaggle submissions.

This is the steering wheel for the entire project. Every future improvement is judged by
whether it moves this number.

---

## Part 6 — How *you* would build this from scratch (the repeatable process)

Here is the process distilled into steps you can reuse on any tabular ML problem:

1. **Read the problem statement and the data dictionary.** What are you predicting? What does
   each column mean? What metric scores you? (Here: predict `class`, scored on accuracy.)
2. **Look at the raw data before modeling.** Print the columns, the shape, a few rows, and the
   target's possible values. We did exactly this and confirmed the script's assumptions held.
   Five minutes here saves an hour of confusion later.
3. **Decide your validation strategy first.** Pick k-fold (usually 5), stratified if classes
   are imbalanced, with a fixed random seed. This is your scoreboard — set it before you model.
4. **Build the simplest honest baseline.** One good model, a few obvious features, the CV loop,
   a valid submission file. Get a real number on the board. *Resist* adding complexity here.
5. **Engineer features from domain knowledge.** Ask "what do I know that the model can't easily
   discover?" Turn each answer into a column. Re-run; keep changes that raise the CV score.
6. **Tune and then ensemble.** Only after the baseline is solid: tune the dials, then combine
   several different model families and average them. (That's literally the next step on this
   project's README checklist.)
7. **Always keep a way to check your work.** Every change is judged by the CV number and a quick
   sanity check on the output (did the class distribution stay sane? any nulls?). Never trust a
   change you haven't measured.

That loop — **baseline → measure → improve one thing → measure again** — is the entire job.
The model is almost a detail; the discipline of measuring honestly is the skill.

---

## Part 7 — Mini-glossary

- **Feature** — an input column the model learns from (a brightness, a color, a redshift).
- **Target (`y`)** — the thing being predicted (`class`, as 0/1/2).
- **Model** — the algorithm that learns the pattern (here, LightGBM).
- **Hyperparameter** — a setting *you* choose (e.g. `learning_rate`), vs. something the model
  learns on its own.
- **Overfitting** — memorizing the training data instead of learning the general rule; it looks
  great on training data and fails on new data. The enemy.
- **Cross-validation (CV)** — splitting data into folds to get an honest score on unseen data.
- **OOF (out-of-fold) prediction** — a row's prediction made by a model that never saw that row.
- **Early stopping** — automatically halt training when the held-out score stops improving.
- **`predict_proba` / probabilities** — the model's confidence per class, e.g. 70/20/10.
- **`argmax`** — pick the position of the largest value; turns probabilities into a final guess.
- **DataFrame** — pandas' table-with-named-columns object; the workhorse of the whole script.
- **Vectorized operation** — doing math on a whole column at once instead of looping row by row;
  shorter to write and far faster.
- **Ensemble** — combining several models (e.g. by averaging) to get a steadier, stronger answer.

---

## Part 8 — Where to go next (to keep learning)

- Open `notebooks/` and plot the data: class balance, redshift histograms per class, and a
  scatter of `g_r` vs `u_g` colored by class. *Seeing* the separation makes the feature
  engineering click.
- Change one hyperparameter (say `num_leaves` from 127 to 63), rerun, and watch the CV number.
  Building intuition for the dials is best done by experiment, not theory.
- When you're ready, the README's next step is the **LightGBM + XGBoost + CatBoost ensemble**.
  You now understand every piece you'll need: the CV loop, OOF predictions, probability
  averaging. The ensemble is just doing what this script does, three times, and blending.

You built (and now understand) a working pipeline with a **0.968 cross-validated**
estimate. (Spoiler for later: the real leaderboard score is ~0.956 — see
[`docs/diagnostics.md`](docs/diagnostics.md). Understanding *why* is the real prize.)
That's a genuine foundation.

---

## Part 9 — The ensemble (`src/ensemble.py`), explained

This is the project's next real step, and it's the perfect lesson in "build on a verified
baseline." Read Parts 0–8 first; this part assumes you understand the CV loop, OOF
predictions, and probability averaging.

### The one-sentence idea

**Run the baseline three times with three *different* model families, then average their
probabilities.** That's the whole thing. If you understand `stellar.py`, you already
understand 90% of `ensemble.py`.

### Why three models instead of one (the thinking)

LightGBM, XGBoost, and CatBoost are all "gradient-boosted trees" — same broad idea — but
they grow their trees by different rules, so they make **different mistakes**. Picture
three competent students grading the same exam. Each gets a few questions wrong, but
rarely the *same* questions. If you take the majority/average of their answers, the
individual errors cancel and the group is more reliable than any one of them.

Averaging only helps when the models are **good but different**. Three identical models
would just reproduce the same mistakes — averaging them changes nothing. The diversity is
the entire point. That's why we deliberately use three different *libraries* rather than
three LightGBMs with slightly different seeds.

### The key engineering decision before writing a line

The risk in this script isn't the maths — it's that **each library has a different API for
two things**: (1) how you tell it which columns are categorical, and (2) how you turn on
early stopping. Get those wrong and the script crashes after minutes of training. So the
disciplined move (which we actually did) was to **smoke-test all three APIs on a tiny
sample first** — 8000 rows, a few seconds — to confirm each model trains and returns
probabilities of the right shape *before* committing to the full ~10-minute run. This is
the "always give yourself a way to check your work" principle, applied early and cheaply.

### What changed in structure (and why)

The baseline is one flat top-to-bottom script — perfect for reading once. The ensemble is
organized into small **functions** instead. Why the upgrade?

- **DRY (Don't Repeat Yourself):** the 5-fold CV loop is identical for all three models, so
  it's written *once* in `cross_validate()` and reused three times. Without functions you'd
  copy-paste that loop three times and they'd drift apart.
- **Testability:** because the logic lives in functions with a `main()` guarded by
  `if __name__ == "__main__":`, the smoke test can `import` those functions and exercise
  them on a small sample *without* running the whole thing. You cannot easily test a flat
  script; you can always test a function.

That `if __name__ == "__main__":` line means "only run `main()` when this file is executed
directly, not when it's imported." It's what lets the smoke test borrow the functions safely.

### The shared engine — `cross_validate()`

This is the heart of the file, and it's just the baseline's loop made reusable:

```python
def cross_validate(name, fit_one, X, y, Xt, predict_prep=None):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), N_CLASSES))
    test_probs = np.zeros((len(Xt), N_CLASSES))
    prep = predict_prep if predict_prep is not None else (lambda f: f)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = fit_one(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx])
        oof[va_idx] = model.predict_proba(prep(X.iloc[va_idx]))
        test_probs += model.predict_proba(prep(Xt)) / N_SPLITS
        ...
    return oof, test_probs
```

- It takes `fit_one` — *a function that trains one model on one fold* — as an **argument**.
  This is the clever bit: the loop is identical for every model; only the "how do I train
  one model" step differs. So we pass that step in. (Passing a function into another function
  is called a "higher-order function" — a very useful pattern.)
- `predict_prep` is an optional last-second transform applied before `predict_proba`. Only
  CatBoost needs it (to stringify categoricals); the others pass `None` and get an identity
  function (`lambda f: f`, meaning "return the frame unchanged").
- Everything else — the fold split, filling `oof` at held-out rows, accumulating
  `test_probs` as a running average — is *exactly* what `stellar.py` does. You've seen it.

### The three `fit_*` functions — where the libraries differ

Each is tiny and does one job: build one model and train it on one fold. They look almost
identical; the differences are the API quirks we smoke-tested:

- `fit_lgb` — uses a **callback** for early stopping (`lgb.early_stopping(80)`), and accepts
  the `category` dtype natively. Identical to the baseline's inner training step.
- `fit_xgb` — early stopping and the metric are set in the **constructor**
  (`early_stopping_rounds=80`, `eval_metric="mlogloss"`), and it needs
  `enable_categorical=True` to accept category columns.
- `fit_cat` — early stopping is a constructor arg too, but CatBoost wants categoricals as
  **plain strings** (not the `category` dtype) and you name them via `cat_features=...` in
  `.fit()`. That's why this function copies the data and does `.astype(str)` on the two
  category columns first. (`cat_predict_frame` does the same at prediction time.)

Same concept (categoricals + early stopping), three different spellings. Learning to read
three libraries' docs for "the same idea, expressed their way" is a core real-world skill.

### The blend — `main()`

```python
oof_blend  = (oof_lgb + oof_xgb + oof_cat) / 3
test_blend = (test_lgb + test_xgb + test_cat) / 3
blend_acc  = accuracy_score(y, oof_blend.argmax(1))
```

- Each `oof_*` is an (n_rows x 3) table of probabilities. Adding three such tables and
  dividing by 3 gives the **element-wise average probability** per class per row. Then
  `argmax(1)` picks the winning class, exactly as in the baseline.
- Crucially, we blend the **OOF** probabilities and score them. That gives an *honest*
  ensemble accuracy on data none of the models trained on — the same trustworthy yardstick
  as before. We print each model's solo CV score next to the blend, so you can literally
  see whether the ensemble earned its keep — and on this dataset it did **not**, at least not
  with equal weights (see the honest result below).

### An honest result: the equal-weight blend LOST to the baseline

When we actually ran this, here is what came out:

| Model | CV accuracy |
|-------|-------------|
| LightGBM | **0.96805** (best single) |
| XGBoost | 0.96764 |
| CatBoost | 0.96601 |
| Equal-weight blend | 0.96763 |

The blend came in *below* the best single model. This is an important, non-obvious lesson
that contradicts the lazy "ensembles always win" intuition: **averaging only helps when the
members are both good AND comparably strong.** Here CatBoost is noticeably weaker (~0.002
below LightGBM), and giving it an equal `1/3` vote dragged the average down past the best
member. A weak voter with a full vote hurts the committee.

The fix is a **weighted blend**: weight each model by how good (and how different) it is —
lots of weight on LightGBM, some on XGBoost (close and diverse), little or none on CatBoost.
The right way to choose those weights is to search them against the **OOF** score, which is
why a good ensemble script **saves the OOF probabilities to disk** — then you can tune blend
weights in milliseconds instead of retraining for ten minutes each time. (That is exactly the
"save OOF predictions for reproducible stacking" item on the README checklist, and the reason
`ensemble.py` writes `oof_*.npy` / `test_*.npy` and then runs a small weight search.)

One honesty caveat: tuning the weights on the same OOF you score on can *slightly* flatter the
result (you're fitting 2 free numbers to that data). With ~577k rows and only two weights it's
negligible here, but the principled version would tune weights inside a nested split.
- The same averaging is applied to the test predictions, and the result is written to
  `submission_ensemble.csv` (kept separate from the baseline's `submission.csv` so you can
  compare both on Kaggle).

### How to take this further (once you've read the run output)

- **Weighted blend:** instead of `/3` (equal weights), give the better models a bigger share
  — e.g. `0.5*lgb + 0.3*cat + 0.2*xgb`. Tune the weights against the OOF score, never the
  leaderboard.
- **Stacking:** feed the three models' OOF probabilities as *inputs* to a small final model
  (a "meta-learner") that learns how best to combine them. More powerful, more complex —
  worth doing only after the simple average is working and measured.
- **More diversity:** add a model from a different family entirely (logistic regression, a
  small neural net). The more genuinely different the members, the more the blend can gain.

The throughline: the ensemble didn't introduce a single *new* idea you hadn't already met
in the baseline. It reused the CV loop, OOF predictions, and probability averaging — just
three times and blended. That's how real projects grow: verified foundation first, then one
well-measured step at a time.

---

## Part 10 — Deep dive: how the model actually learns and predicts

This is the engine room. Parts 0–9 told you *what* the model does; this part shows the exact
arithmetic of *how a tree gets its numbers* and *how those numbers become a prediction*. It's
the most advanced section — read it once you're comfortable with everything above. It answers
three questions people always trip on:

1. Where do the numbers inside the leaves come from? (training)
2. How do those numbers turn into a GALAXY/QSO/STAR decision? (prediction)
3. Does it matter how many rows fall into a leaf? (regularization)

### 10.1 Prediction first (the short recap)

To classify one object, the model:

1. Drops the object's **whole feature vector** through **every** tree (~550 of them).
2. In each tree, forced yes/no questions route it to **exactly one leaf**; it collects that
   leaf's number for each class.
3. **Sums the numbers down each class** (across all trees) → 3 raw scores `(G, Q, S)`.
4. **Softmax** turns the 3 scores into 3 probabilities that sum to 1:
   `p_G = e^G / (e^G + e^Q + e^S)`, and likewise for Q and S.
5. **argmax** picks the biggest probability → the predicted class.

Three things people get wrong here, stated plainly:

- **No single tree or leaf ever holds a final probability.** A leaf contributes a tiny nudge
  (like `+0.33`). The 0.74 kind of number only exists *after summing all trees and softmax*.
- **The prediction uses ALL features, not one threshold.** Two rows that share one feature
  value can still get different predictions, because other trees split on other features and
  route them to different leaves there. A row matches another's prediction only if its full
  feature vector follows the same path in *every* tree.
- **argmax commits to a label even when unsure.** A probability of 0.74 still means "predict
  this class," but ~1 in 4 such rows is genuinely something else. The label is decided; the
  certainty isn't. (Shaving those uncertain cases is what the ensemble does.)

### 10.2 Training: where the leaf numbers come from (a hand-worked 5-tree trace)

The numbers in the leaves are **learned** by repeatedly correcting errors. Here is the full
loop on 4 tiny training rows and 5 trees (= 5 rounds). Two simplifications to keep it readable
(real libraries do the same thing, just scaled): learning rate **η = 0.5** (real is 0.03, but
that needs hundreds of rounds to show movement), and **leaf value = the average "error" of the
rows in it**, where **error = truth − predicted probability**.

The four rows (one feature, `redshift`; note the direction — near-zero = STAR, moderate =
GALAXY, huge = QSO):

| Row | redshift | true class |
|-----|----------|------------|
| A   | 0.4      | GALAXY     |
| B   | 2.8      | QSO        |
| C   | 0.0005   | STAR       |
| D   | 0.7      | GALAXY     |

Every row carries 3 running scores `(G, Q, S)`, all starting at 0. Each round:
**scores → softmax → error → build tree → leaf value = avg error → update scores → repeat.**

**Tree 1.** Scores all 0 → softmax gives `0.333` for every class. Error = truth − 0.333:

| Row (true) | err(G) | err(Q) | err(S) |
|------------|--------|--------|--------|
| A (G)      | +0.667 | −0.333 | −0.333 |
| D (G)      | +0.667 | −0.333 | −0.333 |
| B (Q)      | −0.333 | +0.667 | −0.333 |
| C (S)      | −0.333 | −0.333 | +0.667 |

The tree splits on `redshift` into leaves `{C}` (redshift < 0.01), `{A,D}` (0.01–1.0), `{B}`
(> 1.0). Each leaf's value = average error of its rows. Update `score += η × value = 0.5 ×
value`:

| Row | G | Q | S |
|-----|------|------|------|
| A,D | +0.333 | −0.167 | −0.167 |
| B   | −0.167 | +0.333 | −0.167 |
| C   | −0.167 | −0.167 | +0.333 |

**Trees 2–5** repeat the identical five steps. Here is row A's GALAXY score and probability
climbing as each tree adds its correction (B's QSO and C's STAR track this exactly by symmetry;
D shares A's leaf, so D = A throughout):

| After   | GALAXY score (sum of leaf values) | GALAXY prob | error still to fix |
|---------|-----------------------------------|-------------|--------------------|
| start   | 0.000 | 0.333 | 0.667 |
| tree 1  | 0.333 | 0.452 | 0.548 |
| tree 2  | 0.607 | 0.554 | 0.446 |
| tree 3  | 0.830 | 0.635 | 0.365 |
| tree 4  | 1.013 | 0.696 | 0.304 |
| tree 5  | 1.165 | 0.741 | 0.259 |

Two lessons fall straight out of this table:

- **The per-tree leaf values ARE the `+0.33, +0.27, +0.22 …` corrections** — now you've seen
  them *computed* from errors, not assumed. Summing them is the GALAXY score; softmax turns it
  into 0.741.
- **Each tree's correction shrinks** (the "error to fix" column melts: 0.667 → 0.548 → …).
  Every tree only cleans up what previous trees left wrong. That's *why* boosting needs many
  small trees and why, with the real η = 0.03, it takes hundreds of rounds for the error to
  approach zero.

To predict a new object you **freeze** all these learned leaf values, drop the object through
the frozen trees, sum, and softmax. Training *writes* the leaf numbers; prediction *reads* them.

### 10.3 Does leaf size matter? Regularization and the `G / (H + λ)` formula

A natural question: if two galaxy rows share a leaf but a quasar sits alone in its own leaf,
does the bigger leaf push harder? In the simplified **average-error** version above, **no** —
averaging two identical `+0.667` errors gives `+0.667`, the same as one row, so the count
cancels and the trace stays symmetric.

But **real models don't pure-average — they regularize**, and that breaks the tie. The actual
leaf value in LightGBM/XGBoost is:

```
                G        (sum of the gradients/errors of the rows in the leaf)
leaf value =  -------
              H + λ      (sum of their "confidence" terms  +  reg_lambda)
```

- Numerator `G` sums the errors — grows with more rows.
- Denominator `H` sums a per-row confidence term — also grows with more rows.
- `λ` = `reg_lambda` (we use **2.0**) is a fixed brake added to the denominator.

With `λ = 0`, numerator and denominator both double for 2 rows → they cancel → same value as 1
row (the averaging case). With `λ > 0`, the fixed brake gets **diluted** when more rows pile in.
Worked numbers for round 1 with our `λ = 2` (using per-row error 0.667 and confidence 0.222):

- **Galaxy leaf `{A,D}` (2 rows):** `G = 1.333`, `H = 0.444` → `1.333 / (0.444 + 2) = 0.546`
- **QSO leaf `{B}` (1 row):**     `G = 0.667`, `H = 0.222` → `0.667 / (0.222 + 2) = 0.300`

Same per-row error, but the 2-row leaf pushes `0.546` vs the 1-row leaf's `0.300` — almost
double. So with regularization, **more supporting rows → bigger, more-trusted correction →
higher probability.** The intuition: `λ` says "don't trust a leaf much without enough
evidence," and two rows is more evidence than one.

Leaf size shows up three more ways, all pointing the same direction:

1. **The tree prefers splits that serve bigger groups** (it ranks splits by *total* gain summed
   over rows), so rare patterns may never earn their own leaf.
2. **`min_child_samples = 40`** forbids a leaf built from fewer than 40 rows outright — a
   hard "enough evidence" floor that would have banned our 1-row toy leaves entirely.
3. **Dataset prevalence (the biggest effect):** if one class dominates the data — and galaxies
   do here (~162k predicted vs ~50k QSO, ~35k STAR) — then most leaves in most trees lean that
   way and the model's baseline tilts toward it. A common class genuinely gets higher
   probabilities overall.

So the honest reconciliation: in the **idealized** average (`λ = 0`) leaf size doesn't change
the push; in a **real** model (`reg_lambda = 2.0`, `min_child_samples = 40`, imbalanced
classes) it does. **Regularization is the mechanism that turns "more supporting rows" into
"more confidence."** That single idea separates the toy version from the production algorithm.

---

## Part 11 — Cross-validation vs the leaderboard, and how to know what actually helped

Two ideas that sound the same but aren't, and that together are the difference between real
progress and fooling yourself.

### 11.1 CV and the leaderboard are two *different* scores

You are given two datasets:

| File | Rows | Has the answer (`class`)? |
|------|------|---------------------------|
| `train.csv` | 577,347 | **Yes** — answers included |
| `test.csv`  | 247,435 | **No** — answers hidden by Kaggle |

You only ever *see* the answers for the training data. From that come two scores:

- **Cross-validation (CV)** — *you* compute it, locally, on the **training** data. You hide a
  chunk of `train.csv` from the model, predict it, and grade against answers you already have.
  Free, unlimited, private. It is your **estimate** of how you'll do.
- **Leaderboard (LB)** — *Kaggle* computes it on the **test** data. You submit predictions;
  Kaggle grades them against the **hidden** true answers. Limited (~5/day). It is the **real
  result** on data the model has never seen.

**Analogy:** CV is practising with past exam papers *that come with an answer key* — you grade
yourself, all day. The LB is *sitting the real exam*, graded by the teacher, with an answer key
you never see. The whole point of CV is to **predict your real-exam score before you sit it.**

When the pipeline is clean, CV closely predicts the LB — that's it doing its job. In this
project they *diverged* (CV 0.968 vs LB 0.956) because of **concept drift**: the train and test
rows follow slightly different label rules, so practising on the training papers overstated the
real exam. See [`docs/diagnostics.md`](docs/diagnostics.md).

### 11.2 When you change something and the score moves — what do you keep?

The hard question: if the score goes up, how do you know *which* part of your change caused it,
and what to keep vs discard vs modify? The answer is **controlled experimentation.**

**You can only credit a result to a change if you isolate that change.** Change five things at
once and the LB rises — you've learned nothing reusable, because any of the five (or a lucky
cancellation of good and bad ones) could be responsible. So:

1. **Fix a baseline** — the same reference every time (here: LightGBM, 27 features).
2. **Change exactly one thing** — one feature set, or external data, or one hyperparameter.
3. **Measure the delta against that baseline.** That delta *is* the effect of that one change.
   (This is an **ablation**, and it's exactly why `src/experiments.py` tests each idea as a
   separate variant against the identical baseline.)

**The decision rule:**

| Result of the one-variable test | Decision |
|---|---|
| Beats the score by **more than the noise floor**, and you know *why* | **Keep** |
| Moves **less than the noise floor** (~0.003 CV, ~±0.001 LB) | **Discard** — indistinguishable from luck |
| **Hurts** the score | **Discard** |
| Helps, and you have a hypothesis to do it better | **Modify** — as a *new* one-variable test |

The **noise floor is the gatekeeper**: a change is good only if the score moved *more than
random wobble could explain*. The ensemble's +0.00004 CV "gain" was below noise — and was
actually worse on the LB.

**The catch:** a gradient-boosted model is thousands of tree rules. You **cannot** inspect one
learned rule and label it good or bad — you validate the **whole model as a unit** via CV/LB,
and use *feature importance* (what it leaned on) and *error analysis* (where it fails) as
indirect windows. So "keep vs discard" operates on **your changes** (features, data, params),
never on individual learned rules. You steer the black box from the outside.

**The trap:** even an LB *increase* can fool you. The public LB is itself a ~49k-row sample
(±0.001 noise), so a tiny bump can be luck that vanishes on the private set ("leaderboard
overfitting"). Two defences: demand a delta bigger than that noise, and **prefer changes with a
mechanism you understand** — "external data helped because it denoises the labels" is
trustworthy; "the number went up and I don't know why" reverses on you.

One more distinction worth holding: **methodology lessons** (CV is optimistic here; the noise
floor is ~0.003; blending didn't help) are validated by the *process* and you keep them
permanently. **Individual model changes** are validated one at a time by controlled CV/LB
comparison. Don't confuse the two.
