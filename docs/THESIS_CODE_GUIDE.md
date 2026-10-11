# Thesis Code Guide

**Early Prediction of Patient Deterioration in the ICU — An Explainable ML Study on MIMIC-IV / eICU**

A complete walkthrough of the analysis code, written assuming no prior knowledge of
Python, machine learning, or statistics. Every term is explained the first time it
appears.

---

## How to read this document

Read Parts 1–3 in order. They build the vocabulary you need, and nothing later makes
sense without them. After that you can jump around.

| Part | What it covers |
|---|---|
| 1 | The research problem, in plain language |
| 2 | Vocabulary — every technical word used anywhere in the code |
| 3 | The pipeline, as six stages from raw hospital database to a number in a table |
| 4 | Five scripts explained line by line — the ones that carry the findings |
| 5 | Every file in the package: what it does, what it produces, which thesis section |
| 6 | The six concepts an examiner is most likely to probe |
| 7 | Viva questions mapped to the exact code that answers them |

**One honest note before you start.** The trained models from the original Chapter 4
run were not saved to disk. Every script here that needs a model rebuilds it from the
specification written down in Chapter 3. The rebuilds reproduce the reported numbers
closely — `external_validation/replicate.py` returns 0.769 against Table 4.41's 0.769 —
but they are reconstructions of a documented procedure, not the original objects. The
thesis states this in Section 4.12.1 and Appendix D. If an examiner asks whether you
can reproduce your results, this is the accurate answer: the procedure reproduces, the
original artefacts were not retained, and Appendix B.6 Amendment 3 records that saving
them is a precondition of any future execution.

---

# Part 1 — The research problem in plain language

## What you were trying to do

Patients in intensive care sometimes get suddenly worse. If you could see it coming a
few hours early, staff could intervene sooner. The question was whether a computer
model, reading only the routine numbers already recorded in the patient's chart, could
flag that deterioration in advance.

## What "deterioration" means here — and why this matters

You cannot measure "deterioration" directly. There is no field in a hospital database
that says *this patient is deteriorating*. So you used a **proxy**: three treatments
that clinicians give when a patient is getting worse.

1. **Vasopressor or inotrope initiation** — drugs started when blood pressure fails
2. **Invasive mechanical ventilation** — a breathing tube, when breathing fails
3. **Protocol-triggered resuscitation** — CPR or defibrillation

If any of the three happens, that counts as an event.

This is the single most important limitation in your thesis, and you should say it
before an examiner says it to you: **the model predicts treatment escalation, not
physiological deterioration.** It predicts the moment a clinician decides to act.
Those are related but not the same thing. Your Section 4.10 investigates the
consequences, and the sharpest one is this: of 2,061 patients who died in hospital,
1,493 never received any of the three treatments, so the endpoint labels them as
*no deterioration*. Patients who died are teaching the model that they were stable.

## The two data sources

**MIMIC-IV v2.2** — a de-identified database from Beth Israel Deaconess Medical Center
in Boston, covering about 73,000 ICU stays. This is where the models were built and
tested. Access requires PhysioNet credentialing and a Data Use Agreement.

**eICU Collaborative Research Database** — a second database, from many US hospitals in
a commercial tele-ICU programme. The purpose was to test whether a model built on
Boston data still works elsewhere. You could only obtain the openly licensed
**demonstration release** (2,520 unit stays), not the full release (~200,859). This is
why hypothesis H2b is reported as *not addressed*; Part 6 explains the arithmetic.

## What the three research questions asked

- **RQ1** — Does a sequence model (GRU) beat a summary-feature model (XGBoost)?
- **RQ2** — Are the model's explanations stable, or do they shuffle around when you
  perturb the data?
- **RQ3** — Does performance survive when the model is moved to a different hospital
  system?

## The finding nobody expected

This is your thesis's real contribution, and it is not any of the three questions above.

The model's inputs split into two kinds:

- **Physiology** — the actual measured values. Heart rate 95. Lactate 3.2.
- **Measurement behaviour** — *not the values*, but the record of when someone chose
  to measure. How long since the last lactate test was drawn. Whether blood pressure
  was charted in this window at all.

You measured each block on its own. At four hours, measurement behaviour alone reached
**0.723 ± 0.015** AUROC and physiology alone **0.711 ± 0.009**. The behaviour block —
which contains no clinical values at all, only the timing of attention — came out
*level with or slightly above* the values themselves.

**Be careful how you say this, because the thesis is careful.** The difference is
+0.012 and it is positive in only **3 of 5 splits**. That is noise. The thesis
therefore does **not** claim behaviour beats physiology; §4.8 states that neither
block is established as outperforming the other once differences are paired within
split. And under the corrected endpoint — death counted as an event — the ordering
actually reverses, to physiology 0.736 against behaviour 0.724.

What *is* established, and it is the finding: **a block containing no clinical values
performs comparably to one containing all of them.** That does not need behaviour to
win. It only needs the two to be close, and they are, across five splits and both
horizons.

The mechanism is not mysterious once stated: a nurse who is worried measures more
often. The timing of measurement encodes clinical judgement that has already happened.
The model is partly reading the staff, not the patient.

And the two blocks overlap heavily — their separate shares of above-chance
discrimination sum to over 150%, which is only possible if they are largely measuring
the same underlying thing. So the correct claim is *not* "three quarters of the model
is clinician behaviour." It is: either block alone recovers roughly three quarters of
the model's above-chance discrimination, and dropping either costs only about 0.05 to
0.07 AUROC, because they are redundant views of one signal.

---

# Part 2 — Vocabulary

## Python, from zero

**Python** is a programming language. A **script** is a text file of Python
instructions, run top to bottom. All files ending `.py` here are scripts.

**`import pandas as pd`** — loads a *library* (a bundle of pre-written code) and gives
it a short nickname. After this line, `pd.something` means "use the pandas tool called
something". The four libraries used here:

| Library | Nickname | What it does |
|---|---|---|
| pandas | `pd` | Tables of data — like Excel, driven by code |
| numpy | `np` | Fast arithmetic on long lists of numbers |
| scikit-learn | `sklearn` | Standard machine-learning utilities |
| xgboost | `xgb` | The model type you used |
| shap | `shap` | Explains what a model is paying attention to |

**A DataFrame** is pandas' table: rows and named columns. `df` is the conventional
name for one. `df['label']` means "the column called label". `df.iloc[tr]` means "the
rows at the positions listed in `tr`".

**An array** (`np.array`) is a long list of numbers that numpy can do arithmetic on
all at once. `np.full(n, np.nan)` creates a list of length `n` filled with `np.nan`.

**`np.nan`** means "Not a Number" — the marker for *no value here*. It is how the code
represents a measurement that was never taken. This matters enormously in your thesis,
because the pattern of `nan` values is itself a predictor.

**A function** is a named, reusable block. `def fit_eval(...):` defines one;
`fit_eval(...)` later runs it. The `return` line says what it hands back.

**A loop** — `for H in ('4h', '8h'):` — runs the indented block below it once per item.
That particular loop runs the whole analysis twice: once for the 4-hour horizon, once
for 8-hour.

**An f-string** — `f'events {n:,}'` — builds text with values slotted in. `:,` inserts
thousands commas; `:.3f` rounds to three decimals. Used only for printing progress.

**`assert`** — a self-check that stops the script if something is wrong.
`assert len(feats) == 81` means "if there aren't exactly 81 features, halt." These are
scattered through the code deliberately: they turn a silent wrong answer into a loud
crash.

**A dictionary** — `{'auroc': 0.769, 'events': 2080}` — labelled storage. `res['auroc']`
retrieves a value by its label. **JSON** is a text format for saving dictionaries; every
`.json` file beside a script is that script's saved output.

## The prediction setup

**ICU stay** — one admission to intensive care. Identified by `stay_id`.

**Patient** — identified by `subject_id`. One patient can have several stays. This
distinction is the heart of the leakage question in Part 6.

**Observation window** — the 24 hours of data the model looks at.

**Landmark** — the moment prediction starts: 24 hours after ICU admission. Before this
point there is not enough history to summarise.

**Prediction time (`t0`)** — a moment when the model makes a forecast.

**Horizon** — how far ahead it forecasts: 4 hours or 8 hours.

**Instance** — one prediction: one stay at one prediction time. One stay produces many
instances. The 4-hour analysis has **263,675 instances** from **25,400 stays**.

**Label** — the right answer for one instance. `1` if an event happened inside this
instance's horizon, `0` otherwise. Only **2,080** of the 263,675 four-hour instances
are labelled 1 — about **0.79%**. This extreme imbalance drives several design choices.

**Feature** — one input number. You have **81**:

| Block | Count | What it is |
|---|---|---|
| Physiology | 53 | mean, min, max, trend of each measured variable |
| Measurement behaviour | 28 | 14 × `_missing` + 14 × `_time_since_obs_h` |

The 14 source variables: heart rate, respiratory rate, oxygen saturation, temperature
(Celsius and Fahrenheit as separate channels), systolic/diastolic/mean blood pressure
in both arterial-line and cuff versions (six channels), lactate, creatinine, white cell
count.

**`_missing`** — 1 if this variable was never measured in the window, 0 if it was.
**`_time_since_obs_h`** — hours since the most recent measurement, looking back over
the *whole stay*, not just the window. These 28 are the measurement-behaviour block.

## The model

**XGBoost** builds many small decision trees in sequence. Each tree is a chain of
yes/no questions ("is lactate above 2.1?"), and each new tree concentrates on the cases
the previous ones got wrong. Several hundred such trees vote to produce a final risk
score. The fixed configuration used everywhere:

```python
PAR = dict(n_estimators=400,      # build 400 trees
           max_depth=4,           # each at most 4 questions deep — keeps them simple
           learning_rate=0.05,    # each tree makes only a small correction
           min_child_weight=5,    # don't split on fewer than ~5 patients
           subsample=0.8,         # each tree sees 80% of rows, picked at random
           colsample_bytree=0.8,  # and 80% of features
           tree_method='hist')    # a fast approximation, needed at this data size
```

`subsample` and `colsample_bytree` deliberately withhold data from each tree. A tree
that sees everything memorises noise; one that sees a random 80% has to find patterns
that generalise. `max_depth=4` and `learning_rate=0.05` serve the same purpose.

**GRU (Gated Recurrent Unit)** — a neural network that reads a *sequence* in order,
carrying a memory forward. XGBoost gets 24 hours flattened into summary numbers; the
GRU gets the hour-by-hour sequence. Comparing them was RQ1.

> **A limitation you must state before an examiner does.** XGBoost received all 81
> features, including the 28 measurement-behaviour ones. The GRU received only the
> hourly physiological sequence. So the difference between them reflects *both*
> architecture and information access, and the architecture effect cannot be isolated.
> Section 4.3.2 establishes this. Your defence is not an apology — it is that you
> diagnosed what the comparison actually measures and said so in your contribution
> claim.

## The measurements

**AUROC** (Area Under the Receiver Operating Characteristic curve) — take one patient
who had an event and one who did not. AUROC is the probability the model scores the
event patient higher. 0.5 is a coin flip. 1.0 is perfect. Your 0.788 at four hours
means: pick a random pair, and 78.8% of the time the model ranks them correctly.

**AUPRC** (Area Under the Precision–Recall curve) — more informative than AUROC when
events are rare. At 0.79% prevalence, a useless model scores about 0.0079. Yours scored
roughly 12.9 times that, which is the honest way to express the gain.

**Calibration** — whether the probabilities mean what they say. If the model says 10%,
do 10% of those patients actually have an event? Measured by **slope** (ideally 1.0)
and **intercept** (ideally 0). Your raw intercept was **−4.104**, badly wrong, for a
reason explained in Part 6.

**Confidence interval** — a range of values consistent with the data. "0.749 (0.640 to
0.847)" means the data cannot distinguish the true value anywhere in that range. The
**width** of the interval is how much you don't know. This number decides your H2b
conclusion.

**Bootstrap** — to find out how uncertain an estimate is: repeatedly resample your
data with replacement, recompute, and look at the spread of answers. **Patient-clustered**
bootstrap resamples whole *patients* rather than individual rows, because rows from one
patient are not independent.

**SHAP (SHapley Additive exPlanations)** — assigns each feature a credit for each
individual prediction, from cooperative game theory. **TreeSHAP** computes this exactly
for tree models like XGBoost. Averaging the absolute SHAP value of a feature across all
instances gives its overall importance, and sorting those gives a ranking. RQ2 asked
whether that ranking is stable.

---

# Part 3 — The pipeline, in six stages

```
   MIMIC-IV database (73,141 ICU stays)
            |
   STAGE 1  |  Select the cohort
            v      73,141 -> 57,702 (stay >= 24 h)
                   57,702 -> 52,128 (stay >= 28 h, so one 4-hour window fits)
                   minus 26,728 that already escalated before the landmark
                   = 25,400 modelled stays
            |
   STAGE 2  |  Define the endpoint
            v      earliest of: vasopressor / ventilation / resuscitation
            |
   STAGE 3  |  Cut each stay into prediction windows
            v      263,675 instances, 2,080 events (0.79%)
            |
   STAGE 4  |  Build 81 features per instance
            v      53 physiology + 28 measurement behaviour
            |
   STAGE 5  |  Split by PATIENT, train, predict, recalibrate
            v
   STAGE 6  |  Measure: AUROC, AUPRC, calibration, SHAP, utility
                   -> the numbers in Chapter 4
```

**Why stage 1 discards so much.** 26,728 of the 28,808 stays that ever escalate do so
*before* the 24-hour landmark — 92.8% of them. That is a property of intensive care, not
a flaw in your data: most crises happen early. It has a consequence you must own: your
model does not predict deterioration among all ICU admissions. It predicts it among
patients who survived 24 hours event-free. That is a narrower population, and early
deterioration is outside what your results describe.

---

# Part 4 — Five scripts, line by line

## 4.1 `external_validation/build_eicu.py` — turning a raw database into a table

This is the conceptual heart of the whole thesis. It builds the eICU side, but the
MIMIC side was built the same way, so understanding this script means understanding
how every number downstream came to exist.

### Setting the rules

```python
VITALS = ['heart_rate', 'resp_rate', 'spo2', 'sbp_nbp', 'dbp_nbp', 'mbp_nbp', ...]
LABS   = ['lactate', 'creatinine', 'wbc']
VARS   = VITALS + LABS
```

Two lists, then joined. Vitals and labs are treated differently later: vitals get a
`_trend` feature, labs do not, because labs are measured too rarely within 24 hours for
a slope to mean anything.

### Finding who is eligible

```python
COHORT = set(los)
print(f'eligible stays (adult, unit stay >= {(OBS+HRS)/60:.0f} h): {len(COHORT)}')
```

A `set` is an unordered collection with no duplicates. `los` holds length of stay per
unit stay. The requirement is 24 hours of observation **plus** the horizon — you cannot
forecast 4 hours ahead of a stay that ends in 2. This returns **1,455** eligible stays.

### Finding the events

```python
VASO = (r'norepinephrine|epinephrine|dopamine|dobutamine|vasopressin|...')
CPR  = r'cardiopulmonary resuscitation|CPR|defibrillation|cardioversion'
```

These are **regular expressions** — text-matching patterns. `|` means "or". The code
searches drug-name and treatment-name fields for any of these words. This is how an
abstract endpoint definition becomes an operation on actual database text.

> **Two defects your thesis records rather than hides.** `cardioversion` also matches
> *chemical* cardioversion and rhythm-control procedures, which are not resuscitation:
> of 24 rows it adds, 2 are chemical and 11 sit under arrhythmia management. And the
> ventilation rule takes the minimum start-time *and then* requires it to be positive,
> but zero is the unpopulated value in 4,288 of 5,436 records — so the rule reads an
> empty field as a clinical fact. Appendix B.3 states both, and Appendix B.6
> Amendments 4 and 5 give the corrected rules for a future execution. The code is
> published **as run**, so the counts in the thesis reproduce from it.

```python
MODELLED = sorted(COHORT - pre)
```

`COHORT - pre` is set subtraction: eligible stays minus those that escalated before the
landmark. **1,455 − 173 = 1,282 modelled stays.** A patient already on vasopressors has
nothing left to predict.

### Cutting stays into windows — the landmark design

```python
rows = []
for s in MODELLED:
    L  = los[s]          # length of this stay, in minutes
    ev = EV.get(s)       # time of this stay's event, or None if no event
    k  = 0               # window counter
    while LANDMARK + (k + 1) * HRS <= L:
        t0  = LANDMARK + k * HRS          # prediction time
        t1  = t0 + HRS                    # end of the prediction window
        lab = 1 if (ev is not None and t0 < ev <= t1) else 0
        rows.append((s, t0, lab))
        if lab:
            break                         # exit on first event
        if ev is not None and ev <= t0:
            break                         # event already happened
        k += 1
```

Nine lines that define the entire prediction task. Taking them one at a time:

- `for s in MODELLED:` — handle one stay at a time.
- `while LANDMARK + (k+1)*HRS <= L:` — keep making windows while a *complete* one still
  fits inside the stay. Partial windows at the end are dropped. **This is informative
  censoring and it is a real limitation**: stays end because patients are discharged,
  transferred or die, and none of those is random. Your thesis lists it as such.
- `t0 = LANDMARK + k*HRS` — window 0 starts at 24 h, window 1 at 28 h, window 2 at 32 h.
  **Non-overlapping tiling**: each window starts where the last ended, so no event is
  counted twice.
- `lab = 1 if (ev is not None and t0 < ev <= t1) else 0` — label 1 only if the event
  falls *strictly after* `t0` and *at or before* `t1`. The strict `<` on the left is the
  leakage guard: an event exactly at the prediction time is not something to predict,
  it is something already happening.
- `if lab: break` — **exit on first event.** Once a stay has its event, stop. Without
  this, a patient who escalates at hour 30 would keep generating instances at 32, 36,
  40 — all labelled 0, all after they had already deteriorated. That would teach the
  model that deteriorated patients are stable.
- `if ev is not None and ev <= t0: break` — a second safety net for the same problem.

Result: **17,776 instances carrying 27 events** on the eICU demo at four hours.

### Building the features

```python
for i, (s, t0) in enumerate(zip(win.stay.to_numpy(), win.t0.to_numpy())):
    got = idx.get((s, var))
    if got is None:
        continue
    T, V = got
    hi = np.searchsorted(T, t0, side='left')      # strictly before t0
    if hi == 0:
        continue
    tso[i] = (t0 - T[hi - 1]) / 60.0
    lo = np.searchsorted(T, t0 - OBS, side='left')
    if hi <= lo:
        continue
    tw, vw = T[lo:hi], V[lo:hi]
    miss[i] = 0
    mean[i] = vw.mean(); mn[i] = vw.min(); mx[i] = vw.max()
    if not is_lab and len(vw) >= 2 and (tw[-1] - tw[0]) > 0:
        tr_[i] = np.polyfit(tw, vw, 1)[0]
```

This loop runs once per instance per variable. It is where the measurement-behaviour
features are born.

- `T, V = got` — `T` is the sorted list of times this variable was measured in this
  stay; `V` the values.
- `hi = np.searchsorted(T, t0, side='left')` — finds where `t0` would slot into the
  sorted times. `side='left'` makes it **strictly before** `t0`. **This single argument
  is the main technical leakage guard in the feature code.** `side='right'` would
  include a measurement taken exactly at the prediction time, letting the model peek at
  the present moment.
- `if hi == 0: continue` — no measurement ever precedes `t0`, so there is nothing to
  summarise; leave every feature as `nan` and move on. `miss[i]` stays at its initial
  value of 1.
- `tso[i] = (t0 - T[hi-1]) / 60.0` — hours since the most recent measurement. Note it
  uses `T[hi-1]`, the last observation *anywhere in the stay*, not just in the window.
  A lactate drawn 40 hours ago still produces a `_time_since_obs_h` of 40. **This
  feature is the single strongest predictor in the model at four hours.**
- `lo = np.searchsorted(T, t0 - OBS, ...)` — the other end of the 24-hour window.
- `if hi <= lo: continue` — measurements exist for this stay, but none inside this
  window. `_missing` stays 1 and the summaries stay `nan`. **This is the deliberate
  design choice that makes the thesis's finding possible**: unobserved means `nan`, not
  a filled-in guess.
- `mean/min/max` — three summaries of the values inside the window.
- `np.polyfit(tw, vw, 1)[0]` — fits a straight line through the points and keeps its
  **slope**: the trend. `1` means degree-1 (a line); `[0]` takes the slope rather than
  the intercept.

> **A units bug worth knowing about, because an examiner may ask how you validated your
> own pipeline.** `tw` is in *minutes*, so the slope is per minute. An earlier version of
> this extraction computed it per hour — a 60× difference. It was caught by comparing
> the inter-quartile ranges of the trend features between the two cohorts and finding
> ratios of 62, 60 and 69 where 1 was expected. The lesson generalises: when two
> implementations disagree by a suspiciously round factor, suspect units.

### The final guard

```python
ref = pd.read_csv('.../engineered_features_4h_FULLSCALE.csv', nrows=1).columns.tolist()
assert set(ref) == set(df.columns), (set(ref) ^ set(df.columns))
df = df[ref]
```

Reads just the header row of the MIMIC feature table, checks the eICU table has exactly
the same column names, then **reorders the eICU columns to match**. `^` is symmetric
difference — it reports which columns differ, so a failure says what is wrong.

This matters more than it looks. XGBoost's `colsample_bytree` samples columns **by
position**. Feed it the same columns in a different order and you get different
answers. Three lines prevent a silent, untraceable discrepancy.

---

## 4.2 `decomposition/decomp.py` — the principal finding

This script produces the result your thesis is actually about.

```python
NSPLIT = 5
BEH = ('_missing', '_time_since_obs_h')
```

Five repetitions, and the two suffixes that define the behaviour block.

```python
beh = [c for c in feats if c.endswith(BEH)]
phy = [c for c in feats if c not in beh]
assert len(feats) == 81 and len(phy) == 53 and len(beh) == 28
```

A **list comprehension** — `[x for x in list if condition]` — reads as "every column
whose name ends in `_missing` or `_time_since_obs_h`". Physiology is everything else.
The `assert` enforces 81 = 53 + 28, so a mis-named column fails loudly.

```python
def fit_eval(Xtr, ytr, Xte, yte, seed):
    pw = (len(ytr) - ytr.sum()) / ytr.sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    del m; gc.collect()
    return p, roc_auc_score(yte, p), average_precision_score(yte, p)
```

- `pw = (len(ytr) - ytr.sum()) / ytr.sum()` — non-events divided by events. At 0.79%
  prevalence this is about 126. Passed as `scale_pos_weight`, it tells XGBoost to treat
  each event as worth 126 non-events. Without it, a model can score 99.2% "accuracy" by
  predicting *no event* always, and learn nothing. **This line has a side effect that
  becomes a finding in Part 6.**
- `m.fit(Xtr, ytr)` — the training step. The model sees features and answers together
  and adjusts itself.
- `m.predict_proba(Xte)[:, 1]` — predict on *unseen* data. Returns two columns (chance
  of 0, chance of 1); `[:, 1]` takes the second.
- `del m; gc.collect()` — free the memory. Housekeeping, not statistics: 400 trees ×
  5 splits × 3 blocks × 2 endpoints × 2 horizons will exhaust memory otherwise.

```python
for s in range(NSPLIT):
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s)
                  .split(df, y_spec, g))
```

**The most important line in the entire codebase.** `GroupShuffleSplit` with
`groups=g`, where `g = df['subject_id'].values`, guarantees that **every instance from
one patient lands entirely in training or entirely in testing — never split across
both.**

Why this is non-negotiable: consecutive instances from one patient share 20 of their 24
observation hours. They are near-copies. If one lands in training and its near-twin in
testing, the model has effectively seen the test answer, and the reported AUROC is
inflated by an amount you cannot estimate. Using plain `train_test_split` here instead
of `GroupShuffleSplit` would invalidate every number in Chapter 4.

`random_state=100 + s` gives splits 100, 101, 102, 103, 104 — different each round,
reproducible every run.

```python
for key, y in (('specified', y_spec), ('corrected', y_corr)):
    for bname, cols in (('full', feats), ('phys', phy), ('beh', beh)):
        p, a, ap = fit_eval(df.iloc[tr][cols], y[tr], df.iloc[te][cols], y[te], 100 + s)
```

Two nested loops: two endpoint definitions × three feature blocks = six models per
split, thirty in total per horizon. `df.iloc[tr][cols]` is "training rows, these columns
only" — that subsetting is what makes a block-restricted model.

The `corrected` endpoint comes from earlier in the script:

```python
died = d.deathtime.notna() & (d.deathtime > lm) & (d.deathtime <= hz)
y_corr = ((y_spec == 1) | died.values).astype(int)
```

"Died inside this instance's horizon" added to the event definition — a direct test of
the 1,493-deaths problem. `&` is and, `|` is or, `.astype(int)` turns True/False into
1/0.

```python
d_bp = np.array(res[key]['beh']['auroc']) - np.array(res[key]['phys']['auroc'])
e['beh_minus_phys'] = dict(mean=float(d_bp.mean()), sd=float(d_bp.std()),
                           splits_positive=int((d_bp > 0).sum()), n=len(d_bp))
```

Subtracts the two blocks **within each split, then counts the signs.** This is the
right design and an earlier version got it wrong. Comparing block A's average across
splits to block B's average hides the fact that split-to-split variation can exceed the
difference you are measuring. Pairing within split removes the shared variation;
`splits_positive` then reports in how many of five the ordering actually held. An effect
present in 3 of 5 splits is noise, however nice the average looks.

```python
bs = (np.array(res[key]['beh']['auroc']) - .5) / (np.array(res[key]['full']['auroc']) - .5)
```

The **share of above-chance discrimination**. Subtracting 0.5 from both is essential:
AUROC starts at 0.5 for a useless model, so the amount of *skill* in 0.723 is 0.223,
not 0.723. Comparing raw AUROCs would wildly overstate every share. Doing it correctly
is what produces shares summing to over 150% — and that sum is itself the finding, since
it proves the blocks are redundant rather than additive.

---

## 4.3 `external_validation/score.py` — the external test and the calibration problem

```python
tr_i, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
                  .split(dev, groups=g))
sub = dev.iloc[rest]
cal_r, te_r = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
                   .split(sub, groups=sub.subject_id.values))
cal_i, te_i = rest[cal_r], rest[te_r]
```

A **three-way** patient-level split: 60% train, then the remaining 40% halved into 20%
calibration and 20% test. Three groups because three distinct jobs are needed, and
reusing data across them biases the result:

- **train** — fit the model
- **calibration** — fit the probability correction
- **test** — measure, on data used for neither

```python
p_cal = m.predict_proba(dev.iloc[cal_i][cols])[:, 1]
platt = LogisticRegression(max_iter=1000).fit(logit(p_cal).reshape(-1, 1),
                                              dev.label.values[cal_i])
```

**Platt scaling.** The `scale_pos_weight` line in Part 4.2 fixed the learning problem
and broke the probability scale: by telling the model each event is worth 126
non-events, you made it behave as though events were 126 times commoner than they are.
Its rankings are fine — AUROC only cares about order — but its probabilities are
inflated by roughly that factor. Hence a calibration intercept of **−4.104**.

Platt scaling fits a small logistic regression that maps distorted scores back onto real
probabilities, learned on the calibration split only. `logit(p)` converts a probability
to log-odds, the scale on which this correction is linear.

> **The methodological point worth making in your viva.** Your raw calibration slope was
> 0.939, which *passes* the pre-specified 0.8–1.2 criterion. It should not have. Slope
> alone cannot detect a uniform multiplicative inflation — it measures whether
> probabilities are too *spread out*, not whether they are too *high*. Only the
> intercept of −4.104 reveals it. You caught a case where a model passes its registered
> calibration test while its probabilities are unusable. That is a finding about the
> criterion, not just about your model.

```python
yex = ext.label.values
raw_e, cal_ex = scored(ext[cols])
```

The external cohort, **scored exactly once**. No retraining, no refitting, no threshold
reselection. Scoring external data more than once and keeping the better run is a way
of fitting to it, which destroys the thing external validation exists to measure.

```python
lo, hi, w = hanley_ci(auc, int(y.sum()), int((1 - y).sum()))
```

Computes a confidence interval from the Hanley–McNeil formula. On eICU this returns a
width of **0.207** — and that one number is why H2b is unanswerable. See Part 6.

---

## 4.4 `external_validation/h3_union.py` — whether the explanation transports

```python
def union_rho(si, se, k=15):
    oi = list(np.argsort(-si)); oe = list(np.argsort(-se))
    u = sorted(set(oi[:k]) | set(oe[:k]))
    return float(spearmanr(si[u], se[u]).statistic), len(u)
```

This small function computes **the statistic your thesis registered for H3**, and
getting it right mattered more than anything else in the external analysis.

- `si`, `se` — each feature's mean absolute SHAP value, internally and externally.
- `np.argsort(-si)` — sorts *descending* (the minus flips the order) and returns
  positions, giving the importance ranking.
- `oi[:15]`, `oe[:15]` — the top fifteen features of each ranking.
- `set(...) | set(...)` — the **union**: every feature in either top fifteen. Not the
  intersection. The union is what penalises a feature *entering or leaving* the top
  fifteen, which is precisely the kind of disagreement that matters.
- `spearmanr(si[u], se[u])` — rank correlation over just those features.

**Why over the union and not over all 75.** Because a correlation over 75 features is
dominated by the many unimportant ones whose order both cohorts happen to share. The
arithmetic is checkable: exchanging two features inside the top fifteen moves a 75-item
rank correlation by about **0.001**, where the same exchange moves a 15-item statistic
to **0.857**. The 75-feature number is nearly blind to disagreement at the head of the
list — which is the only place disagreement matters.

The result:

| Horizon | union rho | registered bar | verdict |
|---|---|---|---|
| 4 hours | 0.882 (0.870–0.915) | > 0.7 | **met** |
| 8 hours | 0.057 (0.025–0.102) | > 0.7 | **fails** |

0.057 against a bar of 0.7 is not a near miss. At four hours the model explains itself
the same way in a second hospital system; at eight hours it does not.

```python
boot.append(union_rho(si, A[ii].mean(axis=0), F)[0])
```

Inside the bootstrap loop: resample patients, recompute the external SHAP means,
recompute the statistic. Repeating 500 times gives the interval. Note the registered
decision rule is on the **lower bound** of that interval, not the point estimate —
Section 3.6 states it — which is why the intervals are reported and not just the numbers.

```python
for tag, fn in (('mean_abs', lambda S: np.abs(S).mean(axis=0)),
                ('centred',  lambda S: np.abs(S - S.mean(axis=0)).mean(axis=0))):
```

Two ways of measuring importance, and the second answers an objection to the first. A
`lambda` is a one-line unnamed function.

- **`mean_abs`** — average absolute SHAP. The conventional choice.
- **`centred`** — average absolute *deviation from the feature's own mean*.

The objection: a feature that is nearly constant in eICU can carry a large, nearly
constant SHAP value and so look important while separating no patients at all. Your
Table 4.42 shows this is a live risk — the continuous-vitals recency features have an
inter-quartile range of just 3 minutes on eICU's fixed 5-minute charting grid. Centring
charges each feature only for its variation, so a constant contributes nothing.

What it showed: the behaviour share survives (32.97% internally against 31.96%
externally, so the near-equality is real and not an artefact of constant attributions),
but the four-hour rank statistic falls from 0.882 to 0.726 with a lower bound of 0.629 —
**below the bar.** So the four-hour agreement holds on the registered statistic and does
not hold on the stricter one. Your thesis says exactly that, rather than quoting only
the favourable number.

---

## 4.5 `external_validation/precision.py` — why H2b cannot be answered

No model, no data. Pure arithmetic, and it carries the most consequential conclusion in
Appendix B.

```python
def hm_var(auc, npos, nneg):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    return (auc*(1-auc) + (npos-1)*(q1-auc**2) + (nneg-1)*(q2-auc**2)) / (npos*nneg)
```

The Hanley–McNeil (1982) variance of an AUROC estimate. What it encodes: uncertainty is
governed overwhelmingly by the number of **events**, not the number of rows.
17,776 instances sound like plenty; 27 events are not.

```python
AUC_INT, DROP = 0.769, 0.05
AUC_EXT = AUC_INT - DROP
INT_POS, INT_INST = 401, 52279
R_EICU = (17776 - 27) / 27          # 657.4 non-events per event
```

Every input is a registered criterion or a reported figure. `AUC_EXT = 0.719` is the
*alternative hypothesis* — the AUROC that would sit exactly at the tolerated fall. A
power calculation evaluates variance there, not at the null.

```python
w27 = 2 * 1.96 * np.sqrt(hm_var(0.749, 27, 27 * R_EICU))
print(f'analytic 95% width at 27 events, AUROC 0.749: {w27:.3f}')
print(f'measured bootstrap width, Section 4.12.1:     0.207   -> difference {abs(w27-0.207):.3f}')
```

**The formula is validated before it is trusted.** It returns 0.214 where your actual
patient-clustered bootstrap returned 0.207 — a difference of 0.007. Only then is it used
to make claims about sample sizes you do not have. An examiner who asks "how do you know
your power calculation is right?" gets this check as the answer.

```python
while lo < hi:
    mid = (lo + hi) // 2
    se = np.sqrt(v_int + hm_var(AUC_EXT, mid, mid * EXT_RATIO))
    if DROP / se >= z:
        hi = mid
    else:
        lo = mid + 1
```

A **binary search** for the required event count: guess the midpoint, check whether that
many events give enough power, discard the half that cannot contain the answer, repeat.
It converges in about 15 steps instead of testing every integer.

### What it found, and why it strengthens your thesis

| Events | 95% interval width | smallest detectable fall |
|---|---|---|
| 27 (what you have) | 0.219 | 0.16 |
| 150 (your registered precondition) | 0.093 | 0.067 |
| **272 (actually required)** | 0.070 | 0.050 |
| ~2,150 (full release, projected) | 0.025 | 0.018 |

Three conclusions, in rising order of importance:

1. At 27 events the smallest detectable fall is **0.16**, against a criterion of
   **0.05**. The test cannot be performed. This is not a protocol choice; it is
   arithmetic.
2. **Your registered precondition of 150 events was itself too low.** The criterion
   needs 272. So you did not merely fail to meet a bar — you found the bar was
   mis-specified, and you left the registered figure as written rather than quietly
   correcting it.
3. The full release would clear the corrected bar comfortably. The obstacle is data
   access, plus the separate fact that no serialised model from Chapter 4 survived.

This is the single best thing in your external-validation chapter. Most candidates who
cannot complete an external validation report an absence. You reported an absence *and*
diagnosed that the protocol could not have performed the test even at its own
precondition.

---

# Part 5 — Every file in the package

Each `.py` has its output saved beside it as `.json`, so any reported number can be
checked without re-running anything.

### `external_validation/`

| File | What it does | Thesis |
|---|---|---|
| `build_eicu.py` | Builds the eICU cohort, windows and features from the raw release | §4.12.1, App. B.3 |
| `score.py` | Scores the external cohort once, with recalibration | §4.12.2, Table 4.41 |
| `replicate.py` | A second, independently written execution of the same protocol | §4.12.2 |
| `ablate5.py` | Removes feature groups across five splits to measure what each costs | §4.12.3 |
| `shap_transport.py` | SHAP rankings on both cohorts | §4.12.4 |
| `h3_union.py` | H3's registered union statistic, plus centred attributions | §4.12.4, Table 4.44 |
| `h3_centred_ci.py` | Bootstrap interval for the centred statistic | §4.12.4 |
| `benchmark65.py` | The amended H2b benchmark (the 65- and 61-predictor sets) | App. B.6 Amd. 6 |
| `precision.py` | The sample-size and precision calculus | App. B.4, Table B.2 |
| `verify_hm.py` | Monte Carlo check that the variance formula is conservative | App. B.4 |

**On `replicate.py`.** Written from scratch against the same written rules, it
reproduces the cohort exactly — same 1,282 stays, 17,776 instances, 27 events — and
reproduces the internal AUROC to three decimals, but returns **0.716** externally where
`score.py` returned **0.749**. The cause was never identified.

Treat this as an asset, not an embarrassment. Against the full-model figure those two
results fall on *opposite sides* of the registered 0.05 criterion. That is the sharpest
possible demonstration of why the event count must be met before the criterion is
applied: at 27 events, implementation noise decides the verdict.

### `decomposition/`

| File | What it does | Thesis |
|---|---|---|
| `decomp.py` | The two-block decomposition — the principal finding | §4.8.4, Tables 4.26–4.27 |
| `subfix.py` | Breaks behaviour into sub-blocks (labs, cuff, continuous, arterial) | §4.8.5 |
| `invariance.py` | Repeats the decomposition with a different model class | §4.8.7 |
| `mech_burden.py` | Per-variable measurement rates and alarm burden | §4.8.6 |

`invariance.py` matters for a specific reason: it checks the finding is a property of
the *data* and not of XGBoost. If only gradient-boosted trees found behaviour
predictive, that would be a quirk of the algorithm.

### `benchmark/`

| File | What it does | Thesis |
|---|---|---|
| `news.py` | Reconstructs NEWS2, the clinical early-warning score, and compares | §4.3.5, Fig. 4.1 |
| `news_same_splits.py` | Re-runs the comparison on identical splits | §4.3.5 |
| `_fn.py` | The NEWS2 scoring functions (`band`, `news2`), shared by both | — |

> **Wording you must get right.** NEWS2 has seven components; only **five** are
> reconstructable from this data — supplemental oxygen and level of consciousness are
> not available. So never say "the model outperformed NEWS2." Say "the model showed
> higher discrimination than the five-component NEWS2 reconstruction available in this
> dataset." The thesis is already scoped this way; keep it scoped in the room.

### `endpoint_validity/`

| File | What it does | Thesis |
|---|---|---|
| `corrected.py` | Re-runs everything with death counted as an event — the competing-risk test | §4.10.5 |
| `temporal.py` | Trains on 2008–2013, tests on 2014–2019 (this is H2a) | §4.11 |
| `disagree.py` | Alerting targeted at model/clinician disagreement | §4.9.1 |
| `circularity.py` | Tests whether the disagreement rule is circular | §4.9.1 |
| `blank.py` | Blanks data near the landmark to test for leakage | §4.9.1 |

`blank.py` is your leakage defence. It removes data immediately around the landmark and
re-runs. If performance collapsed, the model was reading the event in progress rather
than predicting it. It did not collapse.

`circularity.py` is the kind of self-check examiners notice. The disagreement rule flags
patients the model scores high and the behaviour block scores low — but "low behaviour
score" is close to how unanticipated events were *defined*, so some of the apparent
advantage is built into the construction. The script tests whether the model
discriminates *within* the low-concern group, which is the question that separates a real
finding from a relabelling.

### `sensitivity/`

| File | What it does | Thesis |
|---|---|---|
| `run_sens.py` | Four sensitivity analyses in one script | §4.4.1, §4.7, §4.8.1–4.8.3 |
| `dca_incumbent.py` | Decision-curve analysis, incl. the behaviour-only comparator | §4.9 |
| `equity_paired.py` | Subgroup performance, paired within split | §4.7 |
| `error_analysis.py` | Where the model fails | §4.8.2 |
| `shap_equity.py` | Whether explanations differ across subgroups | §4.7 |

`run_sens.py` covers: the primary model; three ways of handling class imbalance; the
forward-fill versus complete-case comparison; and four minimum stay-length thresholds.

The imputation result is worth knowing cold. Your proposal specified forward-fill —
carry the last value forward to fill gaps. You tested it against complete-case with
explicit missingness indicators and forward-fill was **worse on every measure**. The
reason is now central to your thesis: filling the gaps destroys the measurement-recency
signal. Forward-fill makes an unmeasured patient look measured.

`dca_incumbent.py` carries the most interesting utility result. Against treat-all and
treat-none the model wins clearly. Against the **behaviour-only** comparator the gain is
positive but modest and vanishes above roughly a 5% threshold. That is the honest
statement of the model's incremental value: large against generic rules, small against
what the observation process already contains — and it is exactly what your central
argument predicts.

### `figures/`

`figs.py` draws Figures 4.1–4.3 from the tables that report them, so no figure contains
a number that is not also printed somewhere in the text.

### What is not here

The GRU sequence-construction code. The 24 × 14 hourly tensors were built outside this
package and were not retained, so **no script here reproduces any GRU figure**.
Section 3.5 originally pointed at that script. If an examiner asks for the GRU fill
rule, the answer is that Section 3.5 shows the argument holds under any of the three
candidate rules, so no conclusion depends on which was used — but the script itself is
absent, and the README in the package says so.

---

# Part 6 — Six concepts an examiner will probe

### 1. Data leakage

**What it is.** The model somehow seeing information it would not have at prediction
time. It inflates results and is often invisible.

**Three guards in your code.**
- `GroupShuffleSplit(groups=subject_id)` — patients never span the train/test boundary.
  Consecutive instances share 20 of 24 observation hours, so without this the model sees
  near-copies of its test cases.
- `np.searchsorted(T, t0, side='left')` — features use measurements *strictly before*
  the prediction time.
- `if lab: break` — exit on first event, so no instance is generated after a patient has
  already deteriorated.

**Plus an empirical test**, which is stronger than any argument: `blank.py` removes data
near the landmark and re-runs. Performance holds.

### 2. Class imbalance and what it did to your probabilities

0.79% of instances are events. A model predicting "no event" always is 99.21% accurate
and worthless.

`scale_pos_weight = non_events / events ≈ 126` fixes the learning problem and creates a
second one: the probabilities come out inflated by roughly that factor. Rankings are
unaffected — AUROC only reads order — so discrimination is valid. But the numbers cannot
be shown to a clinician as risks until Platt scaling corrects them.

The sharp version: your raw slope of 0.939 *passed* the registered calibration criterion
while the intercept of −4.104 showed the probabilities were unusable. Slope alone cannot
see a uniform multiplicative shift.

### 3. The landmark, and what it excludes

24 hours of observation means excluding anyone who deteriorates, dies or leaves before
hour 24. That is **92.8%** of all stays that ever escalate.

The answer to "what about early deterioration?" is: *outside the estimand.* The design
deliberately predicts escalation after 24 hours of event-free observation, and the
results should not be read as applying to the first 24 ICU hours.

### 4. Informative censoring

`while LANDMARK + (k+1)*HRS <= L:` drops incomplete windows at the end of a stay. Stays
end for reasons — discharge, transfer, death — that are not random. So the dropping is
not random either.

The answer to "why binary classification instead of survival or competing-risk
modelling?" is: the study was designed as fixed-horizon binary prediction; the analysis
then revealed that discharge and death create informative censoring and that death
without escalation is clinically important; a competing-risk or multi-state formulation
is therefore the logical next step rather than something this design resolves. Your
`corrected.py` quantifies how much it matters — and the answer is that it matters
enough to reverse the block ordering: counting death as an event moves physiology from
0.711 to 0.736 and behaviour from 0.723 to 0.724. Patients who die without escalating
are physiologically abnormal, so adding them helps the block that reads physiology.

### 5. Why H2b cannot be answered — the one-minute version

27 events. Interval width 0.207. Smallest detectable fall 0.16. Criterion 0.05. The
interval is about four times the effect it would have to resolve, and it contains the
internal value, so nothing fell and nothing held.

And the demonstration that settles it: across every defensible combination of internal
reference and execution, **five combinations pass the criterion and four fail**, while
**all nine intervals contain both 0 and 0.05.** The verdict is determined by analysis
choices, not by the data. That is what "cannot be answered" means operationally.

### 6. Post-hoc analyses

Several of your most interesting analyses were not pre-registered: the decomposition,
the NEWS2 reconstruction, the endpoint-validity work, the disagreement rule, the eICU
execution.

The answer to "if these were post hoc, why should we trust them?" is: they are not
treated as confirmatory hypothesis tests. They are labelled exploratory, listed in
Appendix D.3, and their role is to investigate limitations and mechanisms that the
pre-specified results revealed — not to replace the registered hypothesis tests.

Appendix D is an asset. It makes the thesis auditable, and an examiner who finds a
deviation already listed with its reason stops looking for concealment.

---

# Part 7 — Viva questions mapped to code

| Question | Where the answer lives | The short answer |
|---|---|---|
| How did you prevent data leakage? | `decomp.py` L70, `build_eicu.py` `searchsorted`, `blank.py` | Patient-level splits, strictly-before features, exit on first event — plus an empirical blanking test |
| Why XGBoost and GRU? | §3.5 | A strong tabular baseline against a sequence model; the question was whether sequence information adds anything |
| Can you claim XGBoost beats the GRU? | §4.3.2 | No. They had different feature access. The comparison measures access, not architecture — and the thesis says so in its contribution claim |
| Why 4 and 8 hours? | §3.2 | Long enough to act on, short enough that the signal survives |
| Why a 24-hour window? | §3.2, Table 4.1 | Enough history for trends; costs 92.8% of escalating stays, which is stated |
| Why is measurement behaviour predictive? | `decomp.py`, §4.8.5, §5.9 | Measurement frequency encodes clinical concern. Worried staff measure more |
| Is your endpoint really deterioration? | §3.2.1, §4.10 | No — it is treatment escalation used as a proxy. 1,493 of 2,061 deaths are labelled non-events |
| Why AUROC and not only AUPRC? | §3.11, `score.py` | Both are reported. AUPRC with lift-over-prevalence is the honest figure at 0.79% |
| Why did forward-fill change to complete-case? | `run_sens.py` §3, §3.4 | Tested head to head; forward-fill was worse on every measure. It erases the recency signal |
| What does the SHAP stability statistic measure? | `h3_union.py` | Spearman rho over the *union* of the two top-15 lists, against a pre-specified 0.7. The union penalises features entering or leaving the top 15 |
| Why is H3 only partially supported? | §4.6 | Within horizons 0.996 (passes); across horizons 0.528 (fails). Between institutions 0.882 at 4 h, 0.057 at 8 h |
| Why no Bonferroni correction? | §4.7, `equity_paired.py` | Applied where a family of tests is involved; the paired subgroup design uses a Bonferroni threshold |
| Why no full eICU validation? | §4.12, §5.7, App. A, App. B | Access not obtained; the demo subset failed the pre-specified event-count precondition; the protocol's own stopping rule was applied and the decision logged |
| What does the demo dataset prove? | §4.12.1–4.12.4 | The pipeline runs on a foreign schema; the model beats chance in a second system; the explanation transports at 4 h; the full release would clear the corrected bar |
| Can you reproduce your results? | package README, §4.12.1 | The procedure reproduces to three decimals. The original trained objects were not retained, and Amendment 3 records that serialising them is a precondition of any future execution |

---

## The sentence to have ready

> I established the performance and explanation-stability findings within MIMIC-IV, and
> the measurement-behaviour decomposition is the thesis's principal contribution. I did
> not establish geographic transportability. The GRU comparison is informative for the
> models as implemented but cannot isolate architecture, because feature access was not
> matched. The thesis states which of those it is on every page where it matters.

## Environment

Python 3.11 with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `scipy` and
`matplotlib`. Seeds are fixed in every script. Paths at the top of each file point at
the extracted CSVs and need editing for a new machine. Data is not included: MIMIC-IV
requires PhysioNet credentialed access; the eICU demonstration release is openly
licensed under the ODbL.

One warning carried from `decomp.py`: XGBoost's `colsample_bytree` samples columns **by
position**, so column order must be preserved to reproduce the figures exactly. That is
why `build_eicu.py` ends by reordering its columns to match the development matrix.
