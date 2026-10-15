# Early Prediction of Patient Deterioration in the ICU

### An Explainable ML Study on MIMIC-IV / eICU

[![CI](https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-/actions/workflows/ci.yml/badge.svg)](https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)
[![Data: PhysioNet DUA](https://img.shields.io/badge/data-not%20redistributed-lightgrey.svg)](docs/data-access.md)

Analysis code accompanying the Master's thesis of the same title.

**Shivaprasad Maloth** · MSc Data Science – Digital Health · SRH University of
Applied Sciences Heidelberg · Supervisor: Dr Delekta Jenny · 2026

---

## Abstract

Machine-learning models for early detection of clinical deterioration are
conventionally reported against chance. In intensive care that benchmark describes
no realistic alternative, because every patient is already under continuous expert
observation. This thesis asks what a model adds beyond what the care process has
already recorded.

Of 73,141 MIMIC-IV v2.2 ICU stays, 25,400 were modelled. A composite
acute-escalation endpoint (vasopressor initiation, invasive ventilation, or
resuscitation) was predicted at 4-hour and 8-hour horizons from a 24-hour window. A
gradient-boosted ensemble was compared against a recurrent sequence model, and the
81 predictors were split by a mechanical rule into 53 physiological values and 28
recording only measurement behaviour.

The ensemble outperformed the sequence model at both horizons (AUROC 0.788 against
0.733 at four hours), but on unequal information: the sequence model saw only the
fourteen hourly values, so the comparison measures feature access rather than
architecture. Attribution rankings were stable within horizons but agreed weakly
across them, so explanation stability is partially supported. Roughly three quarters
of above-chance discrimination is recoverable from measurement behaviour alone, and
three quarters again from physiology alone; the shares sum to over 150 per cent, so
the blocks are largely redundant rather than separable. The endpoint records a
treatment decision, not a patient state, classing 1,493 of 2,061 in-hospital deaths
as non-events.

Geographic external validation was not performed. A clinical prediction model is not
deployed into an empty decision space, and reporting performance against chance
answers the wrong question.

**Keywords** — clinical deterioration; intensive care; machine learning;
explainability; informative missingness; clinical prediction models; MIMIC-IV

---

## 1. Research questions, hypotheses and outcomes

Stated first, because the limits are part of the result. Confirmatory status is as
registered in the approved research proposal; departures are in
[docs/deviation-register.md](docs/deviation-register.md).

| | Question / hypothesis | Outcome |
|---|---|---|
| **RQ1 / H1** | Does a temporal sequence model outperform a static ensemble? | **Not resolved.** XGBoost is superior at both horizons as the two were built, but the GRU received 24 hourly steps of the 14 source variables and nothing else, while XGBoost additionally received all 28 measurement-behaviour predictors. The comparison measures feature access, not architecture. |
| **RQ2 / H2a** | Does discrimination degrade across time within MIMIC-IV? | **Hypothesis not supported; criterion met.** No decay was detected against a size-matched contemporaneous reference: +0.021 AUROC at four hours and +0.011 at eight, with calibration slopes 1.030 and 0.947 inside the registered 0.7–1.3 band. |
| **RQ3 / H2b** | Does the model transport to another institution? | **Not addressed.** The eICU demonstration release yields 27 post-landmark events against the 266 the registered criterion requires. The protocol was executed twice as a pipeline test; no transport claim is made in either direction. |
| **H3** | Are SHAP attribution rankings stable? | **Partially supported.** Median Spearman ρ ≥ 0.996 within each of the four model-by-horizon cells across 200 patient-level resamples, but ρ = 0.528 across horizons (p = 0.017, n = 20) against the pre-specified 0.7. |

Four further results were **not pre-specified** and are labelled exploratory
wherever they appear: the two-block feature decomposition and everything built on
it, the NEWS2 reconstruction, the endpoint-validity work of Section 4.10, and the
trial execution of the external-validation protocol.

---

## 2. Data and cohort

### 2.1 Cohort flow (Table 4.1)

| Stage | n | % of prior stage |
|---|---:|---:|
| ICU stays extracted from MIMIC-IV v2.2 | 73,141 | — |
| ICU stay of at least 24 hours | 57,702 | 78.9 % |
| Long enough for one 4-hour window (≥ 28 h) | 52,128 | 90.3 % |
| Excluded: endpoint already occurred before the landmark | 26,728 | 51.3 % |
| **Modelled cohort, 4-hour horizon** | **25,400** | 48.7 % |
| Modelled cohort, 8-hour horizon | 22,981 | — |

### 2.2 Instances and events (Table 4.3)

| Horizon | Stays | Instances | Events | Stay-level rate | Instance-level rate |
|---|---:|---:|---:|---:|---:|
| 4 hours | 25,400 | 263,675 | 2,080 | 8.19 % | 0.79 % |
| 8 hours | 22,981 | 126,155 | 2,079 | 9.05 % | 1.65 % |

Roughly 10.4 prediction instances per stay at four hours. The 24-hour eligible
cohort has a mean age of 65.1 years (SD 16.5), is 56.5 % male, and has a median ICU
stay of 2.41 days (IQR 1.57–4.43).

---

## 3. Methods

**Design.** Landmark analysis. The first prediction timestamp is 24 hours after ICU
admission; each instance uses the preceding 24 hours as its observation window and
the following *W* hours as its label window. Label windows tile back to back without
overlapping, and a stay exits the risk set on its first positive instance, so each
stay contributes at most one event preceded by zero or more non-events.

**Endpoint.** A composite of vasopressor or inotrope initiation, invasive
ventilation, and resuscitation — the first of the three to occur. It records a
*treatment decision*, not a patient state; Section 4.10 measures the consequences.

**Predictors (81).** Eleven vital-sign channels × 4 statistics (mean, min, max,
ordinary-least-squares trend) plus 3 laboratory channels × 3 statistics = **53
physiological values**; and 14 source variables × 2 indicators (observed in window;
hours since last observation) = **28 measurement-behaviour predictors**. The split
is mechanical, by variable name, with no clinical judgement. `config.yaml` declares
the count and [`tests/test_config.py`](tests/test_config.py) fails if the feature
lists stop implying it.

**Models.** XGBoost on the aggregated feature vector; a GRU on the 24 × 14 hourly
tensor. Class imbalance by class weighting. Scope-matched random hyperparameter
searches (30 trials for XGBoost, 12 for the GRU, against 100 planned).

**Evaluation.** AUROC primary, AUPRC secondary; patient-level splits throughout,
never by window; patient-clustered bootstrap (500 resamples for H1, 200 for SHAP);
a 0.02 AUROC practical-relevance threshold; calibration slope 0.9–1.1 acceptable and
0.8–1.2 a warning.

---

## 4. Results

### 4.1 Discrimination and H1 (Tables 4.6, 4.7)

| Horizon | Model | Untuned | Tuned |
|---|---|---:|---:|
| 4 h | XGBoost (static) | 0.783 | **0.788** |
| 4 h | GRU (temporal) | 0.743 | 0.733 |
| 8 h | XGBoost (static) | 0.747 | **0.762** |
| 8 h | GRU (temporal) | 0.706 | 0.710 |

Paired, patient-clustered, matched test sets:

| Comparison | GRU − XGBoost | 95 % CI | Excludes 0 | Exceeds 0.02 |
|---|---:|---|---|---|
| 4 h, untuned | −0.039 | [−0.057, −0.021] | Yes | Yes |
| 8 h, untuned | −0.040 | [−0.058, −0.021] | Yes | Yes |
| 4 h, tuned | −0.056 | [−0.073, −0.039] | Yes | Yes |
| 8 h, tuned | −0.051 | [−0.067, −0.033] | Yes | Yes |

Both objections to the comparison were tested rather than argued: a hidden-size
sweep with early stopping moved GRU AUROC by 0.003, and scope-matched searches
*widened* the gap at both horizons. What the result cannot settle is the
architecture question, for the input-asymmetry reason above.

### 4.2 Calibration and clinical utility (Tables 4.13, 4.16)

| Horizon | Output | AUROC | Slope | Intercept | Brier |
|---|---|---:|---:|---:|---:|
| 4 h | Raw (class-weighted) | 0.766 | 0.939 | −4.104 | 0.08988 |
| 4 h | Platt-recalibrated | 0.766 | 0.963 | −0.029 | 0.00742 |
| 8 h | Raw (class-weighted) | 0.741 | 0.955 | −3.453 | 0.10816 |
| 8 h | Platt-recalibrated | 0.741 | 1.096 | −0.010 | 0.01533 |

The raw slope of 0.939 would have passed the pre-specified rule while the intercept
of −4.104 and a Brier score worse than the base rate show the model badly
miscalibrated — class weighting is the cause, and the criterion is shown
insufficient on its own.

Translated to the bedside at four hours, 90 % sensitivity costs 674 alerts per 1,000
instances: a PPV of 1.13 %, 89 patients flagged for every escalation caught, and
about **81 alerts a day in a 20-bed unit**.

Median lead time is 1.88 hours at the four-hour horizon, below the pre-specified
2-hour criterion, which is met at eight hours. The criterion is structurally capped
by the window design.

### 4.3 Explanation stability and H3 (Tables 4.18, 4.19)

| Model | Horizon | Median ρ | 95 % CI |
|---|---|---:|---|
| XGBoost | 4 h | 0.996 | [0.989, 1.000] |
| XGBoost | 8 h | 0.996 | [0.982, 1.000] |
| GRU | 4 h | 1.000 | [0.994, 1.000] |
| GRU | 8 h | 0.996 | [0.986, 1.000] |

Two qualifications turn H3 from confirmed to partially supported. Attributions are
recomputed on resampled *instances* under a single fitted model, so the statistic
describes Monte Carlo stability of the estimate, not stability of what the model
learns; a refit-per-resample design would test the latter. And the registered
cross-horizon clause fails, at ρ = 0.528 (p = 0.017, n = 20).

A perturbation ladder establishes that the metric *can* fall — one random
transposition within the top 15 takes it to 0.859, two to 0.751, three to 0.649 —
so the high values are not an artefact of an insensitive measure.

### 4.4 The feature-block decomposition (Tables 4.26, 4.27) — exploratory

| Feature block | Predictors | AUROC 4 h | AUROC 8 h |
|---|---:|---:|---:|
| Physiological values only | 53 | 0.711 ± 0.009 | 0.685 ± 0.013 |
| Measurement behaviour only | 28 | 0.723 ± 0.015 | 0.708 ± 0.009 |
| Full feature set | 81 | 0.783 ± 0.012 | 0.759 ± 0.011 |
| *Behaviour share of above-chance discrimination* | | *78.8 %* | *80.3 %* |

A block containing **no clinical values at all** — only whether each variable was
observed, and how long ago — discriminates at 0.723, above the 53 physiological
values. Within it, recency carries the signal and presence adds little on top:
0.726 ± 0.014 against 0.666 ± 0.018 at four hours.

Three readings are ruled out or bounded rather than asserted. It is not leakage:
features are computed strictly inside the window and the median time since the last
heart-rate observation is 0.534 h for non-events against 0.619 h for events, the
opposite of what truncation would produce. The ordering of the two blocks is **not**
established in either direction — the paired within-split signs are inconsistent in
every condition. And because AUROC − 0.5 is not additive, shares summing past 100 %
is expected arithmetic describing redundancy, not a discovery: the honest statement
is that either block can be dropped for 0.05–0.07 AUROC and neither is necessary.

The same contrast is visible before any model is fitted, in the modelled cohort's
own descriptives (Table E.1, `results/cohort_description/`):

| At the 4-hour landmark | No event | Event |
|---|---:|---:|
| Heart rate, median of window means | 83.5 | 88.5 |
| Systolic pressure, median of window means | 122.6 | 116.6 |
| **Lactate observed at all in the window** | **20.1 %** | **45.9 %** |
| **Median hours since the last lactate** | **42.78** | **22.17** |

The physiological medians sit inside each other's inter-quartile ranges. The
measurement figures do not come close: lactate is ordered more than twice as often
before an event and the last one is less than half as old. Table E.1 is a
description and not a result, which is exactly why it is useful here — the
decomposition is not an artefact of the learner.

### 4.5 Against the clinical benchmark (Table 4.10) — exploratory

| Predictor | AUROC 4 h | AUROC 8 h | AUPRC 4 h |
|---|---:|---:|---:|
| NEWS2, five components | 0.597 ± 0.012 | 0.595 ± 0.009 | 0.0119 |
| Physiology-only model (53) | 0.711 ± 0.009 | 0.685 ± 0.013 | 0.0260 |
| Measurement-behaviour model (28) | 0.723 ± 0.015 | 0.708 ± 0.009 | 0.1107 |
| Full model (81) | 0.783 ± 0.012 | 0.759 ± 0.011 | 0.1166 |
| **Increment over NEWS2** | **+0.186** | **+0.164** | 9.8-fold |

The size of the model's contribution depends entirely on the baseline chosen.
Against NEWS2 the increment is +0.186 and the model looks transformative; against a
comparator built only from the observation record it is +0.060 and the model looks
useful but modest. Both are correct, and the second is the more informative, because
the observation record is generated in the unit whether or not any score is used.

A caveat belongs with the NEWS2 figures and is documented in
[docs/reproducibility.md](docs/reproducibility.md): the score was computed from the
*worst* observation in each 24-hour window, which is not how a bedside team applies
it. The saturation that follows is a property of that rule; the discrimination gap
is not, moving by at most 0.024 AUROC under a bedside-style construction.

### 4.6 Validity of the endpoint (Table 4.34)

| Endpoint status | Survived | Died in hospital | Total | Mortality |
|---|---:|---:|---:|---:|
| No escalation in any window | 21,827 | 1,493 | 23,320 | 6.4 % |
| Escalation | 1,512 | 568 | 2,080 | 27.3 % |
| All | 23,339 | 2,061 | 25,400 | 8.1 % |

**1,493 of 2,061 in-hospital deaths (72 %) are coded as non-events**, because the
patient died without escalation. The negative class therefore contains a meaningful
number of patients with the worst possible outcome. Re-testing the decomposition
against a death-inclusive endpoint reverses the ordering of the two blocks — the
behaviour share falls from 78.8 to 75.2 per cent at four hours and from 80.3 to 72.0
— but it does not collapse, so the phenomenon is not purely an artefact of the
endpoint definition.

### 4.7 Transport

**Temporal (H2a, executed).** No decay against a size-matched contemporaneous
reference: +0.021 AUROC at four hours and +0.011 at eight, calibration slopes 1.030
and 0.947.

**Geographic (H2b, not executed).** The eICU demonstration release (2,520 unit stays
across 186 hospitals) yields 1,455 eligible stays, 1,282 modelled, 17,776 instances
and **27 post-landmark events** at four hours. The precision calculus of Appendix B.4
— reproducible from counts alone with
[`analysis/external_validation/precision.py`](analysis/external_validation/precision.py)
— puts the requirement at 266 events for a one-sample test at 80 % power, and shows
the registered precondition of 150 was itself too low: at 150 events the smallest
detectable fall is 0.067, above the 0.05 the criterion is set at.

The protocol was nonetheless run twice as a pipeline test. Two independent
implementations of the same written protocol reproduced the cohort exactly and the
internal figure to three decimals, yet returned 0.749 and 0.716 externally — a
difference of 0.033 on the one quantity H2b would be judged on. No transport claim
is made.

---

## 5. Limitations

1. **The H1 comparison is confounded by feature access**, not architecture (§3.5).
2. **Geographic external validation was not delivered**, and the frozen artefacts
   Appendix B.2 required do not exist, so no external calibration result is possible
   at all.
3. **The endpoint records a treatment decision**, so part of what the behaviour block
   recovers is the coherence of clinical behaviour with itself.
4. **Terminal windows are dropped and the deletion is not conservative**: discharge is
   associated with non-escalation and death or transfer with the opposite, so the
   censoring is informative and the direction of the resulting bias is not fixed by
   the design. The ≥ 28-hour requirement conditions on the future in the same way.
5. **The landmark design excludes the first 24 hours**, the period of highest acuity.
   Results apply to adult ICU patients who survive the first day event-free.
6. **Subgroup analysis is underpowered.** No claim of equitable performance is made in
   either direction; two findings reported in earlier drafts did not survive
   replication under a paired within-split test and were withdrawn.
7. **The decomposition is exploratory** and conditioned on this feature
   representation and this institution's measurement culture.

---

## 6. Reproducibility

Two limitations are load-bearing and stated wherever they matter:

- **No trained model was retained** — no serialised XGBoost or GRU, no Platt
  coefficients. Scripts needing a model refit to the documented specification;
  `replicate.py` returns 0.769 against the reported 0.769, but these are
  reconstructions of procedure, not the original artefacts.
- **The GRU sequence-construction code was not retained**, so no GRU figure is
  reproduced here and the fill rule for unobserved hours is not recoverable. §3.5
  shows the argument holds under all three candidate rules.

A third artefact was lost to the same habit and is recorded separately because it
is not a third limitation: the script behind Table E.1 was not retained either, but
every step of that table is determined by its own caption, so
`analysis/cohort_description/appendix_e_table.py` is an exact reconstruction rather
than a guess and the figure still reproduces from data. The thesis states two
limitations, and that stays accurate — only those two stop a figure reproducing.
[docs/deviation-register.md](docs/deviation-register.md) records all three, and
[docs/archiving-checklist.md](docs/archiving-checklist.md) is the response: nothing
was archived while it was being produced, so three artefacts went the same way
rather than three different ways.

Two known defects are published as they ran, so the reported counts reproduce from
the code: the eICU ventilation and cardioversion mappings (Appendix B.3 gives the
corrected rule), and a NEWS2 band-boundary error found by this repository's test
suite. [docs/reproducibility.md](docs/reproducibility.md) has the before-and-after
for both; [CHANGELOG.md](CHANGELOG.md) records what each changes and what it does
not.

Every script's output is recorded in `results/` so a reported number can be checked
without re-running anything and without PhysioNet access. The 89 tests in `tests/`
need neither data nor a deep-learning stack and run in about two seconds; they
recompute the Appendix B.4 calculus from first principles, reconcile the cohort
counts, pin the registered thresholds, and check the NEWS2 bands case by case
against the published table.

---

## 7. Repository structure

```
analysis/        the thesis analyses, published exactly as they ran, by area
  external_validation/   eICU cohort, scoring, precision calculus, H3 statistics
  decomposition/         two-block decomposition, sub-blocks, invariance test
  benchmark/             NEWS2 reconstruction and its three constructions
  endpoint_validity/     deaths, corrected endpoint, temporal transport
  sensitivity/           imputation, stay length, error analysis, decision curves
  cohort_description/    Table E.1, the modelled cohort by outcome
  figures/               Figures 4.1–4.3 from the tables they report
results/         each script's recorded output, mirroring analysis/
src/             the functions several analyses share, lifted so they can be tested
tests/           89 data-free checks
docs/            deviation register, reproducibility, data access, decisions log
config.yaml      every setting and threshold, validated on load
```

## 8. Installation and use

```bash
git clone https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-
cd Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Runs with no data at all:

```bash
python analysis/external_validation/precision.py   # Appendices B.1 and B.4
python analysis/external_validation/verify_hm.py   # Monte Carlo variance check
```

For the analyses that need the database, install the pinned research environment
(Python 3.12+; see [docs/reproducibility.md](docs/reproducibility.md) on why it
differs from the 3.11 the thesis ran on):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip CUDA
pip install -r requirements.txt
```

## 9. Data

Not included, and not shareable on request. MIMIC-IV v2.2 requires PhysioNet
credentialed access; the eICU demonstration release is openly licensed under the
ODbL. The Data Use Agreement forbids redistributing record-level data, and CI fails
the build if a data-shaped file is ever committed.
[docs/data-access.md](docs/data-access.md) has the steps and the cohort-flow table to
reconcile against.

## 10. Citation

See [CITATION.cff](CITATION.cff), or GitHub's **Cite this repository**.

> Maloth, S. (2026). *Early Prediction of Patient Deterioration in the ICU: An
> Explainable ML Study on MIMIC-IV / eICU* [Master's thesis]. SRH University of
> Applied Sciences Heidelberg.

## 11. Licence

Code under the [MIT licence](LICENSE). This covers the code only — not the MIMIC-IV
or eICU databases, which are not distributed here and are governed by the PhysioNet
Credentialed Health Data Use Agreement, and not the text of the thesis.

**Not for clinical use.** Geographic external validation was not delivered, the
endpoint records a treatment decision rather than a patient state, and the model was
never prospectively evaluated.
