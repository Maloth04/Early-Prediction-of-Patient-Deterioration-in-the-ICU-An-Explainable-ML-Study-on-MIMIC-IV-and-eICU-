# What reproduces, and what does not

Read this before reporting that a figure will not reproduce. Two limitations are
known, load-bearing, and stated in the thesis itself.

## What this code does reproduce

| Directory | Reproduces | Thesis location |
|---|---|---|
| `analysis/external_validation/` | The eICU cohort and every count reported from it | §4.12.1, Table 5.1, App. A.5, App. B.3 |
| | The external scoring, both executions, and the discrimination figures | §4.12.2, Table 4.41 |
| | The recency-transport comparison between cohorts | Table 4.42 |
| | The feature-removal measurements across five splits | §4.12.3 |
| | The attribution-rank comparison, including H3's registered union statistic, the centred-attribution recomputation, and the clustered interval §3.6 requires | §4.12.4, Table 4.44 |
| | The precision calculus behind the event-count precondition, and the Monte Carlo check on the variance formula | App. B.1, App. B.4, Table B.2 |
| | The amended H2b benchmark | §4.12.3, App. B.4, Table B.3 |
| `analysis/decomposition/` | The two-block decomposition, its sub-blocks, and the model-class invariance test | §4.8.4–4.8.7, Tables 4.26–4.29 |
| `analysis/benchmark/` | The five-component NEWS2 reconstruction and its comparison | §4.3.5, Figure 4.1 |
| | The three constructions of that score, which separate the saturation from the discrimination | §4.3.5 |
| `analysis/endpoint_validity/` | Deaths as non-events, the corrected endpoint, temporal transport, the disagreement rule and its circularity test, the landmark blanking test | §4.8.8, §4.9.1, §4.10, §4.11 |
| `analysis/sensitivity/` | Imputation and stay-length sensitivity, error analysis, decision curves, subgroup paired tests | §4.4.1, §4.7, §4.8.1–4.8.3, §4.9 |
| `analysis/figures/` | Figures 4.1 to 4.3 from the tables they report | — |

Each script's output is recorded in `results/` under the same directory name, so a
reader can check a reported number without re-running anything and without PhysioNet
access. `tests/test_recorded_results.py` checks those files still say what the thesis
reports; a hand-edited result fails the suite.

### Runs with no data at all

- `analysis/external_validation/precision.py` — every figure in Table B.2 and the
  required-event counts in Appendices B.1 and B.4, from the counts the thesis reports.
- `analysis/external_validation/verify_hm.py` — the Monte Carlo check establishing
  that the Hanley–McNeil variance those figures use is conservative by 11 to 16 per
  cent against a binormal score model, which is why Appendix B.4 reports a bracket
  rather than a point.
- The whole test suite.

## The two limitations

### 1. No trained model was retained

No serialised XGBoost or GRU model and no Platt recalibration coefficients survive
from the Chapter 4 run. Every script that needs a model refits it to the
specification in `config.yaml` and §4.8.4. The refits reproduce the internal figures
closely — `replicate.py` returns 0.769 against Table 4.41's 0.769 — but they are
reconstructions of documented procedure, not the original artefacts.

The consequence is not only inconvenience. Appendix B.2 required the external
validation to be scored from frozen artefacts, and it could not be, so no external
calibration result exists at all: any slope would describe a Platt mapping refitted
for that run rather than the registered model.

### 2. The GRU sequence-construction code was not retained

The 24 × 14 hourly tensors were built outside this package and not kept. So:

- No script here reproduces any GRU figure.
- Which fill rule was used for unobserved hours is not recoverable. §3.5 shows the
  argument holds under all three candidates — zero-fill, carry-forward, or a
  training-split median — because all three leave the pattern of measurement
  implicitly recoverable from the input while encoding it in no channel. The thesis
  does not depend on which was used.

### The H1 confound

Related, and not a code limitation but a design one, recorded here because it is the
first thing a reader asks. The GRU received 24 hourly steps of the 14 source
variables and nothing else: no per-step missingness mask, no time-since-observation
channel, unlike GRU-D, which supplies both deliberately. XGBoost additionally
received all 28 measurement-behaviour predictors, which carry roughly three quarters
of the available signal on their own. The two models were not competing on equal
information, so the H1 comparison measures feature access rather than architecture.
`tests/test_config.py` pins the input shape so the asymmetry stays visible.

## Known defects, published as they ran

Both are kept rather than silently corrected, so the counts in the thesis reproduce
from the code. Corrections live beside the originals.

### eICU endpoint mapping — `build_eicu.py`

Implements Appendix B.3 as written, including the two defects B.3 records:

1. Ventilation onset is taken as the minimum `ventstartoffset` over all of a stay's
   records and only then required to be positive. Zero is the value most records
   carry (4,288 of 5,436) and cannot be read as a charted start, so this discards the
   charted start of the 60 stays whose records also include a zero or negative
   offset, and leaves stays ventilated before unit admission in the cohort.
2. The resuscitation match includes `cardioversion`, which admits chemical and
   rhythm-control cardioversion. Of the 24 rows it adds, 2 are chemical and 11 sit
   under arrhythmia management.

Appendix B.3 gives the corrected rule a future execution should use (Amendments 4
and 5 in B.6). No conclusion in §4.12 depends on the event count, which is an order
of magnitude below the precondition under every variant.

### NEWS2 band boundaries — `news2_construction.py`

Band membership was resolved with `side="right"`, which places a value lying exactly
on a bound into the next band. The published bounds are inclusive ("respiratory rate
≤ 8 scores 3") and the vitals are charted as integers, so exact boundary values are
common: **13.9 per cent of windows take a different score under the correct rule**.
The thesis's own earlier reconstruction (`analysis/benchmark/_fn.py`) uses half-open
intervals and is unaffected.

The original is kept because §4.3.5 quotes its output. The corrected implementation
is `src/icu_deterioration/news2.py`, checked case by case against the published table
in `tests/test_news2.py`, and `news2_construction_corrected.py` regenerates the table.

| | as reported (§4.3.5) | corrected |
|---|---|---|
| Worst-in-24h median score | 7 | 8 |
| Worst-in-24h at the urgent-response trigger | 85.9 % | 88.1 % |
| End-of-window median / trigger share | 2 / 16.0 % | 2 / 16.0 % |
| End-of-window event rate at the trigger | 1.70 % | 1.69 % |
| End-of-window gain over worst-in-24h | +0.021 AUROC | +0.024 AUROC |
| Implied increment over the benchmark | ≈ 0.165 | ≈ 0.162 |

Every conclusion in §4.3.5 survives. The saturation is still a property of the
construction, the bedside-style trigger still enriches against a 0.79 per cent base
rate, and discrimination still barely moves between constructions — which is the
point of the paragraph.

## Environment

`requirements.txt` is the pinned research environment and needs **Python 3.12 or
newer**: numpy 2.5.4, scipy 1.18.1, xgboost 3.4.1 and shap 0.53.0 all declare
`requires-python >= 3.12`. The original thesis code ran on Python 3.11, so these pins
are for new work, not for re-running the thesis as it was; for that, use 3.11 and let
pip resolve.

`pip install torch` pulls the NVIDIA CUDA stack even with no GPU. Nothing here needs
one — the GRU was trained on a single CPU core with under 4 GB of RAM — so install
from the CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`decomposition/decomp.py` notes that XGBoost's `colsample_bytree` samples columns by
position, so column order must be preserved to reproduce its figures exactly.
