# Early prediction of acute treatment escalation in the ICU

[![CI](https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-/actions/workflows/ci.yml/badge.svg)](https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)
[![Data: PhysioNet DUA](https://img.shields.io/badge/data-not%20redistributed-lightgrey.svg)](docs/data-access.md)

Analysis code for a Master's thesis asking what a deterioration model adds **beyond
the care record it sits on top of**, rather than beyond chance. Predicts a composite
acute-escalation endpoint — vasopressor initiation, invasive ventilation, or
resuscitation — four and eight hours ahead, from 24 hours of structured vitals and
laboratory values in MIMIC-IV v2.2.

> Shivaprasad Maloth, *Early Prediction of Patient Deterioration in the ICU — An
> Explainable ML Study on MIMIC-IV / eICU*. MSc Data Science – Digital Health,
> SRH University of Applied Sciences Heidelberg, 2026.

## The finding

Of 73,141 ICU stays, 25,400 contribute a scoreable window: 263,675 prediction
instances carrying 2,080 events, a prevalence of 0.79 per cent. A gradient-boosted
ensemble reaches 0.788 AUROC at four hours.

The result worth the reader's attention is not that number. The 81 predictors split
by a mechanical naming rule into 53 physiological values and 28 that record only
**measurement behaviour** — whether each variable was observed in the window, and how
long ago. The behaviour block contains no clinical values at all, and on its own it
recovers roughly three quarters of the model's above-chance discrimination.
Physiology alone recovers roughly three quarters too. The two shares sum to over 150
per cent, so the blocks are largely redundant rather than separable: either can be
dropped for 0.05 to 0.07 AUROC, and neither is necessary.

Against a five-component NEWS2 reconstruction the model gains 0.186 AUROC and looks
transformative. Against a comparator built only from the observation record it gains
0.060 and looks useful but modest. Both are correct, and the second is the more
informative baseline, because the observation record is generated in the unit whether
or not any score is used.

## What this work does not establish

Stated here because a repository is read faster than a thesis, and the limits are
part of the result.

| Question | Status |
|---|---|
| Does a sequence model beat a static one? | **Not resolved.** The GRU received 24 hourly steps of the 14 source variables and nothing else; XGBoost additionally received all 28 measurement-behaviour predictors. The comparison measures feature access, not architecture ([§3.5](docs/reproducibility.md#the-h1-confound)). |
| Does the model transport to another institution? | **Not addressed.** The eICU demonstration release yields 27 post-landmark events against the 266 the registered criterion requires ([Appendix B.4](results/external_validation/precision.out.txt)). The protocol was executed twice as a pipeline test and no transport claim is made. |
| Is the endpoint a patient state? | **No — it is a treatment decision.** 1,493 of 2,061 in-hospital deaths are coded as non-events because the patient died without escalation. |
| Is the decomposition confirmatory? | **No.** It is post hoc and labelled as such wherever it appears. |

## Layout

```
analysis/        the thesis analyses, published as they ran, by area
results/         each script's recorded output, so a number can be checked without re-running
src/             the functions several analyses share, with tests
tests/           79 checks that need no data and no deep-learning stack
docs/            protocol deviations, reproducibility limits, data access, archiving
config.yaml      every setting and threshold, validated on load
```

## Quickstart

Most of this runs with no data at all: the precision calculus, the Monte Carlo check
on the variance formula, and the whole test suite work from the counts the thesis
reports.

```bash
git clone https://github.com/Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-
cd icu-early-deterioration
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                            # 79 tests, about two seconds
```

Then, with no PhysioNet account:

```bash
python analysis/external_validation/precision.py   # Appendix B.1 and B.4, from counts alone
python analysis/external_validation/verify_hm.py   # Monte Carlo check on the variance formula
```

For the analyses that need the database, see [docs/data-access.md](docs/data-access.md)
and install the pinned research environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip the CUDA stack
pip install -r requirements.txt                                      # needs Python 3.12+
```

## Reproducibility, honestly

Two limitations are load-bearing and are stated wherever they matter:

1. **No trained model was retained.** No serialised XGBoost or GRU model and no Platt
   recalibration coefficients survive. Scripts needing a model refit it to the
   specification in `config.yaml`; the refits reproduce the internal figures closely
   (`replicate.py` returns 0.769 against the reported 0.769) but they are
   reconstructions of documented procedure, not the original artefacts.
2. **The GRU sequence-construction code was not retained**, so no GRU figure is
   reproduced here, and which fill rule was used for unobserved hours is not
   recoverable. The argument in §3.5 holds under all three candidate rules.

[docs/reproducibility.md](docs/reproducibility.md) lists what reproduces, what does
not, and the two known defects in the eICU extraction that are published as they ran
rather than silently corrected. New work should start from
[docs/archiving-checklist.md](docs/archiving-checklist.md), which exists because of
the two limitations above.

## Data

Not included and never will be. MIMIC-IV v2.2 requires PhysioNet credentialed access;
the eICU demonstration release is openly licensed under the ODbL. The Data Use
Agreement forbids redistributing record-level data, and CI fails the build if a
data-shaped file is ever committed. See [docs/data-access.md](docs/data-access.md).

## Citing

See [CITATION.cff](CITATION.cff), or use GitHub's "Cite this repository".

## Licence

Code under the [MIT licence](LICENSE). This covers the code only — not the databases,
which are not distributed here, and not the thesis text.
