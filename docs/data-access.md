# Getting the data

No data is distributed here, and none can be shared on request. Both databases
require you to obtain your own access.

## MIMIC-IV v2.2

Credentialed access through PhysioNet, which takes a few days:

1. Create a PhysioNet account and complete the CITI "Data or Specimens Only
   Research" training.
2. Submit a credentialing application with a reference.
3. Sign the Credentialed Health Data Use Agreement for
   [MIMIC-IV](https://physionet.org/content/mimiciv/).

**Use the versioned name.** The thesis is built on v2.2 throughout. An unversioned
BigQuery alias resolves to v3.1, which is a different cohort and will not reproduce
the counts reported here. `config.yaml` pins
`physionet-data.mimiciv_v2_2_icu` and `physionet-data.mimiciv_v2_2_hosp`, and
`tests/test_config.py` fails if either loses its version.

The cohort flow these datasets should reproduce:

| Stage | Stays |
|---|---|
| Extracted from `icustays` | 73,141 |
| At least 24 hours | 57,702 |
| Long enough for a scoreable 4-hour window | 52,128 |
| Minus escalations before the landmark | **25,400 modelled** |

which yields 263,675 prediction instances and 2,080 events at four hours, a
prevalence of 0.79 per cent.

## eICU

The [demonstration release](https://physionet.org/content/eicu-crd-demo/) (2,520 unit
stays, 186 hospitals) is openly licensed under the ODbL and needs no credentialing.
It is what `analysis/external_validation/build_eicu.py` runs against, and it yields
27 post-landmark events at four hours — an order of magnitude below the 266 the
registered criterion requires, which is why no transport claim is made.

The [full release](https://physionet.org/content/eicu-crd/) needs the same PhysioNet
credentialing as MIMIC-IV. It is roughly eighty times the size of the demonstration
subset, so on a proportional projection it would clear the event-count precondition.

## Where to put it

Extract to `data/`, which is git-ignored except for this file. Paths at the top of
each script point there and need editing for a new environment.

```
data/
  mimic_harm_4h.csv      harmonised per-window features, 4-hour horizon
  mimic_harm_8h.csv      the same at 8 hours
  eicu_features_4h.csv   eICU, built by build_eicu.py
  ...
```

## What must never be committed

- **Record-level data**, in any format. The Data Use Agreement forbids
  redistribution, and this is the obligation you personally signed.
- Anything derived from it that could identify an individual.
- Credentials: BigQuery service-account keys, PhysioNet tokens, anything similar.
  Use environment variables.

Aggregate results are fine, and are why `results/` exists: every script writes its
output beside itself so a reader can check a reported number without access to
anything.

CI enforces this. A tracked file with a data-shaped extension outside `results/`,
or a credential-shaped filename, fails the build. Treat a failure there as the
check working, not as an obstacle.
