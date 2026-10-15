#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table E.1: modelled-cohort predictor distributions and observation rates by outcome.

TRIPOD+AI expects a baseline table for the cohort the models were actually fitted
on, broken down by outcome. Table 4.2 describes the 24-hour eligible cohort of
57,702 stays, which is a different and higher-acuity population, because the
modelled cohort has every stay whose endpoint occurred before the landmark removed
from it. Appendix D recorded that table as undelivered; Appendix E supplies the
part of it the retained feature matrices support.

READ THIS BEFORE TRUSTING THE FILE
----------------------------------
The script that produced the figures in Appendix E was NOT retained. This file is
a reconstruction, written after the fact against the procedure Table E.1's own
caption states. It is a reconstruction and not the original, and it is labelled as
one rather than presented as the as-run script the rest of ``analysis/`` holds.

Unlike the GRU sequence construction (Section 3.5), where the fill rule for
unobserved hours is not recoverable and so no amount of care reproduces the
original, every step of Table E.1 is fully determined by the caption: the
within-window mean of each variable, its median and inter-quartile range across
instances, the percentage of instances in which the variable was observed at all,
and the median and IQR of the time since its last observation over the instances
where it was observed. Given the same feature matrix this reconstruction is exact,
not an approximation. That is the reason it is published rather than left out, and
``docs/deviation-register.md`` records the retention failure.

The recorded output in ``results/cohort_description/appendix_e_table.json`` holds
the figures from the ORIGINAL run, which are the figures Appendix E reports. If you
re-run this script on your own extraction and a figure differs, the difference is
in the extraction, not in the arithmetic; raise an issue with both numbers.

Data
----
Needs the modelled-cohort feature matrix at the four-hour horizon, the same file
the decomposition in Section 4.8.4 consumes: one row per prediction instance, the
53 physiological and 28 measurement-behaviour columns, and the binary endpoint.
Under the PhysioNet Data Use Agreement that file is not in this repository and
must not be. See ``docs/data-access.md``.

Usage
-----
    python analysis/cohort_description/appendix_e_table.py <feature_matrix.csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The ten predictors Table E.1 reports: seven vitals and three laboratory values.
# The column stems are the feature-matrix naming convention, and the suffixes are
# the ones the mechanical block rule in Section 4.8.4 partitions on.
VARIABLES = [
    ("heart_rate", "Heart rate, bpm"),
    ("resp_rate", "Respiratory rate, breaths/min"),
    ("spo2", "Oxygen saturation, %"),
    ("sbp", "Systolic blood pressure, mmHg"),
    ("dbp", "Diastolic blood pressure, mmHg"),
    ("map", "Mean arterial pressure, mmHg"),
    ("temp_c", "Temperature, degrees Celsius"),
    ("lactate", "Lactate, mmol/L"),
    ("creatinine", "Creatinine, mg/dL"),
    ("wbc", "White cell count, 10^9/L"),
]

MEAN_SUFFIX = "_mean"
MISSING_SUFFIX = "_missing"
RECENCY_SUFFIX = "_time_since_obs_h"
ENDPOINT = "endpoint"


def _median_iqr(values: pd.Series) -> dict | None:
    """Median and inter-quartile range, or None when nothing was observed."""
    v = values.dropna()
    if v.empty:
        return None
    return {
        "median": round(float(v.median()), 2),
        "q1": round(float(v.quantile(0.25)), 2),
        "q3": round(float(v.quantile(0.75)), 2),
    }


def describe(frame: pd.DataFrame) -> dict:
    """Table E.1 for one outcome group."""
    out = {}
    for stem, label in VARIABLES:
        mean_col = stem + MEAN_SUFFIX
        miss_col = stem + MISSING_SUFFIX
        recency_col = stem + RECENCY_SUFFIX

        row: dict = {"label": label, "n_instances": int(len(frame))}

        # Within-window mean of the variable, summarised across instances. The
        # missingness indicator is 1 where the variable was never observed in the
        # window, so those instances carry no value and are excluded from the
        # distribution rather than counted as zero.
        if mean_col in frame:
            observed = frame[mean_col]
            if miss_col in frame:
                observed = observed.where(frame[miss_col] == 0)
            row["window_mean"] = _median_iqr(observed)

        # Observation rate: the percentage of instances in which the variable was
        # recorded at all within the 24-hour window.
        if miss_col in frame:
            row["observed_pct"] = round(100.0 * float((frame[miss_col] == 0).mean()), 1)

        # Hours since the last observation, over the instances where it was
        # observed. This is the recency feature itself, not a derived quantity.
        if recency_col in frame:
            recency = frame[recency_col]
            if miss_col in frame:
                recency = recency.where(frame[miss_col] == 0)
            row["hours_since_last_obs"] = _median_iqr(recency)

        out[stem] = row
    return out


def main(path: str) -> None:
    frame = pd.read_csv(path, low_memory=False)
    if ENDPOINT not in frame:
        raise SystemExit(f"no {ENDPOINT!r} column in {path}")

    event = frame[frame[ENDPOINT] == 1]
    no_event = frame[frame[ENDPOINT] == 0]

    result = {
        "horizon_hours": 4,
        "unit": "prediction instance",
        "instances_total": int(len(frame)),
        "instances_no_event": int(len(no_event)),
        "instances_event": int(len(event)),
        "no_event": describe(no_event),
        "event": describe(event),
        "provenance": {
            "source": "reconstruction; see the module docstring",
            "input": Path(path).name,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    main(sys.argv[1])
