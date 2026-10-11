"""Checks on the five-component NEWS2 reconstruction.

The band edges are checked against the Royal College of Physicians (2017) table,
because a one-off-by-one in a cut point moves every score in Section 4.3.5. The
three window constructions are then checked for the properties the section
depends on: the worst-value rule saturates, and it does not change discrimination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icu_deterioration import news2_score
from icu_deterioration.news2 import COMPONENT_BANDS

VARIABLES = [spec["column"] for spec in COMPONENT_BANDS.values()]

# (respiratory rate, SpO2, systolic BP, heart rate, temperature) -> expected score.
# Hand-worked from the published bands.
PUBLISHED_CASES = [
    ((18, 97, 120, 70, 36.8), 0, "every component normal"),
    ((8, 97, 120, 70, 36.8), 3, "respiratory rate at the bradypnoea bound"),
    ((9, 97, 120, 70, 36.8), 1, "one above that bound"),
    ((25, 97, 120, 70, 36.8), 3, "tachypnoea"),
    ((18, 91, 120, 70, 36.8), 3, "SpO2 at the lowest band's upper bound"),
    ((18, 94, 120, 70, 36.8), 1, "SpO2 mildly low"),
    ((18, 97, 90, 70, 36.8), 3, "hypotension at the bound"),
    ((18, 97, 220, 70, 36.8), 3, "systolic hypertension"),
    ((18, 97, 120, 131, 36.8), 3, "tachycardia"),
    ((18, 97, 120, 40, 36.8), 3, "bradycardia at the bound"),
    ((18, 97, 120, 70, 35.0), 3, "hypothermia at the bound"),
    ((18, 97, 120, 70, 39.5), 2, "pyrexia"),
    ((25, 91, 90, 131, 35.0), 15, "every component at its worst"),
]


def frame_from(values, construction="worst"):
    """One window per row, with whatever columns the construction needs."""
    rows = np.atleast_2d(np.asarray(values, dtype=float))
    data: dict[str, np.ndarray] = {}
    for i, var in enumerate(VARIABLES):
        data[f"{var}_min"] = rows[:, i]
        data[f"{var}_max"] = rows[:, i]
        data[f"{var}_mean"] = rows[:, i]
        data[f"{var}_trend"] = np.zeros(len(rows))
    return pd.DataFrame(data)


@pytest.mark.parametrize("values,expected,why", PUBLISHED_CASES)
def test_band_edges_match_the_published_score(values, expected, why):
    got = news2_score(frame_from(values))[0]
    assert got == pytest.approx(expected), why


def test_an_unobserved_component_contributes_nothing():
    frame = frame_from((18, 97, 120, 70, 36.8))
    for suffix in ("_min", "_max", "_mean"):
        frame.loc[0, f"lactate{suffix}"] = np.nan  # irrelevant column, no effect
    frame.loc[0, "temp_c_min"] = np.nan
    frame.loc[0, "temp_c_max"] = np.nan
    assert news2_score(frame)[0] == 0.0


def test_the_worst_value_rule_never_scores_below_the_window_mean():
    """The saturation argument in Section 4.3.5 rests on this ordering."""
    rng = np.random.default_rng(11)
    n = 300
    centre = {
        "resp_rate": (16, 6),
        "spo2": (96, 3),
        "sbp": (118, 20),
        "heart_rate": (82, 18),
        "temp_c": (36.9, 0.6),
    }
    data: dict[str, np.ndarray] = {}
    for var, (mu, sd) in centre.items():
        mean = rng.normal(mu, sd, n)
        spread = np.abs(rng.normal(0, sd, n))
        data[f"{var}_mean"] = mean
        data[f"{var}_min"] = mean - spread
        data[f"{var}_max"] = mean + spread
        data[f"{var}_trend"] = np.zeros(n)
    frame = pd.DataFrame(data)
    worst = news2_score(frame, "worst")
    mean = news2_score(frame, "mean")
    assert np.all(worst >= mean)
    assert worst.mean() > mean.mean()


def test_a_flat_trend_makes_the_end_of_window_score_the_mean_score():
    rng = np.random.default_rng(3)
    n = 120
    data: dict[str, np.ndarray] = {}
    for var in VARIABLES:
        mean = rng.normal(80, 20, n)
        data[f"{var}_mean"] = mean
        data[f"{var}_min"] = mean - 5
        data[f"{var}_max"] = mean + 5
        data[f"{var}_trend"] = np.zeros(n)
    frame = pd.DataFrame(data)
    assert np.array_equal(news2_score(frame, "end"), news2_score(frame, "mean"))


def test_the_end_of_window_value_is_clipped_to_what_was_observed():
    """An extrapolated value may not leave the observed range: the fitted value is
    an estimate of the final observation, not licence to invent one."""
    frame = frame_from((18, 97, 120, 70, 36.8))
    frame.loc[0, "heart_rate_trend"] = 10.0  # +7,200 bpm over half a window
    frame.loc[0, "heart_rate_min"] = 65.0
    frame.loc[0, "heart_rate_max"] = 75.0
    # Clipped to 75, which is still inside the zero-point band.
    assert news2_score(frame, "end")[0] == 0.0


def test_rejects_an_unknown_construction():
    with pytest.raises(ValueError, match="construction must be one of"):
        news2_score(frame_from((18, 97, 120, 70, 36.8)), "latest")


def test_names_the_column_it_is_missing():
    frame = frame_from((18, 97, 120, 70, 36.8)).drop(columns=["spo2_min"])
    with pytest.raises(KeyError, match="spo2_min"):
        news2_score(frame, "worst")
