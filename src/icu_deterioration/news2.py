"""NEWS2 reconstruction, five components, under three window constructions.

Section 4.3.5 scores NEWS2 from the WORST observation in each 24-hour window. That
is not how a bedside team applies it: a team scores the most recent observation set.
The distinction decides two different claims, and they come apart:

* The saturation IS the construction. Median 7 and 85.9 per cent of windows at the
  urgent-response trigger fall to 2 and 16.0 per cent under an end-of-window score,
  which then enriches at 1.70 per cent against a 0.79 per cent base rate.
* The discrimination is NOT. Paired over five patient-level folds the end-of-window
  score gains 0.021 AUROC (SD 0.013, five folds of five), so the increment over the
  benchmark falls from 0.186 to about 0.165 and the argument stands.

Supplemental oxygen and level of consciousness are not extracted in this pipeline,
so this is a five-of-seven-component reconstruction throughout.

Lifted from ``analysis/benchmark/news2_construction.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Royal College of Physicians (2017) bands. ``cuts`` are INCLUSIVE upper bounds in
# ascending order and ``points`` has one more entry, the score above the final bound.
# Inclusive is the whole point: the published table reads "respiratory rate <=8 scores
# 3", and respiratory rate is charted as an integer, so exactly 8 is a common value and
# must score 3. That is why the lookup below uses side="left"; side="right" would push
# every exact boundary value into the next band.
# SpO2 is Scale 1 and one-sided: only desaturation scores.
COMPONENT_BANDS: dict[str, dict] = {
    "rr": {"column": "resp_rate", "cuts": [8, 11, 20, 24], "points": [3, 1, 0, 2, 3]},
    "spo2": {"column": "spo2", "cuts": [91, 93, 95], "points": [3, 2, 1, 0], "one_sided": True},
    "sbp": {"column": "sbp", "cuts": [90, 100, 110, 219], "points": [3, 2, 1, 0, 3]},
    "hr": {"column": "heart_rate", "cuts": [40, 50, 90, 110, 130], "points": [3, 1, 0, 1, 2, 3]},
    "temp": {"column": "temp_c", "cuts": [35.0, 36.0, 38.0, 39.0], "points": [3, 1, 0, 1, 2]},
}

CONSTRUCTIONS = ("worst", "mean", "end")

# The window is 24 hours; the OLS slope is per minute and the window mean sits at
# the mean observation time, taken as the midpoint, so the final minute is this
# many minutes later.
_HALF_WINDOW_MINUTES = 720.0


def _band(values: np.ndarray, cuts: list[float], points: list[int]) -> np.ndarray:
    """Score each value by band. Unobserved stays NaN and is resolved by the caller."""
    out = np.full(values.shape, np.nan)
    seen = ~np.isnan(values)
    out[seen] = np.asarray(points)[np.searchsorted(cuts, values[seen], side="left")]
    return out


def news2_score(frame: pd.DataFrame, construction: str = "worst") -> np.ndarray:
    """Five-component NEWS2 for every window in ``frame``.

    Args:
        frame: per-window features. Needs ``<var>_min``/``_max`` for ``worst``,
            ``<var>_mean`` for ``mean``, and both plus ``<var>_trend`` for ``end``.
        construction: ``worst`` (Section 4.3.5, the worst value in the window),
            ``mean`` (the window mean), or ``end`` (the fitted value at the window's
            final minute, clipped to the observed range — an estimate of the final
            observation set, not the set itself, since raw timestamps were not
            retained).

    Returns:
        One score per row. An unobserved component contributes 0, as a scoring team
        would leave it blank.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"construction must be one of {CONSTRUCTIONS}, got {construction!r}")

    columns: dict[str, np.ndarray] = {}
    for name, spec in COMPONENT_BANDS.items():
        base, cuts, points = spec["column"], spec["cuts"], spec["points"]
        required = [f"{base}_mean"] if construction == "mean" else [f"{base}_min", f"{base}_max"]
        if construction == "end":
            required += [f"{base}_mean", f"{base}_trend"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise KeyError(f"{construction!r} needs {missing} for component {name!r}")

        if construction == "mean":
            columns[name] = _band(frame[f"{base}_mean"].to_numpy(float), cuts, points)
            continue

        low = np.fmin(frame[f"{base}_min"].to_numpy(float), frame[f"{base}_max"].to_numpy(float))
        high = np.fmax(frame[f"{base}_min"].to_numpy(float), frame[f"{base}_max"].to_numpy(float))
        if construction == "worst":
            if spec.get("one_sided"):
                columns[name] = _band(low, cuts, points)
            else:
                columns[name] = np.fmax(_band(low, cuts, points), _band(high, cuts, points))
        else:  # end
            fitted = (
                frame[f"{base}_mean"].to_numpy(float)
                + frame[f"{base}_trend"].to_numpy(float) * _HALF_WINDOW_MINUTES
            )
            columns[name] = _band(np.clip(fitted, low, high), cuts, points)

    return np.asarray(pd.DataFrame(columns).fillna(0.0).sum(axis=1), dtype=float)
