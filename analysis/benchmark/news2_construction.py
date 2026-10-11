#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the NEWS2 saturation a property of intensive care, or of the 24-hour
worst-value rule?  (Section 4.3.5, the paragraph "How much of this is the
24-hour worst-value rule".)

Section 4.3.5 reconstructs NEWS2 from the WORST observation in each 24-hour
window.  An examiner's objection is that a bedside team scores the most recent
observation set instead, and that taking the worst value over 24 hours will
saturate almost any ICU patient, so the median score of 7 and the 85 per cent of
windows at or above the trigger follow from the construction rather than from
the population.  The objection is right about the saturation and wrong about the
discrimination, and this script is where both halves are measured.

Three reconstructions of the same five-component score on the same instances:

  worst  each component scored at whichever of the window's min/max is worse
         (the construction of Section 4.3.5)
  mean   each component scored at the window mean
  end    each component scored at the window's fitted value at its final minute,
         mean + slope x 720, clipped to the observed range

"end" estimates the final observation set rather than being it: the retained
feature matrix holds per-window summaries, not raw observation timestamps, so
the last charted value is not recoverable.  The OLS slope is per minute and the
mean sits at the mean observation time, taken as the midpoint of the 1440-minute
window, which puts the window's final minute 720 minutes later.

Needs only the harmonised per-window feature matrix, not the raw database.
"""
# -----------------------------------------------------------------------------
# DEFECT, recorded rather than silently corrected, as Appendix B.3 does for
# build_eicu.py. This script scores band membership with
# np.searchsorted(..., side="right"), which places a value lying EXACTLY on a band
# bound into the next band. The published NEWS2 bounds are inclusive ("respiratory
# rate <=8 scores 3") and the vitals are charted as integers, so exact boundary
# values are common: 13.9 per cent of windows take a different score under the
# correct rule. The thesis's own reconstruction (analysis/benchmark/_fn.py) uses
# half-open intervals and is not affected.
#
# This file is kept as it ran, because Section 4.3.5 quotes its output. The
# corrected implementation is src/icu_deterioration/news2.py (side="left", checked
# against the published table in tests/test_news2.py) and the corrected table is
# news2_construction_corrected.py with its output in
# results/benchmark/news2_construction_corrected.out.txt.
#
# What changes:  median 7 -> 8 and 85.9 -> 88.1 per cent at the trigger for the
# worst-in-24h rule; the end-of-window gain 0.021 -> 0.024 AUROC. What does not:
# every conclusion in Section 4.3.5. The saturation is still the construction
# (88.1 -> 16.0 per cent), the bedside-style trigger still enriches (1.69 against
# a 0.79 per cent base rate), and discrimination still barely moves.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

MATRIX   = '../data/mimic_harm_4h.csv'   # edit for a new environment
MODEL_4H = 0.783                          # full 81-predictor model, Table 4.10
NEWS2_4H = 0.186                          # reported increment over the benchmark
BASE     = 0.79                           # per cent, instance-level event rate

# NEWS2 component bands, Royal College of Physicians 2017.  Scale 1 for SpO2.
BANDS = {
    'rr':   ('resp_rate',  [8, 11, 20, 24],          [3, 1, 0, 2, 3]),
    'spo2': ('spo2',       [91, 93, 95],             [3, 2, 1, 0]),
    'sbp':  ('sbp',        [90, 100, 110, 219],      [3, 2, 1, 0, 3]),
    'hr':   ('heart_rate', [40, 50, 90, 110, 130],   [3, 1, 0, 1, 2, 3]),
    'temp': ('temp_c',     [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]),
}


def band(v, cuts, scores):
    out = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    out[ok] = np.asarray(scores)[np.searchsorted(cuts, v[ok], side='right')]
    return out


def reconstruct(df, mode):
    cols = {}
    for key, (base, cuts, scores) in BANDS.items():
        f = lambda v: band(v, cuts, scores)
        lo = np.fmin(df[f'{base}_min'].values, df[f'{base}_max'].values)
        hi = np.fmax(df[f'{base}_min'].values, df[f'{base}_max'].values)
        if mode == 'worst':
            # SpO2 is one-sided: only the low end scores.
            cols[key] = f(lo) if key == 'spo2' else np.fmax(f(lo), f(hi))
        elif mode == 'mean':
            cols[key] = f(df[f'{base}_mean'].values)
        elif mode == 'end':
            v = df[f'{base}_mean'].values + df[f'{base}_trend'].values * 720.0
            cols[key] = f(np.clip(v, lo, hi))
        else:
            raise ValueError(mode)
    # An unobserved component contributes 0, as a scoring team would leave it blank.
    return np.asarray(pd.DataFrame(cols).fillna(0.0).sum(axis=1), dtype=float)


def main():
    df = pd.read_csv(MATRIX)
    y = df.label.values
    print(f'instances {len(df):,}   events {int(y.sum()):,}   '
          f'rate {100 * y.mean():.3f} per cent\n')

    scores = {m: reconstruct(df, m) for m in ('worst', 'mean', 'end')}

    print(f'{"construction":14s}{"AUROC":>8s}{"median":>8s}{">=5 %":>8s}'
          f'{">=7 %":>8s}{"ev@>=5":>9s}{"ev@>=7":>9s}')
    for m, s in scores.items():
        print(f'{m:14s}{roc_auc_score(y, s):8.4f}{np.median(s):8.1f}'
              f'{100 * (s >= 5).mean():8.1f}{100 * (s >= 7).mean():8.1f}'
              f'{100 * y[s >= 5].mean():9.2f}{100 * y[s >= 7].mean():9.2f}')
    print(f'\n(base rate {BASE} per cent; Section 4.3.5 reports median 7, '
          f'85.4 per cent at >=5, 63.1 per cent at >=7, and 0.85 / 0.94 per cent '
          f'event rates at those thresholds)')

    # The level of this reconstruction is not the level reported in Table 4.10
    # (0.597 +/- 0.012 over five splits), so the quantity carried into the thesis
    # is the paired DIFFERENCE between constructions, measured on the same folds.
    print('\npaired over five patient-level folds, same folds for every construction')
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, y, groups=df.subject_id.values))
    au = {m: np.array([roc_auc_score(y[te], scores[m][te]) for _, te in folds])
          for m in scores}
    print(f'  worst-in-24h AUROC          {au["worst"].mean():.4f} '
          f'(SD {au["worst"].std(ddof=1):.4f})')
    for m in ('mean', 'end'):
        d = au[m] - au['worst']
        print(f'  {m:4s} minus worst            {d.mean():+.4f} '
              f'(SD {d.std(ddof=1):.4f}), {int((d > 0).sum())} of 5 folds positive, '
              f'range {d.min():+.4f} to {d.max():+.4f}')
    d_end = (au['end'] - au['worst']).mean()
    print(f'\n  increment over the benchmark: {NEWS2_4H:.3f} reported for '
          f'worst-in-24h, {NEWS2_4H - d_end:.3f} implied for end-of-window')
    print(f'  (the difference is measured here and applied to the reported '
          f'increment; the original per-window score vectors were not retained)')


if __name__ == '__main__':
    main()
