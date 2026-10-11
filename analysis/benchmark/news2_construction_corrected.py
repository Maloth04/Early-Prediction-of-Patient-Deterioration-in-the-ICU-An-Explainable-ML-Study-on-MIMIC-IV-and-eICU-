#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""news2_construction.py rerun with the band bounds read as inclusive.

The original places a value lying exactly on a band bound into the next band; the
published bounds are inclusive and the vitals are integers, so this moves 13.9 per
cent of windows. See the note at the top of news2_construction.py. Scoring here
comes from src/icu_deterioration/news2.py, which tests/test_news2.py checks
against the Royal College of Physicians table case by case.

Needs only the harmonised per-window feature matrix, not the raw database.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icu_deterioration import news2_score  # noqa: E402

MATRIX = "../../data/mimic_harm_4h.csv"   # edit for a new environment
NEWS2_INCREMENT_4H = 0.186                # reported in Table 4.10


def main(path: str = MATRIX) -> None:
    df = pd.read_csv(path)
    y = df.label.values
    print(f"instances {len(df):,}   events {int(y.sum()):,}   "
          f"rate {100 * y.mean():.3f} per cent\n")

    scores = {m: news2_score(df, m) for m in ("worst", "mean", "end")}
    print(f'{"construction":14s}{"AUROC":>8s}{"median":>8s}{">=5 %":>8s}'
          f'{">=7 %":>8s}{"ev@>=5":>9s}{"ev@>=7":>9s}')
    for name, s in scores.items():
        print(f"{name:14s}{roc_auc_score(y, s):8.4f}{np.median(s):8.1f}"
              f"{100 * (s >= 5).mean():8.1f}{100 * (s >= 7).mean():8.1f}"
              f"{100 * y[s >= 5].mean():9.2f}{100 * y[s >= 7].mean():9.2f}")

    folds = list(GroupKFold(n_splits=5).split(df, y, groups=df.subject_id.values))
    auroc = {m: np.array([roc_auc_score(y[te], scores[m][te]) for _, te in folds])
             for m in scores}
    print("\npaired over five patient-level folds, same folds for every construction")
    print(f'  worst-in-24h AUROC       {auroc["worst"].mean():.4f} '
          f'(SD {auroc["worst"].std(ddof=1):.4f})')
    for m in ("mean", "end"):
        d = auroc[m] - auroc["worst"]
        print(f"  {m:5s} minus worst       {d.mean():+.4f} (SD {d.std(ddof=1):.4f}), "
              f"{int((d > 0).sum())} of 5 folds positive")
    gain = (auroc["end"] - auroc["worst"]).mean()
    print(f"\n  increment over the benchmark: {NEWS2_INCREMENT_4H:.3f} reported for "
          f"worst-in-24h, {NEWS2_INCREMENT_4H - gain:.3f} implied for end-of-window")
    print("  (the difference is measured here and applied to the reported increment; "
          "the original per-window score vectors were not retained)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else MATRIX)
