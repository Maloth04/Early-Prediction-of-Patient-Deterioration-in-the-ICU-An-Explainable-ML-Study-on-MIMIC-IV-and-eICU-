# Changelog

Notable changes to the analysis code. Figures reported in the thesis are fixed at
`v1.0.0`; anything that would change a reported number is recorded here with what
it changes and what it does not.

## [1.1.0] — 2026-10-15

Catches the repository up with corrections made to the thesis after the first
release. Nothing here changes a figure the thesis reports.

### Added
- `analysis/cohort_description/appendix_e_table.py` and
  `results/cohort_description/appendix_e_table.json`: Table E.1, the modelled
  cohort's predictor distributions and observation rates by outcome, which
  Appendix D had recorded as an undelivered TRIPOD+AI commitment and Appendix E
  now delivers in part. Ten tests pin the figures and the two claims the appendix
  makes from them — that the physiological separation between outcome groups is
  modest, each group's median sitting inside the other's inter-quartile range,
  and that the separation in whether a variable was measured is not: lactate is
  recorded in 20.1 per cent of non-event windows against 45.9 per cent of event
  windows, and the median time since the last one falls from 42.78 hours to 22.17.
- The 40-stay reconciliation in `docs/data-access.md`. The cohort flow starts at
  73,141 where the v2.2 `icustays` table holds 73,181 rows; two exclusions applied
  before the flow begins account for the difference, and an unexplained 40 reads
  as a silent filter.
- Six decisions in `docs/decisions-log.md`, including the replacement failure
  region for §5.8 and why the Appendix E retention failure is recorded without
  becoming a third reproducibility limitation.

### Changed
- `docs/deviation-register.md`: the modelled-cohort baseline table moves from
  "Not delivered" to "Delivered in part, in Appendix E", with the demographic half
  still recorded as undelivered because age, sex and first care unit are not
  recoverable from the retained matrices.
- `docs/reproducibility.md`: a third retention failure is disclosed — the Appendix
  E script was not kept either — and kept deliberately separate from the two
  limitations. The GRU fill rule is unknowable; Table E.1's procedure is fully
  determined by its own caption, so the published script is an exact
  reconstruction and the figure still reproduces from data.

### Known, and unchanged
- The thesis still states **two** limitations, not three, and that remains
  accurate: only those two stop a figure reproducing.

## [1.0.0] — 2026-10-11

First public release: the code accompanying the submitted thesis, reorganised into
a repository with tests and CI.

### Added
- `src/icu_deterioration/`: the functions several analyses share — Hanley–McNeil
  variance, the registered H3 union statistic, NEWS2 scoring, config validation and
  run provenance — lifted from the scripts so they can be tested without data.
- 79 tests that need neither PhysioNet access nor a deep-learning stack, including
  a recomputation of the Appendix B.4 precision calculus and checks that every
  recorded result file still says what the thesis reports.
- CI that lints, tests on Python 3.11–3.13, and fails the build if a data-shaped or
  credential-shaped file is ever committed.
- `config.yaml` validated on load: the feature lists must imply the declared
  predictor count of 81, and the GRU input shape must match the source variables.

### Fixed
- **NEWS2 band boundaries** in `analysis/benchmark/news2_construction.py`. Band
  membership was resolved with `side="right"`, placing a value lying exactly on a
  bound into the next band; the published bounds are inclusive and the vitals are
  charted as integers, so 13.9 per cent of windows took the wrong score. The
  original is kept as it ran, because Section 4.3.5 quotes its output, with the
  defect recorded at the top of the file; the corrected implementation is in
  `src/icu_deterioration/news2.py` and `news2_construction_corrected.py`.

  What changes: the worst-in-24h median score 7 → 8 and the urgent-response trigger
  share 85.9 → 88.1 per cent; the end-of-window gain 0.021 → 0.024 AUROC, so the
  implied increment over the benchmark is 0.162 rather than 0.165.

  What does not change: every conclusion in Section 4.3.5. The saturation is still
  a property of the construction (88.1 → 16.0 per cent under an end-of-window
  score), the bedside-style trigger still enriches (1.69 against a 0.79 per cent
  base rate), and discrimination still barely moves between constructions.
- `record_run()` no longer raises when a package in its list is absent, when a
  hashed file is unreadable, or when the output directory is nested. It recorded
  provenance and lost the whole record on any of these; it is called after an
  experiment finishes, so that failure cannot be retried.
- `record_run()` now records whether the working tree was dirty. A commit hash
  written while uncommitted changes were present names code that was never run.

### Known, and not fixed
- Two defects in the eICU endpoint mapping (`build_eicu.py`), published as they ran
  so the reported counts reproduce from the code. Appendix B.3 gives the corrected
  rule a future execution should use. See `docs/reproducibility.md`.
