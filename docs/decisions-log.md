# Decisions log

One line per decision that would be hard to reconstruct later: what was decided,
when, and why. The deviation register records what changed against the proposal;
this records why, including for decisions that were not deviations.

Started retrospectively on 2026-10-11, which is itself the lesson — see
[archiving-checklist.md](archiving-checklist.md).

| Date | Decision | Reason |
|---|---|---|
| 2026-10-15 | Publish the Appendix E script as a labelled reconstruction rather than leave it out | The original was not retained, but Table E.1's caption fully determines every step, so a reconstruction is exact rather than a guess. Omitting it would leave the appendix with no code at all; passing it off as as-run would be a false claim. The header says which it is. |
| 2026-10-15 | Keep the thesis at "two limitations", not three, despite the Appendix E retention failure | The two in `reproducibility.md` stop a figure reproducing. This one does not, given the data. Counting them together would flatten a distinction a reader needs, and the Declaration's cross-reference stays accurate. The failure is recorded in `docs/deviation-register.md` regardless. |
| 2026-10-15 | State a replacement failure region for §5.8's withdrawn falsification line | The revised expectation (≈0.76) sat above the original 0.75 threshold, so that line no longer discriminated and was withdrawn. A prediction with no failure region is a description. The replacement — external four-hour AUROC at or above 0.78, the internal 0.783 within its own error — is dated to the revision, not to the proposal. |
| 2026-10-15 | Stop calling DeLong's test "independent" in the methods chapter | §4.3.3 had already shown the two tests are different estimators of the variance of the same statistic on the same predictions, and that both treat windows as independent when they are nested within patients. §3.6 asserted what §4.3.3 denies. |
| 2026-10-15 | Define "registered", "pre-specified" and "locked" once, at first use | No public pre-registration was filed; the approved proposal is the only record. The words appear over a hundred times, and a reader who meets them before Appendix D will read more into them than they carry. One definition is cheaper and clearer than a hundred substitutions. |
| 2026-10-15 | Reconcile the 73,141 first row against the 73,181 in `icustays` in the data doc, not only in the thesis caption | Anyone with PhysioNet access who checks the first number finds a 40-stay discrepancy. The two pre-flow exclusions explain it; an unexplained 40 looks like a silent filter. |
| 2026-10-11 | Keep `analysis/` scripts exactly as they ran, defects included; corrections go in `_corrected` companions | The thesis cites their output. Editing them in place would make the repository disagree with the document it is the record of. |
| 2026-10-11 | Fix the NEWS2 band boundaries in the shared package but not in the original script | 13.9 % of windows scored differently. §4.3.5's conclusions survive, two of its quoted figures do not; both states are now reproducible and the difference is in `CHANGELOG.md`. |
| 2026-10-11 | CI installs the core only, not `requirements.txt` | The pinned environment pulls torch and the CUDA stack. Every test runs without a deep-learning stack or PhysioNet data, so CI stays under a minute and runs on three Python versions. |
| 2026-10-11 | Make "no patient data committed" a CI job, not just `.gitignore` | `.gitignore` is a convenience and can be bypassed with `git add -f`. The DUA is an obligation, so it gets a check that fails the build. |
| 2026-10-11 | Validate `config.yaml` on load rather than trusting it | A single source of truth that nothing checks is a comment. The feature lists must imply 81 predictors and the GRU shape must match the 14 source variables. |
| 2026-10-11 | `record_run()` never raises | It is called after an experiment finishes, so a crash loses provenance that cannot be regenerated. A missing package is recorded as missing. |
| 2026-10-06 | Submit with the eICU execution reported as a pipeline test, not as a test of H2b | 27 post-landmark events against the 266 the registered criterion requires. Treating it as confirmatory would have been the one indefensible move available. |
