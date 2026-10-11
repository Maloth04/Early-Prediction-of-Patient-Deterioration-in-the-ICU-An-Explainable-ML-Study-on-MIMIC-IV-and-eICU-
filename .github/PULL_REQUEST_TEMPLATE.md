## What this changes

<!-- One or two sentences. -->

## Does it move a reported figure?

- [ ] No — no number in the thesis changes.
- [ ] Yes — and `CHANGELOG.md` records the old value, the new value, the cause, and
      what conclusion it does and does not change.

<!-- If yes, give the before and after here too. -->

## Checks

- [ ] `ruff check . && ruff format --check src tests` passes
- [ ] `pytest -q` passes
- [ ] No patient-level data, in any format, is added
- [ ] No credentials are added
- [ ] Scripts under `analysis/` are unchanged, or the change is a `_corrected`
      companion with the original left as it ran
