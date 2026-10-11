# Contributing

This repository is the record of a completed Master's thesis. Issues, questions and
corrections are welcome; feature work mostly is not, because the point of the code is
that it is what produced the reported figures.

## The rule that matters

**A figure reported in the thesis must stay reproducible from this repository.**
If a change would alter a reported number, it does not go in silently. It goes in with
the old value, the new value, what caused the difference, and what conclusion it does
or does not change — in `CHANGELOG.md`, the way the NEWS2 band fix is recorded there.
Scripts under `analysis/` are published as they ran, defects included, because the
thesis cites their output; corrections live beside them under a `_corrected` name.

## Never commit

- **Record-level MIMIC-IV or eICU data**, in any form or any file extension. The
  PhysioNet Data Use Agreement forbids redistribution. CI fails the build on a
  data-shaped file; do not work around it.
- Credentials of any kind. Use environment variables.
- Model checkpoints or other large regenerable artefacts. Keep the script instead.

## Setting up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional, runs the same checks CI runs
pytest -q
```

The test suite needs no data and no deep-learning stack, and runs in about two
seconds. If a change you make cannot be covered by a test that runs without
PhysioNet access, say so in the pull request.

## Before opening a pull request

```bash
ruff check . && ruff format --check src tests && pytest -q
```

Then: does `CHANGELOG.md` need a line? Does a reported figure move? If you touched
`config.yaml`, `tests/test_config.py` pins the registered thresholds deliberately —
a failure there means you changed a value that decides a hypothesis, which needs a
sentence in the pull request, not a test edit.

## Reporting a problem with a reported number

Open an issue using the **Reproduction question** template and give the section or
table, the value you obtained, and how you obtained it. A disagreement with a
reported figure is the most useful issue this repository can receive.
