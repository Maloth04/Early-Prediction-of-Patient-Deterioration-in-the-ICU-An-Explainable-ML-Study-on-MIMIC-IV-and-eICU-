# What to store for every research study

Rule of thumb: someone with your folder and the original data access must be able to rebuild every table and figure without asking you anything. Your ICU thesis had to admit that model files, recalibration coefficients, the sequence-building code and the seed values were not retained. That cost it a transport test (Appendix B.2) and weakened the GRU comparison. This checklist prevents that.

## 1. Folder layout

```
project/
  README.md                 what this is, how to run it, in what order
  requirements.txt          package versions (included)
  requirements-lock.txt     exact freeze from your working machine (create it, see section 3)
  config.yaml               every setting and seed (included)
  record_run.py             writes one metadata file per experiment (included)
  sql/                      cohort and endpoint queries, numbered 01_, 02_, ...
  src/                      windowing, features, models, evaluation, plots
  notebooks/                exploration only, never the only copy of a result
  runs/                     one JSON per experiment from record_run.py
  models/                   saved models and recalibration coefficients
  results/                  tables as CSV, figures as PNG and PDF
  docs/                     protocol, deviation register, decisions log
```

## 2. Files to keep, and why

| Keep | Why | ICU-thesis lesson |
|---|---|---|
| Versioned requirements and lock file | Libraries change results and break code | Software versions must be pinned (Section 3.10) |
| `config.yaml` with the seed | Same settings give the same numbers | Seeds were 'fixed' but never listed |
| Every SQL file, numbered in run order | The cohort is the study | Cohort-flow queries and code mappings were open items |
| ID mapping tables (drug names, itemids, codes) | The endpoint definition lives here | The resuscitation code mapping was not recoverable |
| Feature and sequence-building code | Anyone must rebuild the inputs | The GRU fill rule could not be recovered |
| Saved model files (`.json` / `.pt`) | Needed for any later validation | No serialised model, so models were refitted for eICU |
| Calibration coefficients (Platt slope and intercept) | Needed to score new data on the same scale | Not retained, so no calibration result on the external set |
| Train, validation, test patient ID lists | Proves no leakage and lets you re-run | |
| Raw result tables (CSV) behind every figure | Figures can be redrawn and checked | |
| Protocol and a deviation register | Shows what was planned vs done | Appendix D is one of the thesis's strengths |
| A decisions log (date, decision, reason) | You will forget why | Many proposal versions were revised |
| Hash of each input file (`record_run.py`) | Proves which data version you used | The v2.2 vs v3.1 alias trap |
| Literature library (Zotero or BibTeX file) with the PDFs | Citations stay correct and complete | |

## 3. Setting up the environment (do this once)

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements-lock.txt   # exact versions of everything, keep this file
python record_run.py                 # smoke test: writes runs/smoke_test.json
```

`torch` is large, and larger than it looks: the default install pulls the NVIDIA CUDA stack
(several GB) even on a machine with no GPU. For CPU-only work install it from PyTorch's CPU
index instead, which is a few hundred MB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt            # torch is already satisfied
```

The GRU in the thesis was trained on a single CPU, so nothing here needs a GPU.

**Python version.** `requirements.txt` needs Python 3.12 or newer: numpy 2.5.4, scipy 1.18.1,
xgboost 3.4.1 and shap 0.53.0 all declare `requires-python >=3.12`. The thesis's own code
package documents Python 3.11, so this environment is for new work, not for re-running the
thesis as it was. To rebuild the thesis environment, use 3.11 and let pip choose the
versions rather than these pins.

## 4. Do not store in the repository

- **Patient-level data.** The PhysioNet Data Use Agreement forbids redistributing it and requires secure storage. Store code, IDs of sources, and aggregate results only.
- **Credentials** (BigQuery keys, tokens). Keep them in environment variables or a password manager.
- Large intermediate files you can regenerate. Add them to `.gitignore`, but keep the script that makes them.

## 5. At the end of every experiment

1. Call `record_run('experiment_name', input_files=[...], artefacts=[...])`.
2. Commit the code with a clear message so the git commit recorded matches.
3. Save the tables as CSV in `results/` before making any figure.
4. Add a line to the decisions log for anything that departed from the protocol.
5. Back up the whole folder to a second place (university cloud or an external drive).

## 6. Before you submit or publish

- Re-run from the config on a clean machine, or at least a new virtual environment, and confirm the headline numbers match.
- Check the reporting guidelines for your design (TRIPOD+AI for prediction models) and note any items not met.
- Write the README so a stranger can follow it: what the data is, what to run first, what each script produces.
