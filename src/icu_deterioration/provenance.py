"""Call record_run() at the end of EVERY experiment. It writes one JSON file that lets you
(or an examiner) rebuild the exact environment, settings and inputs of that run.

This records provenance, so it is written never to raise: a missing package, an unreadable
input or a machine without git degrades one field instead of losing the whole record. It is
called after the work is done, and losing the record of a finished run is the one failure
mode that cannot be retried.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "xgboost", "torch", "shap"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_sha256(path):
    """A hash, or why there isn't one. Never raises."""
    try:
        return sha256(path)
    except Exception as e:
        return f"unreadable: {type(e).__name__}"


def versions(packages=PACKAGES):
    """Installed version per package; 'not installed' rather than an exception."""
    out = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
        except Exception as e:
            out[p] = f"unknown: {type(e).__name__}"
    return out


def git_state(cwd=None):
    """Commit plus whether the working tree was clean. A commit hash recorded while
    uncommitted changes were present names code that was never run, which is worse
    than recording nothing, so the dirty flag travels with the hash."""

    def run(args):
        return subprocess.check_output(
            ["git", *args], text=True, cwd=cwd, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run(["rev-parse", "HEAD"])
    except Exception:
        return {"commit": "not a git repository", "dirty": None, "branch": None}
    try:
        dirty = bool(run(["status", "--porcelain"]))
    except Exception:
        dirty = None
    try:
        branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        branch = None
    return {"commit": commit, "dirty": dirty, "branch": branch}


def record_run(
    name, config_path="config.yaml", input_files=(), artefacts=(), outdir="runs", extra=None
):
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg_hash = safe_sha256(config_path)
    except Exception as e:
        cfg, cfg_hash = f"unreadable: {type(e).__name__}", None

    info = {
        "run_name": name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "python_executable": sys.executable,  # which interpreter, not just which version
        "platform": platform.platform(),
        "packages": versions(),
        "git": git_state(),
        "config_path": str(config_path),
        "config_sha256": cfg_hash,  # the file, as well as its contents
        "config": cfg,
        "inputs_sha256": {str(p): safe_sha256(p) for p in input_files},
        "artefacts_sha256": {str(p): safe_sha256(p) for p in artefacts},
    }
    if extra:
        info["extra"] = extra

    out = Path(outdir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)  # parents=True: outdir may be nested
    out.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    missing = [p for p, v in info["packages"].items() if v == "not installed"]
    if missing:
        print(f"note: not installed, recorded as such: {', '.join(missing)}", file=sys.stderr)
    if info["git"].get("dirty"):
        print(
            "WARNING: uncommitted changes present; the commit recorded is not the code that "
            "ran. Commit, then re-run.",
            file=sys.stderr,
        )
    return out


if __name__ == "__main__":
    print(record_run("smoke_test"))
