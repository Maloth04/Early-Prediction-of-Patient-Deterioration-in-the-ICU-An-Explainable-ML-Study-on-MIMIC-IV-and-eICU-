"""Load ``config.yaml`` and fail loudly when it disagrees with itself.

The config is the single source of truth for every setting. That is only worth
anything if something checks it, so the feature lists and the declared predictor
count are reconciled on load: ``expected_predictor_count`` must equal what the
``features`` block actually describes, and the GRU input shape must match the
number of source variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(ValueError):
    """The config is internally inconsistent, or a required key is missing."""


def _require(cfg: dict[str, Any], *path: str) -> Any:
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing required key: {'.'.join(path)}")
        node = node[key]
    return node


def predictor_count(cfg: dict[str, Any]) -> int:
    """Predictors implied by the feature lists: physiology plus measurement behaviour."""
    features = _require(cfg, "features")
    n_vitals = len(_require(features, "vitals"))
    n_labs = len(_require(features, "labs"))
    n_vital_stats = len(_require(features, "vital_stats"))
    n_lab_stats = len(_require(features, "lab_stats"))
    n_behaviour = len(_require(features, "behaviour"))
    physiology = n_vitals * n_vital_stats + n_labs * n_lab_stats
    behaviour = (n_vitals + n_labs) * n_behaviour
    return physiology + behaviour


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the config and validate it against itself.

    Raises:
        ConfigError: on a missing key, a predictor count that does not match the
            feature lists, or a GRU input shape that does not match the variables.
    """
    path = Path(path) if path is not None else DEFAULT_PATH
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} does not contain a mapping")

    declared = _require(cfg, "features", "expected_predictor_count")
    implied = predictor_count(cfg)
    if declared != implied:
        raise ConfigError(
            f"expected_predictor_count is {declared} but the feature lists imply "
            f"{implied}; the built matrix and the config have diverged"
        )

    n_sources = len(_require(cfg, "features", "vitals")) + len(_require(cfg, "features", "labs"))
    shape = _require(cfg, "models", "gru", "input_shape")
    if list(shape) != [_require(cfg, "windowing", "observation_hours"), n_sources]:
        raise ConfigError(
            f"gru.input_shape {shape} does not match "
            f"{_require(cfg, 'windowing', 'observation_hours')} hourly steps "
            f"of {n_sources} source variables"
        )

    acceptable = _require(cfg, "evaluation", "calibration_slope", "acceptable")
    warning = _require(cfg, "evaluation", "calibration_slope", "warning")
    if not (warning[0] <= acceptable[0] < acceptable[1] <= warning[1]):
        raise ConfigError(
            f"calibration bands are not nested: acceptable {acceptable} "
            f"must sit inside warning {warning}"
        )
    return cfg
