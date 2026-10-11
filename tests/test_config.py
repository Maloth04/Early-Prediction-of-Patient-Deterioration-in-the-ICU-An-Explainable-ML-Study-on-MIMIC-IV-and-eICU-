"""The config is only a single source of truth if something checks it."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from icu_deterioration import ConfigError, load_config
from icu_deterioration.config import predictor_count

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_shipped_config_validates(cfg):
    assert cfg["project"]["name"] == "icu_deterioration"


def test_the_feature_lists_imply_the_declared_predictor_count(cfg):
    """81 = 11 vitals x 4 stats + 3 labs x 3 stats + 14 variables x 2 behaviour."""
    assert predictor_count(cfg) == 81
    assert cfg["features"]["expected_predictor_count"] == 81


def test_the_physiology_and_behaviour_blocks_split_as_the_thesis_reports(cfg):
    features = cfg["features"]
    n_vitals, n_labs = len(features["vitals"]), len(features["labs"])
    physiology = n_vitals * len(features["vital_stats"]) + n_labs * len(features["lab_stats"])
    behaviour = (n_vitals + n_labs) * len(features["behaviour"])
    assert (physiology, behaviour) == (53, 28)


def test_the_gru_sees_the_source_variables_and_nothing_else(cfg):
    """The input asymmetry Section 3.5 records: 24 hourly steps of 14 values, with
    no missingness mask and no time-since-observation channel."""
    n_sources = len(cfg["features"]["vitals"]) + len(cfg["features"]["labs"])
    assert cfg["models"]["gru"]["input_shape"] == [24, n_sources] == [24, 14]
    assert n_sources * len(cfg["features"]["behaviour"]) == 28  # what it is denied


def test_a_drifted_predictor_count_is_rejected(tmp_path, cfg):
    broken = copy.deepcopy(cfg)
    broken["features"]["expected_predictor_count"] = 80
    with pytest.raises(ConfigError, match="feature lists imply"):
        load_config(write(tmp_path, broken))


def test_dropping_a_variable_without_updating_the_count_is_rejected(tmp_path, cfg):
    broken = copy.deepcopy(cfg)
    broken["features"]["vitals"] = broken["features"]["vitals"][:-1]
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, broken))


def test_a_mismatched_gru_shape_is_rejected(tmp_path, cfg):
    broken = copy.deepcopy(cfg)
    broken["models"]["gru"]["input_shape"] = [24, 13]
    with pytest.raises(ConfigError, match="input_shape"):
        load_config(write(tmp_path, broken))


def test_unnested_calibration_bands_are_rejected(tmp_path, cfg):
    broken = copy.deepcopy(cfg)
    broken["evaluation"]["calibration_slope"]["acceptable"] = [0.7, 1.3]
    broken["evaluation"]["calibration_slope"]["warning"] = [0.9, 1.1]
    with pytest.raises(ConfigError, match="not nested"):
        load_config(write(tmp_path, broken))


def test_a_missing_key_names_itself(tmp_path, cfg):
    broken = copy.deepcopy(cfg)
    del broken["windowing"]["observation_hours"]
    with pytest.raises(ConfigError, match=r"windowing\.observation_hours"):
        load_config(write(tmp_path, broken))


class TestRegisteredThresholds:
    """Values that decide a hypothesis. A silent edit here rewrites a result."""

    def test_calibration_bands(self, cfg):
        slope = cfg["evaluation"]["calibration_slope"]
        assert slope["acceptable"] == [0.9, 1.1]
        assert slope["warning"] == [0.8, 1.2]

    def test_rank_agreement_criterion(self, cfg):
        assert cfg["evaluation"]["shap_stability_rho_threshold"] == 0.7

    def test_practical_relevance_threshold(self, cfg):
        assert cfg["evaluation"]["practical_threshold_auroc"] == 0.02

    def test_bootstrap_resamples(self, cfg):
        boot = cfg["evaluation"]["bootstrap"]
        assert boot["h1_resamples"] == 500
        assert boot["shap_resamples"] == 200
        assert boot["level"] == "patient"

    def test_splits_are_by_patient_never_by_window(self, cfg):
        assert cfg["split"]["unit"] == "patient"

    def test_the_data_version_is_pinned_not_an_alias(self, cfg):
        """An unversioned BigQuery alias points at v3.1, which is a different
        cohort; the thesis is v2.2 throughout."""
        assert cfg["data"]["version"] == "v2.2"
        for dataset in cfg["data"]["bigquery_datasets"].values():
            assert "v2_2" in dataset

    def test_the_landmark_and_observation_window_agree_with_chapter_3(self, cfg):
        windowing = cfg["windowing"]
        assert windowing["observation_hours"] == 24
        assert windowing["landmark_hours"] == 24
        assert windowing["horizons_hours"] == [4, 8]
        assert windowing["min_age"] == 18
        assert windowing["exit_on_first_event"] is True
        assert windowing["tiling"] == "non_overlapping"
