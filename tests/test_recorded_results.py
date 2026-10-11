"""The recorded outputs in ``results/`` must keep saying what the thesis reports.

Every script writes its output beside itself so a reader can check a reported
number without re-running anything and without PhysioNet access. That only works
if the files stay honest, so the counts that reconcile are reconciled here and the
figures the thesis quotes are pinned. A hand-edited result file fails this suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(relative: str) -> dict:
    with open(RESULTS / relative, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def eicu() -> dict:
    return load("external_validation/eicu_cohort.json")


@pytest.fixture(scope="module")
def precision() -> dict:
    return load("external_validation/precision.json")


class TestEicuCohort:
    def test_the_cohort_flow_reconciles(self, eicu):
        assert eicu["eligible_stays"] - eicu["excluded_pre_landmark"] == eicu["modelled_stays"]

    def test_the_counts_the_thesis_reports(self, eicu):
        assert eicu["eligible_stays"] == 1455
        assert eicu["modelled_stays"] == 1282
        assert eicu["instances"] == 17776
        assert eicu["events"] == 27

    def test_the_prevalence_follows_from_the_counts(self, eicu):
        implied = 100 * eicu["events"] / eicu["instances"]
        assert implied == pytest.approx(eicu["prevalence_pct"], abs=5e-4)

    def test_the_event_count_is_far_below_the_precondition(self, eicu, precision):
        """Appendix B.1 requires 150 post-landmark events and Appendix B.4 shows
        the registered criterion needs 266. This is why H2b was not executed."""
        assert eicu["events"] < 150
        assert eicu["events"] < precision["events_eicu_80pct_one_sample"]


class TestPrecisionCalculus:
    def test_precision_improves_monotonically_with_events(self, precision):
        sizes = ["n27", "n29", "n48", "n150", "n266", "n642"]
        widths = [precision[s]["ci_width"] for s in sizes]
        assert widths == sorted(widths, reverse=True)

    def test_the_two_sample_requirement_exceeds_the_one_sample(self, precision):
        for power in ("80", "90"):
            assert (
                precision[f"events_eicu_{power}pct_two_sample"]
                > precision[f"events_eicu_{power}pct_one_sample"]
            )

    def test_at_150_events_the_detectable_fall_still_exceeds_the_criterion(self, precision):
        """Appendix B.1's own precondition was too low: at 150 events the smallest
        detectable fall is 0.067, above the 0.05 the criterion is set at."""
        assert precision["n150"]["mde_one_sample"] > 0.05

    def test_even_the_full_release_cannot_beat_the_internal_floor(self, precision):
        """The internal estimate carries its own error, so a two-sample comparison
        has a floor no external sample size can cross."""
        assert precision["full_release"]["mde_two_sample"] > precision["internal_floor_mde"]
        assert precision["full_release"]["mde_one_sample"] < precision["internal_floor_mde"]

    def test_both_horizons_discriminate_above_chance_on_the_demonstration_release(self, precision):
        for horizon in ("above_chance_4h", "above_chance_8h"):
            assert precision[horizon]["auroc"] > 0.5
            assert precision[horizon]["p"] < 0.05


class TestRankAgreement:
    @pytest.fixture(scope="class")
    @staticmethod
    def h3() -> dict:
        return load("external_validation/h3_union.json")

    def test_the_registered_criterion_splits_the_two_horizons(self, h3):
        assert h3["4h"]["mean_abs"]["rho_union_top15"] > 0.7
        assert h3["8h"]["mean_abs"]["rho_union_top15"] < 0.7

    def test_the_all_predictor_statistic_hides_that_split(self, h3):
        """0.988 against 0.937 looks like agreement at both horizons. It is why
        Section 4.12.4 reports the union statistic instead."""
        assert h3["4h"]["mean_abs"]["rho_all75"] > 0.9
        assert h3["8h"]["mean_abs"]["rho_all75"] > 0.9

    def test_the_behaviour_share_transports_at_four_hours(self, h3):
        entry = h3["4h"]["mean_abs"]
        assert abs(entry["beh_share_internal"] - entry["beh_share_external"]) < 0.01


class TestNews2Constructions:
    """The corrected reconstruction, against the claims Section 4.3.5 rests on."""

    @pytest.fixture(scope="class")
    @staticmethod
    def table() -> dict:
        path = RESULTS / "benchmark" / "news2_construction_corrected.out.txt"
        rows: dict[str, dict[str, float]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0] in {"worst", "mean", "end"}:
                rows[parts[0]] = dict(
                    zip(
                        ["auroc", "median", "ge5", "ge7", "ev5", "ev7"],
                        (float(p) for p in parts[1:]),
                        strict=True,
                    )
                )
        return rows

    def test_the_worst_value_rule_saturates(self, table):
        assert table["worst"]["ge5"] > 85
        assert table["worst"]["median"] >= 7

    def test_a_bedside_style_construction_does_not(self, table):
        assert table["end"]["ge5"] < 20
        assert table["end"]["median"] <= 3

    def test_the_trigger_enriches_only_under_the_bedside_construction(self, table):
        base_rate = 0.789
        assert table["worst"]["ev5"] == pytest.approx(base_rate, abs=0.1)
        assert table["end"]["ev5"] > 2 * base_rate

    def test_discrimination_barely_moves(self, table):
        """The saturation is the construction; the increment over the benchmark
        is not. This is what keeps Section 4.3.5's argument standing."""
        for name in ("mean", "end"):
            assert abs(table[name]["auroc"] - table["worst"]["auroc"]) < 0.05
