"""Checks on the variance and rank-agreement statistics, with no data required.

The Appendix B.4 figures decided whether the external validation was executed at
all, so they are recomputed here from first principles and compared against the
values recorded in ``results/external_validation/precision.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from icu_deterioration import hanley_mcneil_var, required_events, union_rho

RESULTS = Path(__file__).resolve().parents[1] / "results" / "external_validation"

# Registered quantities, Section 3.11 and Table 4.41 (four hours).
AUC_INTERNAL = 0.769
TOLERATED_FALL = 0.05
AUC_ALTERNATIVE = AUC_INTERNAL - TOLERATED_FALL
AUC_OBSERVED_EXTERNAL = 0.749  # first execution, Table 4.41
INTERNAL_EVENTS, INTERNAL_INSTANCES = 401, 52279
EICU_RATIO = (17776 - 27) / 27


@pytest.fixture(scope="module")
def recorded() -> dict:
    with open(RESULTS / "precision.json", encoding="utf-8") as handle:
        return json.load(handle)


class TestHanleyMcNeil:
    def test_rejects_an_impossible_auc(self):
        for bad in (0.0, 1.0, -0.2, 1.3):
            with pytest.raises(ValueError):
                hanley_mcneil_var(bad, 50, 50)

    def test_rejects_an_empty_class(self):
        with pytest.raises(ValueError):
            hanley_mcneil_var(0.75, 0, 100)

    def test_variance_falls_as_events_accumulate(self):
        widths = [hanley_mcneil_var(0.75, n, n * 100) for n in (27, 150, 500, 2000)]
        assert widths == sorted(widths, reverse=True)

    def test_matches_the_recorded_internal_standard_error(self, recorded):
        se = np.sqrt(
            hanley_mcneil_var(AUC_INTERNAL, INTERNAL_EVENTS, INTERNAL_INSTANCES - INTERNAL_EVENTS)
        )
        assert round(float(se), 4) == recorded["internal_se"]

    def test_matches_the_recorded_interval_width_at_27_events(self, recorded):
        """Appendix B.4's validation width, against the 0.207 the bootstrap measured.

        This one is evaluated at the AUROC the first execution actually returned,
        0.749, because it is a check of the formula against a measurement made on
        that execution. The planning widths below are a different quantity.
        """
        se = np.sqrt(hanley_mcneil_var(AUC_OBSERVED_EXTERNAL, 27, round(27 * EICU_RATIO)))
        width = 2 * 1.96 * se
        assert round(float(width), 3) == recorded["analytic_width_at_27"]
        # The measured interval sits inside the bracket Appendix B.4 reports; the
        # two influences on it cannot be separated from one 27-event cohort.
        assert 0.180 <= 0.207 <= width

    def test_planning_widths_are_evaluated_at_the_alternative_not_the_observed_value(
        self, recorded
    ):
        """A power calculation needs the variance where the test must have power,
        which is the alternative, 0.719. Evaluating it at the internal value
        understates the requirement. The two differ, and conflating them is the
        easy mistake here, so both are pinned."""
        at_alternative = (
            2 * 1.96 * np.sqrt(hanley_mcneil_var(AUC_ALTERNATIVE, 27, round(27 * EICU_RATIO)))
        )
        assert round(float(at_alternative), 3) == recorded["n27"]["ci_width"]
        assert recorded["n27"]["ci_width"] != recorded["analytic_width_at_27"]

    def test_conservative_against_a_binormal_model(self):
        """verify_hm.py's claim, at one size, cheaply: analytic >= empirical."""
        rng = np.random.default_rng(7)
        n_pos, n_neg, auc = 150, 2000, 0.769
        mu = norm.ppf(auc) * np.sqrt(2)
        from sklearn.metrics import roc_auc_score

        estimates = [
            roc_auc_score(
                np.concatenate([np.ones(n_pos), np.zeros(n_neg)]),
                np.concatenate([rng.normal(mu, 1, n_pos), rng.normal(0, 1, n_neg)]),
            )
            for _ in range(400)
        ]
        empirical_sd = float(np.std(estimates, ddof=1))
        analytic_sd = float(np.sqrt(hanley_mcneil_var(auc, n_pos, n_neg)))
        assert analytic_sd >= empirical_sd
        assert analytic_sd / empirical_sd < 1.35


class TestRequiredEvents:
    def test_reproduces_the_one_sample_requirement(self, recorded):
        assert (
            required_events(AUC_ALTERNATIVE, EICU_RATIO, TOLERATED_FALL, power=0.80)
            == recorded["events_eicu_80pct_one_sample"]
        )

    def test_reproduces_the_two_sample_requirement(self, recorded):
        se_internal = np.sqrt(
            hanley_mcneil_var(AUC_INTERNAL, INTERNAL_EVENTS, INTERNAL_INSTANCES - INTERNAL_EVENTS)
        )
        assert (
            required_events(
                AUC_ALTERNATIVE,
                EICU_RATIO,
                TOLERATED_FALL,
                power=0.80,
                two_sample=True,
                se_reference=float(se_internal),
            )
            == recorded["events_eicu_80pct_two_sample"]
        )

    def test_the_demonstration_release_falls_far_short(self, recorded):
        """27 events against the 266 the registered criterion needs, and against
        the 150 Appendix B.1 set as its precondition."""
        needed = recorded["events_eicu_80pct_one_sample"]
        assert 27 / needed < 0.11
        assert needed > 150

    def test_more_power_costs_more_events(self):
        at80 = required_events(AUC_ALTERNATIVE, EICU_RATIO, TOLERATED_FALL, power=0.80)
        at90 = required_events(AUC_ALTERNATIVE, EICU_RATIO, TOLERATED_FALL, power=0.90)
        assert at90 > at80

    def test_two_sample_needs_a_reference_standard_error(self):
        with pytest.raises(ValueError):
            required_events(0.7, 100, 0.05, two_sample=True)


class TestUnionRho:
    def test_identical_rankings_agree_perfectly(self):
        scores = np.linspace(1.0, 0.0, 40)
        rho, size = union_rho(scores, scores.copy())
        assert rho == pytest.approx(1.0)
        assert size == 15

    def test_the_union_grows_when_the_lists_disagree(self):
        a = np.linspace(1.0, 0.0, 40)
        b = a[::-1].copy()
        rho, size = union_rho(a, b)
        assert size == 30
        assert rho == pytest.approx(-1.0)

    def test_head_disagreement_is_visible_to_the_union_and_not_to_the_whole_list(self):
        """Why Section 4.12.4 uses the union: a correlation over every predictor is
        dominated by the many low-ranked ones whose order both cohorts share."""
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        a = np.sort(rng.random(75))[::-1].copy()
        b = a.copy()
        head = list(range(15))
        b[head] = b[rng.permutation(head)]  # reshuffle the head only
        rho_all = float(spearmanr(a, b).statistic)
        rho_union, _ = union_rho(a, b)
        assert rho_all > 0.95
        assert rho_union < rho_all

    def test_rejects_mismatched_vectors(self):
        with pytest.raises(ValueError):
            union_rho(np.arange(10.0), np.arange(11.0))

    def test_rejects_an_out_of_range_k(self):
        with pytest.raises(ValueError):
            union_rho(np.arange(10.0), np.arange(10.0), k=11)


class TestRecordedRankAgreement:
    """The recorded H3 figures must keep saying what Section 4.12.4 reports."""

    @pytest.fixture(scope="class")
    @staticmethod
    def h3() -> dict:
        with open(RESULTS / "h3_union.json", encoding="utf-8") as handle:
            return json.load(handle)

    def test_four_hours_meets_the_registered_criterion(self, h3):
        assert h3["4h"]["mean_abs"]["rho_union_top15"] > 0.7

    def test_eight_hours_fails_it(self, h3):
        assert h3["8h"]["mean_abs"]["rho_union_top15"] < 0.7

    def test_centred_attributions_do_not_rescue_four_hours_on_the_point_estimate(self, h3):
        """0.7255 clears 0.7, but Section 3.6 decides on the interval's lower
        bound, which is 0.629. The point estimate alone does not settle it."""
        assert 0.7 < h3["4h"]["centred"]["rho_union_top15"] < 0.78

    @pytest.mark.parametrize("horizon", ["4h", "8h"])
    @pytest.mark.parametrize("variant", ["mean_abs", "centred"])
    def test_the_union_is_never_smaller_than_one_top_list(self, h3, horizon, variant):
        assert 15 <= h3[horizon][variant]["union_size"] <= 30

    @pytest.mark.parametrize("horizon", ["4h", "8h"])
    @pytest.mark.parametrize("variant", ["mean_abs", "centred"])
    def test_overlap_and_union_size_reconcile(self, h3, horizon, variant):
        """|A ∪ B| = 15 + 15 − |A ∩ B| for two top-15 lists."""
        entry = h3[horizon][variant]
        assert entry["union_size"] == 30 - entry["top15_overlap"]
