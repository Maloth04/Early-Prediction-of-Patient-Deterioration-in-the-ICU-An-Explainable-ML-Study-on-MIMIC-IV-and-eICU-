"""Discrimination variance and rank-agreement statistics.

Both functions are lifted unchanged from the analysis scripts that produced the
reported figures: ``hanley_mcneil_var`` from ``analysis/external_validation/precision.py``
(Appendix B.4) and ``union_rho`` from ``analysis/external_validation/h3_union.py``
(the registered H3 criterion, Section 4.12.4). They are here so the test suite can
check them with no data; the scripts remain the record of the run.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr


def hanley_mcneil_var(auc: float, n_pos: int, n_neg: int) -> float:
    """Variance of an AUROC estimate, Hanley and McNeil (1982).

    Conservative against a binormal score model by 11 to 16 per cent, which is
    why Appendix B.4 reports a bracket rather than a point;
    ``analysis/external_validation/verify_hm.py`` is the Monte Carlo check.

    Args:
        auc: the AUROC the variance is evaluated at. For a power calculation this
            is the ALTERNATIVE value being tested, not the internal one.
        n_pos: number of events.
        n_neg: number of non-events.

    Returns:
        The variance of the AUROC estimate.
    """
    if not 0.0 < auc < 1.0:
        raise ValueError(f"auc must lie in (0, 1), got {auc}")
    if n_pos < 1 or n_neg < 1:
        raise ValueError(f"need at least one of each class, got {n_pos}/{n_neg}")
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )


def required_events(
    auc_alt: float,
    ratio: float,
    delta: float,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sample: bool = False,
    se_reference: float | None = None,
) -> int:
    """Events needed to detect a fall of ``delta`` at the given power.

    The variance is evaluated at ``auc_alt``, the value under the alternative,
    because that is where the test must have power. Evaluating it at the internal
    value understates the requirement.

    Args:
        auc_alt: AUROC under the alternative, i.e. internal minus ``delta``.
        ratio: non-events per event in the external cohort.
        delta: the fall to be detected.
        power: target power.
        alpha: two-sided significance level.
        two_sample: if True, the internal estimate carries its own error too and
            ``se_reference`` is added in quadrature.
        se_reference: standard error of the internal estimate, required when
            ``two_sample`` is True.

    Returns:
        The smallest event count meeting the requirement.
    """
    if two_sample and se_reference is None:
        raise ValueError("two_sample=True requires se_reference")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    target = (delta / z) ** 2
    for n_pos in range(5, 500_000):
        var = hanley_mcneil_var(auc_alt, n_pos, round(n_pos * ratio))
        if two_sample:
            var += se_reference**2
        if var <= target:
            return n_pos
    raise RuntimeError("no event count under 500,000 reaches the target")


def union_rho(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 15) -> tuple[float, int]:
    """The registered H3 statistic: Spearman rho over the union of two top-k lists.

    Section 3.6 fixes the criterion at rho > 0.7 and decides on the LOWER BOUND of
    the interval, not the point estimate. A correlation taken over all predictors
    instead is dominated by the many low-ranked ones whose order both cohorts
    share, so it barely registers disagreement at the head of the list.

    Args:
        scores_a: per-feature importance in the first cohort or horizon.
        scores_b: the same features, same order, in the second.
        k: list length, 15 as registered.

    Returns:
        ``(rho, union_size)``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"score vectors differ in shape: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("score vectors must be one-dimensional")
    if k < 2 or k > a.size:
        raise ValueError(f"k must lie in [2, {a.size}], got {k}")
    top_a = list(np.argsort(-a)[:k])
    top_b = list(np.argsort(-b)[:k])
    union = sorted(set(top_a) | set(top_b))
    rho = float(spearmanr(a[union], b[union]).statistic)
    return rho, len(union)
