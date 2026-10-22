"""Forward and inverse zCDP-to-approximate-DP conversions."""

from __future__ import annotations

import math
import dp_accounting
from dp_accounting import dp_event, mechanism_calibration, rdp
from scipy.special import log_ndtr

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.budgets.accounting._rdp import compute_epsilon_bound_from_rdp
from dpstats.utils import bisect_monotone_boundary


def _zcdp_from_bun_steinke(budget: ApproxDPBudget) -> ZCDPBudget:
    log_one_on_delta = -math.log(budget.delta)
    denominator = math.sqrt(log_one_on_delta + budget.eps) + math.sqrt(log_one_on_delta)
    return ZCDPBudget(rho=(budget.eps / denominator) ** 2)


def _bun_steinke_lemma_36_log_delta(rho: float, epsilon: float) -> float:
    """Evaluate Bun--Steinke Lemma 3.6's proof bound in log space."""
    if epsilon < rho:
        return math.inf

    standardized_threshold = (epsilon + rho) / math.sqrt(2 * rho)
    return (
        epsilon
        + math.log(2)
        + 0.5 * (math.log(math.pi) + math.log(rho))
        + float(log_ndtr(-standardized_threshold))
    )


def _zcdp_from_bun_steinke_lemma_36(
    budget: ApproxDPBudget,
) -> ZCDPBudget:
    target_log_delta = math.log(budget.delta)
    lower = 0.0
    upper = budget.eps

    if _bun_steinke_lemma_36_log_delta(upper, budget.eps) <= target_log_delta:
        return ZCDPBudget(rho=upper)

    return ZCDPBudget(
        rho=bisect_monotone_boundary(
            lambda rho: (
                rho == 0.0
                or _bun_steinke_lemma_36_log_delta(rho, budget.eps) <= target_log_delta
            ),
            lower,
            upper,
            satisfying_endpoint="lower",
        ),
    )


def _continuous_rdp_epsilon(budget: ZCDPBudget, delta: float) -> float:
    """Optimize the stable RDP conversion over all Renyi orders.

    A ``rho``-zCDP mechanism is ``(alpha, alpha*rho)``-RDP at every
    ``alpha > 1``. Substitution into Canonne, Kamath and Steinke (2022),
    Proposition 2.11 gives a convex objective whose derivative vanishes when
    ``rho * (alpha - 1)^2 + log(alpha * delta) = 0``.
    """
    rho = budget.rho
    if rho <= -math.log1p(-(delta * delta)):
        return 0.0

    def derivative_numerator(order: float) -> float:
        scaled_order = math.sqrt(rho) * (order - 1)
        return scaled_order * scaled_order + math.log(order) + math.log(delta)

    order = bisect_monotone_boundary(
        lambda candidate: derivative_numerator(candidate) <= 0,
        1.0,
        2.0,
        satisfying_endpoint="lower",
    )
    epsilon = compute_epsilon_bound_from_rdp(order * rho, order, delta)
    return max(0.0, epsilon)


def spend_as_zcdp_via_rdp(budget: ApproxDPBudget) -> ZCDPBudget:
    """Invert only the continuous CKS conversion of the linear zCDP RDP curve.

    Args:
        budget: Finite approximate-DP budget.

    Returns:
        A zCDP spend whose linear RDP curve implies the budget via CKS.

    Example:
        >>> budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        >>> round(spend_as_zcdp_via_rdp(budget).rho, 6)
        0.030557
    """
    if budget.is_unconstrained:
        return ZCDPBudget(rho=math.inf)
    return ZCDPBudget(
        rho=bisect_monotone_boundary(
            lambda rho: rho == 0.0
            or _continuous_rdp_epsilon(ZCDPBudget(rho=rho), budget.delta) <= budget.eps,
            0.0,
            max(
                budget.eps,
                -math.log1p(-(budget.delta * budget.delta)),
            ),
            satisfying_endpoint="lower",
        )
    )


def spend_as_zcdp(budget: DPBudget) -> ZCDPBudget:
    """Return the largest implemented zCDP spend admitted by a budget.

    Approximate DP takes the best of the closed-form Bun--Steinke inversions,
    the continuous-order Canonne--Kamath--Steinke inversion, and a direct
    finite-order ``dp-accounting`` calibration. Finite pure DP and GDP are
    refused because the implication direction needed by a generic zCDP
    mechanism is unavailable.

    Args:
        budget: Offered budget of any family.

    Returns:
        The zCDP budget the offer admits; infinite ``rho`` when unconstrained.

    Raises:
        InvalidRegime: If the offer is a finite pure-DP or GDP budget.

    Example:
        >>> round(spend_as_zcdp(ApproxDPBudget(eps=1.0, delta=1e-5)).rho, 6)
        0.030557
    """
    if budget.is_unconstrained:
        return ZCDPBudget(rho=math.inf)
    match budget:
        case ZCDPBudget():
            return budget
        case ApproxDPBudget():

            def make_fresh_accountant() -> rdp.RdpAccountant:
                return rdp.RdpAccountant()

            def make_event_from_param(rho: float) -> dp_event.DpEvent:
                return dp_event.ZCDpEvent(max(rho, 0.0))

            candidates = [
                _zcdp_from_bun_steinke(budget),
                _zcdp_from_bun_steinke_lemma_36(budget),
                spend_as_zcdp_via_rdp(budget),
            ]
            try:
                calibrated_rho = float(
                    dp_accounting.calibrate_dp_mechanism(
                        make_fresh_accountant,
                        make_event_from_param,
                        budget.eps,
                        budget.delta,
                    )
                )
                if calibrated_rho > 0.0:
                    candidates.append(ZCDPBudget(rho=calibrated_rho))
            except mechanism_calibration.NoBracketIntervalFoundError:
                pass
            return max(candidates, key=lambda candidate: candidate.rho)
        case PureDPBudget():
            raise InvalidRegime(
                "Cannot spend a PureDPBudget as zCDP because no zCDP guarantee "
                "implies pure DP."
            )
        case GDPBudget():
            raise InvalidRegime(
                "Cannot realize an offered GDPBudget through a generic zCDP "
                "guarantee; GDP implies zCDP, but the reverse implication needed "
                "here does not hold generically."
            )
        case NoPrivacy():
            return ZCDPBudget(rho=math.inf)
        case _:
            raise InvalidRegime(
                f"Cannot spend an offered {type(budget).__name__}: accounting "
                "implements no implication from that budget family."
            )
