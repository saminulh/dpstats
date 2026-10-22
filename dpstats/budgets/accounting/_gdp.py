"""Gaussian-DP profile and budget-family implications."""

from __future__ import annotations

import math
from scipy.special import log_ndtr, ndtr

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.utils import bisect_monotone_boundary


def _delta_from_gdp(budget: GDPBudget, eps: float) -> float:
    """Return the exact approximate-DP delta implied at epsilon.

    Implements Dong, Roth and Su (2022), Corollary 2.13. The second term is
    formed as ``exp(eps + log Phi(b))``, whose exponent is never positive,
    because ``exp(eps)`` alone overflows a float at the epsilons the
    bisection reads.
    """
    if not math.isfinite(budget.mu):
        raise ValueError("GDP profile evaluation requires finite mu.")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")

    first = float(ndtr(-eps / budget.mu + budget.mu / 2))
    second = math.exp(eps + float(log_ndtr(-eps / budget.mu - budget.mu / 2)))
    return first - second


def _epsilon_from_gdp(  # pyright: ignore[reportUnusedFunction]
    budget: GDPBudget, delta: float
) -> float:
    """Return the exact approximate-DP epsilon implied at target delta."""
    if not math.isfinite(budget.mu):
        raise ValueError("GDP profile evaluation requires finite mu.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if _delta_from_gdp(budget, 0.0) <= delta:
        return 0.0
    return bisect_monotone_boundary(
        lambda eps: _delta_from_gdp(budget, eps) <= delta,
        0.0,
        1.0,
        satisfying_endpoint="upper",
    )


def spend_as_gdp(budget: DPBudget) -> GDPBudget:
    """Return the largest implemented GDP spend admitted by a budget.

    Approximate DP inverts the exact Gaussian-DP profile of Dong, Roth and Su
    (2022). An offered zCDP budget admits ``mu = sqrt(2*rho)`` because the
    realized GDP mechanism implies that zCDP guarantee. Finite pure DP is
    unsupported.

    Args:
        budget: Offered budget of any family.

    Returns:
        The GDP budget the offer admits; infinite ``mu`` when unconstrained.

    Raises:
        InvalidRegime: If the offer is a finite pure-DP budget.

    Example:
        >>> round(spend_as_gdp(ApproxDPBudget(eps=1.0, delta=1e-5)).mu, 6)
        0.268051
    """
    if budget.is_unconstrained:
        return GDPBudget(mu=math.inf)
    match budget:
        case GDPBudget():
            return budget
        case ApproxDPBudget():
            mu = bisect_monotone_boundary(
                lambda candidate: candidate == 0.0
                or _delta_from_gdp(GDPBudget(mu=candidate), budget.eps) <= budget.delta,
                0.0,
                1.0,
                satisfying_endpoint="lower",
            )
            return GDPBudget(mu=mu)
        case ZCDPBudget():
            return GDPBudget(mu=math.sqrt(2.0 * budget.rho))
        case PureDPBudget():
            raise InvalidRegime(
                "Cannot realize an offered PureDPBudget through GDP because no "
                "finite GDP guarantee implies pure DP."
            )
        case NoPrivacy():
            return GDPBudget(mu=math.inf)
        case _:
            raise InvalidRegime(
                f"Cannot spend an offered {type(budget).__name__}: accounting "
                "implements no implication from that budget family."
            )
