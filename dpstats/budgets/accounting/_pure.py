"""Pure-DP budget-family implications."""

from __future__ import annotations

import math
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
from dpstats.utils import bisect_monotone_boundary


def spend_as_pure(budget: DPBudget) -> PureDPBudget:
    """Return the largest implemented pure-DP spend admitted by a budget.

    Approximate DP leaves delta unused. zCDP inverts the exact generic pure-DP
    implication derived from Bun & Steinke 2016, Lemma 8.6. GDP uses the exact
    pure-DP/Gaussian tradeoff fixed point from Dong, Roth and Su (2022),
    Proposition 2.5.

    Args:
        budget: Offered budget of any family.

    Returns:
        The pure-DP budget the offer admits; infinite ``eps`` when unconstrained.

    Example:
        >>> round(spend_as_pure(ZCDPBudget(rho=0.5)).eps, 6)
        1.043627
    """
    if budget.is_unconstrained:
        return PureDPBudget(eps=math.inf)
    match budget:
        case PureDPBudget():
            return budget
        case ApproxDPBudget():
            return PureDPBudget(eps=budget.eps)
        case ZCDPBudget():
            return PureDPBudget(
                eps=bisect_monotone_boundary(
                    lambda epsilon: epsilon * math.tanh(epsilon / 2.0) <= budget.rho,
                    0.0,
                    1.0,
                    satisfying_endpoint="lower",
                )
            )
        case GDPBudget():
            return PureDPBudget(
                eps=float(log_ndtr(budget.mu / 2) - log_ndtr(-budget.mu / 2))
            )
        case NoPrivacy():
            return PureDPBudget(eps=math.inf)
        case _:
            raise InvalidRegime(
                f"Cannot spend an offered {type(budget).__name__}: accounting "
                "implements no implication from that budget family."
            )
