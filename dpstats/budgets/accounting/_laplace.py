"""Laplace-mechanism calibration across supported privacy families."""

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


def calibrate_laplace_scale(budget: DPBudget, sensitivity: float) -> float:
    """Calibrate one scalar Laplace release to an offered budget.

    Approximate DP uses the exact Laplace profile. zCDP uses Harrison &
    Manurangsi 2026, Theorem 13. GDP uses Dong, Roth and Su (2022), Lemma A.6.

    Args:
        budget: Offered budget of any family.
        sensitivity: Positive finite L1 sensitivity of the released scalar.

    Returns:
        The Laplace noise scale; zero when the offer is unconstrained.

    Example:
        >>> calibrate_laplace_scale(PureDPBudget(eps=1.0), 1.0)
        1.0
    """
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive.")
    if budget.is_unconstrained:
        return 0.0
    match budget:
        case PureDPBudget():
            laplace_epsilon = budget.eps
        case ApproxDPBudget():
            laplace_epsilon = budget.eps - 2 * math.log1p(-budget.delta)
        case ZCDPBudget():
            laplace_epsilon = bisect_monotone_boundary(
                lambda theta: theta + math.expm1(-theta) <= budget.rho,
                0.0,
                1.0,
                satisfying_endpoint="lower",
            )
        case GDPBudget():
            laplace_epsilon = -2.0 * (math.log(2.0) + float(log_ndtr(-budget.mu / 2.0)))
        case NoPrivacy():
            return 0.0
        case _:
            raise InvalidRegime(
                f"Cannot spend an offered {type(budget).__name__}: accounting "
                "implements no implication from that budget family."
            )
    return sensitivity / laplace_epsilon
