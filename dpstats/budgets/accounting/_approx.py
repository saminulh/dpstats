"""Approximate-DP budget-family implications."""

from __future__ import annotations

import math

from dpstats.budgets import ApproxDPBudget, DPBudget
from dpstats.exceptions import InvalidRegime


def spend_as_approx_dp(budget: DPBudget) -> ApproxDPBudget:
    """Return the approximate-DP spend admitted by a budget.

    Every unconstrained offer, of any family, returns ``(eps=inf, delta=1)``.
    An offered ``ApproxDPBudget`` is returned unchanged. Every other family
    implies a whole (epsilon, delta) curve with no single largest point, so
    choosing one delta belongs to the calling mechanism, not accounting.

    Args:
        budget: Offered budget of any family.

    Returns:
        The approximate-DP budget the offer admits.

    Raises:
        InvalidRegime: If the offer is a finite budget of another family.

    Example:
        >>> spend_as_approx_dp(ApproxDPBudget(eps=1.0, delta=1e-5))
        ApproxDPBudget(eps=1.0, delta=1e-05)
    """
    if budget.is_unconstrained:
        return ApproxDPBudget(eps=math.inf, delta=1.0)
    if isinstance(budget, ApproxDPBudget):
        return budget
    raise InvalidRegime(f"Cannot spend a {type(budget).__name__} as approximate DP.")
