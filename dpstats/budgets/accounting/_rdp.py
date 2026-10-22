"""Continuous Renyi-DP to approximate-DP conversion."""

import math


def compute_epsilon_bound_from_rdp(
    rdp_epsilon: float, order: float, delta: float
) -> float:
    """Convert one RDP guarantee to an unclipped approximate-DP epsilon bound.

    Canonne, Kamath and Steinke (2022), Proposition 2.11. Minimize this bound
    over the RDP curve's valid orders, then take its maximum with zero.

    Args:
        rdp_epsilon: Nonnegative RDP epsilon at the supplied order.
        order: Renyi order greater than one.
        delta: Target approximate-DP delta in (0, 1).

    Returns:
        The converted epsilon bound, which may be negative before clipping.

    Example:
        >>> round(compute_epsilon_bound_from_rdp(0.5, 2.0, 0.01), 6)
        3.718876
        >>> round(compute_epsilon_bound_from_rdp(0.1, 2.0, 0.5), 6)
        -0.593147
    """
    return rdp_epsilon + (
        math.log1p(-1 / order) - (math.log(delta) + math.log(order)) / (order - 1)
    )
