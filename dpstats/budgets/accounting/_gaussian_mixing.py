"""Gaussian-Mixing privacy profile."""

from __future__ import annotations

import math
from typing import cast

from scipy.optimize import (
    OptimizeResult,
    minimize_scalar,  # pyright: ignore[reportUnknownVariableType]
)

from dpstats.budgets.accounting._rdp import compute_epsilon_bound_from_rdp


def _gaussian_mixing_rdp_epsilon(
    alpha: float,
    sketch_size: int,
    gamma: float,
) -> float:
    """Evaluate one Gaussian Mixing sketch's RDP curve.

    Implements Lev et al., "The Gaussian Mixing Mechanism"
    (arXiv:2505.24603v2, Lemma 1).
    """
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one.")
    if not 1.0 < alpha < gamma:
        raise ValueError("alpha must be in (1, gamma).")
    if sketch_size < 1:
        raise ValueError("sketch_size must be positive.")

    return sketch_size * alpha / (2.0 * (alpha - 1.0)) * math.log1p(
        -1.0 / gamma
    ) - sketch_size / (2.0 * (alpha - 1.0)) * math.log1p(-alpha / gamma)


def compute_epsilon_for_composed_gaussian_mixing(
    gamma: float,
    sketch_size: int,
    num_steps: int,
    delta: float,
) -> float:
    """Convert repeated Gaussian-Mixing sketches to approximate DP.

    Composes Lev et al. 2025, Lemma 1 additively in RDP and minimizes the
    Canonne--Kamath--Steinke sharp conversion over the curve's valid orders
    ``1 < alpha < gamma``.

    Args:
        gamma: Finite mixing parameter greater than one.
        sketch_size: Positive number of rows in each sketch.
        num_steps: Positive number of composed sketches.
        delta: Target approximate-DP delta in (0, 1).

    Returns:
        The nonnegative approximate-DP epsilon of the composition at ``delta``.

    Example:
        >>> round(float(compute_epsilon_for_composed_gaussian_mixing(10.0, 5, 3, 1e-5)), 3)
        2.006
    """
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one.")
    if sketch_size < 1:
        raise ValueError("sketch_size must be positive.")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")

    def approximate_dp_epsilon_at_order(alpha: float) -> float:
        rdp_epsilon = _gaussian_mixing_rdp_epsilon(alpha, sketch_size, gamma)
        return compute_epsilon_bound_from_rdp(num_steps * rdp_epsilon, alpha, delta)

    result = cast(
        OptimizeResult,
        minimize_scalar(
            approximate_dp_epsilon_at_order,
            bounds=(1.0 + 1e-10, gamma * (1.0 - 1e-10)),
            method="bounded",
        ),
    )
    if not result.success or not math.isfinite(result.x):
        raise RuntimeError("Gaussian-Mixing Renyi-order calibration did not converge.")

    # Every order in (1, gamma) yields a sound bound. Numerical minimization can
    # only choose a looser sound order; it cannot make the guarantee unsound.
    return max(approximate_dp_epsilon_at_order(result.x), 0.0)
