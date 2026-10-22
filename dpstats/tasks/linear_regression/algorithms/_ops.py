"""Shared Gaussian one-posterior-sample mechanism primitives."""

from __future__ import annotations

import math

import numpy as np

from dpstats.budgets import ApproxDPBudget
from dpstats.task import Array1D, Array2D


def solve_positive_quadratic(
    quadratic: float,
    linear: float,
    right_side: float,
) -> float:
    """Solve ``quadratic*x^2 + linear*x = right_side`` for positive ``x``."""
    return (
        2.0
        * right_side
        / (linear + math.sqrt(linear * linear + 4.0 * quadratic * right_side))
    )


def solve_ops_scaled_temperature(
    regularized_curvature_lower_bound: float,
    feature_radius: float,
    budget: ApproxDPBudget,
) -> float:
    """Solve Wang's pDP bound for ``sqrt(gamma) * L``.

    This is the positive root of the quadratic privacy bound in Wang (2018),
    Remark 11. ``regularized_curvature_lower_bound`` lower-bounds the smallest
    eigenvalue of the ridge precision matrix.

    Raises:
        ValueError: If the leverage-only term consumes the offered epsilon.
    """
    log_term = math.log(2.0 / budget.delta)
    radius_squared = feature_radius**2
    structural = (
        radius_squared * (1.0 + log_term) / (2.0 * regularized_curvature_lower_bound)
    )
    remaining = budget.eps - structural
    if remaining <= 0.0:
        raise ValueError("The OPS privacy bound leaves no positive temperature.")

    linear = math.sqrt(log_term / regularized_curvature_lower_bound)
    quadratic = 1.0 / (2.0 * (regularized_curvature_lower_bound + radius_squared))
    return solve_positive_quadratic(quadratic, linear, remaining)


def sample_ops_posterior(
    mean: Array1D,
    precision_without_gamma: Array2D,
    inverse_sqrt_gamma: float,
    seed: int,
) -> Array1D:
    """Draw exactly from ``N(mean, gamma^-1 H^-1)`` by eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(precision_without_gamma)
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("OPS posterior precision is not positive definite.")

    rng = np.random.default_rng(seed)
    standard_normal: Array1D = rng.normal(size=mean.size)
    scaled: Array1D = standard_normal * inverse_sqrt_gamma / np.sqrt(eigenvalues)
    return mean + eigenvectors @ scaled
