"""Harini and Tandon's refined sufficient-statistics simple-regression mechanism."""

from __future__ import annotations


import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import accounting
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms.utils import (
    line_from_unit_square,
    require_scalar_interval,
    to_unit_interval,
)
from dpstats.tasks import (
    SimpleLinearRegression,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)


def _run_algorithm_3(
    x: Array1D, y: Array1D, budget: PureDPBudget, seed: int
) -> tuple[float, float]:
    """Run Algorithm 3 on records already normalized to the unit square.

    Every entry of ``x`` and ``y`` must lie in [0, 1]; that is the precondition
    of the unit L1 sensitivities the noise scale is calibrated to. Returns the
    normalized slope and intercept.
    """
    # Line 1: epsilon is split evenly between the two vector releases.
    release_budget = PureDPBudget(eps=budget.eps / 2)
    # Lemmas 4.2 and 4.6: on the unit square every coordinate of either query
    # is nonnegative and the three coordinates sum to one per record, so
    # add/remove-one moves the sum query by L1 norm exactly one.
    noise_scale = 1.0 / release_budget.eps
    rng = np.random.default_rng(seed)

    f_query: Array1D = np.array(
        [np.sum(x**2), np.sum(x - x**2), np.sum(1.0 - x)],
    )
    g_query: Array1D = np.array(
        [np.sum(x * y), np.sum((1.0 - x) * y), np.sum(1.0 - y)],
    )
    f: Array1D = f_query + rng.laplace(scale=noise_scale, size=3)
    g: Array1D = g_query + rng.laplace(scale=noise_scale, size=3)

    # Each query sums to the record count, so the transcript carries two
    # independent count estimates; averaging them halves the count variance.
    count_from_f = np.sum(f)
    count_from_g = np.sum(g)
    n_tilde = 0.5 * (count_from_f + count_from_g)
    if n_tilde <= 0:
        return 0.0, 0.5

    # Each statistic also has a redundant form using the other query's count,
    # independent of the coordinates it multiplies. Theorems 4.13 and 4.15:
    # variances 8/eps^2 and 40/eps^2 give inverse-variance weights 5/6 and 1/6.
    sum_x2 = (5.0 / 6.0) * f[0] + (1.0 / 6.0) * (count_from_g - f[1] - f[2])
    sum_xy = (5.0 / 6.0) * g[0] + (1.0 / 6.0) * (count_from_f - g[1] - g[2])
    # Theorems 4.17 and 4.19: variances 16/eps^2 and 32/eps^2 give 2/3 and 1/3.
    sum_x = (2.0 / 3.0) * (f[0] + f[1]) + (1.0 / 3.0) * (count_from_g - f[2])
    sum_y = (2.0 / 3.0) * (g[0] + g[1]) + (1.0 / 3.0) * (count_from_f - g[2])

    determinant = n_tilde * sum_x2 - sum_x**2
    if determinant <= 0:
        return 0.0, 0.5

    slope = (n_tilde * sum_xy - sum_x * sum_y) / determinant
    intercept = (sum_x2 * sum_y - sum_x * sum_xy) / determinant
    return slope, intercept


class DPRSS(BaseAlgorithm):
    """Fit simple linear regression from two noisy sets of sums over the records.

    An implementation of Algorithm 3 from Harini S and Tandon (2026). Rescales
    x and y into [0, 1] using their known bounds, adds Laplace noise to two
    sets of sums over the records, and solves for the slope and intercept from
    them, estimating the number of records from the same noisy sums. It is a
    pure-DP mechanism.

    Requires:
        - Finite, positive-width public scalar domains on x and y.

    Implementation notes:
        - On the unit square, the two released sums are of ``f = (x^2, x - x^2,
          1 - x)`` and ``g = (x*y, (1 - x)*y, 1 - y)`` over the records.
        - Both queries add to the record count, so the six noisy entries hold two
          independent count estimates and a second, redundant estimate of each
          statistic. The paper's inverse-variance weights combine the two forms
          before the least-squares equations are solved; all of this is
          post-processing.
        - When the noisy release admits no fit, because the averaged count or
          the noisy Gram determinant is not positive, the paper's constant line
          is output (slope zero, intercept one half), which maps back to the
          middle of the response domain.
        - The record count is always the private noisy estimate; the declared
          sample size is never used.

    Reference:
        - harinis2026refined: Algorithm 3 of arXiv v1.
    """

    TASK = SimpleLinearRegression

    @override
    def run(
        self,
        metadata: SimpleLinearRegressionMetadata,
        input: SimpleLinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleLinearRegressionOutput:
        x_interval = require_scalar_interval(metadata.x_domain, "x")
        y_interval = require_scalar_interval(metadata.y_domain, "y")

        pure_budget = accounting.spend_as_pure(budget)

        # Appendix A: the public rectangle is mapped onto the unit square.
        x: Array1D = to_unit_interval(input.x, x_interval)
        y: Array1D = to_unit_interval(input.y, y_interval)

        normalized_slope, normalized_intercept = _run_algorithm_3(
            x, y, pure_budget, seed
        )

        # Appendix A: map the line back to the public rectangle.
        slope, intercept = line_from_unit_square(
            normalized_slope, normalized_intercept, x_interval, y_interval
        )
        return SimpleLinearRegressionOutput(slope=slope, intercept=intercept)
