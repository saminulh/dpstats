"""The ``SSP`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime


import numpy as np

from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


class SSP(BaseAlgorithm):
    """Sufficient Statistics Perturbation. Fit linear regression by solving noisy normal equations.

    An implementation of sufficient-statistics perturbation calibrated by
    Theorem 8 of Balle and Wang (2018). Adds Gaussian noise to X^T X and X^T y
    and solves the noisy normal equations for the coefficients. It is a
    Gaussian-DP mechanism.

    Requires:
        - Finite ``design_domain.l2_norm_bound`` and
          ``response_domain.l2_norm_bound``.
        - A budget family admitting a GDP implication.

    Implementation notes:
        - Each noisy statistic uses exact analytic-Gaussian calibration rather
          than the closed-form constant of Wang (2018).

    Reference:
        - balle2018improving: Theorem 8.
        - wang2018revisiting: Appendix B.1.
    """

    TASK = LinearRegression

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        gdp_budget = accounting.spend_as_gdp(budget)
        d = metadata.dim
        x_bound = metadata.design_domain.l2_norm_bound
        y_bound = metadata.response_domain.l2_norm_bound
        if not np.isfinite(x_bound) or not np.isfinite(y_bound):
            raise InvalidRegime("SSP requires finite covariate and response bounds.")
        # Exact GDP composition of the two Gaussian releases.
        # Sensitivities: ||x_i x_i^T||_F <= x_bound^2 and
        # ||x_i y_i||_2 <= x_bound * y_bound.
        covariance_budget, cross_product_budget = gdp_budget.split((1.0, 1.0))
        cov_noise = accounting.calibrate_gaussian_scale(covariance_budget, x_bound**2)
        vec_noise = accounting.calibrate_gaussian_scale(
            cross_product_budget, x_bound * y_bound
        )
        X, y = (input.X, input.y)
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(d, d)) * cov_noise
        A = np.triu(A) + np.triu(A, 1).T
        z = vec_noise * rng.normal(size=d)
        beta = np.linalg.solve(X.T @ X + A, X.T @ y + z)
        return LinearRegressionOutput(beta=beta)
