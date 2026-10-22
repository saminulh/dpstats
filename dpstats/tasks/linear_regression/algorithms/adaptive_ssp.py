"""Adaptive sufficient-statistics perturbation for linear regression."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np
from scipy.stats import norm

from dpstats.parameters import parameter_field
from dpstats.budgets import (
    DPBudget,
    GDPBudget,
)
from dpstats.task import Array1D, Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _run_algorithm_2(
    X: Array2D,
    y: Array1D,
    x_bound: float,
    y_bound: float,
    failure_probability: float,
    budget: GDPBudget,
    seed: int,
) -> Array1D:
    """Run Wang's Algorithm 2 with the Appendix B stability correction."""
    d = X.shape[1]
    # Wang's Theorem 3(ii) proof unions four events (eigenvalue lower bound,
    # Gram-noise stability, Lemma 6 bound for E_1 theta*, E_2 bound); each gets a
    # quarter of the failure probability.
    proof_event_failure_probability = failure_probability / 4.0
    release_budget = budget.split((1.0, 1.0, 1.0))[0]

    # Add/remove sensitivities: x_bound**2 for the smallest eigenvalue (Weyl) and
    # for X^T X (||x x^T||_F = ||x||^2), x_bound * y_bound for X^T y.
    sigma_lambda = accounting.calibrate_gaussian_scale(release_budget, x_bound**2)
    sigma_xx = accounting.calibrate_gaussian_scale(release_budget, x_bound**2)
    sigma_xy = accounting.calibrate_gaussian_scale(release_budget, x_bound * y_bound)
    rng = np.random.default_rng(seed)

    # Algorithm 2, Steps 2-3: privately lower-bound lambda_min(X.T @ X),
    # then choose the adaptive ridge parameter lambda.
    gram: Array2D = X.T @ X
    lambda_min = np.linalg.eigvalsh(gram)[0]
    lambda_min_tilde = max(
        0.0,
        lambda_min
        + sigma_lambda * rng.normal()
        - sigma_lambda * norm.isf(proof_event_failure_probability),
    )
    # Bounds on ||E||: Tropp (2012), Theorem 4.1; and Bandeira and van Handel
    # (2016), Lemma 2.2 (E[lambda_max(E)] <= 2 sigma_xx sqrt(d)) with Gaussian
    # concentration, variance factor 2 - 1/d, over both spectral edges.
    tropp_spectral_bound = sigma_xx * math.sqrt(
        2.0 * d * math.log(2.0 * d / proof_event_failure_probability)
    )
    slepian_spectral_bound = sigma_xx * (
        2.0 * math.sqrt(d)
        + math.sqrt(
            2.0 * (2.0 - 1.0 / d) * math.log(2.0 / proof_event_failure_probability)
        )
    )
    spectral_noise_bound = min(tropp_spectral_bound, slepian_spectral_bound)
    # Wang (2018), Eq. (15) needs
    # ||E|| <= (lambda_min + lambda) / 2, so top up the private lower bound
    # to twice the resolved operator-norm bound.
    lambda_ = max(0.0, 2.0 * spectral_noise_bound - lambda_min_tilde)

    # Algorithm 2, Steps 4-5: privately release X.T @ X and X.T @ y.
    upper_noise = sigma_xx * rng.normal(size=(d, d))
    symmetric_noise = np.triu(upper_noise) + np.triu(upper_noise, 1).T
    gram_tilde = gram + symmetric_noise
    cross_product_tilde = X.T @ y + sigma_xy * rng.normal(size=d)

    # Algorithm 2, Step 6: solve the adaptively regularized normal equations.
    beta_tilde: Array1D = np.linalg.solve(
        gram_tilde + lambda_ * np.eye(d), cross_product_tilde
    )
    return beta_tilde


class AdaSSP(BaseAlgorithm):
    """Adaptive Sufficient Statistics Perturbation. Fit linear regression by solving noisy normal equations with an adaptive ridge penalty.

    An implementation of Algorithm 2 from Wang (2018). Adds Gaussian noise to
    X^T X and X^T y, privately estimates the smallest eigenvalue of X^T X to
    pick a ridge penalty that keeps the noisy system stable, and solves the
    noisy normal equations. It is a Gaussian-DP mechanism.

    Requires:
        - Finite ``design_domain.l2_norm_bound`` and
          ``response_domain.l2_norm_bound``.
        - A budget family admitting a GDP implication.

    Parametrization notes:
        - ``failure_probability``: Corresponds to $ρ$ in Wang (2018). Taken from
          the authors' reference implementation, since Algorithm 2 prints no
          value; Algorithm 1 of the same paper sets the same value.

    Implementation notes:
        - Each noisy statistic uses the exact analytic-Gaussian calibration of
          Balle and Wang (2018) rather than the closed-form noise scale of Wang
          (2018), and the private lower bound on the eigenvalue uses an exact
          one-sided Gaussian quantile.
        - Algorithm 2's printed ridge coefficient does not satisfy the
          stability condition its Appendix B proof needs. The ridge here is
          ``max(0, 2*B - private_eigenvalue_lower_bound)``, where ``B`` is the
          tighter of the matrix-Gaussian tail bound of Tropp (2012) and the
          Gaussian concentration bound of Bandeira and van Handel (2016), for
          the exact noise. Only that constant changes; the three releases
          and the adaptive ridge are those of Wang (2018).
        - ``failure_probability`` is split equally over the four
          high-probability events in the proof of Theorem 3(ii): the private
          eigenvalue bound, the stability of the noisy Gram matrix, and the two
          events ``E_1`` and ``E_2``. The last two affect only the theorem's
          error bound, not what the algorithm does.

    Reference:
        - wang2018revisiting: Algorithm 2.
        - balle2018improving: Theorem 8.
        - tropp2012user: Theorem 4.1.
        - bandeira2016sharp: Lemma 2.2.
    """

    TASK = LinearRegression

    failure_probability: float = parameter_field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Total failure probability for the four utility events in the proof of Theorem 3(ii).",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        gdp_budget = accounting.spend_as_gdp(budget)
        x_bound = metadata.design_domain.l2_norm_bound
        y_bound = metadata.response_domain.l2_norm_bound
        if not math.isfinite(x_bound) or not math.isfinite(y_bound):
            raise InvalidRegime("AdaSSP requires finite covariate and response bounds.")
        beta = _run_algorithm_2(
            X=input.X,
            y=input.y,
            x_bound=x_bound,
            y_bound=y_bound,
            failure_probability=self.failure_probability,
            budget=gdp_budget,
            seed=seed,
        )
        return LinearRegressionOutput(beta=beta)
