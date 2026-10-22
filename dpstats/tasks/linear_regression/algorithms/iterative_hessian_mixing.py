"""Iterative Hessian Mixing for bounded private linear regression."""

from __future__ import annotations

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime

import math
from functools import lru_cache

import numpy as np

from dpstats.parameters import parameter_field
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
)
from dpstats.task import Array1D, Array2D
from dpstats.utils import bisect_monotone_boundary
from dpstats.utils import split_seed
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _epsilon_for_mixing_branch(
    gamma: float,
    sketch_size: int,
    num_iterations: int,
    delta: float,
) -> float:
    """Account the Appendix C.1.1 eigenvalue release and ``T`` sketches.

    The eigenvalue Gaussian from Algorithm 2 has sensitivity one after the
    paper's ``C_X=1`` transport and noise multiplier ``gamma / sqrt(k)``. Its
    epsilon is evaluated through the exact analytic-Gaussian tradeoff, a strict
    improvement over the paper's displayed sufficient bound. Each Algorithm 3
    sketch uses the Gaussian Mixing RDP curve from Lev et al. 2025, Lemma 1;
    the ``T`` curves compose additively before the paper's sharp RDP-to-DP
    conversion. Both release families receive ``delta``. The caller separately
    reserves the Appendix C.1.1 bad-event delta and accounts for the gradient
    releases.
    """
    if not math.isfinite(gamma) or gamma <= 2.5:
        raise ValueError("gamma must be finite and greater than 5/2.")

    eigen_sigma = gamma / math.sqrt(sketch_size)
    eigen_epsilon = accounting.compute_epsilon_from_gaussian_mechanism(
        scale=eigen_sigma,
        sensitivity=1.0,
        delta=delta,
    )

    sketch_epsilon = accounting.compute_epsilon_for_composed_gaussian_mixing(
        gamma,
        sketch_size,
        num_iterations,
        delta,
    )
    return eigen_epsilon + sketch_epsilon


@lru_cache(maxsize=256)
def _calibrate_mixing_branch_gamma_cached(
    eps: float,
    delta: float,
    sketch_size: int,
    num_iterations: int,
) -> float:
    """Invert the complete Appendix C.1.1 mixing-branch accounting.

    ``budget.delta`` is split equally among the eigenvalue Gaussian, the bad
    event, and the composed Gaussian-sketch RDP conversion. The eigenvalue and
    sketch epsilons compose within ``budget.eps``. Unconstrained offers reach
    the source floor gamma -> 5/2.
    """
    per_term_delta = delta / 3.0
    return bisect_monotone_boundary(
        lambda gamma: gamma > 2.5
        and _epsilon_for_mixing_branch(
            gamma, sketch_size, num_iterations, per_term_delta
        )
        <= eps,
        2.5,
        5.0,
        satisfying_endpoint="upper",
    )


def _run_algorithm_2(
    X: Array2D,
    C_X: float,
    gamma: float,
    tau: float,
    eta: float,
    seed: int,
) -> float:
    """Run Lev et al. Algorithm 2 to privately calibrate mixing noise."""
    rng = np.random.default_rng(seed)
    lambda_min = np.linalg.eigvalsh(X.T @ X)[0]
    lambda_tilde = max(lambda_min - eta * C_X**2 * (tau - rng.normal()), 0.0)
    return math.sqrt(max(gamma * C_X**2 - lambda_tilde, 0.0))


def _run_algorithm_1(
    X: Array2D,
    y: Array1D,
    gamma: float,
    tau: float,
    sigma: float,
    k: int,
    T: int,
    C: float,
    seed: int,
) -> Array1D:
    """Run Lev et al. Algorithm 1 in coordinates where ``||x_i|| <= 1``."""
    d = X.shape[1]
    calibration_seed, iteration_seed = split_seed(seed, 2)
    rng = np.random.default_rng(iteration_seed)
    theta = np.zeros(d)
    gram = X.T @ X

    # Algorithm 1, line 2: one private mixing-noise calibration reused by all
    # iterations.
    eta = _run_algorithm_2(
        X,
        1.0,
        gamma,
        tau,
        gamma / math.sqrt(k),
        calibration_seed,
    )
    sketch_covariance = gram + eta**2 * np.eye(d)

    for _ in range(T):
        # Algorithm 1, lines 4-5. Each row of S X + eta Xi is exactly
        # N(0, X.T X + eta^2 I), so direct covariance sampling preserves the
        # paper's sketch law without materializing the k-by-n matrix S.
        X_tilde = rng.multivariate_normal(
            np.zeros(d),
            sketch_covariance,
            size=k,
            check_valid="raise",
        )
        residual = np.clip(y - X @ theta, -C, C)
        G_tilde = X.T @ residual - eta**2 * theta + sigma * rng.normal(size=d)

        # Algorithm 1, lines 6-7: solve the sketched Newton system and update.
        direction = np.linalg.solve((X_tilde.T @ X_tilde) / k, G_tilde)
        theta = theta + direction

    return theta


class IterativeHessianMixing(BaseAlgorithm):
    r"""Fit linear regression by a few Newton steps that use noisy gradients and a randomly sketched X^T X.

    An implementation of Algorithms 1 and 2 from Lev et al. (2026). Privately
    calibrates an amount of noise to mix into X^T X, then runs a fixed number
    of Newton-like steps, each drawing a fresh random Gaussian sketch of that
    noisy matrix and using a gradient with clipped residuals and Gaussian
    noise. It is an approximate-DP mechanism.

    Requires:
        - An ``ApproxDPBudget`` or an unconstrained budget.
        - A finite ``design_domain.l2_norm_bound``.
        - A finite ``response_domain.l2_norm_bound``.

    Parametrization notes:
        - ``num_iterations``: Corresponds to $T$ in the paper. Set as in the
          paper's main evaluation in Section 3.1.
        - The paper sets the sketch size $k = 6 \max(d, \log(4T/ϱ))$, residual
          clip $C = C_Y$ and $ϱ = δ/10$ for practical performance rather than
          to meet its Theorem 1's conditions. They are fixed, not configurable
          hyperparameters.

    Implementation notes:
        - Keeps Algorithm 1's ``-eta**2 * theta`` correction term, which the
          authors' code omits.
        - The noisy eigenvalue, sketch and gradient keep the paper's form, but
          their noise is calibrated by exact analytic-Gaussian and Rényi
          accounting rather than by Appendix C's closed-form bounds.

    Reference:
        - lev2026near: Algorithms 1 and 2 of arXiv v2.
    """

    TASK = LinearRegression

    num_iterations: int = parameter_field(
        default=3,
        gt=0,
        description="Number ``T`` of Algorithm 1 updates.",
    )

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        """Output private regression coefficients with Algorithm 1."""
        budget = accounting.spend_as_approx_dp(budget)

        x_bound = metadata.design_domain.l2_norm_bound
        y_bound = metadata.response_domain.l2_norm_bound
        if not math.isfinite(x_bound):
            raise InvalidRegime(
                "IterativeHessianMixing requires a finite covariate bound."
            )
        if not math.isfinite(y_bound):
            raise InvalidRegime(
                "IterativeHessianMixing requires a finite response bound."
            )

        T = self.num_iterations
        varrho = budget.delta / 10.0
        k = math.ceil(6.0 * max(metadata.dim, math.log(4.0 * T / varrho)))
        C = y_bound

        sketch_budget = ApproxDPBudget(
            eps=budget.eps / 2.0,
            delta=3.0 * budget.delta / 4.0,
        )
        gradient_budget = ApproxDPBudget(
            eps=budget.eps / 2.0,
            delta=budget.delta - sketch_budget.delta,
        )
        gamma = _calibrate_mixing_branch_gamma_cached(
            sketch_budget.eps,
            sketch_budget.delta,
            sketch_size=k,
            num_iterations=T,
        )
        tau = math.sqrt(2.0 * math.log(max(4.0 / budget.delta, 4.0 / varrho)))
        # Appendix C.1.1's sigma is a sufficient RDP bound. Exact GDP
        # composition calibrates the same T adaptive Gaussian releases tighter.
        sigma = (
            accounting.compute_noise_multiplier_for_composed_gaussian(
                gradient_budget, num_steps=T
            )
            * C
        )

        theta = _run_algorithm_1(
            X=input.X / x_bound,
            y=input.y,
            C=C,
            k=k,
            T=T,
            gamma=gamma,
            tau=tau,
            sigma=sigma,
            seed=seed,
        )
        return LinearRegressionOutput(beta=theta / x_bound)
