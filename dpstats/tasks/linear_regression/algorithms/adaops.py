"""Adaptive one-posterior-sample linear regression."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import ApproxDPBudget, DPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np
from scipy.stats import norm

from dpstats.parameters import parameter_field
from dpstats.task import Array1D, Array2D
from dpstats.utils import bisect_monotone_boundary
from dpstats.utils import split_seed
from dpstats.tasks.linear_regression.algorithms._ops import (
    sample_ops_posterior,
    solve_ops_scaled_temperature,
    solve_positive_quadratic,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _minimize_equation_7(
    lambda_min_tilde: float,
    feature_radius: float,
    c1: float,
    c2: float,
    minimum_regularized_curvature: float,
) -> float:
    """Minimize Wang's Equation (7) after its proof quantities are resolved."""
    radius_squared = feature_radius**2

    h_lower = max(minimum_regularized_curvature, lambda_min_tilde)

    log_a = math.log(c1) + 2.0 * math.log(radius_squared)
    power = 2.0 * c2

    def log_negative_derivative(h: float) -> float:
        return (
            log_a
            + power * math.log1p(radius_squared / h)
            - 2.0 * math.log(h)
            + math.log1p(power * radius_squared / (h + radius_squared))
        )

    if log_negative_derivative(h_lower) <= 0.0:
        optimum_h = h_lower
    else:
        optimum_h = bisect_monotone_boundary(
            lambda h: log_negative_derivative(h) <= 0.0,
            h_lower,
            h_lower * 2.0,
            satisfying_endpoint="upper",
        )

    return max(optimum_h - lambda_min_tilde, 0.0)


def _run_algorithm_1(
    X: Array2D,
    y: Array1D,
    budget: ApproxDPBudget,
    feature_radius: float,
    response_radius: float,
    failure_probability: float,
    seed: int,
) -> Array1D:
    """Run Wang's Algorithm 1 from its constructed public inputs."""
    d = X.shape[1]
    release_seed, posterior_seed = split_seed(seed, 2)
    rng = np.random.default_rng(release_seed)

    # Step 1: compute the sufficient statistics and minimum Gram eigenvalue.
    gram: Array2D = X.T @ X
    cross_product: Array1D = X.T @ y
    lambda_min = np.linalg.eigvalsh(gram)[0]

    # Charge the two Gaussian mechanisms, their two safety events, and the
    # final OPS pDP event separately.
    delta_piece = budget.delta / 5.0
    release_budget = ApproxDPBudget(eps=budget.eps / 4.0, delta=delta_piece)
    posterior_budget = ApproxDPBudget(eps=budget.eps / 2.0, delta=delta_piece)

    # The advertised failure probability covers three paper-level categories:
    # eigenvalue accuracy, log-scale accuracy, and posterior concentration.
    # Each of the first two includes its privacy-enabling safety tail.
    utility_failure_piece = failure_probability / 3.0
    safety_failure = min(delta_piece, utility_failure_piece / 2.0)
    accuracy_failure = utility_failure_piece - safety_failure
    safety_quantile = norm.isf(safety_failure)
    accuracy_quantile = norm.isf(accuracy_failure)

    # Step 2: privately release a nonnegative lower bound on the eigenvalue.
    lambda_min_sigma = accounting.calibrate_gaussian_scale(
        release_budget, feature_radius**2
    )
    lambda_min_tilde = max(
        lambda_min
        + lambda_min_sigma * rng.normal()
        - lambda_min_sigma * safety_quantile,
        0.0,
    )

    gaussian_multiplier = accounting.calibrate_gaussian_scale(release_budget, 1.0)
    log_ops = math.log(2.0 / posterior_budget.delta)
    remaining_ops_epsilon = posterior_budget.eps / 2.0

    # Steps 3-4: minimize Equation (7). eps_bar, the positive root of
    # eps_bar^2 / (2 log_ops) + eps_bar = remaining_ops_epsilon, bounds the two
    # temperature-dependent terms of Remark 11 and gives the C1 coefficient.
    eps_bar = solve_positive_quadratic(
        1.0 / (2.0 * log_ops),
        1.0,
        remaining_ops_epsilon,
    )
    log_failure = math.log(1.0 / utility_failure_piece)
    posterior_tail = d / 2.0 + math.sqrt(d * log_failure) + log_failure
    c1 = posterior_tail * log_ops / eps_bar**2

    # U = f + sigma * (Z + q_safety) satisfies U >= f on the charged safety event
    # and U - f <= sigma * (q_safety + q_accuracy) on the upper-tail event; with
    # sensitivity log(1 + R_x^2 / h) this gives C2 in Wang's Equation (7).
    c2 = gaussian_multiplier * (safety_quantile + accuracy_quantile)

    # Half the OPS epsilon goes to the leverage-only term of Wang's Remark 11.
    # Algorithm 1's printed t_min is four times too small for that epsilon/2
    # target; this follows the author code.
    minimum_regularized_curvature = (
        feature_radius**2 * (1.0 + log_ops) / posterior_budget.eps
    )
    lambda_ = _minimize_equation_7(
        lambda_min_tilde,
        feature_radius,
        c1,
        c2,
        minimum_regularized_curvature,
    )

    # Step 5: compute the regularized least-squares center.
    ridge_matrix: Array2D = gram + lambda_ * np.eye(d)
    theta_hat: Array1D = np.linalg.solve(ridge_matrix, cross_product)

    # Step 6: privately upper-bound L(theta_hat).
    lambda_star_tilde = lambda_min_tilde + lambda_
    log_L_sensitivity = math.log1p(feature_radius**2 / lambda_star_tilde)
    log_L_sigma = gaussian_multiplier * log_L_sensitivity
    log_L_center = math.log(
        response_radius + feature_radius * np.linalg.norm(theta_hat)
    )
    delta_release = (
        log_L_center + log_L_sigma * rng.normal() + log_L_sigma * safety_quantile
    )
    log_L_tilde = math.log(feature_radius) + delta_release

    # Step 7: with lambda fixed, calibrate the exact Remark 11 OPS temperature.
    # Infeasibility depends on a private intermediate, so it is an execution
    # failure, not a regime refusal.
    try:
        scaled_sqrt_gamma = solve_ops_scaled_temperature(
            lambda_star_tilde, feature_radius, posterior_budget
        )
    except ValueError as error:
        raise RuntimeError(
            "AdaOPS regularization left no positive OPS temperature."
        ) from error
    inverse_sqrt_gamma = math.exp(log_L_tilde - math.log(scaled_sqrt_gamma))
    if inverse_sqrt_gamma == 0.0:
        raise RuntimeError("AdaOPS posterior scale underflowed to zero.")

    # Output: draw once from the calibrated Gaussian ridge posterior.
    return sample_ops_posterior(
        theta_hat, ridge_matrix, inverse_sqrt_gamma, posterior_seed
    )


class AdaOPS(BaseAlgorithm):
    """Adaptive One-Posterior-Sample. Fit linear regression privately by drawing one random sample from a ridge posterior.

    An implementation of Algorithm 1 from Wang (2018). Privately estimates the
    conditioning of the data to bound the sensitivity of OLS, picks a ridge
    penalty from that estimate, and outputs a sample from the resulting Gaussian
    posterior over the coefficients. It is an approximate-DP mechanism.

    Requires:
        - Finite ``design_domain.l2_norm_bound`` and
          ``response_domain.l2_norm_bound``.
        - A finite ``ApproxDPBudget``.

    Parametrization notes:
        - ``failure_probability``: Corresponds to $ρ$ in the paper. Set as
          Algorithm 1 sets it, which the authors' evaluation also uses;
          Appendix C.4 allows any value in $(0, 1)$.

    Implementation notes:
        - Algorithm 1 as printed has two gaps, each closed here. Its Step 2
          leaves out the ``||X||^2`` sensitivity of the eigenvalue step, which
          is included. Its smallest temperature leaves no budget for a
          posterior draw, so the corrected value from the authors' code is
          used.
        - Each Gaussian step uses exact analytic-Gaussian calibration rather
          than the paper's closed-form noise levels, and the posterior draw
          uses the per-instance privacy quadratic of Remark 11.
        - Appendix C.5 reuses each Gaussian release's privacy delta as that
          release's confidence event, without proof. That reuse is not made:
          the two Gaussian releases, the two failure events their privacy
          depends on, and the posterior draw are each charged separately, with
          each failure event spending at most ``delta/5``.
        - ``failure_probability`` is the total probability of the five tail
          events in the paper's three failure categories.

    Reference:
        - wang2018revisiting: Algorithm 1.
    """

    TASK = LinearRegression

    failure_probability: float = parameter_field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Total failure probability for the eigenvalue, log-scale, and posterior-concentration categories in the risk argument, including the two safety tails.",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        budget = accounting.spend_as_approx_dp(budget)
        if not math.isfinite(budget.eps):
            raise InvalidRegime("AdaOPS requires a finite epsilon.")
        feature_radius = metadata.design_domain.l2_norm_bound
        response_radius = metadata.response_domain.l2_norm_bound
        if not math.isfinite(feature_radius) or not math.isfinite(response_radius):
            raise InvalidRegime("AdaOPS requires finite covariate and response bounds.")
        beta = _run_algorithm_1(
            input.X,
            input.y,
            budget,
            feature_radius,
            response_radius,
            self.failure_probability,
            seed,
        )
        return LinearRegressionOutput(beta=beta)
