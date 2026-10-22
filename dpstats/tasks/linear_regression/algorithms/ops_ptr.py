"""Generalized-PTR one-posterior-sample linear regression."""

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

from dpstats.parameters import parameter_field, search_values
from dpstats.task import Array1D, Array2D
from dpstats.utils import split_seed
from dpstats.tasks.linear_regression.algorithms._ops import (
    sample_ops_posterior,
    solve_ops_scaled_temperature,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _run_algorithm_5(
    X: Array2D,
    y: Array1D,
    response_radius: float,
    regularization: float,
    budget: ApproxDPBudget,
    seed: int,
) -> Array1D:
    """Run Algorithm 5 after the public feature radius has been scaled to one.

    ``budget`` is the budget offered by the platform. Theorem C.1 states
    Algorithm 5 with paper parameter ``delta`` is ``(eps, 2 * delta)``-DP, so
    this transcription uses half the offered delta internally.
    """
    d = X.shape[1]
    gram: Array2D = X.T @ X
    cross_product: Array1D = X.T @ y
    lambda_min = np.linalg.eigvalsh(gram)[0]
    release_seed, posterior_seed = split_seed(seed, 2)
    rng = np.random.default_rng(release_seed)

    paper_delta = budget.delta / 2.0
    scalar_budget = ApproxDPBudget(eps=budget.eps / 4.0, delta=paper_delta / 3.0)
    posterior_budget = ApproxDPBudget(eps=budget.eps / 2.0, delta=paper_delta / 3.0)
    confidence_quantile = norm.isf(paper_delta / 2.0)

    # Algorithm 5, Steps 2-3: privately lower-bound lambda_min(X.T @ X).
    eigen_sigma = accounting.calibrate_gaussian_scale(scalar_budget, 1.0)
    lambda_min_tilde = max(
        lambda_min + eigen_sigma * rng.normal() - eigen_sigma * confidence_quantile,
        0.0,
    )

    # Algorithm 5, Steps 4-5: fit ridge, then privately upper-bound
    # log(||Y|| + ||X|| * ||theta_hat||), with ||X|| = 1 here.
    ridge_matrix: Array2D = gram + regularization * np.eye(d)
    # theta_hat is both the ridge estimate and the OPS posterior mean.
    theta_hat: Array1D = np.linalg.solve(ridge_matrix, cross_product)
    lambda_star_tilde = regularization + lambda_min_tilde
    log_scale_sensitivity = math.log1p(1.0 / lambda_star_tilde)
    log_scale_sigma = accounting.calibrate_gaussian_scale(
        scalar_budget, log_scale_sensitivity
    )
    log_scale = math.log(response_radius + np.linalg.norm(theta_hat))
    # With ||X|| = 1, Algorithm 5's Delta is log(L_tilde).
    log_L_tilde = (
        log_scale
        + log_scale_sigma * rng.normal()
        + log_scale_sigma * confidence_quantile
    )

    # Algorithm 5, Steps 6-7: set L_tilde and calibrate the largest gamma
    # Theorem 5.1 allows. Infeasibility depends on a private intermediate, so
    # it is an execution failure, not a regime refusal.
    try:
        t = solve_ops_scaled_temperature(
            lambda_star_tilde, feature_radius=1.0, budget=posterior_budget
        )
    except ValueError as error:
        raise RuntimeError(
            "OPSPTR could not calibrate a positive posterior temperature."
        ) from error
    log_inverse_sqrt_gamma = log_L_tilde - math.log(t)
    inverse_sqrt_gamma = math.exp(log_inverse_sqrt_gamma)
    if inverse_sqrt_gamma == 0.0:
        raise RuntimeError("OPSPTR posterior scale underflowed to zero.")

    # Algorithm 5, Step 8: draw from the calibrated ridge posterior.
    return sample_ops_posterior(
        theta_hat, ridge_matrix, inverse_sqrt_gamma, posterior_seed
    )


class OPSPTR(BaseAlgorithm):
    """One-Posterior-Sample with Propose-Test-Release. Fit linear regression privately by drawing one random sample from a ridge posterior, sized by private bounds.

    An implementation of Algorithm 5 from Redberg, Zhu and Wang (2023).
    Privately bounds the smallest eigenvalue of X^T X from below and how much
    one record can move the ridge fit from above, then outputs one random draw
    from a Gaussian posterior around the ridge fit, as concentrated as those
    bounds allow. It is an approximate-DP mechanism.

    Requires:
        - A finite ``design_domain.l2_norm_bound``.
        - A finite, strictly positive ``response_domain.l2_norm_bound``.
        - A finite ``ApproxDPBudget``.

    Parametrization notes:
        - ``regularization``: Corresponds to $λ$, the Algorithm 5 input,
          applied after unit feature scaling. The search values are the 30-point
          log grid over $[2.5, 2.5^{10}]$ from Section 5.1 of Redberg, Zhu and
          Wang (2023). The paper
          names no chosen value, so the default is one of the grid's two
          middle points, $i = 14$. The lower one is used because dividing the
          features by the public bound here gives the data roughly nine times
          less curvature than the paper's coordinates, so a given $λ$ is
          relatively about nine times stronger.
        - Fixed budget split: Algorithm 5 and Theorem C.1 give $ε/4$ to the
          $λ_{min}$ lower bound and $ε/2$ to the posterior and split $δ$ into
          thirds, inherited from Appendix C.3 of Wang (2018). The split is not
          exposed.

    Implementation notes:
        - The two noisy scalars use exact analytic-Gaussian calibration rather
          than the paper's closed-form noise levels.
        - Algorithm 5 assumes ``||x_i|| <= 1``, so the features are divided by
          the public norm bound and the drawn coefficients mapped back. The upper
          bound released privately is ``log(||Y|| + ||theta_hat||)``, the log
          of the ridge fit's local Lipschitz constant once features have unit
          norm.
        - If the posterior temperature cannot be calibrated, the run fails with
          an execution error, not a regime refusal, because the failure depends
          on a privately released value.

    Reference:
        - redberg2023generalized: Algorithm 5 and Theorem C.1 (arXiv v1 numbering;
          Algorithm 7 and Theorem D.1 in the published version).
        - wang2018revisiting: Appendix C.3.
    """

    TASK = LinearRegression

    regularization: float = parameter_field(
        default=2.5 ** (1 + 9 * 14 / 29),
        gt=0.0,
        search=search_values(*(2.5 ** (1 + 9 * i / 29) for i in range(30))),
        description="Ridge regularizer after unit feature scaling.",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        feature_radius = metadata.design_domain.l2_norm_bound
        response_radius = metadata.response_domain.l2_norm_bound
        if not math.isfinite(feature_radius) or not math.isfinite(response_radius):
            raise InvalidRegime("OPSPTR requires finite covariate and response bounds.")
        if response_radius == 0.0:
            raise InvalidRegime(
                "OPSPTR requires a strictly positive response norm bound "
                "because Algorithm 5 privatizes its logarithm."
            )
        budget = accounting.spend_as_approx_dp(budget)
        if not math.isfinite(budget.eps):
            raise InvalidRegime("OPSPTR requires a finite epsilon.")
        X = input.X
        y = input.y

        # Algorithm 5 assumes ||x_i|| <= 1; rescale by the public feature radius.
        X_normalized: Array2D = X / feature_radius
        theta_tilde = _run_algorithm_5(
            X=X_normalized,
            y=y,
            response_radius=response_radius,
            regularization=self.regularization,
            budget=budget,
            seed=seed,
        )
        beta: Array1D = theta_tilde / feature_radius
        return LinearRegressionOutput(beta=beta)
