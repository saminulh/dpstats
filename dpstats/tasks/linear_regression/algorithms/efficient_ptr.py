"""The ``EfficientPTR`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.exceptions import NoRelease
from dpstats.budgets import ApproxDPBudget, DPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np
from scipy.special import expit

from dpstats.parameters import parameter_field
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _run_algorithm_2(
    theta_hat: Array1D,
    gamma: float,
    alpha: float,
    budget: ApproxDPBudget,
    seed: int,
) -> Array1D:
    """Run Algorithm 2's efficient PTR mechanism for a vector-valued estimator.

    ``gamma`` is the paper's one-Lipschitz safety lower bound and ``alpha`` is
    the local L2-sensitivity bound on the positive-safety region.
    """
    # Algorithm 2: privately decide whether to output an estimate.
    threshold = 1.0 + (2.0 / budget.eps) * math.log(
        max(1.0 / budget.delta, 1.0 / budget.eps)
    )
    safety_margin = gamma - threshold
    logit = 0.0 if safety_margin == 0.0 else 0.5 * budget.eps * safety_margin
    release_probability = expit(logit)

    rng = np.random.default_rng(seed)
    if rng.random() >= release_probability:
        # Theorem 2.2 covers the joint vector-or-bottom transcript.
        raise NoRelease("EfficientPTR released the accounted PTR bottom outcome.")

    # Algorithm 2's printed scalar normal is N(0, I_p) for this vector query.
    release_budget = ApproxDPBudget(
        eps=budget.eps / 2.0,
        delta=budget.delta,
    )
    noise_scale = accounting.calibrate_gaussian_scale(release_budget, alpha)
    return theta_hat + noise_scale * rng.normal(size=theta_hat.shape)


def _run_algorithm_4(
    X: Array2D,
    y: Array1D,
    n: int,
    feature_radius: float,
    coefficient_radius: float,
    curvature_lower_bound: float,
    budget: ApproxDPBudget,
    seed: int,
) -> Array1D:
    """Run Algorithm 4's efficient-PTR OLS mechanism."""
    # Step 1: enforce the paper's public data bounds.
    X = clip_rows(X, feature_radius)
    response_radius = feature_radius * coefficient_radius
    y = clip_rows(y, response_radius)

    # Step 2: exact OLS followed by coefficient-ball projection.
    gram: Array2D = X.T @ X
    theta_raw: Array1D = np.linalg.solve(gram, X.T @ y)
    theta_hat: Array1D = theta_raw / max(
        1.0,
        np.linalg.norm(theta_raw) / coefficient_radius,
    )

    # Step 3: Equation (18)'s safety score and Equation (17)'s local
    # sensitivity on the positive-safety region.
    lambda_min = np.linalg.eigvalsh(gram)[0]
    gamma = max(
        lambda_min - curvature_lower_bound * n - 2.0 * feature_radius**2,
        0.0,
    ) / (2.0 * feature_radius**2)
    # Section 4.1's constant; radial projection is 2R-Lipschitz relative to
    # max(||u||, ||v||) + R, the proof step the paper leaves implicit.
    alpha = 4.0 * feature_radius**2 * coefficient_radius / (n * curvature_lower_bound)
    return _run_algorithm_2(theta_hat, gamma, alpha, budget, seed)


class EfficientPTR(BaseAlgorithm):
    """Efficient Propose-Test-Release. Fit least squares on clipped data and output it with Gaussian noise if a private safety test passes.

    An implementation of Algorithms 2 and 4 from Shen, Tong and Wang (2026).
    Clips each record's features and response to known radii, fits ordinary
    least squares, projects the coefficients into a ball of configured radius,
    and privately tests that the fit is stable on this data. If the test
    passes, outputs the coefficients with Gaussian noise; otherwise outputs
    nothing. It is an approximate-DP mechanism.

    Requires:
        - An ``ApproxDPBudget`` or an unconstrained budget.
        - A finite ``design_domain.l2_norm_bound``.

    Parametrization notes:
        - ``coefficient_radius``: Corresponds to $R_θ$ in the paper. Set as in
          the authors' synthetic varying-epsilon evaluation.
        - ``curvature_lower_bound``: Corresponds to $c_0$ in the paper. Set as
          in the authors' varying-epsilon evaluation driver.

    Implementation notes:
        - The paper's neighbors replace one record in a dataset of fixed size
          ``n``. Adding or removing a record is covered too, because an all-zero
          record changes none of the statistics, so an added or removed record can be
          treated as a replaced zero record.
        - Algorithm 4 needs the exact inverse of the Gram matrix, so a singular
          design raises an execution error rather than adding a second
          no-release outcome; Theorem 4.1's guarantee holds on the designs
          where the paper's estimator is defined.
        - The response is clipped at ``R_x * R_theta``, with the declared
          design bound as ``R_x``.

    Reference:
        - shen2026efficient: Algorithms 2 and 4 of arXiv v2.
    """

    TASK = LinearRegression

    coefficient_radius: float = parameter_field(
        default=1.0,
        gt=0.0,
        description="Paper quantity ``R_theta`` defining the projected coefficient ball and response clip.",
    )
    curvature_lower_bound: float = parameter_field(
        default=0.4,
        gt=0.0,
        description="Paper quantity ``c_0`` defining the safe eigenvalue region and local sensitivity.",
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

        feature_radius = metadata.design_domain.l2_norm_bound
        if not math.isfinite(feature_radius):
            raise InvalidRegime("EfficientPTR requires a finite covariate norm bound.")

        beta = _run_algorithm_4(
            X=input.X,
            y=input.y,
            n=metadata.n_samples,
            feature_radius=feature_radius,
            coefficient_radius=self.coefficient_radius,
            curvature_lower_bound=self.curvature_lower_bound,
            budget=budget,
            seed=seed,
        )
        return LinearRegressionOutput(beta=beta)
