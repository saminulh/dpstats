"""The ``BoostedAdaSSP`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import DPBudget, GDPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math
from typing import Literal, cast

import numpy as np

from dpstats.parameters import parameter_field, search_range, search_values
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _run_algorithm_1(
    X: Array2D,
    y: Array1D,
    *,
    budget: GDPBudget,
    a: float,
    b: float,
    c: float,
    tau: float,
    T: int,
    learning_rate_schedule: Literal["constant", "inverse", "inverse_sqrt"],
    seed: int,
) -> Array1D:
    """Run Tang et al. Algorithm 1 on its unit-row-clipped design."""
    d = X.shape[1]
    theta = np.zeros(d)

    # GDP composes in squared mu, so squared weights realize the paper's
    # authored mu_1:mu_2:mu_3 = a:b:c allocation.
    gram_budget, gradient_budget, eigenvalue_budget = budget.split(
        (a**2, b**2, c**2),
    )
    gram_noise_scale = accounting.calibrate_gaussian_scale(gram_budget, 1.0)
    eigenvalue_noise_scale = accounting.calibrate_gaussian_scale(eigenvalue_budget, 1.0)
    gradient_noise_multiplier = (
        accounting.compute_noise_multiplier_for_composed_gaussian(
            gradient_budget,
            T,
        )
    )

    rng = np.random.default_rng(seed)
    X = clip_rows(X, 1.0)
    covariance = X.T @ X
    # Wang's AdaSSP Algorithm 2 adds i.i.d. Gaussian noise to the unique
    # upper triangle and reflects it to preserve the sufficient statistic.
    covariance_noise = rng.normal(scale=gram_noise_scale, size=(d, d))
    covariance_noise = np.triu(covariance_noise) + np.triu(covariance_noise, 1).T
    private_covariance = covariance + covariance_noise

    min_eigenvalue = np.linalg.eigvalsh(covariance)[0]
    private_lambda = min_eigenvalue + rng.normal(scale=eigenvalue_noise_scale)
    # The author implementation realizes Algorithm 1's inverse with NumPy's
    # pseudoinverse at its default cutoff.
    inverse_system = cast(
        Array2D,
        np.linalg.pinv(
            private_covariance + private_lambda * np.eye(d),
            rcond=1e-15,
        ),
    )

    for round_index in range(1, T + 1):
        gradient = y - X @ theta
        clipped_gradient = np.clip(gradient, -tau, tau)
        private_cross_product = X.T @ clipped_gradient + rng.normal(
            scale=tau * gradient_noise_multiplier,
            size=d,
        )
        if learning_rate_schedule == "constant":
            learning_rate = 1.0
        elif learning_rate_schedule == "inverse":
            learning_rate = 1.0 / round_index
        else:
            learning_rate = 1.0 / math.sqrt(round_index)
        theta += learning_rate * (inverse_system @ private_cross_product)

    return theta


class BoostedAdaSSP(BaseAlgorithm):
    """Fit private linear regression by repeatedly correcting clipped residuals.

    An implementation of Algorithm 1 from Tang et al. (2024). Forms a noisy X^T
    X and a private lower bound on its smallest eigenvalue, which together fix
    one stabilized system to solve, then runs rounds of boosting, each clipping
    the current residuals, forming a noisy product of the data with them, and
    correcting the coefficients. It is a Gaussian-DP mechanism.

    Requires:
        - A finite ``design_domain.l2_norm_bound``.
        - A budget family admitting a GDP implication.

    Parametrization notes:
        - ``num_rounds``: Corresponds to $T$ in the paper. Set to the authors'
          fixed evaluation setting; the search range comes from Appendix A.1.
        - ``tau``: Corresponds to the paper's residual clip. Set to the authors'
          fixed evaluation setting; the search range comes from Appendix A.1.
        - ``learning_rate_schedule``: The default is the unit learning rate of
          the authors' fixed evaluation setting. Algorithm 1 displays only
          that unit-rate update; the three schedules are those Appendix A.1
          varies.
        - Fixed GDP shares $a$, $b$ and $c$: They are the authors' equal 1:1:1
          allocation and are not exposed.

    Implementation notes:
        - Algorithm 1 clips each feature row to unit L2 norm. With a public
          feature bound ``B``, the rows are divided by ``B``, Algorithm 1 runs
          on the rescaled data, and the coefficients are divided by ``B`` again
          on the way out.

    Reference:
        - tang2024improved: Algorithm 1.
    """

    TASK = LinearRegression

    num_rounds: int = parameter_field(
        default=100,
        ge=1,
        search=search_range(1, 4000),
        description="Number of residual-correction rounds.",
    )
    tau: float = parameter_field(
        default=1.0,
        gt=0.0,
        search=search_range(1e-5, 1e5, scale="log"),
        description="Paper clipping threshold for scalar residuals.",
    )
    learning_rate_schedule: Literal["constant", "inverse", "inverse_sqrt"] = (
        parameter_field(
            default="constant",
            search=search_values("constant", "inverse", "inverse_sqrt"),
            description="Appendix A.1 update-weight schedule.",
        )
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
        if not math.isfinite(feature_radius):
            raise InvalidRegime(
                "BoostedAdaSSP requires a finite covariate norm bound for its "
                "unit-domain coordinate transport."
            )
        gdp_budget = accounting.spend_as_gdp(budget)
        X_normalized: Array2D = input.X / feature_radius

        theta_unit = _run_algorithm_1(
            X_normalized,
            input.y,
            budget=gdp_budget,
            a=1.0,
            b=1.0,
            c=1.0,
            tau=self.tau,
            T=self.num_rounds,
            learning_rate_schedule=self.learning_rate_schedule,
            seed=seed,
        )
        return LinearRegressionOutput(beta=theta_unit / feature_radius)
