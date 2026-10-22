"""Local-sensitivity robust regression from Asi, Duchi, and Talwar.

The implementation specializes Algorithms 2 and 3 of
"On Privately Estimating a Single Coefficient" (arXiv:2503.17252v1) to the
paper's robust-log regression loss. It uses the public-indexed add/remove
derivation: one changed record contributes ``G0/n`` to the gradient change,
while the positive-semidefinite Hessian contribution remains ``G1/n``.
"""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import NoRelease
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np
from scipy.optimize import OptimizeResult, root

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
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


def _solve_robust_log(
    X: Array2D,
    y: Array1D,
    public_n: int,
    *,
    root_tol: float,
    grad_tol: float,
) -> tuple[Array1D, Array2D]:
    """Return the robust-log minimizer and empirical Hessian at that point."""

    def gradient(beta: Array1D) -> Array1D:
        residuals: Array1D = y - X @ beta
        return -(X.T @ np.tanh(residuals / 2.0)) / public_n

    def hessian(beta: Array1D) -> Array2D:
        residuals: Array1D = y - X @ beta
        tanh_values: Array1D = np.tanh(residuals / 2.0)
        weights: Array1D = (1.0 - tanh_values**2) / 2.0
        return (X.T @ (weights[:, None] * X)) / public_n

    result: OptimizeResult = root(
        gradient,
        x0=np.zeros(X.shape[1]),
        jac=hessian,
        method="hybr",
        tol=root_tol,
    )
    beta: Array1D = np.asarray(result.x, dtype=float)
    grad_norm = np.linalg.norm(gradient(beta))
    if (
        not np.all(np.isfinite(beta))
        or not np.isfinite(grad_norm)
        or grad_norm > grad_tol
    ):
        raise RuntimeError(
            "LocalOutputPerturbation: the numerical optimizer did not "
            "reach the required first-order tolerance; the paper's "
            "exact-minimizer argument cannot be applied."
        )
    return beta, hessian(beta)


def _run_algorithm_2(
    lambda_min: float,
    public_n: int,
    x_bound: float,
    budget: DPBudget,
    failure_probability: float,
    seed: int,
) -> float:
    """Privately lower-bound curvature using corrected paper Algorithm 2.

    The arXiv pseudocode applies the recursion a continuous noisy number of
    times even though composition was defined only for integers. Flooring the
    nonnegative count makes the mechanism well-defined and preserves the
    lower-bound event used by Proposition 6.1.
    """
    eps = accounting.spend_as_pure(budget).eps
    # Section 3 fixes rho=1/2 and alpha=1.2332 for phi(s)=exp(s)-1.
    alpha = 1.2332
    condition_threshold = (8.0 * alpha + 1.0) * x_bound**2 / public_n

    def recurse(value: float) -> float:
        if value < condition_threshold:
            return 0.0
        discriminant = 1.0 - 4.0 * alpha * x_bound**2 / (public_n * value)
        if discriminant < 0.0:
            raise RuntimeError(
                "LocalOutputPerturbation: the local stability quadratic "
                "has no real root at a curvature above the condition "
                "threshold, so the recursion has no source-defined step."
            )
        t = (1.0 - math.sqrt(discriminant)) / (2.0 * alpha * x_bound)
        next_value = value * max(2.0 - math.exp(x_bound * t), 0.0) - x_bound**2 / (
            2.0 * public_n
        )
        return max(next_value, 0.0)

    num_steps = 0
    current = lambda_min
    while current > 0.0:
        current = recurse(current)
        num_steps += 1

    rng = np.random.default_rng(seed)
    noisy_steps = num_steps + rng.laplace(scale=1.0 / eps)
    confidence_offset = math.log(1.0 / (2.0 * failure_probability)) / eps
    released_steps = math.floor(max(noisy_steps - confidence_offset, 0.0))
    if released_steps == 0:
        return 0.0

    def reaches_zero(value: float) -> bool:
        for _ in range(released_steps):
            value = recurse(value)
        return value == 0.0

    return bisect_monotone_boundary(
        reaches_zero,
        0.0,
        condition_threshold,
        satisfying_endpoint="lower",
    )


def _run_algorithm_3(
    X: Array2D,
    y: Array1D,
    public_n: int,
    x_bound: float,
    budget: ApproxDPBudget,
    seed: int,
) -> Array1D:
    """Run the paper's robust-regression specialization of Algorithm 3."""
    beta, hessian = _solve_robust_log(
        X,
        y,
        public_n,
        root_tol=1e-11,
        grad_tol=1e-7,
    )

    curvature_seed, release_seed = split_seed(seed, 2)
    lambda_min = max(float(np.linalg.eigvalsh(hessian)[0]), 0.0)
    curvature_budget = PureDPBudget(eps=budget.eps / 2.0)
    release_budget = ApproxDPBudget(
        eps=budget.eps / 2.0,
        delta=budget.delta / 2.0,
    )
    lambda_hat = _run_algorithm_2(
        lambda_min,
        public_n,
        x_bound,
        curvature_budget,
        failure_probability=budget.delta / 2.0,
        seed=curvature_seed,
    )
    # The refusal bit is a function of Algorithm 2's transcript alone, whose
    # eps_cert spend and delta_cert failure lie inside the offered (eps, delta),
    # so the joint output over coefficient vectors and bottom is covered.
    if lambda_hat == 0.0:
        raise NoRelease(
            "LocalOutputPerturbation released no estimate because its "
            "private curvature lower bound was zero."
        )

    # Add/remove local sensitivity: Equation (9) with G0=B and 2G0/n
    # replaced by the single changed contribution G0/n.
    alpha = 1.2332
    discriminant = 1.0 - 4.0 * alpha * x_bound**2 / (public_n * lambda_hat)
    if discriminant < 0.0:
        raise RuntimeError(
            "LocalOutputPerturbation: the certified curvature lower "
            "bound admits no real root of the local stability quadratic, so "
            "no source-defined release radius exists."
        )
    sensitivity = (1.0 - math.sqrt(discriminant)) / (2.0 * alpha * x_bound)
    noise_std = accounting.calibrate_gaussian_scale(
        release_budget,
        sensitivity=sensitivity,
    )
    rng = np.random.default_rng(release_seed)
    return beta + rng.normal(scale=noise_std, size=beta.shape)


class LocalOutputPerturbation(BaseAlgorithm):
    """Fit a robust regression and add Gaussian noise scaled to how stable the fit is.

    An implementation of Algorithms 2 and 3 from Asi, Duchi and Talwar (2025).
    Fits the coefficients by minimizing a smooth robust loss of the residuals,
    privately certifies how curved the loss is near the fit, and outputs the
    whole coefficient vector with Gaussian noise scaled to that certificate. It
    is an approximate-DP mechanism.

    Requires:
        - A finite ``design_domain.l2_norm_bound``.
        - An ``ApproxDPBudget`` or an unconstrained budget.

    Implementation notes:
        - Of the losses the paper allows, this uses the robust log loss
          ``log(1+exp(r)) + log(1+exp(-r))`` of each residual ``r`` and
          outputs the whole coefficient vector. The target is therefore the
          minimizer of that loss, a robust coefficient, not the least-squares
          one.
        - For rows with ``||x_i||_2 <= B`` the gradient and Hessian bounds are
          ``G0 = B`` and ``G1 = B^2/2``. The paper's stability and
          private-curvature recursion are rederived for adding or removing one
          record at the declared ``n``.
        - The paper assumes the exact minimizer. The numerical solver must
          reach a strict first-order tolerance and raises ``RuntimeError``
          otherwise.
        - A failed curvature certificate releases nothing.

    Reference:
        - asi2025privately: Algorithms 2 and 3 of arXiv v1.
    """

    TASK = LinearRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.THEORETICAL
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        x_bound = metadata.design_domain.l2_norm_bound
        if not math.isfinite(x_bound):
            raise InvalidRegime(
                "LocalOutputPerturbation requires a finite covariate norm bound."
            )
        budget = accounting.spend_as_approx_dp(budget)
        beta = _run_algorithm_3(
            input.X,
            input.y,
            metadata.n_samples,
            x_bound,
            budget,
            seed,
        )
        return LinearRegressionOutput(beta=beta)
