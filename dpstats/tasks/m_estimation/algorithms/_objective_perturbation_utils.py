"""Shared objective-perturbation calibration, roots, and loss admission."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
from scipy.optimize import OptimizeResult, root
from scipy.optimize import minimize_scalar  # pyright: ignore[reportUnknownVariableType]
from scipy.special import log_ndtr

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    accounting,
)
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D, Array2D
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
)
from dpstats.utils import bisect_monotone_boundary


def _rkw_objective_perturbation_delta(
    scale: float,
    eps: float,
    gradient_bound: float,
    regularization_multiplier: float,
) -> float:
    """Evaluate the corrected RKW Appendix-D objective privacy profile."""
    log_determinant_cost = abs(math.log1p(-1.0 / regularization_multiplier))
    privacy_loss_mean = gradient_bound**2 / (2.0 * scale**2)
    remaining_eps = eps - log_determinant_cost
    centered_eps = remaining_eps - privacy_loss_mean
    if centered_eps >= 0.0:
        return 2.0 * accounting.compute_delta_from_gaussian_mechanism(
            scale=scale,
            sensitivity=gradient_bound,
            eps=remaining_eps,
        )
    return (1.0 - math.exp(centered_eps)) + math.exp(
        centered_eps
    ) * 2.0 * accounting.compute_delta_from_gaussian_mechanism(
        scale=scale,
        sensitivity=gradient_bound,
        eps=privacy_loss_mean,
    )


def calibrate_rkw_objective_ridge(
    budget: DPBudget,
    gradient_bound: float,
    smoothness_bound: float,
    objective_noise_ratio: float,
) -> tuple[float, float]:
    """Resolve RKW's Section 5 ridge rule: the least ridge above beta meeting sigma = f * sigma_G."""
    budget = accounting.spend_as_approx_dp(budget)
    if budget.is_unconstrained:
        return smoothness_bound, 0.0
    sigma = objective_noise_ratio * accounting.calibrate_gaussian_scale(
        budget, gradient_bound
    )

    def clears_profile(multiplier: float) -> bool:
        return (
            multiplier > 1.0
            and _rkw_objective_perturbation_delta(
                sigma, budget.eps, gradient_bound, multiplier
            )
            <= budget.delta
        )

    try:
        multiplier = bisect_monotone_boundary(
            clears_profile, 1.0, 2.0, satisfying_endpoint="upper"
        )
    except RuntimeError as error:
        raise InvalidRegime(
            "RKW's Section 5 ridge rule is infeasible: no finite "
            "regularization keeps the objective profile's delta at or below "
            "the budget's delta at this objective_noise_ratio."
        ) from error
    return multiplier * smoothness_bound, sigma


def calibrate_rkw_amp(
    budget: ApproxDPBudget,
    gradient_bound: float,
    smoothness_bound: float,
    objective_noise_ratio: float,
) -> tuple[float, float]:
    """Resolve RKW objective noise and regularization for AMP's objective child.

    Args:
        budget: Finite approximate-DP budget for the objective child.
        gradient_bound: Positive ambient loss-gradient bound C.
        smoothness_bound: Positive ambient loss-smoothness bound beta.
        objective_noise_ratio: Positive ratio f to the C-sensitive Gaussian.

    Returns:
        Regularization lambda and objective Gaussian deviation sigma. The
        Gaussian comparator and objective RDP curve use continuous CKS;
        lambda is the satisfying endpoint of the objective calibration.

    Raises:
        RuntimeError: If numerical calibration does not find a satisfying point.

    Example:
        >>> budget = ApproxDPBudget(eps=0.99, delta=9.9e-6)
        >>> lam, sigma = calibrate_rkw_amp(budget, 1.0, 2.0, 1.3)
        >>> round(lam, 5), round(sigma, 5)
        (10.37269, 5.31038)
    """
    rho = accounting.spend_as_zcdp_via_rdp(budget).rho
    sigma = objective_noise_ratio * gradient_bound / math.sqrt(2.0 * rho)
    mu = gradient_bound / sigma

    def epsilon_at_log_order_gap(log_gap: float) -> float:
        # Searching log(alpha - 1) keeps every evaluated order above one.
        gap = math.exp(log_gap)
        alpha = 1.0 + gap
        rdp = alpha * mu**2 / 2.0 + (math.log(2.0) + float(log_ndtr(gap * mu))) / gap
        return accounting.compute_epsilon_bound_from_rdp(rdp, alpha, budget.delta)

    result: OptimizeResult = minimize_scalar(
        epsilon_at_log_order_gap,
        bracket=(0.0, 1.0),
        options={"xtol": 1e-8, "maxiter": 200},
    )
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("RKW AMP Renyi-order calibration did not converge.")
    if result.fun >= budget.eps:
        raise RuntimeError(
            "RKW AMP calibration could not find a feasible Renyi order even "
            "at infinite regularization; increase objective_noise_ratio."
        )
    # Every evaluated order gives a sound upper bound, so the numerical search
    # need not find a global minimum.
    regularization = bisect_monotone_boundary(
        lambda lam: lam > smoothness_bound
        and -math.log1p(-smoothness_bound / lam) + result.fun <= budget.eps,
        smoothness_bound,
        2.0 * smoothness_bound,
        satisfying_endpoint="upper",
    )
    return regularization, sigma


def kst_objective_terms(
    zeta: float,
    lambda_bound: float,
    dim: int,
    budget: PureDPBudget | ApproxDPBudget,
    seed: int,
) -> tuple[float, Array1D]:
    """Return KST Algorithm 1's curvature and sampled linear perturbation."""
    rng = np.random.default_rng(seed)
    if isinstance(budget, PureDPBudget):
        direction: Array1D = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        perturbation = rng.gamma(shape=dim, scale=2.0 * zeta / budget.eps) * direction
    else:
        noise_scale = (
            zeta
            * math.sqrt(8.0 * math.log(2.0 / budget.delta) + 4.0 * budget.eps)
            / budget.eps
        )
        perturbation = rng.normal(size=dim) * noise_scale
    return 2.0 * lambda_bound / budget.eps, perturbation


def perturbed_gradient(
    loss: MEstimationLoss,
    data: Array2D,
    theta: Array1D,
    curvature: float,
    perturbation: Array1D,
) -> Array1D:
    """Evaluate the summed loss gradient plus ``lambda theta + b``."""
    return (
        loss.gradient(data, theta, reduction="sum") + curvature * theta + perturbation
    )


def solve_perturbed_objective(
    loss: MEstimationLoss,
    data: Array2D,
    curvature: float,
    perturbation: Array1D,
    dim: int,
    gradient_scale: float,
    algorithm_name: str,
) -> Array1D:
    """Return a finite first-order root of one strongly convex objective."""

    def gradient(theta: Array1D) -> Array1D:
        return perturbed_gradient(loss, data, theta, curvature, perturbation)

    try:
        hessian_at_zero = loss.hessian(data, np.zeros(dim), reduction="sum")
        jacobian: Callable[[Array1D], Array2D] | None = None
        if hessian_at_zero is not None:
            ridge_hessian = curvature * np.eye(dim)

            def analytic_jacobian(theta: Array1D) -> Array2D:
                loss_hessian = loss.hessian(data, theta, reduction="sum")
                if loss_hessian is None:
                    raise RuntimeError(
                        "MEstimationLoss.hessian must remain available throughout "
                        "one solve."
                    )
                return loss_hessian + ridge_hessian

            jacobian = analytic_jacobian

        result: OptimizeResult = root(
            gradient, x0=np.zeros(dim), jac=jacobian, method="hybr", tol=1e-11
        )
        theta: Array1D = np.asarray(result.x, dtype=float)
        gradient_norm = float(np.linalg.norm(gradient(theta)))
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise RuntimeError(
            f"{algorithm_name}: the numerical solve of the perturbed gradient failed."
        ) from error
    if not np.all(np.isfinite(theta)) or not gradient_norm <= 1e-7 * gradient_scale:
        raise RuntimeError(
            f"{algorithm_name}: the numerical optimizer did not reach the required "
            "first-order tolerance; the exact-minimizer privacy argument cannot be "
            "applied."
        )
    return theta


def admit_loss(loss: MEstimationLoss, properties: LossProperties) -> None:
    """Refuse a loss outside objective perturbation's shared hypotheses.

    ``properties`` is already resolved on the declared public domains, so every
    refusal is decided from public facts alone, before any record is read.
    """
    name = type(loss).__name__
    if not properties.convex:
        raise InvalidRegime(
            f"Objective perturbation requires a convex loss; {name} declares "
            "itself nonconvex."
        )
    if not properties.generalized_linear:
        raise InvalidRegime(
            "Objective perturbation requires a generalized-linear loss; "
            f"{name} is not one."
        )
    if not math.isfinite(properties.loss_lipschitz_constant) or not math.isfinite(
        properties.gradient_lipschitz_constant
    ):
        raise InvalidRegime(
            "Objective perturbation requires finite gradient and smoothness "
            f"bounds; {name} gives zeta={properties.loss_lipschitz_constant} and "
            f"lambda={properties.gradient_lipschitz_constant} on the declared domains."
        )
