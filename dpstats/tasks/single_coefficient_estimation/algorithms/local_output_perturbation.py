"""Local output perturbation of one robust-log coefficient (Asi, Duchi, Talwar)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from scipy.optimize import OptimizeResult, root
from scipy.stats import norm

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import ApproxDPBudget, DPBudget, accounting
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.parameters import parameter_field
from dpstats.task import Array1D, Array2D
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.utils import bisect_monotone_boundary, split_seed

# Section 3 fixes rho = 1/2 and alpha = 1.2332 for phi(s) = exp(s) - 1 as a pair.
_RHO = 0.5
_ALPHA = 1.2332
# ||h''||_inf for h(q) = log(1 + exp(q)) + log(1 + exp(-q)), whose h'' is
# sech(q/2)^2 / 2.
_H_SECOND_DERIVATIVE_BOUND = 0.5
_ROOT_TOLERANCE = 1e-11
_FIRST_ORDER_TOLERANCE = 1e-7


def _solve_robust_log(
    X: Array2D, y: Array1D, n: int, regularization: float
) -> tuple[Array1D, Array2D]:
    """Return equation (1)'s minimizer and the loss-only Hessian, divisor ``n``.

    The objective is the mean robust-log loss plus
    ``(regularization/2) ||theta||^2``, centred at the origin.

    Raises ``RuntimeError`` when the solve fails or its first-order residual
    exceeds the fixed tolerance, since the paper's theorem is stated for the
    exact minimizer.
    """

    def gradient(theta: Array1D) -> Array1D:
        # h'(q) = tanh(q/2) for the residual q = y - x^T theta.
        residual = y - X @ theta
        return -(X.T @ np.tanh(residual / 2.0)) / n + regularization * theta

    def loss_hessian(theta: Array1D) -> Array2D:
        residual = y - X @ theta
        weights = (1.0 - np.tanh(residual / 2.0) ** 2) / 2.0
        return (X.T @ (weights[:, None] * X)) / n

    try:
        result: OptimizeResult = root(
            gradient,
            x0=np.zeros(X.shape[1]),
            jac=lambda theta: loss_hessian(theta) + regularization * np.eye(X.shape[1]),
            method="hybr",
            tol=_ROOT_TOLERANCE,
        )
    except (ArithmeticError, ValueError) as error:
        raise RuntimeError(
            "LocalOutputPerturbation: the robust-log numerical optimizer failed."
        ) from error
    theta: Array1D = np.asarray(result.x, dtype=np.float64)
    residual_norm = float(np.linalg.norm(gradient(theta)))
    if (
        not result.success
        or not np.all(np.isfinite(theta))
        or not residual_norm <= _FIRST_ORDER_TOLERANCE
    ):
        raise RuntimeError(
            "LocalOutputPerturbation: the robust-log optimizer did not reach its "
            "first-order tolerance, so the exact-minimizer argument cannot apply."
        )
    return theta, loss_hessian(theta)


def _c1_threshold(B: float, n: int) -> float:
    """Condition (C1)'s right side for robust log, ``G0 = B``, ``G1 = B^2/2``.

    ``2 G0 alpha B/[rho(1-rho)n] + G1/(rho n)`` under the add/remove relation,
    with ``rho = _RHO``.
    """
    G0 = B
    G1 = B**2 / 2.0
    return 2.0 * G0 * _ALPHA * B / (_RHO * (1.0 - _RHO) * n) + G1 / (_RHO * n)


def _t_add(lam: float, B: float, n: int) -> float:
    """Equation (9)'s parameter-change bound with the add/remove term ``G0/n``.

    ``t_add(lambda) = [1 - sqrt(1 - 4 alpha B G0/(n lambda))]/(2 alpha B)`` with
    ``G0 = B``; real only for ``lambda >= 4 alpha B^2/n``.
    """
    return (1.0 - math.sqrt(1.0 - 4.0 * _ALPHA * B**2 / (n * lam))) / (2.0 * _ALPHA * B)


def _r_min(lam: float, B: float, n: int, *, regularization: float) -> float:
    """Equation (10)'s lower-curvature recursion, zero off (C1).

    ``R_min(lambda) = lambda * max(1 - phi(B t_add(lambda + lambda_reg)), 0)
    - G1/n`` with ``phi(s) = exp(s) - 1`` and ``G1 = B^2/2``, clamped to be
    nonnegative; (C1) reads ``lambda + lambda_reg/rho``.
    """
    if lam + regularization / _RHO < _c1_threshold(B, n):
        return 0.0
    G1 = B**2 / 2.0
    growth = math.exp(B * _t_add(lam + regularization, B, n))
    return max(lam * max(2.0 - growth, 0.0) - G1 / n, 0.0)


def _algorithm_2(
    lambda_min: float,
    B: float,
    n: int,
    budget: ApproxDPBudget,
    seed: int,
    *,
    regularization: float,
) -> float:
    """Algorithm 2's private lower bound on the empirical minimal eigenvalue.

    The released recursion count ``m = floor(max(N_hat - k, 0))`` is an
    integer, ``k = log(1/(2 delta))/eps``, and the bound is the largest
    ``lambda`` whose ``m``-fold recursion reaches zero.
    """
    epsilon = accounting.spend_as_pure(budget).eps
    N = 0
    current = lambda_min
    while current > 0.0:
        current = _r_min(current, B, n, regularization=regularization)
        N += 1
    N_hat = N + np.random.default_rng(seed).laplace(scale=1.0 / epsilon)
    k = math.log(1.0 / (2.0 * budget.delta)) / epsilon
    m = math.floor(max(N_hat - k, 0.0))
    if m == 0:
        return 0.0

    def reaches_zero(lam: float) -> bool:
        for _ in range(m):
            lam = _r_min(lam, B, n, regularization=regularization)
        return lam == 0.0

    return bisect_monotone_boundary(
        reaches_zero, 0.0, _c1_threshold(B, n), satisfying_endpoint="lower"
    )


def _algorithm_4(
    lambda_max: float,
    lambda_0: float,
    B: float,
    n: int,
    budget: ApproxDPBudget,
    seed: int,
) -> float:
    """Algorithm 4's private upper bound on the empirical maximal eigenvalue.

    Equation (12)'s recursion ``R(lambda) = min(lambda exp(B t_add(lambda_0)) +
    G1/n, G1)`` is inverted ``m = floor(max(N_hat - k, 0))`` times from ``G1``,
    an integer count, with ``k = log(1/(2 delta))/eps``.
    """
    epsilon = accounting.spend_as_pure(budget).eps
    G1 = B**2 / 2.0
    growth = math.exp(B * _t_add(lambda_0, B, n))

    def R(lam: float) -> float:
        return min(lam * growth + G1 / n, G1)

    N = 0
    current = lambda_max
    while current < G1:
        current = R(current)
        N += 1
    N_hat = N + np.random.default_rng(seed).laplace(scale=1.0 / epsilon)
    k = math.log(1.0 / (2.0 * budget.delta)) / epsilon
    m = math.floor(max(N_hat - k, 0.0))
    lambda_hat = G1
    for _ in range(m):
        target = lambda_hat
        lambda_hat = bisect_monotone_boundary(
            lambda lam: R(lam) >= target, 0.0, G1, satisfying_endpoint="upper"
        )
    return lambda_hat


def _passes_ratio_certificate(
    lambda_0: float,
    lambda_1: float,
    B: float,
    n: int,
    budget: ApproxDPBudget,
    *,
    regularization: float,
) -> bool:
    """Test Corollary 3.5's ``p=2`` ratio bound against (14a).

    Uses equation (13)'s constants with ``r = B``, and Lemma 8.3's
    ``s2 r^2/lambda_0`` in place of the printed ``s2 r``. Every denominator must
    be strictly positive and ``t_add(R(lambda_0))`` real, or the certificate
    fails.
    """
    epsilon, delta = budget.eps, budget.delta
    r = B
    t = _t_add(lambda_0, B, n)
    gamma = _ALPHA * r * t
    R_reg = (
        _r_min(lambda_0 - regularization, B, n, regularization=regularization)
        + regularization
    )
    gamma_margin = 1.0 - gamma
    if not gamma_margin > 0.0 or not R_reg >= 4.0 * _ALPHA * B**2 / n:
        return False
    gamma_prime = _ALPHA * r * _t_add(R_reg, B, n)
    beta = _H_SECOND_DERIVATIVE_BOUND * r**2 / (gamma_margin * n * lambda_0)
    if not beta < 1.0 or not gamma_prime < 1.0:
        return False
    s1 = 1.0 / gamma_margin - 1.0
    s2 = _H_SECOND_DERIVATIVE_BOUND / (n * (1.0 - beta) * gamma_margin**2)
    kappa = lambda_1 / lambda_0
    # Lemma 8.3's Hessian term is s2 kappa r^2/lambda_0; (14a) prints s2 r.
    shift = kappa * (s1 + s2 * r**2 / lambda_0)
    if not 1.0 - shift > 0.0:
        return False
    ratio = max(
        (1.0 + kappa * gamma / (1.0 - gamma)) / (1.0 - shift),
        1.0 + shift + (kappa * lambda_0 / R_reg) * gamma_prime / (1.0 - gamma_prime),
    )
    return ratio**2 - 1.0 <= 2.0 * epsilon / (1.0 + norm.ppf(1.0 - delta / 2.0) ** 2)


class LocalOutputPerturbation(BaseAlgorithm):
    r"""Estimate one coefficient of a robust regression with Gaussian noise scaled to how stable the fit is.

    An implementation of Algorithm 5 from Asi, Duchi and Talwar (2025). Fits a
    robust regression with a smooth loss of the residuals, privately certifies
    how curved that loss is near the fit, and outputs the coefficient of
    interest with Gaussian noise scaled to that certificate, or outputs nothing
    if a certificate fails. It is an approximate-DP mechanism.

    Requires:
        - A finite, strictly positive ``design_domain.l2_norm_bound`` ``B`` on
          complete design rows.
        - A finite ``ApproxDPBudget``.

    Parametrization notes:
        - Fixed robust-log loss $h(q) = \log(1+e^q) + \log(1+e^{-q})$: Examples
          1 and 3 fix its unit scale, with $G_0 = B$, $G_1 = B^2/2$ and
          $\|h''\|_\infty = 1/2$.
        - Fixed $ρ = 1/2$: Section 3 sets it.
        - Fixed $α = 1.2332$: Section 3 sets it for $φ(s) = e^s - 1$.
        - ``regularization``: The value 0 is the unregularized setting of
          Examples 1 and 3. Equation (1) permits any nonnegative value.
        - Fixed $p = 2$: Corollary 3.5 and (14a) set it.
        - Fixed $u = e_j$: It comes from
          ``metadata.coefficient_of_interest_index``.
        - Fixed per-stage budgets: Algorithms 2 and 4 and the Gaussian release
          each receive the child budget $(e, d)$ of the offered $(E, D)$, with
          $e = E/3$ and $d = D/(1 + \exp(e) + \exp(2e))$, following Corollary
          3.6.
        - Fixed solver root and first-order tolerances: They are operational
          controls.

    Implementation notes:
        - The ratio test's Hessian term is Lemma 8.3's ``s2 kappa r^2/lambda_0``,
          not the ``s2 kappa r`` printed in Corollary 3.5 and (14a).
        - Condition (C1) is tested on Algorithm 2's release before Algorithm 4
          runs, since Algorithm 4 needs it; Algorithm 5 prints the check last.
        - For adding or removing one record at the declared ``n``, ``t``, the
          local modulus and its remainder are half the paper's values for
          replacing a record.
        - Nothing is released when (C1) fails after Algorithm 2, when the ratio
          certificate fails after Algorithm 4, or when the local modulus is
          undefined after both certificates, which is Algorithm 2's failed
          confidence event.
        - A numerical root stands in for the exact minimizer and is accepted
          only below a fixed first-order residual. Solver or linear-algebra
          failure raises ``RuntimeError``.

    Reference:
        - asi2025privately: Algorithm 5 of arXiv v1.
    """

    TASK = SingleCoefficientEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.THEORETICAL
    )

    regularization: float = parameter_field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Ridge weight ``lambda_reg`` of equation (1), centred at the origin.",
    )

    @override
    def run(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> SingleCoefficientEstimationOutput:
        B = metadata.design_domain.l2_norm_bound
        if not 0.0 < B < math.inf:
            raise InvalidRegime(
                "LocalOutputPerturbation requires a finite, strictly positive "
                "design-row L2 bound."
            )
        n = metadata.n_samples
        j = metadata.coefficient_of_interest_index
        if budget.is_unconstrained:
            raise NotImplementedError(
                "LocalOutputPerturbation does not implement its zero-noise limit."
            )
        if not isinstance(budget, ApproxDPBudget):
            raise InvalidRegime(
                "LocalOutputPerturbation requires an ApproxDPBudget: its curvature "
                "certificates' failure events are absorbed into delta."
            )

        # Corollary 3.6: the three child releases compose to
        # (3e, (1 + exp(e) + exp(2e)) d) = (E, D).
        e = budget.eps / 3.0
        d = budget.delta / (1.0 + math.exp(e) + math.exp(2.0 * e))
        child = ApproxDPBudget(eps=e, delta=d)
        lower_seed, upper_seed, release_seed = split_seed(seed, 3)

        reg = self.regularization
        theta_hat, H = _solve_robust_log(input.X, input.y, n, reg)
        u = np.zeros(metadata.dim)
        u[j] = 1.0
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            H_inverse_u = np.linalg.solve(H + reg * np.eye(metadata.dim), u)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "LocalOutputPerturbation: the empirical Hessian's eigensolve or "
                "directional solve failed."
            ) from error
        lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[-1])

        lambda_hat_min = _algorithm_2(
            lambda_min, B, n, child, lower_seed, regularization=reg
        )
        lambda_0 = lambda_hat_min + reg
        # Post-processing of Algorithm 2's release; the refusal is accounted.
        if lambda_hat_min + reg / _RHO < _c1_threshold(B, n):
            raise NoRelease(
                "LocalOutputPerturbation released no estimate because condition "
                "C1 failed on the private curvature lower bound."
            )
        lambda_hat_max = _algorithm_4(lambda_max, lambda_0, B, n, child, upper_seed)
        lambda_1 = lambda_hat_max + reg
        # Post-processing of both curvature releases; the refusal is accounted.
        if not _passes_ratio_certificate(
            lambda_0, lambda_1, B, n, child, regularization=reg
        ):
            raise NoRelease(
                "LocalOutputPerturbation released no estimate because its private "
                "ratio certificate failed."
            )

        # Past both certificates an undefined modulus means Algorithm 2's
        # confidence event failed, a probability-delta event charged to delta.
        lambda_min_reg = lambda_min + reg
        if not lambda_min_reg >= 4.0 * _ALPHA * B**2 / n:
            raise NoRelease(
                "LocalOutputPerturbation released no estimate because Algorithm "
                "2's confidence event failed; the local modulus is undefined."
            )
        # Equation (11) under add/remove: the p=2 gradient set is the ball of
        # radius G0 = B, so (1/n) sup_g |u^T H^-1 g| = B ||H^-1 u||_2 / n.
        gamma = _ALPHA * B * _t_add(lambda_min_reg, B, n)
        if not gamma < 1.0:
            raise NoRelease(
                "LocalOutputPerturbation released no estimate because Algorithm "
                "2's confidence event failed; the local modulus is undefined."
            )
        Delta_add = B * float(np.linalg.norm(H_inverse_u)) / n
        omega = Delta_add + B / (n * lambda_min_reg) * gamma / (1.0 - gamma)
        sigma = accounting.calibrate_gaussian_scale(child, sensitivity=omega)
        noise = np.random.default_rng(release_seed).normal(scale=sigma)
        return SingleCoefficientEstimationOutput(
            coefficient=float(theta_hat[j] + noise)
        )
