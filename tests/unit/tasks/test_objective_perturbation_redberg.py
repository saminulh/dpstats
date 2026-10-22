# pyright: reportPrivateUsage=false
"""Tests for RKW objective perturbation on configured M-estimation losses."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
from scipy.stats import norm

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    Domain,
    EuclideanSpace,
    LpBallDomain,
    ProductDomain,
)
from dpstats.task import Array1D, ArrayND
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    AbsoluteResidualLoss,
    HuberResidualLoss,
    RobHytResidualLoss,
)
from dpstats.tasks.m_estimation import MEstimationInput, MEstimationMetadata
from dpstats.tasks.m_estimation.algorithms import (
    _objective_perturbation_utils,
    objective_perturbation_redberg,
)
from dpstats.tasks.m_estimation.algorithms._objective_perturbation_utils import (
    _rkw_objective_perturbation_delta,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_redberg import (
    ObjectivePerturbationRedberg,
)
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
)
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order
from tests.unit.tasks.task_builders import box, m_estimation_metadata


def _problem(
    loss: MEstimationLoss, X: np.ndarray, y: np.ndarray, *, domain: Domain | None = None
) -> tuple[MEstimationMetadata, MEstimationInput]:
    dim = X.shape[1]
    return (
        m_estimation_metadata(
            X.shape[0],
            loss,
            parameter=domain if domain is not None else dim,
            data=ProductDomain(
                parts=(box(*[(-1.0, 1.0)] * dim), EuclideanSpace(dim=1))
            ),
        ),
        MEstimationInput(data=np.column_stack((X, y))),
    )


def _profile_delta(
    scale: float, eps: float, gradient_bound: float, multiplier: float
) -> float:
    """RKW's corrected Appendix-D objective privacy profile, derived here.

    The regularizer costs ``|log(1 - 1/multiplier)|`` of epsilon. The Gaussian
    objective noise then has privacy-loss mean ``m = C**2 / (2 sigma**2)``,
    and its profile at ``e`` is the Balle-Wang (2018, Thm. 8) closed form
    ``Phi(C/(2 sigma) - e sigma/C) - exp(e) Phi(-C/(2 sigma) - e sigma/C)``,
    doubled for the two-sided objective. When the remaining epsilon ``r``
    exceeds ``m`` the profile is that doubled value at ``r``; otherwise the
    event of loss above ``r`` is charged as ``1 - exp(r - m)`` plus the
    doubled profile at ``m`` weighted by ``exp(r - m)``.
    """

    def gaussian(e: float) -> float:
        a = gradient_bound / (2.0 * scale)
        b = e * scale / gradient_bound
        return float(norm.cdf(a - b) - math.exp(e) * norm.cdf(-a - b))

    remaining = eps - abs(math.log1p(-1.0 / multiplier))
    mean = gradient_bound**2 / (2.0 * scale**2)
    if remaining >= mean:
        return 2.0 * gaussian(remaining)
    return (
        1.0
        - math.exp(remaining - mean)
        + math.exp(remaining - mean) * 2.0 * (gaussian(mean))
    )


class _NonconvexLoss(MEstimationLoss):
    """Loss double outside RKW's convexity hypothesis."""

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        del theta
        return np.zeros(len(data)) if reduction == "none" else 0.0

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        return (
            np.zeros((len(data), len(theta)))
            if reduction == "none"
            else np.zeros_like(theta)
        )

    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        del data_domain, parameter_domain
        return LossProperties(
            convex=False,
            generalized_linear=True,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=1.0,
            gradient_lipschitz_constant=1.0,
        )


def test_full_budget_calibrates_one_objective_draw(
    monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
) -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.zeros((3, 2)), np.zeros(3))
    seen: list[object] = []

    def calibrate(*args: object) -> tuple[float, float]:
        seen.extend(args)
        return 10.0, 6.0

    monkeypatch.setattr(
        objective_perturbation_redberg, "calibrate_rkw_objective_ridge", calibrate
    )
    fake_randomness.script("normal", draws_in_order([np.array([1.0, 0.0])]))
    budget = ApproxDPBudget(eps=2.0, delta=1e-5)

    output = ObjectivePerturbationRedberg().run(metadata, input, budget, seed=5)

    assert seen[0] is budget
    assert seen[3] == 1.3
    assert fake_randomness.parameters_of("normal", "size") == [2]
    np.testing.assert_allclose(output.theta, [-0.6, 0.0], atol=1e-10)


@pytest.mark.parametrize(
    "budget", [PureDPBudget(eps=2.0), ZCDPBudget(rho=0.5), GDPBudget(mu=1.0)]
)
def test_finite_non_approximate_budgets_are_refused(budget: object) -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.eye(2), np.zeros(2))
    with pytest.raises(InvalidRegime, match="Cannot spend"):
        ObjectivePerturbationRedberg().run(metadata, input, budget, seed=0)  # type: ignore[arg-type]


def test_unconstrained_budget_has_zero_objective_noise() -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.zeros((2, 1)), np.zeros(2))
    output = ObjectivePerturbationRedberg().run(metadata, input, NO_PRIVACY, seed=0)
    np.testing.assert_allclose(output.theta, [0.0], atol=1e-12)


def test_euclidean_root_uses_robhyt_hessian_plus_rkw_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.eye(2), np.zeros(2))

    def root_with_hessian(gradient: Any, **kwargs: Any) -> SimpleNamespace:
        del gradient
        np.testing.assert_allclose(kwargs["jac"](np.zeros(2)), 6.0 * np.eye(2))
        return SimpleNamespace(x=np.zeros(2))

    monkeypatch.setattr(_objective_perturbation_utils, "root", root_with_hessian)

    output = ObjectivePerturbationRedberg().run(metadata, input, NO_PRIVACY, seed=0)

    np.testing.assert_allclose(output.theta, [0.0, 0.0])


def test_c1_huber_loss_is_admitted() -> None:
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0), np.eye(2), np.array([0.5, -0.25])
    )
    output = ObjectivePerturbationRedberg().run(
        metadata, input, ApproxDPBudget(eps=3.0, delta=1e-5), seed=1
    )
    assert np.all(np.isfinite(output.theta))


@pytest.mark.parametrize("loss", [_NonconvexLoss(), AbsoluteResidualLoss()])
def test_loss_outside_shared_admission_is_refused(loss: MEstimationLoss) -> None:
    metadata, input = _problem(loss, np.eye(2), np.zeros(2))
    with pytest.raises(InvalidRegime):
        ObjectivePerturbationRedberg().run(
            metadata, input, ApproxDPBudget(eps=2.0, delta=1e-5), seed=0
        )


def test_non_euclidean_parameter_domain_is_refused() -> None:
    metadata, input = _problem(
        RobHytResidualLoss(k=1.0),
        np.eye(2),
        np.zeros(2),
        domain=LpBallDomain.at_origin(p=2.0, radius=1.0, dim=2),
    )
    with pytest.raises(InvalidRegime, match="parameter"):
        ObjectivePerturbationRedberg().run(
            metadata, input, ApproxDPBudget(eps=2.0, delta=1e-5), seed=0
        )


def test_numerical_failure_is_an_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.eye(2), np.zeros(2))

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(_objective_perturbation_utils, "root", fail)
    with pytest.raises(RuntimeError, match="numerical") as caught:
        ObjectivePerturbationRedberg().run(
            metadata, input, ApproxDPBudget(eps=2.0, delta=1e-5), seed=0
        )
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)


def test_final_gradient_failure_is_an_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input = _problem(RobHytResidualLoss(k=1.0), np.eye(2), np.zeros(2))

    def fail_gradient(*args: Any, **kwargs: Any) -> NoReturn:
        raise FloatingPointError("injected")

    monkeypatch.setattr(RobHytResidualLoss, "gradient", fail_gradient)
    monkeypatch.setattr(
        _objective_perturbation_utils,
        "root",
        lambda *args, **kwargs: SimpleNamespace(x=np.zeros(2)),
    )

    with pytest.raises(RuntimeError, match="numerical") as caught:
        ObjectivePerturbationRedberg().run(
            metadata, input, ApproxDPBudget(eps=2.0, delta=1e-5), seed=0
        )
    assert isinstance(caught.value.__cause__, FloatingPointError)


@pytest.mark.parametrize(
    ("scale", "eps", "gradient_bound", "regularization_multiplier"),
    [
        # Remaining epsilon 1.48 exceeds the loss mean 0.06.
        (2.3, 1.7, 0.8, 5.0),
        # Remaining epsilon 0.28 is below the loss mean 3.125.
        (0.4, 0.5, 1.0, 5.0),
    ],
)
def test_corrected_profile_matches_appendix_d(
    scale: float,
    eps: float,
    gradient_bound: float,
    regularization_multiplier: float,
) -> None:
    assert _rkw_objective_perturbation_delta(
        scale,
        eps,
        gradient_bound,
        regularization_multiplier,
    ) == pytest.approx(
        _profile_delta(scale, eps, gradient_bound, regularization_multiplier),
        rel=1e-9,
    )


@pytest.mark.parametrize(
    ("eps", "lower", "upper"),
    [
        # RobHyt k=1 on an R=1 design domain: C = R = 1, beta = 2 R**2 = 2.
        # n = 3000, delta = n**-1.1. Ranges from the paper's rule reviewed
        # against the RobHyt profile.
        (1.0, 5.2, 5.6),
        (3.0, 2.02, 2.06),
        (10.0, 1.084, 1.094),
        (30.0, 1.0001, 1.0003),
    ],
)
def test_resolver_matches_reviewed_robhyt_profile(
    eps: float, lower: float, upper: float
) -> None:
    budget = ApproxDPBudget(eps=eps, delta=3000.0**-1.1)
    lam, _ = _objective_perturbation_utils.calibrate_rkw_objective_ridge(
        budget, gradient_bound=1.0, smoothness_bound=2.0, objective_noise_ratio=1.3
    )
    assert lower <= lam / 2.0 <= upper


def test_ridge_rule_is_continuous_at_the_unconstrained_limit() -> None:
    """a(eps) -> 1 and sigma(eps) -> 0 as eps grows, matching the exact limit."""
    multipliers: list[float] = []
    sigmas: list[float] = []
    for eps in (1e2, 1e3, 1e4):
        budget = ApproxDPBudget(eps=eps, delta=1e-6)
        lam, sigma = _objective_perturbation_utils.calibrate_rkw_objective_ridge(
            budget, gradient_bound=1.0, smoothness_bound=2.0, objective_noise_ratio=1.3
        )
        multipliers.append(lam / 2.0)
        sigmas.append(sigma)

    # sigma has headroom above the bisection's floor and is strictly
    # decreasing; a is already within a few machine epsilons of 1 by
    # eps=1e3, so consecutive values can tie there -- non-increasing, each
    # above 1, and close to the eps=inf limit at the top of the range.
    assert sigmas[0] > sigmas[1] > sigmas[2] > 0.0
    assert multipliers[0] >= multipliers[1] >= multipliers[2] > 1.0
    assert multipliers[-1] == pytest.approx(1.0, abs=1e-9)

    limit_lambda, limit_sigma = (
        _objective_perturbation_utils.calibrate_rkw_objective_ridge(
            NO_PRIVACY,
            gradient_bound=1.0,
            smoothness_bound=2.0,
            objective_noise_ratio=1.3,
        )
    )
    assert (limit_lambda, limit_sigma) == (2.0, 0.0)


def test_larger_objective_noise_ratio_gives_smaller_ridge_multiplier() -> None:
    budget = ApproxDPBudget(eps=3.0, delta=3000.0**-1.1)
    multipliers = [
        _objective_perturbation_utils.calibrate_rkw_objective_ridge(
            budget, 1.0, 2.0, f
        )[0]
        / 2.0
        for f in (1.1, 1.3, 2.0, 5.0)
    ]
    assert multipliers[0] > multipliers[1] > multipliers[2] > multipliers[3]


def test_objective_noise_ratio_enters_identity() -> None:
    assert (
        ObjectivePerturbationRedberg(objective_noise_ratio=1.3).uuid
        != ObjectivePerturbationRedberg(objective_noise_ratio=2.0).uuid
    )


def test_infeasible_ridge_rule_is_publicly_refused() -> None:
    # f = 1 pins sigma exactly at sigma_G: the determinant cost that any
    # finite ridge spends means delta never reaches budget.delta, at any eps.
    with pytest.raises(InvalidRegime, match="infeasible"):
        _objective_perturbation_utils.calibrate_rkw_objective_ridge(
            ApproxDPBudget(eps=3.0, delta=3000.0**-1.1), 1.0, 2.0, 1.0
        )


def test_ridge_rule_at_a_five_matches_old_fixed_multiplier_output() -> None:
    """Pin ObjectivePerturbationRedberg(regularization_multiplier=5.0)'s released theta.

    ``objective_noise_ratio`` and ``old_theta`` below were both computed once,
    before the fixed-ridge field and its dedicated sigma-at-fixed-ridge
    primitive were removed: the ratio is the sigma that calibration required
    at multiplier 5 divided by the exact Balle-Wang Gaussian's sigma at the
    same (eps, delta), and ``old_theta`` is the released theta that followed
    from it. The rule's own bisection lands on the same multiplier to within
    its numerical floor (about 1e-9 relative in the multiplier), so the
    theta comparison below is float-tolerant rather than a literal equality.
    """
    X = np.array([[0.6, -0.3], [0.1, 0.9], [-0.4, 0.2], [0.3, -0.7]])
    y = np.array([0.2, -0.5, 0.4, 0.1])
    metadata, input = _problem(RobHytResidualLoss(k=1.0), X, y)
    budget = ApproxDPBudget(eps=1.0, delta=3000.0**-1.1)
    seed = 7

    properties = metadata.loss_properties
    gradient_bound = properties.loss_lipschitz_constant
    smoothness_bound = properties.gradient_lipschitz_constant
    objective_noise_ratio = 1.3224053827709534

    resolved_lambda, _ = _objective_perturbation_utils.calibrate_rkw_objective_ridge(
        budget, gradient_bound, smoothness_bound, objective_noise_ratio
    )
    assert resolved_lambda / smoothness_bound == pytest.approx(5.0, abs=1e-9)

    old_theta = np.array([-0.006931521123309748, -0.11533706570874636])
    output = ObjectivePerturbationRedberg(
        objective_noise_ratio=objective_noise_ratio
    ).run(metadata, input, budget, seed)
    np.testing.assert_allclose(output.theta, old_theta, rtol=1e-9, atol=0.0)
