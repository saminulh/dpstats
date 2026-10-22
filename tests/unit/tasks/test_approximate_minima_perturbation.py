# pyright: reportPrivateUsage=false
"""Tests for approximate minima perturbation on M-estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.algorithm import CompositeAlgorithm
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    LpBallDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D, ArrayND
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    AbsoluteResidualLoss,
    HuberResidualLoss,
    ToMEstimation,
    RobHytResidualLoss,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
)
from dpstats.tasks.m_estimation.algorithms.approximate_minima_perturbation import (
    AMP,
    _fixed_accelerated_gradient_steps,
    _run_fixed_accelerated_gradient,
)
from dpstats.tasks.m_estimation.algorithms._objective_perturbation_utils import (
    calibrate_rkw_amp,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss, Reduction
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order


def _problem(
    loss: MEstimationLoss, X: np.ndarray, y: np.ndarray, design_bound: float
) -> tuple[MEstimationMetadata, MEstimationInput]:
    metadata = MEstimationMetadata(
        n_samples=X.shape[0],
        parameter_domain=EuclideanSpace(dim=X.shape[1]),
        data_domain=ProductDomain(
            parts=(
                LpBallDomain.at_origin(p=2, radius=design_bound, dim=X.shape[1]),
                EuclideanSpace(dim=1),
            )
        ),
        loss=loss,
    )
    return metadata, MEstimationInput(data=np.column_stack((X, y)))


def _regression_problem() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    """A bounded design whose Huber minimizer is in the quadratic region."""
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(
        X=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.25]]),
        y=np.array([0.5, -0.25, 0.75, 0.1]),
    )
    return metadata, input


@pytest.mark.parametrize(("tolerance", "expected_steps"), [(1.0, 5), (5.0, 0)])
def test_fixed_work_bound_has_hand_computed_boundary(
    tolerance: float, expected_steps: int
) -> None:
    # M=4, lambda=2 gives A=24 and kappa=2. The squared-gradient bound
    # is 1.4185 after four updates and 0.6995 after five; A is below 5**2.
    assert (
        _fixed_accelerated_gradient_steps(
            n=2,
            gradient_bound=1.0,
            smoothness_bound=1.0,
            regularization=2.0,
            gradient_tolerance=tolerance,
        )
        == expected_steps
    )


@pytest.mark.parametrize(("num_steps", "expected"), [(0, -1.0), (1, -2.0 / 3.0)])
def test_fixed_work_returns_the_gradient_step_iterate(
    num_steps: int, expected: float
) -> None:
    # Here F(theta) = 3*theta**2/2 + 2*theta in the Huber quadratic region.
    # The initializer is -1; one step lands on the exact minimizer -2/3,
    # while the extrapolated iterate overshoots it.
    theta = _run_fixed_accelerated_gradient(
        HuberResidualLoss(threshold=2.0),
        np.array([[1.0, 0.0]]),
        regularization=2.0,
        perturbation=np.array([2.0]),
        smoothness_bound=1.0,
        n=1,
        num_steps=num_steps,
    )
    np.testing.assert_allclose(theta, [expected], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("loss", "smoothness"),
    [(HuberResidualLoss(threshold=1.0), 1.0), (RobHytResidualLoss(k=1.0), 2.0)],
)
@pytest.mark.parametrize(
    ("n", "multiplier", "tolerance", "seed"),
    [(4, 2.0, 0.01, 0), (128, 1.5, 1e-6, 7), (1024, 5.0, 1e-8, 2)],
)
def test_fixed_work_reaches_the_stated_gradient_bound(
    loss: MEstimationLoss,
    smoothness: float,
    n: int,
    multiplier: float,
    tolerance: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X /= np.maximum(1.0, np.linalg.norm(X, axis=1))[:, None]
    y = rng.normal(size=n) * 3.0
    perturbation = rng.normal(size=3) * 7.0
    regularization = multiplier * smoothness
    num_steps = _fixed_accelerated_gradient_steps(
        n, 1.0, smoothness, regularization, tolerance
    )
    theta = _run_fixed_accelerated_gradient(
        loss,
        np.column_stack((X, y)),
        regularization,
        perturbation,
        smoothness,
        n,
        num_steps,
    )

    residuals = y - X @ theta
    if isinstance(loss, HuberResidualLoss):
        scores = np.clip(residuals, -1.0, 1.0)
    else:
        scores = np.tanh(2.0 * residuals)
    gradient = -X.T @ scores + regularization * theta + perturbation
    assert np.linalg.norm(gradient) <= tolerance


def test_calibration_scales_with_the_gradient_and_smoothness_bounds() -> None:
    # sigma = f*C/sqrt(2*rho) is linear in the gradient bound C, and the
    # objective's Renyi curve depends on C only through C/sigma, so the ridge
    # does not move. The ridge enters the privacy cost only through the
    # determinant term -log(1 - beta/lambda), so it is linear in beta.
    budget = ApproxDPBudget(eps=0.99, delta=9.9e-6)
    ridge, sigma = calibrate_rkw_amp(budget, 1.0, 2.0, 1.3)

    wider_ridge, wider_sigma = calibrate_rkw_amp(budget, 3.0, 2.0, 1.3)
    rougher_ridge, rougher_sigma = calibrate_rkw_amp(budget, 1.0, 6.0, 1.3)

    assert wider_sigma == pytest.approx(3.0 * sigma)
    assert wider_ridge == pytest.approx(ridge)
    assert rougher_ridge == pytest.approx(3.0 * ridge)
    assert rougher_sigma == pytest.approx(sigma)


def test_objective_noise_is_the_configured_multiple_of_the_zcdp_gaussian() -> None:
    # A Gaussian of sensitivity C and deviation s is C**2/(2*s**2)-zCDP, so the
    # C-sensitive Gaussian at the child's zCDP spend rho has deviation
    # C/sqrt(2*rho); the objective noise is objective_noise_ratio times it.
    budget = ApproxDPBudget(eps=0.99, delta=9.9e-6)
    rho = accounting.spend_as_zcdp_via_rdp(budget).rho

    _, sigma = calibrate_rkw_amp(budget, 2.0, 2.0, 1.3)

    assert sigma == pytest.approx(1.3 * 2.0 / math.sqrt(2.0 * rho))


def test_ridge_exceeds_smoothness_and_falls_with_more_noise_or_budget() -> None:
    smoothness = 0.5
    budget = ApproxDPBudget(eps=0.99, delta=9.9e-6)
    ridges = [
        calibrate_rkw_amp(budget, 1.0, smoothness, ratio)[0]
        for ratio in (1.1, 1.3, 1.5)
    ]
    generous, _ = calibrate_rkw_amp(
        ApproxDPBudget(eps=30.0, delta=9.9e-6), 1.0, smoothness, 1.3
    )

    # More objective noise leaves more of epsilon for the determinant cost.
    assert smoothness < ridges[2] < ridges[1] < ridges[0]
    # A generous budget leaves the determinant cost almost free, so the
    # ridge approaches its lower limit, the smoothness bound.
    assert smoothness < generous < 1.01 * smoothness


@pytest.mark.parametrize(
    ("tolerance", "share"), [(0.01, 0.01), (0.1, 0.01), (0.1, 0.1)]
)
def test_split_budgets_and_both_noise_draws_reach_the_release(
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
    tolerance: float,
    share: float,
) -> None:
    fake_randomness.script(
        "normal", draws_in_order([np.array([2.0, -1.0]), np.array([-3.0, 4.0])])
    )
    metadata, input = _problem(
        RobHytResidualLoss(k=1.0), np.zeros((3, 2)), np.zeros(3), design_bound=1.0
    )
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    output = AMP(gradient_tolerance=tolerance, output_budget_fraction=share).run(
        metadata, input, budget, seed=1
    )

    objective_budget = ApproxDPBudget(eps=1.0 - share, delta=(1.0 - share) * 1e-5)
    output_budget = ApproxDPBudget(eps=share, delta=share * 1e-5)
    assert len(accounting_spy.calls) == 2
    objective_call, output_call = accounting_spy.calls
    assert objective_call.entry_point == "spend_as_zcdp_via_rdp"
    assert_budget_close(objective_call.budget, objective_budget)
    assert output_call.entry_point == "calibrate_gaussian_scale"
    assert_budget_close(output_call.budget, output_budget)
    assert fake_randomness.seeds == (1,)

    # The objective child calibrates at the loss's declared constants; its
    # ridge sets the output sensitivity 2*tolerance/lambda.
    properties = metadata.loss_properties
    ridge, objective_scale = calibrate_rkw_amp(
        objective_budget,
        properties.loss_lipschitz_constant,
        properties.gradient_lipschitz_constant,
        1.3,
    )
    assert output_call.sensitivity == pytest.approx(2.0 * tolerance / ridge)
    output_scale = accounting.calibrate_gaussian_scale(
        output_budget, 2.0 * tolerance / ridge
    )
    # A zero design leaves the minimizer -b/lambda, then the output draw is added.
    expected = -objective_scale / ridge * np.array([2.0, -1.0])
    expected += output_scale * np.array([-3.0, 4.0])
    np.testing.assert_allclose(output.theta, expected, rtol=1e-7)


def test_unsatisfied_calibration_is_numerical_failure() -> None:
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0), np.ones((2, 1)), np.ones(2), design_bound=1.0
    )
    with pytest.raises(RuntimeError, match="could not find a feasible Renyi order"):
        AMP(objective_noise_ratio=0.5).run(
            metadata, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )


@pytest.mark.parametrize(
    ("threshold", "tolerance"), [(1.0, 0.01), (2.0, 0.025), (1.0, 1e-6)]
)
def test_half_huber_composition_is_within_its_certified_radius(
    threshold: float, tolerance: float
) -> None:
    metadata, input = _regression_problem()
    composed = CompositeAlgorithm(
        transformations=(ToMEstimation(loss=HuberResidualLoss(threshold=threshold)),),
        algorithm=AMP(gradient_tolerance=tolerance),
    )
    output = composed.run(metadata, input, NO_PRIVACY, seed=7)

    # R=sqrt(2), beta=2, and the unconstrained ridge is beta. The Huber
    # minimizer is the closed-form ridge solution; no sampled noise.
    regularization = 2.0
    expected = np.linalg.solve(
        input.X.T @ input.X + regularization * np.eye(2), input.X.T @ input.y
    )
    assert np.max(np.abs(input.y - input.X @ expected)) < threshold
    assert np.linalg.norm(output.beta - expected) <= tolerance / regularization


def test_high_accuracy_run_has_bounded_fixed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    gradient = HuberResidualLoss.gradient

    def counted_gradient(
        self: HuberResidualLoss,
        data: Array2D,
        theta: Array1D,
        *,
        reduction: Reduction = "none",
    ) -> ArrayND:
        nonlocal calls
        calls += 1
        assert calls < 10000, "The high-accuracy solve exceeded its work ceiling."
        return gradient(self, data, theta, reduction=reduction)

    monkeypatch.setattr(HuberResidualLoss, "gradient", counted_gradient)
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0),
        np.zeros((20364, 1)),
        np.zeros(20364),
        design_bound=1.0,
    )
    output = AMP(
        gradient_tolerance=1e-6,
    ).run(metadata, input, NO_PRIVACY, seed=0)

    # The initial gradient is already zero, but this realization never exits
    # early. Count work rather than impose a machine-dependent timing threshold.
    assert 1 < calls < 10000
    np.testing.assert_allclose(output.theta, [0.0], atol=1e-12)


def test_the_configured_loss_reaches_the_one_mechanism() -> None:
    """RobHyt and half-Huber run the same class and release different estimates."""
    metadata, input = _regression_problem()
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)

    def compose(loss: MEstimationLoss):
        return CompositeAlgorithm(
            transformations=(ToMEstimation(loss=loss),),
            algorithm=AMP(),
        ).run(metadata, input, budget, seed=2)

    robhyt = compose(RobHytResidualLoss(k=1.0))
    half_huber = compose(HuberResidualLoss(threshold=1.0))

    assert np.all(np.isfinite(robhyt.beta))
    assert np.all(np.isfinite(half_huber.beta))
    # The two losses publish different constants, so one mechanism under the
    # same seed and budget must still separate them.
    assert not np.allclose(robhyt.beta, half_huber.beta, rtol=1e-6, atol=1e-6)


def test_public_preconditions() -> None:
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0), np.ones((2, 1)), np.ones(2), design_bound=1.0
    )
    for budget in (PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0), GDPBudget(mu=1.0)):
        with pytest.raises(InvalidRegime, match="ApproxDPBudget"):
            AMP().run(metadata, input, budget, seed=0)

    constrained = metadata.model_copy(
        update={"parameter_domain": LpBallDomain.at_origin(p=2, radius=1.0, dim=1)}
    )
    with pytest.raises(InvalidRegime, match="outside its objective profile"):
        AMP().run(constrained, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0)

    kinked, kinked_input = _problem(
        AbsoluteResidualLoss(), np.ones((2, 1)), np.ones(2), design_bound=1.0
    )
    with pytest.raises(InvalidRegime, match="finite gradient and smoothness"):
        AMP().run(kinked, kinked_input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0)


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-5),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
def test_unconstrained_budget_releases_the_noiseless_ridge_minimizer(
    budget: DPBudget,
) -> None:
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0), X, 2.0 * X[:, 0], design_bound=1.0
    )

    output = AMP().run(metadata, input, budget, seed=8)
    reference = AMP().run(metadata, input, NO_PRIVACY, seed=9)

    # Every unconstrained family takes the zero-noise route, so the seed and
    # the family leave the release unchanged.
    np.testing.assert_allclose(output.theta, reference.theta, rtol=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_unconstrained_budget_without_a_limit_refuses(ratio: float) -> None:
    # At objective_noise_ratio <= 1 the calibration fails at every epsilon.
    metadata, input = _problem(
        HuberResidualLoss(threshold=1.0), np.ones((2, 1)), np.ones(2), design_bound=1.0
    )
    with pytest.raises(InvalidRegime, match="no unconstrained limit"):
        AMP(objective_noise_ratio=ratio).run(metadata, input, NO_PRIVACY, seed=0)
