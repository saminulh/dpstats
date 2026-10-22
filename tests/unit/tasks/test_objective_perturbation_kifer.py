"""Public behavior of native KST objective perturbation."""

from __future__ import annotations

import math
import tracemalloc
from types import SimpleNamespace
from typing import Any, Literal, NoReturn

import numpy as np
import pytest

from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    LpBallDomain,
    ProductDomain,
)
from dpstats.task import Array1D, ArrayND
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations import ToMEstimation
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    HuberResidualLoss,
    RobHytResidualLoss,
    SquaredResidualLoss,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_kifer import (
    ObjectivePerturbationKifer,
)
from dpstats.tasks.m_estimation.algorithms import (
    _objective_perturbation_utils,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationLoss,
    MEstimationMetadata,
)
from dpstats.tasks.m_estimation.losses import LossProperties
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order


def _regression_problem() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        response_domain=EuclideanSpace(dim=1),
    )
    input_data = LinearRegressionInput(
        X=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.25]]),
        y=np.array([0.5, -0.25, 0.75, 0.1]),
    )
    return metadata, input_data


def _robhyt_composition(k: float = 1.0) -> CompositeAlgorithm:
    return CompositeAlgorithm(
        transformations=(ToMEstimation(loss=RobHytResidualLoss(k=k)),),
        algorithm=ObjectivePerturbationKifer(),
    )


class _NoHessianRobHytLoss(MEstimationLoss):
    """Test loss preserving RobHyt's public facts without the optional derivative."""

    def __call__(
        self,
        data: ArrayND,
        theta: Array1D,
        *,
        reduction: Literal["none", "sum", "mean"] = "none",
    ) -> float | Array1D:
        return RobHytResidualLoss(k=1.0)(data, theta, reduction=reduction)

    def gradient(
        self,
        data: ArrayND,
        theta: Array1D,
        *,
        reduction: Literal["none", "sum", "mean"] = "none",
    ) -> ArrayND:
        return RobHytResidualLoss(k=1.0).gradient(data, theta, reduction=reduction)

    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        return RobHytResidualLoss(k=1.0).properties(data_domain, parameter_domain)


def test_large_squared_loss_composition_bounds_temporary_memory(
    fake_randomness: FakeRandomness,
) -> None:
    """Keep the stacked records, but not another full per-record gradient array."""
    n, dim = 32769, 32
    metadata = LinearRegressionMetadata(
        n_samples=n,
        design_domain=BoxDomain(bounds=np.tile([-1.0, 1.0], (dim, 1))),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    input_data = LinearRegressionInput(X=np.ones((n, dim)), y=np.ones(n))
    input_data.y[: n // 2] = 0.0
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(parameter_radius=1.0),),
        algorithm=ObjectivePerturbationKifer(),
    )
    fake_randomness.script("normal", draws_in_order([np.ones(dim)]))
    fake_randomness.script("gamma", draws_in_order([0.0]))

    tracemalloc.start()
    try:
        output = algorithm.run(metadata, input_data, PureDPBudget(eps=2.0), seed=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Sum-form squared loss plus KST ridge has this interior minimizer.
    np.testing.assert_allclose(
        output.beta, np.full(dim, input_data.y.sum() / (dim * (n + 1)))
    )
    assert peak < 1.5 * (input_data.X.nbytes + input_data.y.nbytes)


@pytest.mark.parametrize(
    "budget",
    [PureDPBudget(eps=2.0), ApproxDPBudget(eps=2.0, delta=1e-5)],
    ids=["pure", "approximate"],
)
def test_robhyt_composition_runs_each_native_kst_law(budget: DPBudget) -> None:
    metadata, input_data = _regression_problem()

    output = _robhyt_composition().run(metadata, input_data, budget, seed=7)

    assert np.all(np.isfinite(output.beta))


def test_robhyt_pure_composition_matches_the_pme_source_objective(
    fake_randomness: FakeRandomness,
) -> None:
    """With a null design, KST and PME both minimize ``Delta theta + b = 0``."""
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        response_domain=EuclideanSpace(dim=1),
    )
    input_data = LinearRegressionInput(X=np.zeros((3, 2)), y=np.array([2.0, -1.0, 4.0]))
    fake_randomness.script("normal", draws_in_order([np.array([3.0, 4.0])]))
    fake_randomness.script("gamma", draws_in_order([5.0]))

    output = _robhyt_composition(k=0.8).run(
        metadata, input_data, PureDPBudget(eps=2.0), seed=5
    )

    assert fake_randomness.parameters_of("normal", "size") == [2]
    assert fake_randomness.parameters_of("gamma", "shape") == [2]
    assert fake_randomness.parameters_of("gamma", "scale") == pytest.approx(
        [0.8 * math.sqrt(2.0)]
    )
    np.testing.assert_allclose(output.beta, [-0.75, -1.0], atol=1e-10)


def test_euclidean_root_uses_robhyt_hessian_plus_kst_ridge(
    monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    input_data = LinearRegressionInput(X=np.eye(2), y=np.zeros(2))
    fake_randomness.script("normal", draws_in_order([np.array([1.0, 0.0])]))
    fake_randomness.script("gamma", draws_in_order([0.0]))

    def root_with_hessian(gradient: Any, **kwargs: Any) -> SimpleNamespace:
        del gradient
        np.testing.assert_allclose(kwargs["jac"](np.zeros(2)), 4.0 * np.eye(2))
        return SimpleNamespace(x=np.zeros(2))

    monkeypatch.setattr(_objective_perturbation_utils, "root", root_with_hessian)

    output = _robhyt_composition().run(
        metadata, input_data, PureDPBudget(eps=2.0), seed=0
    )

    np.testing.assert_allclose(output.beta, [0.0, 0.0])


def test_a_loss_without_a_hessian_reaches_the_same_root() -> None:
    """The optional loss Hessian only speeds the solve; the release is unchanged."""
    metadata, input_data = _regression_problem()
    budget = PureDPBudget(eps=2.0)

    with_hessian = _robhyt_composition().run(metadata, input_data, budget, seed=7)
    without_hessian = CompositeAlgorithm(
        transformations=(ToMEstimation(loss=_NoHessianRobHytLoss()),),
        algorithm=ObjectivePerturbationKifer(),
    ).run(metadata, input_data, budget, seed=7)

    np.testing.assert_allclose(without_hessian.beta, with_hessian.beta, atol=1e-8)


@pytest.mark.parametrize("error_type", [FloatingPointError, np.linalg.LinAlgError])
def test_euclidean_root_reports_an_initial_hessian_numerical_failure(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """KST translates loss-curvature numerical failures into its public error."""
    metadata, input_data = _regression_problem()

    def failing_hessian(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise error_type("test initial Hessian failure")

    monkeypatch.setattr(RobHytResidualLoss, "hessian", failing_hessian)

    with pytest.raises(
        RuntimeError, match="ObjectivePerturbationKifer: the numerical solve"
    ):
        _robhyt_composition().run(metadata, input_data, PureDPBudget(eps=2.0), seed=0)


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-5),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
    ids=[
        "no-privacy",
        "pure-infinite",
        "approximate-infinite",
        "zcdp-infinite",
        "gdp-infinite",
    ],
)
def test_refuses_every_unconstrained_budget_offer(budget: DPBudget) -> None:
    metadata, input_data = _regression_problem()

    with pytest.raises(InvalidRegime, match="finite privacy budget"):
        _robhyt_composition().run(metadata, input_data, budget, seed=0)


def test_robhyt_runs_on_a_ball() -> None:
    metadata, input_data = _regression_problem()
    prepared_metadata, prepared_input, _, _, _ = ToMEstimation(
        loss=RobHytResidualLoss(k=1.0), parameter_radius=1.0
    ).prepare(metadata, input_data, PureDPBudget(eps=1.0), seed=0)

    output = ObjectivePerturbationKifer().run(
        prepared_metadata, prepared_input, PureDPBudget(eps=1.0), seed=0
    )
    assert np.linalg.norm(output.theta) <= 1.0 + 1e-7


@pytest.mark.parametrize(
    ("response", "expected"),
    [([0.2, 0.5], [0.1, 0.2]), ([2.7, 3.0], [0.6, 0.8])],
    ids=["interior", "boundary"],
)
def test_squared_loss_ball_minimizer(
    response: list[float], expected: list[float], fake_randomness: FakeRandomness
) -> None:
    """H=diag(4,10); the boundary fixture has multiplier 5 at (0.6,0.8)."""
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain.at_origin(p=2, radius=2.0, dim=2),
        response_domain=BoxDomain(bounds=np.array([[-3.0, 3.0]])),
    )
    input_data = LinearRegressionInput(X=np.diag([1.0, 2.0]), y=np.array(response))
    fake_randomness.script("normal", draws_in_order([np.array([1.0, 0.0])]))
    fake_randomness.script("gamma", draws_in_order([0.0]))
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(parameter_radius=1.0),),
        algorithm=ObjectivePerturbationKifer(),
    )
    output = algorithm.run(metadata, input_data, PureDPBudget(eps=8.0), seed=0)
    np.testing.assert_allclose(output.beta, expected, atol=1e-6)


@pytest.mark.parametrize("dimension", [2, 5])
def test_dimension_radius_composition_round_trips(dimension: int) -> None:
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(parameter_radius="sqrt_dim"),),
        algorithm=ObjectivePerturbationKifer(),
    )
    restored = BaseAlgorithm.load_record(algorithm.dump_config())
    assert restored == algorithm
    metadata = LinearRegressionMetadata(
        n_samples=1,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=dimension),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    input_data = LinearRegressionInput(X=np.zeros((1, dimension)), y=np.zeros(1))
    prepared, _, _, _, _ = algorithm.transformations[0].prepare(
        metadata, input_data, PureDPBudget(eps=1.0), seed=0
    )
    assert prepared.parameter_domain.radius == pytest.approx(math.sqrt(dimension))
    assert (
        np.linalg.norm(
            restored.run(metadata, input_data, PureDPBudget(eps=1.0), seed=0).beta
        )
        <= math.sqrt(dimension) + 1e-6
    )


@pytest.mark.parametrize("approximate", [False, True], ids=["radial", "Gaussian"])
def test_ball_release_uses_the_source_noise_scale(
    approximate: bool, fake_randomness: FakeRandomness
) -> None:
    """With null X and Delta=2, b=(.24,.32) gives theta=(-.12,-.16)."""
    metadata = LinearRegressionMetadata(
        n_samples=1,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    budget: DPBudget
    if approximate:
        budget = ApproxDPBudget(eps=2.0, delta=2.0 / math.e)
        fake_randomness.script("normal", draws_in_order([np.array([0.03, 0.04])]))
    else:
        budget = PureDPBudget(eps=2.0)
        fake_randomness.script("normal", draws_in_order([np.array([3.0, 4.0])]))
        fake_randomness.script("gamma", draws_in_order([0.4]))
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(parameter_radius=1.0),),
        algorithm=ObjectivePerturbationKifer(),
    )
    output = algorithm.run(
        metadata, LinearRegressionInput(X=np.zeros((1, 2)), y=np.zeros(1)), budget, 0
    )
    np.testing.assert_allclose(output.beta, [-0.12, -0.16], atol=1e-8)
    if not approximate:
        assert fake_randomness.parameters_of("gamma", "scale") == pytest.approx([4.0])


def test_shifted_ball_keeps_the_ridge_centered_at_zero(
    fake_randomness: FakeRandomness,
) -> None:
    metadata, input_data = _regression_problem()
    prepared, rows, _, _, _ = ToMEstimation(loss=RobHytResidualLoss(k=1.0)).prepare(
        metadata, input_data, PureDPBudget(eps=2.0), 0
    )
    shifted = MEstimationMetadata(
        n_samples=prepared.n_samples,
        data_domain=prepared.data_domain,
        parameter_domain=LpBallDomain(p=2, radius=1.0, center=np.array([2.0, 0.0])),
        loss=prepared.loss,
    )
    rows.data[:, :2] = 0
    fake_randomness.script("normal", draws_in_order([np.array([1.0, 0.0])]))
    fake_randomness.script("gamma", draws_in_order([0.0]))
    output = ObjectivePerturbationKifer().run(shifted, rows, PureDPBudget(eps=2.0), 0)
    np.testing.assert_allclose(output.theta, [1.0, 0.0], atol=1e-7)


@pytest.mark.parametrize(
    "domain",
    [
        BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        LpBallDomain.at_origin(p=1, radius=1.0, dim=2),
        LpBallDomain.at_origin(p=2, radius=0.0, dim=2),
    ],
)
def test_refuses_unsupported_parameter_geometry(domain: Domain) -> None:
    metadata, input_data = _regression_problem()
    prepared, rows, _, _, _ = ToMEstimation(loss=RobHytResidualLoss(k=1.0)).prepare(
        metadata, input_data, PureDPBudget(eps=2.0), 0
    )
    unsupported = MEstimationMetadata(
        n_samples=prepared.n_samples,
        data_domain=prepared.data_domain,
        parameter_domain=domain,
        loss=prepared.loss,
    )
    with pytest.raises(InvalidRegime, match="positive-radius L2 ball"):
        ObjectivePerturbationKifer().run(unsupported, rows, PureDPBudget(eps=2.0), 0)


def test_refuses_zero_smoothness_before_solving() -> None:
    """A loss whose declared curvature is zero is refused before any solve."""
    metadata = MEstimationMetadata(
        n_samples=1,
        data_domain=ProductDomain(
            parts=(
                LpBallDomain.at_origin(p=2, radius=0.0, dim=2),
                BoxDomain(bounds=np.array([[-1.0, 1.0]])),
            )
        ),
        parameter_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        loss=SquaredResidualLoss(),
    )
    with pytest.raises(InvalidRegime, match="positive loss smoothness"):
        ObjectivePerturbationKifer().run(
            metadata, MEstimationInput(data=np.zeros((1, 3))), PureDPBudget(eps=2.0), 0
        )


@pytest.mark.parametrize("candidate", [[0.0, 0.0], [10.0, 10.0]])
def test_ball_solver_status_cannot_bypass_candidate_checks(
    candidate: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    from dpstats.tasks.m_estimation.algorithms import (
        objective_perturbation_kifer,
    )

    metadata, input_data = _regression_problem()
    algorithm = CompositeAlgorithm(
        transformations=(
            ToMEstimation(loss=RobHytResidualLoss(k=1.0), parameter_radius=1.0),
        ),
        algorithm=ObjectivePerturbationKifer(),
    )
    monkeypatch.setattr(
        objective_perturbation_kifer,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(x=np.array(candidate), success=True),
    )
    with pytest.raises(RuntimeError, match="feasibility and first-order"):
        algorithm.run(metadata, input_data, PureDPBudget(eps=2.0), 0)


def test_refuses_a_loss_that_is_not_continuously_twice_differentiable() -> None:
    metadata, input_data = _regression_problem()
    prepared_metadata, prepared_input, _, _, _ = ToMEstimation(
        loss=HuberResidualLoss(threshold=1.0)
    ).prepare(metadata, input_data, PureDPBudget(eps=1.0), seed=0)

    with pytest.raises(InvalidRegime, match="continuously twice"):
        ObjectivePerturbationKifer().run(
            prepared_metadata, prepared_input, PureDPBudget(eps=1.0), seed=0
        )


def test_final_gradient_failure_is_an_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _regression_problem()

    def fail_gradient(*args: Any, **kwargs: Any) -> NoReturn:
        raise FloatingPointError("injected")

    monkeypatch.setattr(RobHytResidualLoss, "gradient", fail_gradient)
    monkeypatch.setattr(
        _objective_perturbation_utils,
        "root",
        lambda *args, **kwargs: SimpleNamespace(x=np.zeros(2)),
    )

    with pytest.raises(RuntimeError, match="numerical solve") as caught:
        _robhyt_composition().run(metadata, input_data, PureDPBudget(eps=2.0), seed=0)
    assert isinstance(caught.value.__cause__, FloatingPointError)
