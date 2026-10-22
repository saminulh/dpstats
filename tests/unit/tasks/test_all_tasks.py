"""Contracts every built-in Task, algorithm and instance producer satisfies."""

import functools
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import dpstats.budgets as budgets
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.tasks import BUILT_IN_TASKS, list_built_in_algorithms
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    RobHytResidualLoss,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)
from dpstats.simulations.linear_regression import (
    simulate_clipped_gaussian as linear_regression_gaussian,
)
from dpstats.simulations.covariance_estimation import (
    simulate_gaussian as covariance_gaussian,
)
from dpstats.simulations.logistic_regression import (
    simulate_gaussian as logistic_regression_gaussian,
)
from dpstats.simulations.m_estimation import (
    simulate_gaussian as m_estimation_gaussian,
)
from dpstats.simulations.mean_estimation import (
    simulate_clipped_laplacian,
    simulate_clipped_normal,
)
from dpstats.simulations.median_estimation import simulate_simple_gaussian
from dpstats.simulations.simple_linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.simulations.single_coefficient_estimation import (
    simulate_clipped_gaussian as single_coefficient_gaussian,
)
from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.exceptions import NoRelease
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.algorithm import load_algorithms, register_algorithm
from dpstats.workspace import init_workspace, open_collection
from dpstats.transport.registry import read_registry
from dpstats.task import Task, TaskInput, TaskMetadata
from pydantic import BaseModel, ValidationError
from tests.attempts import read_attempt_rows
from tests.registry_components import ExternalBudget
from tests.sweep_helpers import local_task_root, write_sweep
from tests.unit.tasks.task_builders import Unreadable
from dpstats.workspace.registries import registry_document_path
from dpstats.workspace.layout import task_collection_directory

# Releases the smoke producers cannot reach: objective perturbation needs finite
# loss constants the unbounded Gaussian producers do not declare, and local
# output perturbation needs curvature certifiable at the producer's design bound.
DESIGN_RESTRICTED_ALGORITHMS = {
    "AMP",
    "LocalOutputPerturbation",
    "ObjectivePerturbationRedberg",
}
# A constant release never reads data, so it proves nothing about the sweep path.
CONSTANT_RELEASE_ALGORITHMS = {"ZeroCoefficient"}
PRODUCERS_BY_TASK_ID: dict[str, tuple[Callable[..., object], ...]] = {
    "covariance_estimation": (covariance_gaussian,),
    "linear_regression": (linear_regression_gaussian,),
    "logistic_regression": (logistic_regression_gaussian,),
    "m_estimation": (m_estimation_gaussian,),
    "mean_estimation": (simulate_clipped_normal, simulate_clipped_laplacian),
    "median_estimation": (simulate_simple_gaussian,),
    "simple_linear_regression": (simulate_clipped_gaussian,),
    "single_coefficient_estimation": (single_coefficient_gaussian,),
}


def _first_smoke_algorithm(task: Task) -> BaseAlgorithm:
    for name, AlgClass in list_built_in_algorithms(task).items():
        if (
            name not in DESIGN_RESTRICTED_ALGORITHMS
            and name not in CONSTANT_RELEASE_ALGORITHMS
        ):
            return AlgClass()
    from dpstats.algorithm import CompositeAlgorithm
    from dpstats.tasks.linear_regression.algorithms.ols import OLS
    from dpstats.tasks.single_coefficient_estimation.transformations import (
        ToLinearRegression,
    )

    composed = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=OLS()
    )
    if composed.task is task:
        return composed
    raise AssertionError(f"{task.id} exposes no runnable algorithm.")


def _smoke_budget(algorithm: BaseAlgorithm) -> DPBudget:
    if algorithm.privacy_interpretation is PrivacyInterpretation.NONPRIVATE:
        return NO_PRIVACY
    return ApproxDPBudget(eps=1.0, delta=1e-5)


class TestAllTasksLoad:
    def test_all_tasks_have_algorithms(self):
        from dpstats.tasks import BUILT_IN_TASKS

        for task_id, task in BUILT_IN_TASKS.items():
            assert task.id == task_id
            assert list_built_in_algorithms(task), f"{task_id} has no algorithms"


class TestAllTasksSmoke:
    @pytest.fixture
    def all_tasks(self):
        from dpstats.tasks import BUILT_IN_TASKS

        return BUILT_IN_TASKS

    def test_default_constructed_algorithms_round_trip_through_registry(
        self, all_tasks, tmp_path
    ):
        for task_id, task in all_tasks.items():
            registry_path = registry_document_path(
                task_collection_directory(tmp_path, task_id), "algorithm", "built-in"
            )
            expected = {}

            for AlgClass in list_built_in_algorithms(task).values():
                algorithm = AlgClass()
                expected[
                    register_algorithm(tmp_path, algorithm, registry="built-in")
                ] = algorithm.dump_config().model_dump()

            entries = read_registry(registry_path.read_bytes()).entries
            reloaded = load_algorithms(entries, task_id=task.id)
            actual = {
                uuid: algorithm.dump_config().model_dump()
                for uuid, algorithm in reloaded.items()
            }

            assert actual == expected, (
                f"{task_id} has a default algorithm configuration that does "
                "not round-trip through its registry file"
            )

    def test_metadata_rejects_nonpositive_n_samples(self, all_tasks):
        for task in all_tasks.values():
            metadata, _, _ = PRODUCERS_BY_TASK_ID[task.id][0](seed=42)
            # Field names only: a derived cache such as ``loss_properties``
            # lives in the model's attribute dict but is not authored state.
            authored: dict[str, Any] = {
                name: getattr(metadata, name) for name in task.Metadata.model_fields
            }
            for bad_n_samples in (0, -3):
                with pytest.raises(ValueError, match="n_samples must be positive"):
                    task.Metadata(**{**authored, "n_samples": bad_n_samples})

    def test_every_task_generates_a_valid_instance(self, all_tasks):
        for task in all_tasks.values():
            for produce in PRODUCERS_BY_TASK_ID.get(task.id, ()):
                metadata, input_data, metric_arguments = produce(seed=42)
                assert isinstance(metadata, task.Metadata)
                assert isinstance(input_data, task.Input)
                assert isinstance(metric_arguments, dict)
                # Instance.metric_arguments is persisted as strict canonical
                # JSON, so producers must publish compact lists/scalars rather
                # than NumPy arrays or sample-sized private data.
                canonical_json_bytes(metric_arguments)

    def test_every_task_output_family_round_trips_without_references(
        self, all_tasks, tmp_path
    ):
        for task in all_tasks.values():
            algorithm = _first_smoke_algorithm(task)
            produce = PRODUCERS_BY_TASK_ID[task.id][0]
            metadata, input_data, _ = produce(seed=42)
            budget = _smoke_budget(algorithm)
            expected_output = algorithm.run(
                metadata, input_data, budget=budget, seed=42
            )
            collection = open_collection(init_workspace(tmp_path), task)
            sweep = write_sweep(
                collection,
                [algorithm],
                [produce],
                [budget],
                seeds=[42],
                instance_seeds=[42],
            )
            collection.run_sweep(sweep)

            (row,) = read_attempt_rows(local_task_root(collection))
            released = {name: row[name] for name in expected_output.model_fields}
            assert released == expected_output.model_dump(mode="python")
            assert "metric_arguments" not in row


# ---------------------------------------------------------------------------
# Roster contracts: invariants every built-in algorithm and producer satisfies,
# stated once over the whole discovered roster.
# ---------------------------------------------------------------------------

# One representative budget per exported budget family.
REPRESENTATIVE_BUDGETS: dict[type[DPBudget], DPBudget] = {
    PureDPBudget: PureDPBudget(eps=1.0),
    ApproxDPBudget: ApproxDPBudget(eps=1.0, delta=1e-5),
    ZCDPBudget: ZCDPBudget(rho=0.5),
    GDPBudget: GDPBudget(mu=1.0),
    NoPrivacy: NO_PRIVACY,
}
# A generous budget, so a test-then-release mechanism (PTR) has a regime in
# which it releases on these small instances.
GENEROUS_BUDGET = ApproxDPBudget(eps=10.0, delta=1e-3)
# Every offer an algorithm must release on or refuse: each exported family's
# representative, each finite family at an infinite parameter, and a
# third-party family no built-in algorithm knows.
OFFERED_BUDGETS: dict[str, DPBudget] = {
    **{family.__name__: budget for family, budget in REPRESENTATIVE_BUDGETS.items()},
    "PureDPBudget-inf": PureDPBudget(eps=math.inf),
    "ApproxDPBudget-inf": ApproxDPBudget(eps=math.inf, delta=1e-5),
    "ZCDPBudget-inf": ZCDPBudget(rho=math.inf),
    "GDPBudget-inf": GDPBudget(mu=math.inf),
    "ApproxDPBudget-inf-delta1": ApproxDPBudget(eps=math.inf, delta=1.0),
    "ApproxDPBudget-delta1": ApproxDPBudget(eps=1.0, delta=1.0),
    "ExternalBudget": ExternalBudget(credits=1.0),
}

RosterInstance = tuple[TaskMetadata, TaskInput]


def _drawn(
    produce: Callable[..., tuple[TaskMetadata, TaskInput, Any]], **options: Any
) -> RosterInstance:
    metadata, input_data, _ = produce(seed=0, **options)
    return metadata, input_data


def _intercept_regression() -> RosterInstance:
    """A column-zero intercept with twelve bounded covariates and bounded responses."""
    rng = np.random.default_rng(0)
    n_samples, covariates = 1000, 12
    X = np.column_stack(
        [np.ones(n_samples), rng.uniform(-1.0, 1.0, size=(n_samples, covariates))]
    )
    y = np.clip(0.5 + X[:, 1:].mean(axis=1) + rng.normal(0.0, 0.1, n_samples), -4, 4)
    metadata = LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.tile([[-1.0, 1.0]], (covariates, 1))),
            )
        ),
        response_domain=BoxDomain(bounds=np.array([[-4.0, 4.0]])),
    )
    return metadata, LinearRegressionInput(X=X, y=y)


def _tiny_ball_regression() -> RosterInstance:
    """Five scalar-design rows in the unit ball, for a sampler over the ball."""
    rng = np.random.default_rng(0)
    metadata = LinearRegressionMetadata(
        n_samples=5,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    input_data = LinearRegressionInput(
        X=rng.uniform(-0.5, 0.5, size=(5, 1)), y=rng.uniform(-0.5, 0.5, size=5)
    )
    return metadata, input_data


def _bounded_design_robust_loss() -> RosterInstance:
    """A generalized-linear loss with finite constants over the whole parameter space."""
    rng = np.random.default_rng(0)
    n_samples, dim = 100, 2
    X = rng.normal(size=(n_samples, dim))
    X /= np.maximum(1.0, np.linalg.norm(X, axis=1))[:, None]
    y = X.sum(axis=1) + rng.normal(0.0, 0.3, size=n_samples)
    metadata = MEstimationMetadata(
        n_samples=n_samples,
        parameter_domain=EuclideanSpace(dim=dim),
        data_domain=ProductDomain(
            parts=(
                LpBallDomain.at_origin(p=2, radius=1.0, dim=dim),
                EuclideanSpace(dim=1),
            )
        ),
        loss=RobHytResidualLoss(k=1.0),
    )
    return metadata, MEstimationInput(data=np.column_stack([X, y]))


def _categorical_mean() -> RosterInstance:
    """A full-coded categorical block with an unobserved declared category."""
    return (
        MeanEstimationMetadata(n_samples=4, domain=OneHotDomain(dim=3)),
        MeanEstimationInput(X=np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])),
    )


def _randomized_trial(n_samples: int = 400) -> RosterInstance:
    """An intercept plus one binary treatment, the coefficient of interest."""
    rng = np.random.default_rng(0)
    treatment = rng.binomial(1, 0.5, size=n_samples).astype(float)
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=n_samples,
        coefficient_of_interest_index=1,
        design_domain=ProductDomain(
            parts=(InterceptDomain(), OneHotDomain(dim=1, allow_zero=True))
        ),
        response_domain=EuclideanSpace(dim=1),
    )
    input_data = SingleCoefficientEstimationInput(
        X=np.column_stack([np.ones(n_samples), treatment]),
        y=1.5 + 0.8 * treatment + rng.normal(0.0, 0.5, size=n_samples),
    )
    return metadata, input_data


# Small instances spanning each Task's declared regimes (bounded or unbounded
# domains, scalar or vector, with or without an intercept). An algorithm is
# exercised on the first instance and budget on which it releases an Output;
# no entry names an algorithm.
ROSTER_INSTANCES: dict[str, tuple[Callable[[], RosterInstance], ...]] = {
    "covariance_estimation": (lambda: _drawn(covariance_gaussian),),
    "linear_regression": (
        lambda: _drawn(linear_regression_gaussian),
        _intercept_regression,
        lambda: _drawn(linear_regression_gaussian, n_samples=2000),
        _tiny_ball_regression,
    ),
    "logistic_regression": (
        lambda: _drawn(logistic_regression_gaussian, n_samples=50, dim=2),
    ),
    "m_estimation": (
        lambda: _drawn(m_estimation_gaussian),
        _bounded_design_robust_loss,
    ),
    "mean_estimation": (
        lambda: _drawn(simulate_clipped_normal),
        lambda: _drawn(simulate_clipped_normal, dimension=1),
        _categorical_mean,
    ),
    "median_estimation": (
        lambda: _drawn(simulate_simple_gaussian, n_samples=40),
        lambda: _drawn(simulate_simple_gaussian, n_samples=40, known_domain=True),
    ),
    "simple_linear_regression": (lambda: _drawn(simulate_clipped_gaussian),),
    "single_coefficient_estimation": (
        lambda: _drawn(single_coefficient_gaussian),
        _randomized_trial,
        # Large enough for curvature certificates to pass at the generous budget.
        lambda: _randomized_trial(n_samples=10000),
    ),
}

# Array-output shapes declared by Metadata; the other Tasks release scalars.
DECLARED_OUTPUT_SHAPE: dict[str, Callable[[Any], tuple[int, ...]]] = {
    "covariance_estimation": lambda metadata: (metadata.dim, metadata.dim),
    "linear_regression": lambda metadata: (metadata.dim,),
    "logistic_regression": lambda metadata: (metadata.dim,),
    "m_estimation": lambda metadata: (metadata.parameter_domain.dim,),
    "mean_estimation": lambda metadata: (metadata.dim,),
}

ROSTER = [
    pytest.param(task, name, id=f"{task.id}-{name}")
    for task in BUILT_IN_TASKS.values()
    for name in list_built_in_algorithms(task)
]

# Infinite-parameter limits that exist but are not yet implemented: each offer
# raises NotImplementedError before the Input is read. Implementing a limit
# makes its entry fail until it is deleted.
UNIMPLEMENTED_LIMITS: frozenset[tuple[str, str, str]] = frozenset(
    (task_id, name, offer)
    for task_id, name in (
        ("linear_regression", "TukeyEM"),
        ("median_estimation", "ExactDistancePTR"),
        ("single_coefficient_estimation", "LocalOutputPerturbation"),
    )
    for offer, offered_budget in OFFERED_BUDGETS.items()
    if offered_budget.is_unconstrained
)


def _fields(value: BaseModel) -> dict[str, Any]:
    """Return a model's declared field values as held, arrays included."""
    return {field: getattr(value, field) for field in type(value).model_fields}


@functools.cache
def _roster_instances(task_id: str) -> tuple[RosterInstance, ...]:
    instances = tuple(build() for build in ROSTER_INSTANCES[task_id])
    for metadata, input_data in instances:
        metadata.validate_input(input_data)
    return instances


@functools.cache
def _release_regime(task: Task, name: str) -> tuple[int, DPBudget]:
    """Return the first roster instance and budget on which *name* releases.

    Any failure only moves the search on: what an algorithm does with each
    budget family on its release regime is checked by the tests below.
    """
    offers = (*REPRESENTATIVE_BUDGETS.values(), GENEROUS_BUDGET)
    failures: list[str] = []
    for position, (metadata, input_data) in enumerate(_roster_instances(task.id)):
        for budget in offers:
            try:
                list_built_in_algorithms(task)[name]().run(
                    metadata, input_data, budget=budget, seed=0
                )
            except Exception as failure:
                failures.append(f"{position}/{budget!r}: {type(failure).__name__}")
                continue
            return position, budget
    raise AssertionError(
        f"{task.id}/{name} releases an Output on no roster instance and "
        f"budget: {failures}"
    )


def test_every_exported_budget_family_has_a_representative() -> None:
    exported = {
        value
        for value in (getattr(budgets, name) for name in budgets.__all__)
        if isinstance(value, type)
        and issubclass(value, DPBudget)
        and value is not DPBudget
    }
    assert exported == set(REPRESENTATIVE_BUDGETS)


@pytest.mark.parametrize(("task", "name"), ROSTER)
def test_every_algorithm_releases_a_seed_determined_valid_output(
    task: Task, name: str
) -> None:
    """Same seed, same finite Output, valid as the Task's Output at its dimension."""
    position, budget = _release_regime(task, name)
    metadata, input_data = _roster_instances(task.id)[position]
    algorithm = list_built_in_algorithms(task)[name]()

    first = algorithm.run(metadata, input_data, budget=budget, seed=0)
    second = algorithm.run(metadata, input_data, budget=budget, seed=0)

    assert isinstance(first, task.Output)
    task.Output.model_validate(_fields(first))
    np.testing.assert_equal(_fields(first), _fields(second))
    assert all(
        np.all(np.isfinite(value))
        for value in _fields(first).values()
        if isinstance(value, (float, np.ndarray))
    )
    arrays = [
        value for value in _fields(first).values() if isinstance(value, np.ndarray)
    ]
    if task.id in DECLARED_OUTPUT_SHAPE:
        expected = DECLARED_OUTPUT_SHAPE[task.id](metadata)
        assert arrays and all(array.shape == expected for array in arrays)
    else:
        assert arrays == []


@pytest.mark.parametrize("offer", list(OFFERED_BUDGETS))
@pytest.mark.parametrize(("task", "name"), ROSTER)
def test_every_algorithm_releases_or_refuses_each_budget_family(
    task: Task,
    name: str,
    offer: str,
) -> None:
    """Offered any budget on its release regime, an algorithm releases or refuses.

    Which families an algorithm accepts is its own privacy contract; this
    restates none of it. An infinite parameter need not behave as
    ``NoPrivacy``, and an unknown family is refused rather than crashing. A
    release is a valid Output or the accounted bottom (``NoRelease``). A
    refusal is ``InvalidRegime`` decided from public information alone. Its
    privacy argument, not Input access order, establishes that condition.
    A declared unimplemented limit raises ``NotImplementedError`` before the
    Input is read.
    """
    position, _ = _release_regime(task, name)
    metadata, input_data = _roster_instances(task.id)[position]
    budget = OFFERED_BUDGETS[offer]

    if (task.id, name, offer) in UNIMPLEMENTED_LIMITS:
        for offered_input in (input_data, cast(TaskInput, Unreadable("Input"))):
            with pytest.raises(NotImplementedError):
                list_built_in_algorithms(task)[name]().run(
                    metadata, offered_input, budget=budget, seed=0
                )
        return

    try:
        output = list_built_in_algorithms(task)[name]().run(
            metadata, input_data, budget=budget, seed=0
        )
    except NoRelease:
        return
    except InvalidRegime:
        return
    assert isinstance(output, task.Output)
    task.Output.model_validate(_fields(output))


@pytest.mark.parametrize(
    "produce",
    [
        pytest.param(produce, id=f"{task_id}-{produce.__name__}")
        for task_id, producers in PRODUCERS_BY_TASK_ID.items()
        for produce in producers
    ],
)
def test_every_producer_draws_a_seed_determined_valid_instance(
    produce: Callable[..., tuple[TaskMetadata, TaskInput, Any]],
) -> None:
    """Same seed, same Instance; it satisfies its own Metadata; seeds differ."""
    metadata, input_data, references = produce(seed=3)
    again_metadata, again_input, again_references = produce(seed=3)
    _, other_input, _ = produce(seed=4)

    metadata.validate_input(input_data)
    assert metadata.model_dump(mode="json") == again_metadata.model_dump(mode="json")
    np.testing.assert_equal(_fields(input_data), _fields(again_input))
    assert canonical_json_bytes(references) == canonical_json_bytes(again_references)
    assert any(
        not np.array_equal(value, _fields(other_input)[field])
        for field, value in _fields(input_data).items()
    )
