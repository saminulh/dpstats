"""End-to-end behavior of the public task-store sweep runner."""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.analysis import measure_results
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
    Transformation,
)
from dpstats.exceptions import NoRelease
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    NoPrivacy,
    PureDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.documents import STAGING_PREFIX
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    PARTITION_PREFIX,
    attempts_directory,
    claims_directory,
    registry_directory_name,
    workspace_local_root,
)
from dpstats.sweep import (
    CellScheduler,
    declare_limits_enforced,
    build_executor_scheduler,
    schedule_cells_serially,
    CellContext,
    SubinverseSampleSizeDelta,
)
from dpstats.results import ResultAttempt, ResultStatus
from dpstats.sweep.preparation import PreparedCell
from dpstats.sweep import _cell as sweep_cell
from dpstats.sweep._claims import claim_attempt
from tests.attempts import (
    current_result,
    read_attempt_rows,
    write_attempt_partition,
)
from dpstats.scripts.reconcile_sweep import main as reconcile_sweep_main
from dpstats.scripts.run_sweep import main as run_sweep_main
from dpstats.tasks.mean_estimation.algorithms.gaussian_mechanism import (
    GaussianMechanism,
)
from dpstats.tasks.mean_estimation.algorithms.sample_mean import SampleMean
from dpstats.simulations.mean_estimation import (
    simulate_clipped_normal,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks import LinearRegression, SingleCoefficientEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ApproxBounds as SCEApproxBounds,
    LargestOneHotFWLDemean,
    ToLinearRegression,
)
from tests.registry_components import (
    SIMPLE_TASK,
    InverseSquareDelta,
    NonprivateAlgorithm,
    OversizedDelta,
    SimpleAlgorithm,
    simple_producer,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
)
from tests.sweep_helpers import (
    cli_args,
    observing_schedule,
    write_mean_benchmark_sweep,
    local_task_root,
    sweep_coordinates,
    write_sweep,
)


def _canonical_test_gzip(payload: bytes) -> bytes:
    """Match the persisted gzip representation for this size-only assertion."""
    return gzip.compress(payload, compresslevel=9, mtime=0)


def constant_producer(seed: int) -> tuple[SimpleMetadata, SimpleInput, dict[str, Any]]:
    """Produce one fixed toy instance with an independently known mean."""

    del seed
    return (
        SimpleMetadata(n_samples=2, dim=1),
        SimpleInput(X=np.array([[1.0], [3.0]])),
        {"reference": [2.0]},
    )


class LargeOutputAlgorithm(SimpleAlgorithm):
    """Return a large payload whose stored presence is easy to inspect."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        return SimpleOutput(result=np.ones(10_000))


class RefusingAlgorithm(SimpleAlgorithm):
    """Fail the way an incompatible pairing does: on every cell, with a reason."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, seed
        raise RuntimeError(
            f"SimpleAlgorithm cannot realize a {type(budget).__name__} guarantee"
        )


class AllocationRefusedAlgorithm(SimpleAlgorithm):
    """Fail the way an allocation beyond a cap does."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        raise MemoryError()


class SingularAlgorithm(SimpleAlgorithm):
    """Fail with an exception type the interpreter does not define."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        raise np.linalg.LinAlgError("singular matrix")


_PRIVATE_SENTINEL = "PRIVATE_COVARIATE_SENTINEL_57"


class LeakingAlgorithm(SimpleAlgorithm):
    """Fail with a message quoting the private input it was holding."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, budget, seed
        raise ValueError(f"{_PRIVATE_SENTINEL}: design row {input.X[0].tolist()}")


class NonfiniteOutputAlgorithm(SimpleAlgorithm):
    """Release one requested non-finite value to exercise execution classification."""

    value: float

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        return SimpleOutput(result=np.array([self.value]))


class CountingLinearRegressionAlgorithm(BaseAlgorithm):
    """Expose whether an exact prepared LR coordinate executed again."""

    TASK = LinearRegression
    calls: ClassVar[int] = 0
    offset: float = parameter_field(
        default=0.0, description="Test offset added to the coefficient."
    )

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        del input, budget, seed
        type(self).calls += 1
        return LinearRegressionOutput(
            beta=np.arange(metadata.dim, dtype=np.float64) + 10.0 + self.offset
        )


class CountingToLinearRegression(ToLinearRegression):
    """Expose how often one exact public transformation call is performed."""

    calls: ClassVar[int] = 0

    def prepare(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        LinearRegressionMetadata,
        LinearRegressionInput,
        DPBudget,
        int,
        ToLinearRegression.Restoration,
    ]:
        type(self).calls += 1
        return super().prepare(metadata, input, budget, seed)


class PassThroughSimpleTransformation(Transformation):
    """Preserve a simple-task problem under a distinct leaf seed, so its
    composition is a preparation whose leaf no direct cell answers."""

    OUTER_TASK: ClassVar = SIMPLE_TASK
    INNER_TASK: ClassVar = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def prepare(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[SimpleMetadata, SimpleInput, DPBudget, int, None]:
        return metadata, input, budget, seed + 10, None

    def restore(self, output: SimpleOutput, restoration: None) -> SimpleOutput:
        return output


class IdentitySCETransformation(Transformation):
    """Preserve an SCE problem while giving its composition a distinct identity."""

    OUTER_TASK: ClassVar = SingleCoefficientEstimation
    INNER_TASK: ClassVar = SingleCoefficientEstimation

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def prepare(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        SingleCoefficientEstimationMetadata,
        SingleCoefficientEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        return metadata, input, budget, seed, None

    def restore(
        self,
        output: SingleCoefficientEstimationOutput,
        restoration: None,
    ) -> SingleCoefficientEstimationOutput:
        return output


class WrongOutputToLinearRegression(Transformation):
    """Violate the transformation contract at restoration for boundary testing."""

    OUTER_TASK: ClassVar = SingleCoefficientEstimation
    INNER_TASK: ClassVar = LinearRegression

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def prepare(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        LinearRegressionMetadata,
        LinearRegressionInput,
        DPBudget,
        int,
        ToLinearRegression.Restoration,
    ]:
        return ToLinearRegression().prepare(metadata, input, budget, seed)

    def restore(
        self,
        output: LinearRegressionOutput,
        restoration: ToLinearRegression.Restoration,
    ) -> LinearRegressionOutput:
        del restoration
        return output


def regression_projection_instance(
    seed: int,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput, dict[str, Any]]:
    """Produce the LR problem underlying the SCE reuse checks."""
    del seed
    design_domain = ProductDomain(
        parts=(
            InterceptDomain(),
            BoxDomain(bounds=np.array([[0.0, 10.0]])),
            OneHotDomain(dim=2),
        )
    )
    input = LinearRegressionInput(
        X=np.array(
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 2.0, 1.0, 0.0],
                [1.0, 4.0, 0.0, 1.0],
                [1.0, 6.0, 0.0, 1.0],
            ]
        ),
        y=np.array([2.0, 3.0, 5.0, 7.0]),
    )
    return (
        LinearRegressionMetadata(
            n_samples=4,
            design_domain=design_domain,
            response_domain=BoxDomain(bounds=np.array([[0.0, 10.0]])),
        ),
        input,
        {"reference": [0.0, 0.0, 0.0, 0.0]},
    )


def single_coefficient_projection_instance(
    seed: int,
    *,
    coefficient_of_interest_index: int = 1,
) -> tuple[
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationInput,
    dict[str, Any],
]:
    """Present the same rows as an SCE problem targeting the numeric covariate."""
    metadata, input, _ = regression_projection_instance(seed)
    return (
        SingleCoefficientEstimationMetadata(
            n_samples=metadata.n_samples,
            coefficient_of_interest_index=coefficient_of_interest_index,
            design_domain=metadata.design_domain,
            response_domain=metadata.response_domain,
        ),
        SingleCoefficientEstimationInput(X=input.X, y=input.y),
        {"reference": 0.0},
    )


def single_coefficient_unbounded_response_instance(
    seed: int,
) -> tuple[
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationInput,
    dict[str, Any],
]:
    """Require ApproxBounds to complete one domain from the private response."""
    metadata, input, references = single_coefficient_projection_instance(seed)
    return (
        SingleCoefficientEstimationMetadata(
            n_samples=metadata.n_samples,
            coefficient_of_interest_index=metadata.coefficient_of_interest_index,
            design_domain=metadata.design_domain,
            response_domain=EuclideanSpace(dim=1),
        ),
        input,
        references,
    )


def scale_shaped_producer(
    *, n: int = 100, d: int = 5
) -> Callable[[int], tuple[SimpleMetadata, SimpleInput, dict[str, Any]]]:
    """Expose large instance-owned arrays as reproducible metric arguments."""

    def produce(seed: int) -> tuple[SimpleMetadata, SimpleInput, dict[str, Any]]:
        X = np.random.default_rng(seed).standard_normal((n, d))
        return (
            SimpleMetadata(n_samples=n, dim=d),
            SimpleInput(X=X),
            {
                "reference": np.zeros(d).tolist(),
                "X": X.tolist(),
                "y": np.arange(n, dtype=np.float64).tolist(),
            },
        )

    return produce


class ClaimCountingAlgorithm(SimpleAlgorithm):
    """Report how many attempt claims stood while this cell was executing."""

    claims_directory: str

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        claims = Path(self.claims_directory)
        held = list(claims.iterdir()) if claims.is_dir() else []
        return SimpleOutput(result=np.array([float(len(held))]))


def _write_sweep(
    collection: TaskCollection,
    *,
    name: str = "local",
    seeds: Sequence[int] = (0,),
    algorithm: BaseAlgorithm | None = None,
    budgets: Sequence[DPBudget] = (PureDPBudget(eps=1.0),),
    time_limit: float | None = None,
    memory_limit_gb: float | None = None,
) -> Path:
    """Install one algorithm and the constant Instance, and save their Sweep."""
    return write_sweep(
        collection,
        [SimpleAlgorithm() if algorithm is None else algorithm],
        [constant_producer],
        budgets,
        seeds=seeds,
        time_limit=time_limit,
        memory_limit_gb=memory_limit_gb,
        name=name,
    )


def test_composite_reuses_the_exact_child_result_but_runs_a_changed_problem(
    tmp_path: Path,
) -> None:
    child = CountingLinearRegressionAlgorithm()
    CountingLinearRegressionAlgorithm.calls = 0
    lr_collection = open_collection(init_workspace(tmp_path), LinearRegression)
    source_sweep = write_sweep(
        lr_collection,
        [child],
        [regression_projection_instance],
        [NoPrivacy()],
    )
    lr_collection.run_sweep(source_sweep)
    assert CountingLinearRegressionAlgorithm.calls == 1

    direct = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=child
    )
    fwl = CompositeAlgorithm(
        transformations=(LargestOneHotFWLDemean(), ToLinearRegression()),
        algorithm=child,
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    outer_sweep = write_sweep(
        sce_collection,
        [direct, fwl],
        [single_coefficient_projection_instance],
        [NoPrivacy()],
    )

    returned = sce_collection.run_sweep(
        outer_sweep,
    )

    # The direct projection has the source LR coordinate exactly, while FWL
    # constructs a distinct, one-column LR instance and therefore executes it.
    assert CountingLinearRegressionAlgorithm.calls == 2
    by_algorithm = returned.set_index("alg_uuid")
    direct_output = by_algorithm.loc[direct.uuid, "payload"]
    fwl_output = by_algorithm.loc[fwl.uuid, "payload"]
    assert isinstance(direct_output, SingleCoefficientEstimationOutput)
    assert isinstance(fwl_output, SingleCoefficientEstimationOutput)
    assert direct_output.coefficient == 11.0
    assert fwl_output.coefficient == 10.0


def test_sweep_accepts_the_child_seed_from_approx_bounds(tmp_path: Path) -> None:
    algorithm = CompositeAlgorithm(
        transformations=(SCEApproxBounds(bounds_share=0.5), ToLinearRegression()),
        algorithm=CountingLinearRegressionAlgorithm(),
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    sweep = write_sweep(
        sce_collection,
        [algorithm],
        [single_coefficient_unbounded_response_instance],
        [PureDPBudget(eps=100.0)],
    )
    CountingLinearRegressionAlgorithm.calls = 0

    returned = sce_collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.SUCCESS.value]
    assert CountingLinearRegressionAlgorithm.calls == 1


def test_a_resource_failed_exact_child_result_is_retried_on_request(
    tmp_path: Path,
) -> None:
    child = CountingLinearRegressionAlgorithm()
    lr_collection = open_collection(init_workspace(tmp_path), LinearRegression)
    source_sweep = write_sweep(
        lr_collection,
        [child],
        [regression_projection_instance],
        [NoPrivacy()],
    )
    (source_coordinate,) = sweep_coordinates(lr_collection, source_sweep)
    write_attempt_partition(
        local_task_root(lr_collection),
        [
            current_result(
                instance_uuid=source_coordinate.instance_uuid,
                alg_uuid=source_coordinate.alg_uuid,
                budget=source_coordinate.to_record()["budget"],
                status="timeout",
            )
        ],
        output_columns={"beta": pa.list_(pa.float64())},
    )
    direct = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=child
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    outer_sweep = write_sweep(
        sce_collection,
        [direct],
        [single_coefficient_projection_instance],
        [NoPrivacy()],
    )
    (outer_coordinate,) = sweep_coordinates(sce_collection, outer_sweep)
    CountingLinearRegressionAlgorithm.calls = 0

    returned = sce_collection.run_sweep(outer_sweep, mode="retry_resources")

    assert returned["alg_uuid"].tolist() == [outer_coordinate.alg_uuid]
    assert list(returned["status"]) == [ResultStatus.SUCCESS.value]
    assert CountingLinearRegressionAlgorithm.calls == 1


def test_sweep_records_a_restored_output_for_the_wrong_task_as_an_error(
    tmp_path: Path,
) -> None:
    algorithm = CompositeAlgorithm(
        transformations=(WrongOutputToLinearRegression(),),
        algorithm=CountingLinearRegressionAlgorithm(),
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    sweep = write_sweep(
        sce_collection,
        [algorithm],
        [single_coefficient_projection_instance],
        [NoPrivacy()],
    )

    returned = sce_collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.ALGORITHM_ERROR.value]
    assert list(returned["payload"]) == ["TypeError"]


def test_sweep_processes_and_executes_each_exact_leaf_operation_once(
    tmp_path: Path,
) -> None:
    first = CompositeAlgorithm(
        transformations=(CountingToLinearRegression(),),
        algorithm=CountingLinearRegressionAlgorithm(offset=0.0),
    )
    second = CompositeAlgorithm(
        transformations=(CountingToLinearRegression(),),
        algorithm=CountingLinearRegressionAlgorithm(offset=1.0),
    )
    through_identity = CompositeAlgorithm(
        transformations=(IdentitySCETransformation(), CountingToLinearRegression()),
        algorithm=CountingLinearRegressionAlgorithm(offset=0.0),
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    sweep = write_sweep(
        sce_collection,
        [first, second, through_identity],
        [
            single_coefficient_projection_instance,
            partial(
                single_coefficient_projection_instance,
                coefficient_of_interest_index=2,
            ),
        ],
        [NoPrivacy()],
    )
    CountingToLinearRegression.calls = 0
    CountingLinearRegressionAlgorithm.calls = 0

    returned = sce_collection.run_sweep(sweep)

    # The first two compositions share one transformation call, while the first and
    # third reach the same exact child coordinate through distinct chains.
    assert CountingToLinearRegression.calls == 4
    assert CountingLinearRegressionAlgorithm.calls == 2
    assert list(returned["status"]) == ["success"] * 6
    assert sorted(payload.coefficient for payload in returned["payload"]) == [
        11.0,
        11.0,
        12.0,
        12.0,
        12.0,
        13.0,
    ]


@pytest.mark.parametrize(
    ("source_status", "expected_status", "expected_calls"),
    [
        pytest.param("no_release", "no_release", 0, id="terminal-result"),
        pytest.param(None, "success", 1, id="unanswered"),
    ],
)
def test_composite_projects_terminal_child_results_but_runs_unanswered_ones(
    tmp_path: Path,
    source_status: str | None,
    expected_status: str,
    expected_calls: int,
) -> None:
    child = CountingLinearRegressionAlgorithm()
    lr_collection = open_collection(init_workspace(tmp_path), LinearRegression)
    source_sweep = write_sweep(
        lr_collection,
        [child],
        [regression_projection_instance],
        [NoPrivacy()],
    )
    (source_coordinate,) = sweep_coordinates(lr_collection, source_sweep)
    if source_status is not None:
        write_attempt_partition(
            local_task_root(lr_collection),
            [
                current_result(
                    instance_uuid=source_coordinate.instance_uuid,
                    alg_uuid=source_coordinate.alg_uuid,
                    budget=source_coordinate.to_record()["budget"],
                    status=source_status,
                )
            ],
            output_columns={"beta": pa.list_(pa.float64())},
        )
    direct = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=child
    )
    sce_collection = open_collection(
        init_workspace(tmp_path), SingleCoefficientEstimation
    )
    outer_sweep = write_sweep(
        sce_collection,
        [direct],
        [single_coefficient_projection_instance],
        [NoPrivacy()],
    )
    CountingLinearRegressionAlgorithm.calls = 0

    returned = sce_collection.run_sweep(
        outer_sweep,
    )

    assert list(returned["status"]) == [expected_status]
    assert CountingLinearRegressionAlgorithm.calls == expected_calls


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        pytest.param(RefusingAlgorithm(), "RuntimeError", id="builtin"),
        pytest.param(
            SingularAlgorithm(), "numpy.linalg.LinAlgError", id="module-qualified"
        ),
    ],
)
def test_a_failed_cell_records_what_kind_of_failure_it_was(
    collection: TaskCollection, algorithm: BaseAlgorithm, expected: str
) -> None:
    """The durable row names the failure's type, so a mass failure is legible.

    One incompatible pairing meets every cell identically. Without the type on
    the row, a run that produced thousands of error rows would be
    indistinguishable from one that crashed, and the kind of failure would
    survive only in a worker log that may already have rotated. Both spellings
    a reader can meet are pinned: a builtin and a third-party type.
    """
    sweep = _write_sweep(collection, algorithm=algorithm)

    results = collection.run_sweep(sweep)

    assert list(results["status"]) == [ResultStatus.ALGORITHM_ERROR.value]
    assert list(results["payload"]) == [expected]

    (persisted,) = read_attempt_rows(local_task_root(collection))
    assert persisted["reason"] == expected


def test_a_failing_algorithms_message_never_reaches_the_record(
    collection: TaskCollection,
) -> None:
    """An exception's text is never persisted, only its type.

    A message can quote whatever the algorithm was holding, which is private
    input, and a Result is published. Nothing here redacts the sentinel: the
    recorded name is derived from the exception's class, so its message has no
    route to the record at all.
    """
    sweep = _write_sweep(collection, algorithm=LeakingAlgorithm())

    results = collection.run_sweep(sweep)

    assert list(results["status"]) == [ResultStatus.ALGORITHM_ERROR.value]
    assert list(results["payload"]) == ["ValueError"]

    task_store = local_task_root(collection)
    (persisted,) = read_attempt_rows(task_store)
    assert persisted["reason"] == "ValueError"

    written = [
        path.read_bytes() for path in sorted(task_store.rglob("*")) if path.is_file()
    ]
    assert written
    assert not any(_PRIVATE_SENTINEL.encode() in payload for payload in written)
    assert not any(
        _PRIVATE_SENTINEL.encode() in gzip.decompress(payload)
        for payload in written
        if payload.startswith(b"\x1f\x8b")
    )


def test_empty_store_runs_every_selected_cell_and_records_each_outcome(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(collection, seeds=[0, 1])

    returned = collection.run_sweep(sweep)

    assert returned[["alg_seed", "status"]].to_dict(orient="records") == [
        {"alg_seed": 0, "status": "success"},
        {"alg_seed": 1, "status": "success"},
    ]
    assert [payload.result.tolist() for payload in returned["payload"]] == [
        [2.0],
        [2.0],
    ]
    task_store = local_task_root(collection)
    (partition,) = attempts_directory(task_store).glob(f"{PARTITION_PREFIX}*")
    assert len(list(partition.glob("*.parquet"))) == 1
    stored = read_attempt_rows(task_store)
    assert len(stored) == 2
    assert [row["result"] for row in stored] == [[2.0], [2.0]]
    assert not list(task_store.glob(f"{STAGING_PREFIX}*"))
    assert not list(claims_directory(task_store).iterdir())


@pytest.mark.parametrize("memory_limit_gb", [None, 2.0])
def test_a_memory_error_is_typed_only_under_a_declared_cap(
    collection: TaskCollection, memory_limit_gb: float | None
) -> None:
    """Under an enforcing scheduler a cap makes the refusal a typed outcome;
    with no cap the failure keeps the error's name."""
    sweep = _write_sweep(
        collection,
        algorithm=AllocationRefusedAlgorithm(),
        memory_limit_gb=memory_limit_gb,
    )

    returned = collection.run_sweep(
        sweep, schedule_cells=declare_limits_enforced(schedule_cells_serially)
    )

    (status,) = returned["status"]
    (payload,) = returned["payload"]
    if memory_limit_gb is None:
        assert status is ResultStatus.ALGORITHM_ERROR
        assert payload == "MemoryError"
    else:
        assert status is ResultStatus.MEMORY_LIMIT_EXCEEDED
        assert payload is None
    assert not list(claims_directory(local_task_root(collection)).iterdir())


def test_scheduler_receives_diagnostics_and_may_complete_out_of_order(
    collection: TaskCollection,
) -> None:
    """A scheduler controls ordering while the library controls durable meaning."""
    sweep = _write_sweep(
        collection,
        seeds=[0, 1, 2],
        time_limit=30.0,
        memory_limit_gb=2.0,
    )
    offered: list[CellContext] = []

    def reverse_schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        offered.extend(initial)
        for execution in reversed(offered):
            assert not list(complete(execute(execution)))

    returned = collection.run_sweep(
        sweep, schedule_cells=declare_limits_enforced(reverse_schedule)
    )

    assert [execution.coordinate.alg_seed for execution in offered] == [0, 1, 2]
    assert {execution.time_limit_s for execution in offered} == {30.0}
    assert {execution.memory_limit_bytes for execution in offered} == {2 * 1024**3}
    assert list(returned["alg_seed"]) == [0, 1, 2]
    assert list(returned["peak_rss_mb"]) == [None, None, None]
    assert not list(claims_directory(local_task_root(collection)).iterdir())


def test_scheduler_receives_the_artifact_root_and_the_serial_route_loads_once(
    collection: TaskCollection,
) -> None:
    """Each offered job of an original Instance names the Input root; the
    execute step loads it, once per Instance on the serial route, so the
    artifacts may vanish after the Instance's first job has run. A
    composite's prepared leaves carry their prepared Input."""
    composite = CompositeAlgorithm(
        transformations=(PassThroughSimpleTransformation(),),
        algorithm=SimpleAlgorithm(),
    )
    sweep = write_sweep(
        collection,
        [composite, SimpleAlgorithm()],
        [constant_producer],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )
    artifacts = workspace_local_root(collection.workspace) / INPUT_ARTIFACTS_DIRECTORY
    offered: list[CellContext] = []

    def observe_then_remove_artifacts(execution: CellContext) -> None:
        offered.append(execution)
        if len(offered) == 2:
            for artifact in artifacts.iterdir():
                artifact.unlink()

    returned = collection.run_sweep(
        sweep, schedule_cells=observing_schedule(observe_then_remove_artifacts)
    )

    assert returned["status"].tolist() == [ResultStatus.SUCCESS] * 4
    # Two anchors, their two prepared leaves, two direct leaves: the serial
    # route runs each preparation's leaf before the next offered job.
    assert [isinstance(execution.input, Path) for execution in offered] == [
        True,
        False,
        True,
        False,
        True,
        True,
    ]
    assert {
        execution.input for execution in offered if isinstance(execution.input, Path)
    } == {artifacts}


@pytest.mark.parametrize("scheduler", ["serial", "executor"])
def test_an_unreadable_input_artifact_records_no_attempt_and_frees_the_cell(
    collection: TaskCollection, scheduler: str
) -> None:
    """An Input artifact that cannot be read is a machinery incident: the
    load failure is raised, the cell records no Attempt, and its claim is
    released so it is pending again."""
    sweep = _write_sweep(collection, seeds=[0, 1])
    artifacts = workspace_local_root(collection.workspace) / INPUT_ARTIFACTS_DIRECTORY

    def losing(
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
    ) -> Callable[[CellContext], PreparedCell | ResultAttempt]:
        def execute_without_artifacts(
            context: CellContext,
        ) -> PreparedCell | ResultAttempt:
            for artifact in artifacts.iterdir():
                artifact.unlink()
            return execute(context)

        return execute_without_artifacts

    def schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        if scheduler == "serial":
            schedule_cells_serially(initial, losing(execute), complete)
            return
        with ThreadPoolExecutor(max_workers=2) as pool:
            build_executor_scheduler(pool, 2)(initial, losing(execute), complete)

    with pytest.raises(FileNotFoundError):
        collection.run_sweep(sweep, schedule_cells=schedule)

    task_store = local_task_root(collection)
    assert read_attempt_rows(task_store) == []
    assert not list(claims_directory(task_store).iterdir())


@pytest.mark.parametrize("memory_limit_gb", [None, 2.0])
def test_a_load_refused_by_a_declared_cap_is_a_memory_limit_outcome(
    collection: TaskCollection,
    monkeypatch: pytest.MonkeyPatch,
    memory_limit_gb: float | None,
) -> None:
    """A load the enforcing adapter's cap refuses answers the cell as
    ``memory_limit_exceeded``; with no cap the refusal is raised unrecorded."""
    sweep = _write_sweep(collection, seeds=[0, 1], memory_limit_gb=memory_limit_gb)

    def refuse_load(instance: Any, root: Path) -> Any:
        raise MemoryError

    monkeypatch.setattr(sweep_cell, "load_instance_input", refuse_load)
    task_store = local_task_root(collection)
    if memory_limit_gb is None:
        with pytest.raises(MemoryError):
            collection.run_sweep(sweep)
        assert read_attempt_rows(task_store) == []
    else:
        returned = collection.run_sweep(
            sweep, schedule_cells=declare_limits_enforced(schedule_cells_serially)
        )
        assert returned["status"].tolist() == [ResultStatus.MEMORY_LIMIT_EXCEEDED] * 2
        assert returned["payload"].tolist() == [None, None]
    assert not list(claims_directory(task_store).iterdir())


def test_the_execute_step_crosses_to_a_process_pool_and_loads_there(
    collection: TaskCollection,
) -> None:
    """A process adapter receives a transferable execute step whose workers
    load each cell's Input themselves."""
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer()],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1, 2],
        instance_seeds=[0, 1],
    )

    with ProcessPoolExecutor(max_workers=2) as pool:
        returned = collection.run_sweep(
            sweep, schedule_cells=build_executor_scheduler(pool, 2)
        )

    assert returned["status"].tolist() == [ResultStatus.SUCCESS] * 6
    assert not list(claims_directory(local_task_root(collection)).iterdir())


def test_scheduler_failure_keeps_completed_siblings_durable(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(collection, seeds=[0, 1])

    def fail_after_one(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        complete(execute(next(iter(initial))))
        raise RuntimeError("scheduler failed")

    with pytest.raises(RuntimeError, match="scheduler failed"):
        collection.run_sweep(sweep, schedule_cells=fail_after_one)

    task_root = local_task_root(collection)
    rows = read_attempt_rows(task_root)
    assert len(rows) == 1
    assert rows[0]["seed"] == 0


def test_a_rejected_outcome_reaches_the_scheduler_as_an_exception(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(collection)
    rejected: list[BaseException] = []

    def invalid_schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        next(iter(initial))
        try:
            complete({"not": "an attempt"})  # type: ignore[arg-type]
        except TypeError as error:
            rejected.append(error)
            raise

    with pytest.raises(TypeError, match="must complete ResultAttempt"):
        collection.run_sweep(sweep, schedule_cells=invalid_schedule)

    assert len(rejected) == 1


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "malformed"])
def test_scheduler_outcome_contract_fails_closed(
    collection: TaskCollection,
    failure: str,
) -> None:
    sweep = _write_sweep(collection)

    def invalid_schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        execution = next(iter(initial))
        if failure == "missing":
            return
        if failure == "malformed":
            complete({"not": "an attempt"})  # type: ignore[arg-type]
            return
        attempt = execute(execution)
        assert isinstance(attempt, ResultAttempt)
        if failure == "duplicate":
            complete(attempt)
            complete(attempt)
            return
        complete(
            ResultAttempt(
                coordinate=attempt.coordinate,
                attempt_number=attempt.attempt_number + 1,
                status=ResultStatus.ALGORITHM_ERROR,
                payload="RuntimeError",
            )
        )

    expected = {
        "missing": "completed no outcome",
        "duplicate": "unplanned coordinate or Attempt",
        "foreign": "unplanned coordinate or Attempt",
        "malformed": "must complete ResultAttempt",
    }[failure]
    with pytest.raises((TypeError, ValueError), match=expected):
        collection.run_sweep(sweep, schedule_cells=invalid_schedule)

    task_root = local_task_root(collection)
    rows = read_attempt_rows(task_root)
    assert len(rows) == (1 if failure == "duplicate" else 0)


def test_a_cell_holds_exactly_its_own_claim_while_it_executes(
    collection: TaskCollection,
) -> None:
    """The window a claim covers starts before the cell and ends after it.

    The algorithm reports what the claim store held while it was running, so
    a claim that were written after the work — or never — is visible as a
    count of zero.
    """
    task_store = local_task_root(collection)
    sweep = _write_sweep(
        collection,
        algorithm=ClaimCountingAlgorithm(
            claims_directory=str(claims_directory(task_store))
        ),
    )

    returned = collection.run_sweep(sweep)

    assert [payload.result.tolist() for payload in returned["payload"]] == [[1.0]]
    assert not list(claims_directory(task_store).iterdir())


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_execution_records_a_released_infinite_output_as_a_success(
    collection: TaskCollection,
    value: float,
) -> None:
    """An infinite Output is a release and its values reach storage unchanged."""
    sweep = _write_sweep(
        collection,
        algorithm=NonfiniteOutputAlgorithm(value=value),
    )

    returned = collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.SUCCESS.value]
    (row,) = read_attempt_rows(local_task_root(collection))
    assert row["status"] == ResultStatus.SUCCESS.value
    assert row["reason"] is None
    assert row["result"][0] == value


def test_execution_records_a_nan_output_as_an_execution_failure(
    collection: TaskCollection,
) -> None:
    """NaN is no released value, so the Attempt stores a failure and no Output."""
    sweep = _write_sweep(
        collection,
        algorithm=NonfiniteOutputAlgorithm(value=math.nan),
    )

    returned = collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.ALGORITHM_ERROR.value]
    (row,) = read_attempt_rows(local_task_root(collection))
    assert row["status"] == ResultStatus.ALGORITHM_ERROR.value
    assert row["reason"] == "ValueError"
    assert row["result"] is None


def test_scale_shaped_sweep_does_not_repeat_instance_payloads(
    collection: TaskCollection,
) -> None:
    algorithms = [SimpleAlgorithm(scale=scale) for scale in (1.0, 2.0)]
    generator = scale_shaped_producer(n=20_000, d=2)
    sweep = write_sweep(
        collection,
        algorithms,
        [generator],
        [PureDPBudget(eps=eps) for eps in (1.0, 2.0)],
        seeds=[0, 1],
    )
    algorithm_uuids = [algorithm.uuid for algorithm in algorithms]

    collection.run_sweep(sweep)
    task_store = local_task_root(collection)

    rows = read_attempt_rows(task_store)
    assert len(rows) == 8
    assert {row["alg_id"] for row in rows} == set(algorithm_uuids)
    assert all("metric_arguments" not in row for row in rows)

    _, _, references = generator(0)
    encoded_references = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in references.items()
    }
    legacy_lines = [
        canonical_json_bytes({**row, "metric_arguments": encoded_references}) + b"\n"
        for row in sorted(rows, key=lambda row: (row["alg_id"], row["seed"]))
    ]
    current_bytes = sum(
        path.stat().st_size
        for path in attempts_directory(task_store).glob(
            f"{PARTITION_PREFIX}*/*.parquet"
        )
    )
    legacy_bytes = len(_canonical_test_gzip(b"".join(legacy_lines)))

    assert current_bytes < 10_000
    assert legacy_bytes > current_bytes * 100


def test_run_sweep_command_executes_the_authored_selection(
    tmp_path: Path, collection: TaskCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _write_sweep(collection, seeds=[7])

    run_sweep_main(
        cli_args(
            sweep,
            tmp_path,
            "--algorithm-registry",
            "test",
            "--instance-registry",
            "test",
            "--include-output",
        )
    )

    rendered = capsys.readouterr().out
    assert "success" in rendered
    assert "ResultStatus." not in rendered
    assert "payload" in rendered.splitlines()[0]
    output = collection.load_results().iloc[0]["payload"]
    np.testing.assert_array_equal(output.result, [2.0])
    assert read_attempt_rows(local_task_root(collection))

    run_sweep_main(
        cli_args(
            sweep,
            tmp_path,
            "--algorithm-registry",
            "test",
            "--instance-registry",
            "test",
            "--mode",
            "all",
        )
    )

    assert "success" in capsys.readouterr().out
    assert sorted(
        row["attempt"] for row in read_attempt_rows(local_task_root(collection))
    ) == [0, 1]


def test_run_sweep_command_has_no_no_save_mode(
    tmp_path: Path, collection: TaskCollection
) -> None:
    sweep = _write_sweep(collection)

    with pytest.raises(SystemExit):
        run_sweep_main(
            [
                "--sweep",
                str(sweep),
                "--workspace",
                str(tmp_path),
                "--no-save",
            ]
        )

    assert read_attempt_rows(local_task_root(collection)) == []


def test_complete_sweep_resume_executes_nothing_and_returns_the_same_results(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(collection, seeds=[0, 1])
    first = collection.run_sweep(sweep)
    authored = sweep.read_bytes()
    task_store = local_task_root(collection)
    files_before = {
        path: path.read_bytes()
        for path in attempts_directory(task_store).glob(
            f"{PARTITION_PREFIX}*/*.parquet"
        )
    }

    executed: list[tuple[int, int]] = []

    resumed = collection.run_sweep(sweep, schedule_cells=_recording_schedule(executed))

    assert executed == []
    assert sweep.read_bytes() == authored
    pd.testing.assert_frame_equal(
        resumed.drop(columns=["payload"]), first.drop(columns=["payload"])
    )
    assert {
        path: path.read_bytes()
        for path in attempts_directory(task_store).glob(
            f"{PARTITION_PREFIX}*/*.parquet"
        )
    } == files_before


@pytest.mark.parametrize(
    "first_status", [ResultStatus.SUCCESS, ResultStatus.NO_RELEASE]
)
def test_recompute_appends_a_new_attempt_without_rewriting_history(
    collection: TaskCollection, first_status: ResultStatus
) -> None:
    """An explicit recomputation supersedes rather than overwrites a stored Attempt.

    It applies to an answered failure as well as a success.
    """
    sweep = _write_sweep(collection)
    first = collection.run_sweep(
        sweep, schedule_cells=_forced_status_schedule([first_status])
    )
    task_store = local_task_root(collection)
    rows_before = read_attempt_rows(task_store)

    recomputed = collection.run_sweep(sweep, mode="all")

    assert first["status"].tolist() == [first_status]
    assert recomputed["status"].tolist() == [ResultStatus.SUCCESS]
    assert rows_before == [
        row for row in read_attempt_rows(task_store) if not row["attempt"]
    ]
    assert sorted(
        (row["attempt"], row["status"]) for row in read_attempt_rows(task_store)
    ) == [(0, first_status.value), (1, ResultStatus.SUCCESS.value)]

    resumed = collection.run_sweep(sweep)

    assert len(resumed) == 1
    assert sorted(row["attempt"] for row in read_attempt_rows(task_store)) == [0, 1]


def test_executor_run_records_siblings_and_abandons_the_raising_cell(
    collection: TaskCollection,
) -> None:
    """One job raising is a machinery incident, not an outcome: every sibling
    in flight is recorded, the raising cell is left pending with no claim, and
    the run stops."""
    sweep = _write_sweep(collection, seeds=[0, 1, 2])

    def lose_seed_one(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        def execute_or_lose(context: CellContext) -> PreparedCell | ResultAttempt:
            if context.coordinate.alg_seed == 1:
                raise RuntimeError("worker lost")
            return execute(context)

        with ThreadPoolExecutor(max_workers=3) as pool:
            build_executor_scheduler(pool, 3)(initial, execute_or_lose, complete)

    with pytest.raises(RuntimeError, match="worker lost"):
        collection.run_sweep(sweep, schedule_cells=lose_seed_one)

    task_store = local_task_root(collection)
    rows = {row["seed"]: row for row in read_attempt_rows(task_store)}
    assert sorted(rows) == [0, 2]
    assert {rows[0]["status"], rows[2]["status"]} == {ResultStatus.SUCCESS.value}
    assert not list(claims_directory(task_store).iterdir())

    resumed: list[tuple[int, int]] = []
    collection.run_sweep(sweep, schedule_cells=_recording_schedule(resumed))
    assert resumed == [(1, 0)]


def test_a_raising_serial_job_leaves_no_claim_behind(
    collection: TaskCollection,
) -> None:
    """A job that raises under the serial scheduler stops the run, keeps the
    Attempts already recorded, and releases every claim it took, so the
    unrecorded cells are simply pending again."""
    sweep = _write_sweep(collection, seeds=[0, 1, 2])

    def lose_seed_one(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        def execute_or_lose(context: CellContext) -> PreparedCell | ResultAttempt:
            if context.coordinate.alg_seed == 1:
                raise RuntimeError("worker lost")
            return execute(context)

        schedule_cells_serially(initial, execute_or_lose, complete)

    with pytest.raises(RuntimeError, match="worker lost"):
        collection.run_sweep(sweep, schedule_cells=lose_seed_one)

    task_store = local_task_root(collection)
    (kept,) = read_attempt_rows(task_store)
    assert (kept["seed"], kept["attempt"], kept["status"]) == (
        0,
        0,
        ResultStatus.SUCCESS.value,
    )
    assert not list(claims_directory(task_store).iterdir())

    resumed: list[tuple[int, int]] = []
    collection.run_sweep(sweep, schedule_cells=_recording_schedule(resumed))
    assert resumed == [(1, 0), (2, 0)]
    assert kept in read_attempt_rows(task_store)


def _recording_schedule(executed: list[tuple[int, int]]) -> CellScheduler:
    """Execute every offered cell, recording its seed and attempt number."""
    return observing_schedule(
        lambda execution: executed.append(
            (execution.coordinate.alg_seed, execution.attempt_number)
        )
    )


def _forced_status_schedule(statuses: Sequence[ResultStatus]) -> CellScheduler:
    """Execute each cell, then report one declared outcome per algorithm seed.

    A scheduler owns resource enforcement, so reporting a timeout or a memory
    failure for a cell it ran is what an enforcing scheduler does.
    """

    def declared_status(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        def execute_declared(execution: CellContext) -> ResultAttempt:
            result = execute(execution)
            assert isinstance(result, ResultAttempt)
            status = statuses[execution.coordinate.alg_seed]
            if status is ResultStatus.SUCCESS:
                return result
            payload: Any = None
            if status in (
                ResultStatus.NO_RELEASE,
                ResultStatus.INVALID_REGIME,
                ResultStatus.ALGORITHM_ERROR,
            ):
                payload = f"forced {status.value}"
            return ResultAttempt(
                coordinate=result.coordinate,
                attempt_number=result.attempt_number,
                status=status,
                payload=payload,
                runtime_s=result.runtime_s,
                peak_rss_mb=result.peak_rss_mb,
            )

        schedule_cells_serially(initial, execute_declared, complete)

    return declared_status


def test_resource_retry_selects_only_timeout_and_memory_results(
    collection: TaskCollection,
) -> None:
    # Every terminal outcome answers its cell; only a resource one reopens.
    statuses = list(ResultStatus)
    sweep = _write_sweep(
        collection,
        seeds=list(range(len(statuses))),
    )
    first = collection.run_sweep(
        sweep, schedule_cells=_forced_status_schedule(statuses)
    )
    assert list(first["status"]) == [status.value for status in statuses]

    resumed_executions: list[tuple[int, int]] = []
    resumed = collection.run_sweep(
        sweep, schedule_cells=_recording_schedule(resumed_executions)
    )
    assert resumed_executions == []
    pd.testing.assert_frame_equal(
        resumed.drop(columns=["payload"]), first.drop(columns=["payload"])
    )

    retried: list[tuple[int, int]] = []
    second = collection.run_sweep(
        sweep,
        mode="retry_resources",
        schedule_cells=_recording_schedule(retried),
    )

    assert retried == [
        (statuses.index(ResultStatus.TIMEOUT), 1),
        (statuses.index(ResultStatus.MEMORY_LIMIT_EXCEEDED), 1),
    ]
    assert list(second["status"]) == [
        ResultStatus.SUCCESS.value,
        ResultStatus.NO_RELEASE.value,
        ResultStatus.INVALID_REGIME.value,
        ResultStatus.SUCCESS.value,
        ResultStatus.SUCCESS.value,
        ResultStatus.ALGORITHM_ERROR.value,
    ]
    attempts = read_attempt_rows(local_task_root(collection))
    assert len(attempts) == len(statuses) + 2
    assert sorted(
        row["attempt"]
        for row in attempts
        if row["seed"]
        in {
            statuses.index(ResultStatus.TIMEOUT),
            statuses.index(ResultStatus.MEMORY_LIMIT_EXCEEDED),
        }
    ) == [0, 0, 1, 1]


def test_ordinary_resume_refills_only_the_unanswered_cell(
    collection: TaskCollection,
) -> None:
    """A cell no Result answers needs no operator request to be executed.

    A vanished runner's claim leaves its coordinate exactly as unanswered as
    one never attempted, so a plain resume takes it again under its first
    attempt number, while every recorded terminal outcome still stands.
    """
    sweep = _write_sweep(collection, seeds=[0, 1])
    _, held = sweep_coordinates(collection, sweep)
    task_store = local_task_root(collection)
    first = collection.run_sweep(sweep, seeds=[0])
    assert list(first["status"]) == ["success"]
    claim_attempt(task_store, held, 0)

    executed: list[tuple[int, int]] = []
    resumed = collection.run_sweep(sweep, schedule_cells=_recording_schedule(executed))

    assert executed == [(1, 0)]
    assert list(resumed["status"]) == ["success", "success"]
    chain = sorted(
        (row["seed"], row["attempt"], row["status"])
        for row in read_attempt_rows(task_store)
    )
    assert chain == [(0, 0, "success"), (1, 0, "success")]
    assert not list(claims_directory(task_store).iterdir())


@pytest.mark.parametrize(
    "resource_status",
    [ResultStatus.TIMEOUT, ResultStatus.MEMORY_LIMIT_EXCEEDED],
)
def test_resource_retry_extends_a_pruned_imported_result(
    collection: TaskCollection,
    resource_status: ResultStatus,
) -> None:
    sweep = _write_sweep(collection)
    collection.run_sweep(
        sweep, schedule_cells=_forced_status_schedule([resource_status])
    )
    task_store = local_task_root(collection)
    (coordinate,) = sweep_coordinates(collection, sweep)
    for path in attempts_directory(task_store).glob(f"{PARTITION_PREFIX}*/*.parquet"):
        path.unlink()
    imported = current_result(
        instance_uuid=coordinate.instance_uuid,
        alg_uuid=coordinate.alg_uuid,
        budget=coordinate.budget.dump_config().model_dump(),
        status=resource_status.value,
    )
    imported["attempt_number"] = 3
    write_attempt_partition(task_store, [imported], name="imported")

    executed: list[tuple[int, int]] = []
    retried = collection.run_sweep(
        sweep,
        mode="retry_resources",
        schedule_cells=_recording_schedule(executed),
    )

    assert executed == [(0, 4)]
    assert retried["status"].tolist() == [ResultStatus.SUCCESS]
    rows = read_attempt_rows(task_store)
    assert sorted(row["attempt"] for row in rows) == [3, 4]
    assert {row["alg_id"] for row in rows} == {coordinate.alg_uuid}
    files_before_resume = {
        path: path.read_bytes()
        for path in attempts_directory(task_store).glob(
            f"{PARTITION_PREFIX}*/*.parquet"
        )
    }

    resumed_executions: list[tuple[int, int]] = []
    resumed = collection.run_sweep(
        sweep, schedule_cells=_recording_schedule(resumed_executions)
    )

    assert resumed_executions == []
    assert resumed.to_dict(orient="records") == retried.to_dict(orient="records")
    assert {
        path: path.read_bytes()
        for path in attempts_directory(task_store).glob(
            f"{PARTITION_PREFIX}*/*.parquet"
        )
    } == files_before_resume


def test_two_sweep_files_reuse_shared_cells_in_one_task_store(
    collection: TaskCollection,
) -> None:
    first_sweep = _write_sweep(collection, name="first")
    second_sweep = _write_sweep(collection, name="second", seeds=[0, 1])
    collection.run_sweep(first_sweep)
    executed: list[tuple[int, int]] = []

    returned = collection.run_sweep(
        second_sweep, schedule_cells=_recording_schedule(executed)
    )

    assert executed == [(1, 0)]
    assert list(returned["alg_seed"]) == [0, 1]
    assert list(returned["status"]) == ["success", "success"]
    assert len(read_attempt_rows(local_task_root(collection))) == 2


def test_output_materialization_never_changes_stored_attempts(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(collection, algorithm=LargeOutputAlgorithm())

    returned = collection.run_sweep(sweep)

    assert "output" not in returned.columns
    (stored,) = read_attempt_rows(local_task_root(collection))
    assert len(stored["result"]) == 10_000


def test_nonprivate_selection_ignores_declared_private_budgets(
    collection: TaskCollection,
) -> None:
    sweep = _write_sweep(
        collection,
        algorithm=NonprivateAlgorithm(),
        budgets=[SubinverseSampleSizeDelta(eps=1.0)],
    )

    returned = collection.run_sweep(sweep)

    assert list(returned["budget"]) == [NoPrivacy()]
    # No selected algorithm can spend the declared budget.
    assert json.loads(sweep.read_text())["budgets"] == [
        {
            "path": ("dpstats.sweep.budget_policies:" "SubinverseSampleSizeDelta"),
            "config": {"eps": 1.0},
        }
    ]


def test_a_declared_no_privacy_budget_sweeps_a_private_algorithm(
    collection: TaskCollection,
) -> None:
    """An unconstrained offer is an ordinary declarable budget.

    How a private algorithm behaves at an infinite budget is itself worth
    sweeping, so a Sweep declares NoPrivacy like any other budget and the
    private algorithm receives its own coordinate for it, beside rather than
    merged into the single coordinate a nonprivate algorithm derives.
    """
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm(), NonprivateAlgorithm()],
        [simple_producer()],
        [NoPrivacy(), PureDPBudget(eps=1.0)],
    )

    returned = collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.SUCCESS.value] * 3
    selected: dict[str, list[str]] = {}
    for alg_uuid, budget in zip(returned["alg_uuid"], returned["budget"]):
        selected.setdefault(alg_uuid, []).append(type(budget).__name__)
    assert sorted(selected.values(), key=len) == [
        ["NoPrivacy"],
        ["NoPrivacy", "PureDPBudget"],
    ]


@pytest.mark.parametrize("declared", [0.0, -1.0])
def test_a_sweep_declaring_an_invalid_cell_limit_is_rejected(
    collection: TaskCollection, declared: float
) -> None:
    sweep = _write_sweep(collection)
    document = json.loads(sweep.read_text())
    document["time_limit"] = declared
    sweep.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="is invalid") as rejection:
        collection.run_sweep(sweep)

    assert str(sweep) in str(rejection.value)
    assert "positive and finite" in str(rejection.value.__cause__)
    assert read_attempt_rows(local_task_root(collection)) == []


@pytest.mark.parametrize("declared", [[True], [3.0], ["1"]])
def test_a_sweep_declaring_a_non_integer_seed_is_rejected(
    collection: TaskCollection, declared: list[Any]
) -> None:
    """The Sweep reader is where a seed's type is settled.

    A boolean, a float, and a numeric string each reach ``seeds`` as an
    authored entry, and each is refused where the field is declared, so no
    later stage has to re-establish what a seed is. The Sweep names the file
    it could not read; which field was wrong travels on the chained cause.
    """
    sweep = _write_sweep(collection)
    document = json.loads(sweep.read_text())
    document["seeds"] = declared
    sweep.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="is invalid") as rejection:
        collection.run_sweep(sweep)

    assert str(sweep) in str(rejection.value)
    assert "seeds" in str(rejection.value.__cause__)
    assert read_attempt_rows(local_task_root(collection)) == []


def test_own_algorithm_registry_runs_on_installed_benchmark_instances(
    tmp_path: Path,
) -> None:
    mean_collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(mean_collection, SampleMean())

    returned = mean_collection.run_sweep(sweep)

    assert returned.iloc[0]["payload"].mean_est.tolist() == [2.0]


def test_public_measure_results_loads_a_builtin_metric_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from dpstats.budgets import PureDPBudget
        from dpstats.workspace import init_workspace, open_collection
        from dpstats.analysis import measure_results
        from dpstats.tasks import list_built_in_metrics
        from dpstats.tasks import MeanEstimation
        from dpstats.tasks.mean_estimation.algorithms import SampleMean
        from tests.sweep_helpers import write_mean_benchmark_sweep

        collection = open_collection(init_workspace(sys.argv[1]), MeanEstimation)
        sweep = write_mean_benchmark_sweep(
            collection,
            SampleMean(),
            budgets=[PureDPBudget(eps=1.0)],
        )
        assert not any(name.endswith("._metrics") for name in sys.modules)

        collection.run_sweep(sweep)
        assert not any(name.endswith("._metrics") for name in sys.modules)
        measurements = measure_results(
            collection.load_results(),
            list_built_in_metrics(MeanEstimation)["l2_error_to_sample_mean"],
            collection=collection,
        )

        assert not measurements.empty
        assert set(measurements["l2_error_to_sample_mean"]) == {0.0}
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_installed_benchmark_algorithm_runs_on_own_instances(tmp_path: Path) -> None:
    mean_collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_sweep(
        mean_collection,
        [GaussianMechanism()],
        [lambda seed: simulate_clipped_normal(seed=seed, dimension=1, n_samples=8)],
        [ApproxDPBudget(eps=1.0, delta=1e-5)],
    )

    returned = mean_collection.run_sweep(sweep)

    assert list(returned["status"]) == ["success"]
    assert returned.iloc[0]["payload"].mean_est.shape == (1,)


def test_a_declared_policy_supplies_the_recorded_delta(tmp_path: Path) -> None:
    mean_collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(
        mean_collection,
        GaussianMechanism(),
        budgets=[InverseSquareDelta(eps=1.0)],
        values=[2.0] * 4,
    )

    returned = mean_collection.run_sweep(sweep)

    # The declared policy is n**-2, which is 1/16 at this sample size — a value
    # the built-in policy's 0.01 cap could never produce.
    assert list(returned["budget"]) == [ApproxDPBudget(eps=1.0, delta=0.0625)]
    (declared,) = json.loads(sweep.read_text())["budgets"]
    assert declared == {
        "path": "tests.registry_components:InverseSquareDelta",
        "config": {"eps": 1.0},
    }
    assert not sweep.with_name("extension.bound.json").exists()


def test_a_policy_producing_an_invalid_budget_records_nothing(tmp_path: Path) -> None:
    mean_collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(
        mean_collection,
        GaussianMechanism(),
        budgets=[OversizedDelta(eps=1.0)],
    )
    authored = sweep.read_bytes()

    with pytest.raises(ValueError) as refusal:
        mean_collection.run_sweep(sweep)

    message = str(refusal.value)
    assert "OversizedDelta" in message
    assert "for Instance " in message
    assert "delta must be in" in str(refusal.value.__cause__)
    assert sweep.read_bytes() == authored
    assert read_attempt_rows(local_task_root(mean_collection)) == []


def test_measurement_refuses_an_unreadable_instance_registry(
    tmp_path: Path,
) -> None:
    mean_collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(
        mean_collection,
        GaussianMechanism(),
        budgets=[ApproxDPBudget(eps=1.0, delta=1e-5)],
    )
    mean_collection.run_sweep(sweep)
    task_store = local_task_root(mean_collection)
    before = read_attempt_rows(task_store)
    registry_path = task_store / registry_directory_name("instance") / "benchmark.json"
    document = json.loads(registry_path.read_text())
    instance_uuid = next(iter(document["entries"]))
    del document["entries"][instance_uuid]["metric_arguments"]
    registry_path.write_text(json.dumps(document))

    # Measurement joins the Instance registry before it measures anything, so the
    # registry boundary rejects the missing required field.
    with pytest.raises(ValueError, match=instance_uuid) as refusal:
        from dpstats.tasks import list_built_in_metrics

        measure_results(
            mean_collection.load_results(),
            list_built_in_metrics(MeanEstimation)["l2_error_to_sample_mean"],
        )

    assert "metric_arguments" in str(refusal.value.__cause__)
    assert read_attempt_rows(task_store) == before


def test_reconcile_sweep_settles_a_store_whose_policy_cannot_be_imported(
    collection: TaskCollection,
) -> None:
    """Settling abandoned work never depends on the code a Sweep names.

    An operator reconciles wherever a killed runner left its store, which may
    be an environment without the policy module that authored the budgets.
    Reconciliation reads the declared Task and reconstructs nothing, while
    every operation that resolves budgets still refuses the same sweep.
    """
    sweep = _write_sweep(collection)
    (claimed,) = sweep_coordinates(collection, sweep)
    task_store = local_task_root(collection)
    claim_attempt(task_store, claimed, 0)
    document = json.loads(sweep.read_text())
    document["budgets"] = [{"path": "absent_policies:HouseDelta", "config": {}}]
    sweep.write_text(json.dumps(document))

    cleared = collection.reconcile()

    assert cleared == [claimed.uuid]
    assert not list(claims_directory(task_store).iterdir())
    with pytest.raises(ValueError, match="is invalid"):
        collection.run_sweep(sweep)


def test_reconcile_sweep_command_reports_what_it_settled(
    tmp_path: Path, collection: TaskCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _write_sweep(collection)
    (coordinate,) = sweep_coordinates(collection, sweep)
    claim_attempt(local_task_root(collection), coordinate, 0)

    reconcile_sweep_main(cli_args(sweep, tmp_path))

    rendered = capsys.readouterr().out
    assert coordinate.uuid in rendered
    assert "unanswered" in rendered
    assert not list(claims_directory(local_task_root(collection)).iterdir())


def test_reconcile_sweep_command_on_a_settled_store_reports_nothing_to_do(
    tmp_path: Path, collection: TaskCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _write_sweep(collection)
    collection.run_sweep(sweep)

    reconcile_sweep_main(cli_args(sweep, tmp_path))

    assert "no unreconciled" in capsys.readouterr().out.lower()


class SilentlyDecliningAlgorithm(SimpleAlgorithm):
    """Release the accounted bottom outcome without composing a message."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        raise NoRelease("")


def test_a_declining_algorithm_needs_no_prose_to_be_recorded(
    collection: TaskCollection,
) -> None:
    """A wordless NoRelease is an outcome, not a run-ending validation failure."""
    sweep = _write_sweep(collection, algorithm=SilentlyDecliningAlgorithm())

    returned = collection.run_sweep(sweep)

    assert list(returned["status"]) == ["no_release"]
    assert list(returned["payload"]) == [""]


_REGIME_MESSAGE = "SimpleAlgorithm realizes no implication for the offered family."


class RegimeLimitedAlgorithm(SimpleAlgorithm):
    """Refuse every offered budget to exercise the declared-limit outcome."""

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, input, budget, seed
        raise InvalidRegime(_REGIME_MESSAGE)


def test_a_declared_regime_limit_is_recorded_once_and_never_retried(
    collection: TaskCollection,
) -> None:
    """An invalid_regime outcome answers its cell terminally.

    The refusal is deterministic, so a resume executes nothing even when
    resource failures are retried, and history holds exactly one observation.
    """
    sweep = _write_sweep(collection, algorithm=RegimeLimitedAlgorithm())

    returned = collection.run_sweep(sweep)
    assert list(returned["status"]) == ["invalid_regime"]
    assert list(returned["payload"]) == [_REGIME_MESSAGE]

    resumed = collection.run_sweep(
        sweep,
        mode="retry_resources",
    )
    assert resumed.to_dict(orient="records") == returned.to_dict(orient="records")
    (row,) = read_attempt_rows(local_task_root(collection))
    assert row["status"] == "invalid_regime"
