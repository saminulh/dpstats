"""Public behavior of composite preparation through one Sweep scheduler."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow.dataset as ds
import pytest

from dpstats import init_workspace, open_collection
from dpstats.algorithm import (
    CompositeAlgorithm,
    PrivacyInterpretation,
    Transformation,
)
from dpstats.budgets import DPBudget, NO_PRIVACY, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import BoxDomain
from dpstats.sweep.preparation import PreparedCell
from dpstats.sweep import (
    CellContext,
    declare_limits_enforced,
    build_executor_scheduler,
    schedule_cells_serially,
)
from dpstats.results import ResultAttempt, ResultStatus
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from dpstats.tasks.mean_estimation.transformations import (
    ApproxBounds,
    DataBounds,
)
from dpstats.workspace.layout import claims_directory
from tests.sweep_helpers import local_task_root, observing_schedule


class SlowIdentity(Transformation):
    """Expose independent composite preparations to the public scheduler."""

    OUTER_TASK: ClassVar = MeanEstimation
    INNER_TASK: ClassVar = MeanEstimation
    active: ClassVar[int] = 0
    peak: ClassVar[int] = 0
    calls: ClassVar[int] = 0
    lock: ClassVar[threading.Lock] = threading.Lock()

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        with type(self).lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            type(self).calls += 1
        try:
            time.sleep(0.05)
            return metadata, input, budget, seed, None
        finally:
            with type(self).lock:
                type(self).active -= 1

    def restore(
        self, output: MeanEstimationOutput, restoration: None
    ) -> MeanEstimationOutput:
        return output


class FailingIdentity(SlowIdentity):
    """Refuse each exact preparation under the native Result classification."""

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        del metadata, input, budget, seed
        raise InvalidRegime("test preparation refusal")


class CopyingIdentity(Transformation):
    """Return an equal private copy to exercise prepared-Input interning."""

    OUTER_TASK: ClassVar = MeanEstimation
    INNER_TASK: ClassVar = MeanEstimation

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        return metadata, MeanEstimationInput(X=input.X.copy()), budget, seed, None

    def restore(
        self, output: MeanEstimationOutput, restoration: None
    ) -> MeanEstimationOutput:
        return output


class OffsetSampleMean(SampleMean):
    """Provide a distinct child identity while sharing composite preparation."""

    offset: float = 0.0

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        output = super().run(metadata, input, budget, seed)
        return MeanEstimationOutput(mean_est=output.mean_est + self.offset)


def _collection(
    tmp_path: Path,
    algorithms: list[CompositeAlgorithm],
    *,
    seeds: list[int],
    input_rows: int = 2,
):
    """Install one public mean-estimation collection and its authored Sweep."""
    workspace = init_workspace(tmp_path / "workspace")
    collection = open_collection(workspace, MeanEstimation)
    collection.add_instance(
        MeanEstimationMetadata(
            n_samples=input_rows,
            domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
        ),
        MeanEstimationInput(X=np.linspace(0.25, 0.75, input_rows).reshape(-1, 1)),
        {},
    )
    for algorithm in algorithms:
        collection.add_algorithm(algorithm)
    sweep = collection.save_sweep(
        "parallel",
        {
            "budgets": [NO_PRIVACY],
            "seeds": seeds,
        },
    )
    return collection, sweep


def _reset_slow_identity() -> None:
    """Reset observable preparation state between public collection runs."""
    SlowIdentity.active = 0
    SlowIdentity.peak = 0
    SlowIdentity.calls = 0


class StaggeredIdentity(Transformation):
    """Make one original Instance's preparation visibly outlast another's."""

    OUTER_TASK: ClassVar = MeanEstimation
    INNER_TASK: ClassVar = MeanEstimation
    slow_started: ClassVar[threading.Event] = threading.Event()
    slow_finished: ClassVar[threading.Event] = threading.Event()
    allow_slow_finish: ClassVar[threading.Event] = threading.Event()

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        if not type(self).slow_started.is_set():
            type(self).slow_started.set()
            if not type(self).allow_slow_finish.wait(timeout=5):
                raise TimeoutError("the scheduler never released the slow preparation")
            type(self).slow_finished.set()
        return metadata, input, budget, seed, None

    def restore(
        self, output: MeanEstimationOutput, restoration: None
    ) -> MeanEstimationOutput:
        return output


def test_scheduler_admits_the_next_instance_while_a_slow_preparation_runs(
    tmp_path: Path,
) -> None:
    """A fast Instance's leaf and the next Instance run beside a slow sibling."""
    for event in (
        StaggeredIdentity.slow_started,
        StaggeredIdentity.slow_finished,
        StaggeredIdentity.allow_slow_finish,
    ):
        event.clear()
    collection = open_collection(init_workspace(tmp_path / "workspace"), MeanEstimation)
    metadata = MeanEstimationMetadata(
        n_samples=1, domain=BoxDomain(bounds=np.array([[0.0, 1.0]]))
    )
    for value in (0.0, 0.5, 0.75):
        collection.add_instance(
            metadata, MeanEstimationInput(X=np.array([[value]])), {}
        )
    collection.add_algorithm(
        CompositeAlgorithm(
            transformations=(StaggeredIdentity(),), algorithm=SampleMean()
        )
    )
    sweep = collection.save_sweep(
        "incremental", {"budgets": [NO_PRIVACY], "seeds": [0]}
    )
    admitted_after_fast_leaf = False

    def schedule_cells(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        nonlocal admitted_after_fast_leaf
        source = iter(initial)
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures: dict[Future[PreparedCell | ResultAttempt], CellContext] = {}

            def admit(context: CellContext) -> None:
                futures[workers.submit(execute, context)] = context

            slow = next(source)
            admit(slow)
            assert StaggeredIdentity.slow_started.wait(timeout=5)
            admit(next(source))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    context = futures.pop(future)
                    ready = tuple(complete(future.result()))
                    for leaf in ready:
                        admit(leaf)
                    if (
                        context.coordinate.instance_uuid
                        != slow.coordinate.instance_uuid
                        and ready
                        and not admitted_after_fast_leaf
                    ):
                        admitted_after_fast_leaf = True
                        admit(next(source))
                        assert not StaggeredIdentity.slow_finished.is_set()
                        StaggeredIdentity.allow_slow_finish.set()
            assert next(source, None) is None

    returned = collection.run_sweep(sweep, schedule_cells=schedule_cells)

    assert admitted_after_fast_leaf
    assert StaggeredIdentity.slow_finished.is_set()
    assert list(returned["status"]) == [ResultStatus.SUCCESS.value] * 3
    assert sorted(float(payload.mean_est[0]) for payload in returned["payload"]) == [
        0.0,
        0.5,
        0.75,
    ]


def test_incremental_preparations_read_a_durable_leaf_partition_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-seed preparations reuse durable leaves without rescanning per seed."""
    collection, sweep = _collection(tmp_path, [], seeds=list(range(10)))
    leaf = SampleMean()
    collection.add_algorithm(leaf)
    collection.run_sweep(sweep)
    outer = CompositeAlgorithm(transformations=(CopyingIdentity(),), algorithm=leaf)
    collection.add_algorithm(outer)
    reads: list[str] = []
    dataset: Callable[..., Any] = ds.dataset

    def read(source: Any, *args: Any, **kwargs: Any) -> ds.Dataset:
        if f"{leaf.uuid}" in str(source):
            reads.append(str(source))
        opened: ds.Dataset = dataset(source, *args, **kwargs)
        return opened

    monkeypatch.setattr(ds, "dataset", read)
    executed: list[CellContext] = []

    returned = collection.run_sweep(
        sweep,
        algorithms=[outer.uuid],
        schedule_cells=observing_schedule(executed.append),
    )

    assert len(returned) == 10
    assert all(np.allclose(output.mean_est, [0.5]) for output in returned["payload"])
    assert all(
        isinstance(context.coordinate.algorithm, CompositeAlgorithm)
        for context in executed
    )
    assert 1 <= len(reads) <= 2


def test_thread_scheduler_prepares_independent_composites_concurrently(
    tmp_path: Path,
) -> None:
    _reset_slow_identity()
    serial_collection, serial_sweep = _collection(
        tmp_path / "serial",
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=list(range(8)),
    )
    serial = serial_collection.run_sweep(serial_sweep)

    _reset_slow_identity()
    collection, sweep = _collection(
        tmp_path / "parallel",
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=list(range(8)),
    )

    with ThreadPoolExecutor(max_workers=4) as workers:
        returned = collection.run_sweep(
            sweep, schedule_cells=build_executor_scheduler(workers, 4)
        )

    assert SlowIdentity.peak > 1
    assert returned[
        ["instance", "algorithm", "budget", "alg_seed", "status", "payload"]
    ].to_dict(orient="records") == serial[
        ["instance", "algorithm", "budget", "alg_seed", "status", "payload"]
    ].to_dict(
        orient="records"
    )
    assert list(returned["status"]) == [ResultStatus.SUCCESS.value] * 8
    assert all(np.allclose(payload.mean_est, [0.5]) for payload in returned["payload"])


def test_executor_scheduler_keeps_jobs_in_flight_within_its_bound(
    tmp_path: Path,
) -> None:
    _reset_slow_identity()
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=list(range(8)),
    )

    with ThreadPoolExecutor(max_workers=8) as workers:
        returned = collection.run_sweep(
            sweep, schedule_cells=build_executor_scheduler(workers, 2)
        )

    assert SlowIdentity.peak == 2
    assert list(returned["status"]) == [ResultStatus.SUCCESS.value] * 8
    with pytest.raises(ValueError, match="at least one"):
        build_executor_scheduler(workers, 0)


def test_exact_prepared_instance_shares_one_live_input(tmp_path: Path) -> None:
    """Intern one prepared Input per prepared-Instance UUID within one original Instance.

    Every seed prepares an exact copy with identical content, so each
    preparation anchor's outcome names the same prepared-Instance UUID. While
    those leaves are live the Sweep holds one Input object for that UUID
    rather than one per anchor, and a resumed Sweep dispatches nothing once
    every leaf is durably recorded.
    """
    collection, sweep = _collection(
        tmp_path,
        [
            CompositeAlgorithm(
                transformations=(CopyingIdentity(),), algorithm=SampleMean()
            )
        ],
        seeds=list(range(8)),
        input_rows=1_048_576,
    )
    leaf_inputs: list[MeanEstimationInput] = []

    def every_preparation_first(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        # Hold every leaf until all preparations finish, so all eight are live.
        leaves = [leaf for context in initial for leaf in complete(execute(context))]
        for leaf in leaves:
            assert isinstance(leaf.input, MeanEstimationInput)
            leaf_inputs.append(leaf.input)
        for leaf in leaves:
            assert not list(complete(execute(leaf)))

    returned = collection.run_sweep(sweep, schedule_cells=every_preparation_first)

    assert len(returned) == 8
    assert list(returned["status"]) == [ResultStatus.SUCCESS.value] * 8
    assert len(leaf_inputs) == 8
    assert {id(input) for input in leaf_inputs} == {id(leaf_inputs[0])}
    assert leaf_inputs[0].X.nbytes == 8 * 1024**2
    assert leaf_inputs[0].X.nbytes * len(leaf_inputs) == 64 * 1024**2

    def must_not_run(context: CellContext) -> None:
        raise AssertionError("a resumed Sweep must reuse its durable Results")

    resumed = collection.run_sweep(
        sweep, schedule_cells=observing_schedule(must_not_run)
    )
    assert list(resumed["status"]) == [ResultStatus.SUCCESS.value] * 8
    assert all(np.allclose(payload.mean_est, [0.5]) for payload in resumed["payload"])


def test_sweep_logs_scheduled_and_reused_leaf_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The benchmark.sweep INFO counts describe execution shape, not data."""
    _reset_slow_identity()
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=[0, 1],
    )

    with caplog.at_level("INFO", logger="benchmark.sweep"):
        collection.run_sweep(sweep)

    messages = "\n".join(caplog.messages)
    assert "2 pending outer cell(s) across 1 original Instance(s)" in messages
    assert (
        "Sweep recorded 2 outer cell(s); 2 unique leaf/leaves dispatched and 0 "
        "answered by result reuse" in messages
    )
    assert "0.25" not in messages


def test_exact_shared_preparation_runs_once_for_distinct_children(
    tmp_path: Path,
) -> None:
    _reset_slow_identity()
    transformation = SlowIdentity()
    collection, sweep = _collection(
        tmp_path,
        [
            CompositeAlgorithm(
                transformations=(transformation,), algorithm=SampleMean()
            ),
            CompositeAlgorithm(
                transformations=(transformation,),
                algorithm=OffsetSampleMean(offset=0.25),
            ),
        ],
        seeds=[0],
    )

    returned = collection.run_sweep(sweep)

    assert SlowIdentity.calls == 1
    assert sorted(payload.mean_est.tolist() for payload in returned["payload"]) == [
        [0.5],
        [0.75],
    ]


def test_prepared_leaf_receives_its_representative_time_limit(
    tmp_path: Path,
) -> None:
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=[0],
    )
    offered: list[CellContext] = []

    collection.run_sweep(
        sweep,
        time_limit=5.0,
        schedule_cells=declare_limits_enforced(observing_schedule(offered.append)),
    )

    (leaf,) = [
        context
        for context in offered
        if not isinstance(context.coordinate.algorithm, CompositeAlgorithm)
    ]
    assert leaf.time_limit_s is not None
    assert 0.0 <= leaf.time_limit_s < 5.0


def test_preparation_refusal_answers_every_dependent_outer_cell(
    tmp_path: Path,
) -> None:
    transformation = FailingIdentity()
    collection, sweep = _collection(
        tmp_path,
        [
            CompositeAlgorithm(
                transformations=(transformation,), algorithm=SampleMean()
            ),
            CompositeAlgorithm(
                transformations=(transformation,),
                algorithm=OffsetSampleMean(offset=0.25),
            ),
        ],
        seeds=[0],
    )

    returned = collection.run_sweep(sweep)

    assert list(returned["status"]) == [ResultStatus.INVALID_REGIME.value] * 2


class RefusesPureDP(SampleMean):
    """Refuse every pure-DP budget from the offered family alone."""

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.STANDARD
    )

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if isinstance(budget, PureDPBudget):
            raise InvalidRegime("RefusesPureDP serves no pure-DP budget.")
        return super().run(metadata, input, budget, seed)


def _run_refusing_core_behind(
    tmp_path: Path, transformation: Transformation
) -> tuple[dict[str, Any], list[ResultAttempt]]:
    """Run one refusing core behind a stage; return the outer row and the leaves."""
    workspace = init_workspace(tmp_path / "workspace")
    collection = open_collection(workspace, MeanEstimation)
    collection.add_instance(
        MeanEstimationMetadata(
            n_samples=8,
            domain=BoxDomain(bounds=np.array([[-np.inf, np.inf]])),
        ),
        MeanEstimationInput(X=np.linspace(0.25, 0.75, 8).reshape(-1, 1)),
        {},
    )
    collection.add_algorithm(
        CompositeAlgorithm(transformations=(transformation,), algorithm=RefusesPureDP())
    )
    sweep = collection.save_sweep(
        "refusal", {"budgets": [PureDPBudget(eps=1.0)], "seeds": [0]}
    )
    leaves: list[ResultAttempt] = []

    def schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        def observed(context: CellContext) -> PreparedCell | ResultAttempt:
            outcome = execute(context)
            if isinstance(outcome, ResultAttempt):
                leaves.append(outcome)
            return outcome

        schedule_cells_serially(initial, observed, complete)

    collection.run_sweep(sweep, schedule_cells=schedule)
    (outer,) = collection.load_results(sweep).to_dict("records")
    return outer, leaves


def test_a_core_refusal_after_a_spending_stage_is_the_compositions_bottom(
    tmp_path: Path,
) -> None:
    outer, (leaf,) = _run_refusing_core_behind(tmp_path, ApproxBounds())

    assert outer["status"] is ResultStatus.NO_RELEASE
    assert outer["payload"].startswith("RefusesPureDP refused its regime")
    assert str(leaf.payload) in outer["payload"]
    assert leaf.status is ResultStatus.INVALID_REGIME
    assert leaf.coordinate.budget != PureDPBudget(eps=1.0)


def test_a_core_refusal_after_a_free_stage_stays_an_invalid_regime(
    tmp_path: Path,
) -> None:
    outer, (leaf,) = _run_refusing_core_behind(tmp_path, DataBounds())

    assert outer["status"] is ResultStatus.INVALID_REGIME
    assert outer["payload"] == "RefusesPureDP serves no pure-DP budget."
    assert leaf.status is ResultStatus.INVALID_REGIME
    assert leaf.coordinate.budget == PureDPBudget(eps=1.0)


def test_scheduler_success_is_refused_as_a_preparation_outcome(
    tmp_path: Path,
) -> None:
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=[0],
    )

    def invalid_preparation(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        context = next(iter(initial))
        complete(
            ResultAttempt(
                coordinate=context.coordinate,
                attempt_number=context.attempt_number,
                status=ResultStatus.SUCCESS,
                payload=MeanEstimationOutput(mean_est=np.array([0.5])),
            )
        )

    with pytest.raises(ValueError, match="composite preparation only"):
        collection.run_sweep(sweep, schedule_cells=invalid_preparation)


def test_preparation_scheduler_failure_keeps_completed_outer_attempt_durable(
    tmp_path: Path,
) -> None:
    collection, sweep = _collection(
        tmp_path,
        [
            CompositeAlgorithm(
                transformations=(FailingIdentity(),), algorithm=SampleMean()
            )
        ],
        seeds=[0, 1],
    )

    def fail_after_one(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        complete(execute(next(iter(initial))))
        raise RuntimeError("scheduler failed")

    with pytest.raises(RuntimeError, match="scheduler failed"):
        collection.run_sweep(sweep, schedule_cells=fail_after_one)

    completed = collection.load_results(sweep)
    assert len(completed) == 1
    assert completed.loc[0, "status"] is ResultStatus.INVALID_REGIME


@pytest.mark.parametrize("prepared", [False, True])
def test_preparation_scheduler_failure_never_dispatches_fresh_leaves(
    tmp_path: Path, prepared: bool
) -> None:
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=[0],
    )
    collection.add_algorithm(OffsetSampleMean(offset=0.25))
    calls = 0
    failure = RuntimeError("preparation scheduler stopped")

    def fail_preparation(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("dispatched after scheduler failure")
        if prepared:
            complete(execute(next(iter(initial))))
        raise failure

    with pytest.raises(RuntimeError) as raised:
        collection.run_sweep(sweep, schedule_cells=fail_preparation)

    assert raised.value is failure
    assert calls == 1
    assert len(collection.load_results(sweep)) == 0
    claims = claims_directory(local_task_root(collection))
    assert not (claims.is_dir() and list(claims.iterdir()))


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "malformed"])
def test_preparation_scheduler_outcome_contract_fails_closed(
    tmp_path: Path, failure: str
) -> None:
    collection, sweep = _collection(
        tmp_path,
        [CompositeAlgorithm(transformations=(SlowIdentity(),), algorithm=SampleMean())],
        seeds=[0],
    )

    def invalid_preparation(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        context = next(iter(initial))
        if failure == "missing":
            return
        if failure == "malformed":
            complete({"not": "an outcome"})  # type: ignore[arg-type]
            return
        if failure == "foreign":
            complete(
                ResultAttempt(
                    coordinate=context.coordinate,
                    attempt_number=context.attempt_number + 1,
                    status=ResultStatus.ALGORITHM_ERROR,
                    payload="RuntimeError",
                )
            )
            return
        outcome = execute(context)
        complete(outcome)
        complete(outcome)

    expected = {
        "missing": "completed no outcome",
        "duplicate": "unplanned coordinate or Attempt",
        "foreign": "unplanned coordinate or Attempt",
        "malformed": "must complete ResultAttempt",
    }[failure]
    with pytest.raises((TypeError, ValueError), match=expected):
        collection.run_sweep(sweep, schedule_cells=invalid_preparation)
