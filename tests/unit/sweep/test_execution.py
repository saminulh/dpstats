"""Tests for the context one inline cell executes under."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dpstats.budgets import NO_PRIVACY
from dpstats.coordinates import CellCoordinate
from dpstats.instance import Instance
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.results import ResultAttempt, ResultStatus
from dpstats.sweep import CellContext, build_executor_scheduler
from dpstats.sweep._cell import result_attempt
from dpstats.sweep.preparation import PreparedCell
from dpstats.transport.canonical import canonical_json_bytes
from tests.registry_components import SimpleAlgorithm, SimpleMetadata, SimpleOutput


def test_a_negative_attempt_number_is_refused(tmp_path: Path) -> None:
    """A cell context names one numbered observation, which starts at 0."""
    instance = Instance(
        metadata=SimpleMetadata(n_samples=1, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
    )

    with pytest.raises(ValueError, match="attempt_number"):
        CellContext(
            coordinate=CellCoordinate(
                instance=instance,
                algorithm=SimpleAlgorithm(),
                budget=NO_PRIVACY,
                alg_seed=0,
            ),
            task_collection=tmp_path,
            input=tmp_path,
            attempt_number=-1,
            time_limit_s=None,
            memory_limit_bytes=None,
        )


@pytest.mark.parametrize("error_type", [NoRelease, InvalidRegime])
def test_a_lone_surrogate_in_an_authored_message_still_reaches_evidence(
    tmp_path: Path, error_type: type[Exception]
) -> None:
    """A declared outcome keeps its record when its message is ill-formed text."""
    instance = Instance(
        metadata=SimpleMetadata(n_samples=1, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
    )
    context = CellContext(
        coordinate=CellCoordinate(
            instance=instance,
            algorithm=SimpleAlgorithm(),
            budget=NO_PRIVACY,
            alg_seed=0,
        ),
        task_collection=tmp_path,
        input=tmp_path,
        attempt_number=0,
        time_limit_s=None,
        memory_limit_bytes=None,
    )

    attempt = result_attempt(context, error_type("bad \ud800 text"), started_at=0.0)

    assert attempt.status in {ResultStatus.NO_RELEASE, ResultStatus.INVALID_REGIME}
    assert "\ud800" not in str(attempt.payload)
    canonical_json_bytes(attempt.to_record())


def _context(tmp_path: Path, seed: int = 0) -> CellContext:
    return CellContext(
        coordinate=CellCoordinate(
            instance=Instance(
                metadata=SimpleMetadata(n_samples=1, dim=1),
                input_digest="0" * 64,
                metric_arguments={},
            ),
            algorithm=SimpleAlgorithm(),
            budget=NO_PRIVACY,
            alg_seed=seed,
        ),
        task_collection=tmp_path,
        input=tmp_path,
        attempt_number=0,
        time_limit_s=None,
        memory_limit_bytes=None,
    )


def test_an_output_holding_nan_is_an_execution_failure(tmp_path: Path) -> None:
    """NaN is no released value, so its Attempt names the failure type alone."""
    output = SimpleOutput(result=np.array([1.0, math.nan]))

    attempt = result_attempt(_context(tmp_path), output, started_at=0.0)

    assert attempt.status is ResultStatus.ALGORITHM_ERROR
    assert attempt.payload == "ValueError"


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_an_infinite_output_is_a_stored_release(tmp_path: Path, value: float) -> None:
    output = SimpleOutput(result=np.array([1.0, value]))

    attempt = result_attempt(_context(tmp_path), output, started_at=0.0)

    assert attempt.status is ResultStatus.SUCCESS
    assert attempt.payload is output


def test_a_memory_error_under_a_declared_cap_is_a_memory_limit_outcome(
    tmp_path: Path,
) -> None:
    """A cap the adapter enforces makes an allocation refusal a typed outcome."""
    capped = replace(_context(tmp_path), memory_limit_bytes=1024**3)

    attempt = result_attempt(capped, MemoryError(), started_at=0.0)

    assert attempt.status is ResultStatus.MEMORY_LIMIT_EXCEEDED
    assert attempt.payload is None


def test_a_memory_error_without_a_cap_is_an_execution_failure(
    tmp_path: Path,
) -> None:
    """With no cap declared, a MemoryError is the ordinary failure it names."""
    attempt = result_attempt(_context(tmp_path), MemoryError(), started_at=0.0)

    assert attempt.status is ResultStatus.ALGORITHM_ERROR
    assert attempt.payload == "MemoryError"


def test_executor_scheduler_abandons_a_raising_job_and_finishes_its_siblings(
    tmp_path: Path,
) -> None:
    """A raising future is machinery: no Attempt for it, siblings finish, then it is re-raised."""
    contexts = [_context(tmp_path, seed) for seed in range(4)]
    executed: list[int] = []
    completed: list[ResultAttempt] = []
    failure_seen = threading.Event()

    def execute(context: CellContext) -> PreparedCell | ResultAttempt:
        executed.append(context.coordinate.alg_seed)
        if context.coordinate.alg_seed == 1:
            failure_seen.set()
            raise RuntimeError("worker lost")
        assert failure_seen.wait(timeout=5.0)
        return result_attempt(context, NoRelease("declined"), started_at=0.0)

    def complete(outcome: PreparedCell | ResultAttempt) -> list[CellContext]:
        assert isinstance(outcome, ResultAttempt)
        completed.append(outcome)
        return []

    with ThreadPoolExecutor(max_workers=3) as pool:
        with pytest.raises(RuntimeError, match="worker lost"):
            build_executor_scheduler(pool, 3)(iter(contexts), execute, complete)

    assert sorted(executed) == [0, 1, 2]
    assert sorted(attempt.coordinate.alg_seed for attempt in completed) == [0, 2]
    assert {attempt.status for attempt in completed} == {ResultStatus.NO_RELEASE}
