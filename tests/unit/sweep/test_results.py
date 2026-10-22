"""Tests for the ResultAttempt record codec and its terminal-outcome vocabulary."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dpstats.instance import Instance
from dpstats.budgets import PureDPBudget
from dpstats.workspace import TaskCollection
from dpstats.results import ResultAttempt, ResultStatus
from dpstats.results._outcomes import AttemptRecord
from dpstats.coordinates import CellCoordinate
from dpstats.transport.canonical import (
    canonical_json_bytes,
    coordinate_identity,
)
from tests.attempts import current_result, write_attempt_partition
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleOutput,
    sized_producer,
)
from tests.sweep_helpers import local_task_root, sweep_coordinates, write_sweep


def _coordinate(*, alg_seed: int = 0) -> CellCoordinate:
    instance = Instance(
        metadata=SIMPLE_TASK.Metadata(n_samples=1, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
    )
    return CellCoordinate(
        instance=instance,
        algorithm=SimpleAlgorithm(),
        budget=PureDPBudget(eps=1.0),
        alg_seed=alg_seed,
    )


def _success_record() -> dict[str, Any]:
    return ResultAttempt(
        coordinate=_coordinate(),
        attempt_number=0,
        status=ResultStatus.SUCCESS,
        payload=SimpleOutput(result=np.array([1.0])),
    ).to_record()


def test_the_persisted_status_vocabulary_and_which_outcomes_may_be_retried() -> None:
    """The values are the wire spelling; only resource failures are retryable."""
    assert {status.value: status.retryable for status in ResultStatus} == {
        "success": False,
        "no_release": False,
        "invalid_regime": False,
        "timeout": True,
        "memory_limit_exceeded": True,
        "algorithm_error": False,
    }


def test_str_and_f_string_formatting_yield_the_plain_value() -> None:
    """A reader printing a status sees its wire spelling, not the enum path."""
    status = ResultStatus.SUCCESS
    assert str(status) == "success"
    assert f"{status}" == "success"
    assert "{}".format(status) == "success"
    assert status.value == "success"
    assert repr(status) == "<ResultStatus.SUCCESS: 'success'>"


@pytest.mark.parametrize(
    ("status", "payload", "runtime_s", "peak_rss_mb"),
    [
        pytest.param(
            ResultStatus.SUCCESS, (1.5, -0.0), 0.25, 64.0, id="success-finite"
        ),
        pytest.param(
            ResultStatus.SUCCESS,
            (-math.inf, math.nan),
            0.25,
            None,
            id="success-nonfinite",
        ),
        pytest.param(
            ResultStatus.NO_RELEASE, "PTR declined release", 0.25, 1.0, id="no-release"
        ),
        pytest.param(
            ResultStatus.NO_RELEASE, "", None, None, id="no-release-without-prose"
        ),
        pytest.param(
            ResultStatus.INVALID_REGIME,
            "This algorithm realizes no implication for the offered family.",
            None,
            None,
            id="invalid-regime",
        ),
        pytest.param(
            ResultStatus.ALGORITHM_ERROR,
            "numpy.linalg.LinAlgError",
            0.5,
            None,
            id="algorithm-error-kind",
        ),
        pytest.param(ResultStatus.TIMEOUT, None, None, None, id="timeout"),
        pytest.param(
            ResultStatus.MEMORY_LIMIT_EXCEEDED, None, None, 2.0, id="memory-limit"
        ),
    ],
)
def test_every_outcome_round_trips_to_the_same_wire_bytes(
    status: ResultStatus,
    payload: tuple[float, ...] | str | None,
    runtime_s: float | None,
    peak_rss_mb: float | None,
) -> None:
    """The record is the wire, so neither reading nor writing may restate it.

    A chunk is named by the digest of these bytes and every join keys on the
    identity derived from them, so a codec that normalized a value on the way
    in or out would rename stored history rather than read it.
    """
    coordinate = _coordinate()
    released = (
        SimpleOutput(result=np.array(payload)) if isinstance(payload, tuple) else None
    )
    attempt = ResultAttempt(
        coordinate=coordinate,
        attempt_number=3,
        status=status,
        payload=payload if released is None else released,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
    )

    record = attempt.to_record()
    wire = canonical_json_bytes(record)
    parsed = AttemptRecord.model_validate(record)
    rehydrated = ResultAttempt.from_resolved_record(
        record, coordinate, task=SIMPLE_TASK
    )

    assert set(record) == {
        "coordinate_uuid",
        "coordinate",
        "attempt_number",
        "status",
        "payload",
        "runtime_s",
        "peak_rss_mb",
    }
    assert record["coordinate_uuid"] == coordinate.uuid
    assert record["coordinate"] == coordinate.to_record()
    assert record["status"] == status.value
    assert canonical_json_bytes(record) == wire
    assert canonical_json_bytes(parsed.coordinate.model_dump()) == (
        canonical_json_bytes(record["coordinate"])
    )
    assert canonical_json_bytes(rehydrated.to_record()) == wire
    assert rehydrated.coordinate.instance is coordinate.instance
    assert rehydrated.coordinate.algorithm is coordinate.algorithm
    if released is None:
        assert record["payload"] == payload
        assert rehydrated.payload == payload
        assert rehydrated.output is None
    else:
        assert isinstance(rehydrated.output, SimpleOutput)
        np.testing.assert_array_equal(rehydrated.output.result, released.result)


def test_the_record_validator_judges_structure_without_resolving_references() -> None:
    """A reader holding only bytes can accept a row naming nothing installed."""
    record = ResultAttempt(
        coordinate=_coordinate(), attempt_number=0, status=ResultStatus.TIMEOUT
    ).to_record()
    unresolvable = {
        **record["coordinate"],
        "alg_uuid": "unresolvable-but-structurally-valid",
    }
    record["coordinate"] = unresolvable
    record["coordinate_uuid"] = coordinate_identity(unresolvable)

    parsed = AttemptRecord.model_validate(record)

    assert parsed.coordinate.alg_uuid == "unresolvable-but-structurally-valid"


_DROPPED = object()


def _mutated(**changes: Any) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        record = _success_record()
        for key, value in changes.items():
            if value is _DROPPED:
                del record[key]
            else:
                record[key] = value
        return record

    return build


@pytest.mark.parametrize(
    ("build", "named"),
    [
        pytest.param(
            _mutated(status="definitely_not_a_status"),
            "definitely_not_a_status",
            id="unknown-status",
        ),
        pytest.param(
            _mutated(coordinate_uuid=_coordinate(alg_seed=7).uuid),
            _coordinate(alg_seed=7).uuid,
            id="identity-disagrees",
        ),
        pytest.param(
            _mutated(coordinate_uuid=_DROPPED), "coordinate_uuid", id="identity-missing"
        ),
        pytest.param(
            _mutated(status="invalid_regime", payload=None),
            "InvalidRegime",
            id="regime-without-message",
        ),
        pytest.param(
            _mutated(status="timeout", payload="timed out"),
            "cannot carry",
            id="payload-on-a-payload-free-outcome",
        ),
    ],
)
def test_the_record_validator_names_what_it_refuses(
    build: Callable[[], dict[str, Any]], named: str
) -> None:
    """Each refusal names the offending value or the condition it violates."""
    with pytest.raises(ValueError, match=named):
        AttemptRecord.model_validate(build())


@pytest.mark.parametrize(
    ("status", "payload", "error", "named"),
    [
        pytest.param(
            ResultStatus.SUCCESS, "not an Output", TypeError, "TaskOutput", id="success"
        ),
        pytest.param(
            ResultStatus.TIMEOUT, "timed out", ValueError, "cannot carry", id="timeout"
        ),
    ],
)
def test_an_attempt_refuses_a_payload_its_status_does_not_admit(
    status: ResultStatus, payload: str, error: type[Exception], named: str
) -> None:
    with pytest.raises(error, match=named):
        ResultAttempt(
            coordinate=_coordinate(),
            attempt_number=0,
            status=status,
            payload=payload,
        )


def test_result_loading_restores_a_stored_nonfinite_output(
    collection: TaskCollection,
) -> None:
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, recipe)
    record = current_result(
        instance_uuid=coordinate.instance_uuid,
        alg_uuid=coordinate.alg_uuid,
        budget=coordinate.budget.dump_config().model_dump(),
    )
    record.update(status="success", payload={"result": [math.nan]})
    write_attempt_partition(local_task_root(collection), [record])

    loaded = collection.load_results().iloc[0]

    assert loaded["status"] is ResultStatus.SUCCESS
    assert isinstance(loaded["payload"], SimpleOutput)
    assert math.isnan(loaded["payload"].result[0])
