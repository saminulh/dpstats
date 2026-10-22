"""Execute one Sweep cell and classify its complete numbered outcome."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dpstats.exceptions import NoRelease
from dpstats.exceptions import InvalidRegime
from dpstats.instance import load_instance_input
from dpstats.task import TaskInput, TaskOutput
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.results import ResultAttempt, ResultStatus
from dpstats.coordinates import CellCoordinate

logger = logging.getLogger("benchmark")


@dataclass(frozen=True, slots=True, eq=False)
class CellContext:
    """Everything needed to execute one resolved coordinate once.

    ``input`` is the shared Input-artifact root for a cell of an original
    Instance, which the execute step loads where the cell runs, or the live
    prepared Input of a leaf. Sweep planning starts with contexts for selected
    outer cells and offers its scheduler composite anchors for preparation, then
    unique missing prepared leaves for execution. This is transient operational
    state, not scientific identity or a persisted Attempt record. Resource limits are
    declarations for an adapter that chooses to enforce or observe them; the
    library's serial scheduler deliberately executes inline without enforcing either
    one.
    """

    coordinate: CellCoordinate
    task_collection: Path
    input: TaskInput | Path = field(repr=False)
    attempt_number: int
    time_limit_s: float | None
    memory_limit_bytes: int | None

    def __post_init__(self) -> None:
        if self.attempt_number < 0:
            raise ValueError("CellContext attempt_number must be nonnegative.")

    def resolve_input(self) -> TaskInput:
        """Return the live Input, loading its artifact when this holds a root."""
        if isinstance(self.input, Path):
            return load_instance_input(self.coordinate.instance, self.input)
        return self.input


def _encodable_message(message: str) -> str:
    """Replace a lone surrogate so an authored message reaches canonical JSON.

    NoRelease and InvalidRegime are declared outcomes, not crashes, so ill-formed
    authored text degrades the message instead of losing the outcome.
    """
    return message.encode("utf-8", errors="replace").decode("utf-8")


def _error_type_name(error: Exception) -> str:
    """Name one failure's exception type, taking nothing from its message.

    This name is recorded as a Result and is published with it, so it is
    derived from the class alone and never from the exception's text: a
    message can quote values read from private input, and a name that carries
    no message is data-free by construction rather than by redaction. The
    defining module qualifies a type the interpreter does not provide, so two
    libraries' identically named errors stay distinct.
    """
    error_type = type(error)
    if error_type.__module__ == "builtins":
        return error_type.__qualname__
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _holds_nan(value: Any) -> bool:
    """Report whether a dumped Output holds a NaN float at any depth.

    NaN is no released value, while an infinite value is one and is stored.
    """
    if isinstance(value, float):
        return math.isnan(value)
    # isinstance narrows a dumped container to unknown item types; the casts
    # restate that its items are arbitrary dumped values.
    if isinstance(value, dict):
        return any(_holds_nan(item) for item in cast("dict[Any, Any]", value).values())
    if isinstance(value, (list, tuple)):
        return any(_holds_nan(item) for item in cast("Sequence[Any]", value))
    return False


def validate_output(output: TaskOutput) -> None:
    """Require a serializable released Output with no NaN values."""
    output_fields = output.model_dump(mode="python")
    canonical_json_bytes(output_fields)
    if _holds_nan(output_fields):
        raise ValueError("The Output holds a NaN value.")


def result_attempt(
    context: CellContext,
    outcome: TaskOutput | ResultAttempt | Exception,
    *,
    started_at: float,
) -> ResultAttempt:
    """Classify one invocation outcome under the context's outer coordinate.

    A ``MemoryError`` is ``memory_limit_exceeded`` only when the context
    declares a cap, which an adapter enforces; with no cap it is the ordinary
    execution failure it names.
    """
    coordinate = context.coordinate
    algorithm_name = type(coordinate.algorithm).__name__
    status: ResultStatus
    payload: TaskOutput | str | None

    if isinstance(outcome, TaskOutput) and not isinstance(
        outcome, coordinate.task.Output
    ):
        outcome = TypeError("Execution returned the wrong TaskOutput type.")

    if isinstance(outcome, NoRelease):
        logger.info("Algorithm %s released bottom: %s", algorithm_name, str(outcome))
        status = ResultStatus.NO_RELEASE
        payload = _encodable_message(str(outcome))
    elif isinstance(outcome, InvalidRegime):
        logger.info(
            "Algorithm %s refused the budget regime: %s",
            algorithm_name,
            str(outcome),
        )
        status = ResultStatus.INVALID_REGIME
        payload = _encodable_message(str(outcome))
    elif isinstance(outcome, MemoryError) and context.memory_limit_bytes is not None:
        logger.info(
            "Algorithm %s exceeded its declared memory cap with seed %s.",
            algorithm_name,
            coordinate.alg_seed,
        )
        status = ResultStatus.MEMORY_LIMIT_EXCEEDED
        payload = None
    elif isinstance(outcome, Exception):
        logger.error(
            "Error running %s with seed %s: %s",
            algorithm_name,
            coordinate.alg_seed,
            outcome,
        )
        status = ResultStatus.ALGORITHM_ERROR
        payload = _error_type_name(outcome)
    elif isinstance(outcome, ResultAttempt):
        status = outcome.status
        payload = outcome.payload
    else:
        try:
            validate_output(outcome)
        except Exception as error:
            logger.error(
                "Invalid output from %s with seed %s: %s",
                algorithm_name,
                coordinate.alg_seed,
                error,
            )
            status = ResultStatus.ALGORITHM_ERROR
            payload = _error_type_name(error)
        else:
            status = ResultStatus.SUCCESS
            payload = outcome

    return ResultAttempt(
        coordinate=coordinate,
        attempt_number=context.attempt_number,
        status=status,
        payload=payload,
        runtime_s=time.perf_counter() - started_at,
        peak_rss_mb=None,
    )


def run_coordinate(context: CellContext) -> ResultAttempt:
    """Execute one coordinate and return its complete numbered ResultAttempt.

    The caller owns any durable claim on this context. Keeping claim ownership
    at the orchestration boundary lets a Sweep execute an ephemeral prepared
    coordinate while recording only its official outer Attempts.

    This function deliberately runs in its caller's process. A Sweep scheduler or
    external scheduler owns any process placement, cancellation, or resource
    enforcement around it.
    """
    started_at = time.perf_counter()
    coordinate = context.coordinate
    input = context.resolve_input()
    try:
        outcome = coordinate.algorithm.run(
            metadata=coordinate.instance.metadata,
            input=input,
            budget=coordinate.budget,
            seed=coordinate.alg_seed,
        )
    except Exception as error:
        outcome = error
    return result_attempt(context, outcome, started_at=started_at)
