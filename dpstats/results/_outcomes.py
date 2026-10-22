"""ResultAttempt values and reconciliation into current Results."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dpstats.transport.canonical import decode_transport
from dpstats.task import Task, TaskOutput
from dpstats.coordinates import (
    CoordinateRecord,
    CellCoordinate,
    coordinate_record_identity,
)

NonnegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
AttemptNumber = Annotated[int, Field(ge=0)]


class ResultStatus(str, Enum):
    """Closed terminal-outcome vocabulary for a ResultAttempt."""

    SUCCESS = "success"
    NO_RELEASE = "no_release"
    INVALID_REGIME = "invalid_regime"
    TIMEOUT = "timeout"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    ALGORITHM_ERROR = "algorithm_error"

    def __str__(self) -> str:
        """Format as the plain wire value; ``repr`` still names the member."""
        return self.value

    @property
    def retryable(self) -> bool:
        """Whether this terminal outcome permits another ResultAttempt."""
        return self in {ResultStatus.TIMEOUT, ResultStatus.MEMORY_LIMIT_EXCEEDED}


def _validate_record_payload(status: ResultStatus, payload: Any) -> None:
    if status is ResultStatus.SUCCESS:
        if not isinstance(payload, dict):
            raise ValueError("success ResultAttempt must retain a JSON Output payload.")
        return
    if status is ResultStatus.NO_RELEASE:
        if not isinstance(payload, str):
            raise ValueError("NoRelease ResultAttempt message must be a string.")
        return
    if status is ResultStatus.INVALID_REGIME:
        if not isinstance(payload, str):
            raise ValueError("InvalidRegime ResultAttempt message must be a string.")
        return
    if status is ResultStatus.ALGORITHM_ERROR:
        if not isinstance(payload, str):
            raise ValueError(
                "algorithm_error ResultAttempt exception type must be a string."
            )
        return
    if payload is not None:
        raise ValueError(f"{status.value} ResultAttempt cannot carry a payload.")


class AttemptRecord(BaseModel):
    """One persisted ResultAttempt row, judged without materializing code.

    This is the Attempt record a reader can validate holding nothing but the bytes:
    the coordinate appears as its stable references, the Output appears as the
    JSON it was released as, and ``status`` decides which payload the row may
    carry. ``ResultAttempt`` is the same observation once the Instance,
    algorithm, and Task it names have been resolved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    coordinate_uuid: str
    coordinate: CoordinateRecord
    attempt_number: AttemptNumber
    # A decoded row carries the terminal outcome as the status's own string
    # value, which is the one place this family is spelled rather than held.
    status: Annotated[ResultStatus, Field(strict=False)]
    payload: dict[str, Any] | str | None
    runtime_s: NonnegativeFinite | None
    peak_rss_mb: NonnegativeFinite | None

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        """Check the stored identity and the payload its status admits agree."""
        if self.coordinate_uuid != coordinate_record_identity(
            self.coordinate.model_dump()
        ):
            raise ValueError(
                f"ResultAttempt coordinate_uuid {self.coordinate_uuid!r} does "
                "not name the coordinate record it carries."
            )
        _validate_record_payload(self.status, self.payload)
        return self


class ResultAttempt(BaseModel):
    """One immutable numbered execution observation of one trial.

    Its coordinate is a CellCoordinate.

    Reconciliation selects one per coordinate as its current Result, a role
    of this value rather than a second type.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    coordinate: CellCoordinate = Field(description="The cell this Attempt executed.")
    attempt_number: AttemptNumber = Field(
        description="This coordinate's numbered execution, starting at 0."
    )
    status: ResultStatus = Field(description="This Attempt's terminal outcome.")
    payload: TaskOutput | str | None = Field(
        default=None,
        description=(
            "The released Output or outcome message that status "
            "discriminates; null for a payload-free outcome."
        ),
    )
    runtime_s: NonnegativeFinite | None = Field(
        default=None, description="Wall-clock execution time in seconds, if measured."
    )
    peak_rss_mb: NonnegativeFinite | None = Field(
        default=None, description="Peak resident memory in MiB, if measured."
    )

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        """Check the payload this outcome admits."""
        if self.status is ResultStatus.SUCCESS:
            if not isinstance(self.payload, self.coordinate.task.Output):
                raise TypeError(
                    "success ResultAttempt must carry the coordinate's TaskOutput."
                )
        elif self.status is ResultStatus.NO_RELEASE:
            if not isinstance(self.payload, str):
                raise ValueError("NoRelease ResultAttempt message must be a string.")
        elif self.status is ResultStatus.INVALID_REGIME:
            if not isinstance(self.payload, str):
                raise ValueError(
                    "InvalidRegime ResultAttempt message must be a string."
                )
        elif self.status is ResultStatus.ALGORITHM_ERROR:
            if not isinstance(self.payload, str):
                raise ValueError(
                    "algorithm_error ResultAttempt message must be a string."
                )
        elif self.payload is not None:
            raise ValueError(
                f"{self.status.value} ResultAttempt cannot carry a payload."
            )

        return self

    @property
    def output(self) -> TaskOutput | None:
        """Return the task Output a successful outcome released."""
        return self.payload if isinstance(self.payload, TaskOutput) else None

    def to_record(self) -> dict[str, Any]:
        """Project this execution observation to its persistence representation.

        Readers join on the stored ``coordinate_uuid``; the coordinate record
        beside it is kept for rehydration alone.
        """
        record: dict[str, Any] = {
            "coordinate_uuid": self.coordinate.uuid,
            "coordinate": self.coordinate.to_record(),
            "attempt_number": self.attempt_number,
            "status": self.status.value,
            "payload": (
                self.payload.model_dump(mode="python")
                if isinstance(self.payload, TaskOutput)
                else self.payload
            ),
            "runtime_s": self.runtime_s,
            "peak_rss_mb": self.peak_rss_mb,
        }
        return record

    @classmethod
    def from_resolved_record(
        cls,
        record: Mapping[str, Any],
        coordinate: CellCoordinate,
        *,
        task: Task,
    ) -> ResultAttempt:
        """Construct one record at the resolved coordinate selected for it."""
        coordinate_uuid = record["coordinate_uuid"]
        raw_payload = record["payload"]
        status = ResultStatus(record["status"])
        attempt_number = record["attempt_number"]
        runtime_s = record["runtime_s"]
        peak_rss_mb = record["peak_rss_mb"]
        if coordinate_uuid != coordinate.uuid:
            raise ValueError(
                f"ResultAttempt names coordinate {coordinate_uuid!r}, not "
                f"the selected coordinate {coordinate.uuid!r}."
            )

        return cls(
            coordinate=coordinate,
            attempt_number=attempt_number,
            status=status,
            payload=load_result_payload(task, status, raw_payload),
            runtime_s=runtime_s,
            peak_rss_mb=peak_rss_mb,
        )


def load_result_payload(
    task: Task, status: ResultStatus, payload: Any
) -> TaskOutput | str | None:
    """Restore and validate an outcome without reconstructing its coordinate."""
    raw = decode_transport(payload)
    _validate_record_payload(status, raw)
    if status is ResultStatus.SUCCESS:
        return task.Output.model_validate(raw)
    return raw
