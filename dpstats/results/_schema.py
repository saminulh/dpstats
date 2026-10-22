"""Column vocabulary and typed schemas of one Task's Attempt records and Results.

Canonical records are laid out as rows here and rebuilt from them; nothing here
opens a file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pyarrow as pa

from dpstats.coordinates import FACTOR_KEYS
from dpstats.results._outcomes import ResultStatus
from dpstats.task import ARRAY_COLUMN_SHAPES, Task
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)

COORDINATE_COLUMN = "coordinate_uuid"
PARTITION_COLUMN = "alg_id"
COORDINATE_FIELD = pa.field(COORDINATE_COLUMN, pa.string(), nullable=False)
KEY_COLUMNS = (*FACTOR_KEYS, "seed")
_KEY_FIELDS = (
    pa.field("instance_id", pa.string(), nullable=False),
    pa.field(PARTITION_COLUMN, pa.string(), nullable=False),
    pa.field("budget_path", pa.string(), nullable=False),
    pa.field("budget_config", pa.string(), nullable=False),
    pa.field("seed", pa.uint64(), nullable=False),
)
ATTEMPT_FIELD = pa.field("attempt", pa.int32(), nullable=False)
_OUTCOME_FIELDS = (
    pa.field("status", pa.string(), nullable=False),
    pa.field("runtime_s", pa.float64()),
    pa.field("peak_rss_mb", pa.float64()),
    pa.field("reason", pa.string()),
)
FIXED_FIELDS = (*_KEY_FIELDS, ATTEMPT_FIELD, *_OUTCOME_FIELDS)
RESULT_FIELDS = (*_KEY_FIELDS, *_OUTCOME_FIELDS)
_SCALAR_COLUMN_TYPES: tuple[tuple[Any, pa.DataType], ...] = (
    (float, pa.float64()),
    (int, pa.int64()),
    (bool, pa.bool_()),
    (str, pa.string()),
)
_ARRAY_VALUE_TYPES: dict[str, pa.DataType] = {
    "floating": pa.float64(),
    "integer": pa.int64(),
    "boolean": pa.bool_(),
}
_MESSAGE_STATUSES = (
    ResultStatus.NO_RELEASE,
    ResultStatus.INVALID_REGIME,
    ResultStatus.ALGORITHM_ERROR,
)


def attempt_schema(task: Task) -> pa.Schema:
    """Return the partition-file schema of one Task's attempts.

    The fixed columns come first, then one column per field of the Task's
    Output: a fixed-rank array alias becomes a list nested to its rank over
    its dtype family, and ``float``, ``int``, ``bool``, and ``str`` become the
    matching scalar type. ``alg_id`` is a column, repeated by the partition.
    """
    return _task_schema(task, FIXED_FIELDS)


def result_schema(task: Task) -> pa.Schema:
    """Return the released-file schema of one Task's Results.

    This is the attempt schema without ``attempt``: a release ships one row per
    cell, and an attempt number means nothing outside the store that minted it.
    """
    return _task_schema(task, RESULT_FIELDS)


def _task_schema(task: Task, fixed: Sequence[pa.Field[pa.DataType]]) -> pa.Schema:
    """Lay the given fixed columns out ahead of one Task's Output columns."""
    fields: list[pa.Field[pa.DataType]] = list(fixed)
    reserved = {field.name for field in FIXED_FIELDS} | {COORDINATE_COLUMN}
    for name, info in task.Output.model_fields.items():
        if name in reserved:
            raise ValueError(
                f"Output field {name!r} of Task {task.id!r} collides with an "
                "attempt column."
            )
        annotation = info.rebuild_annotation()
        column = _column_type(annotation)
        if column is None:
            raise ValueError(
                f"Output field {name!r} of Task {task.id!r} has no attempt column "
                f"type for {annotation}."
            )
        fields.append(pa.field(name, column))
    return pa.schema(fields)


def _column_type(annotation: Any) -> pa.DataType | None:
    for scalar, column in _SCALAR_COLUMN_TYPES:
        if annotation == scalar:
            return column
    for alias, (family, rank) in ARRAY_COLUMN_SHAPES.items():
        if annotation == alias:
            column = _ARRAY_VALUE_TYPES[family]
            for _ in range(rank):
                column = pa.list_(column)
            return column
    return None


def records_to_table(records: Iterable[Mapping[str, Any]], task: Task) -> pa.Table:
    """Lay canonical ResultAttempt records out as the rows a partition file holds.

    The payload columns are the row's one persisted meaning of ``status``: the
    Output columns hold a ``success`` release, and ``reason`` holds a
    ``no_release`` or ``invalid_regime`` message or an ``algorithm_error``
    exception type name. A coordinate's identity is not a column: a view joins
    it in from the selection that already carries it.
    """
    schema = attempt_schema(task)
    output_names = tuple(task.Output.model_fields)
    columns: dict[str, list[Any]] = {name: [] for name in schema.names}
    for record in records:
        status = ResultStatus(record["status"])
        output, reason = _row_payload(status, record["payload"])
        coordinate = record["coordinate"]
        budget = coordinate["budget"]
        columns["instance_id"].append(coordinate["instance_uuid"])
        columns[PARTITION_COLUMN].append(coordinate["alg_uuid"])
        columns["budget_path"].append(budget["path"])
        columns["budget_config"].append(
            canonical_json_bytes(budget["config"]).decode("ascii")
        )
        columns["seed"].append(coordinate["alg_seed"])
        columns["attempt"].append(record["attempt_number"])
        columns["status"].append(status.value)
        columns["runtime_s"].append(record["runtime_s"])
        columns["peak_rss_mb"].append(record["peak_rss_mb"])
        columns["reason"].append(reason)
        for name in output_names:
            columns[name].append(None if output is None else output[name])
    return pa.table(columns, schema=schema)


def _row_payload(
    status: ResultStatus, payload: Any
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Split one record's payload into its Output values and its reason."""
    if status is ResultStatus.SUCCESS:
        return payload, None
    if status in _MESSAGE_STATUSES:
        return None, payload
    return None, None


def table_to_records(table: pa.Table) -> list[dict[str, Any]]:
    """Restore canonical ResultAttempt records from complete rows.

    The rows must carry every attempt column beside ``coordinate_uuid``, as a
    full ``view`` returns them; the coordinate record is rebuilt from
    the key columns and the identity is the one the view joined in, so nothing
    is derived here.
    """
    fixed = {
        COORDINATE_COLUMN,
        PARTITION_COLUMN,
        *(field.name for field in FIXED_FIELDS),
    }
    output_names = [name for name in table.column_names if name not in fixed]
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = table.to_pylist()
    for row in rows:
        status = ResultStatus(row["status"])
        output = {name: row[name] for name in output_names}
        records.append(
            {
                "coordinate_uuid": row[COORDINATE_COLUMN],
                "coordinate": {
                    "instance_uuid": row["instance_id"],
                    "alg_uuid": row[PARTITION_COLUMN],
                    "budget": {
                        "path": row["budget_path"],
                        "config": decode_json_document(
                            row["budget_config"].encode("ascii")
                        ),
                    },
                    "alg_seed": row["seed"],
                },
                "attempt_number": row["attempt"],
                "status": status.value,
                "payload": _record_payload(status, output, row["reason"]),
                "runtime_s": row["runtime_s"],
                "peak_rss_mb": row["peak_rss_mb"],
            }
        )
    return records


def _record_payload(
    status: ResultStatus, output: dict[str, Any], reason: str | None
) -> Any:
    """Rebuild one record's payload from its row columns."""
    if status is ResultStatus.SUCCESS:
        return output
    if status in _MESSAGE_STATUSES:
        return reason
    return None
