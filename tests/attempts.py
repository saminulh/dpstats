"""Test-only construction of ResultAttempt and released Result records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dpstats.coordinates import CellCoordinate, coordinate_record_identity
from dpstats.sweep._claims import claim_attempt
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.workspace.layout import (
    PARTITION_PREFIX,
    attempt_partition_directory,
    attempts_directory,
    result_partition_directory,
)

ARRAY_OUTPUT_COLUMNS: dict[str, pa.DataType] = {"result": pa.list_(pa.float64())}

# The stored Attempt columns, spelled out independently of the production
# schema so a reader or writer that drifts from them is caught.
ATTEMPT_FIXED_FIELDS = [
    pa.field("instance_id", pa.string(), nullable=False),
    pa.field("alg_id", pa.string(), nullable=False),
    pa.field("budget_path", pa.string(), nullable=False),
    pa.field("budget_config", pa.string(), nullable=False),
    pa.field("seed", pa.uint64(), nullable=False),
    pa.field("attempt", pa.int32(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("runtime_s", pa.float64()),
    pa.field("peak_rss_mb", pa.float64()),
    pa.field("reason", pa.string()),
]
_RESULT_FIELDS = [field for field in ATTEMPT_FIXED_FIELDS if field.name != "attempt"]
_MESSAGE_STATUSES = {"no_release", "invalid_regime", "algorithm_error"}


def current_result(
    *,
    instance_uuid: str,
    alg_uuid: str,
    budget: Mapping[str, Any],
    status: str = "success",
    estimate: float = 1.0,
    alg_seed: int = 0,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one surfaced ResultAttempt record without the production encoder.

    The stored coordinate identity comes from the library's own derivation
    because a record whose token disagrees with its coordinate is refused at
    decode; everything else here is assembled independently.
    """
    if status == "success" and output is None:
        output = {"result": [estimate]}

    if status == "success":
        payload: dict[str, Any] | str | None = output
    elif status == "no_release":
        payload = "coordinate released the accounted bottom outcome"
    elif status == "invalid_regime":
        payload = "coordinate refused a budget family it does not realize"
    elif status == "algorithm_error":
        payload = "ValueError"
    else:
        payload = None

    coordinate = {
        "instance_uuid": instance_uuid,
        "alg_uuid": alg_uuid,
        "budget": dict(budget),
        "alg_seed": alg_seed,
    }
    return {
        "coordinate_uuid": coordinate_record_identity(coordinate),
        "coordinate": coordinate,
        "attempt_number": 0,
        "status": status,
        "payload": payload,
        "runtime_s": 0.5,
        "peak_rss_mb": 2.0,
    }


def _attempt_row(
    record: Mapping[str, Any], output_columns: Mapping[str, pa.DataType]
) -> dict[str, Any]:
    """Lay one ResultAttempt record out as its stored columns.

    ``status`` decides where the record's payload goes: an output-bearing
    outcome fills the Output columns, and every other explanation is text in
    ``reason``.
    """
    status = record["status"]
    payload = record["payload"]
    output: Mapping[str, Any] | None = None
    reason: str | None = None
    if status == "success":
        output = payload
    elif status in _MESSAGE_STATUSES:
        reason = payload
    coordinate = record["coordinate"]
    budget = coordinate["budget"]
    row: dict[str, Any] = {
        "instance_id": coordinate["instance_uuid"],
        "alg_id": coordinate["alg_uuid"],
        "budget_path": budget["path"],
        "budget_config": canonical_json_bytes(budget["config"]).decode("ascii"),
        "seed": coordinate["alg_seed"],
        "attempt": record["attempt_number"],
        "status": status,
        "runtime_s": record["runtime_s"],
        "peak_rss_mb": record["peak_rss_mb"],
        "reason": reason,
    }
    for name in output_columns:
        row[name] = None if output is None else output[name]
    return row


def write_attempt_rows(
    task_store: Path,
    alg_uuid: str,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    output_columns: Mapping[str, pa.DataType] = ARRAY_OUTPUT_COLUMNS,
) -> Path:
    """Write stored-column rows as one file in *alg_uuid*'s Attempt partition.

    The rows are written with pyarrow under the independent schema above, so a
    test may lay out any row, including one a production writer would refuse.
    """
    schema = pa.schema(
        [*ATTEMPT_FIXED_FIELDS, *(pa.field(n, t) for n, t in output_columns.items())]
    )
    directory = attempt_partition_directory(task_store, alg_uuid)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path)
    return path


def write_attempt_partition(
    task_store: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    output_columns: Mapping[str, pa.DataType] = ARRAY_OUTPUT_COLUMNS,
    name: str = "fixture",
) -> list[Path]:
    """Write ResultAttempt records with pyarrow, independent of the writer.

    Each algorithm's rows become one file in its own partition, which is the
    layout a reader expects; the columns are assembled here rather than by the
    production encoder.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["coordinate"]["alg_uuid"], []).append(
            _attempt_row(record, output_columns)
        )
    return [
        write_attempt_rows(
            task_store, alg_uuid, name, rows, output_columns=output_columns
        )
        for alg_uuid, rows in sorted(grouped.items())
    ]


def write_result_partition(
    task_store: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    output_columns: Mapping[str, pa.DataType] = ARRAY_OUTPUT_COLUMNS,
    name: str = "results",
) -> list[Path]:
    """Write released Result records with pyarrow, independent of the writer.

    This is the release layout: one file per algorithm under ``results/``,
    holding the same columns a stored Attempt does apart from its number, which
    no reader outside the recording store could mean anything by.
    """
    schema = pa.schema(
        [*_RESULT_FIELDS, *(pa.field(n, t) for n, t in output_columns.items())]
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        row = _attempt_row(record, output_columns)
        del row["attempt"]
        grouped.setdefault(record["coordinate"]["alg_uuid"], []).append(row)
    written: list[Path] = []
    for alg_uuid, rows in sorted(grouped.items()):
        directory = result_partition_directory(task_store, alg_uuid)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        written.append(path)
    return written


def abandon_claim(
    task_store: Path, coordinate: CellCoordinate, attempt_number: int = 0
) -> None:
    """Leave the claim a runner that vanished mid-attempt would have left.

    No public verb stops a runner between its claim and its outcome, so this
    is the one place tests reach the claim writer to stage that state.
    """
    claim_attempt(task_store, coordinate, attempt_number)


def unsealed_journals(task_store: Path) -> list[Path]:
    """List the Attempt journals a writer has not sealed into partition files."""
    return sorted(attempts_directory(task_store).glob(".incomplete-*.jsonl"))


def read_partition_rows(partition: Path) -> list[dict[str, Any]]:
    """Read one partition's published files back with pyarrow, in name order.

    A staged ``.incomplete-`` file is not yet a record, so it is skipped.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(partition.glob("*.parquet")):
        if not path.name.startswith(".incomplete-"):
            rows.extend(pq.read_table(path).to_pylist())
    return rows


def read_attempt_rows(task_store: Path) -> list[dict[str, Any]]:
    """Read every stored Attempt row back with pyarrow, ordered by partition."""
    rows: list[dict[str, Any]] = []
    for partition in sorted(
        attempts_directory(task_store).glob(f"{PARTITION_PREFIX}*")
    ):
        rows.extend(read_partition_rows(partition))
    return rows
