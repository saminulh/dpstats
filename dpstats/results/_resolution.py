"""One row per selected coordinate, from whichever shape a root holds.

An Attempt store's latest non-retryable Attempt wins at each cell; repeats
of one numbered Attempt are folded when they agree on every stored column
and refused when they do not. A release answers each cell with its one
stored row, refused if it answers a cell more than once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget
from dpstats.coordinates import CellCoordinate, coordinate_record_identity
from dpstats.instance import Instance
from dpstats.results._outcomes import ResultStatus
from dpstats.results._partitions import (
    attempt_store,
    scan_partitions,
    store_shape,
)
from dpstats.results._schema import (
    ATTEMPT_FIELD,
    COORDINATE_COLUMN,
    COORDINATE_FIELD,
    KEY_COLUMNS,
    PARTITION_COLUMN,
    table_to_records,
)
from dpstats.results._selection import (
    ROW_INDEX,
    SELECTION_ROW,
    canonical_budget_keys,
    empty_selection,
    factor_selection,
    initial_factors,
    keyed_rows,
    require_columns,
    selection_table,
)
from dpstats.task import Task
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.workspace.layout import attempts_directory

_RETRYABLE = "_retryable"
_HIGHEST_ATTEMPT = "_highest_attempt"
_RETRYABLE_STATUSES = pa.array(
    [status.value for status in ResultStatus if status.retryable], pa.string()
)


def view(
    task_root: Path,
    task: Task,
    coordinates: Iterable[CellCoordinate],
    *,
    columns: Sequence[str],
) -> pa.Table:
    """Return the Result each selected coordinate currently holds, if any.

    This is whichever store shape *task_root* holds, read with only the
    needed columns. The selection joins on the natural key, so every returned
    row carries its coordinate's identity without deriving it, and a
    coordinate with no matching row has no row. An Attempt store's latest
    non-retryable ``attempt`` per coordinate wins, and a retryable one counts
    only when no non-retryable Attempt stands; rows repeating one numbered
    attempt are folded when they agree on every stored column and refused
    when they do not. A release is refused if it answers one coordinate with
    more than one row.
    """
    selection, rows = _resolved_selection(task_root, task, coordinates, columns=columns)
    return (
        rows.select(list(columns))
        .add_column(
            0,
            COORDINATE_FIELD,
            selection[COORDINATE_COLUMN].take(rows[SELECTION_ROW]),
        )
        .sort_by(COORDINATE_COLUMN)
    )


def _resolved_selection(
    task_root: Path,
    task: Task,
    coordinates: Iterable[CellCoordinate],
    *,
    columns: Sequence[str],
) -> tuple[pa.Table, pa.Table]:
    """Return the coordinate selection beside the current rows it resolved."""
    selection = (
        selection_table(coordinates)
        .group_by([*KEY_COLUMNS, COORDINATE_COLUMN])
        .aggregate([])
    )
    return selection, selected_attempts(task_root, task, selection, columns=columns)


def current_outcomes(
    task_root: Path, task: Task, coordinates: Iterable[CellCoordinate]
) -> dict[str, tuple[int, ResultStatus]]:
    """Return each answered coordinate's next attempt number and current status.

    Deciding what to execute needs no payload, so the selected partitions are
    opened for the one column this reads. Refuses a release, which carries no
    Attempt history to continue from.
    """
    attempt_store(task_root)
    selection, rows = _resolved_selection(
        task_root, task, coordinates, columns=["status"]
    )
    answered = selection[COORDINATE_COLUMN].take(rows[SELECTION_ROW])
    return {
        coordinate_uuid: (highest + 1, ResultStatus(status))
        for coordinate_uuid, highest, status in zip(
            cast("list[str]", answered.to_pylist()),
            cast("list[int]", rows[_HIGHEST_ATTEMPT].to_pylist()),
            cast("list[str]", rows["status"].to_pylist()),
        )
    }


def selected_attempts(
    task_root: Path, task: Task, selection: pa.Table, *, columns: Sequence[str]
) -> pa.Table:
    """Read the current row of each keyed selection row, from whichever shape the root holds.

    Return the selection row index beside the requested columns. Both
    runtime-coordinate reads and columnar previews use this resolution,
    against either an Attempt store or a release. Conflict checks compare
    complete stored rows, even for projections.
    """
    store, stored = store_shape(task_root, task)
    history = ATTEMPT_FIELD.name in stored.names
    require_columns(task, stored, columns)
    partitions = pc.unique(selection[PARTITION_COLUMN])
    needed = list(
        dict.fromkeys(
            [*KEY_COLUMNS, *(("attempt", "status") if history else ()), *columns]
        )
    )
    rows = _scan_instances(
        store, stored, partitions, needed, pc.unique(selection["instance_id"])
    )
    keyed = keyed_rows(rows, "attempt") if history else keyed_rows(rows)
    selected = keyed.join(
        selection.select(list(KEY_COLUMNS)).append_column(
            SELECTION_ROW, pa.array(np.arange(selection.num_rows), pa.int64())
        ),
        keys=list(KEY_COLUMNS),
        join_type="inner",
    )
    return _one_row_per_cell(task_root, task, stored, rows, selected, columns=columns)


def _scan_instances(
    store: Path,
    stored: pa.Schema,
    partitions: pa.Array[Any] | pa.ChunkedArray[Any],
    columns: Sequence[str],
    instances: pa.Array[Any] | pa.ChunkedArray[Any],
) -> pa.Table:
    """Read the selected partitions, keeping only the selected Instances' rows."""
    rows = scan_partitions(store, stored, partitions, columns)
    return rows.filter(pc.is_in(rows["instance_id"], value_set=instances))


def _one_row_per_cell(
    task_root: Path,
    task: Task,
    stored: pa.Schema,
    scanned: pa.Table,
    selected: pa.Table,
    *,
    columns: Sequence[str],
) -> pa.Table:
    """Resolve one row per selected cell: the latest Attempt, or the release's own.

    An Attempt store's latest non-retryable Attempt wins. A release answering
    one selected cell more than once is refused rather than folded.
    """
    if ATTEMPT_FIELD.name in stored.names:
        return _current_selection(
            task_root, task, stored, scanned, selected, columns=columns
        )
    counted = selected.group_by(SELECTION_ROW).aggregate([(ROW_INDEX, "count")])
    repeated = counted.filter(pc.field(f"{ROW_INDEX}_count") > 1)
    if repeated.num_rows:
        row = repeated[SELECTION_ROW][0].as_py()
        named: dict[str, Any] = selected.filter(
            pc.field(SELECTION_ROW) == row
        ).to_pylist()[0]
        raise ValueError(
            "Released Results answer one cell more than once: Instance "
            f"{named['instance_id']!r}, algorithm {named[PARTITION_COLUMN]!r}, "
            f"seed {named['seed']}."
        )
    return (
        scanned.take(selected[ROW_INDEX])
        .select(list(columns))
        .add_column(0, SELECTION_ROW, selected[SELECTION_ROW])
    )


def _current_selection(
    task_root: Path,
    task: Task,
    stored: pa.Schema,
    scanned: pa.Table,
    selected: pa.Table,
    *,
    columns: Sequence[str],
) -> pa.Table:
    """Resolve numbered attempts once for coordinate and observed selections.

    The current Result is the latest non-retryable Attempt; a retryable one
    counts only when no non-retryable Attempt stands, so a later timeout under
    smaller limits never displaces an earlier success. The highest recorded
    attempt number travels beside it, because that is what numbers the next
    Attempt whether or not it answers the cell.
    """
    # Retryability is read for the selected rows alone, not for every row the
    # scan returned.
    numbered = (
        selected.append_column(
            _RETRYABLE,
            pc.is_in(
                scanned["status"].take(selected[ROW_INDEX]),
                value_set=_RETRYABLE_STATUSES,
            ),
        )
        .group_by([SELECTION_ROW, "attempt"])
        .aggregate([(ROW_INDEX, "count"), (ROW_INDEX, "min"), (_RETRYABLE, "min")])
    )
    repeated = numbered.filter(pc.field(f"{ROW_INDEX}_count") > 1)
    if repeated.num_rows > 0:
        repeated_keys = canonical_budget_keys(
            scanned.select([*KEY_COLUMNS, "attempt"]).take(repeated[f"{ROW_INDEX}_min"])
        )
        _refuse_disagreement(task_root, task, stored, repeated_keys, scanned)
    if not numbered.num_rows:
        return (
            stored.empty_table()
            .select(list(columns))
            .add_column(0, SELECTION_ROW, pa.array([], pa.int64()))
            .append_column(_HIGHEST_ATTEMPT, pa.array([], pa.int32()))
        )
    rows = numbered[SELECTION_ROW].to_numpy()
    attempts = numbered["attempt"].to_numpy()
    final = ~numbered[f"{_RETRYABLE}_min"].to_numpy(zero_copy_only=False)
    # Ordered by selection row, then non-retryable last, then attempt number,
    # so each selection row's winner is the last entry of its run.
    order = np.lexsort((attempts, final, rows))
    ordered = rows[order]
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1])
    winners = order[np.r_[boundaries, len(ordered) - 1]]
    highest = np.maximum.reduceat(attempts[order], np.r_[0, boundaries + 1])
    return (
        scanned.take(
            pa.array(numbered[f"{ROW_INDEX}_min"].to_numpy()[winners], pa.int64())
        )
        .select(list(columns))
        .add_column(0, SELECTION_ROW, pa.array(rows[winners], pa.int64()))
        .append_column(_HIGHEST_ATTEMPT, pa.array(highest, pa.int32()))
    )


def selected_rows(
    task_root: Path,
    task: Task,
    algorithm_ids: Sequence[str],
    instance_ids: Sequence[str],
    *,
    columns: Sequence[str],
    seeds: Sequence[int] | None = None,
    groups: (
        Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]] | None
    ) = None,
) -> tuple[pa.Table, pa.Table, int]:
    """Resolve trials from whichever store shape the root holds.

    Return current rows, their factor table, and the seed-axis length encoded
    in each selection-row index. A supplied factor selection retains its order;
    otherwise factors and seeds are ordered by their recorded identities.
    """
    store, stored = store_shape(task_root, task)
    history = ATTEMPT_FIELD.name in stored.names
    require_columns(task, stored, columns)
    empty = empty_selection(stored, columns)
    if history:
        empty = empty.append_column(_HIGHEST_ATTEMPT, pa.array([], pa.int32()))
    scanned = pa.table({})
    factors = initial_factors(stored, groups)
    if algorithm_ids and instance_ids:
        needed = list(
            dict.fromkeys(
                [*KEY_COLUMNS, *(("attempt", "status") if history else ()), *columns]
            )
        )
        scanned = _scan_instances(
            store,
            stored,
            pa.array(algorithm_ids, type=pa.string()),
            needed,
            pa.array(instance_ids, type=pa.string()),
        )
    if not scanned.num_rows:
        return empty, factors, 0
    selected, factors, seed_count = factor_selection(
        keyed_rows(scanned, "attempt") if history else keyed_rows(scanned),
        factors,
        instance_ids,
        seeds=seeds,
        groups=groups,
    )
    if not selected.num_rows:
        return empty, factors, seed_count
    rows = _one_row_per_cell(
        task_root, task, stored, scanned, selected, columns=columns
    ).sort_by(SELECTION_ROW)
    return rows, factors, seed_count


def recorded_statuses(
    roots: Iterable[Path],
    task: Task,
    algorithm_ids: Sequence[str],
    instance_ids: Sequence[str],
    *,
    groups: (
        Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]] | None
    ) = None,
    seeds: Sequence[int] | None = None,
) -> dict[int, ResultStatus]:
    """Return each recorded cell's status by selection-row index across roots.

    A cell answered under more than one root is one recorded cell: the index
    is the same across roots for one ``groups``/``seeds`` pair, so the first
    root to answer a row wins, in the order *roots* are given.
    """
    statuses: dict[int, ResultStatus] = {}
    if not (algorithm_ids and instance_ids):
        return statuses
    for root in roots:
        current, _, _ = selected_rows(
            root,
            task,
            algorithm_ids,
            instance_ids,
            columns=["status"],
            groups=groups,
            seeds=seeds,
        )
        for row, status in zip(
            cast("list[int]", current[SELECTION_ROW].to_pylist()),
            cast("list[str]", current["status"].to_pylist()),
        ):
            statuses.setdefault(row, ResultStatus(status))
    return statuses


def _refuse_disagreement(
    task_root: Path,
    task: Task,
    stored: pa.Schema,
    repeated: pa.Table,
    scanned: pa.Table,
) -> None:
    """Refuse repeated numbered attempts whose complete stored rows differ.

    Agreement is judged on every stored column, so the repeated rows are read
    again in full unless the view's own scan already holds them.
    """
    keys = repeated.select([*KEY_COLUMNS, "attempt"])
    full = (
        scanned
        if set(stored.names) <= set(scanned.column_names)
        else _scan_instances(
            attempts_directory(task_root),
            stored,
            pc.unique(keys[PARTITION_COLUMN]),
            stored.names,
            pc.unique(keys["instance_id"]),
        )
    )
    rows = keyed_rows(full, "attempt").join(
        keys, keys=[*KEY_COLUMNS, "attempt"], join_type="inner"
    )
    selected = full.take(rows[ROW_INDEX])
    cells = canonical_budget_keys(selected.select([*KEY_COLUMNS, "attempt"]))
    if _bitwise_twins(selected, cells):
        return
    # The cell is named by the canonical key; agreement is judged on the exact
    # stored row, so two hosts' records of one Attempt are a disagreement.
    identities: list[dict[str, Any]] = cells.to_pylist()
    facts: list[dict[str, Any]] = selected.to_pylist()
    first: dict[tuple[Any, ...], bytes] = {}
    for named, row in zip(identities, facts):
        key = tuple(named[name] for name in (*KEY_COLUMNS, "attempt"))
        encoded = canonical_json_bytes(row)
        if first.setdefault(key, encoded) != encoded:
            uuid = coordinate_record_identity(
                {
                    "instance_uuid": named["instance_id"],
                    "alg_uuid": named[PARTITION_COLUMN],
                    "budget": {
                        "path": named["budget_path"],
                        "config": decode_json_document(
                            named["budget_config"].encode("ascii")
                        ),
                    },
                    "alg_seed": named["seed"],
                }
            )
            raise ValueError(
                f"The recorded history disagrees about attempt {row['attempt']} at "
                f"coordinate {uuid}."
            )


def _bitwise_twins(rows: pa.Table, cells: pa.Table) -> bool:
    """Say whether every row repeats its cell's first row bit for bit.

    Bitwise-identical rows encode identically, so this settles agreement
    without encoding each row; any other case is judged row by row.
    """
    positions = cells.append_column(
        ROW_INDEX, pa.array(np.arange(cells.num_rows), pa.int64())
    )
    first = positions.group_by(cells.column_names).aggregate([(ROW_INDEX, "min")])
    paired = positions.join(first, keys=cells.column_names)
    members, firsts = paired[ROW_INDEX], paired[f"{ROW_INDEX}_min"]
    return all(
        _bitwise_equal(
            column.take(members).combine_chunks(), column.take(firsts).combine_chunks()
        )
        for column in rows.columns
    )


def _bitwise_equal(left: pa.Array[Any], right: pa.Array[Any]) -> bool:
    """Compare two aligned columns exactly, telling -0.0 from 0.0."""
    if not np.array_equal(
        left.is_null().to_numpy(zero_copy_only=False),
        right.is_null().to_numpy(zero_copy_only=False),
    ):
        return False
    kind: pa.DataType = left.type
    if pa.types.is_floating(kind):
        return np.array_equal(
            np.asarray(left.to_numpy(zero_copy_only=False)).view(np.uint8),
            np.asarray(right.to_numpy(zero_copy_only=False)).view(np.uint8),
        )
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return pc.list_value_length(left).equals(
            pc.list_value_length(right)
        ) and _bitwise_equal(pc.list_flatten(left), pc.list_flatten(right))
    if pa.types.is_nested(kind):
        return False
    return left.equals(right)


class ResultLookup:
    """Resolve exact-coordinate lookups, reading each partition's keys once.

    A partition's payload columns are read only once a lookup matches it.
    Sound only while the asking Sweep is the collection's sole writer.
    """

    def __init__(self, task_root: Path, task: Task) -> None:
        self._task_root = task_root
        self._task = task
        self._store, self._stored = store_shape(task_root, task)
        self._history = ATTEMPT_FIELD.name in self._stored.names
        self._partitions: dict[str, _IndexedPartition] = {}

    def _read(self, alg_uuid: str, columns: Sequence[str]) -> _IndexedPartition:
        scanned = scan_partitions(
            self._store, self._stored, pa.array([alg_uuid], pa.string()), columns
        )
        keyed = keyed_rows(scanned, "attempt") if self._history else keyed_rows(scanned)
        index: dict[tuple[Any, ...], list[int]] = {}
        for position, key in enumerate(
            zip(*(keyed[name].to_pylist() for name in KEY_COLUMNS))
        ):
            index.setdefault(key, []).append(position)
        return _IndexedPartition(
            scanned, keyed, index, len(columns) == len(self._stored)
        )

    def records(self, coordinates: Sequence[CellCoordinate]) -> list[dict[str, Any]]:
        """Return the current Result of each answered coordinate as a full record."""
        if not coordinates:
            return []
        selection = selection_table(coordinates)
        rows_by_algorithm: dict[str, list[int]] = {}
        for row, alg_uuid in enumerate(
            cast("list[str]", selection[PARTITION_COLUMN].to_pylist())
        ):
            rows_by_algorithm.setdefault(alg_uuid, []).append(row)
        records: list[dict[str, Any]] = []
        for alg_uuid, rows in rows_by_algorithm.items():
            group = selection.take(pa.array(rows, pa.int64()))
            keys = list(zip(*(group[name].to_pylist() for name in KEY_COLUMNS)))
            if alg_uuid not in self._partitions:
                needed = [*KEY_COLUMNS, *(("attempt",) if self._history else ())]
                self._partitions[alg_uuid] = self._read(alg_uuid, needed)
            partition = self._partitions[alg_uuid]
            if not any(key in partition.index for key in keys):
                continue
            if not partition.complete:
                partition = self._read(alg_uuid, self._stored.names)
                self._partitions[alg_uuid] = partition
            matched: list[int] = []
            selection_rows: list[int] = []
            for row, key in enumerate(keys):
                positions = partition.index.get(key, ())
                matched.extend(positions)
                selection_rows.extend([row] * len(positions))
            if not matched:
                continue
            selected = partition.keyed.take(
                pa.array(matched, pa.int64())
            ).append_column(SELECTION_ROW, pa.array(selection_rows, pa.int64()))
            current = _one_row_per_cell(
                self._task_root,
                self._task,
                self._stored,
                partition.scanned,
                selected,
                columns=self._stored.names,
            )
            records.extend(
                table_to_records(
                    current.select(self._stored.names).add_column(
                        0,
                        COORDINATE_FIELD,
                        group[COORDINATE_COLUMN].take(current[SELECTION_ROW]),
                    )
                )
            )
        return records


@dataclass(frozen=True, slots=True)
class _IndexedPartition:
    """One partition's scanned rows, keyed rows, and key index."""

    scanned: pa.Table
    keyed: pa.Table
    index: dict[tuple[Any, ...], list[int]]
    complete: bool
