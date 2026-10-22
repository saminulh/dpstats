"""Encode released Result rows, and check one released file's bytes.

``result_files`` and ``answered_coordinates`` resolve a root through
``_resolution.view``, so either shape it holds answers the same way. The
row-count reader below opens one file's bytes directly, without a Task.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq

from dpstats.coordinates import CellCoordinate
from dpstats.results._partitions import (
    partition_bytes,
    require_partition_rows,
)
from dpstats.results._resolution import view
from dpstats.results._schema import (
    ATTEMPT_FIELD,
    COORDINATE_COLUMN,
    KEY_COLUMNS,
    RESULT_FIELDS,
    result_schema,
)
from dpstats.results._selection import canonical_budget_keys
from dpstats.task import Task


def read_result_row_count(path: Path) -> int:
    """Return one released Result file's row count, without its Task.

    Every fixed column must stand with its stored type, every row must carry
    the partition's own algorithm, and no cell may be answered twice within the
    file, which is what a release consumer checks holding nothing but the bytes.
    """
    schema = pq.read_schema(path)
    for field in RESULT_FIELDS:
        if (
            field.name not in schema.names
            or schema.field(field.name).type != field.type
        ):
            raise ValueError(
                f"Result file {path} has no {field.name!r} column of {field.type}."
            )
    if ATTEMPT_FIELD.name in schema.names:
        raise ValueError(
            f"Result file {path} carries an {ATTEMPT_FIELD.name!r} column, "
            "which belongs to a store rather than a release."
        )
    keys = pq.read_table(path, columns=list(KEY_COLUMNS))
    require_partition_rows(keys, path)
    distinct = canonical_budget_keys(keys).group_by(list(KEY_COLUMNS)).aggregate([])
    if distinct.num_rows != keys.num_rows:
        raise ValueError(f"Result file {path} answers a cell more than once.")
    return keys.num_rows


def result_files(
    task_root: Path, task: Task, coordinates: Iterable[CellCoordinate]
) -> tuple[dict[str, bytes], set[str]]:
    """Encode the Result row of each selected coordinate as released bytes.

    Reads whichever shape *task_root* holds: an Attempt store's latest
    Attempt per coordinate, resolved here at the cut, or a release's
    already-resolved rows, carried through unchanged. Returns released bytes
    per algorithm partition beside the coordinate identities the tree
    answered. One partition is resolved at a time, so memory follows the
    largest partition rather than the whole selection.
    """
    columns = list(result_schema(task).names)
    by_algorithm: dict[str, list[CellCoordinate]] = {}
    for coordinate in coordinates:
        by_algorithm.setdefault(coordinate.alg_uuid, []).append(coordinate)
    files: dict[str, bytes] = {}
    answered: set[str] = set()
    for alg_uuid in sorted(by_algorithm):
        rows = view(task_root, task, by_algorithm[alg_uuid], columns=columns)
        # Nested columns are paged at chunk boundaries, so one contiguous chunk
        # makes the released bytes follow the rows alone, not how they were read.
        files.update(partition_bytes(rows.select(columns).combine_chunks()))
        answered.update(cast("list[str]", rows[COORDINATE_COLUMN].to_pylist()))
    return files, answered


def answered_coordinates(
    task_root: Path, task: Task, coordinates: Sequence[CellCoordinate]
) -> set[str]:
    """Name the selected coordinates *task_root* currently answers."""
    return set(
        cast(
            "list[str]",
            view(task_root, task, coordinates, columns=["status"])[
                COORDINATE_COLUMN
            ].to_pylist(),
        )
    )
