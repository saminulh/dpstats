"""One Attempt or Result partition store on disk and the files standing in it.

The one-shape-per-tree guard, file naming, encoding, exclusive publication,
listing, scanning, and compaction; journals and resolution are elsewhere.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dpstats.results._row_identity import (
    distinct_row_census,
    distinct_rows,
)
from dpstats.results._schema import (
    PARTITION_COLUMN,
    attempt_schema,
    records_to_table,
    result_schema,
)
from dpstats.task import Task
from dpstats.transport.documents import (
    STAGING_PREFIX,
    publish_bytes_exclusive,
)
from dpstats.workspace.layout import (
    PARTITION_FILE_SUFFIX,
    PARTITION_PREFIX,
    attempt_partition_directory,
    attempts_directory,
    partition_directory,
    results_directory,
)

logger = logging.getLogger("benchmark")


def require_directory(path: Path) -> Path:
    """Refuse a symlink or a non-directory where a store or partition belongs.

    An absent path passes: a partition stands only once it holds a file.
    """
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Store path {path} must be a non-symlink directory.")
    return path


def attempt_store(task_root: Path) -> Path:
    """Return the task's attempt store, creating it when absent.

    A task collection holds one store shape. Released Results are resolved
    and immutable, so no writer may mint Attempts beside them.
    """
    if results_directory(task_root).exists():
        raise ValueError(
            f"Task collection {task_root} carries released Results, which are "
            "read-only; record Attempts in your own collection instead."
        )
    directory = attempts_directory(task_root)
    directory.mkdir(parents=True, exist_ok=True)
    return require_directory(directory)


def store_shape(task_root: Path, task: Task) -> tuple[Path, pa.Schema]:
    """Name the one store *task_root* holds and its rows' schema.

    Refuses a tree carrying both ``attempts/`` and ``results/``; an absent
    tree reads as an empty Attempt store, as today.
    """
    attempts, results = attempts_directory(task_root), results_directory(task_root)
    if attempts.exists() and results.exists():
        raise ValueError(
            f"Task collection {task_root} holds both Attempts and Results."
        )
    if results.exists():
        return require_directory(results), result_schema(task)
    return attempts, attempt_schema(task)


def mint_id() -> str:
    """Mint one run's or one compaction's name: UTC stamp, process, random tail.

    The tail is wide enough that two names minted in one second do not collide;
    every file it names is published exclusively, so a collision still refuses.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}-{secrets.token_hex(8)}"


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def require_partition_rows(rows: pa.Table, path: Path) -> None:
    """Refuse rows whose stored algorithm is not the partition file's own."""
    named = path.parent.name.removeprefix(PARTITION_PREFIX)
    for stored in pc.unique(rows.column(PARTITION_COLUMN)).to_pylist():
        if stored != named:
            raise ValueError(
                f"A row carrying algorithm {stored} cannot stand in {path}."
            )


def _file_bytes(records: Sequence[Mapping[str, Any]], task: Task, path: Path) -> bytes:
    """Encode one algorithm's records as the bytes of the partition file *path*."""
    rows = records_to_table(records, task)
    require_partition_rows(rows, path)
    return _parquet_bytes(rows)


def publish_records(
    task_root: Path, task: Task, records: Sequence[Mapping[str, Any]], name: str
) -> list[str]:
    """Publish these records as one file per algorithm partition, named *name*.

    The name is claimed exclusively, so a name that already stands is refused
    rather than replaced.

    Returns:
        The partitions written, in the order their files were published.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["coordinate"]["alg_uuid"], []).append(record)
    written: list[str] = []
    for alg_uuid, group in groups.items():
        partition = require_directory(attempt_partition_directory(task_root, alg_uuid))
        path = partition / f"{name}{PARTITION_FILE_SUFFIX}"
        publish_bytes_exclusive(path, _file_bytes(group, task, path))
        written.append(alg_uuid)
    return written


def _partition_files(partition: Path) -> list[Path]:
    """List one partition's files, refusing entries that are not files."""
    require_directory(partition)
    if not partition.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(partition.iterdir()):
        if path.name.startswith(STAGING_PREFIX):
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix != PARTITION_FILE_SUFFIX
        ):
            raise ValueError(f"Unexpected entry in partition: {path}.")
        files.append(path)
    return files


def partition_bytes(rows: pa.Table) -> dict[str, bytes]:
    """Encode released Result rows as one partition file's bytes per algorithm."""
    partitions = cast("list[str]", pc.unique(rows[PARTITION_COLUMN]).to_pylist())
    return {
        alg_uuid: _parquet_bytes(rows.filter(pc.field(PARTITION_COLUMN) == alg_uuid))
        for alg_uuid in sorted(partitions)
    }


def scan_partitions(
    store: Path,
    stored: pa.Schema,
    partitions: pa.Array[Any] | pa.ChunkedArray[Any],
    columns: Sequence[str],
) -> pa.Table:
    """Read the named columns of the selected partitions of one store.

    Each file is read on its own and refused if it holds a row of another
    partition, so a misplaced row cannot reach a coordinate through the key.
    """
    parts: list[pa.Table] = []
    if store.exists():
        require_directory(store)
        named = cast("list[str]", partitions.to_pylist())
        for alg_uuid in named:
            for path in _partition_files(partition_directory(store, alg_uuid)):
                rows = ds.dataset(str(path), format="parquet", schema=stored).to_table(
                    columns=list(columns)
                )
                require_partition_rows(rows, path)
                parts.append(rows)
    if not parts:
        return stored.empty_table().select(list(columns))
    return pa.concat_tables(parts)


def fragmented_partitions(store: Path) -> list[str]:
    """Name the algorithms whose partitions hold more than one file."""
    fragmented: list[str] = []
    for path in sorted(store.glob(f"{PARTITION_PREFIX}*")):
        if len(_partition_files(path)) > 1:
            fragmented.append(path.name[len(PARTITION_PREFIX) :])
    return fragmented


def _read_partition(files: Sequence[Path], task: Task) -> pa.Table:
    """Read every row of one partition's files, typed by the Task's schema.

    Raises:
        ValueError: naming a file holding a row of another partition.
    """
    schema = attempt_schema(task)
    if not files:
        return schema.empty_table()
    parts: list[pa.Table] = []
    for path in files:
        rows = pq.read_table(path)
        require_partition_rows(rows, path)
        parts.append(rows)
    return pa.concat_tables(parts).cast(schema)


@dataclass(frozen=True)
class PartitionCensus:
    """One Attempt partition's files and rows, and the set its distinct rows form.

    Rows identical on every column are one distinct row; the digest names the
    set of distinct rows whatever their order, so compaction leaves it alone.
    """

    files: int
    rows: int
    distinct_rows: int
    distinct_digest: str


def partition_census(task_root: Path, task: Task, alg_uuid: str) -> PartitionCensus:
    """Count one Attempt partition's files, rows, and distinct rows, as compaction reads them.

    Raises:
        ValueError: naming an entry the partition may not hold, or a file
            holding a row of another partition.
        TypeError: naming a column whose type row identity does not judge.
    """
    files = _partition_files(attempt_partition_directory(task_root, alg_uuid))
    rows = _read_partition(files, task)
    distinct, digest = distinct_row_census(rows)
    return PartitionCensus(len(files), rows.num_rows, distinct, digest)


def compact_partition(
    task_root: Path, task: Task, alg_uuid: str, *, repeats: bool = False
) -> None:
    """Merge one partition's files into one published file, then unlink them.

    The merged file is published and read back before any original is
    removed, so a process that dies here leaves every row readable through
    the originals, the merged file, or both; a view folds the agreeing
    copies. Rows identical on every column are one row here, so a batch that
    was both journaled and recovered leaves no repeats. A partition holding
    one file is left unread, unless ``repeats`` asks for a single file holding
    a row more than once to be rewritten to its distinct rows.

    Raises:
        ValueError: naming a file holding a row of another partition, or a
            merged file that does not hold exactly the partition's distinct
            rows. A merged file that cannot be read back or does not hold
            them is removed, and every original stands.
    """
    partition = attempt_partition_directory(task_root, alg_uuid)
    files = _partition_files(partition)
    if not files or (len(files) == 1 and not repeats):
        return
    rows = _read_partition(files, task)
    merged = distinct_rows(rows)
    if len(files) == 1 and merged.num_rows == rows.num_rows:
        return
    target = partition / f"compacted-{mint_id()}{PARTITION_FILE_SUFFIX}"
    publish_bytes_exclusive(target, _parquet_bytes(merged))
    try:
        published = _read_partition([target], task)
        kept = (published.num_rows, *distinct_row_census(published)) == (
            merged.num_rows,
            *distinct_row_census(rows),
        )
    except BaseException:
        target.unlink()
        raise
    if not kept:
        target.unlink()
        raise ValueError(
            f"Compacting {partition} did not keep its distinct rows, so the "
            "merged file was removed and every original stands."
        )
    for path in files:
        path.unlink()


def recover_staged_partition_files(store: Path) -> None:
    """Drop the staging files a killed partition writer left.

    A staging file inside a partition is a publication that never completed,
    so its writer never confirmed the row was recorded; the cell it answers
    is simply unanswered and runs again.

    Raises:
        ValueError: naming a partition or staged entry the store may not hold.
    """
    for partition in sorted(store.glob(f"{PARTITION_PREFIX}*")):
        require_directory(partition)
        for staged in sorted(partition.glob(f"{STAGING_PREFIX}*")):
            if staged.is_symlink() or not staged.is_file():
                raise ValueError(f"Unexpected entry in attempt partition: {staged}.")
            staged.unlink()
            logger.warning("Removed unpublished attempt partition file %s.", staged)
