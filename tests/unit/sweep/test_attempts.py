"""Tests for the columnar ResultAttempt records of one task collection."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import dpstats.results._journal as journal_module
import dpstats.results._partitions as partitions
from dpstats.budgets import ApproxDPBudget, PureDPBudget
from dpstats.instance import Instance
from dpstats.results._journal import (
    AttemptWriter,
    recover_orphaned_attempt_journals,
)
from dpstats.results._partitions import (
    compact_partition,
    partition_census,
)
from dpstats.results._resolution import (
    current_outcomes,
    recorded_statuses,
    view,
)
from dpstats.results._schema import (
    attempt_schema,
    records_to_table,
    table_to_records,
)
from dpstats.coordinates import CellCoordinate
from dpstats.results import (
    ResultAttempt,
    ResultStatus,
)
from dpstats.task import (
    AnyArray1D,
    Array2D,
    _Array3D,
    ArrayND,
    BoolArray1D,
    _BoolArray2D,
    IntArray1D,
    IntArray2D,
    Task,
    TaskInput,
    TaskMetadata,
    TaskOutput,
)
from dpstats.tasks import BUILT_IN_TASKS
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.documents import stage_durable_bytes
from dpstats.workspace.layout import (
    attempt_partition_directory,
    attempts_directory,
    results_directory,
)
from tests.attempts import (
    ATTEMPT_FIXED_FIELDS,
    current_result,
    read_partition_rows,
    unsealed_journals,
    write_attempt_rows,
    write_result_partition,
)
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleMetadata,
    SimpleOutput,
)

_BUILT_IN_OUTPUT_COLUMNS: dict[str, list[tuple[str, pa.DataType]]] = {
    "covariance_estimation": [("covariance_est", pa.list_(pa.list_(pa.float64())))],
    "linear_regression": [("beta", pa.list_(pa.float64()))],
    "logistic_regression": [("theta", pa.list_(pa.float64()))],
    "m_estimation": [("theta", pa.list_(pa.float64()))],
    "mean_estimation": [("mean_est", pa.list_(pa.float64()))],
    "median_estimation": [("median_est", pa.float64())],
    "simple_linear_regression": [("slope", pa.float64()), ("intercept", pa.float64())],
    "single_coefficient_estimation": [("coefficient", pa.float64())],
}


def _coordinate(
    *, instance: int = 0, scale: float = 1.0, eps: float = 1.0, alg_seed: int = 0
) -> CellCoordinate:
    """Build one SIMPLE_TASK coordinate whose facts the arguments choose."""
    return CellCoordinate(
        instance=Instance(
            metadata=SimpleMetadata(n_samples=1, dim=1),
            input_digest=f"{instance:064x}",
            metric_arguments={},
        ),
        algorithm=SimpleAlgorithm(scale=scale),
        budget=PureDPBudget(eps=eps),
        alg_seed=alg_seed,
    )


def _attempt(
    coordinate: CellCoordinate,
    number: int = 0,
    *,
    status: ResultStatus = ResultStatus.SUCCESS,
    result: tuple[float, ...] = (1.0,),
    message: str | None = None,
    runtime_s: float | None = 0.5,
    peak_rss_mb: float | None = 2.0,
) -> ResultAttempt:
    """Build one attempt of any status at a coordinate."""
    payload: SimpleOutput | str | None = None
    if status is ResultStatus.SUCCESS:
        payload = SimpleOutput(result=np.array(result))
    elif status in (
        ResultStatus.NO_RELEASE,
        ResultStatus.INVALID_REGIME,
        ResultStatus.ALGORITHM_ERROR,
    ):
        payload = message
    return ResultAttempt(
        coordinate=coordinate,
        attempt_number=number,
        status=status,
        payload=payload,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
    )


def _custom_task(output_cls: type[TaskOutput]) -> Task:
    class Metadata(TaskMetadata):
        def validate_input(self, input: TaskInput) -> None:
            del input

    return Task(
        id="custom_task",
        name="Custom",
        description="A task whose Output the test chooses.",
        metadata_cls=Metadata,
        input_cls=TaskInput,
        output_cls=output_cls,
    )


def test_attempt_schema_types_every_built_in_output_after_the_fixed_columns() -> None:
    for task_id, task in BUILT_IN_TASKS.items():
        schema = attempt_schema(task)

        expected = [
            *ATTEMPT_FIXED_FIELDS,
            *(
                pa.field(name, column)
                for name, column in _BUILT_IN_OUTPUT_COLUMNS[task_id]
            ),
        ]
        assert [(field.name, field.type, field.nullable) for field in schema] == [
            (field.name, field.type, field.nullable) for field in expected
        ], task_id


def test_attempt_schema_nests_fixed_rank_aliases_and_types_scalars() -> None:
    class WideOutput(TaskOutput):
        plane: Array2D
        volume: _Array3D
        counts: IntArray1D
        grid: IntArray2D
        flags: BoolArray1D
        mask: _BoolArray2D
        count: int
        flag: bool
        name: str

    schema = attempt_schema(_custom_task(WideOutput))

    assert schema.field("plane").type == pa.list_(pa.list_(pa.float64()))
    assert schema.field("volume").type == pa.list_(pa.list_(pa.list_(pa.float64())))
    assert schema.field("counts").type == pa.list_(pa.int64())
    assert schema.field("grid").type == pa.list_(pa.list_(pa.int64()))
    assert schema.field("flags").type == pa.list_(pa.bool_())
    assert schema.field("mask").type == pa.list_(pa.list_(pa.bool_()))
    assert schema.field("count").type == pa.int64()
    assert schema.field("flag").type == pa.bool_()
    assert schema.field("name").type == pa.string()


def test_attempt_schema_refuses_an_unmapped_annotation_by_field_name() -> None:
    class UnrankedOutput(TaskOutput):
        estimate: ArrayND

    class UntypedOutput(TaskOutput):
        values: AnyArray1D

    class OptionalOutput(TaskOutput):
        estimate: float | None

    class CollidingOutput(TaskOutput):
        status: float

    for output_cls, field in (
        (UnrankedOutput, "estimate"),
        (UntypedOutput, "values"),
        (OptionalOutput, "estimate"),
        (CollidingOutput, "status"),
    ):
        with pytest.raises(ValueError, match=f"'{field}'"):
            attempt_schema(_custom_task(output_cls))


def _every_status() -> list[ResultAttempt]:
    """One attempt per terminal outcome, with nonfinite released values."""
    coordinate = _coordinate()
    return [
        _attempt(coordinate, 0, result=(1.5, -0.0)),
        _attempt(_coordinate(alg_seed=1), 1, result=(math.nan, math.inf, -math.inf)),
        _attempt(
            _coordinate(alg_seed=2),
            status=ResultStatus.NO_RELEASE,
            message="the mechanism released bottom",
        ),
        _attempt(
            _coordinate(alg_seed=3),
            status=ResultStatus.INVALID_REGIME,
            message="the budget family is not realized",
        ),
        _attempt(
            _coordinate(alg_seed=4),
            status=ResultStatus.ALGORITHM_ERROR,
            message="numpy.linalg.LinAlgError",
        ),
        _attempt(
            _coordinate(alg_seed=5),
            status=ResultStatus.ALGORITHM_ERROR,
            message="ValueError",
        ),
        _attempt(_coordinate(alg_seed=6), 2, status=ResultStatus.TIMEOUT),
        _attempt(
            _coordinate(alg_seed=7),
            status=ResultStatus.MEMORY_LIMIT_EXCEEDED,
            runtime_s=None,
        ),
        _attempt(
            _coordinate(alg_seed=8),
            status=ResultStatus.TIMEOUT,
            runtime_s=None,
            peak_rss_mb=None,
        ),
    ]


def test_rows_round_trip_every_status_to_identical_canonical_records() -> None:
    attempts = _every_status()
    records = [attempt.to_record() for attempt in attempts]

    # A view joins each row's coordinate identity in from the selection; the
    # stored row carries only the file columns.
    view = records_to_table(records, SIMPLE_TASK).add_column(
        0,
        "coordinate_uuid",
        pa.array([record["coordinate_uuid"] for record in records], pa.string()),
    )
    restored = table_to_records(view)

    assert [canonical_json_bytes(record) for record in restored] == [
        canonical_json_bytes(record) for record in records
    ]


def test_rows_lay_out_each_status_by_the_ruled_column_rules() -> None:
    table = records_to_table(
        [attempt.to_record() for attempt in _every_status()], SIMPLE_TASK
    )
    rows: list[dict[str, Any]] = table.to_pylist()

    by_status = {row["status"]: row for row in rows}
    assert table.column_names[:2] == ["instance_id", "alg_id"]
    successes = [row for row in rows if row["status"] == "success"]
    assert successes[0]["result"] == [1.5, -0.0]
    assert successes[0]["reason"] is None
    released = successes[1]
    assert math.isnan(released["result"][0])
    assert released["result"][1:] == [math.inf, -math.inf]
    assert released["reason"] is None
    assert by_status["no_release"]["reason"] == "the mechanism released bottom"
    assert by_status["invalid_regime"]["reason"] == (
        "the budget family is not realized"
    )
    assert by_status["timeout"]["result"] is None
    assert by_status["timeout"]["reason"] is None
    assert by_status["memory_limit_exceeded"]["runtime_s"] is None
    errors = [row for row in rows if row["status"] == "algorithm_error"]
    assert sorted(row["reason"] for row in errors) == [
        "ValueError",
        "numpy.linalg.LinAlgError",
    ]
    assert {row["attempt"] for row in rows} == {0, 1, 2}
    assert by_status["success"]["budget_path"].endswith(":PureDPBudget")
    assert by_status["success"]["budget_config"] == '{"eps":1.0}'


def _row(
    coordinate: CellCoordinate,
    attempt: int = 0,
    *,
    status: str = "success",
    result: tuple[float, ...] | None = (1.0,),
    reason: str | None = None,
    runtime_s: float | None = 0.5,
    peak_rss_mb: float | None = 2.0,
    alg_id: str | None = None,
) -> dict[str, Any]:
    """Build one partition-file row from a coordinate's own facts."""
    budget = coordinate.budget.dump_config()
    return {
        "instance_id": coordinate.instance_uuid,
        "alg_id": coordinate.alg_uuid if alg_id is None else alg_id,
        "budget_path": budget.path,
        "budget_config": canonical_json_bytes(budget.config).decode("ascii"),
        "seed": coordinate.alg_seed,
        "attempt": attempt,
        "status": status,
        "runtime_s": runtime_s,
        "peak_rss_mb": peak_rss_mb,
        "reason": reason,
        "result": None if result is None else list(result),
    }


def _partition(task_root: Path, coordinate: CellCoordinate) -> Path:
    return attempt_partition_directory(task_root, coordinate.alg_uuid)


def _seal(task_root: Path, attempt: ResultAttempt) -> Path:
    """Write one Attempt through the production writer; return its sealed file."""
    writer = AttemptWriter(task_root, SIMPLE_TASK)
    writer.add(attempt)
    writer.close()
    (published,) = [
        path
        for path in _partition(task_root, attempt.coordinate).iterdir()
        if not path.name.startswith(".incomplete-")
    ]
    return published


def test_writer_journals_each_attempt_durably_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        synced.append(os.fstat(descriptor).st_ino)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)
    first = _attempt(_coordinate())
    second = _attempt(_coordinate(alg_seed=1))

    writer.add(first)
    (journal,) = unsealed_journals(tmp_path)
    assert journal.read_bytes() == canonical_json_bytes(first.to_record()) + b"\n"
    assert journal.stat().st_ino in synced
    writer.add(second)
    assert journal.read_bytes() == b"".join(
        canonical_json_bytes(attempt.to_record()) + b"\n" for attempt in (first, second)
    )
    writer.close()


def test_close_seals_each_algorithm_into_its_partition_then_drops_the_journal(
    tmp_path: Path,
) -> None:
    fast = _coordinate(scale=1.0)
    slow = _coordinate(scale=2.0)
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)
    writer.add(_attempt(fast))
    writer.add(_attempt(slow, result=(2.0,)))

    writer.close()

    store = attempts_directory(tmp_path)
    assert not unsealed_journals(tmp_path)
    assert sorted(store.iterdir()) == sorted(
        attempt_partition_directory(tmp_path, coordinate.alg_uuid)
        for coordinate in (fast, slow)
    )
    for coordinate, value in ((fast, 1.0), (slow, 2.0)):
        (published,) = _partition(tmp_path, coordinate).iterdir()
        assert published.name.endswith(".0.parquet")
        (row,) = read_partition_rows(_partition(tmp_path, coordinate))
        assert row["result"] == [value]
        assert row["instance_id"] == coordinate.instance_uuid
        assert row["alg_id"] == coordinate.alg_uuid
        assert "coordinate_uuid" not in row


def test_the_size_cap_seals_mid_run_under_the_run_id_and_a_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(journal_module, "_JOURNAL_SIZE_BYTES", 1)
    coordinate = _coordinate()
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)

    writer.add(_attempt(coordinate, 0))
    writer.add(_attempt(coordinate, 1))

    partition = _partition(tmp_path, coordinate)
    names = sorted(path.name for path in partition.iterdir())
    assert [name.rsplit(".", 2)[1:] for name in names] == [
        ["0", "parquet"],
        ["1", "parquet"],
    ]
    assert len({name.rsplit(".", 2)[0] for name in names}) == 1
    assert not unsealed_journals(tmp_path)

    writer.close()

    (compacted,) = partition.iterdir()
    assert compacted.name.startswith("compacted-")
    assert sorted(row["attempt"] for row in read_partition_rows(partition)) == [0, 1]


def test_the_age_trigger_seals_without_closing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(journal_module, "_JOURNAL_MAX_AGE_SECONDS", 0.0)
    coordinate = _coordinate()
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)

    writer.add(_attempt(coordinate, 0))
    assert len(list(_partition(tmp_path, coordinate).iterdir())) == 1
    assert not unsealed_journals(tmp_path)
    writer.add(_attempt(coordinate, 1))
    assert len(list(_partition(tmp_path, coordinate).iterdir())) == 2

    writer.close()

    assert sorted(
        row["attempt"] for row in read_partition_rows(_partition(tmp_path, coordinate))
    ) == [0, 1]


def test_a_partition_file_reaches_the_device_before_its_name_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Power loss is not testable, so the ordering is observed at each sync."""
    store = attempts_directory(tmp_path)
    synced: list[list[str]] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        if not store.is_dir():
            return
        entries = [path for path in store.rglob("*") if path.is_file()]
        if os.fstat(descriptor).st_ino in {path.stat().st_ino for path in entries}:
            synced.append(sorted(str(path.relative_to(store)) for path in entries))

    monkeypatch.setattr(os, "fsync", recording_fsync)
    published = _seal(tmp_path, _attempt(_coordinate()))

    name = str(published.relative_to(store))
    assert synced, "the partition file was never synced"
    for names in synced:
        assert name not in names
        assert [entry for entry in names if ".incomplete-" in entry]


def test_a_row_is_refused_by_a_partition_naming_another_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinate = _coordinate(scale=1.0)
    other = _coordinate(scale=2.0)

    def foreign_partition(task_root: Path, alg_uuid: str) -> Path:
        del alg_uuid
        return _partition(task_root, other)

    monkeypatch.setattr(partitions, "attempt_partition_directory", foreign_partition)
    with pytest.raises(ValueError, match=coordinate.alg_uuid):
        _seal(tmp_path, _attempt(coordinate))

    assert not _partition(tmp_path, other).exists()


def test_compaction_refuses_a_row_stored_under_another_algorithm(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate(scale=1.0)
    other = _coordinate(scale=2.0)
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-a.0", [_row(coordinate, 0)])
    write_attempt_rows(
        tmp_path,
        coordinate.alg_uuid,
        "run-b.0",
        [_row(coordinate, 1, alg_id=other.alg_uuid)],
    )

    with pytest.raises(ValueError, match=other.alg_uuid):
        compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    assert sorted(path.name for path in _partition(tmp_path, coordinate).iterdir()) == [
        "run-a.0.parquet",
        "run-b.0.parquet",
    ]


def test_a_symlinked_partition_is_refused_on_write_and_on_read(tmp_path: Path) -> None:
    coordinate = _coordinate()
    outside = tmp_path / "outside"
    outside.mkdir()
    attempts_directory(tmp_path).mkdir()
    _partition(tmp_path, coordinate).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        _seal(tmp_path, _attempt(coordinate))
    with pytest.raises(ValueError, match="non-symlink"):
        view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status"])

    assert not list(outside.iterdir())


def test_compact_partition_merges_the_files_and_unlinks_the_originals(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    partition = _partition(tmp_path, coordinate)
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-a.0", [_row(coordinate, 0)])
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "cell-x-1", [_row(coordinate, 1, result=(2.0,))]
    )
    (partition / ".incomplete-debris.parquet").write_bytes(b"not a file yet")

    compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    debris, compacted = sorted(path.name for path in partition.iterdir())
    assert debris == ".incomplete-debris.parquet"
    assert compacted.startswith("compacted-") and compacted.endswith(".parquet")
    assert sorted(
        (row["attempt"], row["result"]) for row in read_partition_rows(partition)
    ) == [(0, [1.0]), (1, [2.0])]

    compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    assert sorted(path.name for path in partition.iterdir()) == [debris, compacted]


def test_compaction_keeps_one_copy_of_a_row_two_files_hold_alike(
    tmp_path: Path,
) -> None:
    """A batch that was journaled and then recovered leaves no repeats."""
    coordinate = _coordinate()
    partition = _partition(tmp_path, coordinate)
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run.0", [_row(coordinate, 0)])
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "recovered-run", [_row(coordinate, 0)]
    )

    compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    (compacted,) = partition.iterdir()
    assert compacted.name.startswith("compacted-")
    assert read_partition_rows(partition) == [_row(coordinate, 0)]


def test_a_census_counts_repeats_and_compaction_keeps_its_distinct_set(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    for name in ("run.0", "recovered-run", "pulled-copy"):
        write_attempt_rows(
            tmp_path,
            coordinate.alg_uuid,
            name,
            [_row(coordinate, 0), _row(coordinate, 1, result=(2.0,))],
        )

    before = partition_census(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)
    compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)
    after = partition_census(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    assert (before.files, before.rows, before.distinct_rows) == (3, 6, 2)
    assert (after.files, after.rows, after.distinct_rows) == (1, 2, 2)
    assert after.distinct_digest == before.distinct_digest


def test_opening_a_writer_compacts_every_fragmented_partition(tmp_path: Path) -> None:
    split = _coordinate(scale=1.0)
    whole = _coordinate(scale=2.0)
    write_attempt_rows(tmp_path, split.alg_uuid, "cell-one-0", [_row(split, 0)])
    write_attempt_rows(tmp_path, split.alg_uuid, "cell-two-1", [_row(split, 1)])
    write_attempt_rows(tmp_path, whole.alg_uuid, "run.0", [_row(whole, 0)])

    AttemptWriter(tmp_path, SIMPLE_TASK).close()

    (compacted,) = _partition(tmp_path, split).iterdir()
    assert compacted.name.startswith("compacted-")
    assert sorted(
        row["attempt"] for row in read_partition_rows(_partition(tmp_path, split))
    ) == [0, 1]
    assert [path.name for path in _partition(tmp_path, whole).iterdir()] == [
        "run.0.parquet"
    ]


def test_a_kill_before_the_merged_file_is_published_leaves_the_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinate = _coordinate()
    partition = _partition(tmp_path, coordinate)
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-a.0", [_row(coordinate, 0)])
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-b.0", [_row(coordinate, 1)])

    def refuse(path: Path, payload: bytes) -> None:
        raise OSError(f"device full at {path.name} ({len(payload)} bytes)")

    monkeypatch.setattr(partitions, "publish_bytes_exclusive", refuse)
    with pytest.raises(OSError, match="device full"):
        compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    assert sorted(path.name for path in partition.iterdir()) == [
        "run-a.0.parquet",
        "run-b.0.parquet",
    ]
    assert sorted(row["attempt"] for row in read_partition_rows(partition)) == [0, 1]


def test_a_kill_after_publishing_the_merged_file_leaves_agreeing_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinate = _coordinate()
    partition = _partition(tmp_path, coordinate)
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-a.0", [_row(coordinate, 0)])
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run-b.0", [_row(coordinate, 1)])
    real_unlink = Path.unlink

    def killed_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.parent == partition and not path.name.startswith(
            ("compacted-", ".incomplete-")
        ):
            raise OSError("killed before the originals were removed")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", killed_unlink)
    with pytest.raises(OSError, match="killed"):
        compact_partition(tmp_path, SIMPLE_TASK, coordinate.alg_uuid)

    names = sorted(path.name for path in partition.iterdir())
    assert len(names) == 3
    rows = sorted(read_partition_rows(partition), key=lambda row: row["attempt"])
    assert [row["attempt"] for row in rows] == [0, 0, 1, 1]
    assert rows[0] == rows[1] and rows[2] == rows[3]


def test_recovery_seals_what_a_dead_writer_journaled_and_drops_its_torn_tail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _coordinate(scale=1.0)
    second = _coordinate(scale=2.0)
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)
    writer.add(_attempt(first))
    writer.add(_attempt(second, result=(2.0,)))
    (journal,) = unsealed_journals(tmp_path)
    torn = canonical_json_bytes(_attempt(first, 1).to_record())[:20]
    with journal.open("ab") as handle:
        handle.write(torn)
    del writer

    with caplog.at_level(logging.WARNING, logger="benchmark"):
        recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    assert not journal.exists()
    journal_id = journal.name[len(".incomplete-") : -len(".jsonl")]
    for coordinate, value in ((first, 1.0), (second, 2.0)):
        (recovered,) = _partition(tmp_path, coordinate).iterdir()
        assert recovered.name == f"recovered-{journal_id}.parquet"
        (row,) = read_partition_rows(_partition(tmp_path, coordinate))
        assert (row["attempt"], row["result"]) == (0, [value])
    assert f"Dropping {len(torn)} bytes" in caplog.text
    assert journal.name in caplog.text


def test_recovery_after_a_kill_between_publishing_and_unlinking_agrees_with_itself(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    writer = AttemptWriter(tmp_path, SIMPLE_TASK)
    writer.add(_attempt(coordinate))
    (journal,) = unsealed_journals(tmp_path)
    journaled = journal.read_bytes()
    writer.close()
    journal.write_bytes(journaled)

    recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    partition = _partition(tmp_path, coordinate)
    assert not journal.exists()
    names = sorted(path.name for path in partition.iterdir())
    assert len(names) == 2
    assert [name for name in names if name.startswith("recovered-")]
    rows = read_partition_rows(partition)
    assert len(rows) == 2 and rows[0] == rows[1]


def test_recovery_removes_partition_staging_debris_but_never_a_recorded_attempt(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run.0", [_row(coordinate)])
    partition = _partition(tmp_path, coordinate)
    (partition / ".incomplete-half.parquet").write_bytes(b"half a file")

    recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    assert [path.name for path in partition.iterdir()] == ["run.0.parquet"]
    recover_orphaned_attempt_journals(tmp_path / "absent", SIMPLE_TASK)


def test_recovery_removes_a_complete_staged_file_without_promoting_it(
    tmp_path: Path,
) -> None:
    """A staged file that reads back cleanly is still debris, not a record."""
    coordinate = _coordinate()
    attempt = _attempt(coordinate, 3, result=(4.0,))
    published = _seal(tmp_path / "elsewhere", attempt)
    partition = _partition(tmp_path, coordinate)
    partition.mkdir(parents=True)
    stage_durable_bytes(partition, published.read_bytes(), suffix=".parquet")

    recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    assert list(partition.iterdir()) == []


def test_recovery_refuses_a_symlinked_partition_holding_a_staged_file(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    outside = tmp_path / "outside"
    outside.mkdir()
    staged = stage_durable_bytes(outside, b"half a file", suffix=".parquet")
    attempts_directory(tmp_path).mkdir()
    _partition(tmp_path, coordinate).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    assert [path.name for path in outside.iterdir()] == [staged.name]


def test_a_seal_refuses_the_name_of_a_file_that_already_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs minting one identifier refuse rather than replace each other."""

    def one_id() -> str:
        return "20260903T000000Z-1-collide"

    monkeypatch.setattr(journal_module, "mint_id", one_id)
    coordinate = _coordinate()
    first = AttemptWriter(tmp_path, SIMPLE_TASK)
    first.add(_attempt(coordinate, 0))
    first.close()

    second = AttemptWriter(tmp_path, SIMPLE_TASK)
    second.add(_attempt(coordinate, 1, result=(2.0,)))
    with pytest.raises(FileExistsError):
        second.close()

    partition = _partition(tmp_path, coordinate)
    assert [path.name for path in partition.iterdir()] == [
        "20260903T000000Z-1-collide.0.parquet"
    ]
    assert [
        (row["attempt"], row["result"]) for row in read_partition_rows(partition)
    ] == [(0, [1.0])]


def test_recovery_names_an_unreadable_journal_and_keeps_it_while_sealing_siblings(
    tmp_path: Path,
) -> None:
    store = attempts_directory(tmp_path)
    store.mkdir()
    coordinate = _coordinate()
    bad = store / ".incomplete-bad.jsonl"
    bad.write_bytes(b'{"not": "an attempt"}\n')
    good = store / ".incomplete-good.jsonl"
    good.write_bytes(canonical_json_bytes(_attempt(coordinate).to_record()) + b"\n")
    fragment = store / ".incomplete-fragment.jsonl"
    fragment.write_bytes(b"{")

    with pytest.raises(ValueError, match="not recovered") as report:
        recover_orphaned_attempt_journals(tmp_path, SIMPLE_TASK)

    assert bad.name in str(report.value)
    assert bad.read_bytes() == b'{"not": "an attempt"}\n'
    assert not good.exists()
    assert not fragment.exists()
    assert [path.name for path in _partition(tmp_path, coordinate).iterdir()] == [
        "recovered-good.parquet"
    ]


def test_view_projects_the_requested_columns_after_the_identity(
    tmp_path: Path,
) -> None:
    done = _coordinate(instance=1)
    timed_out = _coordinate(instance=2)
    write_attempt_rows(
        tmp_path,
        done.alg_uuid,
        "run.0",
        [_row(done, 0), _row(timed_out, 0, status="timeout", result=None)],
    )

    rows = view(tmp_path, SIMPLE_TASK, [done, timed_out], columns=["status", "attempt"])

    assert rows.column_names == ["coordinate_uuid", "status", "attempt"]
    assert rows.to_pylist() == sorted(
        [
            {"coordinate_uuid": done.uuid, "status": "success", "attempt": 0},
            {"coordinate_uuid": timed_out.uuid, "status": "timeout", "attempt": 0},
        ],
        key=lambda row: row["coordinate_uuid"],
    )
    with pytest.raises(ValueError, match="'nonsense'"):
        view(tmp_path, SIMPLE_TASK, [done], columns=["nonsense"])
    with pytest.raises(ValueError, match="'coordinate_uuid'"):
        view(tmp_path, SIMPLE_TASK, [done], columns=["coordinate_uuid"])


def test_recorded_statuses_lets_the_first_root_answer_each_selection_row(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    seed0, seed1 = _coordinate(alg_seed=0), _coordinate(alg_seed=1)
    write_attempt_rows(first, seed0.alg_uuid, "run.0", [_row(seed0, 0)])
    write_attempt_rows(
        second,
        seed0.alg_uuid,
        "run.0",
        [_row(seed0, 0, status="timeout", result=None), _row(seed1, 0)],
    )
    groups = [(seed0.instance, seed0.algorithm, (seed0.budget,))]
    ids = ([seed0.alg_uuid], [seed0.instance_uuid])

    forward = recorded_statuses(
        [first, second], SIMPLE_TASK, *ids, groups=groups, seeds=[0, 1]
    )
    backward = recorded_statuses(
        [second, first], SIMPLE_TASK, *ids, groups=groups, seeds=[0, 1]
    )

    assert forward == {0: ResultStatus.SUCCESS, 1: ResultStatus.SUCCESS}
    assert backward == {0: ResultStatus.TIMEOUT, 1: ResultStatus.SUCCESS}
    assert (
        recorded_statuses([first], SIMPLE_TASK, [], [], groups=groups, seeds=[0, 1])
        == {}
    )


def test_a_row_recorded_under_another_hosts_float_answers_its_cell(
    tmp_path: Path,
) -> None:
    """One budget, two hosts' floats: the row answers and reads as executed."""
    # The two deltas a rule whose true value is 0.0017735147777247594...
    # returned on two hosts; both name the same budget, and each
    # canonicalizes to the same stored value at construction.
    recorded = _coordinate().model_copy(
        update={"budget": ApproxDPBudget(eps=1.0, delta=0.0017735147777247586)}
    )
    resolved = _coordinate().model_copy(
        update={"budget": ApproxDPBudget(eps=1.0, delta=0.0017735147777247584)}
    )
    write_attempt_rows(tmp_path, recorded.alg_uuid, "run.0", [_row(recorded, 0)])

    assert resolved.uuid == recorded.uuid
    assert current_outcomes(tmp_path, SIMPLE_TASK, [resolved]) == {
        resolved.uuid: (1, ResultStatus.SUCCESS)
    }
    rows = view(tmp_path, SIMPLE_TASK, [resolved], columns=["budget_config", "status"])
    assert rows.to_pylist() == [
        {
            "coordinate_uuid": resolved.uuid,
            "budget_config": '{"delta":0.00177351477772,"eps":1.0}',
            "status": "success",
        }
    ]


def test_the_largest_seed_the_models_accept_round_trips(tmp_path: Path) -> None:
    """``AlgorithmSeed`` bounds the domain; the column carries all of it."""
    coordinate = _coordinate(alg_seed=2**64 - 1)

    _seal(tmp_path, _attempt(coordinate))
    rows = view(tmp_path, SIMPLE_TASK, [coordinate], columns=["seed", "status"])

    assert rows.to_pylist() == [
        {
            "coordinate_uuid": coordinate.uuid,
            "seed": 2**64 - 1,
            "status": "success",
        }
    ]


def test_view_reads_only_the_partitions_the_selection_names(
    tmp_path: Path,
) -> None:
    selected = _coordinate(scale=1.0)
    other = _coordinate(scale=2.0)
    write_attempt_rows(tmp_path, selected.alg_uuid, "run.0", [_row(selected)])
    unread = _partition(tmp_path, other)
    unread.mkdir(parents=True)
    (unread / "run.0.parquet").write_bytes(b"not a parquet file")

    rows = view(tmp_path, SIMPLE_TASK, [selected], columns=["status"])

    assert rows.to_pylist() == [{"coordinate_uuid": selected.uuid, "status": "success"}]
    with pytest.raises(pa.ArrowInvalid):
        view(tmp_path, SIMPLE_TASK, [selected, other], columns=["status"])


def test_view_refuses_a_file_holding_another_partitions_row(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate(scale=1.0)
    other = _coordinate(scale=2.0)
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "run.0", [_row(coordinate, 0), _row(other, 0)]
    )

    with pytest.raises(ValueError, match="run.0.parquet"):
        view(tmp_path, SIMPLE_TASK, [coordinate, other], columns=["status"])


def test_view_takes_the_highest_attempt_per_coordinate(tmp_path: Path) -> None:
    retried = _coordinate()
    once = _coordinate(alg_seed=1)
    write_attempt_rows(
        tmp_path,
        retried.alg_uuid,
        "run-a.0",
        [_row(retried, 0, status="timeout", result=None), _row(once, 0)],
    )
    write_attempt_rows(
        tmp_path,
        retried.alg_uuid,
        "run-b.0",
        [
            _row(retried, 2, result=(3.0,)),
            _row(retried, 1, status="timeout", result=None),
        ],
    )

    rows = view(
        tmp_path, SIMPLE_TASK, [retried, once], columns=["attempt", "status", "result"]
    )

    assert rows.to_pylist() == sorted(
        [
            {
                "coordinate_uuid": retried.uuid,
                "attempt": 2,
                "status": "success",
                "result": [3.0],
            },
            {
                "coordinate_uuid": once.uuid,
                "attempt": 0,
                "status": "success",
                "result": [1.0],
            },
        ],
        key=lambda row: row["coordinate_uuid"],
    )


def test_view_folds_agreeing_duplicates_and_refuses_disagreeing_ones(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    write_attempt_rows(tmp_path, coordinate.alg_uuid, "run.0", [_row(coordinate, 0)])
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "recovered-x", [_row(coordinate, 0)]
    )

    rows = view(tmp_path, SIMPLE_TASK, [coordinate], columns=["attempt", "result"])

    assert rows.to_pylist() == [
        {"coordinate_uuid": coordinate.uuid, "attempt": 0, "result": [1.0]}
    ]
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "run.1", [_row(coordinate, 0, result=(2.0,))]
    )
    with pytest.raises(ValueError, match=f"attempt 0 at coordinate {coordinate.uuid}"):
        view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status"])


def test_view_folds_bitwise_twins_and_refuses_a_twin_differing_in_zero_sign(
    tmp_path: Path,
) -> None:
    """Agreement is exact: a NaN twin is one Attempt, but -0.0 is not 0.0."""
    twin = _coordinate(alg_seed=0)
    signed = _coordinate(alg_seed=1)
    for name in ("run.0", "recovered-x"):
        write_attempt_rows(
            tmp_path,
            twin.alg_uuid,
            name,
            [_row(twin, 0, result=(math.nan, math.inf), runtime_s=None)],
        )

    rows = view(tmp_path, SIMPLE_TASK, [twin], columns=["attempt", "runtime_s"])

    assert rows.to_pylist() == [
        {"coordinate_uuid": twin.uuid, "attempt": 0, "runtime_s": None}
    ]
    write_attempt_rows(
        tmp_path, signed.alg_uuid, "run.1", [_row(signed, 0, result=(0.0,))]
    )
    write_attempt_rows(
        tmp_path, signed.alg_uuid, "run.2", [_row(signed, 0, result=(-0.0,))]
    )
    with pytest.raises(ValueError, match=f"attempt 0 at coordinate {signed.uuid}"):
        view(tmp_path, SIMPLE_TASK, [twin, signed], columns=["status"])


def test_a_later_retryable_attempt_never_displaces_an_earlier_terminal_one(
    tmp_path: Path,
) -> None:
    """The current Result is the latest non-retryable Attempt.

    A retry that ran under smaller limits and timed out says nothing about the
    success already recorded, so the success stays current while the timeout
    stays recorded; the next Attempt is still numbered one past the highest.
    """
    coordinate = _coordinate()
    write_attempt_rows(
        tmp_path,
        coordinate.alg_uuid,
        "run.0",
        [
            _row(coordinate, 0, result=(1.0,)),
            _row(coordinate, 1, status="timeout", result=None),
        ],
    )

    rows = view(
        tmp_path, SIMPLE_TASK, [coordinate], columns=["attempt", "status", "result"]
    )

    assert rows.to_pylist() == [
        {
            "coordinate_uuid": coordinate.uuid,
            "attempt": 0,
            "status": "success",
            "result": [1.0],
        }
    ]
    assert current_outcomes(tmp_path, SIMPLE_TASK, [coordinate]) == {
        coordinate.uuid: (2, ResultStatus.SUCCESS)
    }


def test_a_retryable_attempt_is_current_when_no_terminal_one_stands(
    tmp_path: Path,
) -> None:
    """A retryable Attempt counts only when no non-retryable Attempt exists."""
    coordinate = _coordinate()
    write_attempt_rows(
        tmp_path,
        coordinate.alg_uuid,
        "run.0",
        [
            _row(coordinate, 0, status="timeout", result=None),
            _row(coordinate, 1, status="memory_limit_exceeded", result=None),
        ],
    )

    assert current_outcomes(tmp_path, SIMPLE_TASK, [coordinate]) == {
        coordinate.uuid: (2, ResultStatus.MEMORY_LIMIT_EXCEEDED)
    }


def _gapped_store(
    task_root: Path,
) -> tuple[CellCoordinate, CellCoordinate, CellCoordinate]:
    """Store attempts 2 and 4 at one cell, attempt 3 at a second, none at a third."""
    gapped = _coordinate(alg_seed=0)
    lone = _coordinate(alg_seed=1)
    unanswered = _coordinate(alg_seed=2)
    write_attempt_rows(
        task_root,
        gapped.alg_uuid,
        "run.0",
        [
            _row(gapped, 2, result=(1.0,)),
            _row(gapped, 4, status="timeout", result=None),
            _row(lone, 3, result=(3.0,)),
        ],
    )
    return gapped, lone, unanswered


def test_reading_and_resolution_use_the_attempt_numbers_that_stand(
    tmp_path: Path,
) -> None:
    """Rows may be dropped, so no read path may assume contiguous numbering."""
    gapped, lone, unanswered = _gapped_store(tmp_path)

    rows = view(
        tmp_path, SIMPLE_TASK, [gapped, lone, unanswered], columns=["attempt", "status"]
    )

    assert sorted(rows.to_pylist(), key=lambda row: row["attempt"]) == [
        {"coordinate_uuid": gapped.uuid, "attempt": 2, "status": "success"},
        {"coordinate_uuid": lone.uuid, "attempt": 3, "status": "success"},
    ]
    assert current_outcomes(tmp_path, SIMPLE_TASK, [gapped, lone, unanswered]) == {
        gapped.uuid: (5, ResultStatus.SUCCESS),
        lone.uuid: (4, ResultStatus.SUCCESS),
    }


def test_the_repeated_attempt_check_reaches_a_gapped_number(tmp_path: Path) -> None:
    gapped, _, _ = _gapped_store(tmp_path)
    write_attempt_rows(
        tmp_path,
        gapped.alg_uuid,
        "run.1",
        [_row(gapped, 4, status="timeout", result=None, runtime_s=9.0)],
    )

    with pytest.raises(ValueError, match=f"attempt 4 at coordinate {gapped.uuid}"):
        view(tmp_path, SIMPLE_TASK, [gapped], columns=["status"])


def test_view_without_attempts_answers_nothing(tmp_path: Path) -> None:
    coordinate = _coordinate()

    empty = view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status", "result"])

    assert empty.num_rows == 0
    assert empty.column_names == ["coordinate_uuid", "status", "result"]
    attempts_directory(tmp_path).mkdir()
    assert view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status"]).num_rows == 0
    write_attempt_rows(
        tmp_path, coordinate.alg_uuid, "run.0", [_row(_coordinate(alg_seed=9))]
    )
    assert view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status"]).num_rows == 0
    assert view(tmp_path, SIMPLE_TASK, [], columns=["status"]).num_rows == 0


def test_store_shape_refuses_a_tree_holding_both_attempts_and_results(
    tmp_path: Path,
) -> None:
    attempts_directory(tmp_path).mkdir()
    results_directory(tmp_path).mkdir()
    with pytest.raises(ValueError, match="holds both Attempts and Results"):
        partitions.store_shape(tmp_path, SIMPLE_TASK)


def test_a_release_answering_one_cell_twice_names_the_coordinate(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate()
    budget = coordinate.budget.dump_config().model_dump()
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=budget,
        )
        for _ in range(2)
    ]
    write_result_partition(tmp_path, records)

    with pytest.raises(
        ValueError,
        match=(
            f"Instance {coordinate.instance_uuid!r}, algorithm "
            f"{coordinate.alg_uuid!r}, seed 0"
        ),
    ):
        view(tmp_path, SIMPLE_TASK, [coordinate], columns=["status"])
