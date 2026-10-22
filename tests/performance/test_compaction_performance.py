"""Opt-in timing of compacting a partition that pulls left three times over."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dpstats.results import compact_partition, partition_census
from dpstats.results._schema import attempt_schema
from dpstats.task import Task
from dpstats.tasks import BUILT_IN_TASKS
from dpstats.workspace.layout import attempt_partition_directory

pytestmark = pytest.mark.performance

_ROWS = 200_000
_ALG = "a3-PERFORMANCEPART1"
# Three copies hold three times the rows to read, so a ratio well above three
# means the repeats cost more than reading them does.
_RATIO_CEILING = 8.0


def _rows(task: Task) -> pa.Table:
    rng = np.random.default_rng(0)
    seeds = np.arange(_ROWS, dtype=np.uint64)
    estimates = rng.normal(size=(_ROWS, 3))
    return pa.table(
        {
            "instance_id": pa.array([f"i5-INSTANCE{n % 97:05d}" for n in range(_ROWS)]),
            "alg_id": pa.array([_ALG] * _ROWS),
            "budget_path": pa.array(["eps=1"] * _ROWS),
            "budget_config": pa.array(['{"eps":1.0}'] * _ROWS),
            "seed": pa.array(seeds),
            "attempt": pa.array(np.zeros(_ROWS, dtype=np.int32)),
            "status": pa.array(["success"] * _ROWS),
            "runtime_s": pa.array(rng.random(_ROWS)),
            "peak_rss_mb": pa.array(rng.random(_ROWS) * 100),
            "reason": pa.array([None] * _ROWS, pa.string()),
            "mean_est": pa.array(list(estimates), pa.list_(pa.float64())),
        }
    ).cast(attempt_schema(task))


def _partition(root: Path, rows: pa.Table, copies: int) -> Path:
    """Lay *rows* down as three files: three whole copies, or one copy split three ways."""
    partition = attempt_partition_directory(root, _ALG)
    partition.mkdir(parents=True)
    bounds = [0, len(rows) // 3, 2 * len(rows) // 3, len(rows)]
    for index in range(3):
        part = (
            rows
            if copies == 3
            else rows.slice(bounds[index], bounds[index + 1] - bounds[index])
        )
        pq.write_table(part, partition / f"compacted-copy-{index}.parquet")
    return partition


def _timed(root: Path, task: Task) -> float:
    started = perf_counter()
    compact_partition(root, task, _ALG)
    return perf_counter() - started


def test_compacting_three_identical_copies_costs_about_reading_them(
    tmp_path: Path,
) -> None:
    task = BUILT_IN_TASKS["mean_estimation"]
    rows = _rows(task)
    repeated: list[float] = []
    single: list[float] = []
    for sample in range(3):
        root = tmp_path / f"repeated-{sample}"
        _partition(root, rows, copies=3)
        repeated.append(_timed(root, task))
        census = partition_census(root, task, _ALG)
        assert (census.files, census.rows, census.distinct_rows) == (1, _ROWS, _ROWS)
        root = tmp_path / f"single-{sample}"
        _partition(root, rows, copies=1)
        single.append(_timed(root, task))
    ratio = statistics.median(repeated) / statistics.median(single)
    print(
        json.dumps(
            {
                "benchmark": "compact_partition_three_copies",
                "rows_per_copy": _ROWS,
                "three_copies_median_s": statistics.median(repeated),
                "one_copy_median_s": statistics.median(single),
                "observed_ratio": ratio,
                "ceiling": _RATIO_CEILING,
            },
            sort_keys=True,
        )
    )
    assert ratio < _RATIO_CEILING
