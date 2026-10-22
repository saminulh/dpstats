"""Acquire typed current Results from a Task collection's Attempts and Results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget
from dpstats.instance import Instance
from dpstats.results._resolution import selected_rows
from dpstats.results._schema import result_schema
from dpstats.results._selection import SELECTION_ROW
from dpstats.results._outcomes import ResultStatus
from dpstats.task import Task

RESULT_COLUMNS = [
    "task",
    "instance",
    "instance_uuid",
    "algorithm",
    "alg_uuid",
    "budget",
    "alg_seed",
    "status",
    "payload",
    "runtime_s",
    "peak_rss_mb",
]


def _column_values(batch: pa.RecordBatch, name: str) -> Any:
    """Restore one column as the values a Task Output field holds.

    The stored schema types each Output field, so the column already carries
    that field's dtype and rank: a fixed-rank array arrives as one array per
    row and a scalar as itself, without revalidating either.
    """
    column = cast("pa.Array[Any]", batch.column(name))
    if not pa.types.is_list(column.type):
        return column.to_pylist()
    if not pa.types.is_list(column.type.value_type):
        return column.to_numpy(zero_copy_only=False)
    return [
        None if value is None else np.asarray(value) for value in column.to_pylist()
    ]


def results_frame(
    task_root: Path,
    instances: Mapping[str, Instance] | None = None,
    algorithms: Mapping[str, BaseAlgorithm] | None = None,
    *,
    seeds: Sequence[int] | None = None,
    groups: (
        Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]] | None
    ) = None,
) -> pd.DataFrame:
    """Acquire current rows once and restore their typed scientific values.

    ``groups`` narrows to exactly its factor selection instead of expanding
    every Instance-Algorithm combination in ``instances``/``algorithms``.
    """
    if groups is not None:
        instances = {instance.uuid: instance for instance, _, _ in groups}
        algorithms = {algorithm.uuid: algorithm for _, algorithm, _ in groups}
    if not instances or not algorithms:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    task: Task = next(iter(instances.values())).task
    rows, factor_table, seed_count = selected_rows(
        task_root,
        task,
        tuple(algorithms),
        tuple(instances),
        columns=result_schema(task).names,
        seeds=seeds,
        groups=groups,
    )
    if not rows.num_rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    factor_records: list[dict[str, Any]] = factor_table.to_pylist()
    budget_keys = {(row["budget_path"], row["budget_config"]) for row in factor_records}
    resolved_budgets = {
        (path, config): DPBudget.load_record(
            {"path": path, "config": json.loads(config)}
        )
        for path, config in budget_keys
    }
    factors = [
        (
            instances[row["instance_id"]],
            algorithms[row["alg_id"]],
            resolved_budgets[row["budget_path"], row["budget_config"]],
        )
        for row in factor_records
    ]
    output_names = tuple(task.Output.model_fields)
    resolved: dict[str, list[Any]] = {
        name: [] for name in ("instance", "algorithm", "budget")
    }
    group_positions = rows[SELECTION_ROW].to_numpy() // seed_count
    offsets = np.r_[0, np.flatnonzero(np.diff(group_positions)) + 1, rows.num_rows]
    observed_seeds = cast("list[int]", rows["seed"].to_pylist())
    for start, stop in zip(offsets[:-1], offsets[1:]):
        instance, algorithm, budget = factors[group_positions[start]]
        resolved["instance"].extend([instance] * (stop - start))
        resolved["algorithm"].extend([algorithm] * (stop - start))
        resolved["budget"].extend([budget] * (stop - start))

    statuses: list[ResultStatus] = []
    payloads: list[Any] = []
    for batch in rows.select(["status", "reason", *output_names]).to_batches(
        max_chunksize=1024
    ):
        values = {name: _column_values(batch, name) for name in batch.schema.names}
        for index in range(batch.num_rows):
            status = ResultStatus(values["status"][index])
            if status is ResultStatus.SUCCESS:
                payload = task.Output.model_construct(
                    **{name: values[name][index] for name in output_names}
                )
            else:
                # Every outcome that released no Output carries its reason, or
                # nothing, exactly as the row stores it.
                payload = values["reason"][index]
            statuses.append(status)
            payloads.append(payload)

    return pd.DataFrame(
        {
            **resolved,
            "task": task.id,
            "instance_uuid": rows["instance_id"].to_pylist(),
            "alg_uuid": rows["alg_id"].to_pylist(),
            "alg_seed": observed_seeds,
            "status": pd.Series(statuses, dtype=object),
            "payload": payloads,
            "runtime_s": rows["runtime_s"].to_pylist(),
            "peak_rss_mb": rows["peak_rss_mb"].to_pylist(),
        },
        columns=RESULT_COLUMNS,
    )
