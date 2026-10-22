"""Key a coordinate or factor selection and join it to scanned rows.

Each row carries its position in that selection; the store and release readers
index rows the same way, so nothing here resolves an Attempt.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget
from dpstats.coordinates import (
    FACTOR_KEYS,
    CellCoordinate,
    budget_key,
    canonical_budget_text,
    factor_selection_table,
)
from dpstats.instance import Instance
from dpstats.results._schema import (
    COORDINATE_COLUMN,
    COORDINATE_FIELD,
    FIXED_FIELDS,
    KEY_COLUMNS,
    PARTITION_COLUMN,
)
from dpstats.task import Task

SELECTION_ROW = "_selection_row"
ROW_INDEX = "_row"
_FACTOR_ROW = "_factor_row"


def require_columns(task: Task, stored: pa.Schema, columns: Sequence[str]) -> None:
    """Refuse a requested column the stored rows do not carry."""
    for name in columns:
        if name not in stored.names:
            raise ValueError(f"Task {task.id!r} has no column {name!r}.")


def empty_selection(stored: pa.Schema, columns: Sequence[str]) -> pa.Table:
    """Return the shape a selection answering no cell takes."""
    return (
        stored.empty_table()
        .select(list(columns))
        .add_column(0, SELECTION_ROW, pa.array([], pa.int64()))
    )


def initial_factors(
    stored: pa.Schema,
    groups: Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]] | None,
) -> pa.Table:
    """Return the supplied factor selection, or the empty one to observe into."""
    if groups is not None:
        return factor_selection_table(groups)
    return stored.empty_table().select(list(FACTOR_KEYS))


def factor_selection(
    keyed: pa.Table,
    factors: pa.Table,
    instance_ids: Sequence[str],
    *,
    seeds: Sequence[int] | None,
    groups: Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]] | None,
) -> tuple[pa.Table, pa.Table, int]:
    """Narrow keyed rows to the selected factors and index each row's position.

    A row's selection-row index is its factor's position on the factor axis
    times the seed-axis length, plus its seed's position, so one index orders
    the frame by Instance, algorithm, budget, and seed.
    """
    keyed = keyed.filter(
        pc.is_in(
            keyed["instance_id"], value_set=pa.array(instance_ids, type=pa.string())
        )
    )
    if seeds is None:
        observed_seeds = pc.unique(keyed["seed"])
        seed_values = pc.take(observed_seeds, pc.sort_indices(observed_seeds))
    else:
        seed_values = pa.array(np.asarray(tuple(dict.fromkeys(seeds)), dtype=np.uint64))
    keyed = keyed.filter(pc.is_in(keyed["seed"], value_set=seed_values))
    if groups is None:
        factors = (
            keyed.select(list(FACTOR_KEYS))
            .group_by(list(FACTOR_KEYS))
            .aggregate([])
            .sort_by([(name, "ascending") for name in FACTOR_KEYS])
        )
    selected = keyed.join(
        factors.append_column(
            _FACTOR_ROW, pa.array(np.arange(factors.num_rows), pa.int64())
        ),
        keys=list(FACTOR_KEYS),
        join_type="inner",
    )
    if not selected.num_rows:
        return selected, factors, len(seed_values)
    positions = (
        selected[_FACTOR_ROW].to_numpy() * len(seed_values)
        + pc.index_in(selected["seed"], value_set=seed_values).to_numpy()
    )
    return (
        selected.append_column(SELECTION_ROW, pa.array(positions, pa.int64())),
        factors,
        len(seed_values),
    )


def selection_table(coordinates: Iterable[CellCoordinate]) -> pa.Table:
    """Key the selection by natural key beside each coordinate's identity.

    A selection repeats few distinct budgets across many coordinates, so each
    budget's columns are derived once.
    """
    columns: dict[str, list[Any]] = {
        name: [] for name in (*KEY_COLUMNS, COORDINATE_COLUMN)
    }
    budgets: dict[DPBudget, tuple[str, str]] = {}
    for coordinate in coordinates:
        budget = coordinate.budget
        if budget not in budgets:
            budgets[budget] = budget_key(budget)
        budget_path, budget_config = budgets[budget]
        columns["instance_id"].append(coordinate.instance_uuid)
        columns[PARTITION_COLUMN].append(coordinate.alg_uuid)
        columns["budget_path"].append(budget_path)
        columns["budget_config"].append(budget_config)
        columns["seed"].append(coordinate.alg_seed)
        columns[COORDINATE_COLUMN].append(coordinate.uuid)
    fields = {field.name: field for field in FIXED_FIELDS}
    return pa.table(
        columns,
        schema=pa.schema([*(fields[name] for name in KEY_COLUMNS), COORDINATE_FIELD]),
    )


def keyed_rows(rows: pa.Table, *columns: str) -> pa.Table:
    """Return the key columns and named extras of each row beside its row index."""
    return canonical_budget_keys(
        rows.select([*KEY_COLUMNS, *columns]).append_column(
            ROW_INDEX, pa.array(np.arange(rows.num_rows), pa.int64())
        )
    )


def canonical_budget_keys(keys: pa.Table) -> pa.Table:
    """Replace recorded budget configurations with the factor they key.

    A row keeps the parameters its Attempt was executed with; a selection
    carries the same budget as resolved on this host. Both sides join on what
    those parameters name, so a row recorded where one resolved parameter
    differs in its last places still answers its cell.
    """
    recorded = keys.column("budget_config")
    distinct = pc.unique(recorded)
    canonical = pa.array(
        [
            canonical_budget_text(text)
            for text in cast("list[str]", distinct.to_pylist())
        ],
        pa.string(),
    )
    return keys.set_column(
        keys.schema.get_field_index("budget_config"),
        "budget_config",
        pc.take(canonical, pc.index_in(recorded, value_set=distinct)),
    )
