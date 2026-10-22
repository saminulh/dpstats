"""Write the files of a Distribution Release, for previews and tests.

A release on disk is a recipe per task, its sealed Results, its standings
view and instance weights, a protocol, a narrative and a manifest. The page-template previews
(``templates/filler_results.py``) and the docs tests assemble small releases
from these writers, so a change to that shape lands here once.
"""

from __future__ import annotations

import math

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dpstats.budgets import PureDPBudget
from dpstats.results import result_files
from dpstats.sweep.coordinates import load_sweep_coordinates
from dpstats.tasks import BUILT_IN_TASKS
from dpstats.transport.canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
    INSTANCE_WEIGHTS_NAME,
    RELEASE_MANIFEST_NAME,
    SWEEPS_DIRECTORY,
    WITHHELD_PROFILES_NAME,
    attempts_directory,
    claims_directory,
    result_partition_directory,
    results_directory,
    task_collection_directory,
)

from panel_data import PROFILE_VIEW_COLUMNS, PROFILE_VIEW_SCHEMA


def sweep_relative(task_id: str) -> str:
    """The recipe path a synthetic release declares for *task_id*."""
    return f"{task_id}/{SWEEPS_DIRECTORY}/simulated.json"


def withheld_profiles_relative(task_id: str) -> str:
    """The path a synthetic release declares for *task_id*'s profile view."""
    return f"{task_id}/{BENCHMARK_INFO_DIRECTORY}/{WITHHELD_PROFILES_NAME}"


def instance_weights_relative(task_id: str) -> str:
    """The path a synthetic release declares for *task_id*'s weights view."""
    return f"{task_id}/{INSTANCE_WEIGHTS_NAME}"


def write_weights_view(root: Path, task_id: str, weights: Mapping[str, float]) -> Path:
    """Write one task's instance weights view, one exact decimal per instance."""
    return write_view_csv(
        root / instance_weights_relative(task_id),
        ("instance", "weight"),
        ((uuid, repr(float(weight))) for uuid, weight in weights.items()),
    )


def write_sweep(
    path: Path, task_id: str, epsilons: Sequence[float], seeds: Sequence[int]
) -> Path:
    """Write one pure-DP recipe at *path* with no resource limits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": task_id,
                "budgets": [
                    dict(PureDPBudget(eps=eps).dump_config()) for eps in epsilons
                ],
                "seeds": list(seeds),
                "time_limit": None,
                "memory_limit_gb": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def seal_released_results(root: Path, task_id: str, sweep: Path) -> int:
    """Resolve a store's recorded Attempts into the Results a release ships.

    A release carries one row per cell under ``results/`` and no Attempt
    history, so the Attempts the sweep recorded are resolved once here and the
    store they lived in is removed.

    Returns:
        The number of released Result rows.
    """
    task = BUILT_IN_TASKS[task_id]
    task_root = task_collection_directory(root, task_id)
    coordinates = load_sweep_coordinates(sweep, root, task)
    files, answered = result_files(task_root, task, coordinates)
    for alg_uuid, data in files.items():
        path = result_partition_directory(task_root, alg_uuid) / "results.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shutil.rmtree(attempts_directory(task_root))
    shutil.rmtree(claims_directory(task_root), ignore_errors=True)
    return len(answered)


def write_view_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Sequence[str]]
) -> Path:
    """Write one sanctioned view: a header of *columns*, then *rows*, as CSV text.

    Every field is written as given, so a caller states each value in the exact
    text the view publishes.
    """
    lines = [",".join(columns), *(",".join(row) for row in rows)]
    _write_text(path, "\n".join(lines) + "\n")
    return path


def write_profile_view(
    path: Path, rows: Iterable[Sequence[Any]], *, quantile: float = 0.9
) -> Path:
    """Write one withheld-profiles view: *rows* in order, typed by its schema.

    Each field is written as given, converted to its column's type, so a
    caller may state a threshold or fraction as its exact decimal text. A row
    of every column states all of them (``None`` or NaN is a null). A shorter
    row (the six names, then threshold, fraction and run counts, optionally its
    three headline values) is in the "all" slice of one Instance and one
    study; with no headline values, they stand in for the release builder's
    exact reading, taken from the curve's own points: the first threshold
    reaching *quantile*, the last fraction, and the first threshold reaching
    that.
    """
    full = len(PROFILE_VIEW_COLUMNS)
    given: list[tuple[Any, ...]] = []
    for row in rows:
        row = tuple(row)
        if len(row) < full:
            stated_reading = row[10:13] if len(row) >= 13 else (math.nan,) * 3
            row = (*row[:6], "all", *row[6:10], *stated_reading, 1, 1)
        given.append(row)
    curves: dict[tuple[str, ...], list[tuple[float, float]]] = {}
    for row in given:
        group = tuple(str(row[index]) for index in (2, 3, 4, 5, 6))
        curves.setdefault(group, []).append((float(row[7]), float(row[8])))
    derived: dict[tuple[str, ...], tuple[float | None, float, float | None]] = {}
    for group, points in curves.items():
        points.sort()
        share = points[-1][1]
        derived[group] = (
            next((tau for tau, fraction in points if fraction >= quantile), None),
            share,
            (
                next((tau for tau, fraction in points if fraction >= share), None)
                if share > 0
                else None
            ),
        )

    def number(value: Any) -> float | None:
        return None if value is None or math.isnan(float(value)) else float(value)

    def reading(row: tuple[Any, ...]) -> tuple[float | None, ...]:
        # A row stating no share reached states no reading.
        if number(row[12]) is None:
            return derived[tuple(str(row[index]) for index in (2, 3, 4, 5, 6))]
        return tuple(number(value) for value in row[11:14])

    readings = [reading(row) for row in given]
    columns = [
        *(
            pa.array([str(row[index]) for row in given], pa.string())
            for index in range(7)
        ),
        *(
            pa.array([float(row[index]) for row in given], pa.float64())
            for index in (7, 8)
        ),
        *(
            pa.array([int(row[index]) for row in given], pa.int64())
            for index in (9, 10)
        ),
        *(
            pa.array([values[index] for values in readings], pa.float64())
            for index in range(3)
        ),
        *(
            pa.array([int(row[index]) for row in given], pa.int64())
            for index in (14, 15)
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(columns, schema=PROFILE_VIEW_SCHEMA),
        path,
        compression="zstd",
        compression_level=9,
    )
    return path


# The pooled baseline every released ranking states.
POOLED_BASELINE = {"kind": "best_median", "aggregation": "median", "pool": "standard"}


def protocol_entry(
    headline_metric: str,
    *,
    headline_baseline: str = "best_median",
    headline_quantile: float = 0.9,
    tau_grid: Mapping[str, Any] | None = None,
    external: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One released task's protocol entry: its resolved ranking and its externals.

    A headline baseline other than the pooled one resolves to the value each
    instance stores under that key, as the released encoding states it.
    """
    return {
        "ranking": {
            "headline_metric": headline_metric,
            "headline_baseline": headline_baseline,
            "headline_definition": (
                dict(POOLED_BASELINE)
                if headline_baseline == "best_median"
                else {"kind": "reference", "key": headline_baseline}
            ),
            "headline_quantile": headline_quantile,
            "tau_grid": dict(
                tau_grid or {"min": 0.0, "max": 10.0, "scale": "linear", "steps": 1001}
            ),
            "weighting": "provenance_ref",
            "eligibility": "standard",
            "baselines": {"best_median": dict(POOLED_BASELINE)},
        },
        "external_baselines": {
            name: dict(record) for name, record in (external or {}).items()
        },
    }


def write_protocol(root: Path, comparison: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write the Benchmark protocol a release ships, stating *comparison*."""
    path = root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes({"comparison": dict(comparison)}))
    return path


def write_narrative(root: Path, text: str) -> Path:
    """Write the protocol narrative a release ships."""
    path = root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_NARRATIVE_NAME
    _write_text(path, text)
    return path


def write_release_manifest(
    root: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    files: Mapping[str, str],
    *,
    version: str,
) -> Path:
    """Write a Distribution Release manifest as canonical JSON with its own id."""
    manifest = {
        "version": version,
        "definition_id": "f" * 64,
        "previous_manifest_id": None,
        "tasks": {
            task: (
                dict(record)
                if "results" in record
                else {**record, "results": _fixture_results(root, task)}
            )
            for task, record in tasks.items()
        },
        "files": sorted(files),
    }
    path = root / RELEASE_MANIFEST_NAME
    path.write_bytes(
        canonical_json_bytes(
            {**manifest, "manifest_id": canonical_json_sha256(manifest)}
        )
        + b"\n"
    )
    return path


def _fixture_results(root: Path, task: str) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {"distribution": {}, "evaluation": {}}
    task_root = task_collection_directory(root, task)
    for path in sorted(results_directory(task_root).glob("alg_id=*/*.parquet")):
        uuid = path.parent.name.removeprefix("alg_id=")
        count = pq.ParquetFile(path).metadata.num_rows
        states["distribution"].setdefault(uuid, {"expected": 0})["expected"] += count
    profiles = root / withheld_profiles_relative(task)
    if profiles.is_file():
        groups: dict[str, dict[tuple[str, str], int]] = {}
        for row in pq.read_table(profiles).to_pylist():
            groups.setdefault(row["alg_uuid"], {})[(row["metric"], row["budget"])] = (
                row["run_count"]
            )
        for uuid, counts in groups.items():
            metric = next(iter(counts))[0]
            states["evaluation"][uuid] = {
                "expected": sum(
                    count for (name, _), count in counts.items() if name == metric
                )
            }
    for lane, algorithms in states.items():
        for uuid, state in algorithms.items():
            state.update(
                state="ready",
                recorded=state["expected"],
                reason=None,
                evidence_id=canonical_json_sha256(["fixture", task, uuid, lane]),
                implementation=["f" * 40],
            )
    return states


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
