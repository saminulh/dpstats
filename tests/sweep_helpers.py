"""Small authored-sweep builder for tests of the public run boundary."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

import dpstats
import numpy as np

from dpstats.algorithm import BaseAlgorithm
from dpstats.domains import BoxDomain
from dpstats.instance import Instance, persist_instance, register_instance
from dpstats.task import Task, TaskInput, TaskMetadata
from dpstats.budgets import NO_PRIVACY, DPBudget
from dpstats.workspace import TaskCollection
from dpstats.coordinates import CellCoordinate
from dpstats.results import ResultAttempt
from dpstats.sweep import (
    BudgetPolicy,
    CellContext,
    CellScheduler,
    schedule_cells_serially,
)
from dpstats.sweep.preparation import PreparedCell
from dpstats.sweep.coordinates import load_sweep_coordinates
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    task_collection_directory,
    workspace_local_root,
)


def observing_schedule(observe: Callable[[CellContext], None]) -> CellScheduler:
    """Schedule serially, showing each offered job to ``observe`` before it runs."""

    def schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        def observed(context: CellContext) -> PreparedCell | ResultAttempt:
            observe(context)
            return execute(context)

        schedule_cells_serially(initial, observed, complete)

    return schedule


def importable_environment(module_root: Path | None = None) -> dict[str, str]:
    """Make the library, these test components, and one scratch module importable.

    A test that drives a real subprocess needs the package under test on the
    path even when it is not installed, and the driver script it wrote in a
    temporary directory too.
    """
    package_root = str(Path(dpstats.__file__).resolve().parents[1])
    environment = dict(os.environ)
    roots = (
        None if module_root is None else str(module_root),
        package_root,
        environment.get("PYTHONPATH"),
    )
    environment["PYTHONPATH"] = os.pathsep.join(path for path in roots if path)
    return environment


# A producer is called with its ``seed`` by keyword, as the public simulations
# declare it.
InstanceProducer = Callable[
    ..., tuple[TaskMetadata, TaskInput, Mapping[str, JsonValue]]
]


def write_sweep(
    collection: TaskCollection,
    algorithms: Sequence[BaseAlgorithm],
    producers: Sequence[InstanceProducer],
    budgets: Sequence[DPBudget | BudgetPolicy],
    *,
    seeds: Sequence[int] = (0,),
    instance_seeds: Sequence[int] = (0,),
    time_limit: float | None = None,
    memory_limit_gb: float | None = None,
    name: str = "test",
) -> Path:
    """Install test components through the public verbs and save their Sweep.

    Every Algorithm and every (producer, seed) Instance joins the ``test``
    group; the saved Sweep selects all installed objects and returns its path.
    """
    for algorithm in algorithms:
        collection.add_algorithm(algorithm, registry="test")
    add_instances(collection, producers, seeds=instance_seeds)
    return collection.save_sweep(
        name,
        {
            "budgets": list(budgets),
            "seeds": list(seeds),
            "time_limit": time_limit,
            "memory_limit_gb": memory_limit_gb,
        },
    )


def add_instances(
    collection: TaskCollection,
    producers: Sequence[InstanceProducer],
    *,
    seeds: Sequence[int] = (0,),
    registry: str = "test",
) -> tuple[str, ...]:
    """Install one Instance per (producer, seed) and return their UUIDs."""
    return tuple(
        collection.add_instance(*producer(seed=seed), registry=registry).uuid
        for producer in producers
        for seed in seeds
    )


def local_task_root(collection: TaskCollection) -> Path:
    """Return this collection's own Task directory in ``local/``.

    Attempts and claims are written beneath it, so the independent readers in
    ``tests/attempts.py`` and the claim store address it directly.
    """
    return task_collection_directory(
        workspace_local_root(collection.workspace), collection.task.id
    )


def sweep_coordinates(
    collection: TaskCollection, sweep: Path, **selection: Any
) -> tuple[CellCoordinate, ...]:
    """Resolve a saved Sweep's typed coordinates against this collection's store."""
    return load_sweep_coordinates(
        sweep,
        workspace_local_root(collection.workspace),
        collection.task,
        **selection,
    )


def install_instances(
    tree_root: Path,
    task: Task,
    producers: Sequence[InstanceProducer],
    *,
    seeds: Sequence[int] = (0,),
    registry: str = "test",
) -> tuple[str, ...]:
    """Register Instances directly in a tree a collection cannot write.

    A bare release bundle refuses every collection writer, so a test that
    stages release-only declarations files them into the bundle's own tree,
    with each Input beside it in the tree's ``input-artifacts/``.
    """
    uuids: list[str] = []
    for producer in producers:
        for seed in seeds:
            metadata, input_data, references = producer(seed=seed)
            assert metadata.task is task
            instance = persist_instance(
                metadata,
                input_data,
                references,
                Path(tree_root) / INPUT_ARTIFACTS_DIRECTORY,
            )
            uuids.append(register_instance(tree_root, instance, registry=registry))
    return tuple(uuids)


SIMPLE_TASK_DEFINITION = "tests.registry_components:SIMPLE_TASK"


def cli_args(sweep: Path, workspace: Path, *options: str) -> list[str]:
    """Return a sweep command's arguments for a ``SIMPLE_TASK`` Sweep, then *options*.

    ``SIMPLE_TASK`` is not built in, so every command names its import path.
    """
    return [
        "--sweep",
        str(sweep),
        "--workspace",
        str(workspace),
        "--task-definition",
        SIMPLE_TASK_DEFINITION,
        *options,
    ]


def write_mean_benchmark_sweep(
    collection: TaskCollection,
    algorithm: BaseAlgorithm,
    *,
    budgets: Sequence[DPBudget | BudgetPolicy] = (NO_PRIVACY,),
    values: Sequence[float] = (1.0, 3.0),
) -> Path:
    """Install an own algorithm beside a "benchmark" Mean Estimation Instance."""
    collection.add_algorithm(algorithm, registry="own")
    collection.add_instance(
        MeanEstimationMetadata(
            n_samples=len(values),
            domain=BoxDomain(bounds=np.array([[0.0, 4.0]])),
        ),
        MeanEstimationInput(X=np.asarray(values, dtype=float).reshape(-1, 1)),
        {
            "ground_truth": [float(np.mean(values))],
            "sample_mean": [float(np.mean(values))],
        },
        registry="benchmark",
    )
    return collection.save_sweep("extension", {"budgets": list(budgets), "seeds": [0]})
