"""The one builder tests use to construct a real workspace or a bare bundle.

Every tree this module writes goes through the public workspace verbs
(``init_workspace``, ``TaskCollection.add_instance``/``add_algorithm``/
``save_sweep``/``run_sweep``) or, for the bytes a Distribution Release ships
that no public verb writes (the manifest, released Result partitions), the
same path vocabulary ``dpstats.workspace.layout`` exposes to
production code. Callers describe what a test needs — how many tasks,
instances, algorithms, whether a release — never a path fragment.

``tests/sweep_helpers.py`` keeps the per-test authoring helpers
(``write_sweep``, ``add_instances``, ``install_instances``) that install a
collection's objects through the same verbs; this module is the one home for
assembling a whole workspace or bare release bundle.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import JsonValue

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, NO_PRIVACY
from dpstats.results import result_files
from dpstats.sweep.coordinates import load_sweep_coordinates
from dpstats.task import Task, TaskInput, TaskMetadata
from dpstats.transport.canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.workspace.layout import (
    RELEASE_MANIFEST_NAME,
    attempts_directory,
    claims_directory,
    result_partition_directory,
    results_directory,
    task_collection_directory,
    workspace_local_root,
)

from tests.registry_components import SIMPLE_TASK, SimpleAlgorithm, simple_producer

InstanceProducer = Callable[
    [int], tuple[TaskMetadata, TaskInput, Mapping[str, JsonValue]]
]

# The sweep name every built workspace's saved recipe carries, one per task.
_SWEEP_NAME = "built"


@dataclass(frozen=True)
class BuiltTask:
    """One built Task's installed objects and, when run, its Results.

    ``sweep_path`` points into the workspace that built it.
    """

    task: Task
    algorithm_uuids: tuple[str, ...]
    instance_uuids: tuple[str, ...]
    sweep_path: Path
    results: pd.DataFrame | None


@dataclass(frozen=True)
class BuiltWorkspace:
    """A whole workspace this module built, and a bare bundle beside it."""

    root: Path
    tasks: tuple[BuiltTask, ...]
    bundle_root: Path | None = None

    def collection(self, task_id: str) -> TaskCollection:
        """Reopen this workspace's collection for one built Task."""
        return open_collection(self.root, self._task(task_id))

    def bundle_collection(self, task_id: str) -> TaskCollection:
        """Open this build's bare bundle's collection for one built Task."""
        assert (
            self.bundle_root is not None
        ), "build_workspace(release=True) was not asked for."
        return open_collection(self.bundle_root, self._task(task_id))

    def _task(self, task_id: str) -> Task:
        """Return the one live Task this test build installed under ``task_id``."""
        for built in self.tasks:
            if built.task.id == task_id:
                return built.task
        raise ValueError(f"This test workspace has no Task {task_id!r}.")


def _default_algorithms(task: Task) -> tuple[BaseAlgorithm, ...]:
    if task is SIMPLE_TASK:
        return (SimpleAlgorithm(),)
    raise ValueError(
        f"No default algorithm for task {task.id!r}; pass algorithms explicitly."
    )


def _default_producer(task: Task) -> InstanceProducer:
    if task is SIMPLE_TASK:
        return simple_producer()
    raise ValueError(
        f"No default instance producer for task {task.id!r}; pass instances explicitly."
    )


def _release_task(root: Path, task: Task, sweep_path: Path) -> None:
    """Resolve one built Task's Attempts into the released Result shape.

    A release carries one row per answered cell under ``results/`` and no
    Attempt history, so this seals what ``run_sweep`` recorded and removes the
    store it lived in — the same reduction production's release-build path
    performs, using the same layout helpers.
    """
    task_root = task_collection_directory(root, task.id)
    coordinates = load_sweep_coordinates(sweep_path, root, task)
    files, _ = result_files(task_root, task, coordinates)
    for alg_uuid, data in files.items():
        path = result_partition_directory(task_root, alg_uuid) / "results.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shutil.rmtree(attempts_directory(task_root), ignore_errors=True)
    shutil.rmtree(claims_directory(task_root), ignore_errors=True)
    results_directory(task_root).mkdir(parents=True, exist_ok=True)


def build_workspace(
    tmp_path: Path,
    *,
    tasks: Sequence[Task] = (SIMPLE_TASK,),
    instances: int | Mapping[str, Sequence[InstanceProducer]] = 1,
    algorithms: int | Mapping[str, Sequence[BaseAlgorithm]] = 1,
    seeds: Sequence[int] = (0,),
    budgets: Sequence[DPBudget] = (NO_PRIVACY,),
    release: bool = False,
    results: bool = True,
) -> BuiltWorkspace:
    """Build a real workspace (and, optionally, a bare bundle) once.

    Every installed object is registered through the public
    ``TaskCollection`` verbs; a saved sweep selects them by UUID. When
    ``results`` is true (the default), the sweep is run so the workspace
    carries real Attempts and Results, not just declarations.

    Args:
        tmp_path: The directory the workspace's ``local/`` is created under.
        tasks: The Tasks to install. ``SIMPLE_TASK`` (with one default
            algorithm and one default instance) needs no further arguments.
        instances: Either a count of default instances per task, or an
            explicit map from task id to the producer functions to install.
        algorithms: Either ``1`` (install one default algorithm; only
            meaningful for ``SIMPLE_TASK``) or a map from task id to the
            distinct configured Algorithm objects to install. A count greater
            than one is refused: repeating one configured object installs the
            same UUID more than once rather than distinct algorithms.
        seeds: Mechanism seeds the saved (and, if ``results``, run) sweep uses.
        budgets: Typed budgets the saved (and, if ``results``, run) sweep uses.
        release: Also build a bare Distribution Release bundle from the same
            installed objects, with a manifest written through the same
            layout helpers a real release uses. Implies ``results=True``.
        results: Run the saved sweep so the workspace holds real Results.

    Returns:
        The built workspace: its root, one ``BuiltTask`` per installed Task
        (installed UUIDs, the saved sweep path, and Results when run), and
        the bundle root when ``release`` was asked for.
    """
    if release:
        results = True
    workspace = init_workspace(tmp_path / "workspace")
    built_tasks: list[BuiltTask] = []
    for task in tasks:
        collection = open_collection(workspace, task)
        task_algorithms: Sequence[BaseAlgorithm]
        if isinstance(algorithms, Mapping):
            task_algorithms = algorithms[task.id]
        elif algorithms == 1:
            task_algorithms = _default_algorithms(task)
        else:
            raise ValueError(
                "algorithms as a count only supports 1 (a default algorithm "
                "object repeated N times would install the same UUID N "
                "times); pass an explicit {task.id: [...]} map instead."
            )
        algorithm_uuids = tuple(
            collection.add_algorithm(algorithm, registry="test")
            for algorithm in task_algorithms
        )

        task_producers: Sequence[InstanceProducer]
        if isinstance(instances, Mapping):
            task_producers = instances[task.id]
        else:
            task_producers = (_default_producer(task),) * instances
        installed = [
            collection.add_instance(*producer(seed), registry="test")
            for seed, producer in enumerate(task_producers)
        ]
        instance_uuids = tuple(instance.uuid for instance in installed)

        sweep_path = collection.save_sweep(
            _SWEEP_NAME,
            {
                "algorithms": list(algorithm_uuids),
                "instances": list(instance_uuids),
                "budgets": list(budgets),
                "seeds": list(seeds),
            },
        )
        frame: pd.DataFrame | None = None
        if results:
            frame = collection.run_sweep(sweep_path)
        built_tasks.append(
            BuiltTask(
                task=task,
                algorithm_uuids=algorithm_uuids,
                instance_uuids=instance_uuids,
                sweep_path=sweep_path,
                results=frame,
            )
        )

    bundle_root: Path | None = None
    if release:
        bundle_root = tmp_path / "bundle"
        shutil.copytree(
            workspace_local_root(workspace),
            bundle_root,
        )
        manifest_tasks: dict[str, Any] = {}
        files: dict[str, str] = {}
        digest = "d" * 64
        for built in built_tasks:
            _release_task(bundle_root, built.task, built.sweep_path)
            relative_sweep = built.sweep_path.relative_to(
                workspace_local_root(workspace)
            )
            files[str(relative_sweep)] = digest
            manifest_tasks[built.task.id] = {"sweep": str(relative_sweep)}
        payload = {
            "version": "built-v1",
            "tasks": manifest_tasks,
            "files": files,
        }
        manifest_id = canonical_json_sha256(payload)
        (bundle_root / RELEASE_MANIFEST_NAME).write_bytes(
            canonical_json_bytes({**payload, "manifest_id": manifest_id}) + b"\n"
        )

    return BuiltWorkspace(
        root=workspace, tasks=tuple(built_tasks), bundle_root=bundle_root
    )
