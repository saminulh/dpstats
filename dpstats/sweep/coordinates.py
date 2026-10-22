"""Resolve Sweeps into one factored selection and its coordinates, and let an operator validate one Sweep against its Task."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow as pa

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
    load_algorithms,
    load_algorithms_partitioned,
)
from dpstats.budgets import DPBudget, NO_PRIVACY
from dpstats.coordinates import CellCoordinate, factor_selection_table
from dpstats.instance import (
    Instance,
    load_instances,
    load_instances_partitioned,
)
from dpstats.task import Task
from dpstats.transport.registry import skipped_note
from dpstats.workspace.registries import (
    installed_registry_names,
    read_selection_multi,
)
from dpstats.workspace.layout import task_collection_directory
from dpstats.sweep._sweep import Sweep, read_sweep
from dpstats.sweep.budget_policies import BudgetPolicy

# Concrete budgets to keep, or a predicate on each concrete budget.
BudgetSelection = Sequence[DPBudget] | Callable[[DPBudget], bool]

SweepGroup = tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]


@dataclass(frozen=True)
class ResolvedSweepGroups:
    """The exact grid and declared budget labels from one policy resolution."""

    groups: tuple[SweepGroup, ...]
    budget_groups: Mapping[tuple[str, DPBudget], DPBudget | BudgetPolicy]


def sweep_selection_table(
    groups: Sequence[SweepGroup],
    seeds: Sequence[int],
) -> pa.Table:
    """Expand selected factors into expected coordinates in Sweep order."""
    base = factor_selection_table(groups)
    labels = {algorithm.uuid: str(algorithm) for _, algorithm, _ in groups}
    base = base.append_column(
        "algorithm", pa.array([labels[row["alg_id"]] for row in base.to_pylist()])
    )
    unique_seeds = tuple(dict.fromkeys(seeds))
    repeated = base.take(np.repeat(np.arange(base.num_rows), len(unique_seeds)))
    return repeated.append_column(
        "seed",
        pa.array(np.tile(np.asarray(unique_seeds, dtype=np.uint64), base.num_rows)),
    )


@dataclass(frozen=True)
class ResolvedSelection:
    """One resolved selection consumed by results, status, coordinates, and execution.

    ``groups`` is ``None`` exactly when the Sweep declares no budgets, and
    ``seeds`` when it declares no seeds; the unresolved UUIDs were requested
    but found neither live nor installed.
    """

    algorithms: Mapping[str, BaseAlgorithm]
    instances: Mapping[str, Instance]
    groups: tuple[SweepGroup, ...] | None
    seeds: tuple[int, ...] | None
    time_limit: float | None
    memory_limit_gb: float | None
    budget_groups: Mapping[tuple[str, DPBudget], DPBudget | BudgetPolicy]
    unresolved_algorithms: tuple[str, ...]
    unresolved_instances: tuple[str, ...]

    def coordinates(self) -> tuple[CellCoordinate, ...]:
        """Expand the finite grid into its distinct coordinates, empty when unbounded."""
        return expand_sweep_groups(self.groups or (), self.seeds or ())


def require_task(declared: str, task: Task) -> None:
    """Refuse a Sweep declared for a Task other than *task*."""
    if declared != task.id:
        raise ValueError(f"Sweep belongs to Task {declared!r}, not {task.id!r}.")


def require_bound_task(
    values: Iterable[BaseAlgorithm | Instance], task: Task, subject: str
) -> None:
    """Refuse a *subject* value whose Task object is not the live *task*."""
    for value in values:
        if value.task is not task:
            raise ValueError(f"{subject} belongs to another Task.")


def registry_entries(
    task_roots: Sequence[Path],
    kind: Literal["algorithm", "instance"],
    registries: str | Sequence[str] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Read and merge the selected ``kind`` registries across the given trees.

    The one place a registry selection becomes entries; every reader of
    installed declarations goes through it. Returns the entries beside the
    registry file each one was read from, so a loader can name that file.
    """
    merged, sources = read_selection_multi(
        task_roots,
        kind,
        registries,
        reconstruct=(
            BaseAlgorithm.load_record if kind == "algorithm" else Instance.load_config
        ),
        derive_uuid=lambda value: value.uuid,
        native_task_id=lambda value: value.task.id,
    )
    return merged.entries, sources


def _entries_for(
    task_roots: Sequence[Path],
    kind: Literal["algorithm", "instance"],
    requested: Sequence[str] | None,
    live: Mapping[str, Any],
    registries: str | Sequence[str] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Read only the installed entries a selection still needs after live objects.

    Returns the entries beside the registry file each one was read from.
    """
    needed = None if requested is None else set(requested) - set(live)
    if needed is not None and not needed:
        return {}, {}
    if registries is None and not installed_registry_names(task_roots, kind):
        return {}, {}
    entries, sources = registry_entries(task_roots, kind, registries)
    if needed is not None:
        entries = {uuid: value for uuid, value in entries.items() if uuid in needed}
    return entries, sources


def _choose(
    requested: Sequence[str] | None,
    installed: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    registry_order: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the requested objects, live before installed, and the unresolved UUIDs.

    Objects follow the request, or the registries' own order under ``registry_order``.
    """
    if requested is None:
        return dict(installed), ()
    wanted = dict.fromkeys(requested)
    chosen = {uuid: live.get(uuid, installed.get(uuid)) for uuid in wanted}
    found = {uuid: value for uuid, value in chosen.items() if value is not None}
    if registry_order:
        found = {uuid: found[uuid] for uuid in (*installed, *live) if uuid in found}
    return found, tuple(uuid for uuid, value in chosen.items() if value is None)


def resolve_selection(
    sweep: Sweep,
    task: Task,
    task_roots: Sequence[Path],
    *,
    live_algorithms: Mapping[str, BaseAlgorithm] | None = None,
    live_instances: Mapping[str, Instance] | None = None,
    algorithm_registries: str | Sequence[str] | None = None,
    instance_registries: str | Sequence[str] | None = None,
    allow_unresolved: bool = False,
    strict: bool = True,
    budgets: BudgetSelection | None = None,
    seeds: Sequence[int] | None = None,
    registry_order: bool = False,
) -> ResolvedSelection:
    """Resolve one Sweep against live objects and the installed registries.

    The sole reader of registries into a selection. A requested UUID is taken
    from ``live_*`` first, then from the registries of ``task_roots`` (Task
    directories, local before release); one this library cannot rebuild is
    refused, or skipped with a warning under ``strict=False``. A UUID found nowhere
    is refused unless ``allow_unresolved``. ``budgets`` and ``seeds`` narrow
    the resulting concrete grid, never the declarations, and refuse a value
    outside it; ``budgets`` may instead be a predicate on each concrete budget.
    """
    require_task(sweep.task, task)
    live_algorithms = live_algorithms or {}
    live_instances = live_instances or {}
    algorithm_entries, algorithm_sources = _entries_for(
        task_roots,
        "algorithm",
        sweep.algorithms,
        live_algorithms,
        algorithm_registries,
    )
    instance_entries, instance_sources = _entries_for(
        task_roots, "instance", sweep.instances, live_instances, instance_registries
    )
    algorithms, missing_algorithms = _choose(
        sweep.algorithms,
        load_algorithms(
            algorithm_entries,
            task_id=task.id,
            strict=strict,
            sources=algorithm_sources,
        ),
        live_algorithms,
        registry_order=registry_order,
    )
    instances, missing_instances = _choose(
        sweep.instances,
        load_instances(
            instance_entries,
            task_id=task.id,
            strict=strict,
            sources=instance_sources,
        ),
        live_instances,
        registry_order=registry_order,
    )
    require_bound_task(
        (*algorithms.values(), *instances.values()), task, "Installed entry"
    )
    if (missing_algorithms or missing_instances) and not allow_unresolved:
        raise ValueError(
            f"Unresolved algorithms: {list(missing_algorithms)}; "
            f"unresolved instances: {list(missing_instances)}."
            + skipped_note(
                missing_algorithms,
                lambda: load_algorithms_partitioned(
                    algorithm_entries, task_id=task.id, sources=algorithm_sources
                )[1],
            )
            + skipped_note(
                missing_instances,
                lambda: load_instances_partitioned(
                    instance_entries, task_id=task.id, sources=instance_sources
                )[1],
            )
        )
    resolved = (
        None
        if sweep.budgets is None
        else resolve_selected_groups(algorithms, instances, sweep, budgets=budgets)
    )
    return ResolvedSelection(
        algorithms=algorithms,
        instances=instances,
        groups=None if resolved is None else resolved.groups,
        seeds=sweep.seeds if seeds is None else _selected_seeds(seeds, sweep.seeds),
        time_limit=sweep.time_limit,
        memory_limit_gb=sweep.memory_limit_gb,
        budget_groups={} if resolved is None else resolved.budget_groups,
        unresolved_algorithms=missing_algorithms,
        unresolved_instances=missing_instances,
    )


def resolve_sweep_selection(
    sweep: Sweep,
    task_collections_root: str | Path | Sequence[str | Path],
    task: Task,
    *,
    algorithm_registries: str | Sequence[str] | None = None,
    instance_registries: str | Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
    budgets: Sequence[DPBudget] | None = None,
    seeds: Sequence[int] | None = None,
    strict: bool = True,
) -> ResolvedSelection:
    """Resolve a portable Sweep, optionally narrowed, into its finite grid.

    ``task_collections_root`` is one tree, or several read local-then-release,
    the same union ``TaskCollection.get_sweep_coordinates`` reads. An
    unrestricted selection over nothing installed, and a private selection
    with no declared budget, are refused.
    """
    if sweep.budgets is None:
        raise ValueError(
            "A Sweep must declare its budgets to resolve coordinates; "
            "this Sweep declares none."
        )
    roots = (
        (task_collections_root,)
        if isinstance(task_collections_root, (str, Path))
        else tuple(task_collections_root)
    )
    narrowed = sweep.model_copy(
        update={
            name: tuple(ids)
            for name, ids in (
                ("algorithms", algorithm_ids),
                ("instances", instance_ids),
            )
            if ids is not None
        }
    )
    selection = resolve_selection(
        narrowed,
        task,
        tuple(task_collection_directory(root, sweep.task) for root in roots),
        algorithm_registries=algorithm_registries,
        instance_registries=instance_registries,
        strict=strict,
        budgets=budgets,
        seeds=seeds,
        registry_order=True,
    )
    for requested, chosen, kind in (
        (narrowed.algorithms, selection.algorithms, "algorithm"),
        (narrowed.instances, selection.instances, "Instance"),
    ):
        if requested is None and not chosen:
            raise ValueError(f"A Sweep requires at least one installed {kind}.")
    if (
        not narrowed.budgets
        and selection.seeds
        and selection.instances
        and any(
            algorithm.privacy_interpretation is not PrivacyInterpretation.NONPRIVATE
            for algorithm in selection.algorithms.values()
        )
    ):
        raise ValueError("A private Sweep must declare at least one typed budget.")
    return selection


def resolve_selected_groups(
    algorithms: Mapping[str, BaseAlgorithm],
    instances: Mapping[str, Instance],
    sweep: Sweep,
    *,
    budgets: BudgetSelection | None = None,
) -> ResolvedSweepGroups:
    """Build the one resolved execution grid from typed selections.

    The callers that read a portable Sweep and those that open a live
    collection share this boundary: registry and UUID selection has already
    happened, while budget policies still resolve once per selected Instance.
    ``budgets`` narrows the resulting concrete grid, never the declarations.
    An explicitly empty budget declaration selects nothing.
    """
    if sweep.budgets == ():
        return ResolvedSweepGroups((), {})
    declared_by_instance = {
        instance.uuid: sweep.budget_groups_for(instance)
        for instance in instances.values()
    }
    interpretations = {
        algorithm.privacy_interpretation for algorithm in algorithms.values()
    }
    if any(
        kind is not PrivacyInterpretation.NONPRIVATE for kind in interpretations
    ) and any(not declared for declared in declared_by_instance.values()):
        raise ValueError("A private Sweep must declare at least one typed budget.")
    budgets_by_instance = {
        uuid: tuple(budget for budget, _ in declared)
        for uuid, declared in declared_by_instance.items()
    }
    available_budgets = dict.fromkeys(
        budget for budgets in budgets_by_instance.values() for budget in budgets
    )
    if PrivacyInterpretation.NONPRIVATE in interpretations:
        available_budgets.setdefault(NO_PRIVACY, None)
    selected_budgets = set(_selected_budgets(budgets, tuple(available_budgets)))
    if not selected_budgets:
        return ResolvedSweepGroups((), {})
    budgets_by_instance = {
        uuid: tuple(budget for budget in available if budget in selected_budgets)
        for uuid, available in budgets_by_instance.items()
    }
    nonprivate_budgets = (NO_PRIVACY,) if NO_PRIVACY in selected_budgets else ()
    groups: list[SweepGroup] = []
    for instance, algorithm in product(instances.values(), algorithms.values()):
        selected = (
            nonprivate_budgets
            if algorithm.privacy_interpretation is PrivacyInterpretation.NONPRIVATE
            else budgets_by_instance[instance.uuid]
        )
        if selected:
            groups.append((instance, algorithm, selected))
    return ResolvedSweepGroups(
        tuple(groups),
        {
            (instance.uuid, budget): declared
            for instance in instances.values()
            for budget, declared in declared_by_instance[instance.uuid]
            if budget in selected_budgets
        },
    )


def _selected_seeds(
    requested: Sequence[int] | None, declared: Sequence[int] | None
) -> tuple[int, ...]:
    """Return selected seeds after refusing values outside the Sweep."""
    declared_unique = tuple(dict.fromkeys(declared or ()))
    if requested is None:
        return declared_unique
    unique = tuple(dict.fromkeys(requested))
    unknown = set(unique) - set(declared_unique)
    if unknown:
        names = ", ".join(str(seed) for seed in sorted(unknown))
        raise ValueError(f"Selected algorithm seed(s) are outside the Sweep: {names}.")
    return tuple(unique)


def _selected_budgets(
    requested: BudgetSelection | None, available: Sequence[DPBudget]
) -> tuple[DPBudget, ...]:
    """Return selected full budget identities, refusing undeclared budgets.

    A predicate keeps the budgets it accepts; an unconstrained budget, such as
    a nonprivate algorithm's, is kept without asking it.
    """
    if requested is None:
        return tuple(available)
    if callable(requested):
        return tuple(
            budget
            for budget in available
            if budget.is_unconstrained or requested(budget)
        )
    unique = tuple(dict.fromkeys(requested))
    available_set = set(available)
    unknown = [budget for budget in unique if budget not in available_set]
    if unknown:
        names = ", ".join(repr(budget) for budget in unknown)
        raise ValueError(f"Selected budget(s) are outside the resolved grid: {names}.")
    return tuple(unique)


def expand_sweep_groups(
    groups: Iterable[SweepGroup],
    seeds: Sequence[int],
) -> tuple[CellCoordinate, ...]:
    """Expand resolved factors into the distinct cells to execute."""
    coordinates: list[CellCoordinate] = []
    unique_seeds = tuple(dict.fromkeys(seeds))
    for instance, algorithm, selected in groups:
        for budget, alg_seed in product(selected, unique_seeds):
            coordinate = CellCoordinate(
                instance=instance,
                algorithm=algorithm,
                budget=budget,
                alg_seed=alg_seed,
            )
            coordinates.append(coordinate)
    return tuple(coordinates)


def load_sweep_coordinates(
    sweep_path: str | Path,
    task_collections_root: str | Path,
    task: Task,
    *,
    algorithm_registries: str | Sequence[str] | None = None,
    instance_registries: str | Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
    budgets: Sequence[DPBudget] | None = None,
    seeds: Sequence[int] | None = None,
    strict: bool = True,
) -> tuple[CellCoordinate, ...]:
    """Resolve one portable Sweep against the Task's installed registries."""
    return resolve_sweep_selection(
        read_sweep(sweep_path),
        task_collections_root,
        task,
        algorithm_registries=algorithm_registries,
        instance_registries=instance_registries,
        algorithm_ids=algorithm_ids,
        instance_ids=instance_ids,
        budgets=budgets,
        seeds=seeds,
        strict=strict,
    ).coordinates()


def validate_sweep(path: str | Path, *, expected_task: str | None = None) -> None:
    """Reject an invalid portable Sweep or an unexpected Task declaration."""
    sweep = read_sweep(path)
    if expected_task is not None and sweep.task != expected_task:
        raise ValueError(f"Sweep declares Task {sweep.task!r}, not {expected_task!r}.")
