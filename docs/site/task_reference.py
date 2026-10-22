"""Read task and transformation declarations without constructing them."""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from importlib import import_module, util

from dpstats.algorithm import Transformation


def on_page_content(html, page, config, **kwargs):
    """Register task type references at their current definition sections."""
    from dpstats.tasks import BUILT_IN_TASKS

    slug, _, leaf = page.file.src_uri.partition("/")
    if slug not in BUILT_IN_TASKS:
        return html
    if leaf != "definition.md":
        return html
    task = BUILT_IN_TASKS[slug]
    autorefs = config.plugins["autorefs"]
    for section, model in (
        ("metadata", task.Metadata),
        ("input", task.Input),
        ("output", task.Output),
    ):
        for module in (model.__module__, f"dpstats.tasks.{slug}"):
            autorefs.register_anchor(page, f"{module}.{model.__name__}", section)
    return html


def exported_transformations(task_id: str) -> dict[str, type[Transformation]]:
    """Return the transformations one built-in task exports, by name.

    A task that exports none, or has no transformations module at all, returns an
    empty mapping: the module is the one declaration of what a task offers.

    Args:
        task_id: The built-in task whose transformations module to read.

    Returns:
        Each exported name, mapped to its declared transformation class.
    """
    module_name = f"dpstats.tasks.{task_id}.transformations"
    if util.find_spec(module_name) is None:
        return {}
    module = import_module(module_name)
    return {name: getattr(module, name) for name in module.__all__}


def outer_tasks(tasks: Sequence[str]) -> dict[str, list[str]]:
    """Return each task's outer tasks: where its native algorithms also run.

    A transformation declares the task it consumes (``OUTER_TASK``) and the task it
    prepares for (``INNER_TASK``), so an algorithm native to the inner task runs
    on the outer one through that transformation. Reaching an outer task makes that
    task's own outer tasks reachable too, so the relation is closed
    transitively: an M-estimation algorithm reaches linear regression directly
    and single-coefficient estimation through it.

    Args:
        tasks: The built-in task IDs to read, and the only IDs reported.

    Returns:
        Each task ID from *tasks*, mapped to its reachable outer task IDs in
        sorted order.
    """
    direct: dict[str, set[str]] = {task_id: set() for task_id in tasks}
    for task_id in tasks:
        for transformation in exported_transformations(task_id).values():
            inner, outer = transformation.INNER_TASK.id, transformation.OUTER_TASK.id
            if inner != outer and inner in direct:
                direct[inner].add(outer)
    reachable: dict[str, list[str]] = {}
    for task_id in tasks:
        found: set[str] = set()
        pending = list(direct[task_id])
        while pending:
            outer = pending.pop()
            if outer == task_id or outer in found:
                continue
            found.add(outer)
            pending.extend(direct.get(outer, ()))
        reachable[task_id] = sorted(found)
    return reachable


def task_statuses(
    reachable: Mapping[str, Sequence[str]],
    with_results: Container[str],
) -> dict[str, str]:
    """Return each task's standing in one build, derived and never authored.

    A task is Benchmarked when the release this build reads declares it;
    Auxiliary core when other tasks reach it through a transformation but the
    release does not declare it; and a Library task otherwise.

    Args:
        reachable: Each documented task ID mapped to its outer tasks, as
            :func:`outer_tasks` reads them.
        with_results: The task IDs the release this build reads declares.

    Returns:
        Each task ID mapped to ``"Benchmarked"``, ``"Auxiliary core"``, or
        ``"Library task"``.
    """
    statuses: dict[str, str] = {}
    for task_id in reachable:
        if task_id in with_results:
            statuses[task_id] = "Benchmarked"
        elif reachable[task_id]:
            statuses[task_id] = "Auxiliary core"
        else:
            statuses[task_id] = "Library task"
    return statuses
