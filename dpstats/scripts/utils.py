"""Argument resolution shared by the command-line scripts."""

from __future__ import annotations

import argparse
from difflib import get_close_matches
from typing import cast

from dpstats.task import Task
from dpstats.transport.configuration import load_object_from_path


ALGORITHM_REGISTRY_HELP = (
    "Installed algorithm registry name; repeat to add more "
    "(optional; default: every installed registry)"
)
INSTANCE_REGISTRY_HELP = (
    "Installed instance registry name; repeat to add more "
    "(optional; default: every installed registry)"
)


def add_task_definition_argument(command: argparse.ArgumentParser) -> None:
    """Add the optional qualified custom Task definition to a sweep command."""
    command.add_argument(
        "--task-definition",
        type=resolve_task_definition,
        help=(
            "Qualified custom Task value (module:object) for this Sweep "
            "(optional; default: the built-in Task the Sweep names)"
        ),
    )


def resolve_task_definition(path: str) -> Task:
    """Load and verify one module-qualified custom Task value for argparse."""
    task = load_object_from_path(path)
    if not isinstance(task, Task):
        raise argparse.ArgumentTypeError(f"{path!r} does not resolve to a Task value.")
    return cast(Task, task)


def resolve_task(task_id: str) -> Task:
    """Look ``task_id`` up in built-in discovery, suggesting a close match.

    A task-name typo is ordinary CLI misuse; it must name a task built-in
    discovery actually exposes, never surface as a bare ``KeyError`` traceback.
    """
    from dpstats.tasks import BUILT_IN_TASKS

    if task_id in BUILT_IN_TASKS:
        return BUILT_IN_TASKS[task_id]
    hint = get_close_matches(task_id, list(BUILT_IN_TASKS), n=1)
    suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
    raise ValueError(
        f"Unknown task {task_id!r}. Available tasks: "
        f"{', '.join(sorted(BUILT_IN_TASKS))}.{suggestion}"
    )
