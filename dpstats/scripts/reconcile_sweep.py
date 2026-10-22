"""Settle one task collection's unfinished work without executing a cell."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dpstats.workspace import TaskCollection
from dpstats.scripts.utils import add_task_definition_argument


def parser() -> argparse.ArgumentParser:
    """Build the ``reconcile-sweep`` command's argument parser."""
    command = argparse.ArgumentParser(
        description=(
            "Settle the claims a killed run left behind, without executing "
            "any cell. Do not run this against a task collection a "
            "collection run is still writing."
        )
    )
    command.add_argument(
        "--sweep",
        required=True,
        help="Path to the authored sweep JSON file (required)",
    )
    add_task_definition_argument(command)
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace to reconcile local/ in (path, required)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public ``reconcile-sweep`` command."""
    parsed = parser().parse_args(argv)

    collection = TaskCollection.from_sweep(
        Path(parsed.sweep),
        Path(parsed.workspace),
        task=parsed.task_definition,
    )
    cleared = collection.reconcile()
    if not cleared:
        print("Settled the task collection: no unreconciled attempt claim stood.")
        return
    print(
        f"Cleared {len(cleared)} abandoned claim(s). Each of these cells is "
        "unanswered unless this run recovered its journal:"
    )
    for coordinate_uuid in cleared:
        print(f"  {coordinate_uuid}")


if __name__ == "__main__":
    main()
