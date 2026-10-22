"""Run or resume one authored sweep in its task collection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from operator import attrgetter
from pathlib import Path

from dpstats.workspace import TaskCollection
from dpstats.results import ResultStatus
from dpstats.scripts.utils import (
    ALGORITHM_REGISTRY_HELP,
    INSTANCE_REGISTRY_HELP,
    add_task_definition_argument,
)


def parser() -> argparse.ArgumentParser:
    """Build the ``run-sweep`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Fill missing sweep cells and return all selected Results."
    )
    command.add_argument(
        "--sweep",
        required=True,
        help="Path to the authored sweep JSON file (required)",
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace to run in; results run against local/ (path, required)",
    )
    command.add_argument(
        "--algorithm-registry",
        action="append",
        help=ALGORITHM_REGISTRY_HELP,
    )
    command.add_argument(
        "--instance-registry",
        action="append",
        help=INSTANCE_REGISTRY_HELP,
    )
    add_task_definition_argument(command)
    command.add_argument(
        "--include-output",
        action="store_true",
        help="Include complete Outputs in the returned table",
    )
    command.add_argument(
        "--mode",
        choices=("missing", "retry_resources", "all"),
        default="missing",
        help=(
            "Which cells to run: missing fills unanswered cells, "
            "retry_resources also retries timeout and memory failures, "
            "all reruns every cell (default: missing)"
        ),
    )
    command.add_argument(
        "--verbose",
        action="store_true",
        help="Show a progress bar over the attempts (default: off)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public ``run-sweep`` command."""
    parsed = parser().parse_args(argv)

    collection = TaskCollection.from_sweep(
        Path(parsed.sweep),
        Path(parsed.workspace),
        task=parsed.task_definition,
    )
    results = collection.run_sweep(
        Path(parsed.sweep),
        mode=parsed.mode,
        verbose=parsed.verbose,
        algorithm_registries=parsed.algorithm_registry,
        instance_registries=parsed.instance_registry,
    )
    if not parsed.include_output:
        results = results.drop(columns=["payload"])
    display = results.assign(
        instance=results["instance"].map(attrgetter("uuid")),
        algorithm=results["algorithm"].map(str),
        budget=results["budget"].map(repr),
        status=results["status"].map(lambda status: ResultStatus(status).value),
    )
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
