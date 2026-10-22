"""Report one Sweep's coverage as JSON."""

import argparse
import json
from collections.abc import Sequence

from dpstats.workspace import TaskCollection
from dpstats.scripts.utils import (
    ALGORITHM_REGISTRY_HELP,
    INSTANCE_REGISTRY_HELP,
    add_task_definition_argument,
)


def parser() -> argparse.ArgumentParser:
    """Build the ``get-sweep-status`` command's argument parser."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--sweep",
        required=True,
        help="Path to the authored sweep JSON file (required)",
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace whose Results the coverage counts (path, required)",
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
        "--results-source",
        choices=("local", "release", "both"),
        default="both",
        help="Which installed tree(s)' Results this counts as coverage (default: both)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public get-sweep-status command."""
    args = parser().parse_args(argv)
    collection = TaskCollection.from_sweep(
        args.sweep,
        args.workspace,
        task=args.task_definition,
    )
    print(
        json.dumps(
            collection.get_sweep_status(
                args.sweep,
                algorithm_registries=args.algorithm_registry,
                instance_registries=args.instance_registry,
                results_source=args.results_source,
            )
        )
    )
