"""Export one Sweep's coordinates for external dispatch."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from dpstats.workspace import TaskCollection
from dpstats.scripts.utils import (
    ALGORITHM_REGISTRY_HELP,
    INSTANCE_REGISTRY_HELP,
    add_task_definition_argument,
)


def parser() -> argparse.ArgumentParser:
    """Build the ``get-sweep-coordinates`` command's argument parser."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--sweep",
        required=True,
        help="Path to the authored sweep JSON file (required)",
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace the Sweep runs in (path, required)",
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
        "--output",
        default="-",
        help="CSV file to write, which must not exist yet (default: - for standard output)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public get-sweep-coordinates command."""
    args = parser().parse_args(argv)
    collection = TaskCollection.from_sweep(
        args.sweep,
        args.workspace,
        task=args.task_definition,
    )
    frame = collection.get_sweep_coordinates(
        args.sweep,
        algorithm_registries=args.algorithm_registry,
        instance_registries=args.instance_registry,
    )
    if args.output == "-":
        frame.to_csv(sys.stdout, index=False)
    else:
        with Path(args.output).open("x", newline="") as output:
            frame.to_csv(output, index=False)
