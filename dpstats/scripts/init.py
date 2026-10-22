"""Create a fresh workspace, or confirm an existing one is ready to use."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dpstats import init_workspace


def parser() -> argparse.ArgumentParser:
    """Build the ``init`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Create <path>/local/, idempotently; refuse a path that "
        "exists and is not a workspace."
    )
    command.add_argument(
        "--path", required=True, help="Workspace directory (path, required)"
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public ``init`` command."""
    parsed = parser().parse_args(argv)
    workspace = init_workspace(parsed.path)
    print(workspace)


if __name__ == "__main__":
    main()
