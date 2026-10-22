"""Ensure a Distribution Release is installed at <workspace>/official-release/."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dpstats import fetch_benchmark


def parser() -> argparse.ArgumentParser:
    """Build the ``fetch`` command's argument parser."""
    command = argparse.ArgumentParser(
        description=(
            "Ensure the platform's Distribution Release is installed at "
            "<workspace>/official-release/. With no --revision, any already-"
            "installed release is left alone (a no-op); the pinned default "
            "is installed only if none is. With --revision, that revision "
            "is ensured, replacing a different one if present. Anonymous "
            "unless HF_TOKEN is set in the environment."
        )
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace to fetch into (path, required)",
    )
    command.add_argument(
        "--revision",
        help="Tag or immutable revision to ensure; omit to ensure any release",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public ``fetch`` command."""
    parsed = parser().parse_args(argv)

    workspace = fetch_benchmark(
        Path(parsed.workspace),
        revision=parsed.revision,
    )
    print(workspace)


if __name__ == "__main__":
    main()
