"""Render the command-line reference from the installed parsers.

The page has no authored copy: every command, its one-line description and its
options come from ``dpstats.cli`` and the parser each command
builds, so a flag that changes in the library changes here in the same build.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from contextlib import contextmanager
from collections.abc import Iterator

from dpstats.cli import COMMAND_PARSERS

PROGRAM = "dpstats"

# Help text is wrapped to the terminal's width, which a build must not inherit
# from whatever shell started it.
HELP_WIDTH = "88"


@contextmanager
def _fixed_width() -> Iterator[None]:
    """Wrap help output at one width, whatever terminal this build runs in."""
    previous = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = HELP_WIDTH
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = previous


def command_help(command: str, parser: ArgumentParser) -> str:
    """One command's help text, as its own parser prints it."""
    parser.prog = f"{PROGRAM} {command}"
    with _fixed_width():
        return parser.format_help().rstrip()


def cli_reference() -> str:
    """The whole command-line reference page, generated from the parsers.

    Returns:
        Markdown: one section per installed command, in the order the library
        declares them, each carrying the help its own parser prints. The
        parser states the command's description, so the page shows it once.
    """
    sections = ["# CLI"]
    for command, build in COMMAND_PARSERS.items():
        sections.append(f"## {command}")
        sections.append(f"```text\n{command_help(command, build())}\n```")
    return "\n\n".join(sections) + "\n"
