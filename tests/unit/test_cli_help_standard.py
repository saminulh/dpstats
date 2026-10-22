"""STYLE S28: every command has a one-line summary; every argument's help says
whether it is required and what its default is.

The checks read the parser objects, not the printed text. What a tool cannot
decide is the unit: whether an argument's help names the unit it is measured
in is judged in review, so the check covers the required and default parts of
the rule and the summary shape.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import pytest

from dpstats.cli import COMMAND_DESCRIPTIONS, COMMAND_PARSERS

# Flags whose presence is the whole argument: the default (off) and the
# optionality are self-evident, so only their help text must exist.
_FLAG_ACTIONS = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._HelpAction,
    argparse._VersionAction,
)


def _arguments(parser: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in parser._actions:
        if not isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
            yield action


def _parsers() -> list[tuple[str, argparse.ArgumentParser]]:
    return [(f"dpstats {name}", build()) for name, build in COMMAND_PARSERS.items()]


PARSERS = _parsers()
IDS = [name for name, _ in PARSERS]


def test_every_library_command_has_a_one_line_summary() -> None:
    for name, summary in COMMAND_DESCRIPTIONS.items():
        assert summary.strip() and "\n" not in summary, name


@pytest.mark.parametrize("name, parser", PARSERS, ids=IDS)
def test_every_argument_help_states_required_and_default(
    name: str, parser: argparse.ArgumentParser
) -> None:
    problems = []
    for action in _arguments(parser):
        label = "/".join(action.option_strings) or action.dest
        help_text = " ".join((action.help or "").split())
        if not help_text:
            problems.append(f"{label}: no help")
            continue
        lowered = help_text.lower()
        if isinstance(action, _FLAG_ACTIONS):
            continue
        required = action.required or not action.option_strings
        if required:
            if "required" not in lowered:
                problems.append(f"{label}: required but help does not say so")
        elif action.default not in (None, argparse.SUPPRESS):
            if "default" not in lowered and "%(default)" not in help_text:
                problems.append(f"{label}: default {action.default!r} not stated")
    assert not problems, f"{name}: " + "; ".join(problems)
