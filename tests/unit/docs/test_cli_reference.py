"""The command-line reference is the installed parsers, not authored prose."""

from __future__ import annotations

import re


from cli_reference import cli_reference

from dpstats.cli import COMMANDS

SECTION = re.compile(r"^## (.+)$", re.M)


def test_the_page_carries_one_section_per_installed_command() -> None:
    page = cli_reference()
    assert SECTION.findall(page) == list(COMMANDS)


def test_every_section_is_the_help_its_own_parser_prints() -> None:
    """The parser states the command, so the page carries nothing beside it."""
    page = cli_reference()
    for command in COMMANDS:
        section = page.split(f"## {command}\n", 1)[1].split("\n## ", 1)[0].strip()
        assert section.startswith("```text") and section.endswith("```"), command
        assert f"usage: dpstats {command}" in section


def test_the_page_is_the_same_whatever_terminal_the_build_runs_in(
    monkeypatch,
) -> None:
    """Help text wraps to a width this build fixes, not to the shell's."""
    monkeypatch.setenv("COLUMNS", "40")
    narrow = cli_reference()
    monkeypatch.setenv("COLUMNS", "200")
    assert cli_reference() == narrow
