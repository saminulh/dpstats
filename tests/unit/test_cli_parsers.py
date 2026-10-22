"""The user CLI's command declarations, parsers and in-process dispatch."""

from pathlib import Path
from typing import Any

import pytest

import dpstats.cli as cli
from dpstats.cli import COMMAND_DESCRIPTIONS, COMMAND_PARSERS, COMMANDS
from dpstats.scripts import (
    fetch,
    generate_task,
    get_sweep_coordinates,
    get_sweep_status,
    init,
    measure_sweep,
    reconcile_sweep,
    run_sweep,
    submission,
)


def test_every_command_is_advertised_once_by_each_declaration() -> None:
    """One command, one entry in each of the three declarations, in one order."""
    assert list(COMMAND_PARSERS) == list(COMMANDS)
    assert list(COMMAND_DESCRIPTIONS) == list(COMMANDS)


def test_a_parser_is_reachable_without_running_its_command() -> None:
    for command, build_parser in COMMAND_PARSERS.items():
        parser = build_parser()
        parser.prog = f"dpstats {command}"
        assert parser.format_usage().startswith(f"usage: dpstats {command}"), command


def test_fetch_command_dispatches_the_anonymous_python_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, Any] = {}
    workspace = tmp_path / "benchmark"

    def fetch_release(workspace_arg: Path, **kwargs: Any) -> Path:
        received["workspace"] = workspace_arg
        received.update(kwargs)
        return workspace_arg

    monkeypatch.setattr(fetch, "fetch_benchmark", fetch_release)

    fetch.main(["--workspace", str(workspace), "--revision", "release-v1"])

    assert received == {"workspace": workspace, "revision": "release-v1"}
    assert capsys.readouterr().out == f"{workspace}\n"


def test_cli_dispatcher_routes_every_advertised_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one exact command inventory, each name routed to its handler."""
    expected = {
        "init": init.main,
        "measure-sweep": measure_sweep.main,
        "fetch": fetch.main,
        "get-sweep-status": get_sweep_status.main,
        "get-sweep-coordinates": get_sweep_coordinates.main,
        "run-sweep": run_sweep.main,
        "reconcile-sweep": reconcile_sweep.main,
        "generate-task": generate_task.main,
        "prepare-submission": submission.prepare_main,
        "verify-submission": submission.verify_main,
    }
    assert COMMANDS == expected

    received: list[tuple[str, list[str]]] = []
    for command in expected:
        monkeypatch.setitem(
            COMMANDS,
            command,
            lambda args, command=command: received.append((command, args)),
        )
        monkeypatch.setattr(
            cli.sys,
            "argv",
            ["dpstats", command, "--sentinel"],
        )
        cli.main()

    assert received == [(command, ["--sentinel"]) for command in expected]
