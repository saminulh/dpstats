"""Clean-interpreter smoke tests for the installed package surface."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import dpstats.cli as cli


def test_every_declared_export_resolves_in_a_fresh_interpreter() -> None:
    """Each module that curates ``__all__`` imports cleanly and binds every name.

    Which names a module exports, and where they live, is the author-surface
    and reference-surface tests' concern; this checks only that the installed
    package delivers what each module declares.
    """
    code = """
import importlib
import json
import pkgutil

import dpstats

unresolved = {}
declaring = 0
for info in pkgutil.walk_packages(dpstats.__path__, "dpstats."):
    module = importlib.import_module(info.name)
    names = getattr(module, "__all__", None)
    if names is None:
        continue
    declaring += 1
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        unresolved[info.name] = missing
print(json.dumps({"declaring": declaring, "unresolved": unresolved}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["declaring"] > 0
    assert report["unresolved"] == {}


def test_task_catalog_import_keeps_implementation_rosters_lazy() -> None:
    """The built-in Task facade loads contracts but no implementation roster."""
    code = """
import json
import sys

import dpstats.tasks

loaded = sorted(
    name
    for name in sys.modules
    if (
        name.startswith("dpstats.tasks.")
        and (".algorithms" in name or name.endswith("._metrics"))
    )
    or name in {
        "dpstats.simulations.linear_regression",
        "dpstats.simulations.logistic_regression",
        "dpstats.simulations.m_estimation",
        "dpstats.simulations.mean_estimation",
        "dpstats.simulations.median_estimation",
        "dpstats.simulations.simple_linear_regression",
        "dpstats.simulations.single_coefficient_estimation",
    }
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_package_root_import_does_not_load_the_task_catalog() -> None:
    """Root lifecycle imports stay independent of built-in Task discovery."""
    code = """
import json
import sys

import dpstats

loaded = sorted(
    name
    for name in sys.modules
    if name == "dpstats.tasks"
    or name.startswith("dpstats.tasks.mean_estimation")
    or name.startswith("dpstats.tasks.median_estimation")
    or name.startswith("dpstats.tasks.linear_regression")
    or name.startswith("dpstats.tasks.logistic_regression")
    or name.startswith("dpstats.tasks.m_estimation")
    or name.startswith("dpstats.tasks.simple_linear_regression")
    or name.startswith("dpstats.tasks.single_coefficient_estimation")
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    "module",
    [
        "dpstats.tasks.linear_regression.transformations",
        "dpstats.tasks.mean_estimation.transformations",
        "dpstats.tasks.median_estimation.transformations",
        "dpstats.tasks.m_estimation.algorithms",
    ],
)
def test_a_task_catalog_module_imports_first_in_a_clean_interpreter(
    module: str,
) -> None:
    """A Task's transformations or algorithms import with no prior package import."""
    code = (
        f"import importlib\n"
        f"mod = importlib.import_module({module!r})\n"
        f"assert all(hasattr(mod, name) for name in mod.__all__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cli_dispatches_generate_task_from_an_unrelated_directory(
    tmp_path: Path,
) -> None:
    """The executable module dispatches one representative user command."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dpstats.cli",
            "generate-task",
            "--task-name",
            "smoke_task",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "smoke_task" / "__init__.py").is_file()
    assert (tmp_path / "smoke_task" / "types.py").is_file()


def test_cli_help_has_the_complete_command_inventory() -> None:
    """The installed user CLI advertises exactly its supported commands."""
    result = subprocess.run(
        [sys.executable, "-m", "dpstats.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert {
        line.strip().split(maxsplit=1)[0]
        for line in result.stdout.splitlines()
        if line.startswith("  ")
    } == set(cli.COMMANDS)


def test_cli_reports_user_errors_without_tracebacks(tmp_path: Path) -> None:
    """CLI misuse produces a one-line error and exit 1, never a traceback."""
    missing_bundle = subprocess.run(
        [
            sys.executable,
            "-m",
            "dpstats.cli",
            "verify-submission",
            "missing",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_bundle.returncode == 1
    assert missing_bundle.stderr.startswith("error:")
    assert "Traceback" not in missing_bundle.stderr
