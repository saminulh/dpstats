"""End-to-end behavior of the installed ``get-sweep-status`` command."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

from dpstats.budgets import PureDPBudget
from dpstats.workspace.layout import (
    sweeps_directory,
    task_collection_directory,
)
from tests.attempts import current_result, write_attempt_partition
from tests.registry_components import SIMPLE_TASK, SimpleAlgorithm, sized_producer
from tests.sweep_helpers import (
    cli_args,
    local_task_root,
    sweep_coordinates,
    write_sweep,
)
from dpstats.workspace import TaskCollection


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from dpstats.cli import main; main()",
            "get-sweep-coordinates" if "--output" in args else "get-sweep-status",
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_command_reports_live_current_actions(
    tmp_path: Path, collection: TaskCollection
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, sweep)
    write_attempt_partition(
        local_task_root(collection),
        [
            current_result(
                instance_uuid=coordinate.instance_uuid,
                alg_uuid=coordinate.alg_uuid,
                budget=coordinate.budget.dump_config(),
            )
        ],
    )

    completed = _run_command(
        *cli_args(sweep, tmp_path),
        "--algorithm-registry",
        "test",
        "--instance-registry",
        "test",
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["cells"]["recorded"] == 1
    assert summary["cells"]["expected"] == 1
    assert "coordinate_uuid" not in completed.stdout


def test_command_exports_complete_preview_csv_to_file_and_refuses_overwrite(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    output = tmp_path / "preview.csv"

    completed = _run_command(
        *cli_args(sweep, tmp_path),
        "--algorithm-registry",
        "test",
        "--instance-registry",
        "test",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert set(rows[0]) == {
        "coordinate_uuid",
        "task",
        "alg_uuid",
        "instance_uuid",
        "budget",
        "alg_seed",
        "time_limit",
        "memory_limit_gb",
    }

    refused = _run_command(
        *cli_args(sweep, tmp_path),
        "--output",
        str(output),
    )
    assert refused.returncode == 1
    assert "File exists" in refused.stderr


def test_command_exports_clean_csv_to_stdout(
    tmp_path: Path, collection: TaskCollection
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    completed = _run_command(
        *cli_args(sweep, tmp_path),
        "--output",
        "-",
    )

    assert completed.returncode == 0, completed.stderr
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    assert len(rows) == 1
    assert rows[0]["coordinate_uuid"].startswith("c3-")


def test_command_reports_an_invalid_sweep_without_a_traceback(tmp_path: Path) -> None:
    task_root = tmp_path / "collections"
    sweep = (
        sweeps_directory(task_collection_directory(task_root, SIMPLE_TASK.id))
        / "invalid.json"
    )
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({"task": SIMPLE_TASK.id, "unknown": True}))

    completed = _run_command("--sweep", str(sweep), "--workspace", str(task_root))

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
