"""Fresh-install smoke for a complete local simulated-instance sweep."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import textwrap
from pathlib import Path

import pytest

from dpstats.workspace.layout import (
    PARTITION_PREFIX,
    attempts_directory,
    registry_directory_name,
    sweeps_directory,
    task_collection_directory,
    workspace_local_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "tests" / "fixtures" / "simulated_instance_sweep"


def _run(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


@pytest.mark.requires_unrestricted_processes
@pytest.mark.slow
def test_fresh_install_example_interrupts_retries_measures_and_reruns(
    committed_library_source: None,
    tmp_path: Path,
) -> None:
    """The shipped example proves the complete replacement workflow."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    source_archive = tmp_path / "source.tar"
    archived = subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(source_archive), "HEAD"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert archived.returncode == 0, archived.stderr
    source_snapshot = tmp_path / "source"
    source_snapshot.mkdir()
    with tarfile.open(source_archive) as archive:
        archive.extractall(source_snapshot)
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(source_snapshot),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(wheels.glob("dpstats-*.whl"))

    venv = tmp_path / ".venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    python = venv / "bin" / "python"
    cli = venv / "bin" / "dpstats"
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(wheel),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    # Reuse the test runtime's third-party dependencies without exposing its
    # editable package finder; the assertion below proves the wheel wins.
    environment["PYTHONPATH"] = sysconfig.get_paths()["purelib"]

    example = tmp_path / "simulated_instance_sweep"
    shutil.copytree(
        source_snapshot / EXAMPLE.relative_to(REPO_ROOT),
        example,
    )
    shutil.copy2(
        source_snapshot / "tests" / "fixtures" / "external_configured_components.py",
        example / "external_configured_components.py",
    )
    shutil.copy2(
        source_snapshot / "tests" / "fixtures" / "external_project_smoke.py",
        example / "external_project_smoke.py",
    )
    shutil.copy2(
        source_snapshot / "tests" / "fixtures" / "installed_fetch_smoke.py",
        example / "installed_fetch_smoke.py",
    )

    imported = _run(
        [
            str(python),
            "-c",
            "import json; import dpstats as dp; "
            "from dpstats.analysis import measure_results; "
            "assert callable(measure_results); "
            "print(json.dumps({'library': dp.__file__}))",
        ],
        cwd=example,
        environment=environment,
    )
    installed_paths = json.loads(imported.stdout)
    assert Path(installed_paths["library"]).is_relative_to(venv)

    external = _run(
        [str(python), "external_project_smoke.py"],
        cwd=example,
        environment=environment,
    )
    assert json.loads(external.stdout.splitlines()[-1]) == {
        "algorithm": "ExternalAlgorithm",
        "transformation": "ExternalTransformation",
        "tuner": "ExternalExhaustiveTuner",
    }

    help_output = _run(
        [str(cli), "--help"],
        cwd=example,
        environment=environment,
    ).stdout
    assert {"get-sweep-coordinates", "run-sweep", "measure-sweep"} <= set(
        help_output.split()
    )
    assert {
        "evaluate",
        "run-experiment",
        "analyze-results",
    }.isdisjoint(help_output.split())

    documented_options = {
        "get-sweep-coordinates": {"--sweep"},
        "run-sweep": {
            "--sweep",
            "--mode",
        },
        "measure-sweep": {"--sweep", "--metric"},
    }
    for command, options in documented_options.items():
        installed_help = _run(
            [str(cli), command, "--help"],
            cwd=example,
            environment=environment,
        ).stdout
        assert options <= set(installed_help.replace(",", " ").split())

    _run([str(python), "prepare.py"], cwd=example, environment=environment)
    storage = example / "benchmark-storage"
    task_collection = task_collection_directory(
        workspace_local_root(storage), "mean_estimation"
    )
    sweep = sweeps_directory(task_collection) / "simulated.json"
    instance_groups = task_collection / registry_directory_name("instance")
    assert {path.stem for path in instance_groups.glob("*.json")} == {
        "real",
        "simulated",
    }

    collected = _run(
        [
            str(cli),
            "run-sweep",
            "--sweep",
            str(sweep),
            "--workspace",
            str(storage),
            "--algorithm-registry",
            "simulated",
            "--instance-registry",
            "simulated",
        ],
        cwd=example,
        environment=environment,
    )
    assert collected.stdout.count("success") == 3

    retryable = _run(
        [
            str(cli),
            "get-sweep-coordinates",
            "--sweep",
            str(sweep),
            "--workspace",
            str(storage),
            "--algorithm-registry",
            "simulated",
            "--instance-registry",
            "simulated",
            "--output",
            "-",
        ],
        cwd=example,
        environment=environment,
    )
    rows = list(csv.DictReader(retryable.stdout.splitlines()))
    assert len(rows) == 3
    assert all(row["coordinate_uuid"].startswith("c3-") for row in rows)

    measurements = _run(
        [
            str(cli),
            "measure-sweep",
            "--sweep",
            str(sweep),
            "--workspace",
            str(storage),
            "--algorithm-registry",
            "simulated",
            "--instance-registry",
            "simulated",
            "--metric",
            "l2_error_to_ground_truth",
        ],
        cwd=example,
        environment=environment,
    )
    # One measured row per collected cell, under the Metric's own named column.
    assert measurements.stdout.count("success") == 3
    assert "l2_error_to_ground_truth" in measurements.stdout

    (example / "custom_metric.py").write_text(
        textwrap.dedent(
            """\
            import numpy as np

            from dpstats.metric import Metric
            from dpstats.tasks import MeanEstimation


            def _mean_error(outputs, ground_truth):
                return float(
                    np.mean(
                        [
                            np.linalg.norm(
                                output.mean_est - np.asarray(ground_truth)
                            )
                            for output in outputs
                        ]
                    )
                )


            def _custom_l2_error(output, *, ground_truth, input, outputs):
                return float(
                    np.linalg.norm(output.mean_est - np.asarray(ground_truth))
                    + 0 * np.sum(input.X)
                    + _mean_error(outputs, ground_truth)
                )


            custom_l2_error = Metric(
                task=MeanEstimation,
                name="custom_l2_error",
                calculation=_custom_l2_error,
                requires=("ground_truth", "input", "outputs"),
            )
            """
        )
    )
    custom_measurements = _run(
        [
            str(cli),
            "measure-sweep",
            "--sweep",
            str(sweep),
            "--workspace",
            str(storage),
            "--algorithm-registry",
            "simulated",
            "--instance-registry",
            "simulated",
            "--metric",
            "custom_metric:custom_l2_error",
        ],
        cwd=example,
        environment=environment,
    )
    assert custom_measurements.stdout.count("success") == 3
    assert "custom_l2_error" in custom_measurements.stdout

    attempts = attempts_directory(task_collection)
    files_before = {
        path.name for path in attempts.glob(f"{PARTITION_PREFIX}*/*.parquet")
    }
    report = _run([str(python), "report.py"], cwd=example, environment=environment)
    assert json.loads(report.stdout.splitlines()[-1]) == {
        "coordinates": 3,
        "measurements": 3,
        "coverage": {
            "expected": 3,
            "recorded": 3,
            "resource_failures": 0,
            "never_attempted": 0,
            "recorded_percent": 100.0,
        },
        "statuses": {"success": 3},
    }
    assert {
        path.name for path in attempts.glob(f"{PARTITION_PREFIX}*/*.parquet")
    } == files_before

    fetched = _run(
        [str(python), "installed_fetch_smoke.py"],
        cwd=example,
        environment=environment,
    )
    assert json.loads(fetched.stdout.splitlines()[-1]) == {
        "downloaded_at_pinned_revision": True,
        "installed": "fetched-benchmark",
        "release_manifest_installed": True,
        "tree_matches_local_cut": True,
    }
