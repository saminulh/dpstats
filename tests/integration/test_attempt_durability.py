"""Durability of completed ResultAttempts across an external process kill."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from dpstats.workspace.layout import (
    PARTITION_PREFIX,
    attempts_directory,
    task_collection_directory,
    workspace_local_root,
)
from tests.attempts import read_attempt_rows
from tests.integration.killed_sweep import SEEDS, command, staged_observations
from tests.registry_components import SIMPLE_TASK
from tests.sweep_helpers import importable_environment

pytestmark = pytest.mark.requires_unrestricted_processes


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"), reason="an external kill requires SIGKILL"
)
@pytest.mark.timeout(60)  # Sixteen paced serial cells across two fresh processes.
def test_external_kill_keeps_every_parent_observed_completion(tmp_path: Path) -> None:
    """SIGKILL loses no completed attempt the parent already observed.

    The run is killed with completions staged and nothing sealed, so under a
    seal-only durability promise every one of them would be discarded. The
    resumed run must surface those exact observations.
    """
    storage_root = tmp_path / "storage"
    task_store = task_collection_directory(
        workspace_local_root(storage_root), SIMPLE_TASK.id
    )
    attempts = attempts_directory(task_store)
    environment = importable_environment()

    process = subprocess.Popen(
        command("run", storage_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if attempts.is_dir() and len(staged_observations(attempts)) >= 2:
                break
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                pytest.fail(f"sweep exited before the kill: {stderr}")
            time.sleep(0.05)
        else:
            pytest.fail("no completed attempt was ever staged durably")

        staged = staged_observations(attempts)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert not list(attempts.glob(f"{PARTITION_PREFIX}*"))
    assert read_attempt_rows(task_store) == []

    resumed = subprocess.run(
        command("run", storage_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr

    recovered = read_attempt_rows(task_store)
    journaled = {
        (observation["coordinate"]["alg_seed"], observation["attempt_number"])
        for observation in staged
    }
    assert journaled <= {(row["seed"], row["attempt"]) for row in recovered}
    assert sorted(row["seed"] for row in recovered) == list(range(SEEDS))
    assert not list(attempts.glob(".incomplete-*"))
