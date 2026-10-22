"""Reconciliation of a killed sweep's in-flight cells, losing nothing."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from dpstats.workspace.layout import (
    PARTITION_PREFIX,
    attempts_directory,
    claims_directory,
    task_collection_directory,
    workspace_local_root,
)
from tests.attempts import read_attempt_rows
from tests.integration.killed_sweep import SEEDS, command, staged_observations
from tests.registry_components import SIMPLE_TASK
from tests.sweep_helpers import importable_environment

pytestmark = pytest.mark.requires_unrestricted_processes


def _claims(claims: Path) -> dict[str, dict[str, Any]]:
    """Read the claims a dead runner left, decoded independently of production."""
    if not claims.is_dir():
        return {}
    return {
        path.name: json.loads(path.read_text())
        for path in sorted(claims.iterdir())
        if not path.name.startswith(".incomplete-")
    }


def _identity(record: dict[str, Any]) -> tuple[int, int]:
    return record["coordinate"]["alg_seed"], record["attempt_number"]


def _row_identity(row: dict[str, Any]) -> tuple[int, int]:
    return row["seed"], row["attempt"]


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"), reason="an external kill requires SIGKILL"
)
@pytest.mark.timeout(30)  # Paced serial runs plus four fresh Python processes.
def test_a_killed_sweep_reconciles_its_in_flight_cells_and_loses_nothing(
    tmp_path: Path,
) -> None:
    """SIGKILL leaves every cell either recorded or unanswered.

    A reader reports surviving claims without deciding them. Cleanup preserves
    journaled outcomes and clears every claim without executing anything: an
    interruption is not an outcome, so a cell the kill took is unanswered. A
    later run fills every cell, so nothing is lost and nothing is invented.
    """
    storage_root = tmp_path / "storage"
    task_store = task_collection_directory(
        workspace_local_root(storage_root), SIMPLE_TASK.id
    )
    attempts = attempts_directory(task_store)
    claims = claims_directory(task_store)
    environment = importable_environment()

    def invoke(entry_point: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command(entry_point, storage_root),
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
            check=False,
        )

    process = subprocess.Popen(
        command("run", storage_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if (
                attempts.is_dir()
                and len(staged_observations(attempts)) >= 2
                and _claims(claims)
            ):
                break
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                pytest.fail(f"sweep exited before the kill: {stderr}")
            time.sleep(0.05)
        else:
            pytest.fail("no cell was ever both journaled and claimed")

        # Kill the whole session: the cell workers die with their parent, so
        # nothing can add to the store after this point.
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    staged = staged_observations(attempts)
    orphaned = _claims(claims)
    assert orphaned, "the kill left no in-flight cell to reconcile"
    assert not list(attempts.glob(f"{PARTITION_PREFIX}*"))

    previewed = invoke("preview")
    assert previewed.returncode == 0, previewed.stderr
    claimed_seeds = {claim["coordinate"]["alg_seed"] for claim in orphaned.values()}
    reported = {seed: text for seed, text in json.loads(previewed.stdout)}
    assert {seed for seed, text in reported.items() if text} == claimed_seeds
    for claim in orphaned.values():
        assert str(claim["pid"]) in reported[claim["coordinate"]["alg_seed"]]
    # A reader decides nothing: the claims and the journal are as they were.
    assert _claims(claims) == orphaned
    assert not list(attempts.glob(f"{PARTITION_PREFIX}*"))

    settled = invoke("reconcile")
    assert settled.returncode == 0, settled.stderr

    reconciled = read_attempt_rows(task_store)
    journaled_attempts = {_identity(observation) for observation in staged}
    claimed_attempts = {_identity(claim) for claim in orphaned.values()}
    # Every journaled outcome became a sealed Attempt record, and every claim was cleared
    # without inventing a record for the cell it held.
    assert {_row_identity(row) for row in reconciled} == journaled_attempts
    assert len(reconciled) == len(staged)
    assert sorted(json.loads(settled.stdout)) == sorted(
        {claim["coordinate_uuid"] for claim in orphaned.values()}
    )
    assert claimed_attempts - journaled_attempts
    assert not list(claims.iterdir())

    completed = invoke("run")
    assert completed.returncode == 0, completed.stderr

    current = {
        row["seed"]: row
        for row in sorted(read_attempt_rows(task_store), key=lambda r: r["attempt"])
    }
    assert len(current) == SEEDS
    assert {row["status"] for row in current.values()} == {"success"}
    assert sorted(current) == list(range(SEEDS))
    assert not list(claims.iterdir())
    assert not list(attempts.glob(".incomplete-*"))
