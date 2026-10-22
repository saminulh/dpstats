"""Subprocess driver for the tests that kill a running Sweep from outside.

Run as ``python -m tests.integration.killed_sweep <entry-point> <workspace>``.
Every entry point first installs the same paced Sweep, so a killed run, a
reader, a reconciliation and a resumed run all see one recipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dpstats.budgets import PureDPBudget
from dpstats.sweep import preview_sweep
from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.workspace.layout import workspace_local_root
from tests.registry_components import SIMPLE_TASK, PacedAlgorithm, constant_producer

SEEDS = 16


def _install(workspace: str) -> tuple[TaskCollection, Path]:
    """Install a half-second algorithm and one Instance, and save their Sweep."""
    collection = open_collection(init_workspace(workspace), SIMPLE_TASK)
    collection.add_algorithm(PacedAlgorithm(delay_s=0.5))
    collection.add_instance(*constant_producer(reference=2.0)(0))
    sweep = collection.save_sweep(
        "test",
        {"budgets": [PureDPBudget(eps=1.0)], "seeds": list(range(SEEDS))},
        overwrite=True,
    )
    return collection, sweep


def run(workspace: str) -> None:
    """Fill every cell of the Sweep serially."""
    collection, sweep = _install(workspace)
    collection.run_sweep(sweep)


def preview(workspace: str) -> None:
    """Print each seed's standing claim as JSON, deciding nothing."""
    collection, sweep = _install(workspace)
    frame = preview_sweep(
        sweep, workspace_local_root(collection.workspace), SIMPLE_TASK
    )
    print(json.dumps(sorted(zip(frame["alg_seed"], frame["claim"]))))


def reconcile(workspace: str) -> None:
    """Print the coordinates reconciliation cleared as JSON."""
    collection, _ = _install(workspace)
    print(json.dumps(collection.reconcile()))


def command(entry_point: str, workspace: Path) -> list[str]:
    """Return the argv that runs one entry point in a fresh interpreter."""
    return [
        sys.executable,
        "-m",
        "tests.integration.killed_sweep",
        entry_point,
        str(workspace),
    ]


def staged_observations(attempts: Path) -> list[dict[str, Any]]:
    """Return every complete observation a writer journaled but never sealed."""
    records: list[dict[str, Any]] = []
    for journal in sorted(attempts.glob(".incomplete-*.jsonl")):
        payload = journal.read_bytes()
        end = payload.rfind(b"\n")
        for line in payload[: end + 1].splitlines():
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    entry_point, workspace = sys.argv[1:]
    {"run": run, "preview": preview, "reconcile": reconcile}[entry_point](workspace)
