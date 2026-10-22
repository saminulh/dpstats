"""Opt-in timing of current-Result resolution over a store of repeated files.

A store that holds every partition as several identical compacted copies must
cost within a small multiple of one copy, and agreeing repeats are settled
without encoding each row.
"""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest

import dpstats.results._resolution as resolution
from dpstats.budgets import PureDPBudget
from dpstats.coordinates import CellCoordinate
from dpstats.instance import Instance
from dpstats.results._resolution import view
from dpstats.transport.canonical import canonical_json_bytes
from tests.attempts import write_attempt_rows
from tests.registry_components import SIMPLE_TASK, SimpleAlgorithm, SimpleMetadata

pytestmark = pytest.mark.performance

COPIES = 3
RETRIES = 5
# Three copies read three times the rows, so a ceiling well under the per-row
# encoding it replaces still leaves room for the extra scan.
RATIO_CEILING = 6.0


def _cells() -> list[CellCoordinate]:
    return [
        CellCoordinate(
            instance=Instance(
                metadata=SimpleMetadata(n_samples=1, dim=1),
                input_digest=f"{instance:064x}",
                metric_arguments={},
            ),
            algorithm=SimpleAlgorithm(scale=scale),
            budget=PureDPBudget(eps=1.0),
            alg_seed=seed,
        )
        for scale in (1.0, 2.0, 3.0, 4.0)
        for instance in range(50)
        for seed in range(20)
    ]


def _row(cell: CellCoordinate, attempt: int, status: str = "success") -> dict[str, Any]:
    budget = cell.budget.dump_config()
    return {
        "instance_id": cell.instance_uuid,
        "alg_id": cell.alg_uuid,
        "budget_path": budget.path,
        "budget_config": canonical_json_bytes(budget.config).decode("ascii"),
        "seed": cell.alg_seed,
        "attempt": attempt,
        "status": status,
        "runtime_s": 0.5,
        "peak_rss_mb": 2.0,
        "reason": None,
        "result": None if status != "success" else [float(cell.alg_seed), np.nan],
    }


def _store(root: Path, cells: list[CellCoordinate], *, copies: int) -> Path:
    """Write each partition once, then as further identical compacted copies."""
    by_algorithm: dict[str, list[CellCoordinate]] = {}
    for cell in cells:
        by_algorithm.setdefault(cell.alg_uuid, []).append(cell)
    for alg_uuid, members in by_algorithm.items():
        first = write_attempt_rows(
            root, alg_uuid, "compacted-0", [_row(cell, 0) for cell in members]
        )
        for copy in range(1, copies):
            shutil.copyfile(first, first.with_name(f"compacted-{copy}.parquet"))
        # A few cells were really retried: a later, distinct Attempt.
        write_attempt_rows(
            root,
            alg_uuid,
            "retried",
            [_row(cell, 1, status="timeout") for cell in members[:RETRIES]],
        )
    return root


def _median_seconds(root: Path, cells: list[CellCoordinate]) -> float:
    samples = []
    for _ in range(3):
        started = perf_counter()
        rows = view(root, SIMPLE_TASK, cells, columns=["status", "result"])
        samples.append(perf_counter() - started)
        assert rows.num_rows == len(cells)
    return statistics.median(samples)


def test_repeated_compacted_copies_cost_about_one_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = _cells()
    clean = _store(tmp_path / "clean", cells, copies=1)
    repeated = _store(tmp_path / "repeated", cells, copies=COPIES)
    encoded = 0

    def counting(value: Any) -> bytes:
        nonlocal encoded
        encoded += 1
        return canonical_json_bytes(value)

    monkeypatch.setattr(resolution, "canonical_json_bytes", counting)
    clean_s = _median_seconds(clean, cells)
    repeated_s = _median_seconds(repeated, cells)

    print(
        json.dumps(
            {
                "benchmark": "repeated_compacted_copies",
                "cells": len(cells),
                "copies": COPIES,
                "clean_median_s": clean_s,
                "repeated_median_s": repeated_s,
                "encodes": encoded,
            }
        )
    )
    # Agreeing repeats cost at most one encode per repeated cell in each of the
    # three resolutions of the repeated store; the clean store has none.
    assert encoded <= 3 * len(cells)
    assert repeated_s <= RATIO_CEILING * clean_s
