"""Opt-in timing diagnostic for measuring one release task into site panels."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pytest

from tests.unit.docs.conftest import (  # noqa: F401 - the release is a fixture
    measurable_release_root,
    measured,
)

pytestmark = pytest.mark.performance


def test_measuring_one_task_reports_its_wall_clock(
    measurable_release_root: Path,
) -> None:
    """Report how long one task's panels take to measure, without a speed target.

    A build that measures one panel state at a time is orders of magnitude
    slower than one that decodes each task's Results once; the reported time
    makes that visible without binding correctness to machine speed.
    """
    started = perf_counter()
    data = measured(measurable_release_root)
    elapsed = perf_counter() - started
    assert data.panels
    print(
        json.dumps(
            {
                "benchmark": "docs_task_assembly",
                "panels": len(data.panels),
                "elapsed_s": elapsed,
            },
            sort_keys=True,
        )
    )
