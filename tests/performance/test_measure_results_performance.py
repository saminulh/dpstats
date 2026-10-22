"""Opt-in geometric timing diagnostics for Results-frame measurement."""

from __future__ import annotations

import json
import statistics
from time import perf_counter

import pandas as pd
import pytest

from dpstats.budgets import PureDPBudget
from dpstats.workspace import TaskCollection
from dpstats.metric import Metric
from dpstats.analysis import measure_results
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleOutput,
    simple_producer,
)
from tests.sweep_helpers import write_sweep

pytestmark = pytest.mark.performance


def _error(output: SimpleOutput, *, reference: list[float]) -> float:
    return abs(float(output.result[0] - reference[0]))


metric = Metric(
    task=SIMPLE_TASK,
    name="measure_results_performance_error",
    calculation=_error,
    requires=("reference",),
)


def _results(collection: TaskCollection) -> pd.DataFrame:
    """Run a tiny persisted sweep and expand its typed, array-bearing outputs."""
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=2, d=1)],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )
    return collection.run_sweep(recipe)


def _expanded_results(results: pd.DataFrame, row_count: int) -> pd.DataFrame:
    """Repeat supplied Results to exercise one requested measurement size."""
    frame = pd.concat([results] * (row_count // len(results)), ignore_index=True)
    return frame


def test_measure_results_reports_geometric_size_timings(
    collection: TaskCollection,
) -> None:
    """Report three timing samples at each size without imposing a speed target."""
    results = _results(collection)
    timings: dict[int, list[float]] = {}
    for row_count in (12_000, 48_000):
        frame = _expanded_results(results, row_count)
        samples: list[float] = []
        for _ in range(3):
            started = perf_counter()
            measured = measure_results(frame, metric)
            samples.append(perf_counter() - started)
            assert len(measured) == row_count
        timings[row_count] = samples
    small = statistics.median(timings[12_000])
    large = statistics.median(timings[48_000])
    print(
        json.dumps(
            {
                "benchmark": "measure_results_geometric_rows",
                "small_rows": 12_000,
                "large_rows": 48_000,
                "small_median_s": small,
                "large_median_s": large,
                "observed_ratio": large / small,
            },
            sort_keys=True,
        )
    )
