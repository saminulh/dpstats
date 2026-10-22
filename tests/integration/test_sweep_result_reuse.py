"""Explicit successful Results answer selected cells through the public sweep."""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from dpstats import init_workspace, open_collection
from dpstats.algorithm import CompositeAlgorithm
from dpstats.budgets import DPBudget, NO_PRIVACY
from dpstats.domains import BoxDomain
from dpstats.results import ResultStatus
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from dpstats.tasks.mean_estimation.transformations import DataBounds
from tests.sweep_helpers import observing_schedule, write_mean_benchmark_sweep


class ObservedMean(SampleMean):
    """Expose execution through an ordinary user-supplied algorithm."""

    calls: ClassVar[list[int]] = []

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        self.calls.append(seed)
        return MeanEstimationOutput(mean_est=np.mean(input.X, axis=0) + seed)


def test_supplied_success_skips_only_its_exact_cell_and_is_durable(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    collection.add_algorithm(ObservedMean())
    collection.add_instance(
        MeanEstimationMetadata(
            n_samples=2, domain=BoxDomain(bounds=np.array([[0.0, 4.0]]))
        ),
        MeanEstimationInput(X=np.array([[1.0], [3.0]])),
        {},
    )
    source = collection.run_sweep(budgets=[NO_PRIVACY], seeds=[0])
    ObservedMean.calls.clear()

    results = collection.run_sweep(
        budgets=[NO_PRIVACY], seeds=[0, 1, 2], mode="all", reuse_results=source
    ).sort_values("alg_seed")

    assert ObservedMean.calls == [1, 2]
    assert list(results.status) == [ResultStatus.SUCCESS] * 3
    assert [output.mean_est.tolist() for output in results.payload] == [
        [2.0],
        [3.0],
        [4.0],
    ]
    assert pd.isna(results.iloc[0].runtime_s)
    assert pd.isna(results.iloc[0].peak_rss_mb)
    assert results.iloc[1:].runtime_s.notna().all()
    reloaded = collection.load_results().sort_values("alg_seed")
    assert [output.mean_est.tolist() for output in reloaded.payload] == [
        [2.0],
        [3.0],
        [4.0],
    ]
    assert pd.isna(reloaded.iloc[0].runtime_s)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "linear_regression"),
        ("instance_uuid", "unknown-instance"),
        ("alg_uuid", "unknown-algorithm"),
        ("alg_seed", 2),
        ("alg_seed", 0),
        ("status", ResultStatus.NO_RELEASE),
        ("payload", "invalid output"),
        ("payload", MeanEstimationOutput(mean_est=np.array([np.nan]))),
    ],
)
def test_invalid_supplied_results_are_refused_before_any_cell_is_written(
    tmp_path: Path, field: str, value: object
):
    collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(collection, ObservedMean())
    source = collection.run_sweep(sweep)
    valid = source.copy()
    valid.at[0, "payload"] = MeanEstimationOutput(mean_est=np.array([17.0]))
    invalid = source.copy()
    invalid.at[0, "alg_seed"] = 1
    invalid.at[0, field] = value
    supplied = pd.concat([valid, invalid], ignore_index=True)
    ObservedMean.calls.clear()

    with pytest.raises((TypeError, ValueError), match="reuse_results"):
        collection.run_sweep(sweep, seeds=[0, 1], mode="all", reuse_results=supplied)

    assert ObservedMean.calls == []
    current = collection.load_results()
    assert len(current) == 1
    assert current.iloc[0].payload.mean_est.tolist() == [2.0]
    assert current.iloc[0].runtime_s == source.iloc[0].runtime_s


def test_reusing_a_composite_does_not_answer_a_different_core(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    target = CompositeAlgorithm(
        transformations=(DataBounds(),), algorithm=ObservedMean()
    )
    sibling = CompositeAlgorithm(
        transformations=(DataBounds(),), algorithm=SampleMean()
    )
    sweep = write_mean_benchmark_sweep(collection, target)
    collection.add_algorithm(sibling)
    source = collection.run_sweep(sweep, algorithms=[target])
    source.at[0, "payload"] = MeanEstimationOutput(mean_est=np.array([7.0]))
    ObservedMean.calls.clear()
    offered = []

    results = collection.run_sweep(
        sweep,
        mode="all",
        reuse_results=source,
        schedule_cells=observing_schedule(offered.append),
    )

    assert ObservedMean.calls == []
    assert [type(job.coordinate.algorithm) for job in offered] == [
        CompositeAlgorithm,
        SampleMean,
    ]
    outputs = dict(zip(results.alg_uuid, results.payload))
    assert outputs[target.uuid].mean_est.tolist() == [7.0]
    assert outputs[sibling.uuid].mean_est.tolist() == [2.0]
    offered.clear()
    collection.run_sweep(
        sweep,
        algorithms=[target],
        mode="all",
        reuse_results=source,
        schedule_cells=observing_schedule(offered.append),
    )
    assert offered == []


@pytest.mark.parametrize("mode", ["missing", "retry_resources"])
def test_supplied_success_supersedes_its_cell_without_rerunning_answered_siblings(
    tmp_path: Path, mode: str
):
    collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    sweep = write_mean_benchmark_sweep(collection, ObservedMean())
    source = collection.run_sweep(sweep, seeds=[0, 1]).sort_values("alg_seed")
    supplied = source.iloc[:1].copy()
    supplied.at[supplied.index[0], "payload"] = MeanEstimationOutput(
        mean_est=np.array([7.0])
    )
    ObservedMean.calls.clear()

    results = collection.run_sweep(
        sweep, seeds=[0, 1], mode=mode, reuse_results=supplied
    ).sort_values("alg_seed")

    assert ObservedMean.calls == []
    assert [output.mean_est.tolist() for output in results.payload] == [[7.0], [3.0]]
    assert pd.isna(results.iloc[0].runtime_s)
    assert results.iloc[1].runtime_s == source.iloc[1].runtime_s
