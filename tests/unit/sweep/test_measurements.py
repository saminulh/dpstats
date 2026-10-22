"""Public measurement behavior over a retained Results frame."""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dpstats.workspace.collection as _collection_module
from dpstats.budgets import ApproxDPBudget, DPBudget, PureDPBudget
from dpstats.workspace import TaskCollection
from dpstats.instance import Instance, load_instance_input
from dpstats.metric import Metric, MetricArgument
from dpstats.analysis import measure_results
from dpstats.results import ResultStatus
from dpstats.tasks import MeanEstimation
from dpstats.task import TaskInput
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
    simple_producer,
)
from tests.sweep_helpers import local_task_root, sweep_coordinates, write_sweep
from tests.attempts import current_result, write_attempt_partition


def _error(output: SimpleOutput, *, reference: Sequence[float]) -> float:
    return float(abs(output.result[0] - reference[0]))


stored_error = Metric(
    task=SIMPLE_TASK,
    name="stored_error",
    calculation=_error,
    requires=("reference",),
)


def _input_error(output: SimpleOutput, *, input: SimpleInput) -> float:
    return float(abs(output.result[0] - input.X[0, 0]))


input_error = Metric(
    task=SIMPLE_TASK,
    name="input_error",
    calculation=_input_error,
    requires=("input",),
)


def _input_total(output: SimpleOutput, *, input: SimpleInput) -> float:
    return float(output.result[0] + input.X.sum())


input_total = Metric(
    task=SIMPLE_TASK,
    name="input_total",
    calculation=_input_total,
    requires=("input",),
)


def _centered(output: SimpleOutput, *, outputs: Sequence[SimpleOutput]) -> float:
    group_mean = sum(item.result[0] for item in outputs) / len(outputs)
    return float(output.result[0] - group_mean)


centered = Metric(
    task=SIMPLE_TASK,
    name="centered",
    calculation=_centered,
    requires=("outputs",),
)


def _group_first(output: SimpleOutput, *, outputs: Sequence[SimpleOutput]) -> float:
    del output
    return float(outputs[0].result[0])


group_first = Metric(
    task=SIMPLE_TASK,
    name="group_first",
    calculation=_group_first,
    requires=("outputs",),
)


def _install_sweep(collection: TaskCollection) -> None:
    """Install one algorithm and Instance under a two-seed Sweep."""
    write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=2, d=1)],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )


@pytest.mark.slow
def test_sparse_results_acquisition_scales_with_observed_outputs(
    collection: TaskCollection,
) -> None:
    """A large declared grid must remain usable under the normal test timeout."""
    budgets = [PureDPBudget(eps=float(index + 1)) for index in range(200)]
    algorithms = [SimpleAlgorithm(scale=float(index + 1)) for index in range(10)]
    recipe = write_sweep(
        collection,
        algorithms,
        [simple_producer(n=2, d=256)],
        budgets,
        instance_seeds=list(range(40)),
        seeds=list(range(10)),
    )
    retained = collection.run_sweep(
        recipe,
        algorithms=[algorithms[0].uuid],
        instances=[next(iter(collection.load_instances()))],
        budgets=budgets[:1],
    )
    # 800,000 declared coordinates, only ten observed vector-valued Results.
    # budget_group reflects this call's own (absent) budgets declaration, not
    # the finite selection retained was run under, so it is excluded here.
    # run_sweep and load_results return the same shape, including
    # source/release_version.
    acquired = collection.load_results()
    dropped = ["payload", "budget_group"]
    pd.testing.assert_frame_equal(
        acquired.drop(columns=dropped), retained.drop(columns=dropped)
    )
    for actual, expected in zip(acquired["payload"], retained["payload"], strict=True):
        np.testing.assert_array_equal(actual.result, expected.result)
    assert len(acquired) == 10
    expected = measure_results(retained, [stored_error, centered])
    for _ in range(2):
        measured = measure_results(acquired, [stored_error, centered])
        pd.testing.assert_frame_equal(
            measured.drop(columns=dropped), expected.drop(columns=dropped)
        )


@pytest.mark.slow
def test_dense_results_retain_large_outputs_across_measurements(
    collection: TaskCollection,
) -> None:
    """Exercise persisted vector payloads and repeated consumption in the normal suite."""
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=16, d=256)],
        [PureDPBudget(eps=1.0), PureDPBudget(eps=2.0)],
        seeds=list(range(512)),
        instance_seeds=list(range(4)),
    )
    coordinates = sweep_coordinates(collection, recipe)
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config().model_dump(),
            alg_seed=coordinate.alg_seed,
            output={"result": [4.0] * 256},
        )
        for coordinate in coordinates
    ]
    write_attempt_partition(local_task_root(collection), records)
    results = collection.load_results()
    assert len(results) == 4096
    assert results["alg_seed"].tolist() == list(range(512)) * 8
    for _ in range(2):
        measured = measure_results(results, [stored_error, centered])
        assert measured["alg_seed"].tolist() == list(range(512)) * 8
        assert measured[stored_error.name].tolist() == [4.0] * 4096
        assert measured[centered.name].tolist() == [0.0] * 4096


def test_observed_results_narrow_by_full_budgets_in_the_selected_seed_order(
    collection: TaskCollection,
) -> None:
    """Narrow by full budgets and seed order; which Attempt stands is resolution's."""
    budgets: list[DPBudget] = [
        PureDPBudget(eps=1.0),
        ApproxDPBudget(eps=1.0, delta=1e-6),
        ApproxDPBudget(eps=1.0, delta=1e-5),
    ]
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=2, d=1)],
        budgets,
        seeds=[3, 1, 2],
    )
    coordinates = sweep_coordinates(collection, recipe)
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config().model_dump(),
            alg_seed=coordinate.alg_seed,
            estimate=float(budgets.index(coordinate.budget) * 10 + coordinate.alg_seed),
        )
        for coordinate in coordinates
    ]
    write_attempt_partition(local_task_root(collection), records)

    assert len(collection.load_results(sweep=recipe)) == 9
    narrowed = collection.load_results(budgets=[budgets[2]], seeds=[2, 3, 2])
    assert narrowed["alg_seed"].tolist() == [2, 3]
    assert narrowed["budget"].tolist() == [budgets[2]] * 2
    assert [payload.result[0] for payload in narrowed["payload"]] == [22.0, 23.0]


def test_results_frame_is_measurable_after_the_collection_changes(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    retained = collection.run_sweep(collection.list_sweeps()[0])
    assert "coordinate_uuid" not in retained.columns

    collection.run_sweep(collection.list_sweeps()[0], mode="all")

    measured = measure_results(retained, stored_error)

    assert "coordinate_uuid" not in measured.columns
    assert measured["alg_seed"].tolist() == [0, 1]
    assert measured["status"].tolist() == [ResultStatus.SUCCESS] * 2


def test_measurement_uses_only_supplied_seed_cohort_for_group_outputs(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    first_seed = results.loc[results["alg_seed"] == 0]

    measured = measure_results(first_seed, centered)

    assert measured[centered.name].tolist() == [0.0]


def test_measurement_keeps_caller_order_for_failed_and_grouped_results(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    reordered = results.iloc[[1, 0]].copy()
    reordered["payload"] = pd.Series(
        [None, SimpleOutput(result=np.array([1.0]))],
        index=reordered.index,
        dtype=object,
    )
    reordered.at[reordered.index[0], "status"] = ResultStatus.TIMEOUT

    measured = measure_results(reordered, group_first)

    assert measured["alg_seed"].tolist() == [1, 0]
    assert math.isinf(measured.iloc[0][group_first.name])
    assert measured.iloc[1][group_first.name] == 1.0


def test_group_outputs_keep_supplied_success_order(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    reordered = results.iloc[[1, 0]].copy()
    reordered["payload"] = pd.Series(
        [SimpleOutput(result=np.array([2.0])), SimpleOutput(result=np.array([1.0]))],
        index=reordered.index,
        dtype=object,
    )

    measured = measure_results(reordered, group_first)

    assert measured["alg_seed"].tolist() == [1, 0]
    assert measured[group_first.name].tolist() == [2.0, 2.0]


@pytest.mark.slow
def test_measurement_handles_a_large_complete_results_frame_in_linear_time(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    row_count = 48_000
    results["payload"] = pd.Series(
        [SimpleOutput(result=np.array([3.0])) for _ in results.index], dtype=object
    )
    constant = Metric(
        task=SIMPLE_TASK,
        name="three",
        calculation=lambda output: float(output.result[0]),
    )
    repeated = pd.concat([results] * (row_count // len(results)), ignore_index=True)

    measured = measure_results(repeated, constant)

    assert len(measured) == row_count
    assert measured["alg_seed"].tolist() == [0, 1] * (row_count // 2)
    assert measured[constant.name].tolist() == [3.0] * row_count


def test_a_collection_reads_each_input_whatever_metrics_are_selected(
    collection: TaskCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a collection no Input is opened; with one every Instance's is."""
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    loaded: list[str] = []

    def record_load(value: Instance, directory: Path) -> TaskInput:
        loaded.append(value.uuid)
        return load_instance_input(value, directory)

    monkeypatch.setattr(_collection_module, "load_instance_input", record_load)
    measured = measure_results(results, stored_error)
    assert loaded == []
    with_collection = measure_results(results, stored_error, collection=collection)

    assert loaded == results["instance_uuid"].drop_duplicates().tolist()
    assert len(measured) == len(results)
    pd.testing.assert_frame_equal(with_collection, measured)


def test_declared_arguments_are_produced_beside_the_tasks_own(
    collection: TaskCollection,
) -> None:
    """A value no selected Metric names is still resolved from the Input."""
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    rows = MetricArgument(
        name="input_rows",
        description="How many rows the Input holds.",
        producer=lambda metadata, input: {"input_rows": input.X.shape[0]},
    )
    with pytest.raises(ValueError, match="collection="):
        measure_results(results, stored_error, arguments=[rows])
    measured = measure_results(
        results, stored_error, collection=collection, arguments=[rows]
    )
    # The installed Sweep's one Instance has two rows.
    assert [held["input_rows"] for held in measured["metric_arguments"]] == [2] * len(
        results
    )


def test_an_input_metric_needs_the_input_root(collection: TaskCollection) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])

    with pytest.raises(ValueError, match="collection="):
        measure_results(results, input_error)


def test_every_terminal_failure_scores_infinite(collection: TaskCollection) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    rows = results.to_dict(orient="records")
    rows[0]["status"] = ResultStatus.ALGORITHM_ERROR
    rows[0]["payload"] = "RuntimeError: synthetic failure"
    rows[1]["status"] = ResultStatus.TIMEOUT
    rows[1]["payload"] = None
    failed = pd.DataFrame(rows)

    measured = measure_results(failed, stored_error)

    assert measured[stored_error.name].tolist()[:2] == [math.inf, math.inf]
    pd.testing.assert_frame_equal(measured[failed.columns], failed)


def test_each_terminal_failure_keeps_its_existing_reason(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    result = collection.run_sweep(collection.list_sweeps()[0]).iloc[[0]].copy()
    cases = [
        (ResultStatus.NO_RELEASE, "privacy refusal", "privacy refusal"),
        (ResultStatus.INVALID_REGIME, "invalid regime", "invalid regime"),
        (ResultStatus.TIMEOUT, None, None),
        (ResultStatus.MEMORY_LIMIT_EXCEEDED, None, None),
        (ResultStatus.ALGORITHM_ERROR, "ValueError", "ValueError"),
    ]

    for status, payload, reason in cases:
        frame = result.copy()
        frame.loc[:, "status"] = status
        frame["payload"] = pd.Series([payload], index=frame.index, dtype=object)
        measured = measure_results(frame, stored_error)
        assert measured[stored_error.name].tolist() == [math.inf]
        pd.testing.assert_frame_equal(measured[frame.columns], frame)


def test_empty_and_invalid_result_frames_have_explicit_contracts(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])

    empty = results.iloc[0:0]
    assert list(measure_results(empty, stored_error).columns) == [
        *empty.columns,
        "metric_arguments",
        stored_error.name,
    ]
    with pytest.raises(ValueError, match="missing required column.*payload"):
        measure_results(results.drop(columns=["payload"]), stored_error)
    wrong_task = results.copy()
    wrong_task.loc[0, "task"] = "another_task"
    with pytest.raises(ValueError, match="task column"):
        measure_results(wrong_task, stored_error)
    wrong_metric = Metric(
        task=MeanEstimation, name="wrong_metric", calculation=lambda output: 0.0
    )
    with pytest.raises(ValueError, match="another Task"):
        measure_results(results, wrong_metric)
    with pytest.raises(ValueError, match="selected more than once"):
        measure_results(results, [stored_error, stored_error])


def test_named_metrics_preserve_original_frame_and_duplicate_index(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0]).iloc[[1, 0]].copy()
    results.index = pd.Index([7, 7], name="user_index")
    results["annotation"] = ["second", "first"]
    original = results.copy(deep=True)

    measured = measure_results(results, [stored_error, centered])

    assert list(measured.columns) == [
        *results.columns,
        "metric_arguments",
        stored_error.name,
        centered.name,
    ]
    pd.testing.assert_frame_equal(measured[results.columns], original)
    pd.testing.assert_frame_equal(results, original)
    assert measured[stored_error.name].tolist() == [
        stored_error.calculate(
            payload, reference=instance.metric_arguments["reference"]
        )
        for payload, instance in zip(results["payload"], results["instance"])
    ]
    assert "metric" not in measured and "value" not in measured
    copied = measure_results(results, [])
    assert copied is not results
    pd.testing.assert_frame_equal(copied, original)
    empty = measure_results(results.iloc[:0], [stored_error, centered])
    assert list(empty.columns) == list(measured.columns)
    assert empty[stored_error.name].dtype == np.dtype(float)
    assert empty[centered.name].dtype == np.dtype(float)


def test_metric_name_rejection_precedes_any_calculation(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0])
    calculated = []
    first = Metric(
        task=SIMPLE_TASK,
        name="first",
        calculation=lambda output: calculated.append(output) or 0.0,
    )
    collision = Metric(task=SIMPLE_TASK, name="payload", calculation=lambda output: 0.0)
    with pytest.raises(ValueError, match="collides"):
        measure_results(results, [first, collision])
    with pytest.raises(ValueError, match="selected more than once"):
        measure_results(results, [first, first])
    assert not calculated


def test_metrics_can_be_added_without_changing_earlier_measurements(
    collection: TaskCollection,
) -> None:
    _install_sweep(collection)
    results = collection.run_sweep(collection.list_sweeps()[0]).iloc[[1, 0]].copy()
    results.index = pd.Index([7, 7], name="user_index")
    results["annotation"] = ["second", "first"]
    first = measure_results(results, stored_error)
    original = first.copy(deep=True)

    extended = measure_results(first, centered)
    together = measure_results(results, [stored_error, centered])

    pd.testing.assert_frame_equal(extended, together)
    pd.testing.assert_frame_equal(first, original)
    pd.testing.assert_frame_equal(extended[first.columns], original)
    assert extended["metric_arguments"].iloc[0] is not first["metric_arguments"].iloc[0]
    empty = measure_results(first.iloc[:0], centered)
    pd.testing.assert_frame_equal(empty, together.iloc[:0])
    with pytest.raises(ValueError, match="collides with an existing Results column"):
        measure_results(first, stored_error)


def _campaign(
    collection: TaskCollection,
    scales: tuple[float, ...],
    *,
    offsets: tuple[float, ...] = (0.0,),
    seeds: tuple[int, ...] = (0,),
) -> list[Instance]:
    """Install one Instance per offset, one Algorithm per scale, and run them.

    Each Instance's Input is the column ``[1 + offset, 3 + offset]``.
    """
    instances = [
        collection.add_instance(
            SimpleMetadata(n_samples=2, dim=1),
            SimpleInput(X=np.array([[1.0 + offset], [3.0 + offset]])),
            {},
        )
        for offset in offsets
    ]
    for scale in scales:
        collection.add_algorithm(SimpleAlgorithm(scale=scale))
    collection.run_sweep(
        collection.save_sweep(
            "campaign", {"budgets": [PureDPBudget(eps=1.0)], "seeds": list(seeds)}
        )
    )
    return instances


def test_input_metrics_read_each_instance_once_and_keep_caller_order(
    collection: TaskCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics selected together share one Input read per Instance.

    The rows alternate between Instances, so a per-group read would load each
    Input more than once; the frame still comes back in the caller's order.
    """
    instances = _campaign(collection, (1.0, 2.0), offsets=(0.0, 10.0), seeds=(0, 1))
    results = collection.load_results()
    by_instance = [
        results.index[results["instance_uuid"] == instance.uuid]
        for instance in instances
    ]
    interleaved = results.loc[[row for pair in zip(*by_instance) for row in pair]]
    loaded: list[str] = []

    def record_load(instance: Instance, directory: Path) -> TaskInput:
        loaded.append(instance.uuid)
        return load_instance_input(instance, directory)

    monkeypatch.setattr(_collection_module, "load_instance_input", record_load)
    measured = measure_results(
        interleaved,
        [input_error, input_total],
        collection=collection,
    )

    assert sorted(loaded) == sorted(instance.uuid for instance in instances)
    assert len(set(interleaved["instance_uuid"].iloc[:2])) == 2
    assert measured["instance_uuid"].tolist() == interleaved["instance_uuid"].tolist()
    assert measured["alg_seed"].tolist() == interleaved["alg_seed"].tolist()
    assert list(measured.columns[-2:]) == [input_error.name, input_total.name]


def test_only_the_measured_instance_holds_a_cached_input(
    collection: TaskCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One Instance Input is released before the next one is read."""
    _campaign(collection, (1.0, 2.0), offsets=(0.0, 10.0, 20.0))
    cached: list[weakref.ReferenceType[TaskInput]] = []

    def record_load(value: Instance, directory: Path) -> TaskInput:
        assert not any(held() is not None for held in cached)
        loaded_input = load_instance_input(value, directory)
        cached.append(weakref.ref(loaded_input))
        return loaded_input

    monkeypatch.setattr(_collection_module, "load_instance_input", record_load)
    live_counts: list[int] = []

    def count_live_inputs(output: SimpleOutput, *, input: SimpleInput) -> float:
        live_counts.append(sum(1 for held in cached if held() is not None))
        return float(input.X.mean())

    live_input_count = Metric(
        task=SIMPLE_TASK,
        name="live_input_count",
        calculation=count_live_inputs,
        requires=("input",),
    )
    measurements = measure_results(
        collection.load_results(),
        live_input_count,
        collection=collection,
    )

    assert len(measurements) == 6
    assert set(live_counts) == {1}
