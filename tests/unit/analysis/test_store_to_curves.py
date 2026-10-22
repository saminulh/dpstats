"""The producer-to-consumer path: measured Results through to profile curves."""

from __future__ import annotations

from collections.abc import Sequence
from operator import attrgetter
from pathlib import Path

import pandas as pd
import pytest

from dpstats.workspace import TaskCollection
from dpstats.analysis import measure_results
from dpstats.analysis import (
    compute_baseline,
    compute_error_ratio,
    compute_profile_curves,
    summarize_outcomes,
)
from dpstats.budgets import NO_PRIVACY, PureDPBudget
from dpstats.metric import Metric
from tests.attempts import current_result, write_attempt_partition
from tests.registry_components import (
    SIMPLE_TASK,
    ExternalBudget,
    SimpleAlgorithm,
    simple_producer,
    SimpleOutput,
)
from tests.sweep_helpers import local_task_root, sweep_coordinates, write_sweep


def _absolute_error(output: SimpleOutput, *, reference: Sequence[float]) -> float:
    """Return the released estimate's distance from the public reference."""
    return float(abs(output.result[0] - reference[0]))


absolute_error = Metric(
    task=SIMPLE_TASK,
    name="absolute_error",
    calculation=_absolute_error,
    requires=("reference",),
)


@pytest.fixture
def store(collection: TaskCollection) -> TaskCollection:
    """A store whose reference algorithm releases 1.0 and whose private one 3.0.

    The private algorithm's second run under the private budget failed, so its
    cell holds one measurable run and one that provided nothing.
    """
    recipe = write_sweep(
        collection,
        [
            SimpleAlgorithm(label="Reference"),
            SimpleAlgorithm(scale=2.0, label="Private"),
        ],
        [simple_producer(n=2, d=1)],
        [NO_PRIVACY, PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )
    records = []
    for coordinate in sweep_coordinates(collection, recipe):
        private = str(coordinate.algorithm) == "Private"
        failed = (
            private
            and coordinate.alg_seed == 1
            and isinstance(coordinate.budget, PureDPBudget)
        )
        records.append(
            current_result(
                instance_uuid=coordinate.instance_uuid,
                alg_uuid=coordinate.alg_uuid,
                budget=coordinate.budget.dump_config(),
                alg_seed=coordinate.alg_seed,
                status="algorithm_error" if failed else "success",
                estimate=3.0 if private else 1.0,
            )
        )
    write_attempt_partition(local_task_root(collection), records)
    return collection


def test_measured_results_carry_the_columns_the_curves_are_drawn_from(
    store: TaskCollection,
) -> None:
    """Released 1.0 against a reference of 1.0 is a ratio of 1; 3.0 is 3; the
    failed run is unbounded and no finite threshold covers it."""
    measurements = measure_results(
        store.load_results(),
        absolute_error,
    )

    baseline = compute_baseline(
        measurements,
        baseline=SimpleAlgorithm(),
        per_budget=False,
        metric=absolute_error.name,
    )
    assert baseline["baseline"].tolist() == [1.0]

    normalized = compute_error_ratio(measurements, baseline, metric=absolute_error.name)
    private_budget = normalized[
        [isinstance(budget, PureDPBudget) for budget in normalized["budget"]]
    ]
    curves = compute_profile_curves(
        private_budget, [1.0, 3.0], metric=absolute_error.name
    )
    pooled = compute_profile_curves(normalized, [3.0], metric=absolute_error.name)

    drawn = {
        (row.algorithm_label, row.tau): row.fraction for row in curves.itertuples()
    }
    assert drawn == {
        ("Reference", 1.0): 1.0,
        ("Private", 1.0): 0.0,
        ("Reference", 3.0): 1.0,
        ("Private", 3.0): 0.5,
    }
    # Pooled over both budgets, Private's failed run is one of four.
    assert dict(zip(pooled["algorithm_label"], pooled["fraction"])) == {
        "Reference": 1.0,
        "Private": 0.75,
    }


def test_loaded_results_carry_the_outcomes_behind_those_curves(
    store: TaskCollection,
) -> None:

    shares = summarize_outcomes(store.load_results(), by=["algorithm_label"])

    private = shares[shares["algorithm_label"] == "Private"]
    assert private["denominator"].unique().tolist() == [4]
    taken = dict(zip(private["status"], private["share"]))
    assert taken["success"] == pytest.approx(0.75)
    assert taken["algorithm_error"] == pytest.approx(0.25)
    assert taken["timeout"] == pytest.approx(0.0)
    reference = shares[shares["algorithm_label"] == "Reference"]
    assert dict(zip(reference["status"], reference["share"]))["success"] == 1.0


def test_a_budget_predicate_selects_rows_and_always_keeps_unconstrained_ones(
    store: TaskCollection,
) -> None:
    """The predicate is never asked about the no-privacy budget, which lacks eps."""
    sweep = store.list_sweeps()[0]
    for selected in (store.load_results(), store.load_results(sweep)):
        assert len(selected) == 8
    for loaded in (
        store.load_results(budgets=lambda budget: budget.eps == 2.0),
        store.load_results(sweep, budgets=lambda budget: budget.eps == 2.0),
    ):
        assert {type(budget).__name__ for budget in loaded["budget"]} == {"NoPrivacy"}
        assert len(loaded) == 4
    narrowed = store.get_sweep_status(sweep, budgets=lambda budget: budget.eps == 2.0)
    assert narrowed["cells"]["expected"] == 4
    assert store.get_sweep_status(sweep)["cells"]["expected"] == 8
    kept = store.load_results(sweep, budgets=lambda budget: budget.eps == 1.0)
    assert len(kept) == 8
    assert sorted(set(kept["algorithm_label"])) == ["Private", "Reference"]
    assert set(kept["instance_label"]) == {
        str(instance) for instance in kept["instance"]
    }


def test_a_custom_budget_family_measures_groups_and_joins(
    collection: TaskCollection,
) -> None:
    """A budget family the library never declared travels the whole path.

    Nothing here teaches the library about credits: the budget is the column,
    so it groups and joins by being itself.
    """
    offered = ExternalBudget(credits=4.0)
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm(label="Reference")],
        [simple_producer(n=2, d=1)],
        [offered],
    )
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config(),
            alg_seed=coordinate.alg_seed,
            status="success",
            estimate=2.0,
        )
        for coordinate in sweep_coordinates(collection, recipe)
    ]
    write_attempt_partition(local_task_root(collection), records)

    measurements = measure_results(
        collection.load_results(),
        absolute_error,
    )

    assert measurements["budget"].tolist() == [offered]
    assert [type(b).__name__ for b in measurements["budget"]] == ["ExternalBudget"]
    assert measurements["budget"].iloc[0].credits == 4.0

    grouped = measurements.groupby(["algorithm", "budget"], sort=False)[
        absolute_error.name
    ].median()
    # The released 2.0 stands 2.0 away from the public reference of 0.0.
    assert grouped.tolist() == [2.0]

    baseline = pd.DataFrame(
        [
            {
                "task": SIMPLE_TASK.id,
                "instance": measurements["instance"].iloc[0],
                "baseline": 4.0,
            }
        ]
    )
    normalized = compute_error_ratio(measurements, baseline, metric=absolute_error.name)

    assert normalized[absolute_error.name].tolist() == [0.5]
