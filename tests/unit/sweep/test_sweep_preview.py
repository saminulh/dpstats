"""Observable contract tests for operational Sweep preview."""

from __future__ import annotations

from typing import Any

import json
import os
from pathlib import Path

import pytest
import pandas as pd

from dpstats.algorithm import register_algorithm
from dpstats.sweep import preview_sweep
from dpstats.sweep._claims import claim_attempt
from dpstats.budgets import PureDPBudget
from tests.attempts import (
    current_result,
    write_attempt_partition,
    write_result_partition,
)
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    sized_producer,
)
from tests.sweep_helpers import local_task_root, sweep_coordinates, write_sweep
from dpstats.workspace import TaskCollection
from dpstats.workspace.layout import (
    claims_directory,
    workspace_local_root,
)
from dpstats.workspace import init_workspace
from dpstats.workspace import open_collection


@pytest.mark.parametrize("seed", [2, 2**63, 2**64 - 1])
def test_projected_preview_reports_structured_status_and_preserves_selection(
    tmp_path: Path,
    collection: TaskCollection,
    seed: int,
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=4)],
        [PureDPBudget(eps=1.0)],
        seeds=[seed, 0, seed],
    )
    first, _ = sweep_coordinates(collection, sweep)
    record = current_result(
        instance_uuid=first.instance_uuid,
        alg_uuid=first.alg_uuid,
        budget=first.budget.dump_config().model_dump(),
        alg_seed=seed,
        status="timeout",
    )
    write_attempt_partition(local_task_root(collection), [record])
    columns = ["alg_seed", "status", "action"]
    projected = preview_sweep(
        sweep, workspace_local_root(tmp_path), SIMPLE_TASK, columns=columns
    )
    assert projected.to_dict("records") == [
        {"alg_seed": seed, "status": "timeout", "action": "retryable"},
        {"alg_seed": 0, "status": None, "action": "missing"},
    ]
    pd.testing.assert_frame_equal(
        projected[["alg_seed", "action"]],
        preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)[
            ["alg_seed", "action"]
        ],
    )


def test_preview_uses_every_installed_algorithm_and_reports_current_gaps(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm(scale=1.0), SimpleAlgorithm(scale=2.0)],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    first, second = sweep_coordinates(collection, sweep)
    write_attempt_partition(
        local_task_root(collection),
        [
            current_result(
                instance_uuid=first.instance_uuid,
                alg_uuid=first.alg_uuid,
                budget=first.budget.dump_config(),
            )
        ],
    )
    store = local_task_root(collection)
    files_before_preview = {
        path.relative_to(store) for path in store.rglob("*") if path.is_file()
    }

    preview = preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    assert set(preview["alg_uuid"]) == {first.alg_uuid, second.alg_uuid}
    assert sorted(preview["action"]) == ["missing", "reusable"]
    assert {
        path.relative_to(store) for path in store.rglob("*") if path.is_file()
    } == files_before_preview


def test_preview_on_a_release_tree_counts_released_results_not_missing(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    """A release carries ``results/``, not ``attempts/``.

    Every cell it answers must be reported as answered, exactly as a store's
    own Attempts are.
    """
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm(scale=1.0), SimpleAlgorithm(scale=2.0)],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    coordinates = sweep_coordinates(collection, sweep)
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config().model_dump(),
        )
        for coordinate in coordinates
    ]
    write_result_partition(local_task_root(collection), records)

    preview = preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    assert sorted(preview["action"]) == ["reusable", "reusable"]


def test_preview_reports_why_the_current_result_is_reusable(
    tmp_path: Path, collection: TaskCollection
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, sweep)
    current = current_result(
        instance_uuid=coordinate.instance_uuid,
        alg_uuid=coordinate.alg_uuid,
        budget=coordinate.budget.dump_config().model_dump(),
        status="timeout",
    )
    newer = {
        **current,
        "attempt_number": 1,
        "status": "success",
        "payload": {"result": [2.0]},
    }
    task_collection = local_task_root(collection)
    write_attempt_partition(task_collection, [current], name="first")
    write_attempt_partition(task_collection, [newer], name="second")

    preview = preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    assert preview.loc[0, ["action", "reason"]].to_dict() == {
        "action": "reusable",
        "reason": "the current Result is success",
    }


def test_preview_refuses_same_number_history_disagreement(
    tmp_path: Path, collection: TaskCollection
) -> None:
    """Preview reads history through the one resolution the store owns."""
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, sweep)
    first = current_result(
        instance_uuid=coordinate.instance_uuid,
        alg_uuid=coordinate.alg_uuid,
        budget=coordinate.budget.dump_config().model_dump(),
    )
    task_collection = local_task_root(collection)
    write_attempt_partition(task_collection, [first], name="first")
    write_attempt_partition(
        task_collection, [{**first, "runtime_s": 1.0}], name="other"
    )

    with pytest.raises(ValueError, match="disagrees about attempt"):
        preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)


def test_preview_ignores_a_history_disagreement_outside_its_selection(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm(scale=1.0), SimpleAlgorithm(scale=2.0)],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )
    coordinates = sweep_coordinates(collection, sweep)
    selected = next(
        coordinate for coordinate in coordinates if coordinate.algorithm.scale == 1.0
    )
    unselected = next(
        coordinate for coordinate in coordinates if coordinate.algorithm.scale == 2.0
    )
    register_algorithm(
        workspace_local_root(tmp_path),
        selected.algorithm,
        registry="selected",
    )
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config().model_dump(),
        )
        for coordinate in (selected, unselected)
    ]
    task_collection = local_task_root(collection)
    write_attempt_partition(task_collection, records, name="first")
    write_attempt_partition(
        task_collection, [{**records[1], "runtime_s": 1.0}], name="second"
    )

    preview = preview_sweep(
        sweep,
        workspace_local_root(tmp_path),
        SIMPLE_TASK,
        algorithm_registries=("selected",),
        instance_registries=("test",),
    )

    assert preview["alg_uuid"].tolist() == [selected.alg_uuid]


@pytest.mark.parametrize(
    ("declared", "reported"),
    [
        pytest.param({"family": "pure_dp", "eps": [1.0, 3.0]}, "family", id="group"),
        pytest.param(
            {
                "path": "dpstats.budgets.families:PureDPBudget",
                "config": {"eps": 1.0, "rho": 0.5},
            },
            "rho",
            id="unknown-field",
        ),
        pytest.param(
            {"path": "tests.registry_components:SimpleAlgorithm", "config": {}},
            "outside",
            id="foreign-family",
        ),
        pytest.param(
            {"path": "no_such_module:Policy", "config": {}},
            "no_such_module",
            id="unimportable",
        ),
    ],
)
def test_a_sweep_declaring_an_unreadable_budget_is_rejected(
    tmp_path: Path, declared: dict[str, Any], reported: str
) -> None:
    """A budget is one configured value of a budget or policy family.

    A document of parameter axes, an undeclared field, a class from another
    family, and a class this installation cannot import are each settled where
    the Sweep is read. The Sweep names the file it could not read; what was
    wrong with it travels on the chained cause.
    """
    sweep = tmp_path / "unreadable.json"
    sweep.write_text(json.dumps({"task": SIMPLE_TASK.id, "budgets": [declared]}))

    with pytest.raises(ValueError, match="is invalid") as rejection:
        preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    assert str(sweep) in str(rejection.value)
    assert reported in str(rejection.value.__cause__)


def test_a_sweep_without_budgets_is_refused_rather_than_resolved_empty(
    tmp_path: Path, collection: TaskCollection
) -> None:
    collection.add_algorithm(SimpleAlgorithm())
    collection.add_instance(*sized_producer(n_samples=4)(seed=0))
    sweep = collection.save_sweep("budgetless", {"seeds": [0]})

    with pytest.raises(ValueError, match="declare its budgets"):
        sweep_coordinates(collection, sweep)
    with pytest.raises(ValueError, match="declare its budgets"):
        sweep_coordinates(collection, sweep, budgets=[PureDPBudget(eps=1.0)])
    with pytest.raises(ValueError, match="declare its budgets"):
        preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)


def test_operator_coordinates_follow_registry_order_not_sweep_order(
    tmp_path: Path, collection: TaskCollection
) -> None:
    collection.add_algorithm(SimpleAlgorithm())
    for n in (4, 5, 6):
        collection.add_instance(*sized_producer(n_samples=n)(seed=0))
    installed = list(collection.load_instances())
    sweep = collection.save_sweep(
        "reversed",
        {
            "instances": installed[::-1],
            "budgets": [PureDPBudget(eps=1.0)],
            "seeds": [0],
        },
    )

    coordinates = sweep_coordinates(collection, sweep)

    assert [cell.instance.uuid for cell in coordinates] == installed


def test_a_private_sweep_declaring_no_budget_is_refused(
    tmp_path: Path, collection: TaskCollection
) -> None:
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=4)],
        [],
    )

    with pytest.raises(ValueError, match="at least one typed budget"):
        preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)


def test_one_sweep_resolves_against_each_task_collection_projection(
    tmp_path: Path,
) -> None:
    first_collection = open_collection(init_workspace(tmp_path / "first"), SIMPLE_TASK)
    sweep = write_sweep(
        first_collection,
        [SimpleAlgorithm(scale=1.0)],
        [sized_producer(n_samples=4)],
        [PureDPBudget(eps=1.0)],
    )
    authored = sweep.read_bytes()
    second_collection = open_collection(
        init_workspace(tmp_path / "second"), SIMPLE_TASK
    )
    write_sweep(
        second_collection,
        [SimpleAlgorithm(scale=2.0)],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
    )

    first = sweep_coordinates(first_collection, sweep)
    second = sweep_coordinates(second_collection, sweep)

    assert sweep.read_bytes() == authored
    assert {coordinate.uuid for coordinate in first} != {
        coordinate.uuid for coordinate in second
    }


def test_preview_reports_a_standing_claim_without_deciding_it(
    tmp_path: Path, collection: TaskCollection
) -> None:
    """A reader names work in limbo; it never turns a claim into a recorded Attempt.

    A claim may belong to a runner that is executing this very cell, so the
    only honest thing a reader can do is say that the cell is spoken for.
    """
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=8)],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )
    claimed, unclaimed = sweep_coordinates(collection, sweep)
    store = local_task_root(collection)
    claim_attempt(store, claimed, 0)
    claims_before = {
        path: path.read_bytes() for path in claims_directory(store).iterdir()
    }
    files_before = {
        path.relative_to(store) for path in store.rglob("*") if path.is_file()
    }

    preview = preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    reported = dict(zip(preview["alg_seed"], preview["claim"]))
    assert str(os.getpid()) in reported[claimed.alg_seed]
    assert reported[unclaimed.alg_seed] == ""
    assert {path: path.read_bytes() for path in claims_directory(store).iterdir()} == (
        claims_before
    )
    assert {
        path.relative_to(store) for path in store.rglob("*") if path.is_file()
    } == files_before


def test_preview_reports_the_declared_resource_envelope(
    tmp_path: Path, collection: TaskCollection
) -> None:
    """Every preview row carries what launching that cell externally requires.

    An orchestrator reads one row and has the whole job: the key that names the
    cell and the envelope the Sweep declares for it, with no Sweep parsing of
    its own.
    """
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm(scale=1.0), SimpleAlgorithm(scale=2.0)],
        [sized_producer(n_samples=4)],
        [PureDPBudget(eps=1.0)],
        time_limit=120.0,
        memory_limit_gb=2.0,
    )

    preview = preview_sweep(sweep, workspace_local_root(tmp_path), SIMPLE_TASK)

    assert list(preview["time_limit"]) == [120.0, 120.0]
    assert list(preview["memory_limit_gb"]) == [2.0, 2.0]
    assert sorted(preview["coordinate_uuid"]) == sorted(
        entry.uuid for entry in sweep_coordinates(collection, sweep)
    )


def test_preview_marks_only_answered_coordinates_for_recomputation(
    tmp_path: Path, collection: TaskCollection
) -> None:
    """A recompute preview distinguishes replacement work from missing work."""
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [sized_producer(n_samples=4)],
        [PureDPBudget(eps=1.0)],
        seeds=[0, 1],
    )
    answered, unanswered = sweep_coordinates(collection, sweep)
    write_attempt_partition(
        local_task_root(collection),
        [
            current_result(
                instance_uuid=answered.instance_uuid,
                alg_uuid=answered.alg_uuid,
                budget=answered.budget.dump_config().model_dump(),
                alg_seed=answered.alg_seed,
            )
        ],
    )

    ordinary = preview_sweep(
        sweep, workspace_local_root(tmp_path), SIMPLE_TASK
    ).set_index("coordinate_uuid")
    recompute = preview_sweep(
        sweep, workspace_local_root(tmp_path), SIMPLE_TASK, recompute=True
    ).set_index("coordinate_uuid")

    assert ordinary.loc[answered.uuid, "action"] == "reusable"
    assert ordinary.loc[unanswered.uuid, "action"] == "missing"
    assert recompute.loc[answered.uuid, "action"] == "recompute"
    assert recompute.loc[unanswered.uuid, "action"] == "missing"
