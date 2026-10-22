"""Observable contracts for optional Sweep settings and mixed object selection."""

from pathlib import Path
from typing import Any

import pytest

from dpstats.budgets import ApproxDPBudget, NO_PRIVACY
from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.instance import Instance, persist_instance
from dpstats.results import ResultStatus
from dpstats.transport.documents import STAGING_PREFIX, write_json_atomic
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    claims_directory,
    registry_directory_name,
    task_collection_directory,
    workspace_local_root,
)
from dpstats.workspace.registries import authoring_registry_path
from tests.attempts import current_result, write_attempt_partition
from dpstats.coordinates import CellCoordinate
from tests.attempts import abandon_claim, read_attempt_rows, unsealed_journals
from tests.sweep_helpers import local_task_root
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    NonprivateAlgorithm,
    simple_instance,
)


_UNIMPORTABLE = {"broken": {"path": "uninstalled_algorithm:Broken", "config": {}}}


@pytest.mark.parametrize(
    "algorithm_registry, other_registry, empty_axis, selection",
    [
        ("selected", None, "algorithms", {}),
        ("selected", None, "instances", {}),
        ("selected", None, "seeds", {}),
        ("selected", None, "budgets", {}),
        # An unselected group's unimportable entry cannot block it.
        (
            "selected",
            ("broken", _UNIMPORTABLE),
            None,
            {"algorithms": [], "algorithm_registries": ["selected"]},
        ),
        # A selected registry installing no Algorithm resolves no coordinate,
        # but the selected Instance still names the handle's one Task.
        (None, ("empty", {}), None, {"algorithm_registries": ["empty"]}),
        # Empty overrides on both axes still take the Task from the selected
        # registry rather than a scan of every installed group.
        (
            "selected",
            ("broken", _UNIMPORTABLE),
            None,
            {"algorithms": [], "instances": [], "algorithm_registries": ["selected"]},
        ),
    ],
    ids=[
        "empty-algorithms",
        "empty-instances",
        "empty-seeds",
        "empty-budgets",
        "past-an-unselected-broken-group",
        "empty-selected-registry",
        "both-axes-empty",
    ],
)
def test_run_sweep_reconciles_a_held_claim_when_it_selects_no_cell(
    collection: TaskCollection,
    algorithm_registry: str | None,
    other_registry: tuple[str, dict[str, Any]] | None,
    empty_axis: str | None,
    selection: dict[str, Any],
) -> None:
    """A run that executes nothing still clears a vanished runner's claim."""
    task_store = local_task_root(collection)
    algorithm = SimpleAlgorithm()
    if algorithm_registry is not None:
        collection.add_algorithm(algorithm, registry=algorithm_registry)
    instance = collection.add_instance(*simple_instance(0))
    if other_registry is not None:
        name, entries = other_registry
        write_json_atomic(
            authoring_registry_path(task_store, "algorithm", name),
            {"task": SIMPLE_TASK.id, "kind": "algorithm", "entries": entries},
        )
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    sweep: Path | dict[str, Any] = {"budgets": [budget], "seeds": [0]}
    if empty_axis is not None:
        sweep = collection.save_sweep("empty", {**sweep, empty_axis: []})
    abandon_claim(
        task_store,
        CellCoordinate(
            instance=instance, algorithm=algorithm, budget=budget, alg_seed=0
        ),
    )

    result = collection.run_sweep(sweep, **selection)

    assert result.empty
    assert collection.reconcile() == []
    assert read_attempt_rows(task_store) == []


def test_run_sweep_recovers_abandoned_file_free_work_with_no_installed_registry(
    tmp_path: Path,
) -> None:
    """An empty axis still settles a vanished runner's claim and journal.

    Neither the Algorithm nor the Instance is ever registered, and no
    registry is installed for this Task at all: ``execute_sweep_groups``
    must carry the collection's own bound Task rather than trying to
    discover one from installed declarations that do not exist.
    """
    import dpstats.sweep._execution as runner_module
    from dpstats.results._journal import AttemptWriter

    inputs = workspace_local_root(tmp_path) / INPUT_ARTIFACTS_DIRECTORY
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = persist_instance(*simple_instance(0), inputs)
    algorithm = SimpleAlgorithm()
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    task_store = local_task_root(collection)

    # A vanished runner's held claim.
    claimed_coordinate = CellCoordinate(
        instance=instance, algorithm=algorithm, budget=budget, alg_seed=0
    )
    abandon_claim(task_store, claimed_coordinate)

    # A vanished runner's durably journaled but never-sealed Attempt.
    journaled_coordinate = CellCoordinate(
        instance=instance, algorithm=algorithm, budget=budget, alg_seed=1
    )
    attempt = runner_module.run_coordinate(
        runner_module.CellContext(
            coordinate=journaled_coordinate,
            task_collection=task_store,
            input=inputs,
            attempt_number=0,
            time_limit_s=None,
            memory_limit_bytes=None,
        )
    )
    writer = AttemptWriter(task_store, SIMPLE_TASK)
    writer.add(attempt)  # left open: no writer.close(), simulating a crash.

    assert not collection.list_algorithm_registries()
    assert not collection.list_instance_registries()

    result = collection.run_sweep(algorithms=[], instances=[], budgets=[], seeds=[])

    assert result.empty
    assert not unsealed_journals(task_store)
    assert (task_store / "claims").exists()
    assert list((task_store / "claims").iterdir()) == []
    assert len(read_attempt_rows(task_store)) == 1


@pytest.mark.parametrize("malformed", [False, True])
def test_empty_sweep_does_not_ignore_claim_debris(
    tmp_path: Path, malformed: bool
) -> None:
    root = workspace_local_root(tmp_path)
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    collection.add_algorithm(SimpleAlgorithm())
    claims = claims_directory(task_collection_directory(root, SIMPLE_TASK.id))
    claims.mkdir(parents=True)
    debris = claims / ("broken.json" if malformed else f"{STAGING_PREFIX}abandoned")
    debris.write_text("{ incomplete")
    sweep = {"algorithms": [], "instances": [], "budgets": [], "seeds": []}

    if malformed:
        with pytest.raises(ValueError, match="broken.json"):
            collection.run_sweep(sweep)
        assert debris.exists()
    else:
        assert collection.run_sweep(sweep).empty
        assert not debris.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_empty_sweep_refuses_malformed_claim_store(
    tmp_path: Path, symlink: bool
) -> None:
    root = workspace_local_root(tmp_path)
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    collection.add_algorithm(SimpleAlgorithm())
    claims = claims_directory(task_collection_directory(root, SIMPLE_TASK.id))
    if symlink:
        target = tmp_path / "other-claims"
        target.mkdir()
        claims.symlink_to(target, target_is_directory=True)
    else:
        claims.write_text("not a directory")

    with pytest.raises(ValueError, match="non-symlink directory"):
        collection.run_sweep(
            {"algorithms": [], "instances": [], "budgets": [], "seeds": []}
        )
    assert claims.exists()


def test_custom_task_reconcile_needs_no_algorithm_registry(tmp_path: Path) -> None:
    """A bound custom Task settles its own claim without loading declarations."""
    workspace = init_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    instance = persist_instance(
        *simple_instance(0),
        workspace_local_root(workspace) / INPUT_ARTIFACTS_DIRECTORY,
    )
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=SimpleAlgorithm(),
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        alg_seed=0,
    )
    task_store = task_collection_directory(
        workspace_local_root(workspace), SIMPLE_TASK.id
    )
    abandon_claim(task_store, coordinate)

    assert collection.reconcile() == [coordinate.uuid]
    assert not (task_store / registry_directory_name("algorithm")).exists()


def test_release_selection_without_a_release_does_not_reuse_local_results(
    tmp_path: Path,
) -> None:
    """An empty answered-root scope reruns a local cell rather than reusing it."""
    workspace = init_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    collection.add_algorithm(SimpleAlgorithm())
    collection.add_instance(*simple_instance(0))
    settings: dict[str, object] = {
        "budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)],
        "seeds": [0],
    }

    assert len(collection.run_sweep(settings)) == 1
    assert collection.run_sweep(settings, results_source="release").empty
    assert len(read_attempt_rows(workspace_local_root(workspace) / SIMPLE_TASK.id)) == 2


def test_file_free_objects_reuse_and_rerun_without_registration(tmp_path: Path):
    inputs = workspace_local_root(tmp_path) / INPUT_ARTIFACTS_DIRECTORY
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = persist_instance(*simple_instance(0), inputs)
    algorithm = SimpleAlgorithm()
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    sweep = {
        "algorithms": [algorithm.uuid, algorithm, algorithm],
        "instances": [instance, instance.uuid],
        "budgets": [budget],
        "seeds": [0, 1],
    }
    status = collection.get_sweep_status(sweep)
    assert status["algorithms"] == {"requested": 1, "resolved": 1, "unresolved": []}
    assert status["cells"]["expected"] == status["cells"]["never_attempted"] == 2
    with pytest.warns(UserWarning, match="2 missing"):
        assert collection.load_results(sweep).empty
    first = collection.run_sweep(sweep)
    assert first.status.tolist() == [ResultStatus.SUCCESS] * 2
    task_store = workspace_local_root(tmp_path) / SIMPLE_TASK.id
    # A rerun in the default mode reuses what is recorded; only ``all`` mints
    # further Attempts, which is visible in the store rather than in the frame.
    assert len(collection.run_sweep(sweep)) == 2
    assert len(read_attempt_rows(task_store)) == 2
    assert len(collection.run_sweep(sweep, mode="all")) == 2
    assert len(read_attempt_rows(task_store)) == 4
    assert collection.get_sweep_status(sweep)["cells"]["recorded_percent"] == 100
    assert (
        len(collection.load_results(algorithms=[algorithm], instances=[instance])) == 2
    )
    assert (
        not collection.list_algorithm_registries()
        and not collection.list_instance_registries()
    )
    assert collection.list_sweeps() == ()
    assert sweep["algorithms"] == [algorithm.uuid, algorithm, algorithm]


def test_saved_settings_are_dynamic_and_overrides_replace(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = collection.add_instance(*simple_instance(0))
    algorithm = SimpleAlgorithm()
    collection.add_algorithm(algorithm)
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    sweep = {
        "algorithms": [algorithm],
        "instances": [instance],
        "budgets": [budget],
        "seeds": [0],
    }
    path = collection.save_sweep("small", sweep)
    before = path.read_bytes()
    loaded = collection.load_sweep(path)
    assert loaded["algorithms"] == [algorithm.uuid]
    assert loaded["instances"] == [instance.uuid]
    assert loaded["budgets"] == [budget]
    assert "alg_seeds" not in loaded
    assert collection.run_sweep(path, seeds=[9]).alg_seed.tolist() == [9]
    assert collection.load_results(path, seeds=None).alg_seed.tolist() == [9]
    assert collection.load_results(path, seeds=[]).empty
    assert collection.run_sweep(path, algorithms=[]).empty
    assert collection.get_sweep_status(path, instances=[])["cells"]["expected"] == 0
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="finite budgets and seeds"):
        collection.run_sweep()
    dynamic = collection.save_sweep("dynamic", {"budgets": [budget], "seeds": [0]})
    assert collection.load_sweep(dynamic)["algorithms"] is None
    collection.add_algorithm(SimpleAlgorithm(scale=2.0))
    assert collection.get_sweep_status(dynamic)["algorithms"]["resolved"] == 2
    assert (
        collection.get_sweep_status(path, algorithms=None)["algorithms"]["resolved"]
        == 2
    )
    with pytest.raises(FileExistsError):
        collection.save_sweep("small", sweep)


def test_unresolved_and_resource_refusal_do_not_write(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    sweep = {
        "algorithms": ["unknown"],
        "instances": ["missing"],
        "budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)],
        "seeds": [0],
    }
    status = collection.get_sweep_status(sweep)
    assert status["algorithms"]["unresolved"] == ["unknown"]
    assert status["cells"]["expected"] is None
    with pytest.raises(ValueError, match="Unresolved"):
        collection.load_results(sweep)
    with pytest.raises(ValueError, match="enforcing"):
        collection.run_sweep(sweep, time_limit=1.0)
    # init_workspace creates local/ eagerly now; the refusal still writes no
    # task directory under it.
    assert not (workspace_local_root(tmp_path) / SIMPLE_TASK.id).exists()
    with pytest.raises(TypeError, match="bare string"):
        collection.load_results(algorithms="unknown")
    with pytest.raises(ValueError, match="Unknown Sweep"):
        collection.save_sweep("invalid", {"alg_seeds": [0]})


def test_nonprivate_uses_no_privacy_and_generators_are_consumed_once(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = collection.add_instance(*simple_instance(0))
    results = collection.run_sweep(
        algorithms=(x for x in [NonprivateAlgorithm()]),
        instances=(x for x in [instance]),
        budgets=(x for x in [ApproxDPBudget(eps=1.0, delta=1e-5)]),
        seeds=(x for x in [0]),
    )
    assert results.budget.tolist() == [NO_PRIVACY]
    selected = collection.load_results(
        algorithms=[NonprivateAlgorithm()],
        instances=[instance],
        budgets=[ApproxDPBudget(eps=1.0, delta=1e-5)],
        seeds=None,
    )
    assert selected.budget.tolist() == [NO_PRIVACY]
    assert selected.alg_seed.tolist() == [0]


def test_live_objects_bypass_unloadable_unrelated_registry(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = persist_instance(
        *simple_instance(0), workspace_local_root(tmp_path) / INPUT_ARTIFACTS_DIRECTORY
    )
    algorithm = SimpleAlgorithm()
    registry = authoring_registry_path(
        workspace_local_root(tmp_path) / SIMPLE_TASK.id, "algorithm", "unavailable"
    )
    write_json_atomic(
        registry,
        {
            "task": SIMPLE_TASK.id,
            "kind": "algorithm",
            "entries": {
                "unavailable": {"path": "unavailable_package:Algorithm", "config": {}}
            },
        },
    )
    sweep = {
        "algorithms": [algorithm.uuid, algorithm],
        "instances": [instance.uuid, instance],
        "budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)],
        "seeds": [0],
    }
    assert collection.run_sweep(sweep).status.tolist() == [ResultStatus.SUCCESS]
    assert len(collection.load_results(sweep)) == 1
    assert collection.get_sweep_status(sweep)["cells"]["recorded"] == 1


def test_reconcile_ignores_an_unrelated_broken_registry(
    collection: TaskCollection,
) -> None:
    """Recovery has the collection's Task, so it loads no Algorithm registry."""
    task_store = local_task_root(collection)
    algorithm = SimpleAlgorithm()
    collection.add_algorithm(algorithm, registry="selected")
    instance = collection.add_instance(*simple_instance(0))
    write_json_atomic(
        authoring_registry_path(task_store, "algorithm", "broken"),
        {"task": SIMPLE_TASK.id, "kind": "algorithm", "entries": _UNIMPORTABLE},
    )
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        alg_seed=0,
    )
    abandon_claim(task_store, coordinate)

    assert collection.reconcile() == [coordinate.uuid]


def test_status_counts_recorded_resource_failures_and_unbounded_attempts(
    tmp_path: Path,
):
    root = workspace_local_root(tmp_path)
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    instance = collection.add_instance(*simple_instance(0))
    algorithm = SimpleAlgorithm()
    collection.add_algorithm(algorithm)
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    statuses = [
        "success",
        "timeout",
        "memory_limit_exceeded",
        "invalid_regime",
        "algorithm_error",
    ]
    write_attempt_partition(
        root / SIMPLE_TASK.id,
        [
            current_result(
                instance_uuid=instance.uuid,
                alg_uuid=algorithm.uuid,
                budget=budget.dump_config(),
                alg_seed=seed,
                status=status,
            )
            for seed, status in enumerate(statuses)
        ],
    )
    sweep = {"budgets": [budget], "seeds": range(6)}
    cells = collection.get_sweep_status(sweep)["cells"]
    assert cells == {
        "expected": 6,
        "recorded": 5,
        "resource_failures": 2,
        "never_attempted": 1,
        "recorded_percent": pytest.approx(500 / 6),
    }
    observed = collection.get_sweep_status()["cells"]
    assert observed["expected"] is None and observed["never_attempted"] is None
    assert (observed["recorded"], observed["resource_failures"]) == (5, 2)
    unresolved = collection.get_sweep_status(
        sweep, algorithms=[algorithm.uuid, "missing"]
    )
    assert unresolved["cells"]["expected"] is None
    assert unresolved["cells"]["recorded"] == 5
    with pytest.warns(UserWarning, match="1 missing of 6 expected trials"):
        assert len(collection.load_results(sweep)) == 5


def test_saved_missing_grid_never_infers_seed_zero(tmp_path: Path):
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    path = collection.save_sweep(
        "no-seeds", {"budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)]}
    )
    assert collection.load_sweep(path)["seeds"] is None
    with pytest.raises(ValueError, match="explicit finite budgets and seeds"):
        collection.run_sweep(path)
