"""Public authoring and lifecycle behavior of the task-collection helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np
import pandas as pd
import pytest

from dpstats.budgets import ApproxDPBudget
from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.analysis import (
    aggregate_trials,
    measure_results,
)
from dpstats.domains import BoxDomain
from dpstats.instance import Instance, load_instance_input
from dpstats.metric import Metric
from dpstats.task import Task, TaskInput, TaskMetadata
from dpstats.results import ResultStatus
from dpstats.sweep import (
    BudgetPolicy,
    CellContext,
    CellScheduler,
    declare_limits_enforced,
    build_executor_scheduler,
    schedule_cells_serially,
)
from dpstats.sweep.coordinates import (
    read_sweep,
    validate_sweep,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
)
from dpstats.tasks import MeanEstimation
from dpstats.transport.canonical import (
    algorithm_identity,
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.transport.documents import write_json_atomic
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    PARTITION_PREFIX,
    results_directory,
    sweeps_directory,
    task_collection_directory,
    workspace_local_root,
)
from dpstats.workspace.registries import (
    authoring_registry_path,
    file_registry_entry,
)
from dpstats.transport.registry import (
    JSONRegistry,
    read_registry,
)
from tests.attempts import (
    abandon_claim,
    current_result,
    read_attempt_rows,
    write_attempt_partition,
)
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
    simple_instance,
    simple_producer,
)
from tests.workspace_helpers import BuiltWorkspace, build_workspace
from tests.sweep_helpers import (
    add_instances,
    local_task_root,
    observing_schedule,
    sweep_coordinates,
    write_sweep,
)

_T = TypeVar("_T")


def _released_first_coordinate(output: SimpleOutput) -> float:
    """Return the first released coordinate as an Output-only Metric value."""
    return float(output.result[0])


released_first_coordinate = Metric(
    task=SIMPLE_TASK,
    name="released_first_coordinate",
    calculation=_released_first_coordinate,
)

# The document these declarations must produce, spelled out independently of
# the Sweep writer.
_AUTHORED_RECIPE = {
    "task": SIMPLE_TASK.id,
    "algorithms": None,
    "instances": None,
    "budgets": [
        {
            "path": "dpstats.budgets.families:ApproxDPBudget",
            "config": {"eps": 1.0, "delta": 1e-5},
        }
    ],
    "seeds": [0, 1, 2],
    "time_limit": None,
    "memory_limit_gb": None,
}


def _sweep(*seeds: int, eps: float = 1.0, **limits: float) -> dict[str, Any]:
    """Declare one approximate-DP budget over the given mechanism seeds."""
    return {
        "budgets": [ApproxDPBudget(eps=eps, delta=1e-5)],
        "seeds": list(seeds),
        **limits,
    }


class _CountingBudget(BudgetPolicy):
    """A public policy whose observable calls prove one shared grid resolution."""

    eps: float
    calls: ClassVar[int] = 0

    def resolve(self, metadata: TaskMetadata) -> ApproxDPBudget:
        type(self).calls += 1
        return ApproxDPBudget(eps=self.eps, delta=1.0 / metadata.n_samples**2)


def _install_collection(storage: Path, *, sweeps: Sequence[str] = ("toy",)) -> Path:
    """Install one algorithm, two Instances, and the named Sweeps; return the workspace."""
    workspace = init_workspace(storage)
    collection = open_collection(workspace, SIMPLE_TASK)
    collection.add_algorithm(SimpleAlgorithm(), registry="toy")
    add_instances(collection, [simple_producer(n=8, d=2)], seeds=(0, 1), registry="toy")
    for name in sweeps:
        collection.save_sweep(name, _sweep(0))
    return workspace


def test_save_sweep_writes_the_authored_sweep_document(
    collection: TaskCollection,
) -> None:
    path = collection.save_sweep(
        "toy",
        {
            "budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)],
            "seeds": [0, 1, 2],
        },
    )

    assert path.name == "toy.json"
    assert collection.list_sweeps() == (path,)
    assert canonical_json_bytes(decode_json_document(path)) == canonical_json_bytes(
        _AUTHORED_RECIPE
    )
    validate_sweep(path, expected_task=SIMPLE_TASK.id)


def test_save_sweep_refuses_an_invalid_declaration_before_writing(
    collection: TaskCollection,
) -> None:
    with pytest.raises(ValueError, match="'time_limit' must be positive"):
        collection.save_sweep(
            "toy",
            _sweep(0, time_limit=0.0),
        )

    assert collection.list_sweeps() == ()


def _tree(root: Path) -> list[str]:
    """Every path under ``root``, so a refusal can prove it wrote nothing."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


@pytest.mark.parametrize(
    ("selection", "refusal", "message"),
    [
        ({"budgets": [1.0]}, ValueError, "valid dictionary"),
        ({"seeds": [-1]}, ValueError, "non-negative integer"),
        ({"algorithms": [SimpleAlgorithm]}, TypeError, "UUIDs or BaseAlgorithm"),
        ({"instances": [7]}, TypeError, "UUIDs or Instance"),
        ({"algorithms": "abc"}, TypeError, "bare string"),
        ({"sweep": {"bogus": 1}}, ValueError, "Unknown Sweep fields"),
        ({"sweep": {"task": "other"}}, ValueError, "Sweep belongs to Task 'other'"),
    ],
)
def test_every_lifecycle_entry_shares_one_settings_refusal(
    tmp_path: Path,
    selection: dict[str, Any],
    refusal: type[Exception],
    message: str,
) -> None:
    """A malformed public input is refused identically by all four entries, writing nothing."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    before = _tree(tmp_path)

    for entry in (
        collection.load_results,
        collection.get_sweep_status,
        collection.get_sweep_coordinates,
        collection.run_sweep,
    ):
        with pytest.raises(refusal, match=message):
            entry(**selection)

    assert _tree(tmp_path) == before


def test_a_live_object_of_another_task_is_refused_before_anything_resolves(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path / "simple")
    collection = open_collection(workspace, SIMPLE_TASK)
    foreign = open_collection(init_workspace(tmp_path / "mean"), MeanEstimation)
    foreign_instance = foreign.add_instance(
        MeanEstimationMetadata(
            n_samples=2, domain=BoxDomain(bounds=np.array([[0.0, 1.0]]))
        ),
        MeanEstimationInput(X=np.array([[0.25], [0.75]])),
        {"ground_truth": [0.5]},
    )
    before = _tree(tmp_path / "simple")

    for entry in (
        collection.load_results,
        collection.get_sweep_status,
        collection.get_sweep_coordinates,
        collection.run_sweep,
    ):
        with pytest.raises(
            ValueError, match="algorithm object belongs to another Task"
        ):
            entry(algorithms=[GaussianMechanism()])
        with pytest.raises(ValueError, match="instance object belongs to another Task"):
            entry(instances=[foreign_instance])

    assert _tree(tmp_path / "simple") == before


def test_conflicting_live_records_are_refused_by_every_entry(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    instance = next(iter(collection.load_instances().values()))
    changed = instance.model_copy(update={"metric_arguments": {"reference": [9.0]}})

    for entry in (
        collection.load_results,
        collection.get_sweep_status,
        collection.get_sweep_coordinates,
        collection.run_sweep,
    ):
        with pytest.raises(ValueError, match="Conflicting live records"):
            entry(instances=[instance, changed])


@pytest.mark.parametrize("scope", ["algorithm_registries", "instance_registries"])
def test_an_empty_registry_list_is_refused_by_every_entry(
    tmp_path: Path, scope: str
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    for entry in (
        collection.load_results,
        collection.get_sweep_status,
        collection.get_sweep_coordinates,
        collection.run_sweep,
    ):
        with pytest.raises(ValueError, match="cannot be empty"):
            entry(collection.list_sweeps()[0], **{scope: []})


def test_duplicate_selectors_are_recorded_once(tmp_path: Path) -> None:
    """A live object and its UUID name one algorithm, so a saved Sweep lists it once."""
    workspace = _install_collection(tmp_path, sweeps=())
    collection = open_collection(workspace, SIMPLE_TASK)
    algorithm = SimpleAlgorithm()

    path = collection.save_sweep(
        "dedup",
        {"algorithms": [algorithm, algorithm.uuid, algorithm], **_sweep(0)},
    )

    assert collection.load_sweep(path)["algorithms"] == [algorithm.uuid]


def test_open_collection_has_one_typed_task_binding(tmp_path: Path) -> None:
    """Opening resolves a built-in id once and also accepts an uninstalled custom Task."""
    workspace = init_workspace(tmp_path)

    assert open_collection(workspace, task_id=MeanEstimation.id).task is MeanEstimation
    assert open_collection(workspace, SIMPLE_TASK).task is SIMPLE_TASK
    with pytest.raises(ValueError, match="exactly one"):
        open_collection(workspace)
    with pytest.raises(ValueError, match="exactly one"):
        open_collection(workspace, SIMPLE_TASK, task_id=SIMPLE_TASK.id)
    with pytest.raises(ValueError, match="Unknown built-in Task"):
        open_collection(workspace, task_id=SIMPLE_TASK.id)


def test_same_task_id_does_not_substitute_a_different_task_object(
    tmp_path: Path,
) -> None:
    """A matching serialized id cannot make another runtime Task authoritative."""
    duplicate = Task(
        SIMPLE_TASK.id,
        "Duplicate simple task",
        SIMPLE_TASK.description,
        SimpleMetadata,
        SimpleInput,
        SimpleOutput,
    )
    SimpleMetadata.task = SIMPLE_TASK
    collection = open_collection(init_workspace(tmp_path), duplicate)

    with pytest.raises(ValueError, match="collection holds Task"):
        collection.add_algorithm(SimpleAlgorithm())


def test_open_collection_reports_what_the_store_installs(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path, sweeps=("toy", "wide"))

    collection = open_collection(workspace, SIMPLE_TASK)

    assert [path.name for path in collection.list_sweeps()] == ["toy.json", "wide.json"]
    assert collection.list_algorithm_registries() == ("toy",)
    assert collection.list_instance_registries() == ("toy",)
    instances = collection.load_instances()
    assert len(instances) == 2
    assert all(instance.task is SIMPLE_TASK for instance in instances.values())
    assert collection.load_instances(registries=("toy",)) == instances
    assert collection.load_instances(registries="toy") == instances
    assert collection.load_instances(registries=None) == instances
    assert collection.load_algorithms(registries="toy") == collection.load_algorithms(
        registries=["toy"]
    )
    assert collection.get_sweep_status(
        collection.list_sweeps()[0],
        algorithm_registries="toy",
        instance_registries="toy",
    ) == collection.get_sweep_status(
        collection.list_sweeps()[0],
        algorithm_registries=["toy"],
        instance_registries=["toy"],
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        collection.load_instances(registries=[])
    with pytest.raises(ValueError, match="cannot repeat"):
        collection.load_instances(registries=["toy", "toy"])
    with pytest.raises(ValueError, match="not installed"):
        collection.load_instances(registries=["absent"])


@pytest.mark.parametrize("sweeps", ["none", "multiple", "changed"])
def test_results_and_measurement_do_not_require_a_saved_sweep(
    tmp_path: Path, sweeps: str
) -> None:
    """Changing execution declarations does not change already observed Results."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    sweep = collection.list_sweeps()[0]
    observed = collection.run_sweep(sweep)
    measured = measure_results(observed, released_first_coordinate)
    collection = TaskCollection.from_sweep(sweep, workspace, task=SIMPLE_TASK)
    if sweeps != "multiple":
        sweep.unlink()
    if sweeps != "none":
        collection.save_sweep(
            "different",
            {
                "budgets": [ApproxDPBudget(eps=2.0, delta=1e-6)],
                "seeds": [9],
            },
        )

    retained = collection.load_results()

    # budget_group reflects this call's own (absent) budgets declaration, not
    # the sweep observed was run under, so it is excluded from this
    # comparison; run_sweep and load_results return the same shape,
    # including source/release_version.
    dropped = ["payload", "budget_group"]
    pd.testing.assert_frame_equal(
        retained.drop(columns=dropped), observed.drop(columns=dropped)
    )
    for actual, expected in zip(retained["payload"], observed["payload"]):
        np.testing.assert_array_equal(actual.result, expected.result)
    pd.testing.assert_frame_equal(
        measure_results(retained, released_first_coordinate).drop(columns=dropped),
        measured.drop(columns=dropped),
    )


def test_the_handle_loads_the_algorithm_groups_it_installs(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    ((uuid, algorithm),) = collection.load_algorithms().items()

    assert isinstance(algorithm, SimpleAlgorithm)
    assert uuid == algorithm.uuid
    assert list(collection.load_algorithms(registries=("toy",))) == [uuid]
    with pytest.raises(ValueError, match="not installed for this Task"):
        collection.load_algorithms(registries=("absent",))


@pytest.mark.parametrize("kind", ["algorithm", "instance"])
def test_results_follow_current_registry_scope_without_deleting_attempts(
    tmp_path: Path, kind: Literal["algorithm", "instance"]
) -> None:
    """Moving declarations preserves visibility; restoring them reveals retained Attempts."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    observed = collection.run_sweep(collection.list_sweeps()[0])
    task_root = task_collection_directory(
        workspace_local_root(workspace), SIMPLE_TASK.id
    )
    source = authoring_registry_path(task_root, kind, "toy")
    deferred = authoring_registry_path(task_root, kind, "deferred")
    declarations = read_registry(source.read_bytes())
    empty = JSONRegistry(task=SIMPLE_TASK.id, kind=kind, entries={}).model_dump()

    write_json_atomic(deferred, declarations.model_dump())
    write_json_atomic(source, empty)

    # budget_group reflects this call's own (absent) budgets declaration, not
    # the sweep observed was run under, so it is excluded from this
    # comparison; run_sweep and load_results return the same shape,
    # including source/release_version.
    dropped = ["payload", "budget_group"]
    pd.testing.assert_frame_equal(
        collection.load_results().drop(columns=dropped),
        observed.drop(columns=dropped),
    )
    scoped = (
        collection.load_results(algorithm_registries=["toy"])
        if kind == "algorithm"
        else collection.load_results(instance_registries=["toy"])
    )
    assert scoped.empty

    write_json_atomic(deferred, empty)
    assert collection.load_results().empty

    write_json_atomic(deferred, declarations.model_dump())
    restored = collection.load_results()
    pd.testing.assert_frame_equal(
        restored.drop(columns=dropped), observed.drop(columns=dropped)
    )
    for actual, expected in zip(restored["payload"], observed["payload"]):
        np.testing.assert_array_equal(actual.result, expected.result)


def test_overlapping_sweeps_select_the_same_collection_results(tmp_path: Path) -> None:
    """Execution selections share cells without owning separate Results."""
    workspace = _install_collection(tmp_path, sweeps=())
    collection = open_collection(workspace, SIMPLE_TASK)
    first = collection.save_sweep(
        "first",
        _sweep(1, 0),
    )
    second = collection.save_sweep(
        "second",
        _sweep(2, 1),
    )
    collection.run_sweep(first)
    collection.run_sweep(second)

    observed = collection.load_results()
    first_results = collection.load_results(sweep=first)
    second_results = collection.load_results(sweep=second)

    assert observed["alg_seed"].tolist() == [0, 1, 2, 0, 1, 2]
    assert first_results["alg_seed"].tolist() == [1, 0, 1, 0]
    assert second_results["alg_seed"].tolist() == [2, 1, 2, 1]
    first_shared = first_results.loc[first_results["alg_seed"] == 1, "payload"]
    second_shared = second_results.loc[second_results["alg_seed"] == 1, "payload"]
    for left, right in zip(first_shared, second_shared):
        np.testing.assert_array_equal(left.result, right.result)


def test_add_algorithm_files_it_in_the_named_group(collection: TaskCollection) -> None:
    uuid = collection.add_algorithm(SimpleAlgorithm(), registry="toy")

    assert list(collection.load_algorithms(registries=("toy",))) == [uuid]
    assert collection.list_algorithm_registries() == ("toy",)


def test_add_algorithm_refuses_another_tasks_algorithm(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError, match="mean_estimation"):
        collection.add_algorithm(GaussianMechanism())

    assert not task_collection_directory(
        workspace_local_root(workspace), "mean_estimation"
    ).exists()


def test_add_instance_persists_the_input_and_files_the_instance(
    collection: TaskCollection,
) -> None:
    metadata, input_data, references = simple_instance(0, n=8, d=2)

    instance = collection.add_instance(
        metadata, input_data, references, label="toy-0", registry="toy"
    )

    assert collection.load_instances(registries=("toy",)) == {instance.uuid: instance}
    assert instance.label == "toy-0"
    stored = SimpleInput.model_validate(
        load_instance_input(instance, collection._writable_input_root("Running a cell"))
    )
    assert np.array_equal(stored.X, input_data.X)


def test_add_instance_refuses_another_tasks_metadata(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError, match="mean_estimation"):
        collection.add_instance(
            MeanEstimationMetadata(
                n_samples=2, domain=BoxDomain(bounds=np.array([[0.0, 1.0]]))
            ),
            MeanEstimationInput(X=np.array([[0.25], [0.75]])),
            {"ground_truth": [0.5]},
        )

    assert not task_collection_directory(
        workspace_local_root(workspace), "mean_estimation"
    ).exists()


def test_reconcile_clears_the_claim_a_vanished_runner_left(tmp_path: Path) -> None:
    collection = open_collection(_install_collection(tmp_path), SIMPLE_TASK)
    held, _ = sweep_coordinates(collection, collection.list_sweeps()[0])
    abandon_claim(local_task_root(collection), held)

    (cleared,) = collection.reconcile()

    assert cleared == held.uuid
    assert (
        collection.get_sweep_status(collection.list_sweeps()[0])["cells"][
            "never_attempted"
        ]
        == 2
    )


def test_save_sweep_refuses_to_replace_an_installed_sweep(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path, sweeps=())
    collection = open_collection(workspace, SIMPLE_TASK)
    installed = collection.save_sweep(
        "toy",
        _sweep(0),
    )
    authored = installed.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        collection.save_sweep(
            "toy",
            _sweep(1, eps=2.0),
        )

    assert installed.read_bytes() == authored
    replaced = collection.save_sweep(
        "toy",
        _sweep(1, eps=2.0),
        overwrite=True,
    )
    assert replaced == installed
    assert installed.read_bytes() != authored


def test_save_sweep_refuses_a_name_the_collection_could_not_list(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path, sweeps=())
    collection = open_collection(workspace, SIMPLE_TASK)

    for name in ("nested/toy", "toy.json"):
        with pytest.raises(ValueError, match="portable name"):
            collection.save_sweep(
                name,
                _sweep(0),
            )

    assert collection.list_sweeps() == ()


def test_open_collection_on_a_bare_bundle_reads_the_two_roots_it_nests(
    tmp_path: Path,
) -> None:
    bundle = _bundle(build_workspace(tmp_path, release=True))

    collection = open_collection(bundle, SIMPLE_TASK)

    assert collection.local_root is None
    assert collection.release_root == bundle
    assert collection._input_artifacts_root == bundle / INPUT_ARTIFACTS_DIRECTORY
    assert collection.release


def test_a_release_handle_refuses_every_writer(tmp_path: Path) -> None:
    """A release is immutable Results, so no operation may record into it."""
    collection = open_collection(
        _bundle(build_workspace(tmp_path, release=True)), SIMPLE_TASK
    )
    metadata = SimpleMetadata(n_samples=4, dim=2)
    writes = {
        "Running a Sweep": lambda: collection.run_sweep(collection.list_sweeps()[0]),
        "Adding an Algorithm": lambda: collection.add_algorithm(SimpleAlgorithm()),
        "Adding an Instance": lambda: collection.add_instance(
            metadata, SimpleInput(X=np.zeros((4, 2))), {}
        ),
        "Saving a Sweep": lambda: collection.save_sweep("mine", {"seeds": [0]}),
        "Reconciling": collection.reconcile,
    }
    for operation, write in writes.items():
        with pytest.raises(ValueError, match=operation) as refusal:
            write()
        assert "init_workspace" in str(refusal.value)

    # Reading is exactly what a release handle is for.
    assert collection.list_sweeps()
    assert len(collection.load_results()) == 1


def test_a_store_and_a_release_never_stand_in_one_task_collection(
    tmp_path: Path,
) -> None:
    """The separation is the directory, so neither writer can reach the other."""
    built = build_workspace(tmp_path, release=True)
    (task,) = built.tasks
    # Opening a release root as an ordinary collection still cannot write.
    with pytest.raises(ValueError, match="Running a Sweep"):
        open_collection(_bundle(built), SIMPLE_TASK).run_sweep(task.sweep_path)

    # Nor does a bare store's own local/ open as a release: it holds neither
    # a nested local/ (it is one) nor a release manifest.
    with pytest.raises(ValueError, match="neither a workspace"):
        open_collection(workspace_local_root(built.root), SIMPLE_TASK)


def _bundle(built: BuiltWorkspace) -> Path:
    """Return the bare Distribution Release bundle ``build_workspace`` cut."""
    assert built.bundle_root is not None
    return built.bundle_root


def test_a_release_reads_exactly_as_the_store_it_was_cut_from(
    tmp_path: Path,
) -> None:
    """A release ships the Results its store resolved, read back unchanged."""
    built = build_workspace(
        tmp_path,
        algorithms={
            SIMPLE_TASK.id: [SimpleAlgorithm(scale=1.0), SimpleAlgorithm(scale=2.0)]
        },
        budgets=[ApproxDPBudget(eps=1.0, delta=1e-5)],
        release=True,
    )

    stored = built.collection(SIMPLE_TASK.id).load_results()
    released = built.bundle_collection(SIMPLE_TASK.id).load_results()

    assert list(released.columns) == list(stored.columns)
    assert "attempt_number" not in released.columns
    assert len(released) == 2
    for column in ("instance_uuid", "alg_uuid", "alg_seed"):
        assert released[column].tolist() == stored[column].tolist()
    assert released["status"].tolist() == stored["status"].tolist()
    assert [payload.result.tolist() for payload in released["payload"]] == [
        payload.result.tolist() for payload in stored["payload"]
    ]


def test_a_selected_algorithm_reads_only_its_own_released_partition(
    tmp_path: Path,
) -> None:
    """A selection opens its partitions alone, so a sibling's bytes never load."""
    selected = SimpleAlgorithm(scale=1.0)
    bundle = _bundle(
        build_workspace(
            tmp_path,
            algorithms={SIMPLE_TASK.id: [selected, SimpleAlgorithm(scale=2.0)]},
            release=True,
        )
    )
    released_task = task_collection_directory(bundle, SIMPLE_TASK.id)
    for path in results_directory(released_task).glob(f"{PARTITION_PREFIX}*/*.parquet"):
        if path.parent.name.endswith(selected.uuid):
            continue
        path.write_bytes(b"not a parquet file")

    collection = open_collection(bundle, SIMPLE_TASK)
    frame = collection.load_results(algorithms=[selected.uuid])

    assert frame["alg_uuid"].tolist() == [selected.uuid]
    with pytest.raises(ValueError, match="Parquet"):
        collection.load_results()


def test_a_column_read_restores_what_validating_each_row_would(
    collection: TaskCollection,
) -> None:
    """The bulk read skips per-row validation and must still agree with it."""
    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=8, d=2)],
        [ApproxDPBudget(eps=1.0, delta=1e-5)],
        seeds=(0, 1, 2, 3),
    )
    coordinates = sweep_coordinates(collection, sweep)
    statuses = ("success", "no_release", "algorithm_error", "timeout")
    records = [
        current_result(
            instance_uuid=coordinate.instance_uuid,
            alg_uuid=coordinate.alg_uuid,
            budget=coordinate.budget.dump_config().model_dump(),
            alg_seed=coordinate.alg_seed,
            status=status,
            estimate=float(coordinate.alg_seed),
        )
        for coordinate, status in zip(coordinates, statuses)
    ]
    write_attempt_partition(local_task_root(collection), records)

    frame = collection.load_results()

    # Each row carries the status it was written with; a released Output
    # restores as the Task's typed float64 array and every other outcome's
    # stored explanation passes through unchanged.
    written = {record["coordinate"]["alg_seed"]: record for record in records}
    assert sorted(frame["status"]) == sorted(ResultStatus(name) for name in statuses)
    for row in frame.itertuples():
        record = written[row.alg_seed]
        assert row.status is ResultStatus(record["status"])
        if row.status is ResultStatus.SUCCESS:
            assert isinstance(row.payload, SimpleOutput)
            np.testing.assert_array_equal(row.payload.result, [float(row.alg_seed)])
            assert row.payload.result.dtype == np.float64
        else:
            assert row.payload == record["payload"]


def test_from_sweep_opens_the_collection_the_sweep_names(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    (installed,) = open_collection(workspace, SIMPLE_TASK).list_sweeps()

    collection = TaskCollection.from_sweep(installed, workspace, task=SIMPLE_TASK)

    assert collection.task is SIMPLE_TASK
    assert collection.list_sweeps() == (installed,)


def test_from_sweep_needs_explicit_sweep_for_each_selection(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    wide = open_collection(workspace, SIMPLE_TASK).save_sweep(
        "wide",
        _sweep(0, 1),
    )
    toy, _ = open_collection(workspace, SIMPLE_TASK).list_sweeps()

    collection = TaskCollection.from_sweep(toy, workspace, task=SIMPLE_TASK)

    assert len(collection.get_sweep_coordinates(collection.list_sweeps()[0])) == 2
    assert len(collection.get_sweep_coordinates(wide)) == 4


def test_the_handle_carries_one_sweep_from_status_to_results(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    assert (
        collection.get_sweep_status(collection.list_sweeps()[0])["cells"][
            "never_attempted"
        ]
        == 2
    )
    assert list(collection.run_sweep(collection.list_sweeps()[0])["status"]) == [
        "success",
        "success",
    ]

    results = collection.load_results()
    measurements = measure_results(results, released_first_coordinate)
    assert released_first_coordinate.name in measurements
    assert len(measurements) == 2

    assert list(results["status"]) == [ResultStatus.SUCCESS] * 2
    assert (
        collection.get_sweep_status(collection.list_sweeps()[0])["cells"]["recorded"]
        == 2
    )


def test_lifecycle_options_reach_the_operation_they_belong_to(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    assert (
        collection.get_sweep_status(collection.list_sweeps()[0])["cells"]["recorded"]
        == 0
    )

    collection.run_sweep(collection.list_sweeps()[0])
    assert len(collection.run_sweep(collection.list_sweeps()[0], mode="all")) == 2
    with pytest.raises(TypeError, match="unexpected keyword"):
        collection.get_sweep_coordinates(  # type: ignore[call-arg]
            collection.list_sweeps()[0], schedule_cells=schedule_cells_serially
        )


def test_lifecycle_narrows_the_resolved_grid_before_execution(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    ((algorithm_id, _),) = collection.load_algorithms().items()
    instance_id = next(iter(collection.load_instances()))
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)

    preview = collection.get_sweep_coordinates(
        collection.list_sweeps()[0],
        algorithms=[algorithm_id, algorithm_id],
        instances=[instance_id],
        budgets=[budget, budget],
        seeds=[0, 0],
    )

    assert len(preview) == 1
    assert collection.run_sweep(collection.list_sweeps()[0], instances=[]).empty
    assert collection.load_results(instances=[]).empty
    with pytest.raises(ValueError, match="Unresolved"):
        collection.get_sweep_coordinates(
            collection.list_sweeps()[0], algorithms=["unknown"]
        )
    assert (
        len(collection.get_sweep_coordinates(collection.list_sweeps()[0], seeds=[1]))
        == 2
    )
    assert (
        len(
            collection.get_sweep_coordinates(
                collection.list_sweeps()[0],
                budgets=[ApproxDPBudget(eps=1.0, delta=1e-4)],
            )
        )
        == 2
    )


def test_registry_scope_reads_one_name_as_readily_as_a_list(tmp_path: Path) -> None:
    """A bare name scopes a call; it is never read as a sequence of letters."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    sweep = collection.list_sweeps()[0]

    named = collection.get_sweep_coordinates(
        sweep, algorithm_registries="toy", instance_registries="toy"
    )
    listed = collection.get_sweep_coordinates(
        sweep, algorithm_registries=["toy"], instance_registries=["toy"]
    )
    assert named["coordinate_uuid"].tolist() == listed["coordinate_uuid"].tolist()
    assert len(named) == 2

    collection.run_sweep(sweep, algorithm_registries="toy", instance_registries="toy")
    assert (
        len(
            collection.load_results(
                algorithm_registries="toy", instance_registries="toy"
            )
        )
        == len(
            collection.load_results(
                algorithm_registries=["toy"], instance_registries=["toy"]
            )
        )
        == 2
    )
    with pytest.raises(ValueError, match="not installed for this Task"):
        collection.run_sweep(sweep, instance_registries="absent")


def test_live_labels_reach_existing_results_without_persisting(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    collection.run_sweep(collection.list_sweeps()[0])
    algorithm = next(iter(collection.load_algorithms().values()))
    instance = next(iter(collection.load_instances().values()))
    panel_algorithm = algorithm.model_copy(update={"label": "panel algorithm"})
    panel_instance = instance.model_copy(update={"label": "panel instance"})
    paths = list((workspace_local_root(workspace) / SIMPLE_TASK.id).glob("*/*.json"))
    original = {path: path.read_bytes() for path in paths}
    results = collection.load_results(
        algorithms=[algorithm.uuid, panel_algorithm, algorithm],
        instances=[panel_instance, instance.uuid, instance],
    )
    assert len(results) == 1
    assert all(value is panel_algorithm for value in results["algorithm"])
    assert all(value is panel_instance for value in results["instance"])
    assert {path: path.read_bytes() for path in paths} == original
    assert collection.load_algorithms()[algorithm.uuid].label == algorithm.label
    changed_references = instance.model_copy(
        update={"metric_arguments": {"reference": [999.0]}}
    )
    with pytest.raises(ValueError, match="Conflicting live records"):
        collection.load_results(instances=[instance, changed_references])


def test_omitted_sweep_has_no_grid_when_the_store_holds_several(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path, sweeps=("toy", "wide"))
    collection = open_collection(workspace, SIMPLE_TASK)

    assert collection.get_sweep_status()["cells"]["expected"] is None

    assert len(collection.get_sweep_coordinates(collection.list_sweeps()[0])) == 2


def test_omitted_sweep_has_no_grid_when_the_store_holds_none(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path, sweeps=())
    collection = open_collection(workspace, SIMPLE_TASK)

    assert collection.list_sweeps() == ()
    assert collection.get_sweep_status()["cells"]["expected"] is None


def test_explicit_algorithms_survive_registry_removal_and_carry_types(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    collection.run_sweep(collection.list_sweeps()[0])
    algorithms = collection.load_algorithms(registries=("toy",))
    registry = authoring_registry_path(
        workspace_local_root(workspace) / SIMPLE_TASK.id, "algorithm", "toy"
    )
    registry.unlink()
    results = collection.load_results(algorithms=algorithms.values())
    assert len(results) == 2
    assert results.columns.tolist() == [
        "task",
        "instance",
        "instance_uuid",
        "algorithm",
        "alg_uuid",
        "budget",
        "alg_seed",
        "status",
        "payload",
        "runtime_s",
        "peak_rss_mb",
        "source",
        "release_version",
        "budget_group",
        "algorithm_label",
        "instance_label",
    ]
    assert all(
        isinstance(algorithm, SimpleAlgorithm) for algorithm in results.algorithm
    )
    assert results["algorithm_label"].tolist() == [
        str(algorithm) for algorithm in results.algorithm
    ]
    measured = measure_results(results, released_first_coordinate)
    aggregated = aggregate_trials(measured, metric=released_first_coordinate.name)
    assert [
        instance.uuid for instance in aggregated.instance
    ] == results.instance_uuid.tolist()
    assert [type(algorithm) for algorithm in aggregated.algorithm] == [
        SimpleAlgorithm,
        SimpleAlgorithm,
    ]
    assert not registry.exists()


def test_finite_sweep_warns_without_installing_custom_algorithms(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    collection.run_sweep(collection.list_sweeps()[0])
    sweep = collection.save_sweep(
        "more-seeds",
        _sweep(0, 1, time_limit=1.0, memory_limit_gb=1.0),
    )
    custom = SimpleAlgorithm(scale=2.0)
    with pytest.warns(UserWarning) as warnings:
        results = collection.load_results(
            algorithms=[SimpleAlgorithm(), custom], sweep=sweep
        )
    assert len(results) == 2
    assert len(warnings) == 1
    assert "6 missing" in str(warnings[0].message)
    assert custom.uuid not in collection.load_algorithms()
    outside = collection.save_sweep(
        "outside",
        _sweep(1, eps=2.0, time_limit=1.0, memory_limit_gb=1.0),
    )
    with pytest.warns(UserWarning, match="2 missing"):
        outside_results = collection.load_results(
            algorithms=[SimpleAlgorithm()], sweep=outside
        )
    assert outside_results.empty


def test_a_cell_recorded_under_another_hosts_delta_is_not_reported_missing(
    collection: TaskCollection,
) -> None:
    """Sweep status answers a cell whatever host resolved the policy's delta."""
    from dpstats.sweep import SubinverseSampleSizeDelta

    sweep = write_sweep(
        collection,
        [SimpleAlgorithm()],
        [simple_producer(n=317, d=1)],
        [SubinverseSampleSizeDelta(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, sweep)
    # The delta plain floating-point arithmetic returns for 317 ** -1.1: every
    # parameter is canonicalized at construction, so this and the rule's own
    # Decimal-resolved delta name the same budget and store the same
    # canonical value, even though they arrive by different arithmetic.
    recorded = coordinate.model_copy(
        update={"budget": ApproxDPBudget(eps=1.0, delta=317**-1.1)}
    )
    assert recorded.budget.dump_config() == coordinate.budget.dump_config()
    assert recorded.budget == coordinate.budget
    write_attempt_partition(
        local_task_root(collection),
        [
            current_result(
                instance_uuid=recorded.instance_uuid,
                alg_uuid=recorded.alg_uuid,
                budget=recorded.budget.dump_config().model_dump(),
            )
        ],
    )

    cells = collection.get_sweep_status(sweep)["cells"]

    assert cells["expected"] == 1
    assert cells["recorded"] == 1
    assert cells["never_attempted"] == 0


def test_budget_policy_selects_the_full_per_instance_budget(tmp_path: Path) -> None:
    from dpstats.sweep import SubinverseSampleSizeDelta

    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    policy = SubinverseSampleSizeDelta(eps=1.0)
    second_policy = SubinverseSampleSizeDelta(eps=2.0)
    fixed = ApproxDPBudget(eps=1.0, delta=1e-5)
    sweep = collection.save_sweep(
        "policy",
        {
            "budgets": [policy, second_policy, fixed],
            "seeds": [0],
            "time_limit": 1.0,
            "memory_limit_gb": 1.0,
        },
    )
    collection.run_sweep(sweep, time_limit=None, memory_limit_gb=None)
    results = collection.load_results(budgets=(policy,), seeds=(0,))
    assert len(results) == 2
    assert all(budget.eps == 1.0 for budget in results.budget)
    assert all(budget.delta != 1e-5 for budget in results.budget)
    assert list(results.budget_group) == [policy, policy]
    assert collection.load_results(algorithms=[]).empty
    assert collection.load_results(budgets=[]).empty
    assert len(collection.load_results()) == 6
    assert collection.load_results().budget_group.isna().all()
    assert len(collection.load_results(budgets=(policy, fixed))) == 4
    assert len(collection.load_results(budgets=(policy, second_policy))) == 4
    deduped = collection.load_results(budgets=(policy, policy, results.budget.iloc[0]))
    assert len(deduped) == 2
    # The literal exact budget resolves to the same value policy already
    # named on the first Instance; the first-declared entry keeps the label.
    assert list(deduped.budget_group) == [policy, policy]
    pd.testing.assert_frame_equal(
        collection.load_results(sweep=sweep, budgets=(policy,)).drop(columns="payload"),
        results.drop(columns="payload"),
    )

    more_seeds = collection.save_sweep(
        "more-policy-seeds",
        {
            "budgets": [policy, second_policy, fixed],
            "seeds": [0, 1],
            "time_limit": 1.0,
            "memory_limit_gb": 1.0,
        },
    )
    with pytest.warns(UserWarning, match="4 missing of 8 expected"):
        selected = collection.load_results(
            budgets=(policy, fixed, policy), sweep=more_seeds
        )
    assert len(selected) == 4


def test_coordinate_selection_resolves_each_policy_once_per_instance(
    tmp_path: Path,
) -> None:
    """The public coordinate view and labels share one per-Instance policy pass."""
    collection = open_collection(_install_collection(tmp_path), SIMPLE_TASK)
    policy = _CountingBudget(eps=1.0)
    _CountingBudget.calls = 0

    coordinates = collection.get_sweep_coordinates({"budgets": [policy], "seeds": [0]})

    assert len(coordinates) == 2
    assert _CountingBudget.calls == 2


def _design_collection(tmp_path: Path, designs: Sequence[np.ndarray]) -> TaskCollection:
    """A collection holding one Instance per supplied design, in order."""
    collection = open_collection(init_workspace(tmp_path), SIMPLE_TASK)
    for position, design in enumerate(designs):
        collection.add_instance(
            SimpleMetadata(n_samples=design.shape[0], dim=design.shape[1]),
            SimpleInput(X=design),
            {"reference": np.zeros(design.shape[1]).tolist()},
            label=f"design-{position}",
            registry="designs",
        )
    return collection


def _condition_number(collection: TaskCollection, instance: Instance) -> float:
    """Read one stored design through the Task's own Input type and condition it."""
    loaded = collection.load_input(instance)
    assert isinstance(loaded, SimpleInput)
    return float(np.linalg.cond(loaded.X))


def test_instance_features_computes_each_callable_over_every_instance(
    tmp_path: Path,
) -> None:
    # Diagonal designs: a diagonal matrix's singular values are its diagonal
    # entries, so kappa_2 is 4/1 and 3/3 without reproducing any library code.
    collection = _design_collection(
        tmp_path,
        [np.diag([4.0, 2.0, 1.0]), np.diag([3.0, 3.0, 3.0])],
    )
    installed = collection.load_instances(registries="designs")

    features = collection.instance_features(
        {
            "condition": lambda instance: _condition_number(collection, instance),
            "n": lambda instance: instance.metadata.n_samples,
        },
        registries="designs",
    )

    assert list(features.columns) == ["condition", "n"]
    assert features.index.name == "instance_uuid"
    assert set(features.index) == set(installed)
    conditioned = {
        installed[uuid].label: value
        for uuid, value in zip(features.index, features["condition"].tolist())
    }
    assert conditioned["design-0"] == pytest.approx(4.0)
    assert conditioned["design-1"] == pytest.approx(1.0)
    assert features["n"].tolist() == [3.0, 3.0]


def test_instance_features_reports_a_rank_deficient_design_as_infinite(
    tmp_path: Path,
) -> None:
    # A coordinate that never varies: the design spans one direction, so its
    # smaller singular value is exactly zero and the ratio is unbounded.
    collection = _design_collection(tmp_path, [np.array([[1.0, 0.0], [2.0, 0.0]])])

    features = collection.instance_features(
        {"condition": lambda instance: _condition_number(collection, instance)},
        registries="designs",
    )

    assert features["condition"].tolist() == [float("inf")]


def test_instance_features_reads_no_input_for_a_metadata_only_callable(
    tmp_path: Path,
) -> None:
    # A collection's Input-artifacts root is always derived from its trees,
    # never independently withheld; a callable that never calls load_input
    # is evidence that none was opened, whether or not one is available.
    collection = _design_collection(tmp_path, [np.diag([2.0, 1.0])])

    features = collection.instance_features(
        {"n": lambda instance: instance.metadata.n_samples}, registries="designs"
    )

    assert features["n"].tolist() == [2.0]


def test_instance_features_holds_one_input_per_instance_and_releases_it(
    tmp_path: Path,
) -> None:
    collection = _design_collection(
        tmp_path, [np.diag([2.0, 1.0]), np.diag([5.0, 1.0])]
    )
    seen: list[TaskInput] = []

    def record(instance: Instance) -> float:
        loaded = collection.load_input(instance)
        seen.append(loaded)
        return 0.0

    collection.instance_features(
        {"rows": record, "rows_again": record}, registries="designs"
    )

    # Both callables on one Instance share the materialized Input; the next
    # Instance's row materializes its own, so nothing survives the row.
    assert seen[0] is seen[1]
    assert seen[2] is seen[3]
    assert seen[0] is not seen[2]


def test_instance_features_propagates_a_callable_failure(tmp_path: Path) -> None:
    collection = _design_collection(tmp_path, [np.diag([2.0, 1.0])])

    def refuse(instance: Instance) -> float:
        raise ValueError(f"no feature for {instance.label}")

    with pytest.raises(ValueError, match="no feature for design-0"):
        collection.instance_features({"refused": refuse}, registries="designs")


def _file_stale_algorithm(collection: TaskCollection, registry: str) -> str:
    """File one entry whose configuration class this library cannot import."""
    path = "tests.registry_components:Absent"
    record: dict[str, Any] = {"path": path, "config": {}}
    uuid = algorithm_identity(record)
    file_registry_entry(
        local_task_root(collection),
        task_id=collection.task.id,
        kind="algorithm",
        uuid=uuid,
        record=record,
        registry=registry,
    )
    return uuid


def _once(call: Callable[[], _T]) -> _T:
    """Run *call*, requiring exactly one skip warning, and return its value."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        value = call()
    skips = [
        w for w in recorded if str(w.message).startswith("Skipped 1 of 2 algorithm")
    ]
    assert len(skips) == 1
    assert len(recorded) == 1
    return value


def test_a_skipped_algorithm_is_absent_from_every_collection_reader(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    stale = _file_stale_algorithm(collection, "toy")
    sweep = collection.save_sweep("stale", _sweep(0))
    valid = SimpleAlgorithm().uuid

    loaded = _once(lambda: collection.load_algorithms(strict=False))
    declared = _once(lambda: collection.get_sweep_coordinates(sweep, strict=False))
    _once(lambda: collection.run_sweep(sweep, strict=False))
    results = _once(lambda: collection.load_results(sweep, strict=False))
    status = _once(lambda: collection.get_sweep_status(sweep, strict=False))

    assert list(loaded) == [valid]
    assert set(declared["alg_uuid"]) == {valid}
    assert set(results["alg_uuid"]) == {valid}
    assert stale not in set(declared["alg_uuid"]) | set(results["alg_uuid"])
    assert status["cells"]["expected"] == len(declared)


def test_collection_readers_refuse_the_stale_entry_by_default(tmp_path: Path) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    _file_stale_algorithm(collection, "toy")
    sweep = collection.save_sweep("stale", _sweep(0))
    refusal = "Could not reconstruct"

    with pytest.raises(ValueError, match=refusal):
        collection.load_algorithms(strict=True)
    with pytest.raises(ValueError, match=refusal):
        collection.load_results(sweep)
    with pytest.raises(ValueError, match=refusal):
        collection.get_sweep_coordinates(sweep)
    with pytest.raises(ValueError, match=refusal):
        collection.get_sweep_status(sweep)
    with pytest.raises(ValueError, match=refusal):
        collection.run_sweep(sweep)
    with pytest.raises(ValueError, match=refusal):
        sweep_coordinates(collection, sweep)


def test_naming_a_skipped_algorithm_is_refused_with_its_reason(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    stale = _file_stale_algorithm(collection, "toy")
    sweep = collection.save_sweep("stale", _sweep(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="Unresolved algorithms") as loading:
            collection.load_results(sweep, algorithms=[stale], strict=False)
        with pytest.raises(ValueError, match="Unresolved algorithms") as groups:
            sweep_coordinates(collection, sweep, algorithm_ids=[stale], strict=False)

    for refusal in (loading, groups):
        assert f"Skipped as unloadable: {stale} (" in str(refusal.value)
        assert "Absent" in str(refusal.value)


def test_a_sweep_naming_another_task_is_refused_on_every_read(
    tmp_path: Path, collection: TaskCollection
) -> None:
    other = tmp_path / "other.json"
    write_json_atomic(other, {"task": "some_other_task"})

    with pytest.raises(ValueError, match="Sweep belongs to Task 'some_other_task'"):
        collection.load_sweep(other)
    with pytest.raises(ValueError, match="Sweep belongs to Task 'some_other_task'"):
        collection.load_results(sweep=other, algorithms=[], instances=[])
    with pytest.raises(ValueError, match="Sweep belongs to Task 'some_other_task'"):
        collection.load_results(
            sweep={"task": "some_other_task"}, algorithms=[], instances=[]
        )


def test_a_bare_sweep_name_found_in_no_tree_is_refused_naming_where_it_looked(
    tmp_path: Path,
) -> None:
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError, match="No saved Sweep named 'missing'") as caught:
        collection.load_sweep("missing")
    assert str(
        sweeps_directory(
            task_collection_directory(collection.local_root, SIMPLE_TASK.id)
        )
    ) in str(caught.value)
    assert collection.load_sweep("toy")["task"] == SIMPLE_TASK.id


@pytest.mark.parametrize(
    ("scheduler", "limits"),
    [("serial", {"time_limit": 1.0}), ("executor", {"memory_limit_gb": 1.0})],
)
def test_declared_limits_are_refused_on_every_unenforcing_schedule_path(
    tmp_path: Path, scheduler: str, limits: dict[str, float]
) -> None:
    """A Sweep declaring a limit never runs under a scheduler that cannot enforce it."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    sweep = collection.save_sweep("limited", _sweep(0, **limits))

    with ThreadPoolExecutor(max_workers=2) as pool:
        schedule_cells: CellScheduler = (
            schedule_cells_serially
            if scheduler == "serial"
            else build_executor_scheduler(pool, 2)
        )
        with pytest.raises(ValueError, match="enforcing schedule_cells"):
            collection.run_sweep(sweep, schedule_cells=schedule_cells)

    assert read_attempt_rows(local_task_root(collection)) == []


def test_a_scheduler_declaring_enforcement_runs_declared_limits(
    tmp_path: Path,
) -> None:
    """The declaration admits the limits and hands each cell its own caps."""
    workspace = _install_collection(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    sweep = collection.save_sweep(
        "limited", _sweep(0, time_limit=1.0, memory_limit_gb=1.0)
    )
    offered: list[CellContext] = []

    returned = collection.run_sweep(
        sweep,
        schedule_cells=declare_limits_enforced(observing_schedule(offered.append)),
    )

    assert len(returned) == 2
    assert {context.time_limit_s for context in offered} == {1.0}
    assert {context.memory_limit_bytes for context in offered} == {1024**3}
