"""Public authoring and selection contracts for named registry groups."""

from __future__ import annotations

from typing import Any

import json
from pathlib import Path

import numpy as np
import pytest

from dpstats.workspace import TaskCollection
from dpstats.instance import Instance, persist_instance
from dpstats.analysis import measure_results
from dpstats.sweep import preview_sweep
from dpstats.tasks.mean_estimation.algorithms.sample_mean import (
    SampleMean,
)
from dpstats.metric import Metric
from dpstats.budgets import PureDPBudget
from dpstats.workspace.layout import workspace_local_root
from dpstats.workspace.registries import (
    installed_registry_names,
    registry_document_path,
)
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
    sized_producer,
)
from tests.sweep_helpers import local_task_root, sweep_coordinates


def _sweep(collection: TaskCollection) -> Path:
    return collection.save_sweep(
        "test", {"budgets": [PureDPBudget(eps=1.0)], "seeds": [0]}
    )


def _two_points() -> tuple[SimpleMetadata, SimpleInput, dict[str, Any]]:
    """Two fixed samples whose mean, 2.0, is also the public reference."""
    return (
        SimpleMetadata(n_samples=2, dim=1),
        SimpleInput(X=np.array([[1.0], [3.0]])),
        {"reference": [2.0]},
    )


def test_named_registry_additions_are_selected_only_at_invocation(
    collection: TaskCollection,
) -> None:
    """An exact selection changes the live cross-product, never the Sweep."""
    sweep = _sweep(collection)
    authored = sweep.read_bytes()
    first = collection.add_algorithm(SimpleAlgorithm(scale=1.0), registry="baseline")
    second = collection.add_algorithm(SimpleAlgorithm(scale=2.0), registry="bounded")
    instance = collection.add_instance(*_two_points(), registry="baseline")

    selected = sweep_coordinates(
        collection,
        sweep,
        algorithm_registries=("bounded",),
        instance_registries=("baseline",),
    )

    assert sweep.read_bytes() == authored
    assert [
        (coordinate.alg_uuid, coordinate.instance_uuid) for coordinate in selected
    ] == [(second, instance.uuid)]
    all_coordinates = sweep_coordinates(
        collection,
        sweep,
    )
    assert {coordinate.alg_uuid for coordinate in all_coordinates} == {first, second}


def test_generated_additions_commit_prior_siblings_before_a_later_failure(
    collection: TaskCollection,
) -> None:
    """A later producer error names its seed and leaves earlier Instances installed."""

    base_producer = sized_producer(n_samples=2, d=1)

    def failing_producer(
        seed: int,
    ) -> tuple[SimpleMetadata, SimpleInput, dict[str, Any]]:
        if seed == 2:
            raise ValueError("producer refused input")
        return base_producer(seed)

    with pytest.raises(ValueError, match="producer refused input"):
        for seed in [1, 2]:
            collection.add_instance(*failing_producer(seed), registry="simulated")

    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    coordinates = sweep_coordinates(
        collection,
        sweep,
        algorithm_registries=("baseline",),
        instance_registries=("simulated",),
    )
    assert len(coordinates) == 1


def test_invalid_instance_references_do_not_create_a_registry_file(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    """Input validation fails before an invalid direct Instance is selected."""
    with pytest.raises(ValueError, match="reserved key"):
        persist_instance(
            SimpleMetadata(n_samples=2, dim=1),
            SimpleInput(X=np.array([[1.0], [3.0]])),
            {"input": [2.0]},
            tmp_path,
        )

    assert not registry_document_path(
        local_task_root(collection), "instance", "baseline"
    ).exists()


def test_registration_refuses_a_symlinked_registry_directory(
    tmp_path: Path, collection: TaskCollection
) -> None:
    """Authoring never follows an installed registry-directory indirection."""
    target = tmp_path / "outside"
    target.mkdir()
    directory = registry_document_path(
        local_task_root(collection), "algorithm", "baseline"
    ).parent
    directory.parent.mkdir(parents=True)
    directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must be a non-symlink directory"):
        collection.add_algorithm(SimpleAlgorithm(), registry="baseline")


def test_instance_registration_refuses_a_symlinked_registry_file(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    """Authoring never reads or replaces an Instance registry through a symlink."""
    target = tmp_path / "outside.json"
    target.write_text("{}")
    path = registry_document_path(local_task_root(collection), "instance", "baseline")
    path.parent.mkdir(parents=True)
    path.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        collection.add_instance(
            *_two_points(),
            registry="baseline",
        )


def test_copying_identical_entries_to_another_group_reuses_the_result(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    """Group organization is outside the identity that names a completed cell."""
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )
    collection.run_sweep(
        sweep,
        algorithm_registries=("baseline",),
        instance_registries=("baseline",),
    )

    collection.add_algorithm(SimpleAlgorithm(), registry="copied")
    collection.add_instance(
        *_two_points(),
        registry="copied",
    )
    preview = preview_sweep(
        sweep,
        workspace_local_root(tmp_path),
        SIMPLE_TASK,
        algorithm_registries=("copied",),
        instance_registries=("copied",),
    )

    assert list(preview["action"]) == ["reusable"]


def test_preview_names_each_algorithm_by_its_label_or_configured_class(
    tmp_path: Path,
    collection: TaskCollection,
) -> None:
    """A label names its entry; an unlabeled one shows its class and variants."""
    sweep = _sweep(collection)
    collection.add_algorithm(
        SimpleAlgorithm(scale=1.0, label="tuned"),
        registry="baseline",
    )
    collection.add_algorithm(SimpleAlgorithm(scale=2.0), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )

    preview = preview_sweep(
        sweep,
        workspace_local_root(tmp_path),
        SIMPLE_TASK,
        algorithm_registries=("baseline",),
        instance_registries=("baseline",),
    )

    assert set(preview["algorithm"]) == {"tuned", "SimpleAlgorithm[scale=2.0]"}


def test_result_loading_and_measurement_honor_the_same_selection(
    collection: TaskCollection,
) -> None:
    """Every result consumer resolves the selected named context, not all groups."""
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(scale=1.0), registry="baseline")
    collection.add_algorithm(SimpleAlgorithm(scale=2.0), registry="bounded")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )
    collection.run_sweep(
        sweep,
        algorithm_registries=("baseline",),
        instance_registries=("baseline",),
    )

    def output_mean(output: SimpleOutput) -> float:
        return float(output.result[0])

    metric = Metric(task=SIMPLE_TASK, name="output_mean", calculation=output_mean)
    selected_results = collection.load_results(
        algorithm_registries=("baseline",),
        instance_registries=("baseline",),
    )
    measurements = measure_results(
        selected_results,
        metric,
    )

    assert len(selected_results) == 1
    assert list(measurements[metric.name]) == [2.0]


def test_exact_selection_ignores_an_unselected_bad_group(
    collection: TaskCollection,
) -> None:
    """Explicit groups are a narrow request; the all-groups form audits everything."""
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )
    bad_path = registry_document_path(
        local_task_root(collection), "algorithm", "broken"
    )
    bad_path.write_text("not a registry file")

    selected = sweep_coordinates(
        collection,
        sweep,
        algorithm_registries=("baseline",),
        instance_registries=("baseline",),
    )

    assert len(selected) == 1
    with pytest.raises(ValueError):
        sweep_coordinates(
            collection,
            sweep,
        )


@pytest.mark.parametrize(
    "selection",
    [(), ("baseline", "baseline"), ("missing",), ("bad/name",)],
)
def test_registry_selection_fails_closed(
    collection: TaskCollection, selection: tuple[str, ...]
) -> None:
    """Selections cannot silently widen, duplicate, or interpret paths."""
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )

    with pytest.raises(ValueError):
        sweep_coordinates(
            collection,
            sweep,
            algorithm_registries=selection,
        )


def test_registry_selection_accepts_a_single_name(collection: TaskCollection) -> None:
    """One direct name selects one registry, not its individual characters."""
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )

    coordinates = sweep_coordinates(
        collection,
        sweep,
        algorithm_registries="baseline",
        instance_registries="baseline",
    )
    assert coordinates == sweep_coordinates(
        collection,
        sweep,
        algorithm_registries=["baseline"],
        instance_registries=["baseline"],
    )


def test_load_instances_on_a_fresh_collection_names_the_expected_layout(
    collection: TaskCollection,
) -> None:
    """No installed registries at all refuses by naming the tree, Task, and layout."""
    with pytest.raises(ValueError, match="instance_registries") as excinfo:
        collection.load_instances()

    message = str(excinfo.value)
    assert SIMPLE_TASK.id in message
    assert str(local_task_root(collection) / "instance_registries") in message


def test_resolution_skips_an_algorithm_entry_native_to_another_task(
    collection: TaskCollection,
) -> None:
    """A registry names its Task by id; only a rebuilt value knows which Task.

    An entry whose configured class belongs elsewhere is well formed, so it is
    skipped by the reconstruction this invocation's task id drives.
    """
    sweep = _sweep(collection)
    collection.add_algorithm(SimpleAlgorithm(), registry="baseline")
    collection.add_instance(
        *_two_points(),
        registry="baseline",
    )
    foreign = SampleMean()
    path = registry_document_path(local_task_root(collection), "algorithm", "baseline")
    document = json.loads(path.read_text())
    document["entries"][foreign.uuid] = foreign.dump_config().model_dump()
    path.write_text(json.dumps(document))

    with pytest.warns(UserWarning, match="native to task 'mean_estimation'") as skipped:
        coordinates = sweep_coordinates(
            collection,
            sweep,
            algorithm_registries=("baseline",),
            instance_registries=("baseline",),
            strict=False,
        )

    assert foreign.uuid in str(skipped[0].message)
    assert {cell.algorithm.uuid for cell in coordinates} == {SimpleAlgorithm().uuid}


def test_installed_registry_names_is_empty_without_the_directory_and_sorted_with(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "local" / "t", tmp_path / "release" / "t"
    assert installed_registry_names((first, second), "instance") == ()
    for root, name in ((first, "b"), (second, "a"), (second, "b")):
        registry_document_path(root, "instance", name).parent.mkdir(
            parents=True, exist_ok=True
        )
        registry_document_path(root, "instance", name).write_text("{}")
    assert installed_registry_names((first, second), "instance") == ("a", "b")
    assert installed_registry_names((first, second), "algorithm") == ()
