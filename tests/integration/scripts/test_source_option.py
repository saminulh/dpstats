"""``--results-source`` on get-sweep-status and measure-sweep."""

from __future__ import annotations

import json
import shutil
import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from dpstats.algorithm import register_algorithm
from dpstats.budgets import ApproxDPBudget
from dpstats.metric import Metric
from dpstats.task import Task
from dpstats.scripts.get_sweep_status import main as get_sweep_status_main
from dpstats.scripts.measure_sweep import main as measure_sweep_main
from dpstats.scripts.get_sweep_coordinates import (
    parser as coordinates_parser,
)
from dpstats.scripts.get_sweep_status import parser as status_parser
from dpstats.scripts.measure_sweep import parser as measure_parser
from dpstats.scripts.reconcile_sweep import parser as reconcile_parser
from dpstats.scripts.run_sweep import parser as sweep_parser
from dpstats.scripts.utils import resolve_task_definition
from dpstats.tasks import MeanEstimation
from dpstats.transport.configuration import load_object_from_path
from dpstats.workspace import init_workspace, open_collection
from dpstats.workspace.layout import (
    RELEASE_MANIFEST_NAME,
    task_collection_directory,
    workspace_local_root,
    workspace_release_root,
)
from tests.attempts import current_result, write_result_partition
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
    simple_instance,
)
from tests.sweep_helpers import cli_args

# Referenced by its qualified path (--metric tests...:_METRIC) so
# measure-sweep can measure SIMPLE_TASK, which has no built-in Metrics.
_METRIC = Metric(task=SIMPLE_TASK, name="value", calculation=lambda output: 0.0)
_BUDGET = ApproxDPBudget(eps=1.0, delta=1e-5)


class SameIdCustomMetadata(SimpleMetadata):
    """Metadata isolated from the built-in-id collision fixture's original Task."""


SAME_ID_CUSTOM_TASK = Task(
    id=MeanEstimation.id,
    name="Custom task sharing a built-in id",
    description="A custom Task used to guard CLI identity binding.",
    metadata_cls=SameIdCustomMetadata,
    input_cls=SimpleInput,
    output_cls=SimpleOutput,
)


class SameIdCustomAlgorithm(SimpleAlgorithm):
    """A test Algorithm bound to the custom Task, not MeanEstimation."""

    TASK = SAME_ID_CUSTOM_TASK


_SAME_ID_CUSTOM_METRIC = Metric(
    task=SAME_ID_CUSTOM_TASK,
    name="value",
    calculation=lambda output: 0.0,
)
_SAME_ID_TASK_PATH = "tests.integration.scripts.test_source_option:SAME_ID_CUSTOM_TASK"
_SAME_ID_ALGORITHM_PATH = (
    "tests.integration.scripts.test_source_option:SameIdCustomAlgorithm"
)
_SAME_ID_METADATA_PATH = (
    "tests.integration.scripts.test_source_option:SameIdCustomMetadata"
)


@pytest.mark.parametrize(
    "parser, required",
    [
        (coordinates_parser, ()),
        (status_parser, ()),
        (measure_parser, ("--metric", "metric")),
        (reconcile_parser, ()),
        (sweep_parser, ()),
    ],
)
def test_sweep_commands_accept_a_qualified_task_definition(
    parser: Callable[[], argparse.ArgumentParser], required: tuple[str, ...]
) -> None:
    """Every sweep command accepts the same explicit custom Task spelling."""
    parsed = parser().parse_args(
        [
            "--sweep",
            "current.json",
            "--workspace",
            "workspace",
            "--task-definition",
            "tests.registry_components:SIMPLE_TASK",
            *required,
        ]
    )
    assert parsed.task_definition is SIMPLE_TASK


def test_qualified_task_definition_requires_a_task_value() -> None:
    """The shared custom definition reader rejects other qualified objects."""
    assert (
        resolve_task_definition("tests.registry_components:SIMPLE_TASK") is SIMPLE_TASK
    )
    with pytest.raises(
        argparse.ArgumentTypeError, match="does not resolve to a Task value"
    ):
        resolve_task_definition("tests.registry_components:SimpleAlgorithm")
    with pytest.raises(SystemExit):
        measure_parser().parse_args(
            [
                "--sweep",
                "current.json",
                "--workspace",
                "workspace",
                "--metric",
                "value",
                "--task-definition",
                "tests.registry_components:SimpleAlgorithm",
            ]
        )


def _relabelled_cli_workspace(tmp_path: Path) -> tuple[Path, Path, str]:
    """A fetched workspace whose local Roster relabels a released Algorithm.

    The same configured Algorithm (same UUID) is installed under the same
    registry name in both trees, with a local label and a released ``null``
    one. Returns the workspace, its saved sweep path, and the shared
    Algorithm's UUID.
    """
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    algorithm = SimpleAlgorithm(label="TailRobustSparseHuberIHT")
    register_algorithm(local_root, algorithm, registry="roster")

    bundle = tmp_path / "bundle-source"
    bundle.mkdir()
    (bundle / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": "v-fixture",
                "manifest_id": "manifest-fixture",
                "tasks": [SIMPLE_TASK.id],
                "files": {},
            }
        )
    )
    register_algorithm(bundle, SimpleAlgorithm(), registry="roster")
    shutil.copytree(bundle, workspace_release_root(workspace))

    collection = open_collection(workspace, SIMPLE_TASK)
    collection.add_instance(*simple_instance(0))
    sweep_path = collection.save_sweep("current", {"budgets": [_BUDGET], "seeds": [0]})
    return workspace, sweep_path, algorithm.uuid


def _plant_a_release_only_result(
    workspace: Path, *, instance_uuid: str, uuid: str
) -> None:
    """Record a Result in the release tree only, for the same coordinate."""
    release_task_store = task_collection_directory(
        workspace_release_root(workspace), SIMPLE_TASK.id
    )
    write_result_partition(
        release_task_store,
        [
            current_result(
                instance_uuid=instance_uuid,
                alg_uuid=uuid,
                budget=_BUDGET.dump_config().model_dump(),
            )
        ],
    )


def _same_id_custom_task_sweep(tmp_path: Path) -> tuple[Path, Path]:
    """Create one answered Sweep owned by a custom Task sharing a built-in id."""
    task = resolve_task_definition(_SAME_ID_TASK_PATH)
    algorithm_type = cast(
        type[SimpleAlgorithm], load_object_from_path(_SAME_ID_ALGORITHM_PATH)
    )
    metadata_type = cast(
        type[SimpleMetadata], load_object_from_path(_SAME_ID_METADATA_PATH)
    )
    workspace = init_workspace(tmp_path / "ws")
    collection = open_collection(workspace, task)
    collection.add_algorithm(algorithm_type())
    metadata, input, references = simple_instance(0)
    collection.add_instance(
        metadata_type(n_samples=metadata.n_samples, dim=metadata.dim),
        input,
        references,
    )
    sweep_path = collection.save_sweep("current", {"budgets": [_BUDGET], "seeds": [0]})
    collection.run_sweep(sweep_path)
    return workspace, sweep_path


def test_get_sweep_status_cli_results_source_local_counts_only_local_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--results-source local`` counts only local Results.

    Registry resolution always reads both trees (the relabelled Roster entry
    merges regardless); only which tree's recorded Results count as coverage
    is scoped by ``--results-source``.
    """
    workspace, sweep_path, uuid = _relabelled_cli_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    (instance,) = collection.load_instances().values()
    _plant_a_release_only_result(workspace, instance_uuid=instance.uuid, uuid=uuid)

    get_sweep_status_main(
        cli_args(
            sweep_path,
            workspace,
            "--algorithm-registry",
            "roster",
            "--results-source",
            "local",
        )
    )
    local_summary = json.loads(capsys.readouterr().out)

    get_sweep_status_main(
        cli_args(
            sweep_path,
            workspace,
            "--algorithm-registry",
            "roster",
            "--results-source",
            "both",
        )
    )
    both_summary = json.loads(capsys.readouterr().out)

    # The relabelled entry merges into one Algorithm under either source.
    assert local_summary["algorithms"]["resolved"] == 1
    assert both_summary["algorithms"]["resolved"] == 1
    assert local_summary["cells"]["recorded"] == 0
    assert both_summary["cells"]["recorded"] == 1


def test_measure_sweep_cli_results_source_local_counts_only_local_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--results-source local`` on measure-sweep measures only local Results."""
    workspace, sweep_path, uuid = _relabelled_cli_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    (instance,) = collection.load_instances().values()
    _plant_a_release_only_result(workspace, instance_uuid=instance.uuid, uuid=uuid)
    metric_path = "tests.integration.scripts.test_source_option:_METRIC"

    measure_sweep_main(
        cli_args(
            sweep_path,
            workspace,
            "--metric",
            metric_path,
            "--algorithm-registry",
            "roster",
            "--results-source",
            "local",
        )
    )
    local_lines = capsys.readouterr().out.strip().splitlines()

    measure_sweep_main(
        cli_args(
            sweep_path,
            workspace,
            "--metric",
            metric_path,
            "--algorithm-registry",
            "roster",
            "--results-source",
            "both",
        )
    )
    both_lines = capsys.readouterr().out.strip().splitlines()

    # pandas prints an empty frame as "Empty DataFrame" plus its columns and
    # index, with no local Result recorded; source="both" prints a header
    # plus the one release-answered row.
    assert local_lines[0] == "Empty DataFrame"
    assert len(both_lines) == 2


def test_measure_sweep_rejects_builtin_metric_for_same_id_custom_task(
    tmp_path: Path,
) -> None:
    """A built-in Metric name never substitutes the collection's custom Task."""
    workspace, sweep_path = _same_id_custom_task_sweep(tmp_path)

    with pytest.raises(ValueError, match="has no built-in Metrics"):
        measure_sweep_main(
            [
                "--sweep",
                str(sweep_path),
                "--metric",
                "l2_error_to_ground_truth",
                "--workspace",
                str(workspace),
                "--task-definition",
                _SAME_ID_TASK_PATH,
            ]
        )


def test_measure_sweep_accepts_qualified_metric_for_same_id_custom_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A custom Task sharing an id still measures its own qualified Metric."""
    workspace, sweep_path = _same_id_custom_task_sweep(tmp_path)

    measure_sweep_main(
        [
            "--sweep",
            str(sweep_path),
            "--metric",
            "tests.integration.scripts.test_source_option:_SAME_ID_CUSTOM_METRIC",
            "--workspace",
            str(workspace),
            "--task-definition",
            _SAME_ID_TASK_PATH,
        ]
    )

    assert len(capsys.readouterr().out.strip().splitlines()) == 2


@pytest.mark.parametrize(
    "main",
    [get_sweep_status_main, measure_sweep_main],
)
def test_cli_results_source_rejects_an_unknown_value(
    main: Callable[[Sequence[str] | None], None],
) -> None:
    with pytest.raises(SystemExit):
        main(["--sweep", "x.json", "--workspace", "x", "--results-source", "bogus"])
