from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib import import_module

import pytest

from dpstats.task import Task, TaskInput, TaskMetadata, TaskOutput
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metric_arguments,
    list_built_in_metrics,
)


class _CustomMetadata(TaskMetadata):
    def validate_input(self, input: TaskInput) -> None:
        _CustomInput.model_validate(input)


class _CustomInput(TaskInput):
    pass


class _CustomOutput(TaskOutput):
    pass


@pytest.mark.parametrize(
    "first_import",
    [
        "dpstats.tasks",
        "dpstats.tasks.single_coefficient_estimation",
        "dpstats.tasks.linear_regression._metrics",
    ],
)
def test_built_in_tasks_are_immutable_without_loading_implementations(
    first_import: str,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from importlib import import_module

        import numpy as np

        import_module(sys.argv[1])
        from dpstats.tasks import (
            BUILT_IN_TASKS, list_built_in_metrics, list_built_in_metric_arguments
        )

        def refuse_fit(*args, **kwargs):
            raise AssertionError("Discovery attempted a fit")

        np.linalg.lstsq = refuse_fit

        assert set(BUILT_IN_TASKS) == {
            "covariance_estimation",
            "linear_regression",
            "logistic_regression",
            "m_estimation",
            "mean_estimation",
            "median_estimation",
            "simple_linear_regression",
            "single_coefficient_estimation",
        }
        try:
            BUILT_IN_TASKS["new"] = BUILT_IN_TASKS["mean_estimation"]
        except TypeError:
            pass
        else:
            raise AssertionError("The built-in Task catalog is mutable")

        for task in BUILT_IN_TASKS.values():
            package = import_module(f"dpstats.tasks.{task.id}")
            name = task.Metadata.__name__.removesuffix("Metadata")
            assert getattr(package, name) is task
            assert task.Metadata.task is task
            assert all(metric.task is task for metric in list_built_in_metrics(task).values())
            assert set(list_built_in_metric_arguments(task)) >= {
                name for metric in list_built_in_metrics(task).values()
                for name in metric.required_names if name not in {"input", "outputs"}
            }
        assert not any(
            name.startswith("dpstats.tasks.")
            and ".algorithms" in name
            for name in sys.modules
        )
        assert (
            "dpstats.simulations.mean_estimation"
            not in sys.modules
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, first_import],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_algorithm_catalog_is_task_scoped_and_independent() -> None:
    task = BUILT_IN_TASKS["linear_regression"]

    algorithms = list_built_in_algorithms(task)

    assert algorithms["OLS"].TASK is task
    assert algorithms["OLS"]().task is task
    assert all(algorithm.TASK is task for algorithm in algorithms.values())
    algorithms["replacement"] = algorithms["OLS"]
    assert "replacement" not in list_built_in_algorithms(task)


@pytest.mark.parametrize("task_id", sorted(BUILT_IN_TASKS))
def test_algorithm_registry_and_reference_surface_agree(task_id: str) -> None:
    """``ALGORITHMS`` is the authority; ``__all__`` restates it in the same order.

    The catalog, and every table rendered from it, follows the registry's order.
    """
    module = import_module(f"dpstats.tasks.{task_id}.algorithms")

    assert [cls.__name__ for cls in module.ALGORITHMS] == list(module.__all__)


@pytest.mark.parametrize("task_id", sorted(BUILT_IN_TASKS))
def test_registries_hold_uniquely_named_entries_of_their_own_task(task_id: str) -> None:
    task = BUILT_IN_TASKS[task_id]
    algorithms = import_module(f"dpstats.tasks.{task_id}.algorithms")

    assert len(list_built_in_algorithms(task)) == len(algorithms.ALGORITHMS)
    assert all(cls.TASK is task for cls in algorithms.ALGORITHMS)
    try:
        metrics = import_module(f"dpstats.tasks.{task_id}._metrics")
    except ModuleNotFoundError:
        return
    assert len(list_built_in_metrics(task)) == len(metrics.METRICS)
    assert all(metric.task is task for metric in metrics.METRICS)


def test_metric_catalog_is_task_scoped_independent_and_allows_no_metrics() -> None:
    metrics = list_built_in_metrics(BUILT_IN_TASKS["linear_regression"])

    assert metrics["residual_rmse"].task is BUILT_IN_TASKS["linear_regression"]
    metrics["replacement"] = metrics["residual_rmse"]
    assert "replacement" not in list_built_in_metrics(
        BUILT_IN_TASKS["linear_regression"]
    )
    assert list_built_in_metrics(BUILT_IN_TASKS["m_estimation"]) == {}


def test_built_in_catalog_rejects_a_custom_task() -> None:
    task = Task(
        id="custom",
        name="Custom",
        description="A custom Task.",
        metadata_cls=_CustomMetadata,
        input_cls=_CustomInput,
        output_cls=_CustomOutput,
    )

    with pytest.raises(ValueError, match="is not a built-in Task"):
        list_built_in_algorithms(task)
    with pytest.raises(ValueError, match="is not a built-in Task"):
        list_built_in_metrics(task)
