"""Private discovery implementation for the built-in Task facade."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any

from pydantic import JsonValue

from dpstats.algorithm import BaseAlgorithm
from dpstats.metric import (
    Metric,
    MetricArgument,
    declared_arguments,
    produce_arguments,
)
from dpstats.task import Task
from dpstats.tasks.covariance_estimation import CovarianceEstimation
from dpstats.tasks.linear_regression import LinearRegression
from dpstats.tasks.logistic_regression import LogisticRegression
from dpstats.tasks.m_estimation import MEstimation
from dpstats.tasks.mean_estimation import MeanEstimation
from dpstats.tasks.median_estimation import MedianEstimation
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegression,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimation,
)


BUILT_IN_TASKS: Mapping[str, Task] = MappingProxyType(
    {
        CovarianceEstimation.id: CovarianceEstimation,
        LinearRegression.id: LinearRegression,
        LogisticRegression.id: LogisticRegression,
        MEstimation.id: MEstimation,
        MeanEstimation.id: MeanEstimation,
        MedianEstimation.id: MedianEstimation,
        SimpleLinearRegression.id: SimpleLinearRegression,
        SingleCoefficientEstimation.id: SingleCoefficientEstimation,
    }
)


def _built_in_task_package(task: Task) -> str:
    if BUILT_IN_TASKS.get(task.id) is not task:
        raise ValueError(f"Task {task.id!r} is not a built-in Task.")
    return f"dpstats.tasks.{task.id}"


def list_built_in_algorithms(task: Task) -> dict[str, type[BaseAlgorithm]]:
    """Return a dictionary of all the built-in algorithms for the given task, keyed by name.

    Raises:
        ValueError: If ``task`` is not a built-in Task.
    """
    module = import_module(f"{_built_in_task_package(task)}.algorithms")
    return {cls.__name__: cls for cls in module.ALGORITHMS}


def list_built_in_metrics(task: Task) -> dict[str, Metric]:
    """Return a dictionary of all the built-in metrics for the given task, keyed by name.

    Raises:
        ValueError: If ``task`` is not a built-in Task.
    """
    module = _metrics_module(task)
    if module is None:
        return {}
    return {metric.name: metric for metric in module.METRICS}


def _metrics_module(task: Task) -> ModuleType | None:
    module_name = f"{_built_in_task_package(task)}._metrics"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def list_built_in_metric_arguments(task: Task) -> dict[str, MetricArgument]:
    """Return a dictionary of all the arguments required by the built-in metrics for the given task, keyed by name.

    Raises:
        ValueError: If ``task`` is not a built-in Task.
    """
    module = _metrics_module(task)
    if module is None:
        return {}
    return declared_arguments(module.METRICS, module.ARGUMENTS)


def compute_metric_arguments(
    task: Task, metadata: Any, input: Any
) -> dict[str, JsonValue]:
    """Compute every metric argument for the instance of a given Task.

    Args:
        task: A built-in Task.
        metadata: The Instance's Metadata, an instance of ``task.Metadata``.
        input: The Instance's Input, valid for ``metadata``.

    Returns:
        The JSON-ready value of each declared argument with a producer,
        keyed by name. Metric arguments the source supplies (those with no
        producer) are absent.

    Raises:
        TypeError: If ``metadata`` is not a ``task.Metadata`` or ``input``
            is not a ``task.Input``.
        ValueError: If ``input`` is not valid for ``metadata``, or ``task``
            is not a built-in Task.
    """
    if not isinstance(metadata, task.Metadata):
        raise TypeError(f"metadata must be a {task.Metadata.__name__}.")
    metadata.validate_input(input)
    return produce_arguments(
        list_built_in_metric_arguments(task).values(), metadata, input
    )
