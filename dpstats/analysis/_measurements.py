"""Resolve the arguments Metrics read and measure retained Results against them."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pydantic import JsonValue

# The module, not its names, so a test may observe production at this seam.
import dpstats.metric as declarations
from dpstats.instance import Instance
from dpstats.results._frame import RESULT_COLUMNS
from dpstats.results import ResultStatus
from dpstats.metric import (
    INPUT,
    OUTPUTS,
    RESERVED_REQUIREMENTS,
    Metric,
    MetricArgument,
    declared_arguments,
)
from dpstats.task import Task, TaskInput, TaskOutput
from dpstats.tasks import BUILT_IN_TASKS, list_built_in_metric_arguments
from dpstats.workspace.collection import TaskCollection


def stored_requirements(metric: Metric) -> frozenset[str]:
    """Name the requirements a Metric reads from resolved arguments."""
    return frozenset(metric.required_names) - RESERVED_REQUIREMENTS


class ArgumentResolver:
    """The stored and produced arguments a selection of Metrics reads.

    The declared arguments are those the selected Metrics and ``arguments``
    declare and, for a built-in Task, its whole catalogue: a value an
    Instance stores under any name the library produces is never read,
    whatever is selected, because the producer is the authority, and a
    selected Metric or argument may not declare a library name differently.
    A name is *source* when the Instance stores it and no declaration
    produces it, *produced* when a declared producer computes it from the
    Input, and *missing* otherwise.
    """

    def __init__(
        self, metrics: Iterable[Metric], arguments: Iterable[MetricArgument] = ()
    ) -> None:
        self.metrics = tuple(metrics)
        catalogue = [
            argument
            for task in {metric.task for metric in self.metrics}
            if BUILT_IN_TASKS.get(task.id) is task
            for argument in list_built_in_metric_arguments(task).values()
        ]
        self.declared = declared_arguments(self.metrics, (*catalogue, *arguments))
        self.produced = frozenset(
            name
            for name, argument in self.declared.items()
            if argument.producer is not None
        )

    def requires_input(self, metric: Metric) -> bool:
        """Whether this Metric cannot be measured without the Instance's Input."""
        return INPUT in metric.required_names or bool(
            stored_requirements(metric) & self.produced
        )

    def source_arguments(self, instance: Instance) -> dict[str, JsonValue]:
        """Keep the Instance's stored values other than produced names."""
        return {
            name: value
            for name, value in instance.metric_arguments.items()
            if name not in self.produced
        }

    def classify(
        self, instance: Instance, name: str
    ) -> Literal["source", "produced", "missing"]:
        """Say where one name comes from for this Instance."""
        if name in self.produced:
            return "produced"
        if name in instance.metric_arguments:
            return "source"
        return "missing"

    def missing(self, instance: Instance, names: Iterable[str]) -> set[str]:
        """Name those of ``names`` this Instance can supply from nowhere."""
        return {name for name in names if self.classify(instance, name) == "missing"}

    def resolve(
        self, instance: Instance, input_data: TaskInput | None
    ) -> dict[str, JsonValue]:
        """Read source values and, when the Input is supplied, produce the rest."""
        arguments = self.source_arguments(instance)
        if input_data is not None and self.produced:
            arguments.update(
                declarations.produce_arguments(
                    (self.declared[name] for name in sorted(self.produced)),
                    instance.metadata,
                    input_data,
                )
            )
        return arguments

    def dedupe(self, instances: Iterable[Instance]) -> dict[str, Instance]:
        """Index Instances by UUID once, in first-seen order.

        Raises:
            ValueError: If two Instances share a UUID but store different
                source arguments.
        """
        by_uuid: dict[str, Instance] = {}
        for instance in instances:
            previous = by_uuid.setdefault(instance.uuid, instance)
            if previous is not instance and self.source_arguments(
                previous
            ) != self.source_arguments(instance):
                raise ValueError(
                    f"Instance {instance.uuid!r} has conflicting Metric arguments."
                )
        return by_uuid


def _measure_group(
    instance: Instance,
    outputs: tuple[TaskOutput, ...],
    metric: Metric,
    input_data: TaskInput | None,
    arguments: Mapping[str, Any],
) -> list[float]:
    """Evaluate one Metric over a group's successful retained Outputs."""
    required: dict[str, Any] = {}
    for name in metric.required_names:
        if name == INPUT:
            required[name] = input_data
        elif name == OUTPUTS:
            required[name] = outputs
        elif name in arguments:
            required[name] = arguments[name]
        else:
            raise ValueError(
                f"Task {instance.task.id!r}, Metric {metric.name!r}, Instance "
                f"{instance.uuid!r} is missing required metric argument {name!r}."
            )
    values: list[float] = []
    for output in outputs:
        value = metric.calculate(output, **required)
        values.append(math.inf if math.isnan(value) else value)
    return values


def _require_distinct_columns(selected: Sequence[Metric]) -> None:
    named: set[str] = set()
    for metric in selected:
        if metric.name == "metric_arguments":
            raise ValueError(
                "Metric name 'metric_arguments' is reserved for resolved values."
            )
        if metric.name in named:
            raise ValueError(
                f"Metric name {metric.name!r} is selected more than once. The "
                "selected Metrics must have distinct column names."
            )
        named.add(metric.name)


def get_metric_availability(
    instances: Iterable[Instance], metrics: Metric | Iterable[Metric]
) -> pd.DataFrame:
    """Report which of the supplied metrics are computable for each instance.

    For each instance, a metric is computable if the instance supplies all
    of the metric's required arguments.

    Args:
        instances: The instances to check.
        metrics: One Metric or an iterable of Metrics.

    Returns:
        A DataFrame indexed by ``instance_uuid``, with one column per Metric
        named by ``metric.name``. String entries
        are ``available`` when the instance supplies all the arguments the metric needs;
        ``available_with_input`` when some argument must be computed from the raw input,
        or the metric reads the input itself; or ``unavailable`` if some argument is missing
        or the metric belongs to another task.

    Raises:
        ValueError: If selected Metrics have duplicate names, or Instances
            sharing a UUID carry conflicting Metric arguments.

    Example:
        >>> get_metric_availability([], []).index.name
        'instance_uuid'
    """
    selected = (metrics,) if isinstance(metrics, Metric) else tuple(metrics)
    _require_distinct_columns(selected)
    resolver = ArgumentResolver(selected)
    by_uuid = resolver.dedupe(instances)
    columns: dict[str, list[str]] = {}
    for metric in selected:
        stored = stored_requirements(metric)
        statuses: list[str] = []
        for instance in by_uuid.values():
            if instance.task is not metric.task or resolver.missing(instance, stored):
                status = "unavailable"
            elif resolver.requires_input(metric):
                status = "available_with_input"
            else:
                status = "available"
            statuses.append(status)
        columns[metric.name] = statuses
    return pd.DataFrame(columns, index=pd.Index(list(by_uuid), name="instance_uuid"))


def measure_results(
    results: pd.DataFrame,
    metrics: Metric | Sequence[Metric],
    *,
    collection: TaskCollection | None = None,
    arguments: Iterable[MetricArgument] = (),
) -> pd.DataFrame:
    """Measure supplied current Results without resolving or rereading a Sweep.

    Successful Results receive Metric values and every other terminal outcome
    receives positive infinity. A Metric requiring ``"outputs"`` reads only the
    successful Outputs this frame supplies for its group.

    Args:
        results: The complete Results DataFrame returned by a task collection,
            optionally carrying earlier measurements of different Metrics.
        metrics: One Metric or a sequence of task-compatible Metrics.
        collection: The collection Inputs are loaded from. When supplied,
            each Instance's Input is read once and every argument the Task
            produces, the selected Metrics' and ``arguments`` included, is
            computed from it, whatever Metrics are selected. When omitted,
            only stored values are read. Its Task must be the Results' own
            Task.
        arguments: Further declared arguments to produce alongside the
            Task's own, so a value no selected Metric names is still
            resolved and may be divided by.

    Returns:
        A copy of the original Results, in the same row/index order, with one
        float column per Metric named by metric.name and a ``metric_arguments``
        column containing resolved source-supplied and computed values. Existing
        produced values on Instances are ignored. Earlier resolved arguments
        are retained and updated with values computed in this call. All other
        original columns remain unchanged, including earlier measurements,
        Instances, payloads and caller-added columns. The supplied frame is
        never modified.

    Raises:
        ValueError: If a selected Metric belongs to another Task, if
            *collection* holds a different Task than the Results, if a
            selected Metric or argument redeclares a library name differently,
            or if *collection* is omitted and a selected Metric requires
            ``"input"`` or a produced value, or an argument has a producer.

    Example:
        >>> measurements = measure_results(results, metric)  # doctest: +SKIP
        >>> measurements[metric.name].median()  # doctest: +SKIP
    """
    selected = (metrics,) if isinstance(metrics, Metric) else tuple(metrics)
    _require_distinct_columns(selected)
    for metric in selected:
        if metric.name in results.columns:
            raise ValueError(
                f"Metric name {metric.name!r} collides with an existing Results column."
            )
    measured = results.copy()
    if not selected:
        return measured
    if results.empty:
        measured["metric_arguments"] = pd.Series(index=results.index, dtype=object)
        for metric in selected:
            measured[metric.name] = np.empty(0, dtype=float)
        return measured
    task, statuses = _parse_results_frame(results)
    for metric in selected:
        if metric.task is not task:
            raise ValueError(f"Metric {metric.name!r} belongs to another Task.")
    if collection is not None and collection.task is not task:
        raise ValueError(
            f"collection holds Task {collection.task.id!r}, but the "
            f"Results belong to Task {task.id!r}."
        )
    extra = tuple(arguments)
    resolver = ArgumentResolver(selected, extra)
    if collection is None:
        if any(resolver.requires_input(m) for m in selected):
            raise ValueError(
                "Measuring an Input-dependent Metric requires collection=."
            )
        if any(argument.producer is not None for argument in extra):
            raise ValueError("Producing an argument from Inputs requires collection=.")
    input_source: TaskCollection[Any, TaskInput, TaskOutput] | None = collection
    successful = np.flatnonzero([status is ResultStatus.SUCCESS for status in statuses])
    instances = results["instance"].to_numpy()
    algorithms = results["algorithm"].to_numpy()
    payloads = results["payload"].to_numpy()
    keys = pd.DataFrame(
        {
            "instance": [instance.uuid for instance in instances[successful]],
            "algorithm": [algorithm.uuid for algorithm in algorithms[successful]],
            "budget": results["budget"].to_numpy()[successful],
        }
    )
    by_uuid = resolver.dedupe(instances)
    # With a collection, every Instance is resolved so its produced values
    # are computed even where no trial succeeded.
    groups_by_instance: dict[str, tuple[Instance, list[NDArray[np.intp]]]] = {
        uuid: (instance, [])
        for uuid, instance in by_uuid.items()
        if input_source is not None
    }
    for positions in keys.groupby(
        ["instance", "algorithm", "budget"], sort=False
    ).indices.values():
        observed = successful[positions]
        instance: Instance = instances.item(observed[0])
        groups_by_instance.setdefault(instance.uuid, (instance, []))[1].append(observed)
    metric_values = np.full((len(selected), len(results)), math.inf)
    prior_arguments = (
        dict(zip(results["instance_uuid"], results["metric_arguments"]))
        if "metric_arguments" in results.columns
        else {}
    )
    resolved = {
        uuid: {**prior_arguments.get(uuid, {}), **resolver.source_arguments(instance)}
        for uuid, instance in by_uuid.items()
    }
    for instance, groups in groups_by_instance.values():
        # Measuring a large store holds one Instance's Input at a time: this name
        # is its only reference, released before the next Instance's Input loads.
        input_data: TaskInput | None = None
        if input_source is not None:
            input_data = input_source.load_input(instance)
        resolved_values = resolver.resolve(instance, input_data)
        resolved[instance.uuid].update(resolved_values)
        for positions in groups:
            outputs: tuple[TaskOutput, ...] = tuple(payloads[positions].tolist())
            for metric_position, metric in enumerate(selected):
                values = _measure_group(
                    instance,
                    outputs,
                    metric,
                    input_data if INPUT in metric.required_names else None,
                    resolved_values,
                )
                metric_values[metric_position, positions] = values
        del input_data
    measured["metric_arguments"] = pd.Series(
        [resolved[instance.uuid] for instance in instances],
        index=results.index,
        dtype=object,
    )
    for metric, values in zip(selected, metric_values):
        measured[metric.name] = values
    return measured


def _parse_results_frame(results: pd.DataFrame) -> tuple[Task, list[ResultStatus]]:
    """Read and validate the Results fields this measurement consumer needs."""
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Results frame is missing required column(s): {names}.")
    instances = results["instance"].to_numpy()
    first = instances[0]
    if not isinstance(first, Instance):
        raise ValueError(
            "Results frame instance values must be resolved Instance objects."
        )
    task = first.task
    statuses = [ResultStatus(value) for value in results["status"].to_numpy()]
    for instance, algorithm, task_id, status, payload in zip(
        instances,
        results["algorithm"].to_numpy(),
        results["task"].to_numpy(),
        statuses,
        results["payload"].to_numpy(),
    ):
        if not isinstance(instance, Instance) or instance.task is not task:
            raise ValueError("Results frame must contain one compatible Task.")
        if algorithm.task is not task:
            raise ValueError("Results frame algorithm belongs to another Task.")
        if task_id != task.id:
            raise ValueError("Results frame task column does not match its Instances.")
        if status is ResultStatus.SUCCESS and not isinstance(payload, task.Output):
            raise ValueError(
                "A successful Results row must retain its typed TaskOutput."
            )
    return task, statuses
