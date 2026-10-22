"""Measure the Results a sweep selects with one or more task metrics."""

from __future__ import annotations

from operator import attrgetter
import argparse
from collections.abc import Sequence
from pathlib import Path

from dpstats.workspace import TaskCollection
from dpstats.analysis import measure_results
from dpstats.transport.configuration import load_object_from_path
from dpstats.metric import Metric
from dpstats.tasks import BUILT_IN_TASKS, list_built_in_metrics
from dpstats.scripts.utils import (
    ALGORITHM_REGISTRY_HELP,
    INSTANCE_REGISTRY_HELP,
    add_task_definition_argument,
)


def parser() -> argparse.ArgumentParser:
    """Build the ``measure-sweep`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Measure the Results a sweep selects with one or more task metrics."
    )
    command.add_argument(
        "--sweep",
        required=True,
        help="Path to the authored sweep JSON file (required)",
    )
    command.add_argument(
        "--metric",
        required=True,
        action="append",
        help=(
            "Built-in metric name or qualified custom Metric (module:object); "
            "repeat to measure several in one pass (required)"
        ),
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace or bare bundle to measure Results in (path, required)",
    )
    command.add_argument(
        "--algorithm-registry",
        action="append",
        help=ALGORITHM_REGISTRY_HELP,
    )
    command.add_argument(
        "--instance-registry",
        action="append",
        help=INSTANCE_REGISTRY_HELP,
    )
    add_task_definition_argument(command)
    command.add_argument(
        "--results-source",
        choices=("local", "release", "both"),
        default="both",
        help="Which installed tree(s)' Results this measures (default: both)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public ``measure-sweep`` command."""
    parsed = parser().parse_args(argv)

    sweep_path = Path(parsed.sweep)
    collection = TaskCollection.from_sweep(
        sweep_path,
        Path(parsed.workspace),
        task=parsed.task_definition,
    )
    task = collection.task
    metrics: list[Metric] = []
    for name in parsed.metric:
        if ":" in name:
            metric = load_object_from_path(name)
            if not isinstance(metric, Metric):
                raise TypeError(f"{name!r} does not resolve to a Metric value.")
        else:
            if BUILT_IN_TASKS.get(task.id) is not task:
                raise ValueError(
                    f"Task {task.id!r} has no built-in Metrics; pass a "
                    "qualified custom Metric instead."
                )
            try:
                metric = list_built_in_metrics(task)[name]
            except KeyError as exc:
                raise ValueError(
                    f"Built-in task {task.id!r} has no Metric named {name!r}."
                ) from exc
        metrics.append(metric)
    measurements = measure_results(
        collection.load_results(
            sweep=sweep_path,
            algorithm_registries=parsed.algorithm_registry,
            instance_registries=parsed.instance_registry,
            results_source=parsed.results_source,
        ),
        metrics,
        collection=collection,
    )
    # The frame carries configured values; a terminal reads their names.
    rendered = measurements.assign(
        instance=measurements["instance"].map(attrgetter("uuid")),
        algorithm=measurements["algorithm"].map(str),
        budget=measurements["budget"].map(repr),
    )
    print(rendered.to_string(index=False))


if __name__ == "__main__":
    main()
