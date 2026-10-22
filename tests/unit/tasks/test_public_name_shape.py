"""Generic public-name invariants every built-in Task package must satisfy.

The convention itself (what a good name is) is a review concern; these tests
pin only what a tool can check: export symbol, class name and registry key
agree, names are well formed and unique within a Task, a Metric's name is its
exported symbol, a Task id is the snake_case package name, and every built-in
Algorithm and Transformation round-trips through its portable configuration path.
"""

from __future__ import annotations

import re
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import pytest

from dpstats.algorithm import BaseAlgorithm
from dpstats.metric import Metric
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metrics,
)

CLASS_NAME = re.compile(r"^[A-Z][A-Za-z0-9]*$")
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
TASK_IDS = sorted(BUILT_IN_TASKS)


def _package(task_id: str) -> str:
    return f"dpstats.tasks.{task_id}"


def _exported_transformations(task_id: str) -> dict[str, Any]:
    module_name = f"{_package(task_id)}.transformations"
    if find_spec(module_name) is None:
        return {}
    module = import_module(module_name)
    return {name: getattr(module, name) for name in module.__all__}


def _all_components() -> list[tuple[str, str, type]]:
    found: list[tuple[str, str, type]] = []
    for task_id in TASK_IDS:
        task = BUILT_IN_TASKS[task_id]
        for name, cls in list_built_in_algorithms(task).items():
            found.append((task_id, name, cls))
        for name, cls in _exported_transformations(task_id).items():
            found.append((task_id, name, cls))
    return found


COMPONENTS = _all_components()
COMPONENT_IDS = [f"{task_id}.{name}" for task_id, name, _ in COMPONENTS]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_task_id_is_its_snake_case_package_name(task_id: str) -> None:
    """A Task id is snake_case and equals the package that declares it."""
    task = BUILT_IN_TASKS[task_id]

    assert task.id == task_id
    assert SNAKE_CASE.match(task.id)
    assert import_module(_package(task_id)).__name__.rsplit(".", 1)[1] == task.id


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_algorithm_export_class_name_and_registry_key_agree(task_id: str) -> None:
    """Each Algorithm's export symbol, class name and catalog key are one string."""
    module = import_module(f"{_package(task_id)}.algorithms")
    catalog = list_built_in_algorithms(BUILT_IN_TASKS[task_id])

    assert set(catalog) == set(module.__all__)
    for symbol in module.__all__:
        cls = getattr(module, symbol)
        assert cls.__name__ == symbol
        assert catalog[symbol] is cls
        assert CLASS_NAME.match(symbol), symbol


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_transformation_export_symbol_equals_class_name(task_id: str) -> None:
    """Each exported Transformation's symbol is its class name."""
    for symbol, cls in _exported_transformations(task_id).items():
        assert cls.__name__ == symbol
        assert CLASS_NAME.match(symbol), symbol


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_names_are_unique_within_a_task(task_id: str) -> None:
    """No Algorithm, Transformation or Metric name repeats within one Task."""
    algorithms = list(list_built_in_algorithms(BUILT_IN_TASKS[task_id]))
    transformations = list(_exported_transformations(task_id))
    metrics = list(list_built_in_metrics(BUILT_IN_TASKS[task_id]))

    components = algorithms + transformations
    assert len(components) == len(set(components)), "Algorithm/Transformation clash"
    assert len(metrics) == len(set(metrics))


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_metric_name_is_its_exported_symbol_in_snake_case(task_id: str) -> None:
    """A Metric's name is snake_case and is the symbol it is exported under."""
    task = BUILT_IN_TASKS[task_id]
    metrics = list_built_in_metrics(task)
    if not metrics:
        return
    module = import_module(f"{_package(task_id)}._metrics")

    for name, metric in metrics.items():
        assert metric.name == name
        assert SNAKE_CASE.match(name), name
    # A Metric bound to a module-level symbol is bound under its own name; one
    # declared inline in ``METRICS`` has no symbol to disagree.
    for symbol, value in vars(module).items():
        if isinstance(value, Metric) and value in module.METRICS:
            assert symbol == value.name


@pytest.mark.parametrize(("task_id", "name", "cls"), COMPONENTS, ids=COMPONENT_IDS)
def test_configuration_path_round_trips_for_every_built_in(
    task_id: str, name: str, cls: type
) -> None:
    """A built-in's portable class path resolves back to the class."""
    module_name, _, qualname = f"{cls.__module__}:{cls.__qualname__}".partition(":")
    resolved: Any = import_module(module_name)
    for part in qualname.split("."):
        resolved = getattr(resolved, part)

    assert resolved is cls
    assert qualname == name


def test_default_configured_algorithms_round_trip_their_configuration() -> None:
    """A default-constructible Algorithm reloads equal from its configuration."""
    round_tripped = 0
    for _task_id, _name, cls in COMPONENTS:
        if not issubclass(cls, BaseAlgorithm):
            continue
        required = [
            field for field, info in cls.model_fields.items() if info.is_required()
        ]
        if required:
            continue
        configured = cls()
        restored = BaseAlgorithm.load_record(configured.dump_config())
        assert restored == configured
        assert restored.uuid == configured.uuid
        round_tripped += 1

    assert round_tripped > 20
