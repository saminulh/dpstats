"""Public namespace, collection, and measurement contracts."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

import dpstats as dp
from dpstats.algorithm import CompositeAlgorithm
from dpstats.instance import Instance
from dpstats.simulations.linear_regression import simulate_clipped_gaussian
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metric_arguments,
    list_built_in_metrics,
    linear_regression as lr,
    mean_estimation as me,
)


@pytest.mark.parametrize("task_id", sorted(BUILT_IN_TASKS))
def test_namespaces_expose_the_catalogued_objects(task_id: str) -> None:
    task = BUILT_IN_TASKS[task_id]
    module = import_module(f"dpstats.tasks.{task_id}")
    for name, algorithm in list_built_in_algorithms(task).items():
        assert getattr(module.algorithms, name) is algorithm
    metrics = list_built_in_metrics(task)
    if metrics:
        assert set(module.metrics.__all__) == set(metrics)
        for name, metric in metrics.items():
            assert getattr(module.metrics, name) is metric
    assert "algorithms" in dir(module)
    with pytest.raises(AttributeError):
        getattr(module, "not_a_public_namespace")


def test_root_composition_export_is_the_original_class() -> None:
    assert dp.CompositeAlgorithm is CompositeAlgorithm
    direct = CompositeAlgorithm(
        transformations=(lr.transformations.Standardize(),),
        algorithm=lr.algorithms.SSP(),
    )
    exported = dp.CompositeAlgorithm(
        transformations=(lr.transformations.Standardize(),),
        algorithm=lr.algorithms.SSP(),
    )
    assert direct.uuid == exported.uuid
    assert direct.dump_config() == exported.dump_config()
    assert direct.privacy_interpretation == exported.privacy_interpretation


def test_opening_task_namespace_does_not_import_algorithms() -> None:
    code = """
import sys
from dpstats.tasks import linear_regression as lr
assert 'algorithms' in dir(lr)
assert 'metrics' in dir(lr)
assert not any(name.startswith('dpstats.tasks.') and
               '.algorithms' in name for name in sys.modules)
metric = lr.metrics.prediction_rmse_to_ols
assert metric.task is lr.LinearRegression
assert not any(name.startswith('dpstats.tasks.') and
               '.algorithms' in name for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "discover",
    [list_built_in_algorithms, list_built_in_metrics, list_built_in_metric_arguments],
)
def test_discovery_returns_independent_dicts(discover) -> None:
    before = discover(lr.LinearRegression)
    result = discover(lr.LinearRegression)
    assert type(result) is dict
    result.clear()
    assert discover(lr.LinearRegression) == before


def test_collection_loading_keeps_bare_instance_identity(tmp_path: Path) -> None:
    metadata, input_data, references = simulate_clipped_gaussian(seed=0)
    collection = dp.open_collection(dp.init_workspace(tmp_path), lr.LinearRegression)
    original = collection.add_instance(metadata, input_data, references)
    loaded = collection.load_instances()[original.uuid]
    assert type(original) is Instance
    assert type(loaded) is Instance
    assert loaded.record() == original.record()
    assert loaded.uuid == original.uuid
    assert loaded == original
    assert isinstance(loaded.metadata, lr.LinearRegressionMetadata)
    assert isinstance(collection.load_input(loaded), lr.LinearRegressionInput)
    assert Instance.load_config(loaded.record()) == loaded
    other = dp.open_collection(tmp_path, me.MeanEstimation)
    with pytest.raises(ValueError, match="another Task"):
        other.load_input(loaded)
