"""Persisted covariance estimates retain separate metric arguments."""

from functools import partial
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from dpstats.analysis import measure_results
from dpstats.budgets import NO_PRIVACY
from dpstats.domains import EuclideanSpace
from dpstats.instance import Instance
from dpstats.simulations.covariance_estimation import simulate_gaussian
from dpstats.tasks import CovarianceEstimation, list_built_in_metrics
from dpstats.tasks.covariance_estimation.algorithms import SampleCovariance
from dpstats.workspace import init_workspace, open_collection
from tests.attempts import read_attempt_rows
from tests.sweep_helpers import local_task_root, write_sweep


def test_gaussian_instance_round_trip_run_and_four_metrics(tmp_path: Path) -> None:
    produce = partial(
        simulate_gaussian,
        mean=[10.0, -20.0],
        covariance=[[2.0, 0.0], [0.0, 1.0]],
        n_samples=1,
    )
    metadata, input_data, references = produce(seed=7)
    assert isinstance(metadata.domain, EuclideanSpace)
    np.testing.assert_allclose(references["ground_truth"], [[2.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(references["sample_covariance"], np.zeros((2, 2)))
    assert set(type(metadata).model_fields) == {"n_samples", "domain"}

    collection = open_collection(init_workspace(tmp_path), CovarianceEstimation)
    instance = collection.add_instance(metadata, input_data, references)
    restored = Instance.load_config(instance.record())
    assert restored == instance
    loaded = collection.load_input(restored)
    np.testing.assert_allclose(loaded.X, input_data.X)

    sweep = write_sweep(
        collection, [SampleCovariance()], [produce], [NO_PRIVACY], instance_seeds=(7,)
    )
    collection.run_sweep(sweep)
    results = collection.load_results()
    assert list(results["status"]) == ["success"]
    np.testing.assert_allclose(
        results.iloc[0]["payload"].covariance_est, np.zeros((2, 2))
    )

    metrics = list_built_in_metrics(CovarianceEstimation)
    expected = {
        "frobenius_error_to_ground_truth": sqrt(5),
        "spectral_error_to_ground_truth": 2.0,
        "frobenius_error_to_sample_covariance": 0.0,
        "spectral_error_to_sample_covariance": 0.0,
    }
    for name, value in expected.items():
        measured = measure_results(results, metrics[name])
        assert measured[name].tolist() == pytest.approx([value])

    (attempt,) = read_attempt_rows(local_task_root(collection))
    assert "ground_truth" not in attempt
    assert "sample_covariance" not in attempt
