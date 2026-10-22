"""Configured private covariance estimation through a saved collection sweep."""

from pathlib import Path

import numpy as np

from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
from dpstats.analysis import measure_results
from dpstats.budgets import ApproxDPBudget
from dpstats.domains import LpBallDomain
from dpstats.tasks import (
    CovarianceEstimation,
    list_built_in_algorithms,
    list_built_in_metrics,
)
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
)
from dpstats.tasks.covariance_estimation.transformations import (
    CovarianceToMeanEstimation,
)
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
)
from dpstats.workspace import init_workspace, open_collection


def test_configured_mean_child_round_trips_and_runs_in_a_collection(tmp_path: Path):
    algorithm = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),),
        algorithm=GaussianMechanism(),
    )
    restored = BaseAlgorithm.load_record(algorithm.dump_config())
    assert restored == algorithm
    assert (
        CovarianceToMeanEstimation
        not in list_built_in_algorithms(CovarianceEstimation).values()
    )

    metadata = CovarianceEstimationMetadata(
        n_samples=3, domain=LpBallDomain.at_origin(p=2, radius=2, dim=2)
    )
    input = CovarianceEstimationInput(X=[[-1, 0], [1, 0], [0, 1]])
    # Centered sample covariance of this triangle, with divisor n.
    reference = [[2 / 3, 0], [0, 2 / 9]]
    budget = ApproxDPBudget(eps=2, delta=1e-6)
    expected = algorithm.run(metadata, input, budget, 7)

    collection = open_collection(init_workspace(tmp_path), CovarianceEstimation)
    collection.add_algorithm(restored, registry="configured")
    collection.add_instance(metadata, input, {"sample_covariance": reference})
    sweep = collection.save_sweep("covariance", {"budgets": [budget], "seeds": [7]})
    collection.run_sweep(sweep)
    results = collection.load_results()

    assert results["status"].tolist() == ["success"]
    np.testing.assert_allclose(
        results.iloc[0]["payload"].covariance_est, expected.covariance_est
    )
    metric = list_built_in_metrics(CovarianceEstimation)[
        "frobenius_error_to_sample_covariance"
    ]
    measured = measure_results(results, metric)
    assert np.isfinite(measured[metric.name].iloc[0])
