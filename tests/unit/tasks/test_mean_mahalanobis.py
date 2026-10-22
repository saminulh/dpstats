"""Sample-covariance Mahalanobis errors against independent small examples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dpstats import init_workspace, open_collection
from dpstats.analysis import (
    aggregate_trials,
    get_metric_availability,
    measure_results,
    compute_error_ratio,
)
from dpstats.budgets import NO_PRIVACY
from dpstats.domains import EuclideanSpace
from dpstats.metric import Metric, produce_arguments
from dpstats.task import Array2D
from dpstats.tasks import (
    MeanEstimation,
    list_built_in_metrics,
    compute_metric_arguments,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean


def _arguments(sample: Array2D) -> dict[str, Any]:
    metadata = MeanEstimationMetadata(
        n_samples=sample.shape[0], domain=EuclideanSpace(dim=sample.shape[1])
    )
    return compute_metric_arguments(
        MeanEstimation, metadata, MeanEstimationInput(X=sample)
    )


def _error(estimate: list[float], arguments: dict[str, Any]) -> float:
    metric = list_built_in_metrics(MeanEstimation)["mahalanobis_error_to_sample_mean"]
    return metric.calculate(
        MeanEstimationOutput(mean_est=np.array(estimate)),
        sample_mean=arguments["sample_mean"],
        sample_mahalanobis_geometry=arguments["sample_mahalanobis_geometry"],
    )


def test_full_rank_error_and_baseline_use_unbiased_sample_covariance() -> None:
    """S=diag(2/3, 8/3), so (1, 2) has squared error 3 and r/n=1/2."""
    sample = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    arguments = _arguments(sample)

    assert _error([1.0, 2.0], arguments) == pytest.approx(math.sqrt(3.0))
    assert arguments["sample_mean_mahalanobis_se"] == pytest.approx(math.sqrt(0.5))


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [([10.0, -4.0], 0.0), ([11.0, -2.0], 1.0), ([12.0, -5.0], math.inf)],
)
def test_rank_one_error_enforces_the_samples_affine_span(estimate, expected) -> None:
    """Centered rows are -u, 0, u for u=(1, 2), so S=uu' and ||u||_S=1."""
    sample = np.array([[9.0, -6.0], [10.0, -4.0], [11.0, -2.0]])
    arguments = _arguments(sample)

    assert _error(estimate, arguments) == pytest.approx(expected)
    assert arguments["sample_mean_mahalanobis_se"] == pytest.approx(math.sqrt(1 / 3))


def test_rank_zero_requires_an_exact_match_and_has_zero_baseline() -> None:
    """A constant sample exercises the rank-zero metric and baseline contract."""
    arguments = _arguments(np.tile([1.0, -2.0], (6, 1)))

    assert arguments["sample_mahalanobis_geometry"]["rank"] == 0
    assert arguments["sample_mean_mahalanobis_se"] == 0.0
    assert _error(arguments["sample_mean"], arguments) == 0.0
    shifted = list(arguments["sample_mean"])
    shifted[0] += 0.001
    assert _error(shifted, arguments) == math.inf


def test_one_hot_sample_has_rank_two_and_rejects_changing_total_mass() -> None:
    """Three observed categories vary in two directions because each row sums to one."""
    counts = np.array([1234, 2345, 5432])
    sample = np.repeat(np.eye(3), counts, axis=0)
    arguments = _arguments(sample)
    mean = np.array(arguments["sample_mean"])
    probabilities = counts / len(sample)

    assert arguments["sample_mean_mahalanobis_se"] == pytest.approx(
        math.sqrt(2 / len(sample))
    )
    assert _error((mean + 1).tolist(), arguments) == math.inf
    assert _error((mean + [1, -1, 0]).tolist(), arguments) == pytest.approx(
        math.sqrt(
            (len(sample) - 1)
            / len(sample)
            * (1 / probabilities[0] + 1 / probabilities[1])
        )
    )


@pytest.mark.parametrize(
    "sample",
    [
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        [[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]],
    ],
)
def test_error_and_baseline_are_invariant_under_affine_coordinate_changes(sample):
    sample = np.array(sample)
    estimate = np.array([1.0, 2.0])
    transform = np.array([[2.0, 1.0], [-1.0, 3.0]])
    translation = np.array([12.0, -7.0])
    original = _arguments(sample)
    transformed = _arguments(sample @ transform.T + translation)

    assert _error((transform @ estimate + translation).tolist(), transformed) == (
        pytest.approx(_error(estimate.tolist(), original))
    )
    assert transformed["sample_mean_mahalanobis_se"] == pytest.approx(
        original["sample_mean_mahalanobis_se"]
    )


def test_a_small_positive_variance_still_contributes_to_error_and_rank() -> None:
    """Rescaling the second coordinate by 1e-5 preserves rank two and error."""
    sample = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.00002], [0.0, -0.00002]])
    arguments = _arguments(sample)

    assert _error([1.0, 0.00002], arguments) == pytest.approx(math.sqrt(3.0))
    assert arguments["sample_mean_mahalanobis_se"] == pytest.approx(math.sqrt(0.5))


def test_span_membership_accepts_projection_roundoff() -> None:
    """A decimal multiple of the sole varying direction lies in its span."""
    sample = np.array([[-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    arguments = _arguments(sample)

    assert _error([0.1, 0.2, 0.3], arguments) == pytest.approx(0.1)


def test_full_rank_covariance_has_no_off_span_direction() -> None:
    """A non-diagonal positive definite covariance is finite in every direction."""
    covariance = np.array([[9.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, 5.0]])
    difference = np.array([1.0, 2.0, 3.0])
    factor = np.linalg.cholesky(covariance)
    sample = np.vstack((factor.T, -factor.T)) * math.sqrt(5 / 2)
    arguments = _arguments(sample)

    assert _error(difference.tolist(), arguments) == pytest.approx(
        math.sqrt(difference @ np.linalg.solve(covariance, difference))
    )


def test_univariate_error_is_absolute_error_in_sample_standard_deviations() -> None:
    """The sample (1, 3, 5) has mean 3 and sample standard deviation 2."""
    arguments = _arguments(np.array([[1.0], [3.0], [5.0]]))

    assert _error([4.0], arguments) == pytest.approx(0.5)
    assert arguments["sample_mean_mahalanobis_se"] == pytest.approx(math.sqrt(1 / 3))


def test_infinite_release_has_infinite_error() -> None:
    arguments = _arguments(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    assert _error([math.inf, 0.0], arguments) == math.inf


@pytest.mark.parametrize(
    "stored", [{}, {"sample_mean": [999.0, 999.0], "sample_mean_mahalanobis_se": 999.0}]
)
@pytest.mark.parametrize("separate_calls", [False, True])
def test_input_arguments_measure_and_normalize_retained_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored: dict[str, Any],
    separate_calls: bool,
) -> None:
    """Rank-one S=uu' gives ratios 0, sqrt(3), infinity for three releases."""
    collection = open_collection(init_workspace(tmp_path), task_id=MeanEstimation.id)
    sample = np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    instance = collection.add_instance(
        MeanEstimationMetadata(n_samples=3, domain=EuclideanSpace(dim=2)),
        MeanEstimationInput(X=sample),
        {**stored, "published_se": 7.0},
    )
    algorithm = SampleMean()
    collection.add_algorithm(algorithm)
    collection.run_sweep(
        algorithms=[algorithm],
        instances=[instance],
        budgets=[NO_PRIVACY],
        seeds=[0, 1, 2],
    )
    results = collection.load_results()
    for index, estimate in enumerate(([0.0, 0.0], [1.0, 2.0], [2.0, -1.0])):
        results.at[index, "payload"] = MeanEstimationOutput(mean_est=np.array(estimate))

    metric = list_built_in_metrics(MeanEstimation)["mahalanobis_error_to_sample_mean"]
    assert (
        get_metric_availability([instance], metric).iloc[0, 0] == "available_with_input"
    )
    calls = []

    def produce(arguments, metadata, input):
        calls.append(type(metadata).task)
        return produce_arguments(arguments, metadata, input)

    monkeypatch.setattr("dpstats.metric.produce_arguments", produce)
    with pytest.raises(ValueError, match="collection="):
        measure_results(results, metric)
    coordinate = Metric(
        task=MeanEstimation,
        name="first_coordinate",
        calculation=lambda output: float(output.mean_est[0]),
    )
    first = measure_results(results, coordinate)
    assert calls == []
    l2 = list_built_in_metrics(MeanEstimation)["l2_error_to_sample_mean"]
    if separate_calls:
        measured = measure_results(first, metric, collection=collection)
        previous = measured.copy(deep=True)
        measured = measure_results(measured, l2, collection=collection)
        pd.testing.assert_frame_equal(measured[previous.columns], previous)
    else:
        measured = measure_results(first, [metric, l2], collection=collection)
    assert calls == [MeanEstimation] * (2 if separate_calls else 1)
    assert first["metric_arguments"].tolist() == [{"published_se": 7.0}] * 3
    assert measured[coordinate.name].tolist() == [0.0, 1.0, 2.0]

    other_coordinate = Metric(
        task=MeanEstimation,
        name="second_coordinate",
        calculation=lambda output: float(output.mean_est[1]),
    )
    extended = measure_results(measured, other_coordinate)
    pd.testing.assert_frame_equal(extended[measured.columns], measured)
    assert extended[other_coordinate.name].tolist() == [0.0, 2.0, -1.0]
    assert calls == [MeanEstimation] * (2 if separate_calls else 1)
    measured = extended
    assert instance.metric_arguments == {**stored, "published_se": 7.0}
    assert measured["instance"].iloc[0] is results["instance"].iloc[0]
    assert measured["metric_arguments"].iloc[0]["published_se"] == 7.0
    ratios = compute_error_ratio(measured, "sample_mean_mahalanobis_se", metric=metric)

    assert measured[metric.name].tolist() == pytest.approx([0.0, 1.0, math.inf])
    assert ratios[metric.name].tolist() == pytest.approx(
        [0.0, math.sqrt(3.0), math.inf]
    )
    aggregated = aggregate_trials(measured, metric=metric)
    assert compute_error_ratio(aggregated, "sample_mean_mahalanobis_se", metric=metric)[
        metric.name
    ].tolist() == pytest.approx([math.sqrt(3.0)])
    failed = results.copy()
    failed["status"] = "timeout"
    failed["payload"] = None
    failures = measure_results(failed, metric, collection=collection)
    assert (
        compute_error_ratio(failures, "sample_mean_mahalanobis_se", metric=metric)[
            metric.name
        ].tolist()
        == [math.inf] * 3
    )
