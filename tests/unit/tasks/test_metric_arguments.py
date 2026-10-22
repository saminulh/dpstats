"""The per-Task argument registry, its producers, and the simulated Instances."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from dpstats.domains import EuclideanSpace
from dpstats.metric import RESERVED_REQUIREMENTS, MetricArgument
from dpstats.task import Task
from dpstats.simulations import linear_regression as lr_simulations
from dpstats.simulations import mean_estimation as me_simulations
from dpstats.simulations import (
    single_coefficient_estimation as sce_simulations,
)
from dpstats.tasks import (
    BUILT_IN_TASKS,
    LinearRegression,
    MeanEstimation,
    SingleCoefficientEstimation,
    list_built_in_metrics,
    list_built_in_metric_arguments,
    compute_metric_arguments,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)

# The statistical baseline each built-in Metric of the benchmarked Tasks is
# read against by default.
_DECLARED_BASELINES = {
    "linear_regression": {
        "l2_error_to_ground_truth": "ols_l2_se",
        "l2_error_to_ols": "ols_l2_se",
        "l2_error_to_ground_truth_relative_to_ols": None,
        "residual_rmse": "ols_residual_rmse",
        "prediction_rmse_to_ols": "ols_prediction_se",
    },
    "mean_estimation": {
        "l2_error_to_ground_truth": "sample_mean_l2_se",
        "l2_error_to_sample_mean": "sample_mean_l2_se",
        "mahalanobis_error_to_sample_mean": "sample_mean_mahalanobis_se",
    },
    "single_coefficient_estimation": {
        "absolute_error_to_ols": "ols_se",
        "sign_disagreement_with_ols": None,
        "absolute_error_of_median_to_ols": "ols_se",
        "absolute_deviation_to_median": "ols_se",
    },
}


def _stored_requirements(task) -> set[str]:
    return {
        name
        for metric in list_built_in_metrics(task).values()
        for name in metric.required_names
        if name not in RESERVED_REQUIREMENTS
    }


@pytest.mark.parametrize("task", BUILT_IN_TASKS.values(), ids=lambda t: t.id)
def test_registry_covers_every_metric_requirement(task) -> None:
    registry = list_built_in_metric_arguments(task)
    assert _stored_requirements(task) <= set(registry)
    assert all(isinstance(ref, MetricArgument) for ref in registry.values())
    assert all(ref.description.strip() for ref in registry.values())


@pytest.mark.parametrize("task_id,declared", _DECLARED_BASELINES.items())
def test_each_metric_names_its_statistical_baseline(task_id, declared) -> None:
    metrics = list_built_in_metrics(BUILT_IN_TASKS[task_id])
    assert {name: metric.baseline_name for name, metric in metrics.items()} == declared


@pytest.mark.parametrize("task", BUILT_IN_TASKS.values(), ids=lambda t: t.id)
def test_every_statistical_baseline_is_a_produced_catalogue_key(task) -> None:
    registry = list_built_in_metric_arguments(task)
    for metric in list_built_in_metrics(task).values():
        if metric.baseline_name is not None:
            assert registry[metric.baseline_name].producer is not None, metric.name


def test_a_produced_value_no_metric_names_stays_in_the_catalogue() -> None:
    registry = list_built_in_metric_arguments(LinearRegression)
    assert registry["response_std"].producer is not None
    assert not any(
        metric.baseline_name == "response_std"
        or "response_std" in metric.required_names
        for metric in list_built_in_metrics(LinearRegression).values()
    )


def test_only_source_supplied_references_lack_a_producer() -> None:
    carried = {
        (task.id, name)
        for task in (LinearRegression, MeanEstimation, SingleCoefficientEstimation)
        for name, ref in list_built_in_metric_arguments(task).items()
        if ref.producer is None
    }
    assert carried == {
        ("linear_regression", "ground_truth"),
        ("mean_estimation", "ground_truth"),
        ("single_coefficient_estimation", "ols_coef"),
    }


def test_compute_metric_arguments_yields_exactly_the_produced_keys() -> None:
    metadata, data, _ = lr_simulations.simulate_clipped_gaussian(
        seed=3, dim=3, n_samples=80
    )
    produced = compute_metric_arguments(LinearRegression, metadata, data)
    assert set(produced) == {
        name
        for name, ref in list_built_in_metric_arguments(LinearRegression).items()
        if ref.producer is not None
    }


def test_compute_metric_arguments_refuses_foreign_metadata() -> None:
    metadata, data, _ = lr_simulations.simulate_clipped_gaussian(seed=3)
    with pytest.raises(TypeError, match="metadata"):
        compute_metric_arguments(MeanEstimation, metadata, data)


def _regression_problem(
    task: Task, X: list[list[float]], y: list[float]
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    if task is SingleCoefficientEstimation:
        return (
            SingleCoefficientEstimation.Metadata(
                n_samples=len(X),
                design_domain=EuclideanSpace(dim=2),
                response_domain=EuclideanSpace(dim=1),
                coefficient_of_interest_index=1,
            ),
            SingleCoefficientEstimation.Input(X=np.array(X), y=np.array(y)),
        )
    return (
        LinearRegression.Metadata(
            n_samples=len(X),
            design_domain=EuclideanSpace(dim=2),
            response_domain=EuclideanSpace(dim=1),
        ),
        LinearRegression.Input(X=np.array(X), y=np.array(y)),
    )


@pytest.mark.parametrize("task", [LinearRegression, SingleCoefficientEstimation])
@pytest.mark.parametrize(
    ("X", "message"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], "residual degrees of freedom"),
        ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], "rank-deficient"),
    ],
)
def test_regression_references_require_residual_degrees_and_full_rank(
    task: Task, X: list[list[float]], message: str
) -> None:
    metadata, data = _regression_problem(task, X, [0.0] * len(X))
    with pytest.raises(ValueError, match=message):
        compute_metric_arguments(task, metadata, data)


# The scraper's formulae, verbatim from
# scraper/input_recipes/baselines.py and mean_estimation.py, so one design is
# computed both ways.
def _scraper_linear_regression(X, y, beta):
    residuals = X @ beta - y
    rss = float(np.sum(residuals**2))
    n, d = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    unbiased = rss / (n - d)
    return {
        "response_std": float(np.std(y)),
        "ols_residual_rmse": float(np.sqrt(np.mean(residuals**2))),
        "ols_prediction_se": float(np.sqrt(unbiased * d / n)),
        "ols_l2_se": float(np.sqrt(unbiased * np.trace(xtx_inv))),
    }


def _scraper_sample_mean_l2_se(X):
    n = X.shape[0]
    if X.shape[1] == 1:
        trace = float(np.var(X[:, 0], ddof=1))
    else:
        trace = float(np.trace(np.cov(X, rowvar=False, ddof=1)))
    return float(np.sqrt(trace / n))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_regression_producers_match_the_scraper_formulae(seed) -> None:
    metadata, data, _ = lr_simulations.simulate_clipped_gaussian(
        seed=seed, dim=4, n_samples=150, condition_num=5.0
    )
    produced = compute_metric_arguments(LinearRegression, metadata, data)
    beta = np.asarray(produced["ols_coef"])
    expected = _scraper_linear_regression(data.X, data.y, beta)
    for name, value in expected.items():
        assert produced[name] == pytest.approx(value, abs=1e-12, rel=0), name


@pytest.mark.parametrize("dimension", [1, 3])
def test_sample_mean_producers_match_the_scraper_formulae(dimension) -> None:
    metadata, data, _ = me_simulations.simulate_clipped_normal(
        seed=4, dimension=dimension, n_samples=60
    )
    produced = compute_metric_arguments(MeanEstimation, metadata, data)
    np.testing.assert_allclose(
        produced["sample_mean"], np.mean(data.X, axis=0), atol=1e-12, rtol=0
    )
    assert produced["sample_mean_l2_se"] == pytest.approx(
        _scraper_sample_mean_l2_se(data.X), abs=1e-12, rel=0
    )


def test_sample_mean_l2_se_refuses_a_single_row() -> None:
    metadata, data, _ = me_simulations.simulate_clipped_normal(seed=0, n_samples=3)
    one_row = type(data)(X=data.X[:1])
    metadata = type(metadata)(n_samples=1, domain=metadata.domain)
    with pytest.raises(ValueError, match="n >= 2"):
        compute_metric_arguments(MeanEstimation, metadata, one_row)


@pytest.mark.parametrize("task", [LinearRegression, SingleCoefficientEstimation])
def test_regression_references_share_one_fit_and_match_a_hand_solved_design(
    task: Task,
) -> None:
    """Orthogonal columns give beta=(1, 2), RSS=4, and Gram diagonal=(4, 2).

    The residual variance is 2, so the slope SE and prediction SE are 1;
    the squared norm of the coefficient SEs is 2 * (1/4 + 1/2) = 3/2.
    """
    metadata, data = _regression_problem(
        task,
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
        [0.0, 0.0, 4.0, 0.0],
    )
    with patch("numpy.linalg.lstsq", wraps=np.linalg.lstsq) as fit:
        produced = compute_metric_arguments(task, metadata, data)
    assert fit.call_count == 1
    if task is SingleCoefficientEstimation:
        assert produced == {"ols_se": pytest.approx(1.0)}
    else:
        assert produced == {
            "ols_coef": pytest.approx([1.0, 2.0]),
            "response_std": pytest.approx(math.sqrt(3.0)),
            "ols_residual_rmse": pytest.approx(1.0),
            "ols_prediction_se": pytest.approx(1.0),
            "ols_l2_se": pytest.approx(math.sqrt(1.5)),
        }


def test_mean_references_match_a_hand_solved_sample() -> None:
    """Column means are (3, 4); both sample variances are 4, so tr(S)/n=8/3."""
    metadata = MeanEstimation.Metadata(n_samples=3, domain=EuclideanSpace(dim=2))
    data = MeanEstimation.Input(X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    produced = compute_metric_arguments(MeanEstimation, metadata, data)
    geometry = produced.pop("sample_mahalanobis_geometry")
    assert geometry["rank"] == 1
    np.testing.assert_allclose(geometry["projection"], [[0.5, 0.5], [0.5, 0.5]])
    whitener = np.asarray(geometry["whitener"])
    np.testing.assert_allclose(whitener.T @ whitener, np.full((2, 2), 1 / 16))
    assert produced == {
        "sample_mean": pytest.approx([3.0, 4.0]),
        "sample_mean_l2_se": pytest.approx(math.sqrt(8.0 / 3.0)),
        "sample_mean_mahalanobis_se": pytest.approx(math.sqrt(1.0 / 3.0)),
    }


_SIMULATED = {
    "linear_regression": lambda: lr_simulations.simulate_clipped_gaussian(
        seed=1, dim=3, n_samples=100
    ),
    "linear_regression.hypercube": lambda: lr_simulations.simulate_hypercube(
        seed=1, dim=3, n_samples=100
    ),
    "mean_estimation.normal": lambda: me_simulations.simulate_clipped_normal(seed=1),
    "mean_estimation.laplacian": lambda: me_simulations.simulate_clipped_laplacian(
        seed=1
    ),
    "single_coefficient_estimation": lambda: sce_simulations.simulate_clipped_gaussian(
        seed=1
    ),
}


@pytest.mark.parametrize("name", _SIMULATED)
def test_a_simulated_instance_carries_every_requirement_and_reference_of_its_metrics(
    name,
) -> None:
    metadata, data, metric_arguments = _SIMULATED[name]()
    task = type(metadata).task
    needed = _stored_requirements(task) | {
        metric.baseline_name
        for metric in list_built_in_metrics(task).values()
        if metric.baseline_name is not None
    }
    assert needed <= set(metric_arguments), sorted(needed - set(metric_arguments))
    produced = compute_metric_arguments(task, metadata, data)
    for key, value in produced.items():
        assert metric_arguments[key] == value


def test_a_simulated_single_coefficient_instance_stores_its_target_and_ols_se() -> None:
    _, _, metric_arguments = sce_simulations.simulate_clipped_gaussian(seed=1)
    assert set(metric_arguments) == {"ols_coef", "ols_se"}


def test_every_simulated_metric_is_measurable_from_the_stored_references() -> None:
    from dpstats.tasks.single_coefficient_estimation import (
        SingleCoefficientEstimationOutput,
    )

    _, data, metric_arguments = sce_simulations.simulate_clipped_gaussian(seed=1)
    output = SingleCoefficientEstimationOutput(coefficient=metric_arguments["ols_coef"])
    for metric in list_built_in_metrics(SingleCoefficientEstimation).values():
        # Group outputs are measured over the group; here the group is one Output.
        needed = {
            name: [output] if name == "outputs" else metric_arguments[name]
            for name in metric.required_names
        }
        assert math.isfinite(metric.calculate(output, **needed))
