"""Direct public-surface tests for built-in post-run metric functions."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st

from dpstats.metric import Metric
from dpstats.task import TaskOutput
from dpstats.tasks import (
    LinearRegression,
    LogisticRegression,
    MeanEstimation,
    MedianEstimation,
    SimpleLinearRegression,
    SingleCoefficientEstimation,
    list_built_in_metrics,
)
from dpstats.tasks.mean_estimation import MeanEstimationOutput
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionOutput,
)
from dpstats.tasks.logistic_regression import LogisticRegressionOutput
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionOutput,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationOutput,
)

mean_metrics = list_built_in_metrics(MeanEstimation)
l2_error_to_ground_truth = mean_metrics["l2_error_to_ground_truth"]
l2_error_to_sample_mean = mean_metrics["l2_error_to_sample_mean"]
linear_metrics = list_built_in_metrics(LinearRegression)
linear_l2_error_to_ground_truth = linear_metrics["l2_error_to_ground_truth"]
linear_l2_error_to_ols = linear_metrics["l2_error_to_ols"]
residual_rmse = linear_metrics["residual_rmse"]
prediction_rmse_to_ols = linear_metrics["prediction_rmse_to_ols"]
l2_error_to_ground_truth_relative_to_ols = linear_metrics[
    "l2_error_to_ground_truth_relative_to_ols"
]
logistic_metrics = list_built_in_metrics(LogisticRegression)
logistic_l2_error_to_ground_truth = logistic_metrics["l2_error_to_ground_truth"]
l2_error_to_mle = logistic_metrics["l2_error_to_mle"]
simple_linear_metrics = list_built_in_metrics(SimpleLinearRegression)
simple_l2_error_to_ground_truth = simple_linear_metrics["l2_error_to_ground_truth"]
simple_l2_error_to_ols = simple_linear_metrics["l2_error_to_ols"]
single_coefficient_metrics = list_built_in_metrics(SingleCoefficientEstimation)
sign_disagreement_with_ols = single_coefficient_metrics["sign_disagreement_with_ols"]
absolute_error_of_median_to_ols = single_coefficient_metrics[
    "absolute_error_of_median_to_ols"
]
absolute_deviation_to_median = single_coefficient_metrics[
    "absolute_deviation_to_median"
]


median_metrics = list_built_in_metrics(MedianEstimation)
quantile_level_error = median_metrics["quantile_level_error"]
absolute_error_to_ground_truth = median_metrics["absolute_error_to_ground_truth"]


def test_mean_l2_error_uses_instance_ground_truth() -> None:
    output = MeanEstimationOutput(mean_est=np.array([4.0, 6.0]))

    assert l2_error_to_ground_truth.calculate(
        output, ground_truth=[1.0, 2.0]
    ) == pytest.approx(5.0)


def test_mean_l2_error_uses_real_instance_sample_mean() -> None:
    output = MeanEstimationOutput(mean_est=np.array([4.0, 6.0]))

    assert l2_error_to_sample_mean.calculate(
        output, sample_mean=[1.0, 2.0]
    ) == pytest.approx(5.0)


def test_median_absolute_error_uses_instance_ground_truth() -> None:
    output = MedianEstimationOutput(median_est=7.5)

    assert absolute_error_to_ground_truth.calculate(output, ground_truth=2.0) == 5.5


def test_median_quantile_level_error_uses_instance_data() -> None:
    output = MedianEstimationOutput(median_est=4.0)
    input = MedianEstimationInput(X=np.array([0.0, 1.0, 2.0, 3.0]))

    assert quantile_level_error.calculate(output, input=input) == pytest.approx(0.5)


def test_linear_l2_error_uses_instance_ground_truth() -> None:
    output = LinearRegressionOutput(beta=np.array([3.0, 4.0]))

    assert linear_l2_error_to_ground_truth.calculate(
        output, ground_truth=[0.0, 0.0]
    ) == pytest.approx(5.0)


def test_linear_l2_error_uses_instance_ols_fit() -> None:
    output = LinearRegressionOutput(beta=np.array([3.0, 4.0]))

    assert linear_l2_error_to_ols.calculate(
        output, ols_coef=[0.0, 0.0]
    ) == pytest.approx(5.0)


def test_linear_l2_error_relative_to_ols_uses_ground_truth_and_ols_fit() -> None:
    output = LinearRegressionOutput(beta=np.array([3.0, 4.0]))
    references = {
        "ground_truth": [0.0, 0.0],
        "ols_coef": [0.0, 4.0],
    }

    assert l2_error_to_ground_truth_relative_to_ols.calculate(
        output, **references
    ) == pytest.approx(1.25)


def test_linear_residual_rmse_uses_instance_design_and_response() -> None:
    output = LinearRegressionOutput(beta=np.array([3.0, 4.0]))
    input = LinearRegressionInput(X=np.eye(2), y=np.array([1.0, 1.0]))

    assert residual_rmse.calculate(output, input=input) == pytest.approx(np.sqrt(6.5))


def test_linear_prediction_rmse_to_ols_uses_instance_design_and_ols_fit() -> None:
    output = LinearRegressionOutput(beta=np.array([3.0, 4.0]))
    input = LinearRegressionInput(X=np.eye(2), y=np.array([1.0, 1.0]))

    assert prediction_rmse_to_ols.calculate(
        output, input=input, ols_coef=[2.0, 3.0]
    ) == pytest.approx(1.0)


def test_logistic_l2_error_uses_instance_ground_truth() -> None:
    output = LogisticRegressionOutput(theta=np.array([3.0, 4.0]))

    assert logistic_l2_error_to_ground_truth.calculate(
        output, ground_truth=[0.0, 0.0]
    ) == pytest.approx(5.0)


def test_logistic_l2_error_uses_instance_mle_fit() -> None:
    output = LogisticRegressionOutput(theta=np.array([3.0, 4.0]))

    assert l2_error_to_mle.calculate(output, mle_coef=[0.0, 0.0]) == pytest.approx(5.0)


def test_simple_linear_l2_error_uses_instance_ground_truth() -> None:
    output = SimpleLinearRegressionOutput(slope=3.0, intercept=4.0)

    assert simple_l2_error_to_ground_truth.calculate(
        output, ground_truth=[0.0, 0.0]
    ) == pytest.approx(5.0)


def test_simple_linear_l2_error_uses_instance_ols_fit() -> None:
    output = SimpleLinearRegressionOutput(slope=3.0, intercept=4.0)

    assert simple_l2_error_to_ols.calculate(
        output, ols_coef=[0.0, 0.0]
    ) == pytest.approx(5.0)


def test_sign_disagreement_uses_instance_ols_coef() -> None:
    output = SingleCoefficientEstimationOutput(coefficient=-3.0)

    assert sign_disagreement_with_ols.calculate(output, ols_coef=-2.0) == 0.0


@pytest.mark.parametrize(
    "coefficient, expected", [(4.0, 3.0), (-2.0, 3.0), (1.0, 0.0), (np.inf, np.inf)]
)
def test_absolute_error_uses_only_the_ols_coef(
    coefficient: float, expected: float
) -> None:
    absolute_error_to_ols = list_built_in_metrics(SingleCoefficientEstimation)[
        "absolute_error_to_ols"
    ]
    output = SingleCoefficientEstimationOutput(coefficient=coefficient)

    assert absolute_error_to_ols.calculate(output, ols_coef=1.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "coefficients, expected",
    [([3.0, -1.0, 10.0], 3.0), ([1.0, 2.0, 3.0, 10.0], 2.5), ([-4.0], -4.0)],
)
def test_median_coefficient_is_the_median_of_the_group_coefficients(
    coefficients: list[float], expected: float
) -> None:
    outputs = [SingleCoefficientEstimationOutput(coefficient=c) for c in coefficients]
    at_median = SingleCoefficientEstimationOutput(coefficient=expected)

    assert absolute_deviation_to_median.calculate(
        at_median, outputs=outputs
    ) == pytest.approx(0.0)


def test_both_median_diagnostics_read_the_group_outputs() -> None:
    assert "outputs" in absolute_error_of_median_to_ols.required_names
    assert "outputs" in absolute_deviation_to_median.required_names


@pytest.mark.parametrize("coefficient", [-50.0, 0.0, 2.5, np.inf])
def test_error_of_median_reads_the_group_median_not_the_trial(
    coefficient: float,
) -> None:
    output = SingleCoefficientEstimationOutput(coefficient=coefficient)

    group = [SingleCoefficientEstimationOutput(coefficient=2.5)]

    assert absolute_error_of_median_to_ols.calculate(
        output, outputs=group, ols_coef=1.0
    ) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "coefficient, expected", [(4.0, 1.5), (1.0, 1.5), (2.5, 0.0), (np.inf, np.inf)]
)
def test_absolute_deviation_is_the_distance_to_the_group_median(
    coefficient: float, expected: float
) -> None:
    output = SingleCoefficientEstimationOutput(coefficient=coefficient)

    group = [SingleCoefficientEstimationOutput(coefficient=2.5)]

    assert absolute_deviation_to_median.calculate(
        output, outputs=group
    ) == pytest.approx(expected)


@given(
    coefficients=st.lists(
        st.floats(min_value=-1e3, max_value=1e3), min_size=1, max_size=15
    ).filter(lambda values: len(values) % 2 == 1),
    ols_coef=st.floats(min_value=-1e3, max_value=1e3),
)
def test_median_of_errors_lies_between_error_of_median_and_that_plus_spread(
    coefficients: list[float], ols_coef: float
) -> None:
    # For an odd group, |median b - t| <= median |b - t| <= |median b - t| + MAD:
    # at least half the trials sit beyond their median from t, and
    # |b - t| <= |b - median b| + |median b - t| for every trial.
    outputs = [SingleCoefficientEstimationOutput(coefficient=c) for c in coefficients]
    errors = [abs(c - ols_coef) for c in coefficients]
    bias = absolute_error_of_median_to_ols.calculate(
        outputs[0], outputs=outputs, ols_coef=ols_coef
    )
    spread = float(
        np.median(
            [
                absolute_deviation_to_median.calculate(output, outputs=outputs)
                for output in outputs
            ]
        )
    )
    tolerance = 1e-9 * (1.0 + max(abs(ols_coef), *map(abs, coefficients)))

    assert bias <= np.median(errors) + tolerance
    assert np.median(errors) <= bias + spread + tolerance


def test_median_diagnostics_read_against_the_ols_standard_error() -> None:
    assert absolute_error_of_median_to_ols.baseline_name == "ols_se"
    assert absolute_deviation_to_median.baseline_name == "ols_se"


_LINEAR = LinearRegressionOutput(beta=np.array([3.0, 4.0]))
_SIMPLE = SimpleLinearRegressionOutput(slope=3.0, intercept=4.0)
_SINGLE = SingleCoefficientEstimationOutput(coefficient=3.0)
_LOGISTIC = LogisticRegressionOutput(theta=np.array([3.0, 4.0]))


_REQUIREMENTS: list[tuple[Metric, TaskOutput, dict[str, object]]] = [
    (
        l2_error_to_ground_truth,
        MeanEstimationOutput(mean_est=np.array([4.0, 6.0])),
        {"ground_truth": [1.0, 2.0]},
    ),
    (
        l2_error_to_sample_mean,
        MeanEstimationOutput(mean_est=np.array([4.0, 6.0])),
        {"sample_mean": [1.0, 2.0]},
    ),
    (
        absolute_error_to_ground_truth,
        MedianEstimationOutput(median_est=7.5),
        {"ground_truth": 2.0},
    ),
    (linear_l2_error_to_ground_truth, _LINEAR, {"ground_truth": [0.0, 0.0]}),
    (linear_l2_error_to_ols, _LINEAR, {"ols_coef": [0.0, 0.0]}),
    (
        l2_error_to_ground_truth_relative_to_ols,
        _LINEAR,
        {"ground_truth": [0.0, 0.0], "ols_coef": [0.0, 4.0]},
    ),
    (
        prediction_rmse_to_ols,
        _LINEAR,
        {
            "input": LinearRegressionInput(X=np.eye(2), y=np.ones(2)),
            "ols_coef": [2.0, 3.0],
        },
    ),
    (logistic_l2_error_to_ground_truth, _LOGISTIC, {"ground_truth": [0.0, 0.0]}),
    (l2_error_to_mle, _LOGISTIC, {"mle_coef": [0.0, 0.0]}),
    (simple_l2_error_to_ground_truth, _SIMPLE, {"ground_truth": [0.0, 0.0]}),
    (simple_l2_error_to_ols, _SIMPLE, {"ols_coef": [0.0, 0.0]}),
    (sign_disagreement_with_ols, _SINGLE, {"ols_coef": -2.0}),
    (
        list_built_in_metrics(SingleCoefficientEstimation)["absolute_error_to_ols"],
        _SINGLE,
        {"ols_coef": 1.0},
    ),
    (
        absolute_error_of_median_to_ols,
        _SINGLE,
        {"outputs": [_SINGLE], "ols_coef": 1.0},
    ),
    (absolute_deviation_to_median, _SINGLE, {"outputs": [_SINGLE]}),
]


@pytest.mark.parametrize(
    ("metric", "output", "references"),
    _REQUIREMENTS,
    ids=[f"{metric.task.id}-{metric.name}" for metric, _, _ in _REQUIREMENTS],
)
def test_a_metric_refuses_to_calculate_without_each_declared_reference(
    metric: Metric, output: TaskOutput, references: dict[str, object]
) -> None:
    for missing in references:
        incomplete = {
            name: value for name, value in references.items() if name != missing
        }
        with pytest.raises(ValueError, match=missing):
            metric.calculate(output, **incomplete)
