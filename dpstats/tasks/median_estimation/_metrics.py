"""Post-run Metrics and metric arguments for median estimation."""

from __future__ import annotations

import numpy as np

from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationOutput,
)


ground_truth = MetricArgument(name="ground_truth", description="Population median.")


def _absolute_error_to_ground_truth(
    output: MedianEstimationOutput, *, ground_truth: float
) -> float:
    r"""Absolute error between the estimate and population median.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (float):
            Population median.

    Notes:
        Computes :math:`\lvert\hat m-m\rvert`, the absolute error to the
        ground truth :math:`m`.
    """
    return float(np.abs(output.median_est - ground_truth))


absolute_error_to_ground_truth = Metric(
    task=MedianEstimation,
    name="absolute_error_to_ground_truth",
    calculation=_absolute_error_to_ground_truth,
    requires=(ground_truth,),
    formula=r"\lvert \hat{m} - m \rvert",
    label="Absolute error vs ground truth",
)


def _quantile_level_error(
    output: MedianEstimationOutput, *, input: MedianEstimationInput
) -> float:
    r"""Quantile error of the estimate under the empirical distribution.

    Args:
        output: The algorithm output to evaluate.
        input (MedianEstimationInput):
            The sample's observations.

    Notes:
        Computes :math:`\max\{\hat F_{<}(\hat m)-\tfrac12,\hat F_{>}(\hat m)-\tfrac12,0\}`,
        where :math:`\hat F_{<}(t)` and :math:`\hat F_{>}(t)` are the fractions
        of samples below and above :math:`t`. The error is zero when the
        estimate is a sample median.
    """
    data = input.X
    x = output.median_est
    p_less = float(np.mean(data < x))
    p_greater = float(np.mean(data > x))
    return float(max(p_less - 0.5, p_greater - 0.5, 0.0))


quantile_level_error = Metric(
    task=MedianEstimation,
    name="quantile_level_error",
    calculation=_quantile_level_error,
    requires=("input",),
    formula=r"\max\{P(Y < \hat{m}) - \tfrac{1}{2},"
    r"\; P(Y > \hat{m}) - \tfrac{1}{2},\; 0\}",
    label="Quantile level error",
)


METRICS: tuple[Metric, ...] = (
    quantile_level_error,
    absolute_error_to_ground_truth,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
