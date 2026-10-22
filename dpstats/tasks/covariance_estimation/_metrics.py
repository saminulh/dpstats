"""Matrix-error metrics with separate population and sample metric arguments."""

from __future__ import annotations

from typing import Any

import numpy as np

from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import CovarianceEstimation
from dpstats.tasks.covariance_estimation import CovarianceEstimationOutput


ground_truth = MetricArgument(
    name="ground_truth",
    description="Population covariance matrix.",
)

sample_covariance = MetricArgument(
    name="sample_covariance",
    description="Centered sample covariance matrix with divisor n.",
)


def _frobenius_error_to_ground_truth(
    output: CovarianceEstimationOutput, *, ground_truth: Any
) -> float:
    """Frobenius distance from the estimate to the population covariance.

    Computes the square root of the sum of squared entries of
    ``output.covariance_est - ground_truth``, using the population covariance
    ``ground_truth`` supplied as an metric argument.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Population covariance matrix.
    """
    return float(
        np.linalg.norm(
            output.covariance_est - np.asarray(ground_truth, dtype=float), ord="fro"
        )
    )


def _spectral_error_to_ground_truth(
    output: CovarianceEstimationOutput, *, ground_truth: Any
) -> float:
    """Operator-2 distance from the estimate to the population covariance.

    Computes the largest singular value of ``output.covariance_est - ground_truth``,
    using the population covariance ``ground_truth`` supplied as an evaluation
    metric argument.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Population covariance matrix.
    """
    return float(
        np.linalg.norm(
            output.covariance_est - np.asarray(ground_truth, dtype=float), ord=2
        )
    )


def _frobenius_error_to_sample_covariance(
    output: CovarianceEstimationOutput, *, sample_covariance: Any
) -> float:
    """Frobenius distance to the centered sample covariance with divisor n.

    Computes the square root of the sum of squared entries of
    ``output.covariance_est - sample_covariance``. The metric argument
    ``sample_covariance`` is the centered sample covariance with divisor n.

    Args:
        output: The algorithm output to evaluate.
        sample_covariance (array_like):
            Centered sample covariance matrix with divisor n.
    """
    return float(
        np.linalg.norm(
            output.covariance_est - np.asarray(sample_covariance, dtype=float),
            ord="fro",
        )
    )


def _spectral_error_to_sample_covariance(
    output: CovarianceEstimationOutput, *, sample_covariance: Any
) -> float:
    """Operator-2 distance to the centered sample covariance with divisor n.

    Computes the largest singular value of
    ``output.covariance_est - sample_covariance``. The metric argument
    ``sample_covariance`` is the centered sample covariance with divisor n.

    Args:
        output: The algorithm output to evaluate.
        sample_covariance (array_like):
            Centered sample covariance matrix with divisor n.
    """
    return float(
        np.linalg.norm(
            output.covariance_est - np.asarray(sample_covariance, dtype=float), ord=2
        )
    )


METRICS: tuple[Metric, ...] = (
    Metric(
        task=CovarianceEstimation,
        name="frobenius_error_to_ground_truth",
        calculation=_frobenius_error_to_ground_truth,
        requires=(ground_truth,),
        formula=r"\lVert \hat{\Sigma} - \Sigma \rVert_F",
        label="Frobenius error vs ground truth",
    ),
    Metric(
        task=CovarianceEstimation,
        name="spectral_error_to_ground_truth",
        calculation=_spectral_error_to_ground_truth,
        requires=(ground_truth,),
        formula=r"\lVert \hat{\Sigma} - \Sigma \rVert_2",
        label="Spectral error vs ground truth",
    ),
    Metric(
        task=CovarianceEstimation,
        name="frobenius_error_to_sample_covariance",
        calculation=_frobenius_error_to_sample_covariance,
        requires=(sample_covariance,),
        formula=r"\lVert \hat{\Sigma} - S_n \rVert_F",
        label="Frobenius error vs sample cov.",
    ),
    Metric(
        task=CovarianceEstimation,
        name="spectral_error_to_sample_covariance",
        calculation=_spectral_error_to_sample_covariance,
        requires=(sample_covariance,),
        formula=r"\lVert \hat{\Sigma} - S_n \rVert_2",
        label="Spectral error vs sample cov.",
    ),
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
