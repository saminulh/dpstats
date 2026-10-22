"""Post-run Metrics and metric arguments for mean estimation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import pinv

from dpstats.metric import Metric, MetricArgument
from dpstats.task import Array2D
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationOutput,
    MeanEstimationInput,
    MeanEstimationMetadata,
)


def _sample_mean_arguments(
    metadata: MeanEstimationMetadata, input: MeanEstimationInput
) -> dict[str, Any]:
    """Average the sample; defined for any number of records."""
    del metadata
    return {"sample_mean": [float(value) for value in np.mean(input.X, axis=0)]}


def _sample_dispersion_arguments(
    metadata: MeanEstimationMetadata, input: MeanEstimationInput
) -> dict[str, Any]:
    """Derive the sampling scales and covariance geometry from the sample.

    Raises:
        ValueError: If the sample has fewer than two records, so no sample
            covariance exists.
    """
    del metadata
    X: Array2D = input.X
    n = X.shape[0]
    if n < 2:
        raise ValueError(
            f"sample dispersion needs n >= 2 rows, got n={n}: no sample covariance."
        )
    centered = X - np.mean(X, axis=0)
    inverse, rank = pinv(centered, return_rank=True)
    # S^+ = (n - 1) X^+ (X^+)^T. QR preserves this norm without storing n rows.
    whitener = np.linalg.qr(inverse.T, mode="r") * math.sqrt(n - 1)
    trace = float(np.sum(np.var(X, axis=0, ddof=1)))
    return {
        "sample_mean_l2_se": math.sqrt(trace / n),
        "sample_mahalanobis_geometry": {
            "rank": int(rank),
            "projection": (inverse @ centered).tolist(),
            "whitener": whitener.tolist(),
        },
        "sample_mean_mahalanobis_se": math.sqrt(int(rank) / n),
    }


ground_truth = MetricArgument(name="ground_truth", description="Population mean.")

sample_mean = MetricArgument(
    name="sample_mean",
    description="Sample mean.",
    producer=_sample_mean_arguments,
)

sample_mean_l2_se = MetricArgument(
    name="sample_mean_l2_se",
    description=(
        "Euclidean norm of the sample mean's standard errors, sqrt(tr(S)/n) "
        "with S the sample covariance at divisor n - 1."
    ),
    producer=_sample_dispersion_arguments,
)

sample_mahalanobis_geometry = MetricArgument(
    name="sample_mahalanobis_geometry",
    description=(
        "The sample covariance's rank, span and whitener, used by the "
        "Mahalanobis metrics."
    ),
    producer=_sample_dispersion_arguments,
)

sample_mean_mahalanobis_se = MetricArgument(
    name="sample_mean_mahalanobis_se",
    description=(
        "Sample mean Mahalanobis standard error, sqrt(rank(S)/n), with S the "
        "sample covariance at divisor n - 1."
    ),
    producer=_sample_dispersion_arguments,
)


def _l2_error_to_ground_truth(
    output: MeanEstimationOutput, *, ground_truth: Any
) -> float:
    r"""L2 error between the estimate and population mean.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Population mean.

    Notes:
        Computes :math:`\lVert \hat{\mu} - \mu \rVert_2`, the :math:`\ell_2`
        error to the ground truth :math:`\mu`.
    """
    return float(
        np.linalg.norm(output.mean_est - np.asarray(ground_truth, dtype=float))
    )


l2_error_to_ground_truth = Metric(
    task=MeanEstimation,
    name="l2_error_to_ground_truth",
    calculation=_l2_error_to_ground_truth,
    requires=(ground_truth,),
    formula=r"\lVert \hat{\mu} - \mu \rVert_2",
    label="L2 error vs ground truth",
    baseline=sample_mean_l2_se,
)


def _l2_error_to_sample_mean(
    output: MeanEstimationOutput, *, sample_mean: Any
) -> float:
    r"""L2 error between the estimate and sample mean.

    Args:
        output: The algorithm output to evaluate.
        sample_mean (array_like):
            Sample mean.

    Notes:
        Computes :math:`\lVert \hat{\mu} - \bar{x} \rVert_2`, the :math:`\ell_2`
        error to the sample mean :math:`\bar{x}`.
    """
    return float(np.linalg.norm(output.mean_est - np.asarray(sample_mean, dtype=float)))


l2_error_to_sample_mean = Metric(
    task=MeanEstimation,
    name="l2_error_to_sample_mean",
    calculation=_l2_error_to_sample_mean,
    requires=(sample_mean,),
    formula=r"\lVert \hat{\mu} - \bar{x} \rVert_2",
    label="L2 error vs sample mean",
    baseline=sample_mean_l2_se,
)


def _mahalanobis_error_to_sample_mean(
    output: MeanEstimationOutput,
    *,
    sample_mean: Any,
    sample_mahalanobis_geometry: Any,
) -> float:
    r"""Mahalanobis error to the sample mean, infinite outside its affine span.

    Args:
        output: The algorithm output to evaluate.
        sample_mean (array_like): Sample mean vector.
        sample_mahalanobis_geometry (dict):
            The sample covariance's rank, span and whitener, used by the Mahalanobis metrics.

    Notes:
        For :math:`\Delta=\hat\mu-\bar x`, returns
        :math:`\sqrt{\Delta^\top S^+\Delta}` when
        :math:`\Delta\in\operatorname{range}(S)`, and positive infinity otherwise.
        A zero covariance gives zero error only for an exact match to the sample
        mean. The baseline ``sample_mean_mahalanobis_se`` is the plug-in RMS
        sampling scale :math:`\sqrt{\operatorname{rank}(S)/n}`.

        :func:`scipy.linalg.pinv` computes the centered sample's pseudoinverse
        and rank with its default tolerance. :func:`numpy.linalg.qr` stores a
        compact whitening factor; :func:`numpy.allclose` with defaults checks
        membership in the sample span. No covariance eigendecomposition is needed.
    """
    difference = output.mean_est - np.asarray(sample_mean, dtype=float)
    if np.isinf(difference).any():
        return math.inf
    rank = sample_mahalanobis_geometry["rank"]
    if rank == 0:
        return 0.0 if np.all(difference == 0) else math.inf
    projection = np.asarray(sample_mahalanobis_geometry["projection"], dtype=float)
    if rank < difference.size and not np.allclose(projection @ difference, difference):
        return math.inf
    whitener = np.asarray(sample_mahalanobis_geometry["whitener"], dtype=float)
    return float(np.linalg.norm(whitener @ difference))


mahalanobis_error_to_sample_mean = Metric(
    task=MeanEstimation,
    name="mahalanobis_error_to_sample_mean",
    calculation=_mahalanobis_error_to_sample_mean,
    requires=(
        sample_mean,
        sample_mahalanobis_geometry,
    ),
    formula=(
        r"\begin{cases}"
        r"\sqrt{(\hat\mu-\bar x)^\top S^+(\hat\mu-\bar x)},"
        r"&\hat\mu-\bar x\in\operatorname{range}(S),\\"
        r"+\infty,&\text{otherwise},"
        r"\end{cases}"
        r"\quad S=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)^\top"
    ),
    label="Mahalanobis error to sample mean",
    baseline=sample_mean_mahalanobis_se,
)


METRICS: tuple[Metric, ...] = (
    l2_error_to_ground_truth,
    l2_error_to_sample_mean,
    mahalanobis_error_to_sample_mean,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
