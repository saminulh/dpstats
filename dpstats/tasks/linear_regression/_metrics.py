"""Post-run Metrics and metric arguments for linear regression."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dpstats.budgets import NO_PRIVACY
from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _ols_arguments(
    metadata: LinearRegressionMetadata, input: LinearRegressionInput
) -> dict[str, Any]:
    """Fit OLS once and derive every classical quantity the arguments read.

    Raises:
        ValueError: If the design has no residual degrees of freedom, or is
            rank-deficient so ``(X'X)^-1`` does not exist.
    """
    n, d = input.X.shape
    if n <= d:
        raise ValueError(
            f"design has n={n} rows and d={d} columns: "
            "no residual degrees of freedom."
        )
    rank = np.linalg.matrix_rank(input.X)
    if rank < d:
        raise ValueError(f"design is rank-deficient: rank {rank} < {d} columns.")
    from dpstats.tasks.linear_regression.algorithms import OLS

    beta = OLS().run(metadata, input, NO_PRIVACY, seed=0).beta
    rss = float(np.sum((input.X @ beta - input.y) ** 2))
    residual_variance = rss / (n - d)
    gram_inverse = np.linalg.inv(input.X.T @ input.X)
    return {
        "ols_coef": [float(value) for value in beta],
        "response_std": math.sqrt(float(np.var(input.y))),
        "ols_residual_rmse": math.sqrt(rss / n),
        "ols_prediction_se": math.sqrt(residual_variance * d / n),
        "ols_l2_se": math.sqrt(residual_variance * float(np.trace(gram_inverse))),
    }


ground_truth = MetricArgument(
    name="ground_truth",
    description="Ground-truth coefficient vector.",
)

ols_coef = MetricArgument(
    name="ols_coef",
    description="OLS coefficient vector fitted on the instance.",
    producer=_ols_arguments,
)

response_std = MetricArgument(
    name="response_std",
    description="Population standard deviation of the response, at divisor n.",
    producer=_ols_arguments,
)

ols_residual_rmse = MetricArgument(
    name="ols_residual_rmse",
    description="Root mean squared residual of the OLS fit, the square root of RSS divided by n.",
    producer=_ols_arguments,
)

ols_prediction_se = MetricArgument(
    name="ols_prediction_se",
    description=(
        "Root-mean-square standard error of the OLS predictions, with "
        "residual variance RSS/(n - d)."
    ),
    producer=_ols_arguments,
)

ols_l2_se = MetricArgument(
    name="ols_l2_se",
    description=(
        "Euclidean norm of the OLS coefficient standard errors, with "
        "residual variance RSS/(n - d)."
    ),
    producer=_ols_arguments,
)


def _l2_error_to_ground_truth(
    output: LinearRegressionOutput, *, ground_truth: Any
) -> float:
    r"""L2 error between the output coefficients and ground truth.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Ground-truth coefficient vector.

    Notes:
        Computes :math:`\lVert\hat\beta-\beta^\star\rVert_2`, the
        :math:`\ell_2` error to the ground-truth coefficient vector
        :math:`\beta^\star`.
    """
    return float(np.linalg.norm(output.beta - np.asarray(ground_truth, dtype=float)))


l2_error_to_ground_truth = Metric(
    task=LinearRegression,
    name="l2_error_to_ground_truth",
    calculation=_l2_error_to_ground_truth,
    requires=(ground_truth,),
    formula=r"\lVert \hat{\beta} - \beta^{\star} \rVert_2",
    label="L2 error vs ground truth",
    baseline=ols_l2_se,
)


def _l2_error_to_ols(output: LinearRegressionOutput, *, ols_coef: Any) -> float:
    r"""L2 error between the output coefficients and OLS.

    Args:
        output: The algorithm output to evaluate.
        ols_coef (array_like):
            OLS coefficient vector fitted on the instance.

    Notes:
        Computes :math:`\lVert\hat\beta-\hat\beta_{\rm OLS}\rVert_2`,
        the :math:`\ell_2` error to the OLS coefficient vector
        :math:`\hat\beta_{\rm OLS}`.
    """
    return float(np.linalg.norm(output.beta - np.asarray(ols_coef, dtype=float)))


l2_error_to_ols = Metric(
    task=LinearRegression,
    name="l2_error_to_ols",
    calculation=_l2_error_to_ols,
    requires=(ols_coef,),
    formula=r"\lVert \hat{\beta} - \hat{\beta}_{\rm OLS} \rVert_2",
    label="L2 error vs OLS",
    baseline=ols_l2_se,
)


def _residual_rmse(
    output: LinearRegressionOutput, *, input: LinearRegressionInput
) -> float:
    r"""In-sample root mean squared residual of the output coefficients.

    Args:
        output: The algorithm output to evaluate.
        input (LinearRegressionInput):
            The sample's design matrix X and response vector y.

    Notes:
        Computes :math:`\frac{1}{\sqrt{n}}\lVert X\hat\beta-y\rVert_2`, where
        :math:`X` is the sample's design matrix, :math:`y` is its response
        vector, and :math:`n` is the number of samples.
    """
    residuals = input.X @ output.beta - input.y
    return float(np.sqrt(np.mean(residuals**2)))


residual_rmse = Metric(
    task=LinearRegression,
    name="residual_rmse",
    calculation=_residual_rmse,
    requires=("input",),
    formula=r"\frac{1}{\sqrt{n}} \lVert X\hat{\beta} - y \rVert_2",
    label="Residual RMSE",
    baseline=ols_residual_rmse,
)


def _prediction_rmse_to_ols(
    output: LinearRegressionOutput,
    *,
    input: LinearRegressionInput,
    ols_coef: Any,
) -> float:
    r"""Root mean squared difference between the output and OLS predictions.

    Args:
        output: The algorithm output to evaluate.
        input (LinearRegressionInput):
            The sample's design matrix X and response vector y.
        ols_coef (array_like):
            OLS coefficient vector fitted on the instance.

    Notes:
        Computes :math:`\frac{1}{\sqrt n}\lVert X(\hat\beta-\hat\beta_{\rm OLS})\rVert_2`,
        where :math:`X` is the sample's design matrix and :math:`n` is the
        number of samples.

        Divided by ``ols_prediction_se``, :math:`\sqrt{s^2 d/n}`, it is the
        Mahalanobis distance of :math:`\hat\beta` from :math:`\hat\beta_{\rm OLS}`
        under the OLS covariance estimate :math:`s^2(X^\top X)^{-1}`, over
        :math:`\sqrt d`. Its square has the form of the F statistic for the
        hypothesis that the coefficients equal :math:`\hat\beta`; it is not
        F-distributed, since :math:`\hat\beta` depends on the same data.
    """
    difference = output.beta - np.asarray(ols_coef, dtype=float)
    return float(np.sqrt(np.mean((input.X @ difference) ** 2)))


prediction_rmse_to_ols = Metric(
    task=LinearRegression,
    name="prediction_rmse_to_ols",
    calculation=_prediction_rmse_to_ols,
    requires=(
        "input",
        ols_coef,
    ),
    formula=r"\frac{1}{\sqrt{n}} \lVert X(\hat{\beta} - \hat{\beta}_{\rm OLS}) \rVert_2",
    label="Prediction RMSE vs OLS",
    baseline=ols_prediction_se,
)


def _l2_error_to_ground_truth_relative_to_ols(
    output: LinearRegressionOutput, *, ground_truth: Any, ols_coef: Any
) -> float:
    r"""L2 error to ground truth, relative to the same error of OLS.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Ground-truth coefficient vector.
        ols_coef (array_like):
            OLS coefficient vector fitted on the instance.

    Notes:
        Computes
        :math:`\lVert\hat\beta-\beta^\star\rVert_2/\lVert\hat\beta_{\rm OLS}-\beta^\star\rVert_2`,
        the :math:`\ell_2` error to the ground-truth coefficient vector
        :math:`\beta^\star` divided by the same error of the OLS coefficient
        vector :math:`\hat\beta_{\rm OLS}`.
    """
    truth = np.asarray(ground_truth, dtype=float)
    ols = np.asarray(ols_coef, dtype=float)
    return float(np.linalg.norm(output.beta - truth) / np.linalg.norm(ols - truth))


l2_error_to_ground_truth_relative_to_ols = Metric(
    task=LinearRegression,
    name="l2_error_to_ground_truth_relative_to_ols",
    calculation=_l2_error_to_ground_truth_relative_to_ols,
    requires=(
        ground_truth,
        ols_coef,
    ),
    formula=r"\frac{\lVert \hat{\beta} - \beta^{\star} \rVert_2}"
    r"{\lVert \hat{\beta}_{\rm OLS} - \beta^{\star} \rVert_2}",
    label="L2 error relative to OLS",
)


METRICS: tuple[Metric, ...] = (
    l2_error_to_ground_truth,
    l2_error_to_ols,
    l2_error_to_ground_truth_relative_to_ols,
    residual_rmse,
    prediction_rmse_to_ols,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = (response_std,)
