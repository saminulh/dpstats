"""Built-in statistical tasks in the library."""

from dpstats.tasks import _builtins
from dpstats.tasks._builtins import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metrics,
    list_built_in_metric_arguments,
    compute_metric_arguments,
)

CovarianceEstimation = _builtins.CovarianceEstimation
"""Estimate the sample covariance of data."""

LinearRegression = _builtins.LinearRegression
"""Perform linear regression of a response given covariates."""

LogisticRegression = _builtins.LogisticRegression
"""Perform logistic regression of a binary response given covariates."""

MEstimation = _builtins.MEstimation
"""Minimize a per-sample loss function."""

MeanEstimation = _builtins.MeanEstimation
"""Estimate the sample mean of data."""

MedianEstimation = _builtins.MedianEstimation
"""Estimate the median of scalar data."""

SimpleLinearRegression = _builtins.SimpleLinearRegression
"""Perform linear regression with a single scalar covariate and intercept."""

SingleCoefficientEstimation = _builtins.SingleCoefficientEstimation
"""Estimate a single coefficient of interest from a linear regression."""

__all__ = [
    "BUILT_IN_TASKS",
    "CovarianceEstimation",
    "LinearRegression",
    "LogisticRegression",
    "MEstimation",
    "MeanEstimation",
    "MedianEstimation",
    "SimpleLinearRegression",
    "SingleCoefficientEstimation",
    "list_built_in_algorithms",
    "list_built_in_metrics",
    "list_built_in_metric_arguments",
    "compute_metric_arguments",
]
