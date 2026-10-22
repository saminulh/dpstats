"""Public simple-linear-regression metrics."""

from dpstats.tasks.simple_linear_regression._metrics import (
    l2_error_to_ground_truth as l2_error_to_ground_truth,
    l2_error_to_ols as l2_error_to_ols,
)

__all__ = [
    "l2_error_to_ground_truth",
    "l2_error_to_ols",
]
