"""Public logistic-regression metrics."""

from dpstats.tasks.logistic_regression._metrics import (
    l2_error_to_ground_truth as l2_error_to_ground_truth,
    l2_error_to_mle as l2_error_to_mle,
)

__all__ = [
    "l2_error_to_ground_truth",
    "l2_error_to_mle",
]
