"""Public mean-estimation metrics."""

from dpstats.tasks.mean_estimation._metrics import (
    l2_error_to_ground_truth as l2_error_to_ground_truth,
    l2_error_to_sample_mean as l2_error_to_sample_mean,
    mahalanobis_error_to_sample_mean as mahalanobis_error_to_sample_mean,
)

__all__ = [
    "l2_error_to_ground_truth",
    "l2_error_to_sample_mean",
    "mahalanobis_error_to_sample_mean",
]
