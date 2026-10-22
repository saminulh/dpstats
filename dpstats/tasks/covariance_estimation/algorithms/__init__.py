"""Built-in covariance-estimation algorithms."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.covariance_estimation.algorithms.sample_covariance import (
    SampleCovariance,
)

__all__ = ["SampleCovariance"]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (SampleCovariance,)
