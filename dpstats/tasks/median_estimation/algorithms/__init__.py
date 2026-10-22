"""Built-in median estimation algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.median_estimation.algorithms.sample_median import (
    SampleMedian,
)
from dpstats.tasks.median_estimation.algorithms.smooth_sensitivity import (
    SmoothSensitivity,
)
from dpstats.tasks.median_estimation.algorithms.exponential_mechanism import (
    ExponentialMechanism,
    WideExponentialMechanism,
)
from dpstats.tasks.median_estimation.algorithms.exact_distance_ptr import (
    ExactDistancePTR,
)
from dpstats.tasks.median_estimation.algorithms.unbounded_quantile_mechanism import (
    UnboundedQuantileMechanism,
)

__all__ = [
    "SampleMedian",
    "SmoothSensitivity",
    "ExponentialMechanism",
    "WideExponentialMechanism",
    "ExactDistancePTR",
    "UnboundedQuantileMechanism",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    SampleMedian,
    SmoothSensitivity,
    ExponentialMechanism,
    WideExponentialMechanism,
    ExactDistancePTR,
    UnboundedQuantileMechanism,
)
