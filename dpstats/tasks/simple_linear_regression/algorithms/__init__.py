"""Built-in simple linear regression algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.simple_linear_regression.algorithms.dp_theil_sen import (
    DPTheilSen,
)
from dpstats.tasks.simple_linear_regression.algorithms.dpgd import (
    DPGD,
)
from dpstats.tasks.simple_linear_regression.algorithms.noisy_stats import (
    NoisyStats,
)
from dpstats.tasks.simple_linear_regression.algorithms.dp_rss import (
    DPRSS,
)

__all__ = [
    "DPGD",
    "DPTheilSen",
    "DPRSS",
    "NoisyStats",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    DPGD,
    DPTheilSen,
    DPRSS,
    NoisyStats,
)
