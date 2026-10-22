"""Built-in logistic regression algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.logistic_regression.algorithms.newton import (
    Newton,
)
from dpstats.tasks.logistic_regression.algorithms.gd import (
    GD,
)

__all__ = [
    "Newton",
    "GD",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    Newton,
    GD,
)
