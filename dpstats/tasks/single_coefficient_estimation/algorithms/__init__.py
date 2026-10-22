"""Built-in single-coefficient estimation algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.single_coefficient_estimation.algorithms.local_output_perturbation import (
    LocalOutputPerturbation,
)
from dpstats.tasks.single_coefficient_estimation.algorithms.zero_coefficient import (
    ZeroCoefficient,
)

__all__ = ["LocalOutputPerturbation", "ZeroCoefficient"]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    LocalOutputPerturbation,
    ZeroCoefficient,
)
