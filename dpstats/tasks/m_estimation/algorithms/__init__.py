"""Built-in m estimation algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.m_estimation.algorithms.approximate_minima_perturbation import (
    AMP,
)
from dpstats.tasks.m_estimation.algorithms.dp_gradient_descent import (
    DPSGD,
)
from dpstats.tasks.m_estimation.algorithms.erm import (
    ERM,
)
from dpstats.tasks.m_estimation.algorithms.gradient_descent import (
    GD,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_kifer import (
    ObjectivePerturbationKifer,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_redberg import (
    ObjectivePerturbationRedberg,
)
from dpstats.tasks.m_estimation.algorithms.sgd import (
    SGD,
)

__all__ = [
    "AMP",
    "DPSGD",
    "ERM",
    "GD",
    "ObjectivePerturbationKifer",
    "ObjectivePerturbationRedberg",
    "SGD",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    AMP,
    DPSGD,
    ERM,
    GD,
    ObjectivePerturbationKifer,
    ObjectivePerturbationRedberg,
    SGD,
)
