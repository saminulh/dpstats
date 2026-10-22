"""The single-coefficient estimation task.

A child of ``linear_regression`` that estimates one coefficient of interest as
a scalar. Linear-regression algorithms can be presented through a standard
projection: run the full regression and retain the designated coordinate.

This contract is point estimation only. Test statistics, p-values, standard
errors, and confidence intervals belong to inference tasks with outputs that
state those distinct mathematical objects.

The targeted coordinate is public problem metadata: every instance declares
``coefficient_of_interest_index``.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.single_coefficient_estimation._types import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)

SingleCoefficientEstimation = Task(
    id="single_coefficient_estimation",
    name="Single Coefficient Estimation",
    description=(
        "Privately estimate one designated regression coefficient given "
        "sensitive features and response."
    ),
    metadata_cls=SingleCoefficientEstimationMetadata,
    input_cls=SingleCoefficientEstimationInput,
    output_cls=SingleCoefficientEstimationOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.single_coefficient_estimation import (
        algorithms as algorithms,
    )
    from dpstats.tasks.single_coefficient_estimation import (
        metrics as metrics,
    )
    from dpstats.tasks.single_coefficient_estimation import (
        transformations as transformations,
    )


__all__ = [
    "SingleCoefficientEstimation",
    "SingleCoefficientEstimationInput",
    "SingleCoefficientEstimationMetadata",
    "SingleCoefficientEstimationOutput",
    "algorithms",
    "metrics",
    "transformations",
]


def __getattr__(name: str) -> ModuleType:
    """Load a declared public submodule when it is first used."""
    if name not in {"algorithms", "metrics", "transformations"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
