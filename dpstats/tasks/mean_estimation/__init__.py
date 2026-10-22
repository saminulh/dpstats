"""Mean Estimation task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.mean_estimation._types import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)

MeanEstimation = Task(
    id="mean_estimation",
    name="Mean Estimation",
    description="Privately estimate the mean vector of a sensitive d-dimensional sample.",
    metadata_cls=MeanEstimationMetadata,
    input_cls=MeanEstimationInput,
    output_cls=MeanEstimationOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.mean_estimation import algorithms as algorithms
    from dpstats.tasks.mean_estimation import metrics as metrics
    from dpstats.tasks.mean_estimation import (
        transformations as transformations,
    )


__all__ = [
    "MeanEstimation",
    "MeanEstimationInput",
    "MeanEstimationMetadata",
    "MeanEstimationOutput",
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
