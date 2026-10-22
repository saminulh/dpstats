"""Median Estimation task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.median_estimation._types import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)

MedianEstimation = Task(
    id="median_estimation",
    name="Median Estimation",
    description="Privately estimate the median of a sensitive scalar sample.",
    metadata_cls=MedianEstimationMetadata,
    input_cls=MedianEstimationInput,
    output_cls=MedianEstimationOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.median_estimation import algorithms as algorithms
    from dpstats.tasks.median_estimation import metrics as metrics
    from dpstats.tasks.median_estimation import (
        transformations as transformations,
    )


__all__ = [
    "MedianEstimation",
    "MedianEstimationInput",
    "MedianEstimationMetadata",
    "MedianEstimationOutput",
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
