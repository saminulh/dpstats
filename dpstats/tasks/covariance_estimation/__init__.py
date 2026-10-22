"""Covariance estimation task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.covariance_estimation._types import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
    CovarianceEstimationOutput,
)

CovarianceEstimation = Task(
    id="covariance_estimation",
    name="Covariance Estimation",
    description="Privately estimate the covariance matrix of a sensitive sample.",
    metadata_cls=CovarianceEstimationMetadata,
    input_cls=CovarianceEstimationInput,
    output_cls=CovarianceEstimationOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.covariance_estimation import algorithms as algorithms
    from dpstats.tasks.covariance_estimation import metrics as metrics
    from dpstats.tasks.covariance_estimation import (
        transformations as transformations,
    )


__all__ = [
    "CovarianceEstimation",
    "CovarianceEstimationInput",
    "CovarianceEstimationMetadata",
    "CovarianceEstimationOutput",
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
