"""M-Estimation task, values, and losses."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.m_estimation._types import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss

MEstimation = Task(
    id="m_estimation",
    name="M-Estimation",
    description=(
        "Privately estimate the minimizer of a configured per-record loss over a "
        "sensitive sample."
    ),
    metadata_cls=MEstimationMetadata,
    input_cls=MEstimationInput,
    output_cls=MEstimationOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.m_estimation import algorithms as algorithms
    from dpstats.tasks.m_estimation import losses as losses


__all__ = [
    "MEstimation",
    "MEstimationInput",
    "MEstimationLoss",
    "MEstimationMetadata",
    "MEstimationOutput",
    "algorithms",
    "losses",
]


def __getattr__(name: str) -> ModuleType:
    """Load a declared public submodule when it is first used."""
    if name not in {"algorithms", "losses"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
