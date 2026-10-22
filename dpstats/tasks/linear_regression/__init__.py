"""Linear Regression task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.linear_regression._types import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)

LinearRegression = Task(
    id="linear_regression",
    name="Linear Regression",
    description=(
        "Privately estimate the regression coefficient vector given sensitive "
        "features and response."
    ),
    metadata_cls=LinearRegressionMetadata,
    input_cls=LinearRegressionInput,
    output_cls=LinearRegressionOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.linear_regression import algorithms as algorithms
    from dpstats.tasks.linear_regression import metrics as metrics
    from dpstats.tasks.linear_regression import (
        transformations as transformations,
    )


__all__ = [
    "LinearRegression",
    "LinearRegressionInput",
    "LinearRegressionMetadata",
    "LinearRegressionOutput",
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
