"""Simple Linear Regression task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.simple_linear_regression._types import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)

SimpleLinearRegression = Task(
    id="simple_linear_regression",
    name="Simple Linear Regression",
    description=(
        "Privately estimate the slope and intercept given a sensitive scalar "
        "feature and response."
    ),
    metadata_cls=SimpleLinearRegressionMetadata,
    input_cls=SimpleLinearRegressionInput,
    output_cls=SimpleLinearRegressionOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.simple_linear_regression import (
        algorithms as algorithms,
    )
    from dpstats.tasks.simple_linear_regression import metrics as metrics
    from dpstats.tasks.simple_linear_regression import (
        transformations as transformations,
    )


__all__ = [
    "SimpleLinearRegression",
    "SimpleLinearRegressionInput",
    "SimpleLinearRegressionMetadata",
    "SimpleLinearRegressionOutput",
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
