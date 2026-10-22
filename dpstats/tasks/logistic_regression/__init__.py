"""Logistic Regression task contract and value types."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dpstats.task import Task

from dpstats.tasks.logistic_regression._types import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
    LogisticRegressionOutput,
)

LogisticRegression = Task(
    id="logistic_regression",
    name="Logistic Regression",
    description=(
        "Privately estimate the logistic regression coefficient vector given "
        "sensitive features and a binary response."
    ),
    metadata_cls=LogisticRegressionMetadata,
    input_cls=LogisticRegressionInput,
    output_cls=LogisticRegressionOutput,
)

if TYPE_CHECKING:
    from dpstats.tasks.logistic_regression import algorithms as algorithms
    from dpstats.tasks.logistic_regression import metrics as metrics


__all__ = [
    "LogisticRegression",
    "LogisticRegressionInput",
    "LogisticRegressionMetadata",
    "LogisticRegressionOutput",
    "algorithms",
    "metrics",
]


def __getattr__(name: str) -> ModuleType:
    """Load a declared public submodule when it is first used."""
    if name not in {"algorithms", "metrics"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
