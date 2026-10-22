"""dpstats library.

Public concerns live in their named modules: for example,
``dpstats.algorithm``, ``dpstats.parameters``,
``dpstats.budgets``, and task-local packages. The root exposes
collection operations, ``CompositeAlgorithm``, ``sweep``, and package metadata
without importing built-in Task discovery.
"""

from importlib.metadata import PackageNotFoundError, version

from dpstats.workspace import (
    TaskCollection,
    fetch_benchmark,
    init_workspace,
    open_collection,
)

from dpstats.algorithm import CompositeAlgorithm as CompositeAlgorithm
from dpstats import sweep as sweep

try:
    __version__ = version("dpstats")
except PackageNotFoundError:  # not installed (e.g. running from a bare source tree)
    __version__ = "unknown"


__all__ = [
    "__version__",
    "CompositeAlgorithm",
    "sweep",
    "open_collection",
    "init_workspace",
    "TaskCollection",
    "fetch_benchmark",
]
