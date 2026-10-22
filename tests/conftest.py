"""Shared pytest fixtures for dpstats tests."""

import os

# The gate runs several xdist workers per host; an unpinned BLAS in each one
# oversubscribes the cores and turns sub-second linear algebra into timeouts.
# Set before numpy is first imported so every worker and child inherits it.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

pytest_plugins = (
    "tests.fixtures.accounting_spy",
    "tests.fixtures.randomness_fake",
)

import subprocess
from pathlib import Path

import numpy as np
import pytest

from dpstats.workspace import (
    TaskCollection,
    init_workspace,
    open_collection,
)
from dpstats.task import Task, TaskInput, TaskMetadata, TaskOutput
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def collection(tmp_path: Path) -> TaskCollection:
    """The ``SIMPLE_TASK`` collection of a fresh workspace at ``tmp_path``."""
    return open_collection(init_workspace(tmp_path), SIMPLE_TASK)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Exempt `slow`/`performance` tests from the default per-test timeout.

    Those markers already carry their own resource or wall-clock cost; a
    fixed `timeout` marker on the item takes precedence over the ini option,
    so this raises theirs instead of racing the same tight clock every other
    test is held to.
    """
    for item in items:
        if item.get_closest_marker("slow") or item.get_closest_marker("performance"):
            item.add_marker(pytest.mark.timeout(0))


@pytest.fixture
def committed_library_source() -> None:
    """Skip a test whose subject is installed from an export of ``git HEAD``.

    Such an export carries the committed library and its build metadata, never
    the working tree, so an uncommitted change to either would be reported as a
    pass that says nothing about it.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "dpstats", "pyproject.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    uncommitted = [line[3:] for line in status.stdout.splitlines()]
    if uncommitted:
        pytest.skip(
            "the installed export is git HEAD, which does not carry the "
            f"uncommitted library source in {', '.join(uncommitted)}"
        )


@pytest.fixture
def linear_regression_data():
    rng = np.random.default_rng(42)
    n, d = 100, 5
    X = rng.standard_normal((n, d))
    true_beta = rng.standard_normal(d)
    y = X @ true_beta + 0.1 * rng.standard_normal(n)
    return X, y, true_beta


@pytest.fixture
def simple_task() -> tuple[Task, type[TaskMetadata], type[TaskInput], type[TaskOutput]]:
    """``SIMPLE_TASK`` and its Metadata, Input and Output classes."""
    return SIMPLE_TASK, SimpleMetadata, SimpleInput, SimpleOutput


@pytest.fixture
def simple_algorithm() -> type[SimpleAlgorithm]:
    """The ``SIMPLE_TASK`` algorithm class, whose ``scale`` multiplies the mean."""
    return SimpleAlgorithm


@pytest.fixture
def linear_regression_task():
    from dpstats.tasks import LinearRegression

    return LinearRegression


@pytest.fixture
def mean_estimation_task():
    from dpstats.tasks import MeanEstimation

    return MeanEstimation
