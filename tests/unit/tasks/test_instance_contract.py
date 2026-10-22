"""Every Metadata/Input contract, and its enforcement at generation.

A task instance is a ``(metadata, input)`` pair whose metadata is the public
problem specification algorithms calibrate against. Each Metadata owns the
check that its Input honors those public facts; instance generation
applies it to every instance it produces. These tests state each task's
contract directly — the shape its metadata declares and the domain it
declares — and confirm both the honest pair and each way of breaking it.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    ProductDomain,
)
from dpstats.task import TaskInput, TaskMetadata
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    SquaredErrorLoss,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)

Pair = tuple[TaskMetadata, TaskInput]


def _refused(build: Callable[[], Pair], match: str, id: str) -> Any:
    return pytest.param(build, match, id=id)


def _assert_refused(build: Callable[[], Pair], match: str) -> None:
    # Constructing a malformed Input may itself raise, so building happens
    # inside the refusal.
    with pytest.raises(ValueError, match=match):
        metadata, input_data = build()
        metadata.validate_input(input_data)


# --- Linear regression -------------------------------------------------------

LR_X = np.array([[0.5, -0.5], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
LR_Y = np.array([1.0, -1.0, 2.0, 0.0])


def _lr_metadata(n: int = 4, dim: int = 2) -> LinearRegressionMetadata:
    return LinearRegressionMetadata(
        n_samples=n,
        design_domain=BoxDomain(bounds=np.tile([-1.0, 1.0], (dim, 1))),
        response_domain=BoxDomain(bounds=np.array([[-5.0, 5.0]])),
    )


def _intercept_metadata() -> LinearRegressionMetadata:
    return LinearRegressionMetadata(
        n_samples=2,
        design_domain=ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=1))),
        response_domain=EuclideanSpace(dim=1),
    )


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (_lr_metadata(n=3), LinearRegressionInput(X=LR_X, y=LR_Y)),
            "n_samples",
            "row-count",
        ),
        _refused(
            lambda: (_lr_metadata(dim=3), LinearRegressionInput(X=LR_X, y=LR_Y)),
            "design_domain",
            "design-width",
        ),
        _refused(
            lambda: (_lr_metadata(), LinearRegressionInput(X=LR_X, y=LR_Y[:2])),
            "y has shape",
            "response-length",
        ),
        _refused(
            lambda: (
                _lr_metadata(),
                LinearRegressionInput(
                    X=np.array([[0.5, -0.5], [9.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
                    y=LR_Y,
                ),
            ),
            "design_domain",
            "design-value",
        ),
        _refused(
            lambda: (
                _lr_metadata(),
                LinearRegressionInput(X=LR_X, y=np.array([1.0, -1.0, 2.0, 99.0])),
            ),
            "response_domain",
            "response-value",
        ),
        _refused(
            lambda: (
                _intercept_metadata(),
                LinearRegressionInput(
                    X=np.array([[1.0, 0.3], [0.0, -0.3]]), y=np.array([0.0, 1.0])
                ),
            ),
            "design_domain",
            "intercept-not-one",
        ),
    ],
)
def test_linear_regression_refuses_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


# --- Single-coefficient estimation --------------------------------------------

SCE_X = np.array([[1.0, 1.0], [0.0, 2.0], [-2.0, 0.0]])
SCE_Y = np.array([0.5, -0.5, 1.0])


def _sce_metadata(n: int = 3) -> SingleCoefficientEstimationMetadata:
    return SingleCoefficientEstimationMetadata(
        n_samples=n,
        coefficient_of_interest_index=1,
        design_domain=LpBallDomain.at_origin(p=2, radius=2.0, dim=2),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (
                _sce_metadata(n=5),
                SingleCoefficientEstimationInput(X=SCE_X, y=SCE_Y),
            ),
            "n_samples",
            "row-count",
        ),
        _refused(
            lambda: (
                _sce_metadata(),
                SingleCoefficientEstimationInput(
                    X=np.array([[1.0, 1.0], [0.0, 2.0], [-3.0, 0.0]]), y=SCE_Y
                ),
            ),
            "design_domain",
            "design-row",
        ),
        _refused(
            lambda: (
                _sce_metadata(),
                SingleCoefficientEstimationInput(X=SCE_X, y=np.array([0.5, -0.5, 4.0])),
            ),
            "response_domain",
            "response-value",
        ),
    ],
)
def test_single_coefficient_estimation_refuses_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


# --- Logistic regression -----------------------------------------------------


def test_logistic_regression_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        LogisticRegressionMetadata(dim=0, n_samples=3)


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (
                LogisticRegressionMetadata(dim=3, n_samples=3),
                LogisticRegressionInput(
                    X=np.zeros((3, 2)), y=np.array([0.0, 1.0, 1.0])
                ),
            ),
            "dim",
            "design-width",
        ),
        _refused(
            lambda: (
                LogisticRegressionMetadata(dim=2, n_samples=3),
                LogisticRegressionInput(X=np.zeros((3, 2)), y=np.array([0.0, 1.0])),
            ),
            "y has shape",
            "response-length",
        ),
        _refused(
            lambda: (
                LogisticRegressionMetadata(dim=2, n_samples=3),
                LogisticRegressionInput(
                    X=np.zeros((3, 2)), y=np.array([0.0, 1.0, 2.0])
                ),
            ),
            "labels",
            "non-binary-label",
        ),
    ],
)
def test_logistic_regression_refuses_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


# --- M-estimation ------------------------------------------------------------


def _m_metadata(n: int = 3, record_dim: int = 2) -> MEstimationMetadata:
    return MEstimationMetadata(
        parameter_domain=EuclideanSpace(dim=1),
        n_samples=n,
        loss=SquaredErrorLoss(),
        data_domain=BoxDomain(bounds=np.tile([-1.0, 1.0], (record_dim, 1))),
    )


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (_m_metadata(), MEstimationInput(data=np.zeros((3, 4)))),
            "data_domain",
            "record-width",
        ),
        _refused(
            lambda: (
                _m_metadata(),
                MEstimationInput(data=np.array([[0.5, 0.5], [-1.0, 7.0], [0.0, 0.0]])),
            ),
            "data_domain",
            "record-value",
        ),
    ],
)
def test_m_estimation_refuses_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


def test_large_input_validation_bounds_memory_and_checks_the_last_record() -> None:
    data = np.zeros((65537, 32))
    input_data = MEstimationInput(data=data)
    metadata = _m_metadata(n=len(data), record_dim=data.shape[1])

    tracemalloc.start()
    try:
        metadata.validate_input(input_data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < data.nbytes / 8
    data[-1, -1] = 7.0
    with pytest.raises(ValueError, match="leaves its declared data_domain"):
        metadata.validate_input(input_data)


# --- Mean and median estimation ---------------------------------------------


def _mean_metadata() -> MeanEstimationMetadata:
    return MeanEstimationMetadata(
        n_samples=3, domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2)
    )


def _median_metadata() -> MedianEstimationMetadata:
    return MedianEstimationMetadata(
        n_samples=4, domain=BoxDomain(bounds=np.array([[0.0, 10.0]]))
    )


def test_an_input_belonging_to_another_task_is_refused() -> None:
    foreign = MedianEstimationInput(X=np.array([0.0, 0.5, 1.0]))

    with pytest.raises(TypeError, match="MeanEstimationInput"):
        _mean_metadata().validate_input(foreign)


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (_mean_metadata(), MeanEstimationInput(X=np.zeros((2, 2)))),
            "n_samples",
            "mean-row-count",
        ),
        _refused(
            lambda: (
                _mean_metadata(),
                MeanEstimationInput(X=np.array([[1.0, 0.0], [0.0, -1.0], [3.0, 3.0]])),
            ),
            "domain",
            "mean-row-value",
        ),
        _refused(
            lambda: (_median_metadata(), MedianEstimationInput(X=np.zeros((4, 1)))),
            "1D",
            "median-not-a-vector",
        ),
        _refused(
            lambda: (
                _median_metadata(),
                MedianEstimationInput(X=np.array([0.0, 3.0, 7.0, 11.0])),
            ),
            "domain",
            "median-value",
        ),
    ],
)
def test_location_tasks_refuse_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


# --- Simple linear regression -----------------------------------------------

SLR_X = np.array([-1.0, 0.0, 1.0])
SLR_Y = np.array([0.0, 1.0, 2.0])


def _slr_metadata() -> SimpleLinearRegressionMetadata:
    return SimpleLinearRegressionMetadata(
        n_samples=3,
        x_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
        y_domain=BoxDomain(bounds=np.array([[-2.0, 2.0]])),
    )


@pytest.mark.parametrize(
    "build, match",
    [
        _refused(
            lambda: (
                _slr_metadata(),
                SimpleLinearRegressionInput(x=SLR_X[:2], y=SLR_Y),
            ),
            "x has shape",
            "covariate-length",
        ),
        _refused(
            lambda: (
                _slr_metadata(),
                SimpleLinearRegressionInput(x=SLR_X, y=SLR_Y[:2]),
            ),
            "y has shape",
            "response-length",
        ),
        _refused(
            lambda: (
                _slr_metadata(),
                SimpleLinearRegressionInput(x=np.array([-1.0, 0.0, 4.0]), y=SLR_Y),
            ),
            "x_domain",
            "covariate-value",
        ),
        _refused(
            lambda: (
                _slr_metadata(),
                SimpleLinearRegressionInput(x=SLR_X, y=np.array([0.0, 1.0, 9.0])),
            ),
            "y_domain",
            "response-value",
        ),
    ],
)
def test_simple_linear_regression_refuses_a_broken_pair(
    build: Callable[[], Pair], match: str
) -> None:
    _assert_refused(build, match)


# --- Honest pairs ------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: (_lr_metadata(), LinearRegressionInput(X=LR_X, y=LR_Y)),
            id="linear_regression",
        ),
        pytest.param(
            lambda: (
                _intercept_metadata(),
                LinearRegressionInput(
                    X=np.array([[1.0, 0.3], [1.0, -0.3]]), y=np.array([0.0, 1.0])
                ),
            ),
            id="linear_regression-intercept",
        ),
        pytest.param(
            lambda: (
                _sce_metadata(),
                SingleCoefficientEstimationInput(X=SCE_X, y=SCE_Y),
            ),
            id="single_coefficient_estimation",
        ),
        pytest.param(
            lambda: (
                LogisticRegressionMetadata(dim=2, n_samples=3),
                LogisticRegressionInput(
                    X=np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
                    y=np.array([0.0, 1.0, 1.0]),
                ),
            ),
            id="logistic_regression",
        ),
        pytest.param(
            lambda: (
                _m_metadata(),
                MEstimationInput(data=np.array([[0.5, 0.5], [-1.0, 1.0], [0.0, 0.0]])),
            ),
            id="m_estimation",
        ),
        pytest.param(
            lambda: (
                _mean_metadata(),
                MeanEstimationInput(X=np.array([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]])),
            ),
            id="mean_estimation",
        ),
        pytest.param(
            lambda: (
                _median_metadata(),
                MedianEstimationInput(X=np.array([0.0, 3.0, 7.0, 10.0])),
            ),
            id="median_estimation",
        ),
        pytest.param(
            lambda: (_slr_metadata(), SimpleLinearRegressionInput(x=SLR_X, y=SLR_Y)),
            id="simple_linear_regression",
        ),
    ],
)
def test_an_honest_pair_passes(build: Callable[[], Pair]) -> None:
    metadata, input_data = build()
    metadata.validate_input(input_data)


def test_an_honest_simulated_instance_producer_returns_contract_values() -> None:
    from dpstats.simulations.mean_estimation import (
        simulate_clipped_normal,
    )

    metadata, input_data, _ = simulate_clipped_normal(
        seed=0,
        mean_radius=1.0,
        noise_radius_upper_bound=2.0,
        dimension=3,
        n_samples=25,
    )

    assert input_data.X.shape == (25, 3)
    # The declared radius is the sum of the two configured radii, and the
    # rows honor it: checked here by measuring the norms directly.
    assert metadata.domain.radius == 3.0
    assert np.max(np.linalg.norm(input_data.X, axis=1)) <= 3.0
