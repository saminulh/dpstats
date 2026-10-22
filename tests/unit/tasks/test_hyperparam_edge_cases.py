"""Explicit zero-valued producer parameters reach the simulated instance.

A ``param or DEFAULT`` fallback would silently replace a falsy-but-valid
``0``/``0.0`` with the default; these tests observe the zero in the draw.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dpstats.simulations.linear_regression import (
    simulate_clipped_gaussian as linear_regression_gaussian,
)
from dpstats.simulations.m_estimation import (
    simulate_gaussian as m_estimation_gaussian,
)
from dpstats.task import Array1D, Array2D


def _linear_regression_design(input_data: Any) -> tuple[Array2D, Array1D]:
    return input_data.X, input_data.y


def _m_estimation_design(input_data: Any) -> tuple[Array2D, Array1D]:
    return input_data.data[:, :-1], input_data.data[:, -1]


@pytest.mark.parametrize(
    "producer, design",
    [
        (linear_regression_gaussian, _linear_regression_design),
        (m_estimation_gaussian, _m_estimation_design),
    ],
    ids=["linear_regression", "m_estimation"],
)
def test_zero_response_noise_leaves_an_exactly_linear_response(
    producer: Callable[..., Any],
    design: Callable[[Any], tuple[Array2D, Array1D]],
) -> None:
    _, input_data, _ = producer(seed=0, response_noise=0.0)
    X, y = design(input_data)

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    np.testing.assert_allclose(X @ coefficients, y, atol=1e-10)


def test_covariate_noise_zero_is_honored_as_a_fully_degenerate_design():
    """An explicit zero covariate scale is not silently replaced by the default.

    A zero-scale design is the exact zero matrix, which has no unique OLS
    fit; the refusal itself is the evidence that the explicit zero, not a
    silently substituted default, reached the design.
    """
    with pytest.raises(ValueError, match="rank-deficient"):
        linear_regression_gaussian(seed=0, covariate_noise=0.0)
