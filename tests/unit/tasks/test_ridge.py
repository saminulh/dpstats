"""Tests for the non-private Ridge baseline."""

from __future__ import annotations

from dpstats.domains import InterceptDomain

import numpy as np
import pytest

from dpstats.domains import ProductDomain
from dpstats.budgets import NO_PRIVACY
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
)
from tests.unit.tasks.task_builders import lr_metadata


def _instance(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 3.0]) + rng.normal(0.0, 0.2, size=300)
    metadata = lr_metadata(300, design=4)
    return metadata, LinearRegressionInput(X=X, y=y)


def test_satisfies_ridge_stationarity():
    metadata, input = _instance()
    multiplier = 0.2
    output = Ridge(regularization_multiplier=multiplier).run(
        metadata, input, budget=NO_PRIVACY, seed=0
    )
    X, y = input.X, input.y
    penalty = multiplier * metadata.n_samples
    gradient = X.T @ (X @ output.beta - y) + penalty * output.beta
    assert np.linalg.norm(gradient) < 1e-12 * np.linalg.norm(X.T @ y)


def test_matches_analytic_solution_of_ill_conditioned_design():
    rotation: Array2D = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    singular_values: Array1D = np.array([1.0, 2.0**-14])
    X: Array2D = np.diag(singular_values) @ rotation.T
    y: Array1D = np.array([1.0, 2.0])
    n, dim = X.shape
    penalty = 2.0**-32
    multiplier = penalty / n

    # X = diag(s) V.T, so ridge separates in its known singular coordinates.
    expected: Array1D = rotation @ (
        singular_values * y / (singular_values**2 + penalty)
    )

    metadata = lr_metadata(n, design=dim)
    output = Ridge(regularization_multiplier=multiplier).run(
        metadata, LinearRegressionInput(X=X, y=y), NO_PRIVACY, 0
    )

    relative_error = np.linalg.norm(output.beta - expected) / np.linalg.norm(expected)
    assert relative_error < 1e-12


def test_keeps_directions_the_penalty_identifies():
    scales: Array1D = np.array([1.0, 1e-20])
    X: Array2D = np.diag(scales)
    y: Array1D = np.array([0.0, 1e-20])
    n, dim = X.shape
    multiplier = 5e-43
    penalty = multiplier * n
    expected: Array1D = scales * y / (scales**2 + penalty)

    output = Ridge(regularization_multiplier=multiplier).run(
        lr_metadata(n, design=dim),
        LinearRegressionInput(X=X, y=y),
        NO_PRIVACY,
        0,
    )

    assert output.beta == pytest.approx(expected, rel=1e-10, abs=1e-15)


def test_penalizes_a_declared_intercept_column():
    n = 4
    output = Ridge(regularization_multiplier=1.0).run(
        lr_metadata(n, design=ProductDomain(parts=(InterceptDomain(),))),
        LinearRegressionInput(X=np.ones((n, 1)), y=np.full(n, 5.0)),
        NO_PRIVACY,
        0,
    )

    assert output.beta == pytest.approx([2.5])


def test_near_zero_multiplier_approaches_ols():
    metadata, input = _instance()
    algorithm = Ridge(regularization_multiplier=1e-12)
    output = algorithm.run(metadata, input, budget=NO_PRIVACY, seed=0)
    expected, _, _, _ = np.linalg.lstsq(input.X, input.y, rcond=None)
    assert output.beta == pytest.approx(expected)


def test_regularization_shrinks_coefficient_norm():
    metadata, input = _instance()
    weak = Ridge(regularization_multiplier=1e-6).run(metadata, input, NO_PRIVACY, 0)
    strong = Ridge(regularization_multiplier=10.0).run(metadata, input, NO_PRIVACY, 0)
    assert np.linalg.norm(strong.beta) < np.linalg.norm(weak.beta)


def test_row_replication_preserves_coefficients():
    metadata, input = _instance()
    multiplier = 0.3
    original = Ridge(regularization_multiplier=multiplier).run(
        metadata, input, NO_PRIVACY, 0
    )
    replicated_metadata = lr_metadata(2 * metadata.n_samples, design=metadata.dim)
    replicated_input = LinearRegressionInput(
        X=np.concatenate([input.X, input.X]),
        y=np.concatenate([input.y, input.y]),
    )
    replicated = Ridge(regularization_multiplier=multiplier).run(
        replicated_metadata, replicated_input, NO_PRIVACY, 0
    )
    assert replicated.beta == pytest.approx(original.beta)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_rejects_invalid_multiplier(value: float):
    with pytest.raises(ValueError, match="regularization_multiplier"):
        Ridge(regularization_multiplier=value)
