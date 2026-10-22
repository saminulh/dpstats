"""Simulated-instance producers for the Simple Linear Regression Task."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue

from dpstats.algorithm import CompositeAlgorithm
from dpstats.budgets import NO_PRIVACY
from dpstats.domains import BoxDomain
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
)
from dpstats.tasks.simple_linear_regression.transformations import (
    ToLinearRegression,
)


def simulate_clipped_gaussian(
    *,
    seed: int,
    n_samples: int = 200,
    slope: float = 2.0,
    intercept: float = 1.0,
    x_noise: float = 1.0,
    response_noise: float = 1.0,
    x_radius: float = 10.0,
    response_radius: float = 10.0,
) -> tuple[
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionInput,
    dict[str, JsonValue],
]:
    """Generate bounded scalar Gaussian linear-regression observations."""
    if x_radius <= 0:
        raise ValueError("x_radius must be positive.")
    if response_radius <= 0:
        raise ValueError("response_radius must be positive.")
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(0, x_noise, size=n_samples), -x_radius, x_radius)
    y = slope * x + intercept + rng.normal(0, response_noise, size=n_samples)
    y = np.clip(y, -response_radius, response_radius)
    metadata = SimpleLinearRegressionMetadata(
        n_samples=n_samples,
        x_domain=BoxDomain(bounds=np.array([[-x_radius, x_radius]], dtype=float)),
        y_domain=BoxDomain(
            bounds=np.array([[-response_radius, response_radius]], dtype=float)
        ),
    )
    input_data = SimpleLinearRegressionInput(x=x, y=y)
    ols_output = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=OLS()
    ).run(metadata=metadata, input=input_data, budget=NO_PRIVACY, seed=0)
    return (
        metadata,
        input_data,
        {
            "ground_truth": [slope, intercept],
            "ols_coef": [ols_output.slope, ols_output.intercept],
        },
    )


__all__ = ["simulate_clipped_gaussian"]
