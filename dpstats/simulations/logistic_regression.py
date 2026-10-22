"""Simulated-instance producers for the Logistic Regression Task."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue
from scipy.special import expit

from dpstats.budgets import NO_PRIVACY
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
)
from dpstats.tasks.logistic_regression.algorithms.newton import (
    Newton,
)


def simulate_gaussian(
    *, seed: int, n_samples: int = 100, dim: int = 5
) -> tuple[LogisticRegressionMetadata, LogisticRegressionInput, dict[str, JsonValue]]:
    """Generate Gaussian covariates and Bernoulli logistic responses."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, dim))
    true_theta = rng.normal(0, 1, size=dim)
    y = rng.binomial(1, expit(X @ true_theta)).astype(np.float64)
    metadata = LogisticRegressionMetadata(dim=dim, n_samples=n_samples)
    input_data = LogisticRegressionInput(X=X, y=y)
    mle_coef = (
        Newton()
        .run(metadata=metadata, input=input_data, budget=NO_PRIVACY, seed=0)
        .theta
    )
    return (
        metadata,
        input_data,
        {
            "ground_truth": true_theta.tolist(),
            "mle_coef": mle_coef.tolist(),
        },
    )


__all__ = ["simulate_gaussian"]
