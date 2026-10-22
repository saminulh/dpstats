"""Simulated-instance producers for the Single Coefficient Estimation Task."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue

from dpstats.budgets import NO_PRIVACY
from dpstats.domains import LpBallDomain
from dpstats.tasks import (
    SingleCoefficientEstimation,
    compute_metric_arguments,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)
from dpstats.utils import split_seed


def simulate_clipped_gaussian(
    *,
    seed: int,
    dim: int = 3,
    n_samples: int = 200,
    coefficient_of_interest_index: int = 0,
    response_noise: float = 1.0,
    covariate_radius: float = 10.0,
    response_radius: float = 10.0,
) -> tuple[
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationInput,
    dict[str, JsonValue],
]:
    """Generate bounded Gaussian data for one regression coefficient.

    Returns:
        Metadata, Input, and computed metric arguments: the scalar
        ols_coef and the classical OLS standard error ols_se.
    """
    if not 0 <= coefficient_of_interest_index < dim:
        raise ValueError(
            "coefficient_of_interest_index must identify a covariate column."
        )
    if covariate_radius <= 0:
        raise ValueError("covariate_radius must be positive.")
    if response_radius <= 0:
        raise ValueError("response_radius must be positive.")
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=n_samples,
        coefficient_of_interest_index=coefficient_of_interest_index,
        design_domain=LpBallDomain.at_origin(p=2, radius=covariate_radius, dim=dim),
        response_domain=LpBallDomain.at_origin(p=2, radius=response_radius, dim=1),
    )
    (rng_seed,) = split_seed(seed, 1)
    rng = np.random.default_rng(rng_seed)
    X = metadata.design_domain.project(rng.normal(0, 1.0, size=(n_samples, dim)))
    beta_true = rng.normal(size=dim)
    beta_true /= np.linalg.norm(beta_true)
    y = X @ beta_true + rng.normal(0, response_noise, size=n_samples)
    y = metadata.response_domain.project(y[:, None])[:, 0]
    input_data = SingleCoefficientEstimationInput(X=X, y=y)
    beta = (
        OLS()
        .run(
            metadata=LinearRegressionMetadata(
                n_samples=metadata.n_samples,
                design_domain=metadata.design_domain,
                response_domain=metadata.response_domain,
            ),
            input=LinearRegressionInput(X=input_data.X, y=input_data.y),
            budget=NO_PRIVACY,
            seed=0,
        )
        .beta
    )
    produced = compute_metric_arguments(
        SingleCoefficientEstimation, metadata, input_data
    )
    return (
        metadata,
        input_data,
        {
            "ols_coef": float(beta[metadata.coefficient_of_interest_index]),
            **produced,
        },
    )


__all__ = ["simulate_clipped_gaussian"]
