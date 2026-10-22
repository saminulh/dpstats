"""Simulated-instance producers for the M-estimation Task."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import JsonValue

from dpstats.domains import EuclideanSpace
from dpstats.simulations._sampling import sample_gaussian_design
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    AbsoluteResidualLoss,
    SquaredResidualLoss,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationLoss,
)
from dpstats.utils import split_seed


_CONFIGURED_LOSSES: dict[str, MEstimationLoss] = {
    "squared": SquaredResidualLoss(),
    "absolute": AbsoluteResidualLoss(),
}


def simulate_gaussian(
    *,
    seed: int,
    dim: int = 2,
    n_samples: int = 200,
    covariate_noise: float = 1.0,
    response_noise: float = 1.0,
    condition_num: float = 1.0,
    loss: Literal["squared", "absolute"] = "squared",
) -> tuple[MEstimationMetadata, MEstimationInput, dict[str, JsonValue]]:
    """Generate Gaussian covariates and noise for M-estimation."""
    design_seed, remainder_seed = split_seed(seed, 2)
    X = sample_gaussian_design(
        n_samples=n_samples,
        dim=dim,
        scale=covariate_noise,
        condition_num=condition_num,
        seed=design_seed,
    )
    rng = np.random.default_rng(remainder_seed)
    beta_true = rng.normal(size=dim)
    beta_true /= np.linalg.norm(beta_true)
    y = X @ beta_true + rng.normal(0, response_noise, size=n_samples)
    metadata = MEstimationMetadata(
        parameter_domain=EuclideanSpace(dim=dim),
        n_samples=n_samples,
        data_domain=EuclideanSpace(dim=dim + 1),
        loss=_CONFIGURED_LOSSES[loss],
    )
    return metadata, MEstimationInput(data=np.column_stack([X, y])), {}


__all__ = ["simulate_gaussian"]
