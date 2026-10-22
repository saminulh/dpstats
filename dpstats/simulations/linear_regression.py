"""Simulated-instance producers for the Linear Regression Task."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import JsonValue
from scipy.special import ndtr

from dpstats.domains import BoxDomain, LpBallDomain
from dpstats.task import Array1D, Array2D
from dpstats.simulations._sampling import sample_gaussian_design
from dpstats.tasks import LinearRegression, compute_metric_arguments
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.utils import split_seed


def simulate_hypercube(
    *,
    seed: int,
    dim: int = 10,
    n_samples: int = 1000,
    condition_num: float = 1.0,
    covariate_radius: float = 1.0,
    response_noise: float = 1.0,
    signal_direction: Literal["strong", "weak", "generic"] = "generic",
) -> tuple[LinearRegressionMetadata, LinearRegressionInput, dict[str, JsonValue]]:
    """Generate linear responses on an equicorrelated uniform-marginal cube.

    Every coordinate is uniform on ``[-covariate_radius, covariate_radius]``.
    ``condition_num`` is the population covariance condition number, not the
    realized sample covariance or Gram condition number. Marginal variance
    stays radius squared / 3 as correlation changes. The joint distribution
    is uniform over the cube only when ``condition_num=1``.

    The unit-norm coefficient is the all-ones direction (strong), the first
    two coordinates' normalized contrast (weak), or a seeded random direction
    (generic). At condition number one all directions have equal variance.
    A weak direction requires at least two dimensions. Independent uniform
    response noise has standard deviation ``response_noise``; no clipping is
    performed. The public response bound radius*sqrt(dim) + sqrt(3)*noise is
    shared across directions. Coefficients remain fixed when n, noise, or
    conditioning changes at the same seed and dimension. No intercept is added.

    Returns:
        Metadata, Input, and the generating ground_truth reference.
        ols_coef is the design's own fitted OLS coefficient vector.
        response_std and ols_residual_rmse scale residual_rmse readings.
        ols_prediction_se scales prediction_rmse_to_ols; ols_l2_se scales l2_error_to_ols.
        Every produced key is ``compute_metric_arguments(LinearRegression, ...)``.

    References:
        The Gaussian copula identity Corr(Phi(Z1), Phi(Z2)) =
        6/pi * arcsin(Corr(Z1, Z2)/2):
        https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html
    """
    if type(dim) is not int or dim < 1:
        raise ValueError("dim must be a positive integer.")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not 1.0 <= condition_num < math.inf:
        raise ValueError("condition_num must be finite and at least 1.")
    if dim == 1 and condition_num != 1:
        raise ValueError("condition_num must be 1 when dim is 1.")
    if not 0.0 < covariate_radius < math.inf:
        raise ValueError("covariate_radius must be finite and positive.")
    if not 0.0 <= response_noise < math.inf:
        raise ValueError("response_noise must be finite and nonnegative.")
    if signal_direction not in ("strong", "weak", "generic"):
        raise ValueError("signal_direction must be strong, weak, or generic.")
    if signal_direction == "weak" and dim < 2:
        raise ValueError("The weak signal_direction requires dim >= 2.")

    design_seed, signal_seed, noise_seed = split_seed(seed, 3)
    design_rng = np.random.default_rng(design_seed)
    # Equicorrelation has eigenvalues 1+(d-1)*r and 1-r. Invert their
    # ratio, then the Gaussian copula's uniform-marginal correlation law.
    correlation = (condition_num - 1.0) / (condition_num + dim - 1.0)
    latent_correlation = 2.0 * math.sin(math.pi * correlation / 6.0)
    latent = math.sqrt(latent_correlation) * design_rng.normal(
        size=(n_samples, 1)
    ) + math.sqrt(1.0 - latent_correlation) * design_rng.normal(size=(n_samples, dim))
    X: Array2D = covariate_radius * (2.0 * ndtr(latent) - 1.0)

    if signal_direction == "strong":
        beta_true: Array1D = np.ones(dim) / math.sqrt(dim)
    elif signal_direction == "weak":
        beta_true = np.zeros(dim)
        beta_true[:2] = [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)]
    else:
        signal_rng = np.random.default_rng(signal_seed)
        beta_true = signal_rng.normal(size=dim)
        beta_true /= np.linalg.norm(beta_true)

    noise_radius = math.sqrt(3.0) * response_noise
    noise_rng = np.random.default_rng(noise_seed)
    y: Array1D = X @ beta_true + noise_rng.uniform(
        -noise_radius, noise_radius, size=n_samples
    )
    response_radius = covariate_radius * math.sqrt(dim) + noise_radius
    metadata = LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=BoxDomain(
            bounds=np.tile([-covariate_radius, covariate_radius], (dim, 1))
        ),
        response_domain=BoxDomain(
            bounds=np.array([[-response_radius, response_radius]])
        ),
    )
    input_data = LinearRegressionInput(X=X, y=y)
    arguments: dict[str, JsonValue] = {
        "ground_truth": [float(coefficient) for coefficient in beta_true],
        **compute_metric_arguments(LinearRegression, metadata, input_data),
    }
    return metadata, input_data, arguments


def simulate_clipped_gaussian(
    *,
    seed: int,
    dim: int = 2,
    n_samples: int = 200,
    covariate_noise: float = 1.0,
    response_noise: float = 1.0,
    condition_num: float = 1.0,
    covariate_radius: float = 10.0,
    response_radius: float = 10.0,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput, dict[str, JsonValue]]:
    """Generate a bounded Gaussian linear-regression instance.

    ``y = X @ beta + noise`` with a controlled design condition number. Rows
    and responses are projected into the declared L2 domains.

    Returns:
        Metadata, Input, and the Metric arguments below.
        ground_truth and ols_coef are the generating and fitted coefficients.
        response_std and ols_residual_rmse scale residual_rmse readings.
        ols_prediction_se scales prediction_rmse_to_ols; ols_l2_se scales l2_error_to_ols.
    """
    if covariate_radius <= 0:
        raise ValueError("covariate_radius must be positive.")
    if response_radius <= 0:
        raise ValueError("response_radius must be positive.")

    design_seed, remainder_seed = split_seed(seed, 2)
    X = sample_gaussian_design(
        n_samples=n_samples,
        dim=dim,
        scale=covariate_noise,
        condition_num=condition_num,
        seed=design_seed,
    )
    metadata = LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=LpBallDomain.at_origin(p=2, radius=covariate_radius, dim=dim),
        response_domain=LpBallDomain.at_origin(p=2, radius=response_radius, dim=1),
    )
    X = metadata.design_domain.project(X)

    rng = np.random.default_rng(remainder_seed)
    beta_true = rng.normal(size=dim)
    beta_true /= np.linalg.norm(beta_true)
    y = X @ beta_true + rng.normal(0, response_noise, size=n_samples)
    y = metadata.response_domain.project(y[:, None])[:, 0]
    input_data = LinearRegressionInput(X=X, y=y)
    return (
        metadata,
        input_data,
        {
            "ground_truth": beta_true.tolist(),
            **compute_metric_arguments(LinearRegression, metadata, input_data),
        },
    )


__all__ = ["simulate_hypercube", "simulate_clipped_gaussian"]
