"""Simulated-instance producers for the Mean Estimation Task."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue

from dpstats.domains import LpBallDomain
from dpstats.tasks import MeanEstimation, compute_metric_arguments
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)


def _mean_and_noise_domain(
    *, seed: int, dimension: int, mean_radius: float, noise_radius_upper_bound: float
) -> tuple[np.random.Generator, np.ndarray, LpBallDomain]:
    if noise_radius_upper_bound < 0:
        raise ValueError("noise_radius_upper_bound must be nonnegative.")
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 1, size=dimension)
    mean = mean / np.linalg.norm(mean) * mean_radius
    return (
        rng,
        mean,
        LpBallDomain.at_origin(p=2, radius=noise_radius_upper_bound, dim=dimension),
    )


def _instance(
    mean: np.ndarray,
    values: np.ndarray,
    *,
    n_samples: int,
    dimension: int,
    mean_radius: float,
    noise_radius_upper_bound: float,
) -> tuple[MeanEstimationMetadata, MeanEstimationInput, dict[str, JsonValue]]:
    metadata = MeanEstimationMetadata(
        n_samples=n_samples,
        domain=LpBallDomain.at_origin(
            p=2, radius=mean_radius + noise_radius_upper_bound, dim=dimension
        ),
    )
    input_data = MeanEstimationInput(X=values + mean[None, :])
    arguments: dict[str, JsonValue] = {
        "ground_truth": mean.tolist(),
        **compute_metric_arguments(MeanEstimation, metadata, input_data),
    }
    return metadata, input_data, arguments


def simulate_clipped_normal(
    *,
    seed: int,
    mean_radius: float = 10.0,
    noise_radius_upper_bound: float = 10.0,
    dimension: int = 2,
    n_samples: int = 100,
) -> tuple[MeanEstimationMetadata, MeanEstimationInput, dict[str, JsonValue]]:
    """Generate a translated, L2-clipped Gaussian mean-estimation instance.

    Returns:
        Metadata, Input, and the ground_truth mean with the ``compute_metric_arguments``
        values: sample_mean and sample_mean_l2_se. Needs ``n_samples >= 2``.
    """
    rng, mean, noise_domain = _mean_and_noise_domain(
        seed=seed,
        dimension=dimension,
        mean_radius=mean_radius,
        noise_radius_upper_bound=noise_radius_upper_bound,
    )
    values = noise_domain.project(rng.normal(0, 1, size=(n_samples, dimension)))
    return _instance(
        mean,
        values,
        n_samples=n_samples,
        dimension=dimension,
        mean_radius=mean_radius,
        noise_radius_upper_bound=noise_radius_upper_bound,
    )


def simulate_clipped_laplacian(
    *,
    seed: int,
    mean_radius: float = 10.0,
    noise_radius_upper_bound: float = 10.0,
    dimension: int = 2,
    n_samples: int = 100,
) -> tuple[MeanEstimationMetadata, MeanEstimationInput, dict[str, JsonValue]]:
    """Generate a translated, L2-clipped Laplace mean-estimation instance.

    Returns:
        Metadata, Input, and the ground_truth mean with the ``compute_metric_arguments``
        values: sample_mean and sample_mean_l2_se. Needs ``n_samples >= 2``.
    """
    rng, mean, noise_domain = _mean_and_noise_domain(
        seed=seed,
        dimension=dimension,
        mean_radius=mean_radius,
        noise_radius_upper_bound=noise_radius_upper_bound,
    )
    values = noise_domain.project(rng.laplace(0, 1, size=(n_samples, dimension)))
    return _instance(
        mean,
        values,
        n_samples=n_samples,
        dimension=dimension,
        mean_radius=mean_radius,
        noise_radius_upper_bound=noise_radius_upper_bound,
    )


__all__ = ["simulate_clipped_normal", "simulate_clipped_laplacian"]
