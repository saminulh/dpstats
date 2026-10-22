"""Simulated-instance producers for the Median Estimation Task."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue

from dpstats.domains import BoxDomain, EuclideanSpace
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
)


def simulate_simple_gaussian(
    *, seed: int, median: float = 0.0, n_samples: int = 100, known_domain: bool = False
) -> tuple[MedianEstimationMetadata, MedianEstimationInput, dict[str, JsonValue]]:
    """Generate ``X ~ N(median, 1)`` for median estimation.

    ``known_domain=True`` uses the data-dependent observed range; it is useful
    only to exercise algorithms requiring supplied bounds and is not private.
    """
    X = np.random.default_rng(seed).normal(loc=median, scale=1.0, size=n_samples)
    if known_domain:
        domain = BoxDomain(bounds=np.array([[np.min(X), np.max(X)]]))
    else:
        domain = EuclideanSpace(dim=1)
    return (
        MedianEstimationMetadata(n_samples=n_samples, domain=domain),
        MedianEstimationInput(X=X),
        {"ground_truth": median},
    )


__all__ = ["simulate_simple_gaussian"]
