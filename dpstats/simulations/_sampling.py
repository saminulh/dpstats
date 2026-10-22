"""Shared sampling primitives for built-in simulated-instance producers."""

from __future__ import annotations

import numpy as np

from dpstats.task import Array2D
from dpstats.utils import split_seed


def sample_gaussian_design(
    n_samples: int,
    dim: int,
    scale: float,
    condition_num: float,
    seed: int,
) -> Array2D:
    """Draw a Gaussian design matrix with a controlled condition number."""
    if condition_num < 1:
        raise ValueError("condition_num must be at least 1.")
    if dim == 1 and condition_num > 1:
        raise ValueError(
            "A one-dimensional design always has condition number 1; "
            "condition_num > 1 requires dim >= 2."
        )
    base_seed, rotation_seed = split_seed(seed, 2)
    rng = np.random.default_rng(base_seed)
    design = rng.normal(0, scale, size=(n_samples, dim))
    if condition_num == 1:
        return design

    eigenvalues = rng.uniform(low=1, high=condition_num, size=dim)
    eigenvalues[0] = 1
    eigenvalues[1] = condition_num
    rotation_rng = np.random.default_rng(rotation_seed)
    rotation, _ = np.linalg.qr(rotation_rng.normal(size=(dim, dim)))
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] = -rotation[:, -1]
    return design * np.sqrt(eigenvalues) @ rotation.T
