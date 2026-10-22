"""Gaussian covariance instances with population and sample arguments."""

import numpy as np
from numpy.typing import ArrayLike
from pydantic import JsonValue

from dpstats.budgets import NO_PRIVACY
from dpstats.domains import EuclideanSpace
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
)
from dpstats.tasks.covariance_estimation._types import SYMMETRY_TOLERANCE
from dpstats.tasks.covariance_estimation.algorithms import SampleCovariance


def simulate_gaussian(
    *,
    seed: int,
    mean: ArrayLike = (0.0, 0.0),
    covariance: ArrayLike = ((1.0, 0.0), (0.0, 1.0)),
    n_samples: int = 100,
) -> tuple[
    CovarianceEstimationMetadata, CovarianceEstimationInput, dict[str, JsonValue]
]:
    """Draw Gaussian records without clipping or adding an intercept.

    Args:
        seed: Integer seed for the Gaussian draw.
        mean: Finite population mean vector; determines the dimension.
        covariance: Finite positive-semidefinite population covariance, symmetric
            to the task's numerical tolerance.
        n_samples: Positive number of records to draw.

    Returns:
        Public metadata with an unbounded domain, private input records, and
        metric arguments. ground_truth is the supplied population
        covariance; sample_covariance is the canonical centered covariance
        of the realized records with divisor n.

    Raises:
        ValueError: If the mean, covariance, or sample count is invalid.

    Example:
        >>> metadata, input_data, arguments = simulate_gaussian(seed=0, n_samples=1)
        >>> input_data.X.shape
        (1, 2)
        >>> int(np.count_nonzero(arguments["sample_covariance"]))
        0
    """
    if np.iscomplexobj(mean) or np.iscomplexobj(covariance):
        raise ValueError("mean and covariance must be real.")
    mean_vector = np.asarray(mean, dtype=float)
    covariance_matrix = np.asarray(covariance, dtype=float)
    if (
        mean_vector.ndim != 1
        or mean_vector.size < 1
        or not np.all(np.isfinite(mean_vector))
    ):
        raise ValueError("mean must be a nonempty finite vector.")
    dim = len(mean_vector)
    if covariance_matrix.shape != (dim, dim):
        raise ValueError("covariance must have shape (len(mean), len(mean)).")
    if not np.all(np.isfinite(covariance_matrix)):
        raise ValueError("covariance must be finite.")
    if not np.allclose(
        covariance_matrix,
        covariance_matrix.T,
        rtol=SYMMETRY_TOLERANCE,
        atol=SYMMETRY_TOLERANCE,
    ):
        raise ValueError("covariance must be symmetric.")
    metadata = CovarianceEstimationMetadata(
        n_samples=n_samples, domain=EuclideanSpace(dim=dim)
    )
    rng = np.random.default_rng(seed)
    values = rng.multivariate_normal(
        mean_vector,
        covariance_matrix,
        size=n_samples,
        check_valid="raise",
        tol=0,
        method="eigh",
    )
    input_data = CovarianceEstimationInput(X=values)
    sample = SampleCovariance().run(metadata, input_data, NO_PRIVACY, seed=0)
    return (
        metadata,
        input_data,
        {
            "ground_truth": covariance_matrix.tolist(),
            "sample_covariance": sample.covariance_est.tolist(),
        },
    )


__all__ = ["simulate_gaussian"]
