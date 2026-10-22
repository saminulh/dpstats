"""Typed input, public metadata, and output for covariance estimation."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, field_validator
from typing_extensions import override

from dpstats.domains import Domain
from dpstats.parameters import FieldSymbol
from dpstats.task import Array2D, TaskInput, TaskMetadata, TaskOutput

SYMMETRY_TOLERANCE = 1e-12


class CovarianceEstimationMetadata(TaskMetadata):
    """The sample count and domain of a covariance-estimation input.

    The domain fixes the dimension; it makes no distributional or population
    moment promises.
    """

    domain: Domain = Field(description="Public domain containing every data row.")

    @property
    def dim(self) -> int:
        """Return the dimension declared by the domain."""
        return self.domain.dim

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, CovarianceEstimationInput):
            raise TypeError(
                "CovarianceEstimation requires a CovarianceEstimationInput."
            )
        if input.X.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and domain dimension {self.dim}."
            )
        if not np.all(self.domain.contains(input.X)):
            raise ValueError("X leaves its declared domain.")


class CovarianceEstimationInput(TaskInput):
    """Confidential row vectors, without an implicit intercept."""

    X: Annotated[Array2D, FieldSymbol("X")] = Field(
        description="Matrix with one record per row and one variable per column."
    )


class CovarianceEstimationOutput(TaskOutput):
    """A symmetric covariance estimate, which may be indefinite or infinite.

    Symmetry uses relative and absolute tolerances of 1e-12; the supplied
    entries are preserved.
    """

    covariance_est: Array2D = Field(description="Symmetric covariance estimate.")

    @field_validator("covariance_est")
    @classmethod
    def _validate_covariance_est(cls, value: Array2D) -> Array2D:
        if value.shape[0] < 1 or value.shape[0] != value.shape[1]:
            raise ValueError("covariance_est must be a nonempty square matrix.")
        if not np.allclose(
            value,
            value.T,
            rtol=SYMMETRY_TOLERANCE,
            atol=SYMMETRY_TOLERANCE,
            equal_nan=True,
        ):
            raise ValueError("covariance_est must be symmetric.")
        return value
