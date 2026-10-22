from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field
from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.domains import Domain
from dpstats.task import Array1D, Array2D


class MeanEstimationMetadata(TaskMetadata):
    """The public shape of one mean-estimation problem.

    The domain fixes both the dimension every record carries and the bounds a
    mechanism calibrates its noise from.
    """

    domain: Domain = Field(
        description="Public domain every data vector lies in; its dimension "
        "is the mean's dimension."
    )

    @property
    def dim(self) -> int:
        """Dimension of the data vectors."""
        return self.domain.dim

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, MeanEstimationInput):
            raise TypeError(
                f"MeanEstimation requires a MeanEstimationInput, not "
                f"{type(input).__name__}."
            )
        if input.X.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"MeanEstimation X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and a domain of dimension "
                f"{self.dim}."
            )
        if not np.all(self.domain.contains(input.X)):
            raise ValueError("MeanEstimation X leaves its declared domain.")


class MeanEstimationInput(TaskInput):
    """One mean-estimation problem's confidential records."""

    X: Annotated[Array2D, FieldSymbol("X")] = Field(
        description="Matrix with `n_samples` rows, one per sample."
    )


class MeanEstimationOutput(TaskOutput):
    """One mean-estimation output."""

    mean_est: Array1D = Field(description="Mean estimate.")
