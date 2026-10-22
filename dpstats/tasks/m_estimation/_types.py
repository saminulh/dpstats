"""Typed values and problem-specification checks for the M-estimation task."""

from __future__ import annotations

from functools import cached_property

import numpy as np

from typing import Annotated

from pydantic import Field
from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.domains import Domain
from dpstats.task import Array1D, Array2D
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
)


class MEstimationMetadata(TaskMetadata):
    """The public information supplied for an M-estimation problem.

    The metadata declares the loss function to be minimized and the public
    domains of the data and parameter vectors.
    """

    parameter_domain: Domain = Field(
        description="Public domain of the parameter vector."
    )
    data_domain: Domain = Field(description="Public domain containing every sample.")
    loss: MEstimationLoss = Field(description="Loss function.")

    @cached_property
    def loss_properties(self) -> LossProperties:
        """Return the properties of the loss function on this metadata.

        This method returns the properties of the loss function (such as its Lipschitz constant), which
        are a function of not only the loss but also the domains of the data and parameter vectors.
        """
        return self.loss.properties(self.data_domain, self.parameter_domain)

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, MEstimationInput):
            raise TypeError(
                f"MEstimation requires a MEstimationInput, not "
                f"{type(input).__name__}."
            )
        if input.data.shape != (self.n_samples, self.data_domain.dim):
            raise ValueError(
                f"MEstimation data has shape {input.data.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and a data_domain of "
                f"dimension {self.data_domain.dim}."
            )
        # Domain membership can allocate several arrays shaped like the records.
        for start in range(0, self.n_samples, 4096):
            if not np.all(self.data_domain.contains(input.data[start : start + 4096])):
                raise ValueError("MEstimation data leaves its declared data_domain.")


class MEstimationInput(TaskInput):
    """Private input data for an M-estimation problem."""

    data: Annotated[Array2D, FieldSymbol("z")] = Field(
        description="Data matrix with `n_samples` rows."
    )


class MEstimationOutput(TaskOutput):
    """Estimated parameter vector for an M-estimation problem."""

    theta: Array1D = Field(description="Estimated parameter vector.")
