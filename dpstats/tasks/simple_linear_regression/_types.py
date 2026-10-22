from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.domains import Domain
from dpstats.task import Array1D


class SimpleLinearRegressionMetadata(TaskMetadata):
    """The public shape of one simple-linear-regression problem.

    Both declared domains are scalar, and their bounds are what a mechanism
    normalizes and calibrates against.
    """

    x_domain: Domain = Field(
        description="Public scalar domain every covariate lies in."
    )
    y_domain: Domain = Field(description="Public scalar domain every response lies in.")

    @model_validator(mode="after")
    def _validate_metadata(self) -> SimpleLinearRegressionMetadata:
        if self.x_domain.dim != 1 or self.y_domain.dim != 1:
            raise ValueError(
                "SimpleLinearRegression x_domain and y_domain must be scalar."
            )
        return self

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, SimpleLinearRegressionInput):
            raise TypeError(
                f"SimpleLinearRegression requires a SimpleLinearRegressionInput, "
                f"not {type(input).__name__}."
            )
        if input.x.shape != (self.n_samples,):
            raise ValueError(
                f"SimpleLinearRegression x has shape {input.x.shape}, but its "
                f"metadata declares n_samples={self.n_samples}."
            )
        if input.y.shape != (self.n_samples,):
            raise ValueError(
                f"SimpleLinearRegression y has shape {input.y.shape}, but its "
                f"metadata declares n_samples={self.n_samples}."
            )
        if not np.all(self.x_domain.contains(input.x[:, None])):
            raise ValueError("SimpleLinearRegression x leaves its declared x_domain.")
        if not np.all(self.y_domain.contains(input.y[:, None])):
            raise ValueError("SimpleLinearRegression y leaves its declared y_domain.")


class SimpleLinearRegressionInput(TaskInput):
    """One simple-linear-regression problem's confidential records."""

    x: Annotated[Array1D, FieldSymbol("x")] = Field(
        description="Vector with `n_samples` entries, one covariate per sample."
    )
    y: Annotated[Array1D, FieldSymbol("y")] = Field(
        description="Vector with `n_samples` entries, one response per sample."
    )


class SimpleLinearRegressionOutput(TaskOutput):
    """One simple-linear-regression output.

    The output line is ``y ~= slope * x + intercept`` in the source scale.
    """

    slope: float = Field(description="Slope estimate.")
    intercept: float = Field(description="Intercept estimate.")
