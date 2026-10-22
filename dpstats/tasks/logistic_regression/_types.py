from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.task import Array1D, Array2D


class LogisticRegressionMetadata(TaskMetadata):
    """The public shape of one logistic-regression problem.

    The design carries no declared domain: the dimension is the only public
    fact beyond the sample count.
    """

    dim: Annotated[int, FieldSymbol("d")] = Field(description="Number of covariates.")

    @model_validator(mode="after")
    def _validate_metadata(self) -> LogisticRegressionMetadata:
        if self.dim < 1:
            raise ValueError("LogisticRegression dim must be positive.")
        return self

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, LogisticRegressionInput):
            raise TypeError(
                f"LogisticRegression requires a LogisticRegressionInput, not "
                f"{type(input).__name__}."
            )
        if input.X.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"LogisticRegression X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and dim={self.dim}."
            )
        if input.y.shape != (self.n_samples,):
            raise ValueError(
                f"LogisticRegression y has shape {input.y.shape}, but its metadata "
                f"declares n_samples={self.n_samples}."
            )
        if not np.all((input.y == 0.0) | (input.y == 1.0)):
            raise ValueError(
                "LogisticRegression y must contain only the labels 0 and 1."
            )


class LogisticRegressionInput(TaskInput):
    """One logistic-regression problem's confidential records."""

    X: Annotated[Array2D, FieldSymbol("X")] = Field(
        description="Matrix with `n_samples` rows, one per sample."
    )
    y: Annotated[Array1D, FieldSymbol("y")] = Field(
        description="Vector with `n_samples` entries, one binary label (0 or 1) per sample."
    )


class LogisticRegressionOutput(TaskOutput):
    """One logistic-regression output."""

    theta: Array1D = Field(description="Coefficient vector estimate.")
