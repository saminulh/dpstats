from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.domains import (
    Domain,
    InterceptDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D


class LinearRegressionMetadata(TaskMetadata):
    """The public shape of one linear-regression problem.

    The design domain fixes the coefficient dimension and whether the problem
    carries an intercept; the response domain is scalar.
    """

    design_domain: Domain = Field(
        description="Public domain every design row lies in; its dimension is "
        "the coefficient dimension."
    )
    response_domain: Domain = Field(
        description="Public scalar domain every response lies in."
    )

    @model_validator(mode="after")
    def _validate_metadata(self) -> LinearRegressionMetadata:
        if self.response_domain.dim != 1:
            raise ValueError("LinearRegression response_domain must be scalar.")
        if self.design_domain.l2_norm_bound == 0.0:
            raise ValueError(
                "LinearRegression design_domain must have a positive norm bound; "
                "a bound of 0 makes every design row zero."
            )
        return self

    @property
    def dim(self) -> int:
        return self.design_domain.dim

    @property
    def has_intercept(self) -> bool:
        return (
            isinstance(self.design_domain, ProductDomain)
            and bool(self.design_domain.parts)
            and isinstance(self.design_domain.parts[0], InterceptDomain)
        )

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, LinearRegressionInput):
            raise TypeError(
                f"LinearRegression requires a LinearRegressionInput, not "
                f"{type(input).__name__}."
            )
        if input.X.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"LinearRegression X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and a design_domain of "
                f"dimension {self.dim}."
            )
        if input.y.shape != (self.n_samples,):
            raise ValueError(
                f"LinearRegression y has shape {input.y.shape}, but its metadata "
                f"declares n_samples={self.n_samples}."
            )
        if not np.all(self.design_domain.contains(input.X)):
            raise ValueError("LinearRegression X leaves its declared design_domain.")
        if not np.all(self.response_domain.contains(input.y[:, None])):
            raise ValueError("LinearRegression y leaves its declared response_domain.")


class LinearRegressionInput(TaskInput):
    """One linear-regression problem's confidential records."""

    X: Annotated[Array2D, FieldSymbol("X")] = Field(
        description="Matrix with `n_samples` rows, one per sample."
    )
    y: Annotated[Array1D, FieldSymbol("y")] = Field(
        description="Vector with `n_samples` entries, one response per sample."
    )


class LinearRegressionOutput(TaskOutput):
    """One linear-regression output."""

    beta: Array1D = Field(description="Coefficient vector estimate.")
