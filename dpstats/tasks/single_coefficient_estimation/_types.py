from __future__ import annotations

from pydantic import Field, model_validator

from typing_extensions import override

from dpstats.task import TaskInput, TaskOutput
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)


class SingleCoefficientEstimationMetadata(LinearRegressionMetadata):
    """The public shape of one single-coefficient problem.

    A linear-regression problem that additionally names which design
    coordinate's coefficient the task reports.
    """

    coefficient_of_interest_index: int = Field(
        description="Index of the coefficient to estimate."
    )

    @model_validator(mode="after")
    def _validate_coefficient_of_interest(self) -> SingleCoefficientEstimationMetadata:
        if not 0 <= self.coefficient_of_interest_index < self.dim:
            raise ValueError(
                "coefficient_of_interest_index must identify a design coordinate."
            )
        return self

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, SingleCoefficientEstimationInput):
            raise TypeError(
                f"SingleCoefficientEstimation requires a "
                f"SingleCoefficientEstimationInput, not {type(input).__name__}."
            )
        super().validate_input(input)


class SingleCoefficientEstimationInput(LinearRegressionInput):
    """One single-coefficient problem's confidential records.

    Identical to the linear-regression records it inherits.
    """


class SingleCoefficientEstimationOutput(TaskOutput):
    """One single-coefficient output."""

    coefficient: float = Field(description="Coefficient estimate.")
