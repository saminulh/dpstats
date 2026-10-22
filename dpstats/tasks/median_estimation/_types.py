from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import override

from dpstats.parameters import FieldSymbol
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.domains import Domain
from dpstats.task import Array1D


class MedianEstimationMetadata(TaskMetadata):
    """The public shape of one median-estimation problem.

    The domain is scalar, and its bounds are what a mechanism calibrates
    against.
    """

    domain: Domain = Field(
        description="Public scalar domain every observation lies in."
    )

    @model_validator(mode="after")
    def _validate_metadata(self) -> MedianEstimationMetadata:
        if self.domain.dim != 1:
            raise ValueError("MedianEstimation domain must be scalar.")
        return self

    @override
    def validate_input(self, input: TaskInput) -> None:
        if not isinstance(input, MedianEstimationInput):
            raise TypeError(
                f"MedianEstimation requires a MedianEstimationInput, not "
                f"{type(input).__name__}."
            )
        if input.X.shape != (self.n_samples,):
            raise ValueError(
                f"MedianEstimation X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples}."
            )
        if not np.all(self.domain.contains(input.X[:, None])):
            raise ValueError("MedianEstimation X leaves its declared domain.")


class MedianEstimationInput(TaskInput):
    """One median-estimation problem's confidential records."""

    X: Annotated[Array1D, FieldSymbol("x")] = Field(
        description="Vector with `n_samples` entries, one per sample."
    )


class MedianEstimationOutput(TaskOutput):
    """One median-estimation output."""

    median_est: float = Field(description="Median estimate.")
