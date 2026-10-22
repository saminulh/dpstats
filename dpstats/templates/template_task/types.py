"""Metadata, Input, and Output value types for template_task."""

from __future__ import annotations

import numpy as np
from pydantic import model_validator

from dpstats.task import (
    Array1D,
    TaskInput,
    TaskMetadata,
    TaskOutput,
)
from dpstats.domains import Domain


class TemplateTaskMetadata(TaskMetadata):
    """Metadata for TemplateTask.

    Attributes:
        n_samples: Number of samples in the dataset.
        domain: Public domain of each scalar observation.
    """

    domain: Domain

    @model_validator(mode="after")
    def _validate_metadata(self) -> TemplateTaskMetadata:
        if self.domain.dim != 1:
            raise ValueError("TemplateTask domain must be scalar.")
        return self

    def validate_input(self, input: TaskInput) -> None:
        """Check that one input half honors this problem specification.

        Instance authoring and generation call this method on every pair they
        produce.
        """
        if not isinstance(input, TemplateTaskInput):
            raise TypeError(
                f"TemplateTask requires a TemplateTaskInput, not "
                f"{type(input).__name__}."
            )
        if input.data.shape != (self.n_samples,):
            raise ValueError(
                f"TemplateTask data has shape {input.data.shape}, but its metadata "
                f"declares n_samples={self.n_samples}."
            )
        if not np.all(self.domain.contains(input.data[:, None])):
            raise ValueError("TemplateTask data leaves its declared domain.")


class TemplateTaskInput(TaskInput):
    """Input data for TemplateTask."""

    data: Array1D


class TemplateTaskOutput(TaskOutput):
    """Output for TemplateTask."""

    mean_est: float
