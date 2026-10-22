"""Importable configured components representing a downstream package."""

from __future__ import annotations

import math
from typing import ClassVar, assert_type

import numpy as np
from pydantic import JsonValue

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
    Transformation,
)
from dpstats.task import (
    Array1D,
    Array2D,
    ArrayND,
    Task,
    TaskInput,
    TaskMetadata,
    TaskOutput,
)
from dpstats.dp_tuning import DPExhaustiveCompose
from dpstats.budgets import DPBudget
from dpstats.domains import Domain
from dpstats.tasks.m_estimation import MEstimationLoss
from dpstats.tasks.m_estimation.losses import LossProperties


class ExternalMetadata(TaskMetadata):
    """Public metadata owned by the downstream task."""

    dimension: int

    def validate_input(self, input: TaskInput) -> None:
        input = ExternalInput.model_validate(input)
        if input.X.shape != (self.n_samples, self.dimension):
            raise ValueError("External input does not match its metadata.")


class ExternalInput(TaskInput):
    """Private inputs for the downstream task."""

    X: Array2D


class ExternalOutput(TaskOutput):
    """Output for the downstream task."""

    estimate: Array1D


EXTERNAL_TASK = Task(
    id="external_configuration_fixture",
    name="External Configuration Fixture",
    description="A task supplied by an importable downstream package.",
    metadata_cls=ExternalMetadata,
    input_cls=ExternalInput,
    output_cls=ExternalOutput,
)

assert_type(EXTERNAL_TASK, Task[ExternalMetadata, ExternalInput, ExternalOutput])


class ExternalAlgorithm(BaseAlgorithm):
    """A downstream configured algorithm with an ordinary Pydantic field."""

    TASK = EXTERNAL_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    scale: float = 1.0

    def run(
        self,
        metadata: ExternalMetadata,
        input: ExternalInput,
        budget: DPBudget,
        seed: int,
    ) -> ExternalOutput:
        del metadata, budget, seed
        return ExternalOutput(estimate=input.X.mean(axis=0) * self.scale)


def external_instance(
    *, seed: int, n_samples: int = 3, dimension: int = 2
) -> tuple[ExternalMetadata, ExternalInput, dict[str, JsonValue]]:
    """Produce direct external-task values without configured persistence."""
    input_data = ExternalInput(
        X=np.random.default_rng(seed).normal(size=(n_samples, dimension))
    )
    return ExternalMetadata(n_samples=n_samples, dimension=dimension), input_data, {}


class ExternalTransformation(Transformation):
    """A downstream identity transformation with persisted configuration."""

    OUTER_TASK: ClassVar[Task] = EXTERNAL_TASK
    INNER_TASK: ClassVar[Task] = EXTERNAL_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    label: str = "external"

    def prepare(
        self,
        metadata: ExternalMetadata,
        input: ExternalInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[ExternalMetadata, ExternalInput, DPBudget, int, str]:
        return metadata, input, budget, seed, self.label

    def restore(self, output: ExternalOutput, restoration: str) -> ExternalOutput:
        del restoration
        return output


class ExternalMEstimationLoss(MEstimationLoss):
    """A configured loss authored by a downstream package."""

    weight: float

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        scores = data @ theta
        costs = 0.5 * self.weight * scores**2
        if reduction == "none":
            return costs
        return float(np.sum(costs) if reduction == "sum" else np.mean(costs))

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        scores = data @ theta
        if reduction == "none":
            return self.weight * scores[:, None] * data
        gradient_sum = self.weight * (data.T @ scores)
        return gradient_sum if reduction == "sum" else gradient_sum / len(data)

    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        del data_domain, parameter_domain
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=math.inf,
            gradient_lipschitz_constant=math.inf,
        )


def external_selection_cost(
    output: ExternalOutput,
    *,
    evaluation_metadata: ExternalMetadata,
    evaluation_input: ExternalInput,
    budget: DPBudget,
    seed: int,
    offset: float = 0.0,
) -> float:
    """Score one external candidate for the real tuning-wrapper contract."""
    del evaluation_metadata, evaluation_input, budget, seed
    return float(np.linalg.norm(output.estimate) + offset)


class ExternalExhaustiveTuner(DPExhaustiveCompose):
    """A downstream extension of the platform's existing tuner wrapper."""
