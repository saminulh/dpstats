"""Transformations preparing mean estimation for another task.

The mean is the minimizer of the squared-error loss, and a robust location
estimate is the minimizer of the Huber loss of the deviation norm. An
M-estimation algorithm solves either once the loss is fixed. Records pass
through unchanged, which is why the child's guarantee transfers verbatim.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field
from typing_extensions import override

from dpstats.domains import Domain
from dpstats.task import Array1D, Array2D, ArrayND

from dpstats.algorithm import Transformation

from typing import ClassVar

from dpstats.budgets import DPBudget
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
    Reduction,
)
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput as MInput,
    MEstimationMetadata as MMetadata,
    MEstimationOutput as MOutput,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput as MeanInput,
    MeanEstimationMetadata as MeanMetadata,
    MeanEstimationOutput as MeanOutput,
)


class SquaredErrorLoss(MEstimationLoss):
    """Squared Euclidean error between a record and the parameter."""

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        costs = np.sum((data - theta) ** 2, axis=1)
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        if reduction == "none":
            return 2 * (theta - data)
        gradient_sum = 2 * (len(data) * theta - np.sum(data, axis=0))
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(data)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        data_bounds = data_domain.bounding_box().bounds
        parameter_bounds = parameter_domain.bounding_box().bounds
        coordinate_bounds = np.maximum(
            np.abs(data_bounds[:, 0] - parameter_bounds[:, 1]),
            np.abs(data_bounds[:, 1] - parameter_bounds[:, 0]),
        )
        return LossProperties(
            convex=True,
            generalized_linear=False,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=2 * float(np.linalg.norm(coordinate_bounds)),
            gradient_lipschitz_constant=2.0,
        )


class HuberNormLoss(MEstimationLoss):
    """Huber loss of the Euclidean distance from a record to the parameter.

    The loss turns from quadratic to linear at distance
    ``threshold_per_sqrt_dim`` times the square root of the dimension.
    """

    threshold_per_sqrt_dim: float = Field(
        gt=0,
        description="Positive multiple of the square root of the dimension at "
        "which the loss turns from quadratic to linear.",
    )

    def _threshold(self, dim: int) -> float:
        return self.threshold_per_sqrt_dim * math.sqrt(dim)

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        threshold = self._threshold(data.shape[1])
        norms = np.linalg.norm(data - theta, axis=1)
        costs = 0.5 * norms**2
        outside = norms > threshold
        costs[outside] = threshold * norms[outside] - 0.5 * threshold**2
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        threshold = self._threshold(data.shape[1])
        norms = np.linalg.norm(data - theta, axis=1)
        scale = np.ones_like(norms)
        outside = norms > threshold
        scale[outside] = threshold / norms[outside]
        if reduction == "none":
            return (theta - data) * scale[:, None]
        gradient_sum = theta * np.sum(scale) - scale @ data
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(data)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        del data_domain
        return LossProperties(
            convex=True,
            generalized_linear=False,
            continuously_twice_differentiable=False,
            loss_lipschitz_constant=self._threshold(parameter_domain.dim),
            gradient_lipschitz_constant=1.0,
        )


class ToMEstimation(Transformation):
    """Estimate the mean as the point minimizing a configured loss, squared error by default.

    Passes the records unchanged to a general loss-minimization algorithm, with
    squared error as the loss unless another is configured, and reports the
    minimizing point as the mean. This spends no privacy budget.

    Parametrization notes:
        - ``loss``: The default loss is minimized at the mean, so the
          conversion alone changes no estimand.

    Implementation notes:
        - The parameter domain is set to be the smallest convex set containing
          the data domain.
    """

    OUTER_TASK: ClassVar[Task] = MeanEstimation
    INNER_TASK: ClassVar[Task] = MEstimation

    loss: MEstimationLoss = parameter_field(
        default=SquaredErrorLoss(),
        exclude_if=lambda value: value == SquaredErrorLoss(),
        description="Configured per-record location loss.",
    )

    @override
    def prepare(
        self, metadata: MeanMetadata, input: MeanInput, budget: DPBudget, seed: int
    ):
        prepared_metadata = MMetadata(
            parameter_domain=metadata.domain.convex_enclosure(),
            n_samples=metadata.n_samples,
            data_domain=metadata.domain,
            loss=self.loss,
        )
        return prepared_metadata, MInput(data=input.X), budget, seed, None

    @override
    def restore(self, output: MOutput, restoration: None):
        return MeanOutput(mean_est=output.theta)


__all__ = [
    "HuberNormLoss",
    "SquaredErrorLoss",
    "ToMEstimation",
]
