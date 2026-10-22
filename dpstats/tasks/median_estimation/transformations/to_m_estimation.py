"""Transformations preparing median estimation for another task.

The median is a minimizer of the absolute-error loss, so an M-estimation
algorithm solves median estimation once that loss is fixed. Each scalar
observation becomes a one-dimensional record, which is rowwise and public.
"""

from __future__ import annotations

import math

import numpy as np

from typing_extensions import override

from dpstats.domains import Domain
from dpstats.task import Array1D, Array2D, ArrayND

from dpstats.algorithm import Transformation

from typing import ClassVar

from dpstats.budgets import DPBudget
from dpstats.task import Task
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
    Reduction,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput as MInput,
    MEstimationMetadata as MMetadata,
    MEstimationOutput as MOutput,
)
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput as MedianInput,
    MedianEstimationMetadata as MedianMetadata,
    MedianEstimationOutput as MedianOutput,
)


class AbsoluteErrorLoss(MEstimationLoss):
    """Coordinatewise absolute error between a record and the parameter."""

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        costs = np.sum(np.abs(data - theta), axis=1)
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
        gradients = np.sign(theta - data)
        if reduction == "none":
            return gradients
        if reduction == "sum":
            return np.sum(gradients, axis=0)
        if reduction == "mean":
            return np.mean(gradients, axis=0)
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
            loss_lipschitz_constant=math.sqrt(parameter_domain.dim),
            gradient_lipschitz_constant=math.inf,
        )


class ToMEstimation(Transformation):
    """Estimate the median as the point minimizing the absolute-error loss.

    Passes the values to a general loss-minimization algorithm with the
    absolute-error loss, whose minimizing point is the median. This spends no
    privacy budget.

    Implementation notes:
        - The absolute-error loss has a minimizer at a sample value, which
          lies in the declared domain, so the declared domain is the parameter
          domain and the estimand is unchanged.
    """

    OUTER_TASK: ClassVar[Task] = MedianEstimation
    INNER_TASK: ClassVar[Task] = MEstimation

    @override
    def prepare(
        self, metadata: MedianMetadata, input: MedianInput, budget: DPBudget, seed: int
    ):
        prepared_metadata = MMetadata(
            parameter_domain=metadata.domain,
            n_samples=metadata.n_samples,
            data_domain=metadata.domain,
            loss=AbsoluteErrorLoss(),
        )
        return (
            prepared_metadata,
            MInput(data=input.X.reshape(-1, 1)),
            budget,
            seed,
            None,
        )

    @override
    def restore(self, output: MOutput, restoration: None):
        return MedianOutput(median_est=float(output.theta.ravel()[0]))


__all__ = [
    "AbsoluteErrorLoss",
    "ToMEstimation",
]
