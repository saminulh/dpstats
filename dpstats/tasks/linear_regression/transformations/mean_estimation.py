"""Transformations preparing linear regression as its response's mean estimation.

A linear regression whose design declares an intercept factors through the
intercept-only null model: regressing on the intercept alone estimates the
response mean, so the response column becomes a one-dimensional
mean-estimation problem and the output mean restores to the intercept
coordinate.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import Transformation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.task import Task
from dpstats.tasks import LinearRegression, MeanEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class ResponseMean(Transformation):
    """Estimate the intercept as the mean of the response and report every slope as zero.

    Passes only the response to a mean-estimation algorithm and reports its
    estimate as the intercept, with every other coefficient zero: the
    intercept-only model.

    Requires:
        - A declared intercept in the design.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = MeanEstimation

    @override
    def prepare(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[MeanEstimationMetadata, MeanEstimationInput, DPBudget, int, int]:
        if not metadata.has_intercept:
            raise InvalidRegime("ResponseMean requires a declared intercept column.")
        prepared_metadata = MeanEstimationMetadata(
            n_samples=metadata.n_samples, domain=metadata.response_domain
        )
        prepared_input = MeanEstimationInput(X=input.y[:, None])
        return prepared_metadata, prepared_input, budget, seed, metadata.dim

    @override
    def restore(
        self, output: MeanEstimationOutput, restoration: int
    ) -> LinearRegressionOutput:
        beta = np.zeros(restoration)
        beta[0] = float(output.mean_est[0])
        return LinearRegressionOutput(beta=beta)
