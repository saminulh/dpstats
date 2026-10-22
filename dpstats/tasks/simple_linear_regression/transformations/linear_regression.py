"""Transformations preparing simple linear regression for another task.

A simple linear regression is the two-column linear regression whose first
design coordinate is the intercept, so a linear-regression algorithm solves it
once the design is promoted. Each record maps to exactly one record, and the
output coefficient vector splits back into intercept and slope.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import Transformation

from typing import ClassVar

import numpy as np

from dpstats.domains import InterceptDomain, ProductDomain
from dpstats.budgets import DPBudget
from dpstats.task import Task
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)
from dpstats.tasks import (
    SimpleLinearRegression,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput as SLRInput,
    SimpleLinearRegressionMetadata as SLRMetadata,
    SimpleLinearRegressionOutput as SLROutput,
)


class ToLinearRegression(Transformation):
    """Fit a linear regression with an added intercept column and report its intercept and slope.

    Adds a column of ones to x and passes the data to a linear-regression
    algorithm. Reports its first coefficient as the intercept and its second as
    the slope.
    """

    OUTER_TASK: ClassVar[Task] = SimpleLinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

    @override
    def prepare(
        self, metadata: SLRMetadata, input: SLRInput, budget: DPBudget, seed: int
    ):
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=ProductDomain(parts=(InterceptDomain(), metadata.x_domain)),
            response_domain=metadata.y_domain,
        )
        prepared_input = LRInput(
            X=np.column_stack((np.ones_like(input.x), input.x)),
            y=input.y,
        )
        return prepared_metadata, prepared_input, budget, seed, None

    @override
    def restore(self, output: LROutput, restoration: None):
        return SLROutput(
            slope=float(output.beta[1]),
            intercept=float(output.beta[0]),
        )
