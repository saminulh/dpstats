"""Transformations preparing single-coefficient estimation for another task.

The task targets one headline coefficient of a linear regression, so a
linear-regression algorithm solves it directly: the same records, domains,
budget, and seed go to the child, and selecting one coordinate of the
output coefficient vector is post-processing.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import Transformation

from dataclasses import dataclass
from typing import ClassVar

from dpstats.budgets import DPBudget
from dpstats.task import Task
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)
from dpstats.tasks import (
    SingleCoefficientEstimation,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput as SCEInput,
    SingleCoefficientEstimationMetadata as SCEMetadata,
    SingleCoefficientEstimationOutput as SCEOutput,
)


class ToLinearRegression(Transformation):
    """Drop the coefficient-of-interest field, fit the linear regression, and report that coefficient.

    Drops the coefficient-of-interest field and passes the same data to a
    linear-regression algorithm, which estimates every coefficient. Reports the
    one at the coefficient-of-interest index.
    """

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = LinearRegression

    @dataclass(frozen=True)
    class Restoration:
        """Which output coefficient the outer task reports.

        Attributes:
            index: Coordinate of the coefficient vector this task targets.
        """

        index: int

    @override
    def prepare(
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ):
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=metadata.design_domain,
            response_domain=metadata.response_domain,
        )
        prepared_input = LRInput(X=input.X, y=input.y)
        restoration = ToLinearRegression.Restoration(
            index=metadata.coefficient_of_interest_index
        )
        return prepared_metadata, prepared_input, budget, seed, restoration

    @override
    def restore(self, output: LROutput, restoration: ToLinearRegression.Restoration):
        return SCEOutput(coefficient=float(output.beta[restoration.index]))
