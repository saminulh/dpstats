"""Compose single-coefficient estimation with a simple-linear-regression child.

Every non-coefficient-of-interest column is dropped. The composed child sees
only the coefficient-of-interest column and the response, and its output
slope is returned unchanged as the task's coefficient.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import Transformation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.task import Task
from dpstats.tasks import (
    SimpleLinearRegression,
    SingleCoefficientEstimation,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput as SCEInput,
    SingleCoefficientEstimationMetadata as SCEMetadata,
    SingleCoefficientEstimationOutput as SCEOutput,
)


class COIOnlySLR(Transformation):
    """Drop every control, fit a simple linear regression on the coefficient-of-interest column, and report its slope.

    Keeps only the coefficient-of-interest column and the response, and passes
    them to a simple-linear-regression algorithm; its slope is the estimate.
    This is the effect with no controls, which differs from the full-model
    coefficient when the controls matter.

    Requires:
        - A declared intercept.
        - A coefficient of interest other than the intercept.
    """

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SimpleLinearRegression

    @override
    def prepare(
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ) -> tuple[
        SimpleLinearRegressionMetadata,
        SimpleLinearRegressionInput,
        DPBudget,
        int,
        None,
    ]:
        if not metadata.has_intercept:
            raise InvalidRegime(
                "COIOnlySLR requires a design domain that declares an intercept."
            )
        j = metadata.coefficient_of_interest_index
        if j == 0:
            raise InvalidRegime(
                "COIOnlySLR requires coefficient_of_interest_index to name a "
                "coordinate other than the declared intercept."
            )
        x_domain = metadata.design_domain.select_coordinates(
            np.asarray([j], dtype=np.int64)
        )
        prepared_metadata = SimpleLinearRegressionMetadata(
            n_samples=metadata.n_samples,
            x_domain=x_domain,
            y_domain=metadata.response_domain,
        )
        prepared_input = SimpleLinearRegressionInput(x=input.X[:, j], y=input.y)
        return prepared_metadata, prepared_input, budget, seed, None

    @override
    def restore(
        self, output: SimpleLinearRegressionOutput, restoration: None
    ) -> SCEOutput:
        return SCEOutput(coefficient=output.slope)
