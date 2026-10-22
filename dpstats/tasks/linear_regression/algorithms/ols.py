"""The ``OLS`` algorithm."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


class OLS(BaseAlgorithm):
    """Fit ordinary least squares exactly, with no privacy, as a reference point.

    Finds the coefficients that minimize the sum of squared residuals and
    outputs them exactly. It adds no noise, so it gives no privacy guarantee;
    it is a reference point for the private methods.

    Requires:
        - An unconstrained privacy budget.
    """

    TASK = LinearRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime("OLS requires an unconstrained privacy budget.")
        beta, *_ = np.linalg.lstsq(input.X, input.y, rcond=None)
        return LinearRegressionOutput(beta=beta)
