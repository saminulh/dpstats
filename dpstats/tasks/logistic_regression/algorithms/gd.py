"""The ``GD`` algorithm."""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime


import numpy as np
from scipy.special import expit

from dpstats.parameters import parameter_field, search_range
from dpstats.tasks import LogisticRegression
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
    LogisticRegressionOutput,
)


class GD(BaseAlgorithm):
    """Fit non-private logistic regression by full-batch gradient descent.

    Runs a fixed number of gradient steps on the logistic (cross-entropy) loss
    at a fixed learning rate, and outputs the result exactly. It adds no noise,
    so it gives no privacy guarantee; it is a reference point for the private
    methods.

    Requires:
        - An unconstrained privacy budget.

    Parametrization notes:
        - ``lr``: The fixed step size. No source prescribes a value.
        - ``max_iter``: The number of gradient steps. No source prescribes a
          value.
    """

    TASK = LogisticRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    lr: float = parameter_field(
        default=0.1,
        search=search_range(0.01, 1.0),
        description="Fixed learning rate for full-batch gradient descent.",
    )
    max_iter: int = parameter_field(
        default=100,
        search=search_range(10, 1000),
        description="Maximum number of gradient-descent iterations.",
    )

    @override
    def run(
        self,
        metadata: LogisticRegressionMetadata,
        input: LogisticRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LogisticRegressionOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime("GD requires an unconstrained privacy budget.")
        X = input.X
        y = input.y
        n, d = X.shape
        theta = np.zeros(d)
        for _ in range(self.max_iter):
            logits = X @ theta
            probs = expit(logits)
            grad = X.T @ (probs - y) / n
            theta -= self.lr * grad
        return LogisticRegressionOutput(theta=theta)
