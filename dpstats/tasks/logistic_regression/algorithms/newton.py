"""The ``Newton`` algorithm."""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime

import numpy as np
from scipy.special import expit

from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D
from dpstats.tasks import LogisticRegression
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
    LogisticRegressionOutput,
)


def _logistic_loss(logits: Array1D, y: Array1D) -> float:
    signed_logits = (1.0 - 2.0 * y) * logits
    return float(np.mean(np.logaddexp(0.0, signed_logits)))


_MAX_BACKTRACKS = 100


class Newton(BaseAlgorithm):
    """Fit non-private logistic regression by Newton's method.

    Minimizes the logistic (cross-entropy) loss by Newton's method with an
    Armijo backtracking line search, and outputs the result exactly. It adds no
    noise, so it gives no privacy guarantee; it is a reference point for the
    private methods.

    Requires:
        - An unconstrained privacy budget.

    Parametrization notes:
        - ``tol``: The convergence tolerance on the Newton step. No source
          prescribes a value.
        - ``max_iter``: The most Newton iterations before the run fails. No
          source prescribes a value.
        - ``alpha``: The Armijo parameter, which must lie in $(0, 0.5]$. No
          source prescribes a value within that range.
        - ``beta``: The factor each backtracking step shrinks the step size
          by. No source prescribes a value.

    Implementation notes:
        - An invalid Newton decrement, an exhausted line search, or no
          convergence within ``max_iter`` iterations raises an execution error
          rather than a no-release outcome, because the estimator is undefined
          there.
    """

    TASK = LogisticRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    tol: float = parameter_field(
        default=1e-6,
        gt=0.0,
        search=search_range(1e-8, 1e-4),
        description="Convergence tolerance on the Newton step.",
    )
    max_iter: int = parameter_field(
        default=100,
        gt=0,
        search=search_range(10, 1000),
        description="Maximum number of Newton iterations.",
    )
    alpha: float = parameter_field(
        default=0.25,
        gt=0.0,
        le=0.5,
        search=search_range(0.01, 0.5),
        description="Armijo condition parameter for the backtracking line search.",
    )
    beta: float = parameter_field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        search=search_range(0.1, 0.9),
        description="Line-search step-size reduction factor.",
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
            raise InvalidRegime("Newton requires an unconstrained privacy budget.")
        X = input.X
        y = input.y
        n, d = (metadata.n_samples, metadata.dim)
        theta = np.zeros(d)
        for _ in range(self.max_iter):
            logits = X @ theta
            probs = expit(logits)
            grad = X.T @ (probs - y) / n
            hess = X.T @ (X * (probs * (1 - probs))[:, None]) / n
            delta_theta = np.linalg.solve(hess, grad)
            decrement = grad @ delta_theta
            if not np.isfinite(decrement) or decrement < 0.0:
                raise RuntimeError("Newton produced an invalid Newton decrement.")
            if np.max(np.abs(grad)) < self.tol:
                return LogisticRegressionOutput(theta=theta)

            loss = _logistic_loss(logits, y)
            step = 1.0
            for _ in range(_MAX_BACKTRACKS):
                candidate = theta - step * delta_theta
                candidate_logits = X @ candidate
                candidate_loss = _logistic_loss(candidate_logits, y)
                if candidate_loss <= loss - self.alpha * step * decrement:
                    theta = candidate
                    break
                step *= self.beta
            else:
                raise RuntimeError("Newton backtracking line search was exhausted.")
        raise RuntimeError(
            f"Newton did not converge within {self.max_iter} iterations."
        )
