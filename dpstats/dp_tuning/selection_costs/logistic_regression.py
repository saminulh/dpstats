"""Privacy-accounted selection costs for logistic regression."""

from __future__ import annotations

import numpy as np

from dpstats.algorithm import PrivacyInterpretation
from dpstats.budgets import DPBudget, accounting
from dpstats.dp_tuning.base import declares_privacy_interpretation
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
    LogisticRegressionOutput,
)


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def laplace_classification_error_rate(
    output: LogisticRegressionOutput,
    evaluation_metadata: LogisticRegressionMetadata,
    evaluation_input: LogisticRegressionInput,
    budget: DPBudget,
    seed: int,
) -> float:
    """Return a private noisy classification-error selection cost.

    The error count is divided by the declared ``n_samples``, a fixed
    reference. One added or removed record changes the count by at most one,
    so Laplace noise at scale ``1 / (n * eps)`` realizes the offered pure
    budget.
    """
    n = evaluation_metadata.n_samples
    eps = accounting.spend_as_pure(budget).eps
    error_count: int = int(
        np.count_nonzero(
            (evaluation_input.X @ output.theta >= 0).astype(int) != evaluation_input.y
        )
    )
    error_rate = float(error_count / n)
    return error_rate + float(np.random.default_rng(seed).laplace(0, 1 / (n * eps)))
