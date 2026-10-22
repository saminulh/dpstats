"""The canonical nonprivate centered sample-covariance baseline."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import CovarianceEstimation
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
    CovarianceEstimationOutput,
)


class SampleCovariance(BaseAlgorithm):
    """Compute the exact sample covariance matrix, dividing by n, with no privacy.

    Centers each column at its sample mean and returns the covariance matrix,
    dividing by the number of records rather than n − 1. It adds no noise, so
    it gives no privacy guarantee; it is a reference point for the private
    methods.

    Requires:
        - An unconstrained privacy budget.
    """

    TASK = CovarianceEstimation
    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    @override
    def run(
        self,
        metadata: CovarianceEstimationMetadata,
        input: CovarianceEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> CovarianceEstimationOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime(
                "SampleCovariance requires an unconstrained privacy budget."
            )
        centered = input.X - np.mean(input.X, axis=0)
        return CovarianceEstimationOutput(
            covariance_est=centered.T @ centered / metadata.n_samples
        )
