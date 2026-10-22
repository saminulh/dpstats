"""The ``SampleMean`` algorithm."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class SampleMean(BaseAlgorithm):
    """Compute the exact sample mean, with no privacy, as a reference point.

    Averages each coordinate over the records and outputs the result exactly.
    It adds no noise, so it gives no privacy guarantee; it is a reference point
    for the private methods.

    Requires:
        - An unconstrained privacy budget.
    """

    TASK = MeanEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime("SampleMean requires an unconstrained privacy budget.")
        return MeanEstimationOutput(mean_est=np.mean(input.X, axis=0))
