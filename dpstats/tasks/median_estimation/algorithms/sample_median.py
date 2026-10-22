"""The ``SampleMedian`` algorithm."""

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
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)


class SampleMedian(BaseAlgorithm):
    """Compute the exact sample median, with no privacy, as a reference point.

    Returns the middle value of the sorted data exactly. It adds no noise, so
    it gives no privacy guarantee; it is a reference point for the private
    methods.

    Requires:
        - An unconstrained privacy budget.
    """

    TASK = MedianEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime(
                "SampleMedian requires an unconstrained privacy budget."
            )
        return MedianEstimationOutput(median_est=float(np.median(input.X)))
