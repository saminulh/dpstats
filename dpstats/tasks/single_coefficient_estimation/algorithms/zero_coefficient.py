"""The ``ZeroCoefficient`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)


class ZeroCoefficient(BaseAlgorithm):
    """Always report zero, the floor any private estimate of the coefficient should beat.

    Ignores the data and reports zero as the coefficient of interest. It spends
    no privacy budget.
    """

    TASK = SingleCoefficientEstimation

    @override
    def run(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> SingleCoefficientEstimationOutput:
        return SingleCoefficientEstimationOutput(coefficient=0.0)
