"""Non-private empirical risk minimisation for M-estimation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.optimize import minimize

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace
from dpstats.task import Array1D
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)


class ERM(BaseAlgorithm):
    """Minimize the average loss exactly, with no privacy, as a reference point.

    Minimizes the average loss over the records from a zero start and outputs
    the minimizer exactly. It adds no noise, so it gives no privacy guarantee;
    it is a reference point for the private methods.

    Requires:
        - A ``EuclideanSpace`` parameter domain.
        - An unconstrained privacy budget.
    """

    TASK = MEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    @override
    def run(
        self,
        metadata: MEstimationMetadata,
        input: MEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MEstimationOutput:
        if not isinstance(metadata.parameter_domain, EuclideanSpace):
            raise InvalidRegime(
                "ERM implements unconstrained optimization only; the "
                "declared parameter domain "
                f"{type(metadata.parameter_domain).__name__} is not supported."
            )
        if not budget.is_unconstrained:
            raise InvalidRegime("ERM requires an unconstrained privacy budget.")

        def objective(theta: Array1D) -> float:
            return float(metadata.loss(input.data, theta, reduction="mean"))

        res = minimize(objective, x0=np.zeros(metadata.parameter_domain.dim))
        return MEstimationOutput(theta=res.x)
