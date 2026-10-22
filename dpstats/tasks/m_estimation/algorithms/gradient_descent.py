"""Simple gradient descent algorithm for M-estimation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)


class GD(BaseAlgorithm):
    """Minimize the average loss by projected gradient descent, with no privacy.

    Takes a fixed number of full-data gradient steps, choosing each step size
    by Armijo backtracking and keeping the iterate inside the allowed parameter
    range, and outputs the result exactly. It adds no noise, so it gives no
    privacy guarantee; it is a reference point for the private methods.

    Requires:
        - An unconstrained privacy budget.

    Parametrization notes:
        - ``eta``: The step size each backtracking search starts from. No
          source prescribes a value.
        - ``num_steps``: The number of gradient steps. No source prescribes a
          value.

    Implementation notes:
        - Iteration stops early on a non-finite or zero gradient, or when the
          backtracking search makes no progress.
    """

    TASK = MEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    eta: float = parameter_field(
        default=1.0,
        search=search_range(0.01, 10.0),
        description="Initial (maximum) step size for each step's Armijo backtracking line search.",
    )
    num_steps: int = parameter_field(
        default=10,
        search=search_range(1, 100),
        description="Number of gradient-descent iterations.",
    )

    _ARMIJO_C = 1e-4
    _MAX_BACKTRACKS = 50

    @override
    def run(
        self,
        metadata: MEstimationMetadata,
        input: MEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MEstimationOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime("GD requires an unconstrained privacy budget.")
        data = input.data

        def mean_loss(theta: Array1D) -> float:
            return float(metadata.loss(data, theta, reduction="mean"))

        domain = metadata.parameter_domain
        theta = domain.project(np.zeros(domain.dim))
        current_loss = mean_loss(theta)
        for _ in range(self.num_steps):
            grad = metadata.loss.gradient(data, theta, reduction="mean")
            if not np.all(np.isfinite(grad)):
                break
            if float(grad @ grad) == 0.0:
                break
            step = self.eta
            for _ in range(self._MAX_BACKTRACKS):
                candidate = domain.project(theta - step * grad)
                displacement = theta - candidate
                displacement_sq = float(displacement @ displacement)
                if displacement_sq == 0.0:
                    break
                candidate_loss = mean_loss(candidate)
                if np.isfinite(candidate_loss) and (
                    candidate_loss
                    <= current_loss - (self._ARMIJO_C / step) * displacement_sq
                ):
                    theta = candidate
                    current_loss = candidate_loss
                    break
                step /= 2
            else:
                break
            if displacement_sq == 0.0:
                break
        return MEstimationOutput(theta=theta)
