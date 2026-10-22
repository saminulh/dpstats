"""Stochastic gradient descent for M-estimation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_range
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)


class SGD(BaseAlgorithm):
    """SGD with Armijo Line-Search. Minimize the average loss by projected stochastic gradient descent, with no privacy.

    An adaptation of Algorithm 1 from Vaswani et al. (2019) to a projected
    parameter domain. Takes one step per sampled record, picking the step size
    by backtracking on that record's loss and keeping the iterate inside the
    allowed parameter range, and outputs the last iterate exactly. It adds no
    noise, so it gives no privacy guarantee; it is a reference point for the
    private methods.

    Requires:
        - An unconstrained privacy budget.

    Parametrization notes:
        - ``eta``: Corresponds to $η_{max}$ in Vaswani et al. (2019), the step
          size each backtracking search starts from. Section 3 of that paper
          reports the method is robust to it and recommends a large value;
          no source prescribes the default itself.
        - ``num_steps``: The number of stochastic steps. No source prescribes a
          value.

    Implementation notes:
        - The backtracking search shrinks the step until the projected update
          satisfies ``loss(candidate) <= loss(theta) - (c/step) * ||theta -
          candidate||**2``, in the style of Vaswani et al. (2019).
        - Iteration starts at the projection of zero onto the parameter domain;
          on an unbounded domain this is ordinary stochastic gradient descent.

    Reference:
        - vaswani2019painless: Algorithm 1.
    """

    TASK = MEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    eta: float = parameter_field(
        default=1.0,
        search=search_range(0.01, 10.0),
        description="Initial (maximum) step size for each step's stochastic line search.",
    )
    num_steps: int = parameter_field(
        default=100,
        search=search_range(1, 1000),
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
            raise InvalidRegime("SGD requires an unconstrained privacy budget.")
        rng = np.random.default_rng(seed)
        domain = metadata.parameter_domain
        theta = domain.project(np.zeros(domain.dim))
        for _ in range(self.num_steps):
            index = rng.integers(len(input.data))
            z = input.data[index : index + 1]
            grad = metadata.loss.gradient(z, theta, reduction="sum")
            if not np.all(np.isfinite(grad)):
                break
            if float(grad @ grad) == 0.0:
                continue
            sample_loss = float(metadata.loss(z, theta, reduction="sum"))
            step = self.eta
            for _ in range(self._MAX_BACKTRACKS):
                candidate = domain.project(theta - step * grad)
                displacement = theta - candidate
                displacement_sq = float(displacement @ displacement)
                if displacement_sq == 0.0:
                    break
                candidate_loss = float(metadata.loss(z, candidate, reduction="sum"))
                if np.isfinite(candidate_loss) and (
                    candidate_loss
                    <= sample_loss - (self._ARMIJO_C / step) * displacement_sq
                ):
                    theta = candidate
                    break
                step /= 2
        return MEstimationOutput(theta=theta)
