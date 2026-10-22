"""One vector-Laplace release of a categorical mean."""

import numpy as np
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.domains import OneHotDomain
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class DPHistogram(BaseAlgorithm):
    """Estimate one categorical block's mean with Laplace noise.

    Adds noise to the mean of the complete declared one-hot block, including
    unobserved codes, by the vector Laplace mechanism (Dwork and Roth, 2014).
    It is a pure-DP mechanism.

    Requires:
        - Exactly one top-level ``OneHotDomain``.
        - A budget family admitting a pure-DP implication, unless the domain
          is a public constant.

    Reference:
        - dwork2014algorithmic: Theorem 3.6.
    """

    TASK = MeanEstimation

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if not isinstance(metadata.domain, OneHotDomain):
            raise InvalidRegime("DPHistogram requires one top-level OneHotDomain.")
        if metadata.domain.dim == 1 and not metadata.domain.allow_zero:
            return MeanEstimationOutput(mean_est=np.ones(1))
        pure_budget = accounting.spend_as_pure(budget)
        # Adding or removing a record changes sum(X) by one one-hot row, of L1
        # norm at most one, so sum(X) / n with public n has L1 sensitivity 1/n.
        scale = accounting.calibrate_laplace_scale(pure_budget, 1 / metadata.n_samples)
        mean_est = np.mean(input.X, axis=0) + np.random.default_rng(seed).laplace(
            0, scale, size=(metadata.domain.dim,)
        )
        return MeanEstimationOutput(mean_est=mean_est)
