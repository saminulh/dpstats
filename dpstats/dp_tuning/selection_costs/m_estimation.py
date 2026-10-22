"""Privacy-accounted selection costs for M-estimation."""

from __future__ import annotations

from typing import cast

import numpy as np

from dpstats.algorithm import PrivacyInterpretation
from dpstats.domains import BoxDomain
from dpstats.dp_tuning.base import declares_privacy_interpretation
from dpstats.budgets import DPBudget, accounting
from dpstats.task import Array1D
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def laplace_clipped_empirical_loss(
    output: MEstimationOutput,
    evaluation_metadata: MEstimationMetadata,
    evaluation_input: MEstimationInput,
    budget: DPBudget,
    seed: int,
    clipping_bound: float,
) -> float:
    """Return the private clipped empirical-loss selection cost.

    Clipping each nonnegative contribution at the public bound
    ``clipping_bound`` gives the sum add/remove sensitivity
    ``clipping_bound``.
    """
    losses = evaluation_metadata.loss(
        evaluation_input.data, output.theta, reduction="none"
    )
    eps = accounting.spend_as_pure(budget).eps
    return float(
        np.sum(np.minimum(losses, clipping_bound))
        + np.random.default_rng(seed).laplace(0, clipping_bound / eps)
    )


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def unbounded_quantile_median_empirical_loss(
    output: MEstimationOutput,
    evaluation_metadata: MEstimationMetadata,
    evaluation_input: MEstimationInput,
    budget: DPBudget,
    seed: int,
    beta: float,
) -> float:
    """Return the private median empirical-loss selection cost.

    M-estimation's nonnegative loss contract places the values in the public
    domain ``[0, +inf)`` required by the unbounded quantile mechanism.
    """
    from dpstats.tasks.median_estimation.algorithms import (
        UnboundedQuantileMechanism,
    )
    from dpstats.tasks.median_estimation import (
        MedianEstimationInput,
        MedianEstimationMetadata,
    )

    # The unreduced batch contract returns an array, not the scalar union arm.
    losses = cast(
        Array1D,
        evaluation_metadata.loss(evaluation_input.data, output.theta, reduction="none"),
    )
    result = UnboundedQuantileMechanism(beta=beta).run(
        metadata=MedianEstimationMetadata(
            n_samples=evaluation_metadata.n_samples,
            domain=BoxDomain(bounds=np.array([[0.0, np.inf]])),
        ),
        input=MedianEstimationInput(X=losses),
        budget=budget,
        seed=seed,
    )
    return float(result.median_est)
