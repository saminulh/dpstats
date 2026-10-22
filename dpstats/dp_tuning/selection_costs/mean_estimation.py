"""Privacy-accounted selection costs for mean estimation."""

from __future__ import annotations

import numpy as np

from dpstats.algorithm import PrivacyInterpretation
from dpstats.budgets import DPBudget, accounting
from dpstats.dp_tuning.base import declares_privacy_interpretation
from dpstats.exceptions import InvalidRegime
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def laplace_l2_error_to_sample_mean(
    output: MeanEstimationOutput,
    evaluation_metadata: MeanEstimationMetadata,
    evaluation_input: MeanEstimationInput,
    budget: DPBudget,
    seed: int,
) -> float:
    """Return a private L2-error selection cost against the evaluation mean.

    The public domain supplies its center ``c`` and centered L2 radius ``R``.
    The reference is ``c + sum(x - c) / n`` over the declared ``n``, equal to
    the mean when ``n`` records are present, so one added or removed record moves
    it by at most ``R / n`` and Laplace scale ``R / (n * eps)`` realizes the
    offered pure budget.
    """
    n = evaluation_metadata.n_samples
    center, l2_radius = evaluation_metadata.domain.l2_enclosing_ball()
    if not np.isfinite(l2_radius):
        raise InvalidRegime(
            "laplace_l2_error_to_sample_mean requires a finite enclosing radius."
        )
    reference = center + np.sum(evaluation_input.X - center, axis=0) / n
    error = float(np.linalg.norm(output.mean_est - reference))
    eps = accounting.spend_as_pure(budget).eps
    return error + float(np.random.default_rng(seed).laplace(0, l2_radius / (n * eps)))
