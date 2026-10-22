"""Privacy-accounted selection costs for linear regression."""

from __future__ import annotations

import numpy as np

from dpstats.algorithm import PrivacyInterpretation
from dpstats.dp_tuning.base import declares_privacy_interpretation
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D
from dpstats.domains import BoxDomain
from dpstats.budgets import (
    DPBudget,
    accounting,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _clipped_squared_errors(
    output: LinearRegressionOutput,
    metadata: LinearRegressionMetadata,
    evaluation_input: LinearRegressionInput,
) -> tuple[Array1D, float]:
    """Return squared errors of predictions clipped to the response domain.

    Clipping a prediction into the public response domain is post-processing
    of the output, and bounds every squared error by the squared
    domain width, which is returned beside them as the sum's sensitivity.
    """
    bounds = metadata.response_domain.bounding_box().bounds
    lower, upper = float(bounds[0, 0]), float(bounds[0, 1])
    width = upper - lower
    if not np.isfinite(width):
        raise InvalidRegime(
            "A clipped sum of squared errors requires a bounded response "
            "domain; the declared response domain is unbounded."
        )
    predictions = np.clip(evaluation_input.X @ output.beta, lower, upper)
    return (predictions - evaluation_input.y) ** 2, width**2


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def laplace_clipped_sum_of_squared_errors(
    output: LinearRegressionOutput,
    evaluation_metadata: LinearRegressionMetadata,
    evaluation_input: LinearRegressionInput,
    budget: DPBudget,
    seed: int,
) -> float:
    """Return clipped SSE with Laplace noise as a private selection cost.

    Predictions are clipped into the public response domain, so each squared
    error lies in ``[0, w**2]`` for the domain width ``w`` and the sum has
    add/remove sensitivity ``w**2``; an unbounded response domain is refused
    with ``InvalidRegime``.
    """
    eps = accounting.spend_as_pure(budget).eps
    squared_errors, sensitivity = _clipped_squared_errors(
        output, evaluation_metadata, evaluation_input
    )
    rng = np.random.default_rng(seed)
    return float(np.sum(squared_errors) + rng.laplace(0, sensitivity / eps))


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def gaussian_clipped_sum_of_squared_errors(
    output: LinearRegressionOutput,
    evaluation_metadata: LinearRegressionMetadata,
    evaluation_input: LinearRegressionInput,
    budget: DPBudget,
    seed: int,
) -> float:
    """Return clipped SSE with Gaussian noise as a private selection cost.

    The sum is clipped and calibrated as in
    ``laplace_clipped_sum_of_squared_errors``, with Gaussian noise.
    """
    squared_errors, sensitivity = _clipped_squared_errors(
        output, evaluation_metadata, evaluation_input
    )
    rng = np.random.default_rng(seed)
    sigma = accounting.calibrate_gaussian_scale(budget, sensitivity)
    return float(np.sum(squared_errors) + rng.normal(0, sigma))


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def unbounded_quantile_median_squared_error(
    output: LinearRegressionOutput,
    evaluation_metadata: LinearRegressionMetadata,
    evaluation_input: LinearRegressionInput,
    budget: DPBudget,
    seed: int,
    beta: float,
) -> float:
    """Return a private median squared error via the quantile mechanism."""
    from dpstats.tasks.median_estimation.algorithms import (
        UnboundedQuantileMechanism,
    )
    from dpstats.tasks.median_estimation import (
        MedianEstimationInput,
        MedianEstimationMetadata,
    )

    squared_errors: Array1D = (
        evaluation_input.X @ output.beta - evaluation_input.y
    ) ** 2
    median_output = UnboundedQuantileMechanism(beta=beta).run(
        metadata=MedianEstimationMetadata(
            n_samples=evaluation_metadata.n_samples,
            domain=BoxDomain(bounds=np.array([[0.0, np.inf]])),
        ),
        input=MedianEstimationInput(X=squared_errors),
        budget=budget,
        seed=seed,
    )
    return float(median_output.median_est)
