"""The exponential-mechanism medians of Alabi et al. (2022): ``ExponentialMechanism`` and ``WideExponentialMechanism``."""

from __future__ import annotations

import math

import numpy as np
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, PureDPBudget, accounting
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)


def _release_rung_weighted_median(
    z: Array1D, budget: PureDPBudget, rng: np.random.Generator
) -> float:
    """Sample the interval law in Alabi et al. 2022 Algorithms 3 and 5.

    At an unconstrained budget, only the lowest nonempty rung retains weight.
    """
    if z[0] == z[-1]:
        return float(z[0])
    n = z.size
    rank = np.arange(1, n)
    widths = z[1:] - z[:-1]
    with np.errstate(divide="ignore"):
        log_interval_length = np.log(widths)
    dist_from_median = np.ceil(np.abs(rank - n / 2))
    if math.isinf(budget.eps):
        lowest_nonempty = dist_from_median[widths > 0].min()
        score = np.where(dist_from_median == lowest_nonempty, 0.0, -math.inf)
        score = score + log_interval_length
    else:
        score = log_interval_length - (budget.eps / 2) * dist_from_median
    noisy_score = score + rng.gumbel(size=n - 1)
    selected = int(np.argmax(noisy_score))
    left, right = z[selected], z[selected + 1]
    return float(rng.uniform(low=left, high=right))


class ExponentialMechanism(BaseAlgorithm):
    """Exponential Mechanism for Median. Select a median interval with the exponential mechanism.

    An implementation of Algorithm 3 from Alabi et al. (2022). Sorts the data,
    scores each gap between neighboring values by how far its rank is from the
    middle, samples a gap with the exponential mechanism, and outputs a
    uniformly random point inside it. It is a pure-DP mechanism.

    Requires:
        - A finite bounding box on ``metadata.domain``.

    Implementation notes:
        - For an even sample the authors' code first inserts a copy of the
          sample median while the paper pseudocode does not. This
          implementation follows the paper.

    Reference:
        - alabi2022differentially: Algorithm 3.
        - alabi2022code: Listing ``dpMedianExp``.
    """

    TASK = MedianEstimation

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        pure_budget = accounting.spend_as_pure(budget)
        bounds = metadata.domain.bounding_box().bounds
        r_l, r_u = float(bounds[0, 0]), float(bounds[0, 1])
        if not (math.isfinite(r_l) and math.isfinite(r_u)):
            raise InvalidRegime(
                "ExponentialMechanism requires a finite bounding box; the "
                f"declared domain gives [{r_l}, {r_u}]."
            )
        rng = np.random.default_rng(seed)

        z = np.sort(np.clip(input.X, r_l, r_u))
        z = np.concatenate(([r_l], z, [r_u]))
        median_est = _release_rung_weighted_median(z, pure_budget, rng)
        return MedianEstimationOutput(median_est=median_est)


class WideExponentialMechanism(BaseAlgorithm):
    """θ-Widened Exponential Mechanism for Median. Select a widened median interval with the exponential mechanism.

    An implementation of Algorithm 5 from Alabi et al. (2022). Runs the same
    interval sampling as the median exponential mechanism, but first widens
    each candidate interval around the median by a configured amount. It is a
    pure-DP mechanism.

    Requires:
        - A finite bounding box on ``metadata.domain``.

    Parametrization notes:
        - ``width``: Corresponds to $θ$ in the paper, which leaves it tunable
          and prescribes no value.

    Implementation notes:
        - The authors' code first inserts a copy of the sample median, and a
          second for an even sample, while the paper pseudocode inserts one
          only for an even sample. This implementation follows the paper.

    Reference:
        - alabi2022differentially: Algorithm 5.
        - alabi2022code: Listing ``dpMedianExpWide``.
    """

    TASK = MedianEstimation

    width: float = parameter_field(
        default=0.01,
        ge=0.0,
        search=search_range(0.0, 0.1),
        description="Non-negative interval extension around the median.",
    )

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        pure_budget = accounting.spend_as_pure(budget)
        bounds = metadata.domain.bounding_box().bounds
        r_l, r_u = float(bounds[0, 0]), float(bounds[0, 1])
        if not (math.isfinite(r_l) and math.isfinite(r_u)):
            raise InvalidRegime(
                "WideExponentialMechanism requires a finite bounding box; the "
                f"declared domain gives [{r_l}, {r_u}]."
            )
        rng = np.random.default_rng(seed)
        theta = self.width

        z = np.sort(np.clip(input.X, r_l, r_u))
        if z.size % 2 == 0:
            z = np.insert(z, z.size // 2, np.median(z))
        half = z.size // 2
        z[:half] = np.maximum(r_l, z[:half] - theta)
        z[z.size - half :] = np.minimum(z[z.size - half :] + theta, r_u)
        z = np.concatenate(([r_l], z, [r_u]))
        median_est = _release_rung_weighted_median(z, pure_budget, rng)
        return MedianEstimationOutput(median_est=median_est)
