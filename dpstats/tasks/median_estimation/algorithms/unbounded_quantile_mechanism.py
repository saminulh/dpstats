"""The ``UnboundedQuantileMechanism`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm


import numpy as np

from dpstats.budgets import accounting
from dpstats.parameters import parameter_field, search_range
from dpstats.budgets import DPBudget
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)
import math

from dpstats.exceptions import InvalidRegime


class UnboundedQuantileMechanism(BaseAlgorithm):
    """Estimate a median by unbounded quantile search over a growing grid.

    An implementation of Algorithm 2 from Durfee (2023). Checks a growing
    sequence of candidate values above a known lower bound and stops at the
    first one that a noisy count says has at least half the data below it. It
    is a pure-DP mechanism.

    Requires:
        - A finite lower bound on ``metadata.domain``'s bounding box.

    Parametrization notes:
        - ``beta``: Sets how fast the candidates grow. Set to the default
          Section 6.4 recommends for computational efficiency; the experiments
          use a finer value (Section 4.1).

    Implementation notes:
        - The paper counts values below each candidate and stops once the count
          is high enough. Counting values above and stopping once the count is
          low enough is equivalent, except at a tie with a grid point, where
          this version may output the grid-aligned median instead of moving
          past it.

    Reference:
        - durfee2023unbounded: Algorithm 2.
    """

    TASK = MedianEstimation

    beta: float = parameter_field(
        default=1.01,
        gt=1.0,
        search=search_range(1.001, 1.01),
        description="Multiplicative step size for the candidate grid.",
    )

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        eps = accounting.spend_as_pure(budget).eps

        lower = float(metadata.domain.bounding_box().bounds[0, 0])
        if not math.isfinite(lower):
            raise InvalidRegime(
                "UnboundedQuantileMechanism requires a finite lower bound on the "
                f"declared domain's bounding box; it gives {lower}."
            )
        rng = np.random.default_rng(seed)
        data = input.X
        n = data.shape[0]
        T = 0.5 * n
        # Under add/remove, count - n/2 moves by at most 1/2 and not monotonically,
        # so general AboveThreshold with Lap(2/eps) charges 3eps/4, within eps.
        noise_scale = 2 / eps
        threshold_noisy = T + rng.laplace(scale=noise_scale)
        k = 0
        while True:
            estimate = lower + self.beta**k - 1
            query_value = np.sum(data > estimate)
            query_noisy = query_value + rng.laplace(scale=noise_scale)
            if query_noisy <= threshold_noisy:
                break
            k += 1
        return MedianEstimationOutput(median_est=float(estimate))
