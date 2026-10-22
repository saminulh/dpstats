"""The ``GaussianMechanism`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime


import numpy as np

from dpstats.task import Array1D
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class GaussianMechanism(BaseAlgorithm):
    """Analytic Gaussian Mechanism. Output the empirical mean under the analytic Gaussian mechanism.

    An implementation of Theorem 8 from Balle and Wang (2018).
    Computes the sample mean and adds Gaussian noise. It is a Gaussian-DP
    mechanism.

    Requires:
        - A finite L2 enclosing radius on ``metadata.domain``.
        - A budget family admitting a GDP implication.

    Reference:
        - balle2018improving: Theorem 8.
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
        rng = np.random.default_rng(seed)

        n = metadata.n_samples

        _, radius = metadata.domain.l2_enclosing_ball()
        if not np.isfinite(radius):
            raise InvalidRegime("GaussianMechanism requires a finite enclosing radius.")
        # Under add/remove, the mean c + sum(X - c) / n with public n moves by at
        # most radius / n when one record is added or removed.
        l2_sens = radius / n

        sigma = accounting.calibrate_gaussian_scale(budget, l2_sens)
        mean_est_nonprivate: Array1D = np.mean(input.X, axis=0)
        mean_est = mean_est_nonprivate + rng.normal(
            0, sigma, size=mean_est_nonprivate.shape
        )
        return MeanEstimationOutput(mean_est=mean_est)
