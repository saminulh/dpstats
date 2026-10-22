"""Reduce covariance estimation to a joint first-and-second-moment mean."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from dpstats.algorithm import Transformation
from dpstats.budgets import DPBudget
from dpstats.domains import Domain, EuclideanSpace, LpBallDomain
from dpstats.task import Task
from dpstats.tasks import CovarianceEstimation, MeanEstimation
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
    CovarianceEstimationOutput,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class CovarianceToMeanEstimation(Transformation):
    """Concatenate-and-Split. Estimate the data's first and second moments as one mean and report the covariance they imply.

    An adaptation of Algorithm 5 from Kalinin, Upadhyay and Lampert (2025) to
    one shot. Converts each record x into the vector (x, vec(xxᵀ)), passes
    these vectors to a mean-estimation algorithm, and outputs the estimated
    mean of the products minus the product of the estimated means.

    Implementation notes:
        - The paper outputs moments continually through a Gaussian matrix
          mechanism; this adaptation outputs once, hands the joint vectors to
          any mean-estimation algorithm, and gives the product entries the same
          weight as the raw entries rather than rescaling either.
        - The product entries are stored as the diagonal first, then sqrt(2)
          times each above-diagonal entry in lexicographic order, so the
          vector's Euclidean norm equals the matrix's Frobenius norm. The
          covariance is not clipped, projected onto the positive semidefinite
          cone, or debiased.
        - If the data lie in a ball of radius C around the origin, the joint
          vectors lie in a ball of radius ``hypot(C, C*C)``, which is declared
          to the mean-estimation step.
        - The joint rows take n(d^2+3d)/2 binary64 values. They are filled one
          source column at a time, without intermediate copies.

    Reference:
        - kalinin2025continual: Section 3.1, Algorithm 5.
    """

    OUTER_TASK: ClassVar[Task] = CovarianceEstimation
    INNER_TASK: ClassVar[Task] = MeanEstimation

    def prepare(
        self,
        metadata: CovarianceEstimationMetadata,
        input: CovarianceEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[MeanEstimationMetadata, MeanEstimationInput, DPBudget, int, int]:
        """Pack one joint-moment row per source record.

        Args:
            metadata: Public source shape and row domain.
            input: Source records.
            budget: Budget passed unchanged to the child.
            seed: Random seed passed unchanged to the child.

        Returns:
            Child metadata, input, budget, seed, and public source dimension.

        Raises:
            RuntimeError: Joint moments are nonfinite.

        Example:
            >>> from dpstats.budgets import NO_PRIVACY
            >>> metadata = CovarianceEstimationMetadata(
            ...     n_samples=1, domain=EuclideanSpace(dim=1)
            ... )
            >>> prepared = CovarianceToMeanEstimation().prepare(
            ...     metadata, CovarianceEstimationInput(X=[[2.0]]), NO_PRIVACY, 0
            ... )
            >>> prepared[1].X.tolist()
            [[2.0, 4.0]]
        """
        d = metadata.dim
        p = d * (d + 3) // 2
        bound = metadata.domain.l2_norm_bound
        radius = math.hypot(bound, bound * bound)
        domain: Domain = EuclideanSpace(dim=p)
        if math.isfinite(radius):
            domain = LpBallDomain.at_origin(p=2, radius=radius, dim=p)
        X = input.X
        moments = np.empty((X.shape[0], p))
        moments[:, :d] = X
        with np.errstate(over="ignore", invalid="ignore"):
            np.square(X, out=moments[:, d : 2 * d])
            # Row-major upper triangle: column i pairs with every later column.
            start = 2 * d
            for i in range(d - 1):
                block = moments[:, start : start + d - 1 - i]
                np.multiply(X[:, i : i + 1], X[:, i + 1 :], out=block)
                block *= math.sqrt(2)
                start += d - 1 - i
        if not np.isfinite(moments).all():
            raise RuntimeError("Nonfinite joint moments in covariance transformation.")
        prepared_metadata = MeanEstimationMetadata(
            n_samples=metadata.n_samples, domain=domain
        )
        return prepared_metadata, MeanEstimationInput(X=moments), budget, seed, d

    def restore(
        self, output: MeanEstimationOutput, restoration: int
    ) -> CovarianceEstimationOutput:
        """Reconstruct centered covariance from the first and second moments the child outputs.

        Args:
            output: Child mean output in the prepared coordinate order.
            restoration: Public source dimension returned by prepare.

        Returns:
            The symmetric covariance estimate, without PSD projection.

        Example:
            >>> output = MeanEstimationOutput(mean_est=[2.0, 5.0])
            >>> CovarianceToMeanEstimation().restore(output, 1).covariance_est.tolist()
            [[1.0]]
        """
        d = restoration
        mean = output.mean_est[:d]
        second_moment = np.diag(output.mean_est[d : 2 * d])
        row, col = np.triu_indices(d, k=1)
        upper = output.mean_est[2 * d :] / math.sqrt(2)
        second_moment[row, col] = upper
        second_moment[col, row] = upper
        second_moment -= np.outer(mean, mean)
        return CovarianceEstimationOutput(covariance_est=second_moment)
