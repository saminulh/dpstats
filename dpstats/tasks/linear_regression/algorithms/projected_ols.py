"""Sheffet's projected ordinary least squares."""

from __future__ import annotations

import math

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import ApproxDPBudget, DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _release_private_smallest_squared_singular(
    A: Array2D,
    B: float,
    projection_budget: ApproxDPBudget | None,
    rng: np.random.Generator,
) -> float:
    """Spend epsilon/2 on Algorithm 1's private squared singular value.

    The paper bounds replacement sensitivity by ``2 B^2``, so Laplace scale
    ``4 B^2 / epsilon`` realizes its stated half-budget release.
    """
    smallest_squared_singular = max(0.0, np.linalg.eigvalsh(A.T @ A)[0])
    if projection_budget is None:
        return smallest_squared_singular
    return smallest_squared_singular - rng.laplace(
        scale=4 * B**2 / projection_budget.eps
    )


def _run_algorithm_1(
    A: Array2D,
    B: float,
    r: int,
    projection_budget: ApproxDPBudget | None,
    private_smallest_squared_singular: float,
    rng: np.random.Generator,
    *,
    force_altered: bool = False,
) -> Array2D:
    """Release Algorithm 1's private Gaussian projection.

    ``force_altered`` realizes Section 6's explicit altered-``r=25`` fallback
    after its private selection step has already chosen that route.
    """
    if projection_budget is None:
        w_squared = 0.0
        altered = private_smallest_squared_singular <= 0
    else:
        epsilon = projection_budget.eps
        delta = projection_budget.delta
        log_term = math.log(8 / delta)
        w_squared = (8 * B**2 / epsilon) * (math.sqrt(2 * r * log_term) + 2 * log_term)
        altered = (
            private_smallest_squared_singular
            <= w_squared + 4 * B**2 * math.log(1 / delta) / epsilon
        )
    altered = force_altered or altered
    if altered:
        A = np.vstack([A, math.sqrt(w_squared) * np.eye(A.shape[1])])

    R = rng.normal(size=(r, A.shape[0]))
    return R @ A


def _run_section_6_experimental_recipe(
    A: Array2D,
    B: float,
    public_n: int,
    projection_budget: ApproxDPBudget | None,
    rng: np.random.Generator,
) -> Array2D:
    """Release the paper's adaptively sized Section 6 projection."""
    private_smallest_squared_singular = _release_private_smallest_squared_singular(
        A, B, projection_budget, rng
    )

    if B == 0:
        largest_unaltered_r = 0
    elif projection_budget is None:
        largest_unaltered_r = public_n if private_smallest_squared_singular > 0 else 0
    else:
        epsilon = projection_budget.eps
        delta = projection_budget.delta
        log_term = math.log(8 / delta)
        unaltered_cutoff = (
            epsilon
            * (
                private_smallest_squared_singular
                - 4 * B**2 * math.log(1 / delta) / epsilon
            )
            / (8 * B**2)
            - 2 * log_term
        )
        if unaltered_cutoff <= 0:
            largest_unaltered_r = 0
        elif unaltered_cutoff > math.sqrt(2 * public_n * log_term):
            largest_unaltered_r = public_n
        else:
            strict_upper_bound = unaltered_cutoff**2 / (2 * log_term)
            largest_unaltered_r = max(0, math.ceil(strict_upper_bound) - 1)

    uses_altered_fallback = largest_unaltered_r < 25
    r = 25 if uses_altered_fallback else largest_unaltered_r
    return _run_algorithm_1(
        A,
        B,
        r,
        projection_budget,
        private_smallest_squared_singular,
        rng,
        force_altered=uses_altered_fallback,
    )


def _resolve_problem(
    metadata: LinearRegressionMetadata,
    input: LinearRegressionInput,
    budget: DPBudget,
) -> tuple[Array2D, float, ApproxDPBudget | None]:
    """Resolve the bounded augmented matrix and the budget the projection spends.

    The budget is ``None`` for an unconstrained offer, whose releases take
    their exact infinite-epsilon limit and add no noise.
    """
    if budget.is_unconstrained:
        projection_budget = None
    elif isinstance(budget, ApproxDPBudget):
        projection_budget = budget
    else:
        raise InvalidRegime(
            "ProjectedOLS requires an ApproxDPBudget or an unconstrained "
            "privacy budget because Algorithm 1 is proved only under "
            "approximate DP."
        )

    covariate_bound = metadata.design_domain.l2_norm_bound
    response_bound = metadata.response_domain.l2_norm_bound
    if not math.isfinite(covariate_bound):
        raise InvalidRegime(
            "ProjectedOLS requires a finite covariate norm bound from "
            "the public design domain."
        )
    if not math.isfinite(response_bound):
        raise InvalidRegime(
            "ProjectedOLS requires a finite response norm bound from "
            "the public response domain."
        )
    return (
        np.column_stack((input.X, input.y)),
        math.hypot(covariate_bound, response_bound),
        projection_budget,
    )


class ProjectedOLS(BaseAlgorithm):
    """Projected Ordinary Least Squares. Fit least squares to a private random projection of the data.

    An implementation of Algorithm 1 from Sheffet (2017), sized by the
    experimental recipe of its Section 6. Privately chooses how many synthetic
    records to keep from how well-conditioned the data are, multiplies the data
    and response by a random Gaussian matrix to get that many synthetic
    records, and outputs the least-squares coefficients fitted to them. It is
    an approximate-DP mechanism.

    Requires:
        - Finite ``design_domain.l2_norm_bound`` and
          ``response_domain.l2_norm_bound``.
        - An ``ApproxDPBudget`` or an unconstrained budget.

    Implementation notes:
        - One private release of the smallest squared singular value both picks
          the number of synthetic records, the largest whole number up to
          ``n_samples`` that strictly passes the test, and decides Algorithm 1's
          conditioning branch, so that value is paid for once. Below 25
          records, the branch that alters the input with ``r = 25`` is taken.
        - Algorithm 1 also outputs whether the input was altered; that marker
          is discarded. The fit is the minimum-norm least-squares solution,
          which equals the paper's inverse formula when the projected design
          has full column rank.

    Reference:
        - sheffet2017differentially: Algorithm 1 and Section 6.
    """

    TASK = LinearRegression

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        A, B, projection_budget = _resolve_problem(metadata, input, budget)
        projected_A = _run_section_6_experimental_recipe(
            A,
            B,
            metadata.n_samples,
            projection_budget,
            np.random.default_rng(seed),
        )
        beta, *_ = np.linalg.lstsq(projected_A[:, :-1], projected_A[:, -1], rcond=None)
        return LinearRegressionOutput(beta=beta)
