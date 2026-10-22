"""SubLasso private feature selection as a composition transformation.

The selector resolves the block count, block size and kept length from the
public ``n_samples``, shuffles the records, keeps the balanced prefix, fits one
frozen nonprivate Lasso per contiguous block, votes each block's
``k`` largest absolute non-intercept coefficients, and privately peels ``k``
features from the vote totals. The released coordinates are public
afterwards, so the child composes sequentially with that release.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import Transformation
from dpstats.exceptions import InvalidRegime, NotApplicable

import math
from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import Lasso

from dpstats.budgets import accounting
from dpstats.budgets import DPBudget
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.task import Array1D, Array2D, IntArray1D
from dpstats.utils import split_seed
from dpstats.tasks.linear_regression.transformations._selector_utils import (
    SelectorRestoration,
    prepend_intercept,
    reduce_to_selection,
    scatter_selected,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)


def _block_vote(coefficients: Array1D, num_features: int) -> IntArray1D:
    """Vote the ``k`` largest absolute non-intercept coefficients.

    ``coefficients`` is the complete fitted vector including the penalized
    intercept coordinate at position zero, which is dropped before voting (the
    artifact's forced flatten/shape repair). Equal absolute values totalize in
    source-column order.
    """
    selectable = np.abs(coefficients[1:])
    order = np.lexsort((np.arange(selectable.size), -selectable))
    vote = np.zeros(selectable.size, dtype=np.int64)
    vote[order[:num_features]] = 1
    return vote


def _peel(votes: IntArray1D, num_features: int, epsilon: float, seed: int) -> list[int]:
    """Select ``k`` coordinates by nonmonotone Peel at sensitivity one.

    The Gumbel peeling mechanism of Durfee and Rogers (2019): each round adds
    fresh Gumbel noise at scale ``2k/epsilon`` (the general
    exponential-mechanism factor for nonmonotone sensitivity one at per-round
    budget ``epsilon/k``) and removes the winner. At infinite epsilon each
    round chooses uniformly among the exact maxima — the limit of the Gumbel
    law, not a deterministic argsort.
    """
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    available = np.ones(votes.size, dtype=bool)
    for _ in range(num_features):
        if math.isinf(epsilon):
            candidates = np.flatnonzero(available & (votes == votes[available].max()))
            winner = int(rng.choice(candidates))
        else:
            scores = votes + rng.gumbel(
                scale=2 * num_features / epsilon, size=votes.size
            )
            scores[~available] = -np.inf
            winner = int(np.argmax(scores))
        selected.append(winner)
        available[winner] = False
    return selected


def _select_features_sub_lasso(
    X: Array2D,
    y: Array1D,
    num_features: int,
    epsilon: float,
    seed: int,
    *,
    n: int,
    m_sel: int,
    q_sel: int,
) -> IntArray1D:
    """Run DGJ Algorithm 2 on the complete intercept-first design.

    ``n`` is the declared ``n_samples`` and ``m_sel``, ``q_sel`` the block
    count and block size resolved from it. Returns complete design columns in
    source order with the intercept prepended.
    """
    kept = m_sel * q_sel

    shuffle_seed, peel_seed = split_seed(seed, 2)
    permutation = np.random.default_rng(shuffle_seed).permutation(n)[:kept]

    votes = np.zeros(X.shape[1] - 1, dtype=np.int64)
    for block_index in range(m_sel):
        rows = permutation[block_index * q_sel : (block_index + 1) * q_sel]
        model: Any = Lasso(alpha=1.0, fit_intercept=False, max_iter=1000, tol=1e-4)
        # A solver failure this late is an execution failure, not the public
        # regime refusal this transformation raises before reading private data.
        try:
            model.fit(X[rows], y[rows])
            coefficients = np.asarray(model.coef_, dtype=float)
        except ValueError as error:
            raise RuntimeError(
                "SubLasso block Lasso failed to produce coefficients."
            ) from error
        if not np.all(np.isfinite(coefficients)):
            raise RuntimeError("SubLasso block Lasso produced nonfinite coefficients.")
        votes += _block_vote(coefficients, num_features)

    selected = _peel(votes, num_features, epsilon, peel_seed)
    return prepend_intercept(
        np.asarray(selected, dtype=np.int64) + 1, has_intercept=True
    )


class SubLasso(Transformation):
    """Privately select features by a vote of Lasso fits on disjoint subsets of the data, and pass only those to the next step. Upon receiving its coefficients for the selected features, fill in zero for every feature that was dropped.

    An implementation of Algorithm 2 from Dick, Gillenwater and Joseph (2023).
    Splits the records into disjoint blocks, fits a Lasso on each, lets each
    block vote for its largest coefficients, privately picks a fixed number of
    features from the vote totals, and passes only those columns to the next
    step.

    Requires:
        - A declared column-zero intercept; otherwise ``NotApplicable``.
        - A ``num_features`` within ``[1, dim - 1]``; otherwise
          ``NotApplicable``.
        - At least ``dim`` declared samples, one selector block.

    Parametrization notes:
        - ``num_features``: Corresponds to $k$ in Dick, Gillenwater and Joseph
          (2023). Set to the smaller
          of the two values the paper's evaluation uses, the one Section 4.4
          suggests as the more relevant setting.

    Implementation notes:
        - This follows the authors' code rather than Algorithm 5 of Kifer,
          Smith and Thakurta (2012), which introduced the
          Sample and Aggregate construction (Samp-Agg). That algorithm releases every
          averaged vote under Laplace noise; this one picks the top vote totals
          with the Gumbel mechanism of Durfee and Rogers (2019). The two are
          related but not interchangeable.
        - The number of blocks, ``floor(n/dim)``, is the formula the code of
          Dick, Gillenwater and Joseph (2023) executes, and the 5% selection
          share is the paper's simulation setting. The block count, block size and kept length are resolved
          from the declared ``n_samples``, never from the records received.
        - Privacy holds for replacing one slot of a fixed-length sequence.
          Adding or removing a record is covered by a construction used only in
          the proof: truncate above the declared ``n`` or pad below it with
          the public sentinel record (the domain centers), then order the slots
          at random, so neighbors differ in at most one block. On inputs of
          the declared size this is exactly the shuffle-prefix law, which is
          what runs.
        - Each block's Lasso is the authors' frozen scikit-learn setting:
          ``alpha=1.0``, ``fit_intercept=False``, ``max_iter=1000``,
          ``tol=1e-4``, cyclic order. The constant intercept column is fitted
          and penalized but never votes, as the authors' code was repaired to
          do.
        - A block Lasso that fails or returns a nonfinite coefficient raises
          ``RuntimeError``, an execution failure rather than a regime refusal.

    Reference:
        - dick2023better: Algorithm 2.
        - kifer2012private: Algorithm 5.
        - durfee2019practical: Section 4.1.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

    num_features: int = parameter_field(
        default=5,
        gt=0,
        description="Number k of non-intercept features to select.",
    )

    @override
    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ):
        if not metadata.has_intercept:
            raise NotApplicable("SubLasso requires a declared column-zero intercept.")
        selectable = metadata.dim - 1
        if not 1 <= self.num_features <= selectable:
            raise NotApplicable(
                f"num_features={self.num_features} must lie in [1, {selectable}]."
            )
        m_sel = metadata.n_samples // metadata.dim
        if m_sel < 1:
            raise InvalidRegime(
                "SubLasso requires at least one selector block: "
                f"n={metadata.n_samples} rows give floor(n/d)=0 models."
            )
        q_sel = metadata.n_samples // m_sel

        selection_budget, downstream_budget = budget.split((0.05, 0.95))
        selection_seed, downstream_seed = split_seed(seed, 2)
        selected = _select_features_sub_lasso(
            input.X,
            input.y,
            self.num_features,
            accounting.spend_as_pure(selection_budget).eps,
            selection_seed,
            n=metadata.n_samples,
            m_sel=m_sel,
            q_sel=q_sel,
        )
        reduced_metadata, reduced_input, restoration = reduce_to_selection(
            metadata, input, selected
        )
        return (
            reduced_metadata,
            reduced_input,
            downstream_budget,
            downstream_seed,
            restoration,
        )

    @override
    def restore(self, output: LROutput, restoration: SelectorRestoration):
        return scatter_selected(output, restoration)
