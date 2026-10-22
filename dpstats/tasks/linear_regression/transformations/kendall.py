"""Kendall feature-selection mathematics used by its transformations.

Both the released selection mechanism and its exact counterpart maximize the
same greedy objective: each round scores every remaining column by its absolute
scaled Kendall statistic against the response, penalized by its mean statistic
against the already selected columns. The released mechanism reports a noisy
maximizer; the exact routine reports the maximizer itself and is therefore an
oracle rather than a private selector.
"""

# pyright: reportUnusedFunction=false
# `_select_features_privately`/`_select_features_exactly` are called only
# from the sibling `kendall_selection` transformations, not from this module.

from __future__ import annotations


import numpy as np
from scipy import stats

from dpstats.budgets import PureDPBudget
from dpstats.task import Array1D, Array2D, IntArray1D, IntArray2D
from dpstats.utils import split_seed


def _tie_broken_ranks(values: Array2D, seed: int) -> IntArray2D:
    """Rank columns after resolving exact ties with data-independent keys."""
    rng = np.random.default_rng(seed)
    n, d = values.shape
    ranks = np.empty((n, d), dtype=np.int64)
    ordinal_ranks = np.arange(n, dtype=np.int64)
    for column in range(d):
        order = np.lexsort((rng.random(n), values[:, column]))
        ranks[order, column] = ordinal_ranks
    return ranks


def _scaled_kendall_score(left: IntArray1D, right: IntArray1D) -> float:
    """Compute the paper's absolute, n/2-scaled Kendall statistic."""
    if left.size == 2:
        return 1.0
    tau = stats.kendalltau(
        left,
        right,
        method="asymptotic",
        variant="b",
        alternative="two-sided",
    ).statistic
    return abs((left.size / 2) * tau)


def _round_scores(
    label_scores: Array1D,
    correlation_sums: Array1D,
    selected: list[int],
    round_index: int,
) -> Array1D:
    """Score the remaining columns for one greedy round."""
    scores = label_scores.copy()
    if round_index > 0:
        scores -= correlation_sums / round_index
    scores[selected] = -np.inf
    return scores


def _accumulate_correlations(
    ranks: IntArray2D,
    correlation_sums: Array1D,
    selected: list[int],
    selected_index: int,
    dim: int,
) -> None:
    """Add the newly selected column's statistic to every remaining candidate."""
    selected_rank = ranks[:, selected_index]
    for candidate in range(dim):
        if candidate not in selected:
            correlation_sums[candidate] += _scaled_kendall_score(
                ranks[:, candidate], selected_rank
            )


def _select_features_privately(
    X: Array2D,
    y: Array1D,
    k: int,
    budget: PureDPBudget,
    seed: int,
) -> IntArray1D:
    """Run Algorithm 1 of Dick, Gillenwater, and Joseph.

    Each round releases one column by report-noisy-max over the greedy scores,
    with per-round Gumbel scale ``2 * sensitivity * k / eps`` for sensitivity
    ``1.5`` in the first round and ``3.0`` afterwards.
    """
    tie_seed, selection_seed = split_seed(seed, 2)
    ranks = _tie_broken_ranks(np.column_stack((X, y)), tie_seed)
    rng = np.random.default_rng(selection_seed)
    d = X.shape[1]
    label_rank = ranks[:, d]
    label_scores = np.asarray(
        [_scaled_kendall_score(ranks[:, j], label_rank) for j in range(d)]
    )
    correlation_sums = np.zeros(d)
    selected: list[int] = []

    for round_index in range(k):
        scores = _round_scores(label_scores, correlation_sums, selected, round_index)

        sensitivity = 1.5 if round_index == 0 else 3.0
        scale = 2 * sensitivity * k / budget.eps
        selection_scores = scores + scale * rng.gumbel(size=d)
        maximizers = np.flatnonzero(selection_scores == selection_scores.max())
        selected_index = int(rng.choice(maximizers))
        selected.append(selected_index)

        if round_index + 1 < k:
            _accumulate_correlations(
                ranks, correlation_sums, selected, selected_index, d
            )

    return np.asarray(selected, dtype=np.int64)


def _select_features_exactly(X: Array2D, y: Array1D, k: int, seed: int) -> IntArray1D:
    """Maximize the same greedy objective without any release mechanism.

    The result depends on the private sample and is returned unnoised, so it
    carries no privacy guarantee. ``seed`` only breaks exact rank ties.
    """
    ranks = _tie_broken_ranks(np.column_stack((X, y)), seed)
    d = X.shape[1]
    label_rank = ranks[:, d]
    label_scores = np.asarray(
        [_scaled_kendall_score(ranks[:, j], label_rank) for j in range(d)]
    )
    correlation_sums = np.zeros(d)
    selected: list[int] = []

    for round_index in range(k):
        scores = _round_scores(label_scores, correlation_sums, selected, round_index)
        selected_index = int(np.argmax(scores))
        selected.append(selected_index)

        if round_index + 1 < k:
            _accumulate_correlations(
                ranks, correlation_sums, selected, selected_index, d
            )

    return np.asarray(selected, dtype=np.int64)
