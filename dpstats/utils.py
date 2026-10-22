"""Shared utility functions with no privacy semantics of their own.

Deterministic numerical support that algorithm, accounting, and tuning
implementations compose: Euclidean clipping, monotone boundary bisection,
categorical sampling from log weights, and deterministic seed derivation.
Nothing here is a DP mechanism, spends a budget, or interprets a guarantee —
anything with privacy semantics belongs to the concern that owns those
semantics, and anything one concern owns belongs with that concern, not here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from dpstats.task import Array1D, ArrayND


def clip_rows(vectors: ArrayND, radius: float) -> ArrayND:
    """Clip each scalar sample or row vector to L2 norm ``radius``."""
    if vectors.ndim == 1:
        return np.clip(vectors, a_min=-radius, a_max=radius)
    norms = np.linalg.norm(vectors, axis=1)
    ratio = np.divide(radius, norms, out=np.ones_like(norms), where=norms > 0)
    return vectors * np.minimum(1.0, ratio)[:, None]


def bisect_monotone_boundary(
    satisfies: Callable[[float], bool],
    lower: float,
    upper: float,
    *,
    satisfying_endpoint: Literal["lower", "upper"],
    upper_bound: float | None = None,
    max_iterations: int = 200,
) -> float:
    """Bisect a monotone predicate boundary and return its satisfying endpoint."""
    if not lower < upper:
        raise ValueError("lower must be less than upper.")
    if upper_bound is not None:
        if not lower < upper_bound:
            raise ValueError("upper_bound must be greater than lower.")
        upper = min(upper, upper_bound)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    lower_satisfies = satisfies(lower)
    if satisfying_endpoint == "lower" and not lower_satisfies:
        raise ValueError("The lower endpoint must satisfy the predicate.")
    if satisfying_endpoint == "upper" and lower_satisfies:
        return lower
    upper_satisfies = satisfies(upper)
    while upper_satisfies == lower_satisfies:
        if upper_bound is not None and upper == upper_bound:
            if satisfying_endpoint == "lower":
                return upper
            raise RuntimeError("No satisfying point exists within upper_bound.")
        expanded_upper = upper * 2
        if not np.isfinite(expanded_upper) or expanded_upper <= upper:
            raise RuntimeError("Could not bracket the monotone boundary.")
        upper = (
            min(expanded_upper, upper_bound)
            if upper_bound is not None
            else expanded_upper
        )
        upper_satisfies = satisfies(upper)
    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2
        if midpoint == lower or midpoint == upper:
            break
        if satisfies(midpoint) == lower_satisfies:
            lower = midpoint
        else:
            upper = midpoint
    result = lower if satisfying_endpoint == "lower" else upper
    if not satisfies(result):
        raise RuntimeError("Bisection lost its satisfying endpoint.")
    return result


def sample_index_from_log_weights(log_weights: Array1D, seed: int) -> int:
    """Sample one categorical index from unnormalized log weights."""
    if not np.any(np.isfinite(log_weights)):
        raise RuntimeError(
            "Categorical sampling failed: no candidate has a finite log weight."
        )
    rng = np.random.default_rng(seed)
    return int(np.argmax(log_weights + rng.gumbel(size=log_weights.shape)))


def split_seed(seed: int, n: int) -> tuple[int, ...]:
    """Deterministically split one integer seed into ``n`` independent children."""
    children: list[np.random.SeedSequence] = np.random.SeedSequence(seed).spawn(n)
    seeds: list[int] = []
    for child in children:
        low_word, high_word = child.generate_state(2)
        value = int(low_word) | (int(high_word) << 32)
        seeds.append(value)
    return tuple(seeds)
