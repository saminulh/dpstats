"""Google/Wilson logarithmic-histogram ``ApproxBounds`` construction.

Wilson et al. (2020).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.budgets import DPBudget
from dpstats.budgets import accounting
from dpstats.domains import BoxDomain, Domain
from dpstats.domains.transforms import transform_numeric_blocks
from dpstats.task import Array1D, Array2D, IntArray1D
from dpstats.utils import split_seed

BoundsEstimator = Callable[[Array1D], tuple[float, float]]
DomainValues = tuple[Domain, Array2D]


def validate_approx_bounds_grid(scale: float, num_bins: int) -> None:
    """Require a finite base-two magnitude grid from public configuration."""
    try:
        final_edge = math.ldexp(scale, num_bins - 1)
    except OverflowError:
        raise ValueError(
            "ApproxBounds.num_bins produces a nonfinite grid edge."
        ) from None
    if not math.isfinite(final_edge):
        raise ValueError("ApproxBounds.num_bins produces a nonfinite grid edge.")


def _find_bounds(
    noisy_positive: Array1D,
    noisy_negative: Array1D,
    threshold: float,
    boundaries: Array1D,
) -> tuple[float, float] | None:
    """Return the outer edges of threshold-surviving signed buckets."""
    surviving_negative = np.flatnonzero(noisy_negative >= threshold)
    surviving_positive = np.flatnonzero(noisy_positive >= threshold)
    if surviving_negative.size == 0 and surviving_positive.size == 0:
        return None

    if surviving_negative.size:
        lower = -boundaries[surviving_negative[-1]]
    else:
        index = surviving_positive[0]
        lower = 0.0 if index == 0 else boundaries[index - 1]

    if surviving_positive.size:
        upper = boundaries[surviving_positive[-1]]
    else:
        index = surviving_negative[0]
        upper = 0.0 if index == 0 else -boundaries[index - 1]

    return float(lower), float(upper)


def _approx_bounds(
    values: Array1D,
    epsilon: float,
    seed: int,
    *,
    scale: float,
    num_bins: int,
) -> tuple[float, float]:
    """Run one scalar Google/Wilson ``ApproxBounds`` mechanism.

    Returns the unbounded interval when no bin survives any threshold, so the
    coordinate keeps whatever public bounds it had.
    """
    boundaries: Array1D = np.ldexp(scale, np.arange(num_bins))
    # The first magnitude bucket whose right edge contains each value.
    indices: IntArray1D = np.minimum(
        np.searchsorted(boundaries, np.abs(values), side="left"), num_bins - 1
    )
    positive = values >= 0.0
    positive_counts = np.bincount(indices[positive], minlength=num_bins).astype(float)
    negative_counts = np.bincount(indices[~positive], minlength=num_bins).astype(float)

    if math.isinf(epsilon):
        bounds = _find_bounds(
            positive_counts,
            negative_counts,
            threshold=1.0,
            boundaries=boundaries,
        )
        if bounds is not None:
            return bounds
        return -math.inf, math.inf

    noise_scale = 1.0 / epsilon
    rng = np.random.default_rng(seed)
    noisy_positive = positive_counts + rng.laplace(scale=noise_scale, size=num_bins)
    noisy_negative = negative_counts + rng.laplace(scale=noise_scale, size=num_bins)

    for failure_probability in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        per_bin_probability = (1.0 - failure_probability) ** (1.0 / (2 * num_bins))
        threshold = -noise_scale * math.log(2.0 * (1.0 - per_bin_probability))
        bounds = _find_bounds(
            noisy_positive,
            noisy_negative,
            threshold,
            boundaries,
        )
        if bounds is not None:
            return bounds

    return -math.inf, math.inf


def _coordinates_to_estimate(bounds: Array2D, *, estimate_all: bool) -> IntArray1D:
    """Return the coordinate indices the private mechanism must estimate.

    ``estimate_all`` selects every coordinate, for ``ApproxBoundsAll``'s
    comparison against already-public bounds; otherwise only a coordinate
    missing a finite public side, ``ApproxBounds``'s completion contract.
    """
    if estimate_all:
        return np.arange(bounds.shape[0])
    bounded = np.isfinite(bounds[:, 0]) & np.isfinite(bounds[:, 1])
    return np.flatnonzero(~bounded)


def _complete_numeric_bounds(
    source_domain: Domain,
    values: Array2D,
    estimate: BoundsEstimator,
    *,
    estimate_all: bool,
) -> DomainValues:
    public_bounds = source_domain.bounding_box().bounds
    targets = _coordinates_to_estimate(public_bounds, estimate_all=estimate_all)
    if targets.size == 0:
        return source_domain, values

    completed = public_bounds.copy()
    for coordinate in targets:
        private_lower, private_upper = estimate(values[:, coordinate])
        lower = max(public_bounds[coordinate, 0], private_lower)
        upper = min(public_bounds[coordinate, 1], private_upper)
        if lower > upper:
            raise NoRelease(
                "ApproxBounds did not select an interval inside the declared "
                "public domain."
            )
        completed[coordinate] = (lower, upper)

    prepared_domain = BoxDomain(bounds=completed)
    prepared_values = prepared_domain.project(values)
    return prepared_domain, prepared_values


def estimate_bounds_and_clip(
    blocks: Sequence[DomainValues],
    budget: DPBudget,
    seed: int,
    *,
    bounds_share: float,
    scale: float,
    num_bins: int,
    estimate_all: bool = False,
) -> tuple[tuple[Domain, ...], tuple[Array2D, ...], DPBudget, int]:
    """Privately complete or re-estimate numeric bounds and return compatible data.

    ``estimate_all`` estimates every numeric coordinate regardless of an
    existing public side, for ``ApproxBoundsAll``'s spend-when-public
    comparison; the default completes only a coordinate missing a finite
    public side.
    """
    required = sum(
        _coordinates_to_estimate(
            domain.bounding_box().bounds, estimate_all=estimate_all
        ).size
        for domain, _ in blocks
    )
    if required == 0:
        return (
            tuple(domain for domain, _ in blocks),
            tuple(values for _, values in blocks),
            budget,
            seed,
        )

    bounds_budget, downstream_budget = budget.split((bounds_share, 1.0 - bounds_share))
    coordinate_weights = (1.0,) * required
    direct_coordinate_budget = bounds_budget.split(coordinate_weights)[0]
    # Two pure-DP epsilons each fit the bounds budget: the coordinate's share
    # spent directly, or the one whose zCDP cost is its share of the zCDP budget.
    direct_epsilon = accounting.spend_as_pure(direct_coordinate_budget).eps
    try:
        zcdp_budget = accounting.spend_as_zcdp(bounds_budget)
    except InvalidRegime:
        coordinate_epsilon = direct_epsilon
    else:
        zcdp_coordinate_budget = zcdp_budget.split(coordinate_weights)[0]
        zcdp_epsilon = accounting.spend_as_pure(zcdp_coordinate_budget).eps
        coordinate_epsilon = max(direct_epsilon, zcdp_epsilon)
    coordinate_seeds = iter(split_seed(seed, required + 1))

    def estimate(values: Array1D) -> tuple[float, float]:
        mechanism_seed = next(coordinate_seeds)
        return _approx_bounds(
            values,
            coordinate_epsilon,
            mechanism_seed,
            scale=scale,
            num_bins=num_bins,
        )

    prepared = tuple(
        transform_numeric_blocks(
            domain,
            values,
            lambda numeric_domain, block: _complete_numeric_bounds(
                numeric_domain, block, estimate, estimate_all=estimate_all
            ),
        )
        for domain, values in blocks
    )
    downstream_seed = next(coordinate_seeds)
    return (
        tuple(domain for domain, _ in prepared),
        tuple(values for _, values in prepared),
        downstream_budget,
        downstream_seed,
    )


__all__ = ["estimate_bounds_and_clip", "validate_approx_bounds_grid"]
