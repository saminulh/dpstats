"""Exact low-dimensional BoxEM mean estimation."""

from __future__ import annotations

import math

import numpy as np

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import DPBudget, PureDPBudget, accounting
from dpstats.exceptions import InvalidRegime
from dpstats.utils import sample_index_from_log_weights
from dpstats.task import Array1D, Array2D
from dpstats.utils import split_seed
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


def _polygon_area(polygon: Array2D) -> float:
    """Return the area of an ordered planar polygon."""
    if polygon.shape[0] < 3:
        return 0.0
    translated = polygon - polygon[0]
    x = translated[:, 0]
    y = translated[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def _clip_polygon(polygon: Array2D, normal: Array1D, threshold: float) -> Array2D:
    """Intersect an ordered convex polygon with ``normal @ x <= threshold``."""
    if polygon.shape[0] == 0:
        return polygon

    clipped: list[Array1D] = []
    previous = polygon[-1]
    previous_value = normal @ previous - threshold
    previous_inside = previous_value <= 0
    for current in polygon:
        current_value = normal @ current - threshold
        current_inside = current_value <= 0
        if current_inside != previous_inside:
            fraction = previous_value / (previous_value - current_value)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    if not clipped:
        return np.empty((0, 2))
    return np.asarray(clipped)


def _one_dimensional_regions(X: Array2D, bounds: Array2D) -> list[Array2D]:
    """Construct the positive-volume scalar Tukey upper-level sets."""
    x = np.sort(X[:, 0])
    lower = bounds[0, 0]
    upper = bounds[0, 1]
    regions = [np.array([[lower], [upper]])]
    for depth in range(1, X.shape[0] // 2 + 1):
        region_lower = max(lower, x[depth - 1])
        region_upper = min(upper, x[X.shape[0] - depth])
        if region_upper <= region_lower:
            regions.append(np.empty((0, 1)))
        else:
            regions.append(np.array([[region_lower], [region_upper]]))
    return regions


def _two_dimensional_regions(X: Array2D, bounds: Array2D) -> list[Array2D]:
    """Construct planar Tukey regions by incremental half-plane intersection.

    The depth-``k`` region is the intersection of every closed half-plane that
    contains at least ``n-k+1`` observations. In two dimensions, boundary lines
    through pairs of observations suffice. Each oriented half-plane is assigned
    the first depth at which it becomes active, so each constraint is applied
    once as the regions shrink.
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    polygon = np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ]
    )
    regions = [polygon.copy()]
    n = X.shape[0]
    max_depth = n // 2
    constraints_by_depth: list[list[tuple[Array1D, float]]] = [
        [] for _ in range(max_depth + 1)
    ]

    for first in range(n - 1):
        anchor = X[first]
        for second in range(first + 1, n):
            edge = X[second] - anchor
            if edge[0] == 0 and edge[1] == 0:
                continue
            offsets = X - anchor
            cross_products = edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]

            left_depth = n - np.count_nonzero(cross_products >= 0) + 1
            if left_depth <= max_depth:
                normal = np.array([edge[1], -edge[0]])
                constraints_by_depth[left_depth].append((normal, normal @ anchor))

            right_depth = n - np.count_nonzero(cross_products <= 0) + 1
            if right_depth <= max_depth:
                normal = np.array([-edge[1], edge[0]])
                constraints_by_depth[right_depth].append((normal, normal @ anchor))

    for depth in range(1, max_depth + 1):
        for normal, threshold in constraints_by_depth[depth]:
            polygon = _clip_polygon(polygon, normal, threshold)
            if polygon.shape[0] == 0:
                break
        if polygon.shape[0] < 3 or np.linalg.matrix_rank(polygon - polygon[0]) < 2:
            polygon = np.empty((0, 2))
        regions.append(polygon.copy())
    return regions


def _compute_tukey_regions(X: Array2D, bounds: Array2D) -> list[Array2D]:
    """Construct Tukey upper-level regions inside a public box.

    Returns the depth-zero support followed by the depth-one through
    depth-``floor(n / 2)`` regions. Empty and lower-dimensional regions have
    zero Lebesgue volume and are represented by empty arrays.
    """
    if X.shape[1] == 1:
        return _one_dimensional_regions(X, bounds)
    if X.shape[1] == 2:
        return _two_dimensional_regions(X, bounds)
    raise InvalidRegime("Exact Tukey-region construction supports dimensions 1 and 2.")


def _region_volume(region: Array2D) -> float:
    """Return an interval length or planar polygon area."""
    if region.shape[1] == 1:
        return region[1, 0] - region[0, 0] if region.shape[0] == 2 else 0.0
    return _polygon_area(region)


def _boxem_log_level_weights(regions: list[Array2D], budget: PureDPBudget) -> Array1D:
    """Return log weights for the BoxEM depth-level mixture.

    Brown and Zakynthinou Equation (5), specialized to BoxEM, gives weight
    ``V_0`` at depth zero and
    ``V_k * exp(eps * k / 2) * (1 - exp(-eps / 2))`` for ``k >= 1``.
    """
    volumes = np.array([_region_volume(region) for region in regions])
    log_weights = np.full(len(regions), -math.inf)
    positive = volumes > 0
    log_weights[positive] = np.log(volumes[positive])

    if len(regions) > 1:
        rate = budget.eps / 2
        increment = math.log(-math.expm1(-rate))
        positive_depths = np.flatnonzero(positive[1:]) + 1
        log_weights[positive_depths] += rate * positive_depths + increment
    return log_weights


def _sample_polygon(polygon: Array2D, seed: int) -> Array1D:
    """Sample uniformly from a convex polygon by area-weighted triangulation."""
    rng = np.random.default_rng(seed)
    anchor = polygon[0]
    triangle_areas = np.array(
        [
            abs(
                (polygon[index, 0] - anchor[0]) * (polygon[index + 1, 1] - anchor[1])
                - (polygon[index, 1] - anchor[1]) * (polygon[index + 1, 0] - anchor[0])
            )
            / 2
            for index in range(1, polygon.shape[0] - 1)
        ]
    )
    triangle_indices = np.flatnonzero(triangle_areas > 0) + 1
    probabilities = triangle_areas[triangle_areas > 0] / np.sum(
        triangle_areas[triangle_areas > 0]
    )
    index = rng.choice(triangle_indices, p=probabilities)
    root = math.sqrt(rng.uniform())
    along = rng.uniform()
    return (
        (1 - root) * anchor
        + root * (1 - along) * polygon[index]
        + root * along * polygon[index + 1]
    )


class BoxEM(BaseAlgorithm):
    """Private Tukey Median. Privately pick a point deep inside the data, in the sense of Tukey depth, as the mean estimate.

    An implementation of Algorithm 1 in Appendix I of Liu, Kong, Kakade and Oh
    (2021), sampled by Equation (5) of Brown and Zakynthinou (2025). Samples a
    point from a known box with probability that grows exponentially with its
    Tukey depth, a measure of how central it is among the data. It is a pure-DP
    mechanism.

    Requires:
        - A public dimension of one or two.
        - At most 250 public samples in dimension two.
        - A ``BoxDomain`` or ``LpBallDomain`` with finite bounds of positive
          width.

    Implementation notes:
        - The box is the ``BoxDomain`` itself, or the coordinate box
          circumscribing an ``LpBallDomain``.

    Reference:
        - liu2021robust: Algorithm 1 in Appendix I.
        - brown2025tukey: Equation (5).
    """

    TASK = MeanEstimation

    _MAX_EXACT_DIM = 2
    _MAX_EXACT_N_2D = 250

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        bounds = metadata.domain.bounding_box().bounds
        d = metadata.dim
        n = metadata.n_samples
        if d > self._MAX_EXACT_DIM:
            raise InvalidRegime(
                "BoxEM supports exact Tukey-region construction only in "
                "dimensions 1 and 2."
            )
        if d == 2 and n > self._MAX_EXACT_N_2D:
            raise InvalidRegime(
                "BoxEM supports at most 250 public samples in dimension 2."
            )

        if not np.all(np.isfinite(bounds)):
            raise InvalidRegime("BoxEM requires finite output bounds.")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise InvalidRegime("BoxEM requires output bounds with positive width.")

        pure_budget = accounting.spend_as_pure(budget)
        regions = _compute_tukey_regions(input.X, bounds)

        depth_seed, region_seed = split_seed(seed, 2)
        if math.isinf(pure_budget.eps):
            positive_depths = [
                depth
                for depth, region in enumerate(regions)
                if _region_volume(region) > 0
            ]
            depth = positive_depths[-1]
        else:
            log_weights = _boxem_log_level_weights(regions, pure_budget)
            depth = sample_index_from_log_weights(log_weights, depth_seed)

        region = regions[depth]
        if d == 1:
            region_rng = np.random.default_rng(region_seed)
            mean_est: Array1D = np.array(
                [region_rng.uniform(region[0, 0], region[1, 0])]
            )
        else:
            mean_est = _sample_polygon(region, region_seed)
        return MeanEstimationOutput(mean_est=mean_est)
