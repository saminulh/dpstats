"""Exact-distance PTR truncated exponential mechanism for scalar medians."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.exceptions import NoRelease
from dpstats.budgets import ApproxDPBudget, DPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np

from dpstats.parameters import parameter_field
from dpstats.task import Array1D
from dpstats.utils import split_seed
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)


def _truncated_laplace_bound(budget: ApproxDPBudget) -> float:
    eps = budget.eps
    delta = budget.delta
    ratio = math.expm1(eps) / (eps * delta)
    return max(1.0, math.log(ratio) / eps)


def _sample_truncated_laplace(seed: int, budget: ApproxDPBudget, bound: float) -> float:
    """Draw from density proportional to exp(-eps |x|) on [-bound, bound]."""
    eps = budget.eps
    rng = np.random.default_rng(seed)
    u = float(rng.uniform())
    edge_mass = math.exp(-eps * bound)
    if u < 0.5:
        return math.log(edge_mass + 2 * u * (1 - edge_mass)) / eps
    return -math.log(edge_mass + 2 * (1 - u) * (1 - edge_mass)) / eps


def _lower_order(x_sorted: Array1D, rank: int) -> float:
    if rank <= 0:
        return -math.inf
    if rank > x_sorted.size:
        return math.inf
    return float(x_sorted[rank - 1])


def _upper_order(x_sorted: Array1D, rank: int) -> float:
    if rank <= 0:
        return math.inf
    if rank > x_sorted.size:
        return -math.inf
    return float(x_sorted[-rank])


def _log_score_integral(
    x_sorted: Array1D,
    t: int,
    eps: float,
    *,
    increment_left: bool,
) -> float:
    lo = _lower_order(x_sorted, t)
    hi = _upper_order(x_sorted, t)
    if lo >= hi:
        return -math.inf
    if not math.isfinite(lo) or not math.isfinite(hi):
        return math.inf

    edges: Array1D = np.unique(x_sorted[(x_sorted >= lo) & (x_sorted <= hi)])
    terms: list[float] = []
    n = x_sorted.size
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        width = float(right - left)
        if width <= 0:
            continue
        s_left = int(np.searchsorted(x_sorted, left, side="right"))
        s_right = n - int(np.searchsorted(x_sorted, right, side="left"))
        score = min(s_left + int(increment_left), s_right)
        terms.append(math.log(width) + eps * score)
    if not terms:
        return -math.inf
    result = terms[0]
    for term in terms[1:]:
        result = float(np.logaddexp(result, term))
    return result


def _unsafe_left(
    x_sorted: Array1D,
    budget: ApproxDPBudget,
    t: int,
    condition: int,
) -> bool:
    """Evaluate the paper's lower-boundary condition C1 or C3."""
    eps = budget.eps
    delta = budget.delta
    n = x_sorted.size
    if condition == 1:
        if n < 2 * (t - 1):
            return False
        outer_lo = _lower_order(x_sorted, t - 1)
        outer_hi = _upper_order(x_sorted, t - 1)
        if outer_lo > outer_hi:
            return False
        gap = _lower_order(x_sorted, t) - outer_lo
        log_integral = _log_score_integral(x_sorted, t, eps, increment_left=True)
        log_factor = math.log(delta) - math.log1p(-delta)
    else:
        if n < 2 * t:
            return False
        support_lo = _lower_order(x_sorted, t)
        support_hi = _upper_order(x_sorted, t)
        if support_lo > support_hi:
            return False
        gap = _lower_order(x_sorted, t + 1) - support_lo
        log_integral = _log_score_integral(x_sorted, t, eps, increment_left=False)
        log_factor = math.log(delta)

    if gap <= 0:
        return False
    if not math.isfinite(gap):
        return True
    if log_integral == -math.inf:
        return True
    if log_integral == math.inf:
        return False
    return math.log(gap) + eps * t > log_factor + log_integral


def _canonical_candidate(
    x_sorted: Array1D,
    t: int,
    condition: int,
    k1: int,
    k2: int,
    k3: int,
    k4: int,
) -> Array1D | None:
    """Construct a Lemma 10 canonical dataset for C1 or C3."""
    t_prime = t - 1 if condition == 1 else t
    n = x_sorted.size
    left_end = t_prime - k1
    middle_start = left_end + k3
    right_start = n - (t_prime - k2)
    middle_end = n - (t_prime - k2 + k4)
    if not (0 <= left_end <= middle_start <= middle_end <= right_start <= n):
        return None
    return np.concatenate(
        [
            np.full(k1, -np.inf),
            x_sorted[:left_end],
            x_sorted[middle_start:middle_end],
            x_sorted[right_start:],
            np.full(k2, np.inf),
        ]
    )


def _distance_to_unsafety(
    x_sorted: Array1D,
    budget: ApproxDPBudget,
    t: int,
    *,
    max_distance: int,
) -> int:
    """Compute exact Hamming distance within the radius relevant to PTR."""
    for distance in range(max_distance + 1):
        for k1 in range(distance + 1):
            for k2 in range(distance - k1 + 1):
                for k3 in range(distance - k1 - k2 + 1):
                    k4 = distance - k1 - k2 - k3
                    for condition in (1, 3):
                        candidate = _canonical_candidate(
                            x_sorted, t, condition, k1, k2, k3, k4
                        )
                        if candidate is None:
                            continue
                        if _unsafe_left(candidate, budget, t, condition):
                            return distance
                        mirrored: Array1D = np.sort(-candidate)
                        if _unsafe_left(mirrored, budget, t, condition):
                            return distance
    return max_distance + 1


def _sample_truncated_mechanism(
    x_sorted: Array1D, budget: ApproxDPBudget, t: int, seed: int
) -> float:
    eps = budget.eps
    rng = np.random.default_rng(seed)
    lo = _lower_order(x_sorted, t)
    hi = _upper_order(x_sorted, t)
    if lo > hi:
        raise NoRelease("Truncated exponential mechanism has empty support.")
    if lo == hi:
        return lo

    intervals: list[tuple[float, float]] = []
    log_weights: list[float] = []
    n = x_sorted.size
    for k in range(t, n - t + 1):
        left = float(x_sorted[k - 1])
        right = float(x_sorted[k])
        if right <= left:
            continue
        intervals.append((left, right))
        log_weights.append(math.log(right - left) + eps * min(k, n - k))
    if not intervals:
        raise RuntimeError(
            "Truncated exponential mechanism has non-singleton zero-volume support."
        )
    weights = np.exp(np.asarray(log_weights) - max(log_weights))
    weights /= weights.sum()
    selected = int(rng.choice(len(intervals), p=weights))
    left, right = intervals[selected]
    return float(rng.uniform(left, right))


class ExactDistancePTR(BaseAlgorithm):
    """Truncated Exponential Mechanism with Propose-Test-Release. Estimate a median privately after a private test that the data are far from any unstable case.

    An implementation of Algorithm 1 from Brown, Duchi, Haque and Oh (2026).
    Privately tests whether the data are far from any dataset on which the
    median would be unstable. If the test passes, samples the median with an
    exponential mechanism restricted to central enough points; otherwise
    outputs nothing. It is an approximate-DP mechanism.

    Requires:
        - A finite ``ApproxDPBudget``; an unconstrained one raises
          ``NotImplementedError``.
        - ``metadata.n_samples`` of at least two.

    Parametrization notes:
        - ``truncation_fraction``: Sets the depth threshold $t$ as this
          fraction of $n$. The paper's Proposition 2 sets $t$ from a
          distributional constant the task does not declare, so the fraction is
          configured instead; the paper prescribes no value for it.

    Reference:
        - brown2026differentially: Algorithm 1.
    """

    TASK = MedianEstimation

    truncation_fraction: float = parameter_field(
        default=0.25,
        gt=0.0,
        le=0.5,
        description="Fraction of the public metadata sample count used as the Tukey depth threshold ``t``.",
    )

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        budget = accounting.spend_as_approx_dp(budget)
        if not math.isfinite(budget.eps):
            raise NotImplementedError(
                "ExactDistancePTR does not implement the infinite-epsilon limit of its "
                "truncated exponential mechanism."
            )
        if metadata.n_samples < 2:
            raise InvalidRegime("ExactDistancePTR requires public n_samples >= 2.")

        # The halves pay only for the private comparison and the released
        # truncated mechanism; the unstable set below uses the full budget.
        ptr_budget = ApproxDPBudget(eps=budget.eps / 2, delta=budget.delta / 2)
        estimate_budget = ApproxDPBudget(eps=budget.eps / 2, delta=budget.delta / 2)
        data = input.X

        x_sorted: Array1D = np.sort(data)
        t = min(
            metadata.n_samples // 2,
            max(1, int(metadata.n_samples * self.truncation_fraction)),
        )
        bound = _truncated_laplace_bound(ptr_budget)
        max_distance = math.ceil(2 * bound)
        distance = _distance_to_unsafety(x_sorted, budget, t, max_distance=max_distance)

        threshold_seed, estimate_seed = split_seed(seed, 2)
        noisy_threshold = bound + _sample_truncated_laplace(
            threshold_seed, ptr_budget, bound
        )
        if distance <= noisy_threshold:
            raise NoRelease("PTR check failed: dataset is too close to unsafety.")

        estimate = _sample_truncated_mechanism(
            x_sorted, estimate_budget, t, estimate_seed
        )
        return MedianEstimationOutput(median_est=estimate)
