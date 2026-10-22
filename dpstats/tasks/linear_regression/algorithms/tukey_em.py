"""The ``TukeyEM`` algorithm of Amin et al. (2023)."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    NO_PRIVACY,
    PureDPBudget,
    accounting,
)
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.parameters import parameter_field, search_values
from dpstats.task import Array1D, Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.utils import sample_index_from_log_weights, split_seed


def _fit_partition_models(
    metadata: LinearRegressionMetadata,
    input: LinearRegressionInput,
    m: int,
    seed: int,
) -> Array2D:
    """Run Algorithm 2 lines 2--4: one least-squares model per random block.

    Returns the ``m`` by ``d`` matrix whose row ``i`` is ``beta_i``.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(metadata.n_samples)
    blocks = np.array_split(order, m)
    models = np.empty((m, metadata.dim))
    ols = OLS()
    for i, rows in enumerate(blocks):
        block_metadata = LinearRegressionMetadata(
            n_samples=len(rows),
            design_domain=metadata.design_domain,
            response_domain=metadata.response_domain,
        )
        block_input = LinearRegressionInput(X=input.X[rows], y=input.y[rows])
        models[i] = ols.run(block_metadata, block_input, budget=NO_PRIVACY, seed=0).beta
    return models


def _project_and_sort(models: Array2D, seed: int) -> Array2D:
    """Run Algorithm 2 lines 5--8: the ``d`` by ``m`` matrix ``S``.

    Row ``j`` holds the models' coordinate ``j`` in nondecreasing order, after
    the small random perturbation that gives every depth region positive
    volume.
    """
    rng = np.random.default_rng(seed)
    projections = models.T
    # The remark after Lemma 3.6 perturbs the models so that no depth region
    # has zero volume; the width 1e-10 is the authors' code's.
    perturbation = rng.uniform(0.0, 1e-10, size=projections.shape)
    return np.sort(projections + perturbation, axis=1)


def _log_depth_volumes(S: Array2D) -> Array1D:
    """Run Algorithm 2 lines 9--10 in log space: ``log V_i`` for ``i = 1, ..., m/2``.

    ``V_i`` is the volume of the box of points with approximate Tukey depth at
    least ``i``, the product over dimensions of ``S[j, m - i] - S[j, i - 1]``
    (zero-based columns). Entry ``i - 1`` holds ``log V_i``; a zero-width box
    gives ``-inf``.
    """
    _, m = S.shape
    i = np.arange(1, m // 2 + 1)
    side_lengths = S[:, m - i] - S[:, i - 1]
    with np.errstate(divide="ignore"):
        return np.sum(np.log(side_lengths), axis=0)


def _distance_lower_bound(log_volumes: Array1D, t: int, budget: ApproxDPBudget) -> int:
    """Compute the Lemma 3.6 lower bound on the distance to an unsafe database.

    Finds the largest ``k`` in ``{0, ..., t - 1}`` for which some ``g > 0``
    satisfies ``V_(t-k-1) / V_(t+k+g+1) * exp(-eps g / 2) <= delta'`` with
    ``delta' = delta / (8 exp(eps))``, or ``-1`` when none does. The lemma
    makes the distance to unsafety strictly larger than that ``k``, so the
    returned bound is ``k + 1``, which is ``0`` for a database at distance
    zero. ``k = t - 1`` never satisfies the condition because ``V_0``, the
    volume of the whole space, is infinite.
    """
    eps = budget.eps
    log_delta_target = math.log(budget.delta) - math.log(8.0) - eps
    num_volumes = log_volumes.size

    def satisfied(k: int) -> bool:
        numerator_depth = t - k - 1
        first_denominator_depth = t + k + 2
        if numerator_depth < 1 or first_denominator_depth > num_volumes:
            return False
        log_denominators = log_volumes[first_denominator_depth - 1 :]
        g = np.arange(1, log_denominators.size + 1)
        with np.errstate(invalid="ignore"):
            log_ratio = (
                log_volumes[numerator_depth - 1] - log_denominators - eps * g / 2
            )
        return bool(np.any(log_ratio <= log_delta_target))

    # If (k, g) satisfies the condition, every smaller k does with some g, so
    # the largest satisfying k is found by binary search over {0, ..., t - 1}.
    largest_satisfied = -1
    first_unsatisfied = t
    while largest_satisfied + 1 < first_unsatisfied:
        k = (largest_satisfied + first_unsatisfied) // 2
        if satisfied(k):
            largest_satisfied = k
        else:
            first_unsatisfied = k
    return largest_satisfied + 1


def _ptr_check(log_volumes: Array1D, t: int, budget: ApproxDPBudget, seed: int) -> bool:
    """Run Algorithm 1: pass when the noisy distance bound clears the threshold."""
    rng = np.random.default_rng(seed)
    k = _distance_lower_bound(log_volumes, t, budget)
    noisy_k = k + rng.laplace(scale=1.0 / budget.eps)
    threshold = math.log(1.0 / (2.0 * budget.delta)) / budget.eps
    return noisy_k >= threshold


def _sample_depth(log_volumes: Array1D, t: int, budget: PureDPBudget, seed: int) -> int:
    """Run Algorithm 3 lines 2--4, extended to the deepest depth ``floor(m/2)``.

    Depth ``i >= t`` has weight ``W_i exp(eps i / 2)`` with ``W_i = V_i -
    V_(i+1)`` the volume of exact depth ``i``. The displayed loop stops at
    ``|V| - 1``; the deepest depth enters with ``V_(|V|+1) = 0``, its whole
    box being exact volume.
    """
    log_volumes_restricted = log_volumes[t - 1 :]
    log_volumes_deeper = np.append(log_volumes_restricted[1:], -np.inf)
    with np.errstate(invalid="ignore"):
        log_exact_volumes = log_volumes_restricted + np.log1p(
            -np.exp(log_volumes_deeper - log_volumes_restricted)
        )
    i = np.arange(t, t + log_volumes_restricted.size)
    log_weights = log_exact_volumes + budget.eps * i / 2
    return int(i[sample_index_from_log_weights(log_weights, seed)])


def _sample_point_with_depth(S: Array2D, i: int, seed: int) -> Array1D:
    """Run Algorithm 4: a uniform point of approximate Tukey depth exactly ``i``.

    Along dimension ``j`` the points of depth at least ``i`` fill
    ``[S[j, i - 1], S[j, m - i]]`` (zero-based columns) and those of depth
    exactly ``i`` fill its two end intervals, ``[S[j, i - 1], S[j, i])`` and
    ``(S[j, m - i - 1], S[j, m - i]]``. The region is split by the first
    dimension ``j*`` whose depth is exactly ``i``: coordinates before ``j*``
    have depth at least ``i + 1``, coordinate ``j*`` depth exactly ``i``, and
    coordinates after ``j*`` depth at least ``i``. At the deepest listed
    depth no point is deeper, so the whole box has depth exactly ``i``.
    """
    d, m = S.shape
    if i == m // 2:
        rng = np.random.default_rng(seed)
        return rng.uniform(S[:, i - 1], S[:, m - i])

    side = S[:, m - i] - S[:, i - 1]
    side_deeper = S[:, m - i - 1] - S[:, i]
    exact_side = side - side_deeper
    with np.errstate(divide="ignore"):
        log_side = np.log(side)
        log_side_deeper = np.log(side_deeper)
        log_exact_side = np.log(exact_side)
    # log V_<j,i+1: the product over dimensions before j of their deeper
    # lengths; log V_>j,i: the product over dimensions after j of their
    # lengths at depth i.
    log_volume_before = np.zeros(d)
    log_volume_before[1:] = np.cumsum(log_side_deeper)[:-1]
    log_volume_after = np.zeros(d)
    log_volume_after[:-1] = np.cumsum(log_side[::-1])[::-1][1:]
    log_component_volumes = log_volume_before + log_exact_side + log_volume_after

    child_seeds = split_seed(seed, d + 1)
    # The paper's j*, shifted down by one for zero-based indexing, like every
    # column index of S here.
    j_star = sample_index_from_log_weights(log_component_volumes, child_seeds[0])

    y = np.empty(d)
    for j in range(j_star):
        rng = np.random.default_rng(child_seeds[j + 1])
        y[j] = rng.uniform(S[j, i], S[j, m - i - 1])
    rng = np.random.default_rng(child_seeds[j_star + 1])
    left_low, left_high = S[j_star, i - 1], S[j_star, i]
    right_low, right_high = S[j_star, m - i - 1], S[j_star, m - i]
    left_width = left_high - left_low
    right_width = right_high - right_low
    if rng.uniform() < left_width / (left_width + right_width):
        y[j_star] = rng.uniform(left_low, left_high)
    else:
        y[j_star] = rng.uniform(right_low, right_high)
    for j in range(j_star + 1, d):
        rng = np.random.default_rng(child_seeds[j + 1])
        y[j] = rng.uniform(S[j, i - 1], S[j, m - i])
    return y


class TukeyEM(BaseAlgorithm):
    """Estimate linear-regression coefficients with Tukey-depth aggregation.

    An implementation of Algorithms 1-4 from Amin, Joseph, Ribero and
    Vassilvitskii (2023). Splits the records at random into disjoint groups,
    fits ordinary least squares in each, privately tests that enough of the
    group fits agree, and samples a point central among them, in the sense of
    Tukey depth. It is an approximate-DP mechanism.

    Requires:
        - A finite ``ApproxDPBudget``; an unconstrained one raises
          ``NotImplementedError``.
        - ``floor(n_samples / m) >= dim``.
        - With ``m="n_over_d"``, ``floor(n_samples / dim) >= 4``.

    Parametrization notes:
        - ``m``: Corresponds to $m$ in Amin, Joseph, Ribero and Vassilvitskii
          (2023), the number of disjoint models. Set to the value Section 4.3 reports as typically near
          optimal; the search values are the grid Section 4.3 sweeps. The
          value ``"n_over_d"`` sets $m = ⌊n/d⌋$ from the public sample size
          and dimension, the Tukey model count of Dick, Gillenwater and
          Joseph (2023), Section 4.2, after feature selection. The paper
          writes $⌊ñ/k⌋$; their code divides by the $k$ selected features plus
          the intercept, which is the reduced
          problem's dimension. Their noisy estimate of $n$ is not needed,
          since $n$ is public.
        - Fixed block guard ``floor(n_samples / m) >= dim``: It comes from the
          code of Amin, Joseph, Ribero and Vassilvitskii (2022), and the $n ≥ 1000 d$
          recommendation of Amin, Joseph, Ribero and Vassilvitskii (2023) is this guard evaluated at $m = 1000$. It
          is not exposed.
        - Fixed depth $t = ⌊m/4⌋$, the $ε/2$ split between PTR and the
          exponential mechanism, and the PTR threshold: They are fixed in the
          authors' code and are not exposed.

    Implementation notes:
        - The privacy proof is for adding or removing one record at the declared
          ``n``. It extends the balanced partition to other dataset sizes by
          truncating above ``n`` and padding with fixed placeholder records below
          it, so neighboring datasets give the same collection of models or
          collections differing in one model. On inputs of the declared size
          this is exactly the shuffle and partition executed here.
        - Because one model can change, the proof uses Algorithm 3's general
          exponential-mechanism coefficient and the swap-neighbor test of Brown
          et al. (2021), not the tighter Appendix Section 7.3 analysis for adding or
          removing a model.
        - Where Google's reference code differs from Amin, Joseph, Ribero and
          Vassilvitskii (2023), the paper is followed: the test uses Laplace scale ``1/epsilon``, exact-region
          sampling uses the paper's separate left and right depths, and the
          balanced partition keeps every record.
        - The paper perturbs the models by a small random amount so that no
          depth region has zero volume, without stating the amount; this
          implementation adds uniform noise of width ``1e-10`` to every
          projected coordinate, the amount the authors' code uses.
        - The deepest listed depth, ``floor(m/2)``, is a candidate of the depth
          sampler with its whole box as exact volume, as in the authors'
          code; displayed Algorithm 3 stops one depth short of it.
        - Each block is fit by the library's least-squares ``OLS``, which
          returns the minimum-norm solution on a rank-deficient block, as the
          authors' code's least-squares fit does; Algorithm 2 line 4 writes
          the explicit inverse.

    Reference:
        - amin2023easy: Algorithms 1-4.
        - amin2022code: Listing ``dp_tukey``, ``multiple_regressions`` and
          ``perturb_and_sort_matrix``.
        - dick2023better: Section 4.2.
        - dick2023code: Listing ``run_private_regression``.
        - brown2021covariance: Lemma 3.4.
    """

    TASK = LinearRegression

    m: Annotated[int, Field(ge=4)] | Literal["n_over_d"] = parameter_field(
        default=1000,
        search=search_values(250, 500, 750, 1000, 1250, 1500, 1750, 2000),
        description=(
            "Number of disjoint non-private OLS models aggregated by TukeyEM, "
            "or 'n_over_d' for floor(n_samples / dim)."
        ),
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        budget = accounting.spend_as_approx_dp(budget)
        if not math.isfinite(budget.eps):
            raise NotImplementedError(
                "TukeyEM does not implement the infinite-epsilon limit of its "
                "exponential-mechanism depth sampler."
            )
        if self.m == "n_over_d":
            m = metadata.n_samples // metadata.dim
            if m < 4:
                raise InvalidRegime(
                    "TukeyEM with m='n_over_d' requires floor(n_samples / dim) "
                    f">= 4; got n_samples={metadata.n_samples}, "
                    f"dim={metadata.dim}."
                )
        else:
            m = self.m
        if metadata.n_samples // m < metadata.dim:
            raise InvalidRegime(
                "TukeyEM requires floor(n_samples / m) >= dim, the authors' "
                f"block-size guard; got n_samples={metadata.n_samples}, "
                f"m={m}, dim={metadata.dim}."
            )

        ptr_budget = ApproxDPBudget(eps=budget.eps / 2, delta=budget.delta)
        sampling_budget = PureDPBudget(eps=budget.eps / 2)
        t = m // 4

        partition_seed, sort_seed, ptr_seed, depth_seed, point_seed = split_seed(
            seed, 5
        )
        models = _fit_partition_models(metadata, input, m, partition_seed)
        S = _project_and_sort(models, sort_seed)
        log_volumes = _log_depth_volumes(S)

        if not _ptr_check(log_volumes, t, ptr_budget, ptr_seed):
            raise NoRelease("TukeyEM PTR check did not permit a release.")

        depth = _sample_depth(log_volumes, t, sampling_budget, depth_seed)
        beta = _sample_point_with_depth(S, depth, point_seed)
        return LinearRegressionOutput(beta=beta)
