"""Scalar Hourglass mean estimator in the add-remove model.

Kulesza, Suresh and Wang (2024), Algorithm 3 instantiated with the hourglass mechanism of Section 4.1, Theorem 2,
Lemmas 3-4, and Theorem 3. Each scalar record is normalized from the public
interval ``[l, u]`` to ``[0, 1]``; the transformed sums ``s1 = sum(t)`` and
``s2 = sum(1 - t)`` receive one exact correlated hourglass draw
``(Z1, Z2)``, and the output is
``w * clip((s1 + Z1) / (s1 + s2 + Z1 + Z2), 0, 1) + l``.

The count is private: the estimator ignores the public reference
``metadata.n_samples`` in both the output and its calibration, exactly as
the paper treats ``|D|`` as private. Adding a normalized record perturbs
``(s1, s2)`` by ``(t, 1 - t)``; Theorem 2 proves the hourglass density's
ratio under every such shift is within ``exp(±epsilon)``, so the transcript
is pure epsilon-DP under add/remove-one, and the clipped ratio is
post-processing. Algorithm 3 omits its exact zero-denominator atom; this port
completes it by outputting the lower endpoint for a negative noisy numerator
and the upper endpoint otherwise.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np

from dpstats.task import Array1D
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


def _staircase_gamma(epsilon: float) -> float:
    """Geng-Viswanath Equation (50) optimal staircase parameter.

    With ``b = exp(-epsilon)`` the published form is
    ``-b/(1-b) + cbrt(b - 2b^2 + 2b^4 - b^5) / (cbrt(2) * (1-b)^2)``.
    The cube-root argument factors exactly as ``b * (1-b)^3 * (1+b)``, so the
    whole expression reduces to ``(cbrt(b*(1+b)/2) - b) / (1-b)`` — the same
    law with the denominator's cancellation-prone ``(1-b)`` powers removed.
    For finite
    positive epsilon, ``b*(1+b)/2`` is positive, so ``** (1/3)`` is its real
    cube root. ``1 - b`` is evaluated as ``-expm1(-epsilon)`` so small budgets
    keep full precision; the small-budget limit is ``1/2`` and the large-budget
    limit is ``0``.
    """
    b = math.exp(-epsilon)
    one_minus_b = -math.expm1(-epsilon)
    cube_root = (b * (1.0 + b) / 2.0) ** (1.0 / 3.0)
    return (cube_root - b) / one_minus_b


def _draw_hourglass(epsilon: float, gamma: float, seed: int) -> tuple[float, int]:
    """Draw ``Z1`` and the exact integer hourglass line index.

    Lemma 3: the ``Z1`` marginal is the unit-sensitivity epsilon-DP staircase
    law with parameter ``gamma`` — density proportional to
    ``exp(-epsilon * ceil(|x| - gamma))``: a central plateau on
    ``(-gamma, gamma)`` of mass ``2*gamma`` and unit-width side steps of mass
    ``2 * b**k`` for step ``k >= 1``.

    Lemma 4: conditional on ``Z1 = x``, ``Z1 + Z2`` lies on the integer
    lattice with center index ``floor(x + 1 - gamma)`` for ``x >= 0`` and
    ``-floor(-x + 1 - gamma)`` for ``x < 0``, offset by a two-sided geometric
    integer with mass ``(1-b)/(1+b) * b**|j|``. The line index
    ``K = center + J`` is computed exactly as an integer. ``Z2 = K - Z1``
    is deterministic reconstruction, so the mechanism does not construct it.
    """
    rng = np.random.default_rng(seed)
    b = math.exp(-epsilon)
    if b == 0.0 and gamma == 0.0:
        return 0.0, 0
    one_minus_b = -math.expm1(-epsilon)

    # Staircase marginal: central plateau vs geometric side step.
    central_mass = gamma
    side_mass = b / one_minus_b
    if rng.uniform(0.0, central_mass + side_mass) < central_mass:
        magnitude = rng.uniform(0.0, gamma)
    else:
        step = int(rng.geometric(one_minus_b))
        magnitude = gamma + step - 1.0 + rng.uniform(0.0, 1.0)
    z1 = magnitude if rng.uniform() < 0.5 else -magnitude

    # Lemma 4 conditional: P(J=j) = (1-b)/(1+b) * b**|j|.
    if z1 >= 0.0:
        center = math.floor(z1 + 1.0 - gamma)
    else:
        center = -math.floor(-z1 + 1.0 - gamma)
    zero_offset_probability = one_minus_b / (1.0 + b)
    if rng.uniform() < zero_offset_probability:
        line_offset = 0
    else:
        offset_magnitude = int(rng.geometric(one_minus_b))
        line_offset = offset_magnitude if rng.uniform() < 0.5 else -offset_magnitude
    line_index = center + line_offset
    return z1, line_index


def _run_algorithm_3(
    values: Array1D,
    lower: float,
    upper: float,
    epsilon: float,
    seed: int,
) -> float:
    """Algorithm 3 on the public interval with the signed-endpoint completion.

    The exact integer line index determines the zero-denominator atom. The
    paper omits that atom, so this port returns the lower endpoint for a
    negative noisy numerator and the upper endpoint otherwise.
    """
    width = upper - lower
    normalized = (values - lower) / width
    s1 = float(np.sum(normalized))

    gamma = _staircase_gamma(epsilon)
    z1, line_index = _draw_hourglass(epsilon, gamma, seed)

    noisy_count = len(values) + line_index
    noisy_sum = s1 + z1
    if noisy_count == 0:
        return lower if noisy_sum < 0.0 else upper

    ratio = noisy_sum / noisy_count
    return width * float(np.clip(ratio, 0.0, 1.0)) + lower


class Hourglass(BaseAlgorithm):
    """Hourglass Mechanism. Estimate a one-dimensional mean as a noisy sum divided by a noisy count.

    An implementation of Algorithm 3 from Kulesza, Suresh and Wang (2024).
    Rescales each value into [0, 1], forms two noisy sums, of the rescaled
    values and of one minus them, whose total is a noisy count of the records,
    and outputs their ratio, mapped back to the original range. It is a pure-DP
    mechanism.

    Requires:
        - ``metadata.dim == 1``.
        - A finite, nondegenerate public interval from the declared domain's
          bounding box.

    Parametrization notes:
        - Fixed $γ$: The staircase parameter is fixed from epsilon by Equation
          (50) of Geng and Viswanath (2016), as Kulesza, Suresh and Wang (2024)
          prescribe.

    Implementation notes:
        - The paper leaves one case undefined, and it is completed here. The
          two noisy sums add to the record count plus noise on the integer
          lattice, so the noisy count is exactly zero with positive
          probability, and Algorithm 3 defines no output there. In that case
          the lower end of the interval is output when the first noisy sum is
          negative and the upper end otherwise. This is post-processing of the
          same noisy release.
        - The method is implemented as published, treating the record count as
          private; it is deliberately not adapted to use the public sample size
          the metadata declares, which enters neither the output nor its
          calibration.

    Reference:
        - kulesza2024mean: Algorithm 3.
        - geng2016optimal: Equation (50).
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
        if metadata.dim != 1:
            raise InvalidRegime(
                f"Hourglass supports only metadata.dim == 1; got " f"{metadata.dim}."
            )
        bounds = metadata.domain.bounding_box().bounds
        lower = float(bounds[0, 0])
        upper = float(bounds[0, 1])
        if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
            raise InvalidRegime(
                "Hourglass requires a finite nondegenerate public "
                f"interval; the declared domain's bounding box gives "
                f"[{lower}, {upper}]."
            )

        epsilon = accounting.spend_as_pure(budget).eps
        estimate = _run_algorithm_3(input.X[:, 0], lower, upper, epsilon, seed)
        return MeanEstimationOutput(mean_est=np.array([estimate]))
