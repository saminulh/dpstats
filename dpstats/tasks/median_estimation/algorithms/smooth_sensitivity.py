"""Bun--Steinke smooth-sensitivity mechanisms for the midpoint median."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D
from dpstats.tasks import MedianEstimation
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)
from dpstats.utils import bisect_monotone_boundary


def _midpoint_median_smooth_sensitivity(
    x: Array1D,
    beta: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Smooth-sensitivity bound of the midpoint median for adding or removing one record.

    The bound ``max_k exp(-beta * k) * A_k``, where ``A_k`` is half the widest
    span of ``k + 2`` consecutive values that straddles the middle, with the
    public bounds standing in for the values beyond the sample at either end.
    Adding or removing one record moves the midpoint median by at most half a
    gap straddling the middle, and ``k`` further changes can place the two ends
    of that gap up to ``k`` positions apart, so the bound is at least the local
    sensitivity and changes by at most a factor ``exp(beta)`` between
    neighbors: the two properties Lemma 15 of Bun and Steinke (2019) asks of a
    function standing in for the smooth sensitivity of Definition 14 (Nissim,
    Raskhodnikova and Smith, 2007).
    """
    values = np.sort(np.concatenate(([lower_bound], x, [upper_bound])))
    middle = values.size // 2
    smooth_sensitivity = 0.0
    for k in range(values.size - 1):
        first_left = max(0, middle - k - 1)
        last_left = min(middle, values.size - k - 2)
        spans = (
            values[first_left + k + 1 : last_left + k + 2]
            - values[first_left : last_left + 1]
        )
        smooth_sensitivity = max(
            smooth_sensitivity, math.exp(-beta * k) * spans.max() / 2
        )
    return float(smooth_sensitivity)


class SmoothSensitivity(BaseAlgorithm):
    """Estimate the median with noise scaled to its smooth sensitivity.

    An implementation of the method in Section 3.1.1 of Bun and Steinke (2019).
    Computes the sample median and adds noise, drawn from one of the paper's
    distributions, scaled to its smooth sensitivity, a data-dependent measure
    of how much changing a few records could move it. It is a zCDP mechanism.

    Requires:
        - A finite bounding box on ``metadata.domain``.
        - A budget family admitting a zCDP implication.
        - A ``beta`` small enough for a positive noise dilation.

    Parametrization notes:
        - ``noise``: One of the three noise laws Bun and Steinke (2019) compare. The
          default is the Laplace log-normal law, which Section 7.2 finds
          generally has the lowest variance.
        - ``beta``: Corresponds to $t$ in Bun and Steinke (2019). The paper optimizes $t$
          for each setting and gives no single value.

    Implementation notes:
        - The sensitivity is computed on the values as admitted; Listing
          ``smooth_sens_median_slow`` of the cited code first clips them to
          the bounding box. This implementation relies on the declared domain
          instead.
        - The ``uln`` law draws ``Uniform(-1, 1) * exp(sigma * Normal)`` as
          Definition 21 states; Listing ``dpMedianULN`` of the cited code
          draws the Laplace log-normal law for that mechanism.
        - The ``lln`` shape ``sigma`` is the root of the Section 3.1.1 cubic,
          which minimizes the variance; Listing ``dpMedianLLN`` of the cited
          code fixes it at ``max(2 * t / epsilon_c, 1/2)``, the upper bracket
          Bun and Steinke (2019) give for that root.

    Reference:
        - bun2019average: Definition 14, Section 3.1.1, Theorems 18, 22 and 25.
        - nissim2007smooth: Definition 2.2.
        - alabi2022code: Listing ``dpMedianLLN``, Listing ``dpMedianULN``,
          Listing ``dpMedianArsinhNormal`` and Listing ``smooth_sens_median_slow``.
    """

    TASK = MedianEstimation

    noise: Literal["lln", "uln", "arsinh_normal"] = parameter_field(
        default="lln",
        description="Admissible noise law: ``lln``, ``uln``, or ``arsinh_normal``.",
    )
    beta: float = parameter_field(
        default=0.1,
        gt=0.0,
        search=search_range(0.01, 1.0, scale="log"),
        description="Positive smoothing quantity ``t``, the selectable smoothness parameter of Bun and Steinke (2019).",
    )

    @override
    def run(
        self,
        metadata: MedianEstimationMetadata,
        input: MedianEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MedianEstimationOutput:
        bounds = metadata.domain.bounding_box().bounds
        lower_bound, upper_bound = float(bounds[0, 0]), float(bounds[0, 1])
        if not (math.isfinite(lower_bound) and math.isfinite(upper_bound)):
            raise InvalidRegime(
                "SmoothSensitivity requires a finite bounding box; the declared "
                f"domain gives [{lower_bound}, {upper_bound}]."
            )
        rho = accounting.spend_as_zcdp(budget).rho
        if math.isinf(rho):
            return MedianEstimationOutput(median_est=float(np.median(input.X)))
        # (epsilon**2 / 2)-CDP in each theorem is rho-zCDP.
        epsilon = math.sqrt(2 * rho)
        t = self.beta

        # Each law outputs f(x) + S_t(x) / s * Z. The theorem's privacy equation,
        # solved for s, gives the dilation at the offered epsilon.
        match self.noise:
            case "lln":
                # Section 3.1.1: epsilon = t / sigma + exp(3 sigma^2 / 2) s; the
                # variance-minimizing sigma is the unique root of 5 (epsilon / t) sigma^3
                # - 5 sigma^2 - 1 = 0 in (t / epsilon, max(2 t / epsilon, 1/2)).
                upper_sigma = max(2 * t / epsilon, 0.5)
                sigma = bisect_monotone_boundary(
                    lambda sigma: 5 * epsilon / t * sigma**3 - 5 * sigma**2 - 1 >= 0,
                    t / epsilon,
                    upper_sigma,
                    satisfying_endpoint="upper",
                    upper_bound=upper_sigma,
                )
                dilation = math.exp(-1.5 * sigma**2) * (epsilon - t / sigma)
                theorem = "Theorem 18"
            case "uln":
                # Theorem 22 holds for sigma >= sqrt(2); Proposition 4 gives
                # epsilon = t / sigma + exp(3 sigma^2 / 2) sqrt(2 / (pi sigma^2)) s,
                # so s = exp(-3 sigma^2 / 2) sqrt(pi sigma^2 / 2) (epsilon - t / sigma).
                sigma = math.sqrt(2)
                dilation = (
                    math.exp(-1.5 * sigma**2)
                    * math.sqrt(math.pi * sigma**2 / 2)
                    * (epsilon - t / sigma)
                )
                theorem = "Theorem 22"
            case "arsinh_normal":
                # Theorem 25 at its sigma = 2 / sqrt(3): epsilon =
                # sqrt(t (t / sigma^2 + 1 / sigma + 2)) + s (2 / (3 sigma) + sigma / 2),
                # and 1 / (2 / (3 sigma) + sigma / 2) = 6 sigma / (4 + 3 sigma^2).
                sigma = 2 / math.sqrt(3)
                dilation = (6 * sigma / (4 + 3 * sigma**2)) * (
                    epsilon - math.sqrt(t * (t / sigma**2 + 1 / sigma + 2))
                )
                theorem = "Theorem 25"

        if dilation <= 0:
            raise InvalidRegime(
                f"SmoothSensitivity({self.noise!r}) is infeasible: "
                f"Bun--Steinke {theorem} requires a positive noise dilation, "
                f"but beta={self.beta}, sigma={sigma}, and "
                f"epsilon_c={epsilon} give dilation={dilation}."
            )

        rng = np.random.default_rng(seed)
        match self.noise:
            case "lln":
                noise = rng.laplace() * math.exp(sigma * rng.normal())
            case "uln":
                noise = rng.uniform(-1, 1) * math.exp(sigma * rng.normal())
            case "arsinh_normal":
                noise = math.sinh(sigma * rng.normal()) / sigma

        median = float(np.median(input.X))
        smooth_sensitivity = _midpoint_median_smooth_sensitivity(
            input.X, t, lower_bound, upper_bound
        )
        return MedianEstimationOutput(
            median_est=median + (smooth_sensitivity / dilation) * noise
        )
