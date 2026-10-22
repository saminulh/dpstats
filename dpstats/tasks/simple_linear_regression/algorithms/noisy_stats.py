"""Pure-DP NoisyStats for simple linear regression.

The mechanism is analyzed under the neighboring relation of Alabi et al. (2022):
fixed-size replace-one, where two datasets of the same length differ on a
single record. That is not the platform's add/remove relation, and no
add/remove theorem is claimed here — the ``1 - 1/n`` sensitivity is
specifically a replace-one quantity.

The paper states its sensitivities for data in ``[r_l, r_u]`` under the
hypothesis ``0 <= r_l <= r_u``. Rather than clip in place and rescale those
constants, this module clips each field to its public interval and affinely
transports the pair onto the unit box ``[0, 1]^2``, which establishes that
hypothesis by construction for any finite declared interval and runs the
paper's own ``r_l = 0, r_u = 1`` constants. The released slope and intercept
are transported back to the source scale as deterministic post-processing.
"""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.exceptions import NoRelease
from dpstats.budgets import (
    DPBudget,
    PureDPBudget,
    accounting,
)


import numpy as np

from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms.utils import (
    line_from_unit_square,
    require_scalar_interval,
    to_unit_interval,
)
from dpstats.tasks import (
    SimpleLinearRegression,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)


class NoisyStats(BaseAlgorithm):
    """Estimate slope and intercept by perturbing centered sufficient statistics.

    An implementation of Algorithm 1 from Alabi et al. (2022). Rescales x and y
    into [0, 1] using their known bounds, adds Laplace noise to the covariance
    of x and y and to the variance of x, reports their ratio as the slope, and
    outputs the intercept with its own noise. It is a pure-DP mechanism.

    Requires:
        - A truthful bounded scalar domain of finite, positive width on each of
          x and y.

    Implementation notes:
        - The published version renames the method ``DPSuffStats`` and scales
          the same sensitivities by the data range ``[r_l, r_u]`` (Lemma 18).
          This runs arXiv v1's constants for ``r_l = 0, r_u = 1`` on data
          mapped to the unit interval, which is the same mechanism.

    Reference:
        - alabi2022differentially: Algorithm 1 and Lemma 4 (arXiv v1 numbering;
          Lemma 18 in the published version).
    """

    TASK = SimpleLinearRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.QUALIFIED
    )

    @override
    def run(
        self,
        metadata: SimpleLinearRegressionMetadata,
        input: SimpleLinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleLinearRegressionOutput:
        x_interval = require_scalar_interval(metadata.x_domain, "x")
        y_interval = require_scalar_interval(metadata.y_domain, "y")

        pure_budget = accounting.spend_as_pure(budget)
        # Three Laplace releases under basic composition, a third of eps each.
        release_budget = PureDPBudget(eps=pure_budget.eps / 3)
        n = metadata.n_samples

        x_unit: Array1D = to_unit_interval(input.x, x_interval)
        y_unit: Array1D = to_unit_interval(input.y, y_interval)

        x_mean = float(np.mean(x_unit))
        y_mean = float(np.mean(y_unit))
        n_covariance = float(np.sum((x_unit - x_mean) * (y_unit - y_mean)))
        n_variance = float(np.sum((x_unit - x_mean) ** 2))
        # Lemma 4: replacing one record of [0, 1] data moves either centered
        # statistic by at most 1 - 1/n.
        sensitivity = 1 - 1 / n
        rng = np.random.default_rng(seed)
        numerator = n_covariance + float(
            rng.laplace(0.0, sensitivity / release_budget.eps)
        )
        denominator = n_variance + float(
            rng.laplace(0.0, sensitivity / release_budget.eps)
        )
        if denominator <= 0:
            raise NoRelease("NoisyStats output bottom: noisy variance is nonpositive.")

        slope_unit = numerator / denominator
        # With the released slope fixed, replacing one record moves
        # y_mean - slope * x_mean by at most (1 + |slope|) / n.
        intercept_sensitivity = (1 + abs(slope_unit)) / n
        intercept_unit = (
            y_mean
            - slope_unit * x_mean
            + float(rng.laplace(0.0, intercept_sensitivity / release_budget.eps))
        )

        slope, intercept = line_from_unit_square(
            slope_unit, intercept_unit, x_interval, y_interval
        )
        return SimpleLinearRegressionOutput(slope=slope, intercept=intercept)
