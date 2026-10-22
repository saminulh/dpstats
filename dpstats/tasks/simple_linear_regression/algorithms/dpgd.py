"""Alabi et al.'s private gradient descent for simple linear regression."""

from __future__ import annotations


import math

import numpy as np
from pydantic import model_validator

from typing_extensions import override

from dpstats.budgets import accounting
from dpstats.algorithm import BaseAlgorithm
from dpstats.parameters import parameter_field, search_range, search_values
from dpstats.budgets import DPBudget, PureDPBudget, ZCDPBudget
from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms.utils import (
    line_from_quartile_predictions,
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


def _clipped_gradient_sum(
    x: Array1D, y: Array1D, predictions: Array1D, tau: float
) -> Array1D:
    """Compute Algorithm 7's coordinatewise-clipped full-batch gradient sum."""
    basis = 2 * np.column_stack((0.75 - x, x - 0.25))
    y_hat = basis @ predictions
    per_example = (y_hat - y)[:, np.newaxis] * basis
    gradient_sum: Array1D = np.sum(
        np.clip(per_example, -tau, tau),
        axis=0,
    )
    return gradient_sum


class DPGD(BaseAlgorithm):
    """DPGradDescent. Fit simple linear regression by private gradient descent.

    An implementation of Algorithms 6 and 7 from Alabi et al. (2022). Runs
    gradient descent on the line's predicted responses at x = 0.25 and x = 0.75
    rather than on the slope and intercept directly, at each full-data step
    clipping every record's gradients and adding noise, and outputs the slope
    and intercept that the final two predictions give. It is a pure-DP or zCDP
    mechanism, matching the offered budget.

    Requires:
        - A bounded scalar domain of positive width on each of x and y.
        - A ``PureDPBudget`` or a budget family admitting a zCDP implication.

    Parametrization notes:
        - ``tau``: Corresponds to $τ$ in the paper. Set as in the paper's
          evaluation in Appendices C.2 and E.
        - ``num_steps``: Corresponds to $T$ in the paper. Set as in the paper's
          evaluation in Appendices C.2 and E; the search values are even.

    Implementation notes:
        - Each variable is mapped to ``[0, 1]`` from its declared interval, the
          algorithm runs there, and the slope and intercept are mapped back.
          The clip therefore acts in normalized units.
        - The step size is the reciprocal square root of the running sum of
          squared noisy-gradient norms, and the output averages the second
          half of the iterates taken after each update. Algorithm 7 projects no
          iterate, and neither does this.
        - The averaging window follows the authors' code; Algorithms 6 and 7
          average the iterates from before each update instead.
        - The offered budget picks the paper's version: a ``PureDPBudget`` runs
          Algorithm 6 with Laplace noise, and any other family runs Algorithm 7
          with Gaussian noise.

    Reference:
        - alabi2022differentially: Algorithms 6 and 7 (arXiv v1 numbering; outlined as
          Algorithm 7 in the published version).
    """

    TASK = SimpleLinearRegression

    tau: float = parameter_field(
        default=1.0,
        gt=0.0,
        search=search_range(0.01, 10.0, scale="log"),
        description="Coordinatewise per-example gradient clipping threshold.",
    )
    num_steps: int = parameter_field(
        default=80,
        gt=0,
        search=search_values(20, 40, 80, 160),
        description="Even number of full-batch noisy gradient steps.",
    )

    @model_validator(mode="after")
    def _validate_even_num_steps(self) -> DPGD:
        if self.num_steps % 2 != 0:
            raise ValueError("num_steps must be even.")
        return self

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

        # The authors' replication code fixes the normalized predictions at
        # the midpoint; it does not expose initialization as a tuning choice.
        predictions: Array1D = np.array([0.5, 0.5])
        iterates: list[Array1D] = []
        cumulative_squared_norm = 0.0
        use_laplace = isinstance(budget, PureDPBudget)
        if use_laplace:
            pure_budget = accounting.spend_as_pure(budget)
            per_step_pure_budget = PureDPBudget(eps=pure_budget.eps / self.num_steps)
            # Adding or removing a record moves the sum by its two gradients, each
            # clipped to [-tau, tau]: L1 sensitivity 2 * tau (the paper's 4 * tau
            # is for replacing one), at eps / T per step.
            noise_scale = 2.0 * self.tau / per_step_pure_budget.eps
        else:
            zcdp_budget = accounting.spend_as_zcdp(budget)
            per_step_zcdp_budget = ZCDPBudget(rho=zcdp_budget.rho / self.num_steps)
            # The same two clipped gradients give L2 sensitivity sqrt(2) * tau
            # (2 * sqrt(2) * tau for replacing one), at rho / T per step.
            noise_scale = accounting.calibrate_gaussian_scale(
                per_step_zcdp_budget,
                math.sqrt(2.0) * self.tau,
            )
        x: Array1D = to_unit_interval(input.x, x_interval)
        y: Array1D = to_unit_interval(input.y, y_interval)
        rng = np.random.default_rng(seed)

        for _ in range(self.num_steps):
            gradient_sum = _clipped_gradient_sum(x, y, predictions, self.tau)
            if use_laplace:
                noise = rng.laplace(
                    loc=0.0,
                    scale=noise_scale,
                    size=predictions.shape,
                )
            else:
                noise = rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=predictions.shape,
                )
            noisy_gradient: Array1D = gradient_sum + noise
            cumulative_squared_norm += noisy_gradient @ noisy_gradient
            if cumulative_squared_norm == 0:
                raise RuntimeError(
                    "DPGD failed: the cumulative noisy-gradient norm " "remained zero."
                )
            step_size = 1 / math.sqrt(cumulative_squared_norm)
            predictions = predictions - step_size * noisy_gradient
            iterates.append(predictions.copy())

        released_predictions: Array1D = np.mean(
            np.asarray(iterates[self.num_steps // 2 :]), axis=0
        )
        p25 = float(released_predictions[0])
        p75 = float(released_predictions[1])
        slope_norm, intercept_norm = line_from_quartile_predictions(p25, p75)
        slope, intercept = line_from_unit_square(
            slope_norm, intercept_norm, x_interval, y_interval
        )
        return SimpleLinearRegressionOutput(slope=slope, intercept=intercept)
