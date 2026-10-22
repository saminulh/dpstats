"""The ``GridExponentialMechanism`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime

from itertools import product
import math
from typing import Sequence

import numpy as np

from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D, Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


_MAX_GRID_DIMENSION = 3


def _resolve_domain_scales(
    metadata: LinearRegressionMetadata,
    residual_cap_mult: float,
) -> tuple[float, Array1D]:
    """Resolve the capped-residual and coefficient scales from public domains."""
    design_bounds = metadata.design_domain.bounding_box().bounds
    response_bounds = metadata.response_domain.bounding_box().bounds
    if not np.all(np.isfinite(design_bounds)) or not np.all(
        np.isfinite(response_bounds)
    ):
        raise InvalidRegime(
            "Discrete regression EM requires finite design and response bounds."
        )

    y_lo, y_hi = response_bounds[0]
    response_half_width = (y_hi - y_lo) / 2
    if response_half_width == 0.0:
        raise InvalidRegime(
            "Discrete regression EM requires a nonzero response-domain width."
        )

    design_half_widths = (design_bounds[:, 1] - design_bounds[:, 0]) / 2
    first_slope = 1 if metadata.has_intercept else 0
    if np.any(design_half_widths[first_slope:] == 0.0):
        raise InvalidRegime(
            "Discrete regression EM requires nonzero widths for all covariates."
        )

    coefficient_scales = np.empty(metadata.dim)
    coefficient_scales[first_slope:] = (
        response_half_width / design_half_widths[first_slope:]
    )
    if metadata.has_intercept:
        design_centers = design_bounds[:, 0] + design_half_widths
        coefficient_scales[0] = max(abs(y_lo), abs(y_hi)) + np.dot(
            np.abs(design_centers[first_slope:]),
            coefficient_scales[first_slope:],
        )
    return residual_cap_mult * response_half_width, coefficient_scales


def _sample_grid_point(
    X: Array2D,
    y: Array1D,
    grids_per_dim: Sequence[Array1D],
    residual_cap: float,
    score_scale: float,
    seed: int,
) -> Array1D:
    """Sample an exponential-mechanism grid point.

    Gumbel-max sampling: adding an independent standard-Gumbel draw to each
    log-weight ``-score_scale * score`` and taking the argmax samples exactly
    from the categorical distribution with weights ``exp(-score_scale * score)``.

    Args:
        X: Regression design matrix.
        y: Regression response vector.
        grids_per_dim: Ordered coefficient candidates for each dimension.
        residual_cap: Absolute residual cap used by the quality score.
        score_scale: Exponential-mechanism multiplier on the quality score.
        seed: RNG seed for the mechanism's Gumbel draws.

    Returns:
        The sampled coefficient vector.
    """
    points: Array2D = np.asarray(list(product(*grids_per_dim)), dtype=np.float64)
    residuals: Array2D = points @ X.T - y
    scores: Array1D = np.minimum(residuals**2, residual_cap**2).sum(axis=1)
    rng = np.random.default_rng(seed)
    selection_scores = rng.gumbel(size=len(points))
    if math.isinf(score_scale):
        selection_scores[scores != scores.min()] = -math.inf
    else:
        selection_scores -= score_scale * scores
    return points[int(np.argmax(selection_scores))]


class GridExponentialMechanism(BaseAlgorithm):
    """Select regression coefficients from a grid by the exponential mechanism.

    Declares a public finite grid and outputs a point from it with the
    exponential mechanism (McSherry and Talwar, 2007), scoring each point by its
    clipped sum of squared residuals. It is a pure-DP mechanism.

    Requires:
        - A public design dimension of at most three.
        - Finite design and response bounding boxes.
        - A nonzero response-domain width and a nonzero width for every
          non-intercept design coordinate.

    Parametrization notes:
        - ``residual_cap_mult``: Sets the cap on each record's absolute residual
          as a multiple of the response domain's half-width. The exponential
          mechanism prescribes no value.
        - ``coef_box_mult``: Sets the coefficient box as a multiple of the
          coefficient scales the public domains imply. The paper prescribes
          no value.
        - ``num_steps``: The number of grid points per coefficient. The paper
          prescribes no value.

    Reference:
        - mcsherry2007mechanism: Theorem 6.
    """

    TASK = LinearRegression

    residual_cap_mult: float = parameter_field(
        default=0.1,
        gt=0.0,
        search=search_range(0.01, 10.0, scale="log"),
        description="Positive multiplier on half the response-domain width, capping the residual score.",
    )
    coef_box_mult: float = parameter_field(
        default=1.0,
        gt=0.0,
        search=search_range(0.1, 10.0, scale="log"),
        description="Positive multiplier on the domain-derived coefficient grid scales.",
    )
    num_steps: int = parameter_field(
        default=30,
        ge=2,
        search=search_range(2, 100),
        description="Number of grid points per dimension, at least two.",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        if metadata.dim > _MAX_GRID_DIMENSION:
            raise InvalidRegime(
                "The current grid sampler supports design dimensions through 3; "
                f"received public dimension {metadata.dim}."
            )
        residual_cap, coefficient_scales = _resolve_domain_scales(
            metadata, self.residual_cap_mult
        )
        eps = accounting.spend_as_pure(budget).eps
        X, y = input.X, input.y
        grids_per_dim: list[Array1D] = [
            np.linspace(-half_width, half_width, self.num_steps)
            for half_width in self.coef_box_mult * coefficient_scales
        ]
        # Adding or removing a record moves every score by an amount in [0, R**2],
        # all in one direction, so eps / R**2 needs no factor of two for eps-DP.
        score_scale = eps / residual_cap**2
        beta_hat = _sample_grid_point(
            X, y, grids_per_dim, residual_cap, score_scale, seed
        )
        return LinearRegressionOutput(beta=beta_hat)
