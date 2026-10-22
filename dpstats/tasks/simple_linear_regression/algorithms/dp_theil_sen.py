"""Smooth-sensitivity DP Theil--Sen for simple linear regression."""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import PrivacyInterpretation

import math

import numpy as np
from pydantic import model_validator

from dpstats.algorithm import BaseAlgorithm
from dpstats.parameters import parameter_field
from dpstats.budgets import accounting
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D, IntArray1D, IntArray2D
from dpstats.utils import split_seed
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


def _selected_matchings(n_vertices: int, round_indices: IntArray1D) -> IntArray2D:
    """Construct selected rounds of an even complete graph's one-factorization."""
    if n_vertices < 2 or n_vertices % 2 != 0:
        raise ValueError("The matching construction requires an even sample size.")

    n_rotating = n_vertices - 1
    matchings = np.empty((len(round_indices), n_vertices), dtype=np.int64)
    rotating = np.arange(1, n_vertices, dtype=np.int64)
    positions = np.arange(n_rotating, dtype=np.int64)
    rounds = round_indices[:, None]
    rotated = rotating[(positions[None, :] - rounds) % n_rotating]
    vertices = np.empty((len(round_indices), n_vertices), dtype=np.int64)
    vertices[:, 0] = 0
    vertices[:, 1:] = rotated
    matchings[:, 0::2] = vertices[:, : n_vertices // 2]
    matchings[:, 1::2] = vertices[:, : n_vertices // 2 - 1 : -1]
    return matchings


def _matching_smooth_sensitivity(
    clipped_values: Array1D,
    *,
    n_samples: int,
    k: int,
    t: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Compute the t-smooth sensitivity in Alabi et al., Lemma 10.

    Order statistics beyond the observed values take the public boundary
    values. Equation (4) supplies the sliding lower index
    ``m - k * (distance + 1) + shift``. The bookended ``len // 2``
    convention matches the paper's reference implementation and bounds the
    midpoint median used for an even number of estimates.
    """
    ordered = np.sort(np.concatenate(([lower_bound], clipped_values, [upper_bound])))
    median_index = len(ordered) // 2
    last_index = len(ordered) - 1

    sensitivity = max(
        float(ordered[min(median_index + k, last_index)] - ordered[median_index]),
        float(ordered[median_index] - ordered[max(median_index - k, 0)]),
    )
    output_width = upper_bound - lower_bound
    for distance in range(1, n_samples + 1):
        decay = math.exp(-distance * t)
        if decay * output_width <= sensitivity:
            break
        window = k * (distance + 1)
        shifts = np.arange(window + 1)
        upper_indices = np.minimum(median_index + shifts, last_index)
        lower_indices = np.maximum(median_index - window + shifts, 0)
        largest_gap = np.max(ordered[upper_indices] - ordered[lower_indices]).item()
        sensitivity = max(sensitivity, decay * largest_gap)
    return sensitivity


def _release_matching_smooth_sensitivity_median(
    values: Array1D,
    *,
    n_samples: int,
    num_matchings: int,
    lower_bound: float,
    upper_bound: float,
    degrees_of_freedom: float,
    budget: PureDPBudget,
    seed: int,
) -> float:
    """Apply Alabi et al. Algorithm 5 to the k-matching prediction map."""
    eps = budget.eps
    d = degrees_of_freedom
    # Algorithm 5 fixes these constants; they are not tuning parameters.
    t = eps / (2 * (d + 1))
    s = eps * math.sqrt(d) / (d + 1)

    clipped = np.clip(values, lower_bound, upper_bound)
    sensitivity = _matching_smooth_sensitivity(
        clipped,
        n_samples=n_samples,
        k=num_matchings,
        t=t,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    noise = np.random.default_rng(seed).standard_t(d)
    return np.median(clipped).item() + sensitivity * noise / s


def _run_algorithm_2(
    x: Array1D,
    y: Array1D,
    *,
    n_samples: int,
    num_matchings: int,
    prediction_lower_bound: float,
    prediction_upper_bound: float,
    degrees_of_freedom: float,
    lower_budget: PureDPBudget,
    upper_budget: PureDPBudget,
    seed: int,
) -> tuple[float, float]:
    """Run Alabi et al. Algorithm 2 and output its two predictions.

    This keeps the paper's matching and prediction construction separate from
    its DPSS median component without introducing a public composition API.
    """
    matching_seed, lower_seed, upper_seed, fill_seed = split_seed(seed, 4)
    matching_rng = np.random.default_rng(matching_seed)
    rounds = matching_rng.choice(
        n_samples - 1,
        size=num_matchings,
        replace=False,
    )
    pairs = _selected_matchings(n_samples, rounds).reshape(-1, 2)
    left = pairs[:, 0]
    right = pairs[:, 1]
    x_difference = x[right] - x[left]
    non_vertical = x_difference != 0
    slopes = np.divide(
        y[right] - y[left],
        x_difference,
        out=np.zeros(len(pairs)),
        where=non_vertical,
    )
    x_pair_mean = (x[right] + x[left]) / 2
    y_pair_mean = (y[right] + y[left]) / 2

    # Algorithm 2 skips vertical pairs, but its proof needs a fixed kn/2-length
    # map. Each vertical pair takes a public bound chosen by a fair coin per pair,
    # so a replaced record still moves at most one entry per matching.
    coins = np.random.default_rng(fill_seed).integers(2, size=(n_samples, n_samples))
    vertical_pair_value = np.where(
        coins[np.minimum(left, right), np.maximum(left, right)] == 1,
        prediction_upper_bound,
        prediction_lower_bound,
    )
    predictions_25 = np.where(
        non_vertical,
        slopes * (0.25 - x_pair_mean) + y_pair_mean,
        vertical_pair_value,
    )
    predictions_75 = np.where(
        non_vertical,
        slopes * (0.75 - x_pair_mean) + y_pair_mean,
        vertical_pair_value,
    )

    prediction_25 = _release_matching_smooth_sensitivity_median(
        predictions_25,
        n_samples=n_samples,
        num_matchings=num_matchings,
        lower_bound=prediction_lower_bound,
        upper_bound=prediction_upper_bound,
        degrees_of_freedom=degrees_of_freedom,
        budget=lower_budget,
        seed=lower_seed,
    )
    prediction_75 = _release_matching_smooth_sensitivity_median(
        predictions_75,
        n_samples=n_samples,
        num_matchings=num_matchings,
        lower_bound=prediction_lower_bound,
        upper_bound=prediction_upper_bound,
        degrees_of_freedom=degrees_of_freedom,
        budget=upper_budget,
        seed=upper_seed,
    )
    return prediction_25, prediction_75


class DPTheilSen(BaseAlgorithm):
    """DPTheilSenkMatch. Estimate the slope and intercept privately as medians over lines through pairs of points (Theil-Sen).

    An implementation of Algorithms 2 and 5 from Alabi et al. (2022). Pairs up
    the records in several ways, fits a line through each pair, takes the
    median of those lines' predictions at x = 0.25 and x = 0.75, adds Student-t
    noise scaled to the medians' smooth sensitivity, and outputs the slope and
    intercept that the two noisy predictions give. It is a pure-DP mechanism.

    Requires:
        - Finite, positive-width public scalar domains on x and y.
        - At least two public samples.

    Parametrization notes:
        - ``prediction_lower_bound``: Corresponds to $r_l$ in the paper, in
          normalized response units. Set as in the authors' real-data setting.
        - ``prediction_upper_bound``: Corresponds to $r_u$ in the paper, in
          normalized response units. Set as in the authors' real-data setting.
        - ``degrees_of_freedom``: Corresponds to $d$ in Algorithm 5. Set as in
          the authors' reference implementation.

    Implementation notes:
        - The medians are taken over all ``n - 1`` disjoint matchings of the
          records, with the matching-aware smooth sensitivity of Lemmas 10, 11 and
          21 and Student-t noise.
        - The guarantee is for replacing one record in a dataset of fixed even
          size; no claim is made for adding or removing a record. An odd number of
          records above one first drops one record chosen uniformly at random,
          independently of the data.
        - Lemma 10's proof needs exactly ``kn/2`` pair predictions, but
          Algorithm 2 skips pairs with equal ``x``. Those entries are filled
          with one of the two public prediction bounds, chosen by a fair coin
          that depends only on the pair's position. That is symmetric between
          the bounds and puts no fixed point mass inside them in the released
          median.

    Reference:
        - alabi2022differentially: Algorithms 2 and 5 (arXiv v1 numbering; Algorithms 2
          and 6 in the published version).
    """

    TASK = SimpleLinearRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.QUALIFIED
    )

    prediction_lower_bound: float = parameter_field(
        default=-0.5,
        description="Lower endpoint of the public prediction interval in normalized response units.",
    )
    prediction_upper_bound: float = parameter_field(
        default=1.5,
        description="Upper endpoint of the public prediction interval in normalized response units.",
    )
    degrees_of_freedom: float = parameter_field(
        default=3.0,
        gt=0.0,
        description="Degrees of freedom of the Student-t noise.",
    )

    @model_validator(mode="after")
    def _validate_prediction_bounds(self) -> DPTheilSen:
        if self.prediction_lower_bound >= self.prediction_upper_bound:
            raise ValueError(
                "prediction bounds must satisfy lower_bound < upper_bound."
            )
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
        n = metadata.n_samples
        if n < 2:
            raise InvalidRegime("DPTheilSen requires at least two samples.")
        # The two prediction medians split the offer equally; each half is
        # spent before the Input is read.
        lower_half, upper_half = budget.split((1.0, 1.0))
        lower_budget = accounting.spend_as_pure(lower_half)
        upper_budget = accounting.spend_as_pure(upper_half)

        x: Array1D = to_unit_interval(input.x, x_interval)
        y: Array1D = to_unit_interval(input.y, y_interval)
        algorithm_seed = seed
        if n % 2 != 0:
            preprocessing_seed, algorithm_seed = split_seed(seed, 2)
            dropped_index = np.random.default_rng(preprocessing_seed).integers(n)
            x = np.delete(x, dropped_index)
            y = np.delete(y, dropped_index)
            n -= 1

        p_tilde_25, p_tilde_75 = _run_algorithm_2(
            x,
            y,
            n_samples=n,
            num_matchings=n - 1,
            prediction_lower_bound=self.prediction_lower_bound,
            prediction_upper_bound=self.prediction_upper_bound,
            degrees_of_freedom=self.degrees_of_freedom,
            lower_budget=lower_budget,
            upper_budget=upper_budget,
            seed=algorithm_seed,
        )

        normalized_slope, normalized_intercept = line_from_quartile_predictions(
            p_tilde_25, p_tilde_75
        )
        slope, intercept = line_from_unit_square(
            normalized_slope, normalized_intercept, x_interval, y_interval
        )
        return SimpleLinearRegressionOutput(slope=slope, intercept=intercept)
