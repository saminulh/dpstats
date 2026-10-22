"""Shared unit-square coordinate transport for the SLR algorithms.

The four bounds-assuming simple-regression mechanisms transport records onto
the unit square, release a line there, and invert the normalization through
one shared spelling in ``algorithms/utils.py``. These tests check that each
release commutes with a change of units and that the transforms satisfy their
defining algebraic properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpstats.budgets import PureDPBudget
from dpstats.tasks.simple_linear_regression.algorithms.dpgd import (
    DPGD,
)
from dpstats.tasks.simple_linear_regression.algorithms.dp_rss import DPRSS
from dpstats.tasks.simple_linear_regression.algorithms.noisy_stats import (
    NoisyStats,
)
from dpstats.tasks.simple_linear_regression.algorithms.dp_theil_sen import (
    DPTheilSen,
)
from dpstats.tasks.simple_linear_regression.algorithms.utils import (
    line_from_quartile_predictions,
    line_from_unit_square,
    to_unit_interval,
)
from dpstats.simulations.simple_linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
)
from tests.unit.tasks.task_builders import box, slr_metadata


class TestAffineEquivariance:
    """Every transported release commutes with a public affine change of units.

    Mapping ``x -> a x + b`` and ``y -> c y + d`` together with their declared
    intervals leaves the unit-square records unchanged, so under a fixed seed
    the released line ``y = s x + t`` must become
    ``y = (c s / a) x + (c t + d - c s b / a)``.
    """

    @pytest.mark.parametrize(
        "alg_class",
        [DPTheilSen, NoisyStats, DPRSS, DPGD],
        ids=lambda alg_class: alg_class.__name__,
    )
    def test_released_line_follows_an_affine_change_of_units(
        self, alg_class: type
    ) -> None:
        a, b, c, d = 2.0, 3.0, 5.0, -1.0
        metadata, input_data, _ = simulate_clipped_gaussian(seed=7, n_samples=200)
        np.testing.assert_array_equal(metadata.x_domain.bounds, [[-10.0, 10.0]])
        np.testing.assert_array_equal(metadata.y_domain.bounds, [[-10.0, 10.0]])
        moved_metadata = slr_metadata(200, box((-17.0, 23.0)), box((-51.0, 49.0)))
        moved_input = SimpleLinearRegressionInput(
            x=a * input_data.x + b, y=c * input_data.y + d
        )
        budget = PureDPBudget(eps=10.0)

        line = alg_class().run(metadata, input_data, budget, seed=11)
        moved = alg_class().run(moved_metadata, moved_input, budget, seed=11)

        assert moved.slope == pytest.approx(c * line.slope / a, rel=1e-9)
        assert moved.intercept == pytest.approx(
            c * line.intercept + d - c * line.slope * b / a, rel=1e-9, abs=1e-9
        )


class TestUnitSquareTransport:
    """The shared transforms satisfy their defining algebraic contracts."""

    def test_forward_map_sends_interval_endpoints_to_zero_and_one(self) -> None:
        values = np.array([-3.0, 0.5, 4.0])
        unit = to_unit_interval(values, (-3.0, 4.0))
        assert unit[0] == 0.0
        assert unit[-1] == 1.0
        assert np.all((unit >= 0.0) & (unit <= 1.0))
        assert unit[1] == pytest.approx(3.5 / 7.0)

    def test_inverse_line_map_transports_points_on_the_line(self) -> None:
        """A point on the normalized line maps to a point on the returned line."""
        x_interval = (-3.0, 4.0)
        y_interval = (2.0, 12.0)
        normalized_slope, normalized_intercept = 0.8, -0.25
        slope, intercept = line_from_unit_square(
            normalized_slope, normalized_intercept, x_interval, y_interval
        )
        for x in np.linspace(-6.0, 7.0, 9):
            x_unit = (x - x_interval[0]) / (x_interval[1] - x_interval[0])
            y_unit = normalized_slope * x_unit + normalized_intercept
            y = y_interval[0] + (y_interval[1] - y_interval[0]) * y_unit
            assert slope * x + intercept == pytest.approx(y, rel=1e-12)

    def test_quartile_conversion_interpolates_the_prediction_pair(self) -> None:
        """The returned line passes through (0.25, p25) and (0.75, p75)."""
        p25, p75 = 0.3, 0.9
        slope, intercept = line_from_quartile_predictions(p25, p75)
        assert slope * 0.25 + intercept == pytest.approx(p25, rel=1e-12)
        assert slope * 0.75 + intercept == pytest.approx(p75, rel=1e-12)
