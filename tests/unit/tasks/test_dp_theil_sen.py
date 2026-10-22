"""Tests for smooth-sensitivity DP Theil--Sen regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dpstats.domains import BoxDomain, Domain, EuclideanSpace
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms import (
    DPTheilSen,
    dp_theil_sen,
)
from dpstats.tasks.simple_linear_regression.algorithms.dp_theil_sen import (
    _release_matching_smooth_sensitivity_median,  # pyright: ignore[reportPrivateUsage]
)
from dpstats.simulations.simple_linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close


UNIT_DOMAIN = BoxDomain(bounds=np.array([[0.0, 1.0]]))
LOWER_PREDICTION_BOUND = -0.5
UPPER_PREDICTION_BOUND = 1.5


def _metadata(
    n_samples: int,
    *,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
) -> SimpleLinearRegressionMetadata:
    return SimpleLinearRegressionMetadata(
        n_samples=n_samples,
        x_domain=BoxDomain(bounds=np.array([x_bounds])),
        y_domain=BoxDomain(bounds=np.array([y_bounds])),
    )


def _released_prediction_vectors(
    x: Array1D, y: Array1D, seed: int
) -> tuple[Array1D, Array1D]:
    """Record the two totalized prediction vectors a run takes medians of.

    Their entries are the quantity Lemma 10's sensitivity argument is stated
    about, and the noise is applied to their medians, so recording them shows
    the released location without reimplementing the release.
    """
    captured: list[Array1D] = []
    release = _release_matching_smooth_sensitivity_median

    def recording_release(values: Array1D, **kwargs: Any) -> float:
        captured.append(values.copy())
        return release(values, **kwargs)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(
            dp_theil_sen,
            "_release_matching_smooth_sensitivity_median",
            recording_release,
        )
        DPTheilSen(
            prediction_lower_bound=LOWER_PREDICTION_BOUND,
            prediction_upper_bound=UPPER_PREDICTION_BOUND,
        ).run(
            _metadata(len(x)),
            SimpleLinearRegressionInput(x=x, y=y),
            PureDPBudget(eps=2.0),
            seed,
        )
    predictions_25, predictions_75 = captured
    return predictions_25, predictions_75


class TestPaperContract:
    def test_vertical_pairs_use_the_fixed_length_totalization(self) -> None:
        first = SimpleLinearRegressionInput(
            x=np.full(4, 0.5), y=np.array([0.0, 0.25, 0.75, 1.0])
        )
        second = SimpleLinearRegressionInput(
            x=np.full(4, 0.5), y=np.array([1.0, 0.75, 0.25, 0.0])
        )
        metadata = _metadata(4)
        budget = PureDPBudget(eps=2.0)
        seed = 7

        first_output = DPTheilSen().run(metadata, first, budget, seed)
        second_output = DPTheilSen().run(metadata, second, budget, seed)

        assert first_output.slope == pytest.approx(second_output.slope)
        assert first_output.intercept == pytest.approx(second_output.intercept)

    def test_tied_pairs_release_the_median_of_the_untied_predictions(self) -> None:
        # A balanced binary covariate ties 12 of the 8 * 7 / 2 pairs. Every
        # untied prediction here is a convex combination of two responses, so it
        # lies strictly inside the prediction bounds and the filled entries are
        # exactly the entries sitting at a bound.
        x = np.repeat([0.0, 1.0], 4)
        y = np.array([0.15, 0.35, 0.55, 0.75, 0.25, 0.45, 0.65, 0.85])

        predictions_25, predictions_75 = _released_prediction_vectors(x, y, seed=6)

        filled = (predictions_25 == LOWER_PREDICTION_BOUND) | (
            predictions_25 == UPPER_PREDICTION_BOUND
        )
        assert len(predictions_25) == len(predictions_75) == 28
        assert np.count_nonzero(filled) == 12
        # Both coordinates fill the same pairs with the same endpoint, because
        # the coin is indexed by the pair rather than by the coordinate.
        assert np.array_equal(predictions_25[filled], predictions_75[filled])
        # This seed's coin splits those pairs evenly, so the fill adds the same
        # mass strictly below and strictly above the untied predictions and
        # leaves their median exactly where Algorithm 2's skipping map puts it.
        # A constant interior fill would report its own value instead.
        assert np.count_nonzero(predictions_25 == LOWER_PREDICTION_BOUND) == 6
        assert np.median(predictions_25) == pytest.approx(
            np.median(predictions_25[~filled])
        )
        assert np.median(predictions_75) == pytest.approx(
            np.median(predictions_75[~filled])
        )

    def test_untying_a_pair_changes_at_most_k_entries(self) -> None:
        # Rows 0 and 1 share a covariate, and replacing row 1 unties that pair
        # while two other tied pairs survive. A fill keyed by anything but pair
        # identity -- rank among the ties, say -- would renumber the survivors
        # and change entries that no replaced record touches.
        x = np.array([0.2, 0.2, 0.3, 0.45, 0.6, 0.6, 0.85, 0.85])
        y = np.array([0.1, 0.9, 0.3, 0.4, 0.55, 0.6, 0.8, 0.95])
        replaced_x = x.copy()
        replaced_x[1] = 0.25
        n = len(x)
        k = n - 1

        baseline = _released_prediction_vectors(x, y, seed=4)
        replaced = _released_prediction_vectors(replaced_x, y, seed=4)

        for before, after in zip(baseline, replaced):
            assert len(before) == len(after) == k * n // 2
            changed = np.count_nonzero(before != after)
            assert 0 < changed <= k


class TestPublicDomains:
    def test_affine_domain_normalization_and_output_mapping(self) -> None:
        normalized_input = SimpleLinearRegressionInput(
            x=np.array([0.0, 0.2, 0.7, 1.0]),
            y=np.array([0.1, 0.3, 0.6, 0.95]),
        )
        transformed_input = SimpleLinearRegressionInput(
            x=10.0 + 4.0 * normalized_input.x,
            y=-3.0 + 7.0 * normalized_input.y,
        )
        budget = PureDPBudget(eps=3.0)
        seed = 19

        normalized = DPTheilSen().run(
            _metadata(4),
            normalized_input,
            budget,
            seed,
        )
        transformed = DPTheilSen().run(
            _metadata(4, x_bounds=(10.0, 14.0), y_bounds=(-3.0, 4.0)),
            transformed_input,
            budget,
            seed,
        )

        expected_slope = normalized.slope * 7.0 / 4.0
        expected_intercept = -3.0 + 7.0 * normalized.intercept - expected_slope * 10.0
        assert transformed.slope == pytest.approx(expected_slope)
        assert transformed.intercept == pytest.approx(expected_intercept)

    @pytest.mark.parametrize(
        ("x_domain", "y_domain", "message"),
        [
            (EuclideanSpace(dim=1), UNIT_DOMAIN, "x interval"),
            (UNIT_DOMAIN, EuclideanSpace(dim=1), "y interval"),
            (
                BoxDomain(bounds=np.array([[0.0, 0.0]])),
                UNIT_DOMAIN,
                "x_domain",
            ),
            (
                UNIT_DOMAIN,
                BoxDomain(bounds=np.array([[0.0, np.inf]])),
                "y interval",
            ),
        ],
    )
    def test_requires_finite_positive_width_scalar_boxes(
        self,
        x_domain: Domain,
        y_domain: Domain,
        message: str,
    ) -> None:
        metadata = SimpleLinearRegressionMetadata(
            n_samples=4,
            x_domain=x_domain,
            y_domain=y_domain,
        )
        input = SimpleLinearRegressionInput(
            x=np.linspace(0.0, 1.0, 4),
            y=np.linspace(0.0, 1.0, 4),
        )
        with pytest.raises(InvalidRegime, match=message):
            DPTheilSen().run(metadata, input, PureDPBudget(eps=1.0), 0)


class TestConstruction:
    def test_prediction_bounds_default_to_the_paper_interval(self) -> None:
        algorithm = DPTheilSen()
        assert (algorithm.prediction_lower_bound, algorithm.prediction_upper_bound) == (
            -0.5,
            1.5,
        )

    @pytest.mark.parametrize("lower", [1.0, 2.0])
    def test_prediction_bounds_must_be_increasing(self, lower: float) -> None:
        with pytest.raises(ValueError, match="prediction bounds"):
            DPTheilSen(prediction_lower_bound=lower, prediction_upper_bound=1.0)

    def test_noninteger_degrees_of_freedom_is_supported(self) -> None:
        metadata, input, _ = simulate_clipped_gaussian(n_samples=20, seed=0)

        output = DPTheilSen(degrees_of_freedom=2.5).run(
            metadata,
            input,
            PureDPBudget(eps=2.0),
            17,
        )

        assert np.isfinite(output.slope)
        assert np.isfinite(output.intercept)


class TestExecution:
    @pytest.mark.parametrize(
        "budget", [NO_PRIVACY, ApproxDPBudget(eps=float("inf"), delta=1.0)]
    )
    def test_unconstrained_offers_match_the_delta_one_approx_dp_offer(
        self, budget: DPBudget
    ) -> None:
        # ApproxDPBudget.split keeps delta at 1 in every part of a delta=1
        # whole, so every unconstrained offer, of any family, releases
        # identically to an explicit ApproxDPBudget(1, 1) offer.
        metadata, input, _ = simulate_clipped_gaussian(n_samples=20, seed=0)
        reference = DPTheilSen().run(
            metadata, input, ApproxDPBudget(eps=1.0, delta=1.0), 7
        )

        output = DPTheilSen().run(metadata, input, budget, 7)

        assert (output.slope, output.intercept) == pytest.approx(
            (reference.slope, reference.intercept)
        )

    def test_algorithm_rejects_fewer_than_two_samples(self) -> None:
        metadata = _metadata(1)
        input = SimpleLinearRegressionInput(
            x=np.array([0.5]),
            y=np.array([0.5]),
        )
        with pytest.raises(InvalidRegime, match="at least two samples"):
            DPTheilSen().run(metadata, input, PureDPBudget(eps=1.0), 0)

    def test_odd_sample_size_discards_exactly_one_row(self) -> None:
        # Moving one row to the opposite corner changes the release unless
        # that row is the one the seeded preprocessing discards.
        seed = 23
        x = np.array([0.05, 0.2, 0.45, 0.7, 0.95])
        y = np.array([0.1, 0.3, 0.4, 0.75, 0.9])
        algorithm = DPTheilSen()
        budget = PureDPBudget(eps=2.0)
        baseline = algorithm.run(
            _metadata(5), SimpleLinearRegressionInput(x=x, y=y), budget, seed
        )

        ignored = []
        for row in range(5):
            moved_x, moved_y = x.copy(), y.copy()
            moved_x[row] = 1.0 - x[row]
            moved_y[row] = 1.0 - y[row]
            moved = algorithm.run(
                _metadata(5),
                SimpleLinearRegressionInput(x=moved_x, y=moved_y),
                budget,
                seed,
            )
            if (moved.slope, moved.intercept) == pytest.approx(
                (baseline.slope, baseline.intercept)
            ):
                ignored.append(row)

        assert len(ignored) == 1

    @pytest.mark.parametrize(
        "offered",
        [
            PureDPBudget(eps=2.0),
            ApproxDPBudget(eps=2.0, delta=1e-5),
            ZCDPBudget(rho=2.0),
        ],
        ids=["pure", "approx", "zcdp"],
    )
    def test_routes_the_offered_budget_through_pure_accounting(
        self,
        accounting_spy: AccountingSpy,
        offered: DPBudget,
    ) -> None:
        metadata, input, _ = simulate_clipped_gaussian(n_samples=20, seed=0)

        DPTheilSen().run(metadata, input, offered, 7)

        if isinstance(offered, PureDPBudget):
            child_budget: DPBudget = PureDPBudget(eps=offered.eps / 2)
        elif isinstance(offered, ApproxDPBudget):
            child_budget = ApproxDPBudget(eps=offered.eps / 2, delta=offered.delta / 2)
        else:
            assert isinstance(offered, ZCDPBudget)
            child_budget = ZCDPBudget(rho=offered.rho / 2)
        assert len(accounting_spy.calls) == 2
        for call in accounting_spy.calls:
            assert call.entry_point == "spend_as_pure"
            assert_budget_close(call.budget, child_budget)
