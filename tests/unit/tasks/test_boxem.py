"""Tests for the exact low-dimensional BoxEM mean estimator."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from dpstats.domains import BoxDomain, EuclideanSpace, LpBallDomain
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D, Array2D
from dpstats.tasks.mean_estimation.algorithms.boxem import (
    BoxEM,
    _boxem_log_level_weights,  # pyright: ignore[reportPrivateUsage]
    _compute_tukey_regions,  # pyright: ignore[reportPrivateUsage]
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy


def _area(polygon: Array2D) -> float:
    """Return polygon area for independent test assertions."""
    if polygon.shape[0] < 3:
        return 0.0
    return (
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        / 2
    )


def _halfspace_depth(X: Array2D, point: Array1D) -> int:
    """Compute planar Tukey depth by sweeping query-centered directions."""
    offsets = X - point
    nonzero = np.linalg.norm(offsets, axis=1) > 0
    if not np.any(nonzero):
        return X.shape[0]
    angles = np.arctan2(offsets[nonzero, 1], offsets[nonzero, 0])
    events = np.sort(
        np.mod(
            np.concatenate((angles - math.pi / 2, angles + math.pi / 2)), 2 * math.pi
        )
    )
    next_events = np.roll(events, -1)
    next_events[-1] += 2 * math.pi
    directions = np.mod((events + next_events) / 2, 2 * math.pi)
    counts = [
        np.count_nonzero(offsets @ np.array([math.cos(angle), math.sin(angle)]) >= 0)
        for angle in directions
    ]
    return int(min(counts))


def _inside_polygon(point: Array1D, polygon: Array2D) -> bool:
    """Test convex-polygon membership with SciPy's independent hull equations."""
    if polygon.shape[0] < 3:
        return False
    equations = ConvexHull(polygon).equations
    return bool(np.all(equations[:, :2] @ point + equations[:, 2] <= 1e-9))


class TestBoxEMGeometry:
    """Pin the Tukey regions and Brown--Zakynthinou level mixture."""

    def test_scalar_regions_include_tied_observations(self) -> None:
        X = np.array([[-2.0], [-0.5], [-0.5], [1.0], [3.0], [3.0]])
        regions = _compute_tukey_regions(X, np.array([[-4.0, 4.0]]))
        np.testing.assert_array_equal(regions[0], [[-4.0], [4.0]])
        np.testing.assert_array_equal(regions[1], [[-2.0], [3.0]])
        np.testing.assert_array_equal(regions[2], [[-0.5], [3.0]])
        np.testing.assert_array_equal(regions[3], [[-0.5], [1.0]])

    def test_planar_depth_one_matches_scipy_convex_hull(self) -> None:
        rng = np.random.default_rng(27)
        for _ in range(5):
            X = rng.normal(size=(8, 2))
            regions = _compute_tukey_regions(X, np.array([[-3.0, 3.0], [-3.0, 3.0]]))
            assert _area(regions[1]) == pytest.approx(ConvexHull(X).volume)

    def test_planar_regions_match_independent_halfspace_depth(self) -> None:
        X = np.array(
            [
                [-1.7, -0.2],
                [-1.0, 1.4],
                [-0.3, -1.5],
                [0.2, 0.3],
                [0.7, 1.2],
                [1.4, -0.8],
                [1.8, 0.5],
            ]
        )
        regions = _compute_tukey_regions(X, np.array([[-2.0, 2.0], [-2.0, 2.0]]))
        for x in np.linspace(-1.75, 1.75, 8):
            for y in np.linspace(-1.7, 1.7, 7):
                point = np.array([x, y])
                depth = _halfspace_depth(X, point)
                for level in range(1, len(regions)):
                    assert _inside_polygon(point, regions[level]) == (depth >= level)

    def test_level_weights_equal_telescoping_density_masses(self) -> None:
        eps = 1.4
        regions = _compute_tukey_regions(
            np.array([[-2.0], [-0.5], [0.0], [1.0], [3.0]]),
            np.array([[-4.0, 4.0]]),
        )
        actual = np.exp(_boxem_log_level_weights(regions, PureDPBudget(eps=eps)))
        rate = eps / 2
        expected = np.array(
            [
                8.0,
                5.0 * (math.exp(rate) - 1),
                1.5 * (math.exp(2 * rate) - math.exp(rate)),
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_planar_area_is_stable_for_a_narrow_large_offset_box(self) -> None:
        offset = 1e8
        bounds = np.array([[offset, offset + 2.0], [offset, offset + 2.0]])
        regions = _compute_tukey_regions(
            np.array([[offset + 1.0, offset + 1.0]]), bounds
        )

        weights = np.exp(_boxem_log_level_weights(regions, PureDPBudget(eps=1.0)))

        assert weights[0] == pytest.approx(4.0)

    def test_positive_tiny_budget_has_finite_level_increments(self) -> None:
        regions = [
            np.array([[-1.0], [1.0]]),
            np.array([[-0.5], [0.5]]),
        ]

        weights = _boxem_log_level_weights(regions, PureDPBudget(eps=1e-20))

        assert np.all(np.isfinite(weights))


class TestBoxEMAlgorithm:
    """Exercise public sampling, accounting, and guards."""

    def test_approximate_offer_spends_as_pure_inside_the_public_box(self) -> None:
        metadata = MeanEstimationMetadata(
            n_samples=5,
            domain=BoxDomain(bounds=np.array([[-3.0, 3.0]])),
        )
        input_data = MeanEstimationInput(
            X=np.array([[-1.0], [-0.2], [0.0], [0.7], [1.4]])
        )
        algorithm = BoxEM()
        first = algorithm.run(metadata, input_data, PureDPBudget(eps=1.0), seed=11)
        with_delta = algorithm.run(
            metadata, input_data, ApproxDPBudget(eps=1.0, delta=1e-5), seed=11
        )
        np.testing.assert_allclose(first.mean_est, with_delta.mean_est)
        assert -3 <= first.mean_est[0] <= 3

    def test_planar_run_samples_inside_ball_circumscribing_box(self) -> None:
        X = np.array(
            [
                [-1.2, -0.4],
                [-0.8, 0.9],
                [-0.2, -1.1],
                [0.1, 0.2],
                [0.6, 1.0],
                [0.9, -0.7],
                [1.3, 0.3],
                [0.2, 1.4],
            ]
        )
        domain = LpBallDomain(p=2, radius=2.0, center=np.array([3.0, -4.0]))
        output = BoxEM().run(
            MeanEstimationMetadata(n_samples=X.shape[0], domain=domain),
            MeanEstimationInput(X=X),
            PureDPBudget(eps=2.0),
            seed=5,
        )
        assert output.mean_est.shape == (2,)
        assert np.all(output.mean_est >= [1.0, -6.0])
        assert np.all(output.mean_est <= [5.0, -2.0])

    def test_unconstrained_budget_samples_the_deepest_positive_volume_region(
        self,
    ) -> None:
        output = BoxEM().run(
            MeanEstimationMetadata(
                n_samples=3,
                domain=BoxDomain(bounds=np.array([[-4.0, 4.0]])),
            ),
            MeanEstimationInput(X=np.array([[-2.0], [0.0], [1.0]])),
            NO_PRIVACY,
            seed=7,
        )
        assert -2 <= output.mean_est[0] <= 1

    def test_public_refusals_happen_before_input_access(self) -> None:
        unreadable = MagicMock(spec=MeanEstimationInput)
        type(unreadable).X = PropertyMock(side_effect=AssertionError("read X"))
        invalid_metadata = [
            MeanEstimationMetadata(
                n_samples=4,
                domain=BoxDomain(bounds=np.tile([-1.0, 1.0], (3, 1))),
            ),
            MeanEstimationMetadata(
                n_samples=251,
                domain=BoxDomain(bounds=np.tile([-1.0, 1.0], (2, 1))),
            ),
            MeanEstimationMetadata(n_samples=3, domain=EuclideanSpace(dim=1)),
            MeanEstimationMetadata(
                n_samples=3,
                domain=BoxDomain(bounds=np.array([[-math.inf, math.inf]])),
            ),
            MeanEstimationMetadata(
                n_samples=3,
                domain=BoxDomain(bounds=np.array([[1.0, 1.0]])),
            ),
        ]
        for metadata in invalid_metadata:
            with pytest.raises(InvalidRegime):
                BoxEM().run(metadata, unreadable, PureDPBudget(eps=1.0), seed=0)

    @pytest.mark.parametrize(
        "budget",
        [
            PureDPBudget(eps=0.7),
            ApproxDPBudget(eps=0.7, delta=1e-5),
            ZCDPBudget(rho=0.245),
        ],
    )
    def test_accounting_conformance_uses_one_pure_spend(
        self, budget: DPBudget, accounting_spy: AccountingSpy
    ) -> None:
        BoxEM().run(
            MeanEstimationMetadata(
                n_samples=3,
                domain=BoxDomain(bounds=np.array([[-2.0, 2.0]])),
            ),
            MeanEstimationInput(X=np.array([[-1.0], [0.0], [1.0]])),
            budget,
            seed=0,
        )
        assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)
