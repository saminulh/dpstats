"""Unit tests for the public Domain geometry and its built-in wire form."""

import math
from typing import Any

import numpy as np
import pytest

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)


class TestEuclideanSpace:
    def test_preserves_dimension_without_claiming_finite_geometry(self):
        domain = EuclideanSpace(dim=3)
        point = np.array([1.0, -2.0, 4.0])

        np.testing.assert_array_equal(domain.center, np.zeros(3))
        assert domain.l1_diameter() == np.inf
        assert domain.l2_diameter() == np.inf
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.tile(np.array([[-np.inf, np.inf]]), (3, 1)),
        )
        assert domain.l2_norm_bound == np.inf
        center, radius = domain.l2_enclosing_ball()
        np.testing.assert_array_equal(center, np.zeros(3))
        assert radius == np.inf
        assert domain.project(point) is point

    def test_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            EuclideanSpace(dim=0)


class TestBoxDomain:
    def test_dim_property(self):
        bounds = np.array([[0, 1], [-1, 1], [0, 10]])
        box = BoxDomain(bounds=bounds)
        assert box.dim == 3

    def test_center_is_bounds_midpoint(self):
        box = BoxDomain(bounds=np.array([[-5.0, 1.0], [2.0, 8.0]]))
        np.testing.assert_allclose(box.center, [-2.0, 5.0])
        center, radius = box.l2_enclosing_ball()
        np.testing.assert_allclose(center, [-2.0, 5.0])
        assert radius == pytest.approx(np.sqrt(18.0))

    def test_is_its_own_bounding_box(self):
        box = BoxDomain(bounds=np.array([[-5.0, 1.0], [2.0, 8.0]]))
        assert box.bounding_box() is box

    def test_project_inside(self):
        box = BoxDomain(bounds=np.array([[0, 1], [0, 1]]))
        point = np.array([0.5, 0.5])
        assert np.allclose(box.project(point), point)

    def test_project_outside(self):
        box = BoxDomain(bounds=np.array([[0, 1], [0, 1]]))
        assert np.allclose(box.project(np.array([2.0, -1.0])), [1.0, 0.0])

    def test_project_batch(self):
        box = BoxDomain(bounds=np.array([[0, 1], [0, 1]]))
        points = np.array([[0.5, 0.5], [2.0, 2.0], [-1.0, 0.5]])
        expected = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.5]])
        assert np.allclose(box.project(points), expected)

    def test_l2_norm_bound_unit_box(self):
        box = BoxDomain(bounds=np.array([[0, 1], [0, 1]]))
        assert np.isclose(box.l2_norm_bound, np.sqrt(2.0))

    def test_l2_norm_bound_uses_farther_endpoint_per_coord(self):
        box = BoxDomain(bounds=np.array([[-5.0, 1.0], [-1.0, 3.0]]))
        assert np.isclose(box.l2_norm_bound, np.sqrt(34.0))

    def test_l2_norm_bound_is_sound_upper_bound(self):
        bounds = np.array([[-2.0, 5.0], [-7.0, 1.0], [0.0, 4.0]])
        box = BoxDomain(bounds=bounds)
        rng = np.random.default_rng(0)
        pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(1000, 3))
        assert np.all(np.linalg.norm(pts, axis=1) <= box.l2_norm_bound + 1e-9)

    @pytest.mark.parametrize(
        ("bounds", "violated"),
        [
            (np.empty((0, 2)), "shape"),
            (np.zeros((2, 3)), "shape"),
            (np.array([[np.nan, 1.0]]), "NaN"),
            (np.array([[2.0, 1.0]]), "must not exceed"),
            (np.array([[np.inf, np.inf]]), "positive infinity"),
            (np.array([[-np.inf, -np.inf]]), "negative infinity"),
        ],
    )
    def test_invalid_bounds_are_rejected(self, bounds, violated):
        with pytest.raises(ValueError, match=violated):
            BoxDomain(bounds=bounds)

    def test_half_infinite_box_has_total_geometry(self):
        domain = BoxDomain(bounds=np.array([[2.0, np.inf]]))
        np.testing.assert_array_equal(domain.center, [2.0])
        assert domain.l2_norm_bound == np.inf
        center, radius = domain.l2_enclosing_ball()
        np.testing.assert_array_equal(center, [2.0])
        assert radius == np.inf
        assert domain.bounding_box() is domain


@pytest.mark.parametrize(
    ("domain", "l2_diameter", "l1_diameter"),
    [
        # A [0, 3] x [0, 4] box: diagonal 5, side sum 7.
        (BoxDomain(bounds=np.array([[0, 3], [0, 4]])), 5.0, 7.0),
        # An l2 ball of radius r in d dimensions: 2r, and 2r * sqrt(d).
        (LpBallDomain.at_origin(p=2, radius=1.0, dim=4), 2.0, 4.0),
        # An l1 ball: the vertex pair +-r e_i spans 2r in both norms.
        (LpBallDomain.at_origin(p=1, radius=5.0, dim=10), 10.0, 10.0),
        # An l-infinity ball is a cube of side 2r: 2r * sqrt(d), and 2r * d.
        (LpBallDomain.at_origin(p=np.inf, radius=1.0, dim=4), 4.0, 8.0),
    ],
)
def test_diameters_match_the_declared_geometry(
    domain: Domain, l2_diameter: float, l1_diameter: float
):
    assert domain.l2_diameter() == pytest.approx(l2_diameter)
    assert domain.l1_diameter() == pytest.approx(l1_diameter)


class TestLpBallDomain:
    def test_origin_constructor_centers_the_ball_at_zero(self):
        ball = LpBallDomain.at_origin(p=2, radius=1.0, dim=3)
        assert ball.dim == 3
        np.testing.assert_array_equal(ball.center, np.zeros(3))

    def test_dimension_follows_the_declared_center(self):
        assert LpBallDomain(p=2, radius=1.0, center=np.ones(3)).dim == 3

    def test_project_l2_inside(self):
        ball = LpBallDomain.at_origin(p=2, radius=1.0, dim=3)
        point = np.array([0.1, 0.2, 0.3])
        assert np.allclose(ball.project(point), point)

    def test_project_l2_outside(self):
        ball = LpBallDomain.at_origin(p=2, radius=1.0, dim=2)
        projected = ball.project(np.array([2.0, 0.0]))
        assert np.isclose(np.linalg.norm(projected), 1.0)
        assert np.allclose(projected, [1.0, 0.0])

    def test_project_l2_with_center(self):
        center = np.array([1.0, 1.0])
        ball = LpBallDomain(p=2, radius=1.0, center=center)
        projected = ball.project(np.array([3.0, 1.0]))
        assert np.isclose(np.linalg.norm(projected - center), 1.0)
        assert np.allclose(projected, [2.0, 1.0])

    def test_project_linf(self):
        ball = LpBallDomain.at_origin(p=np.inf, radius=1.0, dim=2)
        assert np.allclose(ball.project(np.array([2.0, -3.0])), [1.0, -1.0])

    def test_project_l1_preserves_interior_and_boundary_points(self):
        ball = LpBallDomain.at_origin(p=1, radius=1.0, dim=2)
        points = np.array([[0.25, -0.5], [0.25, -0.75]])
        np.testing.assert_array_equal(ball.project(points), points)

    def test_project_l1_outside(self):
        ball = LpBallDomain.at_origin(p=1, radius=2.0, dim=2)
        projected = ball.project(np.array([3.0, -1.0]))
        np.testing.assert_allclose(projected, [2.0, 0.0])
        assert np.all(ball.contains(projected))

    def test_project_l1_batch_with_center(self):
        center = np.array([1.0, -1.0])
        ball = LpBallDomain(p=1, radius=1.0, center=center)
        points = np.array([[1.25, -1.25], [3.0, -1.0], [2.0, 0.0]])
        np.testing.assert_allclose(
            ball.project(points),
            np.array([[1.25, -1.25], [2.0, -1.0], [1.5, -0.5]]),
        )
        assert np.all(ball.contains(ball.project(points)))

    def test_project_l1_zero_radius_returns_center(self):
        center = np.array([1.0, -1.0])
        ball = LpBallDomain(p=1, radius=0.0, center=center)
        projected = ball.project(np.array([[1.0, -1.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(projected, np.tile(center, (2, 1)))

    @pytest.mark.parametrize("p", [1, 2, np.inf])
    def test_project_preserves_interior_points_exactly(self, p: float):
        ball = LpBallDomain(p=p, radius=2e16, center=np.array([1e16]))
        point = np.array([1.0])
        np.testing.assert_array_equal(ball.project(point), point)

    def test_project_batch(self):
        ball = LpBallDomain.at_origin(p=2, radius=1.0, dim=2)
        points = np.array([[0.1, 0.1], [2.0, 0.0], [0.0, 3.0]])
        projected = ball.project(points)
        assert np.allclose(projected[0], [0.1, 0.1])
        assert np.isclose(np.linalg.norm(projected[1]), 1.0)
        assert np.isclose(np.linalg.norm(projected[2]), 1.0)

    def test_unsupported_p_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must be 1, 2, or infinity"):
            LpBallDomain.at_origin(p=3, radius=1.0, dim=2)

    def test_l2_norm_bound_l2_ball_origin(self):
        assert np.isclose(
            LpBallDomain.at_origin(p=2, radius=5.0, dim=3).l2_norm_bound, 5.0
        )

    def test_l2_norm_bound_l2_ball_with_center(self):
        center = np.array([3.0, 4.0])
        ball = LpBallDomain(p=2, radius=1.0, center=center)
        assert np.isclose(ball.l2_norm_bound, 6.0)

    def test_l1_and_linf_balls_have_l2_bounds(self):
        assert LpBallDomain.at_origin(p=1, radius=1.0, dim=2).l2_norm_bound == 1.0
        assert LpBallDomain.at_origin(
            p=np.inf, radius=1.0, dim=2
        ).l2_norm_bound == pytest.approx(np.sqrt(2.0))

    @pytest.mark.parametrize(
        ("p", "expected_radius"),
        [(1, 3.0), (2, 3.0), (np.inf, 6.0)],
    )
    def test_l2_enclosing_ball_uses_declared_norm(self, p, expected_radius):
        center = np.array([4.0, -2.0, 1.0, 3.0])
        domain = LpBallDomain(p=p, radius=3.0, center=center)
        actual_center, actual_radius = domain.l2_enclosing_ball()
        np.testing.assert_array_equal(actual_center, center)
        assert actual_radius == pytest.approx(expected_radius)

    def test_bounding_box_is_coordinatewise_center_plus_or_minus_radius(self):
        domain = LpBallDomain(p=2, radius=3.0, center=np.array([4.0, -2.0]))
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.array([[1.0, 7.0], [-5.0, 1.0]]),
        )

    @pytest.mark.parametrize("radius", [np.inf, np.nan, -1.0])
    def test_radius_must_be_finite_and_nonnegative(self, radius):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            LpBallDomain.at_origin(p=2, radius=radius, dim=2)

    @pytest.mark.parametrize(
        ("center", "refusal"),
        [
            (np.array([]), r"LpBallDomain\.center must be a nonempty vector"),
            (np.zeros((1, 2)), "expected a 1D array, got 2D"),
            (np.array([np.nan]), r"LpBallDomain\.center must be finite"),
        ],
    )
    def test_center_must_be_a_nonempty_finite_vector(self, center, refusal):
        with pytest.raises(ValueError, match=refusal):
            LpBallDomain(p=2, radius=1.0, center=center)

    def test_origin_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            LpBallDomain.at_origin(p=2, radius=1.0, dim=0)


class TestOneHotDomain:
    def test_exactly_one_geometry(self):
        domain = OneHotDomain(dim=4)
        np.testing.assert_allclose(domain.center, np.full(4, 0.25))
        assert domain.l2_norm_bound == pytest.approx(1.0)
        assert np.isclose(domain.l2_diameter(), np.sqrt(2.0))
        assert domain.l1_diameter() == pytest.approx(2.0)
        center, radius = domain.l2_enclosing_ball()
        np.testing.assert_allclose(center, np.full(4, 0.25))
        assert radius == pytest.approx(np.sqrt(3 / 4))

    def test_singleton_exactly_one_domain_has_zero_diameter(self):
        domain = OneHotDomain(dim=1)
        np.testing.assert_allclose(domain.center, [1.0])
        assert domain.l2_diameter() == pytest.approx(0.0)
        assert domain.l1_diameter() == pytest.approx(0.0)

    def test_selecting_the_singleton_coordinate_preserves_its_exact_image(self):
        selected = OneHotDomain(dim=1).select_coordinates(np.array([0]))
        np.testing.assert_array_equal(
            selected.contains(np.array([[0.0], [1.0]])), [False, True]
        )
        np.testing.assert_array_equal(selected.bounding_box().bounds, [[1.0, 1.0]])

    @pytest.mark.parametrize(
        "allow_zero, coordinates, expected",
        [
            (False, [0, 1], [False, True, True, False]),
            (False, [1, 0], [False, True, True, False]),
            (True, [1, 0], [True, True, True, False]),
            (False, [1], [True, True, False]),
            (True, [1], [True, True, False]),
        ],
    )
    def test_coordinate_selection_preserves_exact_categorical_membership(
        self, allow_zero, coordinates, expected
    ):
        selected = OneHotDomain(dim=2, allow_zero=allow_zero).select_coordinates(
            np.array(coordinates)
        )
        candidates = (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            if len(coordinates) == 2
            else np.array([[0.0], [1.0], [2.0]])
        )
        np.testing.assert_array_equal(selected.contains(candidates), expected)

    def test_reference_coded_geometry_in_one_dimension(self):
        domain = OneHotDomain(dim=1, allow_zero=True)
        np.testing.assert_allclose(domain.center, [0.5])
        assert domain.l2_diameter() == pytest.approx(1.0)
        assert domain.l1_diameter() == pytest.approx(1.0)

    def test_bounding_box_preserves_singleton_and_reference_coding(self):
        np.testing.assert_array_equal(
            OneHotDomain(dim=1).bounding_box().bounds,
            np.array([[1.0, 1.0]]),
        )
        np.testing.assert_array_equal(
            OneHotDomain(dim=1, allow_zero=True).bounding_box().bounds,
            np.array([[0.0, 1.0]]),
        )
        np.testing.assert_array_equal(
            OneHotDomain(dim=3).bounding_box().bounds,
            np.tile(np.array([[0.0, 1.0]]), (3, 1)),
        )

    def test_reference_coded_enclosing_ball_contains_every_category(self):
        domain = OneHotDomain(dim=3, allow_zero=True)
        center, radius = domain.l2_enclosing_ball()
        supported_points = np.vstack((np.zeros(3), np.eye(3)))
        assert np.all(
            np.linalg.norm(supported_points - center, axis=1) <= radius + 1e-12
        )

    def test_projection_selects_nearest_exact_category(self):
        domain = OneHotDomain(dim=3)
        points = np.array([[0.1, 0.8, 0.2], [3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(
            domain.project(points),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    def test_reference_projection_can_select_zero(self):
        domain = OneHotDomain(dim=3, allow_zero=True)
        points = np.array([[0.1, 0.4, 0.2], [0.1, 0.6, 0.2]])
        np.testing.assert_array_equal(
            domain.project(points),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )

    def test_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            OneHotDomain(dim=0)


class TestProductDomain:
    def test_mixed_box_and_one_hot_geometry(self):
        box = BoxDomain(bounds=np.array([[-2.0, 2.0], [10.0, 14.0]]))
        category = OneHotDomain(dim=3)
        domain = ProductDomain(parts=(box, category))

        assert domain.dim == 5
        np.testing.assert_allclose(
            domain.center,
            np.array([0.0, 12.0, 1 / 3, 1 / 3, 1 / 3]),
        )
        assert np.isclose(domain.l2_norm_bound, np.sqrt(201.0))
        assert np.isclose(domain.l2_diameter(), np.sqrt(34.0))
        assert domain.l1_diameter() == pytest.approx(10.0)
        center, radius = domain.l2_enclosing_ball()
        np.testing.assert_allclose(center, domain.center)
        assert radius == pytest.approx(np.sqrt(26 / 3))

    def test_component_bounds_compose(self):
        domain = ProductDomain(
            parts=(
                LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
                LpBallDomain.at_origin(p=1, radius=1.0, dim=2),
            )
        )
        assert domain.l2_norm_bound == pytest.approx(np.sqrt(2.0))

    def test_convex_enclosure_keeps_convex_parts_and_boxes_the_rest(self):
        """A convex domain is its own enclosure; a one-hot block becomes its box."""
        ball = LpBallDomain(p=2, radius=2.0, center=np.array([1.0, -1.0]))
        box = BoxDomain(bounds=np.array([[0.0, 1.0]]))
        euclidean = EuclideanSpace(dim=3)
        intercept = InterceptDomain()
        for convex in (ball, box, euclidean, intercept):
            assert convex.convex_enclosure() is convex
        one_hot = OneHotDomain(dim=2, allow_zero=True)
        enclosure = one_hot.convex_enclosure()
        assert isinstance(enclosure, BoxDomain)
        np.testing.assert_array_equal(enclosure.bounds, np.array([[0.0, 1.0]] * 2))
        product = ProductDomain(parts=(ball, one_hot)).convex_enclosure()
        assert isinstance(product, ProductDomain)
        assert product.parts[0] is ball
        assert isinstance(product.parts[1], BoxDomain)

    def test_bounding_box_concatenates_component_enclosures(self):
        domain = ProductDomain(
            parts=(
                LpBallDomain(p=2, radius=2.0, center=np.array([1.0, -1.0])),
                OneHotDomain(dim=2),
            )
        )
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.array(
                [
                    [-1.0, 3.0],
                    [-3.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
        )

    def test_unbounded_component_bounding_box_propagates_infinite_bounds(self):
        domain = ProductDomain(
            parts=(
                BoxDomain(bounds=np.array([[0.0, 1.0]])),
                EuclideanSpace(dim=2),
            )
        )
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.array([[0.0, 1.0], [-np.inf, np.inf], [-np.inf, np.inf]]),
        )

    def test_projection_respects_coordinate_blocks(self):
        domain = ProductDomain(
            parts=(
                BoxDomain(bounds=np.array([[0.0, 1.0]])),
                OneHotDomain(dim=3),
            )
        )
        point = np.array([2.0, 0.1, 0.8, 0.2])
        np.testing.assert_array_equal(
            domain.project(point),
            np.array([1.0, 0.0, 1.0, 0.0]),
        )

    def test_coordinate_selection_preserves_block_geometry(self):
        domain = ProductDomain(
            parts=(
                BoxDomain(bounds=np.array([[-2.0, 2.0], [10.0, 14.0]])),
                OneHotDomain(dim=3),
            )
        )
        selected = domain.select_coordinates(np.array([1, 4, 2], dtype=np.int64))

        assert isinstance(selected, ProductDomain)
        box, category = selected.parts
        assert isinstance(box, BoxDomain)
        np.testing.assert_array_equal(box.bounds, [[10.0, 14.0]])
        assert category == OneHotDomain(dim=2, allow_zero=True)

    def test_coordinate_selection_cannot_reorder_product_blocks(self):
        domain = ProductDomain(parts=(EuclideanSpace(dim=1), EuclideanSpace(dim=1)))
        with pytest.raises(ValueError, match="reorder product blocks"):
            domain.select_coordinates(np.array([1, 0], dtype=np.int64))

    def test_product_must_be_nonempty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ProductDomain(parts=())


class TestDomainDefaults:
    def test_unimplemented_finite_geometry_defaults_to_infinity(self):
        class TrivialDomain(Domain):
            dim = 1

            @property
            def center(self) -> np.ndarray:
                return np.zeros(1)

            def l2_diameter(self) -> float:
                return 0.0

            def l1_diameter(self) -> float:
                return 0.0

            def project(self, x: np.ndarray) -> np.ndarray:
                return x

            def contains(self, values: np.ndarray) -> bool:
                return bool(np.all(np.isfinite(values)))

            def select_coordinates(self, indices: np.ndarray) -> Domain:
                del indices
                return self

        domain = TrivialDomain()
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.array([[-np.inf, np.inf]]),
        )
        assert domain.l2_norm_bound == np.inf
        center, radius = domain.l2_enclosing_ball()
        np.testing.assert_array_equal(center, [0.0])
        assert radius == np.inf


class TestProductDomainFlattening:
    def test_nested_products_flatten_at_construction(self):
        inner = ProductDomain(
            parts=(EuclideanSpace(dim=2), BoxDomain(bounds=np.array([[0.0, 1.0]])))
        )
        outer = ProductDomain(parts=(inner, EuclideanSpace(dim=1)))

        assert all(not isinstance(part, ProductDomain) for part in outer.parts)
        assert len(outer.parts) == 3
        assert outer.dim == 4

    def test_nested_and_flat_spellings_agree_on_geometry(self):
        box = BoxDomain(bounds=np.array([[-1.0, 1.0], [-2.0, 2.0]]))
        flat = ProductDomain(parts=(box, EuclideanSpace(dim=1)))
        nested = ProductDomain(
            parts=(ProductDomain(parts=(box,)), EuclideanSpace(dim=1))
        )

        assert nested.dim == flat.dim
        assert nested.l1_diameter() == flat.l1_diameter()
        assert nested.parts == flat.parts


class TestMembership:
    def test_euclidean_space_holds_every_real_point_but_no_nonfinite_one(self):
        domain = EuclideanSpace(dim=2)
        assert np.all(domain.contains(np.array([[0.0, 0.0], [1e12, -1e12]])))
        assert np.all(domain.contains(np.array([3.0, -4.0])))
        assert not np.all(domain.contains(np.array([[0.0, np.nan]])))
        assert not np.all(domain.contains(np.array([[0.0, np.inf]])))

    def test_box_domain_compares_each_coordinate_against_its_own_bounds(self):
        domain = BoxDomain(bounds=np.array([[0.0, 10.0], [-1.0, 1.0]]))
        assert np.all(domain.contains(np.array([[5.0, 0.0], [1.0, -0.5]])))
        assert np.all(domain.contains(np.array([[0.0, -1.0], [10.0, 1.0]])))
        assert not np.all(domain.contains(np.array([[5.0, 1.5]])))
        assert not np.all(domain.contains(np.array([[-0.5, 0.0]])))
        assert not np.all(domain.contains(np.array([[5.0, np.nan]])))

    def test_box_domain_refuses_an_infinite_value_under_an_infinite_bound(self):
        domain = BoxDomain(bounds=np.array([[0.0, np.inf]]))
        assert np.all(domain.contains(np.array([[1e300]])))
        assert not np.all(domain.contains(np.array([[np.inf]])))

    @pytest.mark.parametrize("p", [1, 2, np.inf])
    def test_lp_ball_membership_is_available_for_every_supported_norm(self, p):
        domain = LpBallDomain.at_origin(p=p, radius=2.0, dim=3)
        assert np.all(domain.contains(np.array([[1.0, 0.0, 0.0]])))
        assert np.all(domain.contains(np.array([[2.0, 2.0, 2.0]])) == (p == np.inf))
        assert not np.all(domain.contains(np.array([[3.0, 3.0, 3.0]])))
        assert not np.all(domain.contains(np.array([[np.nan, 0.0, 0.0]])))

    def test_lp_ball_l1_membership_uses_the_sum_of_magnitudes(self):
        domain = LpBallDomain.at_origin(p=1, radius=1.0, dim=2)
        assert np.all(domain.contains(np.array([[0.5, 0.5]])))
        assert np.all(domain.contains(np.array([[0.25, -0.5]])))
        assert not np.all(domain.contains(np.array([[0.6, 0.6]])))

    def test_lp_ball_tolerates_epsilon_scale_overshoot_but_not_material_violation(
        self,
    ):
        radius = 5.0
        domain = LpBallDomain.at_origin(p=2, radius=radius, dim=1)
        eps = float(np.finfo(np.float64).eps)
        assert np.all(domain.contains(np.array([[radius]])))
        assert np.all(domain.contains(np.array([[radius * (1 + 4 * eps)]])))
        assert not np.all(domain.contains(np.array([[radius * 1.01]])))
        assert not np.all(domain.contains(np.array([[radius * (1 + 1e-9)]])))

    def test_lp_ball_membership_is_measured_from_the_declared_center(self):
        domain = LpBallDomain(p=2, radius=1.0, center=np.array([10.0, 0.0]))
        assert np.all(domain.contains(np.array([[10.5, 0.0]])))
        assert not np.all(domain.contains(np.array([[0.0, 0.0]])))

    def test_one_hot_domain_requires_an_exactly_encoded_category(self):
        domain = OneHotDomain(dim=3)
        assert np.all(domain.contains(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])))
        assert np.all(domain.contains(np.array([0.0, 1.0, 0.0])))
        assert not np.all(domain.contains(np.array([[0.999, 0.0, 0.0]])))
        assert not np.all(domain.contains(np.array([[1.0, 1.0, 0.0]])))
        assert not np.all(domain.contains(np.array([[0.0, 0.0, 0.0]])))
        assert not np.all(domain.contains(np.array([[np.nan, 0.0, 0.0]])))

    def test_one_hot_domain_admits_the_reference_row_only_when_declared(self):
        reference_row = np.array([[0.0, 0.0]])
        assert OneHotDomain(dim=2, allow_zero=True).contains(reference_row)
        assert not OneHotDomain(dim=2).contains(reference_row)

    def test_product_domain_requires_every_block_to_hold_its_own_slice(self):
        domain = ProductDomain(
            parts=(OneHotDomain(dim=2), BoxDomain(bounds=np.array([[0.0, 1.0]])))
        )
        assert np.all(domain.contains(np.array([[1.0, 0.0, 0.5]])))
        assert not np.all(domain.contains(np.array([[0.5, 0.5, 0.5]])))
        assert not np.all(domain.contains(np.array([[1.0, 0.0, 2.0]])))
        assert not np.all(domain.contains(np.array([[1.0, 0.0, np.nan]])))

    def test_membership_of_several_points_requires_all_of_them(self):
        domain = BoxDomain(bounds=np.array([[0.0, 1.0]]))
        assert np.all(domain.contains(np.array([[0.1], [0.9]])))
        assert not np.all(domain.contains(np.array([[0.1], [1.9]])))


DOMAINS = [
    EuclideanSpace(dim=3),
    BoxDomain(bounds=[[0.0, 1.0], [-2.0, 2.0]]),
    BoxDomain(bounds=[[-math.inf, 5.0], [0.0, math.inf]]),
    LpBallDomain(p=1.0, radius=1.5, center=[0.0, 0.0]),
    LpBallDomain(p=2.0, radius=0.5, center=[1.0, -1.0]),
    LpBallDomain(p=math.inf, radius=2.0, center=[0.0]),
    OneHotDomain(dim=3, allow_zero=True),
    ProductDomain(
        parts=(
            BoxDomain(bounds=[[0.0, 1.0]]),
            OneHotDomain(dim=3, allow_zero=True),
            InterceptDomain(),
            EuclideanSpace(dim=2),
        )
    ),
]


def _dump_domain(domain: Domain) -> dict[str, Any]:
    return decode_json_document(canonical_json_bytes(domain.model_dump(mode="python")))


class TestDomainConfiguration:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_every_domain_round_trips_through_shared_configuration(self, domain):
        restored = Domain.load_record(domain.dump_config())
        assert type(restored) is type(domain)
        assert _dump_domain(restored) == _dump_domain(domain)

    def test_origin_constructor_emits_explicit_center(self):
        domain = LpBallDomain.at_origin(p=2, radius=1.0, dim=3)
        assert _dump_domain(domain) == {
            "p": 2.0,
            "radius": 1.0,
            "center": [0.0, 0.0, 0.0],
        }

    def test_omitted_one_hot_reference_category_normalizes_to_false(self):
        domain = OneHotDomain(dim=2)
        assert _dump_domain(domain) == {
            "dim": 2,
            "allow_zero": False,
        }

    @pytest.mark.parametrize(
        "bounds", [[[18, 90]], np.array([[18, 90]], dtype=np.int64)]
    )
    def test_integer_bounds_normalize_to_float(self, bounds):
        domain = BoxDomain(bounds=bounds)
        assert domain.bounds.dtype == np.dtype(float)
        rewritten = _dump_domain(domain)
        assert rewritten == {"bounds": [[18.0, 90.0]]}
        assert canonical_json_bytes(rewritten) == (b'{"bounds":[[18.0,90.0]]}')

    def test_unbounded_box_uses_canonical_json_tokens(self):
        record = _dump_domain(EuclideanSpace(dim=2).bounding_box())
        assert record == {
            "bounds": [
                [-math.inf, math.inf],
                [-math.inf, math.inf],
            ],
        }
        assert canonical_json_bytes(record) == (
            b'{"bounds":[["-inf_json","+inf_json"],' b'["-inf_json","+inf_json"]]}'
        )

    def test_unknown_configuration_path_is_refused(self):
        with pytest.raises(ValueError, match="use 'module:qualname'"):
            Domain.load_record({"path": "HyperbolicDomain", "config": {"dim": 2}})

    @pytest.mark.parametrize("parts", [{"path": "BoxDomain", "config": {}}, ["box"]])
    def test_invalid_product_parts_are_refused(self, parts):
        with pytest.raises(ValueError):
            ProductDomain(parts=parts)


class CustomDomain(Domain):
    """Example extension owned entirely by a custom task."""

    scale: float

    @property
    def dim(self) -> int:
        return 1

    @property
    def center(self):
        return np.zeros(1)

    def l2_diameter(self) -> float:
        return 0.0

    def l1_diameter(self) -> float:
        return 0.0

    def project(self, x):
        return x

    def contains(self, values):
        return np.atleast_1d(np.all(np.isfinite(values), axis=-1))

    def select_coordinates(self, indices):
        self._validate_coordinate_selection(indices)
        return self


def test_custom_domain_round_trips_without_core_registration():
    domain = CustomDomain(scale=2)

    restored = Domain.load_record(
        decode_json_document(canonical_json_bytes(domain.dump_config().model_dump()))
    )

    assert isinstance(restored, CustomDomain)
    assert restored.scale == 2.0
