"""Unit tests for the domain-metadata transformations."""

import math
import tracemalloc
from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import ApproxDPBudget, DPBudget, NO_PRIVACY
from dpstats.exceptions import InvalidRegime

from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.linear_regression.transformations import (
    DataBounds,
    Standardize,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from tests.unit.tasks.task_builders import box, lr_metadata


class _SpyAlgorithm(BaseAlgorithm):
    """Record the prepared instance and return a marker vector."""

    TASK = LinearRegression

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    record: dict[str, Any] | None = None

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        if self.record is not None:
            self.record["metadata"] = metadata
            self.record["input"] = input
            self.record["budget"] = budget
            self.record["seed"] = seed
        output = LinearRegressionOutput(beta=np.arange(metadata.dim, dtype=float))
        if self.record is not None:
            self.record["output"] = output
        return output


def _record(composition: CompositeAlgorithm) -> dict[str, Any]:
    """Return the concrete spy child's run record."""
    assert isinstance(composition.algorithm, _SpyAlgorithm)
    assert composition.algorithm.record is not None
    return composition.algorithm.record


def _bounded_instance(
    has_intercept: bool,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    """Build a small instance with declared box bounds."""
    rng = np.random.default_rng(7)
    n = 40
    covariates = rng.uniform(-2.0, 3.0, size=(n, 2))
    if has_intercept:
        X = np.column_stack((np.ones(n), covariates))
        design_domain = ProductDomain(
            parts=(InterceptDomain(), box((-2.0, 3.0), (-2.0, 3.0)))
        )
    else:
        X = covariates
        design_domain = box((-2.0, 3.0), (-2.0, 3.0))
    y = rng.uniform(-5.0, 5.0, size=n)
    metadata = lr_metadata(n, design=design_domain, response=box((-5.0, 5.0)))
    return metadata, LinearRegressionInput(X=X, y=y)


class TestDataBounds:
    """Empirical numeric boxes retain the design's structural domain blocks."""

    def test_metadata_only_preparation_does_not_copy_product_domain_records(self):
        n, dim = 32769, 32
        metadata = lr_metadata(
            n,
            design=ProductDomain(
                parts=(InterceptDomain(), EuclideanSpace(dim=dim - 1))
            ),
        )
        input_data = LinearRegressionInput(X=np.ones((n, dim)), y=np.ones(n))
        tracemalloc.start()
        try:
            prepared, forwarded, _, _, _ = DataBounds().prepare(
                metadata, input_data, NO_PRIVACY, seed=0
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert forwarded is input_data
        np.testing.assert_array_equal(
            prepared.design_domain.bounding_box().bounds, np.ones((dim, 2))
        )
        assert peak < input_data.X.nbytes / 4

    def test_mixed_product_domain_is_bounded_without_losing_structure(self):
        intercept = InterceptDomain()
        continuous = BoxDomain(bounds=np.array([[-20.0, 20.0], [-5.0, 50.0]]))
        category = OneHotDomain(dim=3, allow_zero=True)
        unbounded = EuclideanSpace(dim=1)
        design_domain = ProductDomain(
            parts=(intercept, continuous, category, unbounded)
        )
        metadata = lr_metadata(4, design=design_domain)
        X = np.array(
            [
                [1.0, -4.0, 10.0, 0.0, 0.0, 0.0, 7.0],
                [1.0, 2.0, 12.0, 1.0, 0.0, 0.0, 7.0],
                [1.0, -1.0, 11.0, 0.0, 1.0, 0.0, 7.0],
                [1.0, 5.0, 15.0, 0.0, 0.0, 1.0, 7.0],
            ]
        )
        y = np.array([-3.0, 0.0, 8.0, 2.0])
        task_input = LinearRegressionInput(X=X, y=y)
        record: dict[str, Any] = {}
        offered = ApproxDPBudget(eps=2.0, delta=1e-6)

        composition = CompositeAlgorithm(
            transformations=(DataBounds(),), algorithm=_SpyAlgorithm(record=record)
        )
        output = composition.run(metadata, task_input, budget=offered, seed=19)

        record = _record(composition)
        prepared = record["metadata"]
        assert isinstance(prepared.design_domain, ProductDomain)
        prepared_intercept, prepared_continuous, prepared_category, prepared_numeric = (
            prepared.design_domain.parts
        )
        assert prepared_intercept is intercept
        assert isinstance(prepared_continuous, BoxDomain)
        np.testing.assert_array_equal(
            prepared_continuous.bounds, [[-4.0, 5.0], [10.0, 15.0]]
        )
        assert prepared_category is category
        assert isinstance(prepared_numeric, BoxDomain)
        np.testing.assert_array_equal(prepared_numeric.bounds, [[7.0, 7.0]])
        assert isinstance(prepared.response_domain, BoxDomain)
        np.testing.assert_array_equal(prepared.response_domain.bounds, [[-3.0, 8.0]])
        np.testing.assert_array_equal(prepared.design_domain.project(X), X)
        np.testing.assert_array_equal(prepared.response_domain.project(y), y)

        assert metadata.design_domain is design_domain
        assert isinstance(metadata.response_domain, EuclideanSpace)
        assert record["input"] is task_input
        assert record["budget"] == offered
        assert record["seed"] == 19
        assert output is record["output"]

    def test_zero_width_plain_coordinates_use_singleton_boxes(self):
        metadata = lr_metadata(3, design=2)
        task_input = LinearRegressionInput(
            X=np.array([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]]),
            y=np.array([4.0, 4.0, 4.0]),
        )
        record: dict[str, Any] = {}

        composition = CompositeAlgorithm(
            transformations=(DataBounds(),), algorithm=_SpyAlgorithm(record=record)
        )
        composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-6), seed=0
        )

        record = _record(composition)
        prepared = record["metadata"]
        assert isinstance(prepared.design_domain, BoxDomain)
        np.testing.assert_array_equal(
            prepared.design_domain.bounds, [[2.0, 2.0], [-1.0, -1.0]]
        )
        np.testing.assert_array_equal(prepared.response_domain.bounds, [[4.0, 4.0]])

    @pytest.mark.parametrize(
        "task_input",
        [
            LinearRegressionInput(
                X=np.array([[1.0, np.nan], [2.0, 3.0]]),
                y=np.array([1.0, 2.0]),
            ),
            LinearRegressionInput(
                X=np.array([[1.0, 2.0], [2.0, 3.0]]),
                y=np.array([1.0, np.inf]),
            ),
        ],
        ids=["nan-design", "infinite-response"],
    )
    def test_nonfinite_input_fails_without_claiming_a_regime_limit(
        self, task_input: LinearRegressionInput
    ) -> None:
        """A nonfinite sample must not be refused as a public regime limit.

        Collection publishes an ``InvalidRegime`` message verbatim, so a
        refusal conditioned on the realized sample would publish one bit of it.
        """
        metadata = lr_metadata(2, design=2)
        record: dict[str, Any] = {}

        with pytest.raises(ValueError) as failure:
            CompositeAlgorithm(
                transformations=(DataBounds(),), algorithm=_SpyAlgorithm(record=record)
            ).run(
                metadata,
                task_input,
                budget=ApproxDPBudget(eps=1.0, delta=1e-6),
                seed=0,
            )

        assert not isinstance(failure.value, InvalidRegime)
        assert record == {}


class TestStandardize:
    """Public affine maps preserve the represented regression problem."""

    def test_centers_numeric_blocks_and_response_with_intercept(self) -> None:
        intercept = InterceptDomain()
        category = OneHotDomain(dim=2, allow_zero=True)
        metadata = lr_metadata(
            3,
            design=ProductDomain(parts=(intercept, box((-2.0, 4.0)), category)),
            response=box((-5.0, 15.0)),
        )
        task_input = LinearRegressionInput(
            X=np.array(
                [
                    [1.0, -2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 1.0],
                ]
            ),
            y=np.array([-5.0, 5.0, 15.0]),
        )
        record: dict[str, Any] = {}

        composition = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
        )
        output = composition.run(
            metadata,
            task_input,
            budget=ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=4,
        )

        record = _record(composition)
        prepared = record["metadata"]
        prepared_input = record["input"]
        assert isinstance(prepared.design_domain, ProductDomain)
        prepared_intercept, prepared_numeric, prepared_category = (
            prepared.design_domain.parts
        )
        assert prepared_intercept is intercept
        assert prepared_category is category
        assert isinstance(prepared_numeric, BoxDomain)
        np.testing.assert_allclose(prepared_numeric.bounds, [[-1.0, 1.0]])
        np.testing.assert_allclose(prepared.response_domain.bounds, [[-1.0, 1.0]])
        np.testing.assert_allclose(
            prepared_input.X,
            [[1.0, -1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(prepared_input.y, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            output.beta,
            [5.0 - 10.0 / 3.0, 10.0 / 3.0, 20.0, 30.0],
        )

    def test_scales_without_centering_through_origin(self) -> None:
        metadata = lr_metadata(
            3, design=box((2.0, 6.0), (-4.0, 4.0)), response=box((10.0, 30.0))
        )
        task_input = LinearRegressionInput(
            X=np.array([[2.0, -4.0], [4.0, 0.0], [6.0, 4.0]]),
            y=np.array([10.0, 20.0, 30.0]),
        )
        record: dict[str, Any] = {}

        composition = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
        )
        output = composition.run(
            metadata,
            task_input,
            budget=ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )

        record = _record(composition)
        prepared = record["metadata"]
        prepared_input = record["input"]
        np.testing.assert_allclose(
            prepared.design_domain.bounds, [[1.0, 3.0], [-1.0, 1.0]]
        )
        np.testing.assert_allclose(prepared.response_domain.bounds, [[1.0, 3.0]])
        np.testing.assert_allclose(
            prepared_input.X, [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]]
        )
        np.testing.assert_allclose(prepared_input.y, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(output.beta, [0.0, 2.5])

    def test_l2_balls_use_their_public_center_and_radius(self) -> None:
        metadata = lr_metadata(
            2,
            design=ProductDomain(
                parts=(
                    InterceptDomain(),
                    LpBallDomain(p=2, radius=2.0, center=np.array([3.0, -1.0])),
                )
            ),
            response=LpBallDomain(p=2, radius=4.0, center=np.array([6.0])),
        )
        task_input = LinearRegressionInput(
            X=np.array([[1.0, 3.0, -1.0], [1.0, 5.0, -1.0]]),
            y=np.array([6.0, 10.0]),
        )
        record: dict[str, Any] = {}

        composition = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
        )
        composition.run(
            metadata,
            task_input,
            budget=ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )

        record = _record(composition)
        prepared = record["metadata"]
        numeric = prepared.design_domain.parts[1]
        assert isinstance(numeric, LpBallDomain)
        np.testing.assert_array_equal(numeric.center, [0.0, 0.0])
        assert numeric.radius == pytest.approx(math.sqrt(2))
        assert isinstance(prepared.response_domain, LpBallDomain)
        np.testing.assert_array_equal(prepared.response_domain.center, [0.0])
        assert prepared.response_domain.radius == 1.0

    def test_singleton_numeric_domains_pass_through(self) -> None:
        metadata = lr_metadata(
            2,
            design=ProductDomain(parts=(InterceptDomain(), box((7.0, 7.0)))),
            response=box((4.0, 4.0)),
        )
        task_input = LinearRegressionInput(
            X=np.array([[1.0, 7.0], [1.0, 7.0]]),
            y=np.array([4.0, 4.0]),
        )
        record: dict[str, Any] = {}

        composition = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
        )
        composition.run(
            metadata,
            task_input,
            budget=ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )

        record = _record(composition)
        np.testing.assert_array_equal(record["input"].X, task_input.X)
        np.testing.assert_array_equal(record["input"].y, task_input.y)

    @pytest.mark.parametrize("unbounded", ["design", "response"])
    def test_refuses_unbounded_numeric_domains_before_reading_input(
        self, unbounded: str
    ) -> None:
        metadata = lr_metadata(
            2,
            design=2 if unbounded == "design" else box((-1.0, 1.0), (-1.0, 1.0)),
            response=1 if unbounded == "response" else box((-1.0, 1.0)),
        )
        task_input = LinearRegressionInput(X=np.full((2, 2), np.nan), y=np.ones(2))
        record: dict[str, Any] = {}

        with pytest.raises(InvalidRegime, match="bounded"):
            CompositeAlgorithm(
                transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
            ).run(
                metadata,
                task_input,
                budget=ApproxDPBudget(eps=1.0, delta=1e-6),
                seed=0,
            )

        assert record == {}

    @pytest.mark.parametrize("has_intercept", [True, False])
    def test_ols_fit_is_invariant_in_original_coordinates(
        self, has_intercept: bool
    ) -> None:
        metadata, task_input = _bounded_instance(has_intercept)

        direct = OLS().run(metadata, task_input, budget=NO_PRIVACY, seed=0)
        standardized = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=OLS()
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        np.testing.assert_allclose(standardized.beta, direct.beta, atol=1e-12)

    def test_privacy_interpretations_compose_by_weakest_component(self) -> None:
        diagnostic = CompositeAlgorithm(
            transformations=(DataBounds(), Standardize()),
            algorithm=_SpyAlgorithm(record={}),
        )
        nonprivate = CompositeAlgorithm(
            transformations=(DataBounds(), Standardize()),
            algorithm=OLS(),
        )

        assert Standardize().privacy_interpretation is PrivacyInterpretation.STANDARD
        assert diagnostic.privacy_interpretation is PrivacyInterpretation.DIAGNOSTIC
        assert nonprivate.privacy_interpretation is PrivacyInterpretation.NONPRIVATE
