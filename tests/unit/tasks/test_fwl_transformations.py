"""Unit tests for LargestOneHotFWLDemean through the public CompositeAlgorithm surface."""

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import NotApplicable
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    NO_PRIVACY,
)

from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Array1D
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.single_coefficient_estimation.transformations.fwl import (
    LargestOneHotFWLDemean,
)
from dpstats.tasks.single_coefficient_estimation.transformations.linear_regression import (
    ToLinearRegression,
)
from dpstats.tasks import (
    SingleCoefficientEstimation,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)


class _SpyChild(BaseAlgorithm):
    """Native SCE stand-in capturing the prepared problem it receives."""

    TASK = SingleCoefficientEstimation

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    record: dict[str, Any] | None = None

    def run(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> SingleCoefficientEstimationOutput:
        if self.record is not None:
            self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        return SingleCoefficientEstimationOutput(coefficient=41.5)


def _one_hot(labels: list[int], width: int) -> np.ndarray:
    block = np.zeros((len(labels), width))
    for row, label in enumerate(labels):
        if label < width:
            block[row, label] = 1.0
    return block


def _sce(coi: int, parts: tuple[Any, ...]) -> SingleCoefficientEstimationMetadata:
    return SingleCoefficientEstimationMetadata(
        n_samples=3,
        coefficient_of_interest_index=coi,
        design_domain=ProductDomain(parts=parts),
        response_domain=BoxDomain(bounds=np.array([[0.0, 8.0]])),
    )


def _exact_block_instance():
    """Intercept + exactly-one two-group block + one covariate, COI on it.

    Groups: rows 0 and 1 share group 0; row 2 is a singleton group 1.
    Covariate x = (1, 3, 5), response y = (2, 6, 7). Hand-derived demeaning:
    group-0 means are (2, 4), so demeaned x = (-1, 1, 0), y = (-2, 2, 0);
    the singleton group demeans exactly to zero.
    """
    labels = [0, 0, 1]
    X = np.column_stack((np.ones(3), _one_hot(labels, 2), np.array([1.0, 3.0, 5.0])))
    y = np.array([2.0, 6.0, 7.0])
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=3,
        coefficient_of_interest_index=3,
        design_domain=ProductDomain(
            parts=(
                InterceptDomain(),
                OneHotDomain(dim=2),
                BoxDomain(bounds=np.array([[0.0, 6.0]])),
            )
        ),
        response_domain=BoxDomain(bounds=np.array([[0.0, 8.0]])),
    )
    return metadata, SingleCoefficientEstimationInput(X=X, y=y)


def _run_spy(
    metadata: SingleCoefficientEstimationMetadata,
    task_input: SingleCoefficientEstimationInput,
    budget: DPBudget = NO_PRIVACY,
    seed: int = 0,
) -> tuple[SingleCoefficientEstimationOutput, dict[str, Any]]:
    record: dict[str, Any] = {}
    composition = CompositeAlgorithm(
        transformations=(LargestOneHotFWLDemean(),),
        algorithm=_SpyChild(record=record),
    )
    output = composition.run(metadata, task_input, budget=budget, seed=seed)
    assert composition.algorithm.record is not None
    return output, composition.algorithm.record


class TestExactDemeaning:
    """Hand-computed transforms, group conventions, and pass-through."""

    def test_hand_computed_full_one_hot_block(self):
        metadata, task_input = _exact_block_instance()

        output, record = _run_spy(metadata, task_input)

        np.testing.assert_allclose(record["input"].X, [[-1.0], [1.0], [0.0]])
        np.testing.assert_allclose(record["input"].y, [-2.0, 2.0, 0.0])
        assert record["metadata"].coefficient_of_interest_index == 0
        assert record["metadata"].n_samples == 3
        # The child's scalar release is returned unchanged.
        assert output.coefficient == 41.5

    def test_reference_coded_block_with_intercept(self):
        # allow_zero block of width 2 with a declared intercept: rows 0 and 3
        # are the all-zero reference group (means x = 3, y = 3); the other
        # labels appear once, so those singleton groups demean to zero.
        labels = [2, 0, 1, 2]
        X = np.column_stack(
            (np.ones(4), _one_hot(labels, 2), np.array([2.0, 5.0, 9.0, 4.0]))
        )
        y = np.array([1.0, 3.0, 8.0, 5.0])
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=4,
            coefficient_of_interest_index=3,
            design_domain=ProductDomain(
                parts=(
                    InterceptDomain(),
                    OneHotDomain(dim=2, allow_zero=True),
                    BoxDomain(bounds=np.array([[0.0, 10.0]])),
                )
            ),
            response_domain=BoxDomain(bounds=np.array([[0.0, 10.0]])),
        )

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        np.testing.assert_allclose(record["input"].X, [[-1.0], [0.0], [0.0], [1.0]])
        np.testing.assert_allclose(record["input"].y, [-2.0, 0.0, 0.0, 2.0])
        assert record["metadata"].coefficient_of_interest_index == 0

    def test_budget_and_seed_pass_through_unchanged(self):
        metadata, task_input = _exact_block_instance()
        budget = ApproxDPBudget(eps=1.5, delta=1e-6)

        _, record = _run_spy(metadata, task_input, budget=budget, seed=17)

        assert record["budget"] is budget
        assert record["seed"] == 17


class TestSelectionPolicy:
    """Public, deterministic block selection from metadata alone."""

    def test_largest_eligible_block_is_selected(self):
        # Two nuisance blocks of widths 2 and 3: the width-3 block is
        # absorbed, the width-2 block is retained as a covariate.
        wide = [0, 1, 2, 0, 1, 2]
        narrow = [0, 1, 0, 1, 0, 1]
        x = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        X = np.column_stack((_one_hot(narrow, 2), _one_hot(wide, 3), x))
        y = np.arange(6.0)
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=6,
            coefficient_of_interest_index=5,
            design_domain=ProductDomain(
                parts=(
                    OneHotDomain(dim=2),
                    OneHotDomain(dim=3),
                    BoxDomain(bounds=np.array([[0.0, 8.0]])),
                )
            ),
            response_domain=BoxDomain(bounds=np.array([[0.0, 8.0]])),
        )

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        # Retained: the width-2 block plus the covariate; COI remaps to 2.
        assert record["input"].X.shape == (6, 3)
        assert record["metadata"].coefficient_of_interest_index == 2
        # Demeaning grouped by the wide block: each wide group holds rows
        # (i, i+3), so x demeans to -2 then +2 within each pair.
        np.testing.assert_allclose(
            record["input"].X[:, 2], [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
        )

    def test_ties_break_to_the_lowest_part_index(self):
        # Two width-2 blocks with different realized groupings: absorbing the
        # first (pairs (0,1) and (2,3)) demeans x to (-1, 1, -2, 2); absorbing
        # the second (pairs (0,2) and (1,3)) would give (-1.5, ...) instead.
        first = [0, 0, 1, 1]
        second = [0, 1, 0, 1]
        x = np.array([1.0, 3.0, 4.0, 8.0])
        X = np.column_stack((_one_hot(first, 2), _one_hot(second, 2), x))
        y = np.zeros(4)
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=4,
            coefficient_of_interest_index=4,
            design_domain=ProductDomain(
                parts=(
                    OneHotDomain(dim=2),
                    OneHotDomain(dim=2),
                    BoxDomain(bounds=np.array([[0.0, 8.0]])),
                )
            ),
            response_domain=BoxDomain(bounds=np.array([[0.0, 8.0]])),
        )

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        np.testing.assert_allclose(record["input"].X[:, 2], [-1.0, 1.0, -2.0, 2.0])
        assert record["metadata"].coefficient_of_interest_index == 2

    def test_block_containing_the_coi_is_excluded(self):
        # The width-3 block contains the COI, so the smaller width-2 block is
        # absorbed instead and the COI block is retained.
        wide = [0, 1, 2, 0, 1, 2]
        narrow = [0, 0, 0, 1, 1, 1]
        X = np.column_stack((_one_hot(wide, 3), _one_hot(narrow, 2)))
        y = np.arange(6.0)
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=6,
            coefficient_of_interest_index=1,  # inside the width-3 block
            design_domain=ProductDomain(
                parts=(OneHotDomain(dim=3), OneHotDomain(dim=2))
            ),
            response_domain=BoxDomain(bounds=np.array([[0.0, 8.0]])),
        )

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        # Retained: the width-3 COI block only; the COI index is unchanged.
        assert record["input"].X.shape == (6, 3)
        assert record["metadata"].coefficient_of_interest_index == 1


class TestPublicRefusals:
    """Structural limits raise from public metadata before input access."""

    def test_coi_on_the_intercept_is_refused(self):
        metadata = _sce(
            0, (InterceptDomain(), OneHotDomain(dim=2), EuclideanSpace(dim=1))
        )
        with pytest.raises(NotApplicable, match="intercept coordinate"):
            LargestOneHotFWLDemean().prepare(metadata, object(), NO_PRIVACY, 0)

    def test_non_product_design_is_refused(self):
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=3,
            coefficient_of_interest_index=0,
            design_domain=EuclideanSpace(dim=3),
            response_domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
        )
        with pytest.raises(NotApplicable, match="ProductDomain"):
            LargestOneHotFWLDemean().prepare(metadata, object(), NO_PRIVACY, 0)

    def test_no_one_hot_block_is_refused(self):
        metadata = _sce(
            1, (InterceptDomain(), BoxDomain(bounds=np.array([[0.0, 1.0]])))
        )
        with pytest.raises(NotApplicable, match="eligible"):
            LargestOneHotFWLDemean().prepare(metadata, object(), NO_PRIVACY, 0)

    def test_reference_coding_without_intercept_is_refused(self):
        # The only one-hot block is reference-coded and no intercept is
        # declared, so no eligible block remains.
        metadata = _sce(
            2,
            (
                OneHotDomain(dim=2, allow_zero=True),
                BoxDomain(bounds=np.array([[0.0, 1.0]])),
            ),
        )
        with pytest.raises(NotApplicable, match="eligible"):
            LargestOneHotFWLDemean().prepare(metadata, object(), NO_PRIVACY, 0)


class TestFWLEquivalence:
    """The target coefficient matches full explicit-one-hot OLS exactly."""

    @staticmethod
    def _full_instance(rng: np.random.Generator, second_block: bool) -> tuple[
        SingleCoefficientEstimationMetadata,
        SingleCoefficientEstimationInput,
        Any,
        Any,
        int,
    ]:
        n = 48
        absorbed = [int(v) for v in rng.integers(0, 3, size=n)]  # width-3 block
        covariates: Any = rng.normal(size=(n, 2))
        columns = [np.ones(n), _one_hot(absorbed, 3)]
        parts: list[Any] = [InterceptDomain(), OneHotDomain(dim=3)]
        if second_block:
            other = [int(v) for v in rng.integers(0, 3, size=n)]  # reference-coded
            columns.append(_one_hot(other, 2))
            parts.append(OneHotDomain(dim=2, allow_zero=True))
        columns.append(covariates)
        parts.append(BoxDomain(bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]])))
        X = np.column_stack(columns)
        beta: Any = rng.normal(size=X.shape[1])
        y = X @ beta + rng.normal(0.0, 0.1, size=n)
        coi = X.shape[1] - 2
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=n,
            coefficient_of_interest_index=coi,
            design_domain=ProductDomain(parts=tuple(parts)),
            response_domain=BoxDomain(bounds=np.array([[-40.0, 40.0]])),
        )
        return metadata, SingleCoefficientEstimationInput(X=X, y=y), X, y, coi

    def test_target_coefficient_matches_full_ols(self):
        rng = np.random.default_rng(7)
        metadata, task_input, X, y, coi = self._full_instance(rng, second_block=False)

        full: Array1D = np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)
        composed = CompositeAlgorithm(
            transformations=(LargestOneHotFWLDemean(), ToLinearRegression()),
            algorithm=OLS(),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        assert composed.coefficient == pytest.approx(full[coi], rel=1e-8)

    def test_absorbing_one_block_retains_the_others_exactly(self):
        # Two one-hot blocks: the width-3 block is absorbed, the
        # reference-coded width-2 block stays a covariate, and the target
        # coefficient still matches the full explicit-one-hot regression.
        rng = np.random.default_rng(11)
        metadata, task_input, X, y, coi = self._full_instance(rng, second_block=True)

        full: Array1D = np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)
        record: dict[str, Any] = {}
        composition = CompositeAlgorithm(
            transformations=(LargestOneHotFWLDemean(),),
            algorithm=_SpyChild(record=record),
        )
        composition.run(metadata, task_input, budget=NO_PRIVACY, seed=0)
        composed = CompositeAlgorithm(
            transformations=(LargestOneHotFWLDemean(), ToLinearRegression()),
            algorithm=OLS(),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        # The retained problem still carries the width-2 block's columns.
        assert composition.algorithm.record is not None
        assert composition.algorithm.record["input"].X.shape == (48, 4)
        assert composed.coefficient == pytest.approx(full[coi], rel=1e-8)


class TestPreparedDomains:
    """Blockwise centered-difference enclosures with the global guarantee."""

    def test_blockwise_enclosures_and_global_diameter(self):
        raw_box = BoxDomain(bounds=np.array([[-1.0, 2.0], [0.0, 4.0]]))
        raw_ball = LpBallDomain(p=2, radius=1.5, center=np.array([1.0, 1.0]))
        raw_one_hot = OneHotDomain(dim=2)
        metadata = SingleCoefficientEstimationMetadata(
            n_samples=3,
            coefficient_of_interest_index=4,
            design_domain=ProductDomain(
                parts=(
                    InterceptDomain(),
                    OneHotDomain(dim=3),
                    raw_box,
                    raw_ball,
                    raw_one_hot,
                )
            ),
            response_domain=BoxDomain(bounds=np.array([[-4.0, 6.0]])),
        )
        X = np.column_stack(
            (
                np.ones(3),
                _one_hot([0, 1, 2], 3),
                np.array([[-1.0, 0.0], [0.5, 2.0], [2.0, 4.0]]),
                np.array([[1.0, 1.0], [0.5, 1.5], [1.0, 0.0]]),
                _one_hot([0, 1, 0], 2),
            )
        )
        y = np.array([-4.0, 1.0, 6.0])

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        prepared = record["metadata"].design_domain
        assert isinstance(prepared, ProductDomain)
        assert len(prepared.parts) == 3

        centered_box, centered_ball, one_hot_enclosure = prepared.parts
        assert isinstance(centered_box, BoxDomain)
        np.testing.assert_allclose(centered_box.bounds, [[-3.0, 3.0], [-4.0, 4.0]])
        assert isinstance(centered_ball, LpBallDomain)
        assert centered_ball.p == 2
        assert centered_ball.radius == pytest.approx(3.0)
        np.testing.assert_array_equal(centered_ball.center, np.zeros(2))
        assert isinstance(one_hot_enclosure, LpBallDomain)
        assert not isinstance(one_hot_enclosure, OneHotDomain)
        assert one_hot_enclosure.radius == pytest.approx(np.sqrt(2.0))

        response = record["metadata"].response_domain
        assert isinstance(response, BoxDomain)
        np.testing.assert_allclose(response.bounds, [[-10.0, 10.0]])

        # The accepted global guarantee survives blockwise: the prepared
        # product's origin norm bound equals the retained raw L2 diameter.
        retained_raw = ProductDomain(parts=(raw_box, raw_ball, raw_one_hot))
        assert prepared.l2_norm_bound == pytest.approx(retained_raw.l2_diameter())

    def test_euclidean_parts_pass_through_unbounded(self):
        metadata = _sce(2, (OneHotDomain(dim=2), EuclideanSpace(dim=1)))
        X = np.column_stack((_one_hot([0, 1, 0], 2), np.array([1.0, 2.0, 3.0])))
        y = np.array([1.0, 2.0, 3.0])

        _, record = _run_spy(metadata, SingleCoefficientEstimationInput(X=X, y=y))

        prepared = record["metadata"].design_domain
        assert isinstance(prepared.parts[0], EuclideanSpace)
