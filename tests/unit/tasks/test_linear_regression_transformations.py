"""Unit tests for the linear-regression Kendall selection transformations."""

from typing import Any

import numpy as np
import pytest

from dpstats.simulations.linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import InvalidRegime, NotApplicable

from dpstats.budgets import ApproxDPBudget
from dpstats.tasks.linear_regression.transformations import (
    DPKendall,
    ExactKendallSelection,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionOutput,
)
from tests.unit.tasks.task_builders import lr_metadata

SENTINEL_COEFFICIENT = 7.0


class _SpyAlgorithm(BaseAlgorithm):
    """Record what the composition algorithm position was offered and return a marker."""

    TASK = LinearRegression

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    record: dict[str, Any] | None = None

    def run(self, metadata, input, budget, seed):
        if self.record is not None:
            self.record["budget"] = budget
            self.record["dim"] = metadata.dim
        return LinearRegressionOutput(beta=np.full(metadata.dim, SENTINEL_COEFFICIENT))


@pytest.fixture
def instance():
    """Generate one moderate linear-regression instance."""
    metadata, task_input, _ = simulate_clipped_gaussian(seed=0, dim=6, n_samples=300)
    return metadata, task_input


class TestSelectionReduction:
    """Both selectors reduce the design and restore the original width."""

    @pytest.mark.parametrize(
        "transformation",
        [DPKendall(num_features=2), ExactKendallSelection(num_features=2)],
    )
    def test_output_keeps_the_original_width_with_a_reduced_support(
        self, instance, transformation
    ):
        metadata, task_input = instance
        record: dict[str, Any] = {}
        composition = CompositeAlgorithm(
            transformations=(transformation,), algorithm=_SpyAlgorithm(record=record)
        )

        output = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )

        assert composition.algorithm.record is not None
        assert composition.algorithm.record["dim"] == 2
        assert output.beta.shape == (metadata.dim,)
        assert int(np.count_nonzero(output.beta)) == 2
        assert set(np.flatnonzero(output.beta)) <= set(range(metadata.dim))

    @pytest.mark.parametrize(
        "transformation",
        [DPKendall(num_features=2), ExactKendallSelection(num_features=2)],
    )
    def test_restoration_places_child_values_at_selected_indices(
        self, instance, transformation
    ):
        metadata, task_input = instance
        composition = CompositeAlgorithm(
            transformations=(transformation,), algorithm=_SpyAlgorithm()
        )

        output = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=3
        )

        selected = np.flatnonzero(output.beta)
        assert np.all(output.beta[selected] == SENTINEL_COEFFICIENT)
        unselected = np.setdiff1d(np.arange(metadata.dim), selected)
        assert np.all(output.beta[unselected] == 0.0)

    @pytest.mark.parametrize("transformation_class", [DPKendall, ExactKendallSelection])
    def test_requesting_more_features_than_exist_is_rejected(
        self, instance, transformation_class
    ):
        metadata, task_input = instance

        with pytest.raises(NotApplicable, match="exceeds"):
            transformation_class(num_features=metadata.dim + 1).prepare(
                metadata, task_input, ApproxDPBudget(eps=1.0, delta=1e-5), 0
            )

    def test_a_stage_that_cannot_select_passes_the_problem_through(self, instance):
        """A composition omits the stage instead of refusing the cell."""
        metadata, task_input = instance
        composition = CompositeAlgorithm(
            transformations=(DPKendall(num_features=metadata.dim + 1),),
            algorithm=_SpyAlgorithm(record={}),
        )
        offered = ApproxDPBudget(eps=1.0, delta=1e-5)

        composition.run(metadata, task_input, budget=offered, seed=0)

        assert composition.algorithm.record == {
            "budget": offered,
            "dim": metadata.dim,
        }

    @pytest.mark.parametrize("transformation_class", [DPKendall, ExactKendallSelection])
    def test_a_single_row_is_refused_as_a_regime_limit(
        self, instance, transformation_class
    ):
        metadata, task_input = instance
        single_row = lr_metadata(
            1, design=metadata.design_domain, response=metadata.response_domain
        )
        single_row_input = LinearRegressionInput(X=task_input.X[:1], y=task_input.y[:1])

        with pytest.raises(InvalidRegime, match="samples") as refused:
            transformation_class(num_features=2).prepare(
                single_row,
                single_row_input,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                0,
            )
        assert refused.type is InvalidRegime


class TestSelectionCost:
    """The private selector pays for its release and the oracle does not."""

    def test_the_private_selector_hands_on_a_strictly_smaller_budget(self, instance):
        metadata, task_input = instance
        record: dict[str, Any] = {}
        composition = CompositeAlgorithm(
            transformations=(DPKendall(num_features=2, selection_share=0.2),),
            algorithm=_SpyAlgorithm(record=record),
        )
        offered = ApproxDPBudget(eps=1.0, delta=1e-5)

        composition.run(metadata, task_input, budget=offered, seed=0)

        assert composition.algorithm.record is not None
        assert composition.algorithm.record["budget"].eps < offered.eps
        assert composition.algorithm.record["budget"].delta < offered.delta

    def test_the_oracle_hands_on_the_whole_budget(self, instance):
        metadata, task_input = instance
        record: dict[str, Any] = {}
        composition = CompositeAlgorithm(
            transformations=(ExactKendallSelection(num_features=2),),
            algorithm=_SpyAlgorithm(record=record),
        )
        offered = ApproxDPBudget(eps=1.0, delta=1e-5)

        composition.run(metadata, task_input, budget=offered, seed=0)

        assert composition.algorithm.record is not None
        assert composition.algorithm.record["budget"] == offered

    def test_the_oracle_selection_does_not_depend_on_the_seed(self, instance):
        metadata, task_input = instance
        composition = CompositeAlgorithm(
            transformations=(ExactKendallSelection(num_features=3),),
            algorithm=_SpyAlgorithm(),
        )

        first = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )
        second = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=99
        )

        assert np.array_equal(np.flatnonzero(first.beta), np.flatnonzero(second.beta))
