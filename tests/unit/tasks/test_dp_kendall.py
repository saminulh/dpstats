"""COI-preserving Kendall selection through public SCE transformation compositions."""

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import InvalidRegime, NotApplicable
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.budgets import NO_PRIVACY
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    DPKendall,
)


class _RecordingChild(BaseAlgorithm):
    """Observe the task call received by an SCE child and release a marker."""

    TASK = SingleCoefficientEstimation
    record: dict[str, Any]

    @property
    def privacy_interpretation(self):
        return PrivacyInterpretation.STANDARD

    def run(self, metadata, input, budget, seed):
        self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        return SingleCoefficientEstimationOutput(coefficient=7.5)


def _observe(transformations, metadata, input, budget=NO_PRIVACY, seed=4):
    algorithm = CompositeAlgorithm(
        transformations=transformations, algorithm=_RecordingChild(record={})
    )
    output = algorithm.run(metadata, input, budget, seed)
    return algorithm.algorithm.record, output


@pytest.fixture
def problem():
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=4,
        coefficient_of_interest_index=2,
        design_domain=ProductDomain(
            parts=(InterceptDomain(), BoxDomain(bounds=np.array([[-2.0, 2.0]] * 3)))
        ),
        response_domain=EuclideanSpace(dim=1),
    )
    input = SingleCoefficientEstimationInput(
        X=np.array(
            [
                [1.0, -1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0],
            ]
        ),
        y=np.array([-1.0, -5.0, 5.0, 1.0]),
    )
    return metadata, input


def test_retains_coi_and_intercept_while_selecting_one_control(problem):
    metadata, input = problem
    record, output = _observe((DPKendall(num_features=1),), metadata, input)
    prepared, reduced = record["metadata"], record["input"]
    np.testing.assert_array_equal(reduced.X, input.X[:, [0, 1, 2]])
    np.testing.assert_array_equal(reduced.y, input.y)
    assert prepared.coefficient_of_interest_index == 2
    assert prepared.has_intercept
    assert prepared.dim == 3
    np.testing.assert_array_equal(
        prepared.design_domain.parts[1].bounds, [[-2.0, 2.0], [-2.0, 2.0]]
    )
    assert record["budget"] == NO_PRIVACY
    assert 0 <= record["seed"] < 2**64
    assert output.coefficient == 7.5


@pytest.mark.parametrize("coi", [0, 1, 2, 3])
@pytest.mark.parametrize("has_intercept", [False, True])
def test_coi_is_retained_at_every_position_even_when_it_is_the_intercept(
    problem, coi, has_intercept
):
    from dpstats.budgets import ApproxDPBudget

    original_metadata, input = problem
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=4,
        coefficient_of_interest_index=coi,
        design_domain=(
            original_metadata.design_domain if has_intercept else EuclideanSpace(dim=4)
        ),
        response_domain=original_metadata.response_domain,
    )
    for seed in range(5):
        record, _ = _observe(
            (DPKendall(num_features=1),),
            metadata,
            input,
            ApproxDPBudget(eps=1.0, delta=1e-6),
            seed,
        )
        prepared, reduced = record["metadata"], record["input"]
        np.testing.assert_array_equal(
            reduced.X[:, prepared.coefficient_of_interest_index], input.X[:, coi]
        )
        assert prepared.dim == (3 if has_intercept and coi != 0 else 2)
        assert prepared.has_intercept == has_intercept
        if has_intercept:
            np.testing.assert_array_equal(reduced.X[:, 0], np.ones(4))
        prepared.validate_input(reduced)


def test_selection_cannot_depend_on_withheld_coi_values(problem):
    from dpstats.budgets import ApproxDPBudget

    metadata, input = problem
    changed_X = input.X.copy()
    changed_X[:, 2] = [1.9, 1.7, -1.7, -1.9]
    changed = SingleCoefficientEstimationInput(X=changed_X, y=input.y)
    transformation = DPKendall(num_features=1)
    budget = ApproxDPBudget(eps=1.0, delta=1e-6)
    for seed in range(5):
        first, _ = _observe((transformation,), metadata, input, budget, seed)
        second, _ = _observe((transformation,), metadata, changed, budget, seed)
        original, selected = first["metadata"], first["input"]
        modified, other = second["metadata"], second["input"]
        assert (
            original.coefficient_of_interest_index
            == modified.coefficient_of_interest_index
        )
        np.testing.assert_array_equal(
            np.delete(selected.X, original.coefficient_of_interest_index, axis=1),
            np.delete(other.X, modified.coefficient_of_interest_index, axis=1),
        )


def test_composition_returns_the_retained_coi_coefficient_and_round_trips(problem):
    from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
    from dpstats.tasks.linear_regression.algorithms import OLS
    from dpstats.tasks.single_coefficient_estimation.transformations import (
        ToLinearRegression,
    )

    metadata, input = problem
    algorithm = CompositeAlgorithm(
        transformations=(DPKendall(num_features=1), ToLinearRegression()),
        algorithm=OLS(),
    )
    loaded = BaseAlgorithm.load_record(algorithm.dump_config().model_dump())
    assert loaded.uuid == algorithm.uuid
    assert loaded.run(metadata, input, NO_PRIVACY, 4).coefficient == pytest.approx(2.0)


def test_too_many_controls_refuses_instead_of_counting_the_coi(problem):
    metadata, input = problem
    with pytest.raises(NotApplicable, match="exceeds the 2 selectable"):
        DPKendall(num_features=3).prepare(metadata, input, NO_PRIVACY, 4)


def test_single_row_refuses_publicly(problem):
    metadata, input = problem
    metadata = metadata.model_copy(update={"n_samples": 1})
    input = SingleCoefficientEstimationInput(X=input.X[:1], y=input.y[:1])
    with pytest.raises(InvalidRegime, match="at least two samples") as refused:
        DPKendall(num_features=1).prepare(metadata, input, NO_PRIVACY, 4)
    assert refused.type is InvalidRegime


def test_single_row_refuses_the_whole_composition(problem):
    # A NotApplicable stage would be passed through, sending every column on.
    from dpstats.algorithm import CompositeAlgorithm
    from dpstats.tasks.linear_regression.algorithms import OLS
    from dpstats.tasks.single_coefficient_estimation.transformations import (
        ToLinearRegression,
    )

    metadata, input = problem
    metadata = metadata.model_copy(update={"n_samples": 1})
    input = SingleCoefficientEstimationInput(X=input.X[:1], y=input.y[:1])
    algorithm = CompositeAlgorithm(
        transformations=(DPKendall(num_features=1), ToLinearRegression()),
        algorithm=OLS(),
    )
    with pytest.raises(InvalidRegime, match="at least two samples") as refused:
        algorithm.run(metadata, input, NO_PRIVACY, 4)
    assert refused.type is InvalidRegime


def test_selecting_all_controls_still_spends_selection_budget(problem):
    from dpstats.budgets import (
        ApproxDPBudget,
        PureDPBudget,
        ZCDPBudget,
        GDPBudget,
    )

    metadata, input = problem
    cases = [
        (PureDPBudget(eps=2.0), PureDPBudget(eps=1.6)),
        (ApproxDPBudget(eps=2.0, delta=1e-6), ApproxDPBudget(eps=1.6, delta=0.8e-6)),
        (ZCDPBudget(rho=2.0), ZCDPBudget(rho=1.6)),
        (GDPBudget(mu=2.0), GDPBudget(mu=np.sqrt(3.2))),
    ]
    for offered, expected in cases:
        record, _ = _observe(
            (DPKendall(num_features=2, selection_share=0.2),),
            metadata,
            input,
            offered,
            0,
        )
        remaining = record["budget"]
        assert type(remaining) is type(expected)
        for field, value in expected.model_dump().items():
            assert remaining.model_dump()[field] == pytest.approx(value)


def test_fwl_remapped_coi_is_retained_without_recreating_the_intercept():
    from dpstats.algorithm import CompositeAlgorithm, PrivacyInterpretation
    from dpstats.domains import OneHotDomain
    from dpstats.tasks.linear_regression.algorithms import OLS
    from dpstats.tasks.single_coefficient_estimation.transformations import (
        LargestOneHotFWLDemean,
        ToLinearRegression,
    )

    metadata = SingleCoefficientEstimationMetadata(
        n_samples=6,
        coefficient_of_interest_index=5,
        design_domain=ProductDomain(
            parts=(InterceptDomain(), OneHotDomain(dim=2), EuclideanSpace(dim=3))
        ),
        response_domain=EuclideanSpace(dim=1),
    )
    input = SingleCoefficientEstimationInput(
        X=np.array(
            [
                [1, 1, 0, -1, 1, -1],
                [1, 1, 0, 0, -2, 1],
                [1, 1, 0, 1, 1, 0],
                [1, 0, 1, -1, 1, 0],
                [1, 0, 1, 0, -2, -1],
                [1, 0, 1, 1, 1, 1],
            ],
            dtype=float,
        ),
        y=np.array([-5, 2, 3, -3, -2, 5], dtype=float),
    )
    algorithm = CompositeAlgorithm(
        transformations=(
            LargestOneHotFWLDemean(),
            DPKendall(num_features=2),
            ToLinearRegression(),
        ),
        algorithm=OLS(),
    )
    record, _ = _observe(
        (LargestOneHotFWLDemean(), DPKendall(num_features=2)), metadata, input
    )
    prepared, reduced = record["metadata"], record["input"]
    assert prepared.dim == 3
    assert not prepared.has_intercept
    np.testing.assert_array_equal(reduced.X[:, 2], [-1, 1, 0, 0, -1, 1])
    assert algorithm.run(metadata, input, NO_PRIVACY, 4).coefficient == pytest.approx(
        2.0
    )
    from dpstats.tasks.linear_regression.algorithms import BoostedAdaSSP

    private_child = CompositeAlgorithm(
        transformations=algorithm.transformations,
        algorithm=BoostedAdaSSP(
            num_rounds=100, tau=1.0, learning_rate_schedule="constant"
        ),
    )
    assert private_child.privacy_interpretation is PrivacyInterpretation.DIAGNOSTIC


@pytest.mark.parametrize("family", ["pure", "approximate", "zcdp", "gdp"])
def test_unconstrained_families_preserve_the_donor_selection_limit(problem, family):
    from dpstats.budgets import (
        PureDPBudget,
        ApproxDPBudget,
        ZCDPBudget,
        GDPBudget,
    )

    offered = {
        "pure": PureDPBudget(eps=float("inf")),
        "approximate": ApproxDPBudget(eps=float("inf"), delta=1e-6),
        "zcdp": ZCDPBudget(rho=float("inf")),
        "gdp": GDPBudget(mu=float("inf")),
    }[family]
    metadata, input = problem
    record, _ = _observe((DPKendall(num_features=1),), metadata, input, offered)
    prepared, reduced = record["metadata"], record["input"]
    np.testing.assert_array_equal(reduced.X, input.X[:, [0, 1, 2]])
    assert prepared.coefficient_of_interest_index == 2


@pytest.mark.parametrize("domain", [EuclideanSpace(dim=1), InterceptDomain()])
def test_a_design_containing_only_the_coi_has_nothing_to_select(domain):
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=3,
        coefficient_of_interest_index=0,
        design_domain=domain,
        response_domain=EuclideanSpace(dim=1),
    )
    input = SingleCoefficientEstimationInput(X=np.ones((3, 1)), y=np.arange(3.0))
    with pytest.raises(NotApplicable, match="no selectable controls"):
        DPKendall(num_features=1).prepare(metadata, input, NO_PRIVACY, 4)
