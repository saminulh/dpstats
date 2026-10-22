"""DPWhitening's linear geometry, delegated privacy contracts and composition outcomes."""

import math
from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NO_PRIVACY,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.exceptions import InvalidRegime, NoRelease, NotApplicable
from dpstats.task import Array1D
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.covariance_estimation.transformations import (
    CovarianceToMeanEstimation,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import GaussianMechanism
from dpstats.tasks.linear_regression.algorithms import OLS
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ToLinearRegression,
    DPWhitening,
)
from dpstats.utils import split_seed
from tests.unit.tasks.task_builders import Unreadable, box, sce_metadata


Problem = tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]


@pytest.fixture
def problem() -> (
    tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]
):
    X = np.array([[-1, 1, 0], [-1, 3, 0], [1, 2, -2], [1, 2, 2]])
    return sce_metadata(4, 0, box((-1, 1), (1, 3), (-2, 2))), (
        SingleCoefficientEstimationInput(X=X, y=X @ np.array([0.7, -0.2, 0.3]))
    )


def test_uncentered_second_moment_uses_eigenvector_coordinates() -> None:
    X = np.array([[-0.5, 2, 0], [0.5, 2, 0], [-0.5, 0, 1], [0.5, 0, 1]])
    metadata = sce_metadata(4, 0, box((-1, 1), (0, 2), (0, 1)))
    source = SingleCoefficientEstimationInput(X=X, y=X[:, 0])
    child, data, *_ = DPWhitening().prepare(metadata, source, NO_PRIVACY, 0)

    # M2=diag(2, 1/2), so ascending eigenvectors swap the axes and each
    # column is divided by the corresponding second-moment square root.
    np.testing.assert_allclose(
        data.X,
        np.column_stack((X[:, 0], X[:, 2] * math.sqrt(2), X[:, 1] / math.sqrt(2))),
    )
    np.testing.assert_allclose(data.X[:, 1:].T @ data.X[:, 1:] / 4, np.eye(2))
    assert np.all(data.X[:, 1:].mean(axis=0) > 0)
    assert data.y is source.y
    assert isinstance(child.design_domain, ProductDomain)
    center, radius = child.design_domain.parts[1].l2_enclosing_ball()
    np.testing.assert_allclose(center, [1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert radius == pytest.approx(math.sqrt(5 / 2))
    child.validate_input(data)


@pytest.mark.parametrize("coi", [0, 2])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("budget", [NO_PRIVACY, ApproxDPBudget(eps=10, delta=1e-5)])
def test_original_nuisance_span_and_focal_ols_survive(
    coi: int, intercept: bool, budget: DPBudget
) -> None:
    rng = np.random.default_rng(23)
    X = rng.uniform(-1, 1, (32, 3))
    domain: Domain = box((-1, 1), (-1, 1), (-1, 1))
    if intercept:
        X = np.column_stack((np.ones(32), X))
        domain = ProductDomain(parts=(InterceptDomain(), domain))
    beta: Array1D = np.arange(1, X.shape[1] + 1, dtype=float) / 10
    metadata = sce_metadata(32, coi, domain)
    data = SingleCoefficientEstimationInput(X=X, y=X @ beta)
    child, prepared, _, _, restoration = DPWhitening().prepare(
        metadata, data, budget, 9
    )

    child.validate_input(prepared)
    child_index = int(intercept and coi != 0)
    np.testing.assert_array_equal(prepared.X[:, child_index], X[:, coi])
    assert prepared.y is data.y
    assert child.coefficient_of_interest_index == child_index
    assert child.has_intercept == intercept
    if intercept:
        np.testing.assert_array_equal(prepared.X[:, 0], np.ones(32))
    np.testing.assert_array_equal(
        child.design_domain.bounding_box().bounds[child_index],
        domain.bounding_box().bounds[coi],
    )
    # The known, full-rank linear model identifies this coefficient, including
    # an intercept target. All original columns are still in the child's span.
    assert np.linalg.lstsq(prepared.X, prepared.y, rcond=None)[0][
        child_index
    ] == pytest.approx(beta[coi])
    np.testing.assert_allclose(
        prepared.X @ np.linalg.lstsq(prepared.X, X, rcond=None)[0], X, atol=1e-12
    )
    output = SingleCoefficientEstimationOutput(coefficient=0.7)
    assert DPWhitening().restore(output, restoration) is output
    if budget is NO_PRIVACY:
        method = CompositeAlgorithm(
            transformations=(DPWhitening(), ToLinearRegression()), algorithm=OLS()
        )
        assert method.run(metadata, data, budget, 9).coefficient == pytest.approx(
            beta[coi]
        )


@pytest.mark.parametrize(
    "budget", [ApproxDPBudget(eps=10, delta=1e-5), GDPBudget(mu=2), ZCDPBudget(rho=2)]
)
@pytest.mark.parametrize("share", [0.25, 0.4])
def test_fixed_children_receive_their_typed_budgets_and_sibling_seeds(
    monkeypatch: pytest.MonkeyPatch, problem: Problem, budget: DPBudget, share: float
) -> None:
    calls: list[tuple[MeanEstimationMetadata, MeanEstimationInput, DPBudget, int]] = []
    original = GaussianMechanism.run

    def record(
        self: GaussianMechanism,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        calls.append((metadata, input, budget, seed))
        return original(self, metadata, input, budget, seed)

    monkeypatch.setattr(GaussianMechanism, "run", record)
    metadata, data = problem
    _, _, child_budget, child_seed, _ = DPWhitening(preparation_share=share).prepare(
        metadata, data, budget, 5
    )
    preparation, expected_child = budget.split((share, 1 - share))
    covariance, mean = preparation.split((4, 1))
    seeds = split_seed(5, 3)
    # Delegation oracle: the existing budget and seed owners define these splits.
    assert [call[0].dim for call in calls] == [2, 5]
    assert [call[2] for call in calls] == [mean, covariance]
    assert [call[3] for call in calls] == list(seeds[:2])
    assert child_budget == expected_child
    assert child_seed == seeds[2]
    np.testing.assert_array_equal(calls[0][1].X, data.X[:, 1:])
    assert (
        calls[0][0].domain.model_dump_json() == box((1, 3), (-2, 2)).model_dump_json()
    )
    # C=sqrt(13), so the joint row domain has radius sqrt(13+169).
    assert calls[1][0].domain.l2_norm_bound == pytest.approx(math.sqrt(182))


def test_separate_mean_does_not_cancel_the_covariance_adapters_mean(
    monkeypatch: pytest.MonkeyPatch, problem: Problem
) -> None:
    def released_mean(
        self: GaussianMechanism,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if metadata.dim == 2:
            return MeanEstimationOutput(mean_est=np.array([2.0, 1.0]))
        return MeanEstimationOutput(mean_est=np.array([1, -1, 5, 4, 2 * math.sqrt(2)]))

    monkeypatch.setattr(GaussianMechanism, "run", released_mean)
    metadata, data = problem
    child, transformed, *_ = DPWhitening().prepare(metadata, data, NO_PRIVACY, 7)
    # H=[[5,2],[2,4]], b=(1,-1), a=(2,1): S=H-bb.T+aa.T=[[8,5],[5,4]].
    # Therefore W W.T = S^-1 = [[4,-5],[-5,8]]/7, independently of
    # the eigensolver's sign choices. Using H alone gives a different Gram.
    moment_inverse = np.array([[4, -5], [-5, 8]]) / 7
    Z = data.X[:, 1:]
    np.testing.assert_allclose(
        transformed.X[:, 1:] @ transformed.X[:, 1:].T, Z @ moment_inverse @ Z.T
    )
    child.validate_input(transformed)


@pytest.mark.parametrize(
    "budget,gram_multiplier",
    [(NO_PRIVACY, 4e8), (ZCDPBudget(rho=20), 8 / math.sqrt(182))],
)
def test_negative_eigenvalues_use_the_accepted_floor(
    monkeypatch: pytest.MonkeyPatch,
    problem: Problem,
    budget: DPBudget,
    gram_multiplier: float,
) -> None:
    def negative_moments(
        self: GaussianMechanism,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if metadata.dim == 2:
            return MeanEstimationOutput(mean_est=np.zeros(2))
        return MeanEstimationOutput(mean_est=np.array([0.0, 0.0, -100.0, -100.0, 0.0]))

    monkeypatch.setattr(GaussianMechanism, "run", negative_moments)
    metadata, data = problem
    child, transformed, *_ = DPWhitening().prepare(metadata, data, budget, 7)
    # At no privacy, f=1e-8/4, hence 1/f=4e8. At rho=20, covariance
    # rho=4 and sensitivity=sqrt(182)/4; f=sqrt(182)/8, hence 8/sqrt(182).
    Z = data.X[:, 1:]
    np.testing.assert_allclose(
        transformed.X[:, 1:] @ transformed.X[:, 1:].T,
        (Z @ Z.T) * gram_multiplier,
    )
    child.validate_input(transformed)


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-6),
        ApproxDPBudget(eps=1, delta=1),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
def test_unconstrained_offers_retain_the_finite_domain_law(
    problem: Problem, budget: DPBudget
) -> None:
    metadata, data = problem
    child, transformed, remaining, *_ = DPWhitening().prepare(metadata, data, budget, 7)
    # Nuisance M2=diag(9/2, 2); both eigenvalues exceed the floor.
    np.testing.assert_allclose(
        transformed.X[:, 1:].T @ transformed.X[:, 1:] / 4, np.eye(2), atol=1e-14
    )
    assert remaining.is_unconstrained
    child.validate_input(transformed)


def test_missing_nuisance_is_a_public_skip_before_input_access() -> None:
    unused_input: Any = Unreadable("input")
    with pytest.raises(NotApplicable, match="nuisance"):
        DPWhitening().prepare(sce_metadata(4, 0, 1), unused_input, NO_PRIVACY, 0)


@pytest.mark.parametrize("budget", [NO_PRIVACY, ZCDPBudget(rho=2)])
def test_gaussian_mean_owns_unbounded_domain_refusal(
    monkeypatch: pytest.MonkeyPatch, problem: Problem, budget: DPBudget
) -> None:
    def forbidden_mean(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unbounded mean must refuse before averaging.")

    monkeypatch.setattr(np, "mean", forbidden_mean)
    _, data = problem
    with pytest.raises(InvalidRegime):
        DPWhitening().prepare(sce_metadata(4, 0, 3), data, budget, 7)


def test_finite_pure_budget_is_refused_before_private_averaging(
    monkeypatch: pytest.MonkeyPatch, problem: Problem
) -> None:
    def forbidden_mean(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsupported Gaussian accounting must precede averaging.")

    monkeypatch.setattr(np, "mean", forbidden_mean)
    metadata, data = problem
    with pytest.raises(InvalidRegime):
        DPWhitening().prepare(metadata, data, PureDPBudget(eps=1), 7)


@pytest.mark.parametrize("nuisance_value,permitted", [(0, True), (1, False)])
def test_finite_pure_budget_distinguishes_origin_and_nonzero_singletons(
    nuisance_value: int, permitted: bool
) -> None:
    metadata = sce_metadata(4, 0, box((-1, 1), (nuisance_value, nuisance_value)))
    data = SingleCoefficientEstimationInput(
        X=np.column_stack(([-1, -1, 1, 1], np.full(4, nuisance_value))), y=np.zeros(4)
    )
    if permitted:
        child, prepared, remaining, *_ = DPWhitening().prepare(
            metadata, data, PureDPBudget(eps=1), 7
        )
        np.testing.assert_array_equal(prepared.X, data.X)
        assert remaining == PureDPBudget(eps=0.75)
        child.validate_input(prepared)
    else:
        with pytest.raises(InvalidRegime):
            DPWhitening().prepare(metadata, data, PureDPBudget(eps=1), 7)


def test_covariance_allocation_refusal_keeps_its_existing_owner(
    monkeypatch: pytest.MonkeyPatch, problem: Problem
) -> None:
    refusal = InvalidRegime("Covariance allocation refused.")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise refusal

    monkeypatch.setattr(CovarianceToMeanEstimation, "prepare", refuse)
    metadata, data = problem
    with pytest.raises(InvalidRegime) as caught:
        DPWhitening().prepare(metadata, data, ZCDPBudget(rho=2), 7)
    assert caught.value is refusal


class RefusingCoefficient(BaseAlgorithm):
    """Reject the prepared public regime to exercise composition accounting."""

    TASK = SingleCoefficientEstimation

    def run(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> SingleCoefficientEstimationOutput:
        raise InvalidRegime("Child requires another public regime.")


@pytest.mark.parametrize(
    "budget,expected", [(ZCDPBudget(rho=2), NoRelease), (NO_PRIVACY, InvalidRegime)]
)
def test_child_refusal_uses_composition_spend_detection(
    problem: Problem, budget: DPBudget, expected: type[Exception]
) -> None:
    metadata, data = problem
    method = CompositeAlgorithm(
        transformations=(DPWhitening(),), algorithm=RefusingCoefficient()
    )
    with pytest.raises(expected, match="another public regime"):
        method.run(metadata, data, budget, 9)


@pytest.fixture
def centered_problem() -> Problem:
    X = np.array(
        [[1, -1, 1, 0], [1, 1, 1, 2], [1, 1, 5, 0], [1, -1, 5, 2]], dtype=float
    )
    metadata = SingleCoefficientEstimation.Metadata(
        n_samples=4,
        coefficient_of_interest_index=1,
        design_domain=ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.array([[-1, 1], [1, 5], [0, 2]])),
            )
        ),
        response_domain=EuclideanSpace(dim=1),
    )
    return metadata, SingleCoefficientEstimation.Input(
        X=X, y=X @ np.array([1.1, 0.7, -0.2, 0.3])
    )


def test_centered_covariance_keeps_literal_intercept_and_focal(
    centered_problem: Problem,
) -> None:
    metadata, data = centered_problem
    child, prepared, *_ = DPWhitening().prepare(metadata, data, NO_PRIVACY, 0)
    # Mean=(3,1), covariance=diag(4,1): the ascending eigenbasis swaps axes.
    expected = np.column_stack(
        (
            data.X[:, :2],
            (data.X[:, 3] - 1),
            (data.X[:, 2] - 3) / 2,
        )
    )
    np.testing.assert_allclose(prepared.X, expected)
    np.testing.assert_allclose(prepared.X[:, 2:].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(
        prepared.X[:, 2:].T @ prepared.X[:, 2:] / 4, np.eye(2), atol=1e-14
    )
    assert child.has_intercept
    assert child.coefficient_of_interest_index == 1
    assert prepared.y is data.y
    assert isinstance(child.design_domain, ProductDomain)
    center, radius = child.design_domain.parts[-1].l2_enclosing_ball()
    np.testing.assert_allclose(center, 0)
    assert radius == pytest.approx(math.sqrt(5))
    child.validate_input(prepared)
    method = CompositeAlgorithm(
        transformations=(DPWhitening(), ToLinearRegression()), algorithm=OLS()
    )
    assert method.run(metadata, data, NO_PRIVACY, 0).coefficient == pytest.approx(0.7)


def test_private_center_and_covariance_are_distinct_releases(
    monkeypatch: pytest.MonkeyPatch, centered_problem: Problem
) -> None:
    def release(
        self: GaussianMechanism,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if metadata.dim == 2:
            return MeanEstimationOutput(mean_est=np.array([2.0, 1.0]))
        return MeanEstimationOutput(mean_est=np.array([1, -1, 5, 4, 2 * math.sqrt(2)]))

    monkeypatch.setattr(GaussianMechanism, "run", release)
    metadata, data = centered_problem
    child, prepared, *_ = DPWhitening().prepare(metadata, data, NO_PRIVACY, 7)
    # Covariance=H-bb.T=[[4,3],[3,3]], so WW.T=[[3,-3],[-3,4]]/3.
    # Rows use the separate a=(2,1), not b=(1,-1), and never add aa.T to S.
    shifted = data.X[:, 2:] - [2, 1]
    expected_gram = shifted @ (np.array([[3, -3], [-3, 4]]) / 3) @ shifted.T
    np.testing.assert_allclose(prepared.X[:, 2:] @ prepared.X[:, 2:].T, expected_gram)
    assert np.linalg.norm(prepared.X[:, 2:].mean(axis=0)) > 0
    child.validate_input(prepared)


@pytest.mark.parametrize(
    "budget", [ApproxDPBudget(eps=10, delta=1e-5), GDPBudget(mu=2), ZCDPBudget(rho=2)]
)
def test_gaussian_calls_exclude_retained_columns_and_keep_budget_seed_owners(
    monkeypatch: pytest.MonkeyPatch, centered_problem: Problem, budget: DPBudget
) -> None:
    calls: list[tuple[MeanEstimationMetadata, MeanEstimationInput, DPBudget, int]] = []
    original = GaussianMechanism.run

    def record(
        self: GaussianMechanism,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        calls.append((metadata, input, budget, seed))
        return original(self, metadata, input, budget, seed)

    monkeypatch.setattr(GaussianMechanism, "run", record)
    metadata, data = centered_problem
    _, _, remaining, child_seed, _ = DPWhitening().prepare(metadata, data, budget, 5)
    preparation, expected_child = budget.split((0.25, 0.75))
    covariance, mean = preparation.split((4, 1))
    seeds = split_seed(5, 3)
    assert [call[0].dim for call in calls] == [2, 5]
    assert [call[2] for call in calls] == [mean, covariance]
    assert [call[3] for call in calls] == list(seeds[:2])
    np.testing.assert_array_equal(calls[0][1].X, data.X[:, 2:])
    assert remaining == expected_child
    assert child_seed == seeds[2]


@pytest.mark.parametrize("coi", [0, 1])
def test_no_whitenable_nuisances_skips_before_input_access(coi: int) -> None:
    metadata = SingleCoefficientEstimation.Metadata(
        n_samples=4,
        coefficient_of_interest_index=coi,
        design_domain=(
            ProductDomain(
                parts=(InterceptDomain(), BoxDomain(bounds=np.array([[-1, 1]])))
            )
            if coi
            else ProductDomain(parts=(InterceptDomain(),))
        ),
        response_domain=EuclideanSpace(dim=1),
    )
    unreadable: Any = object()
    with pytest.raises(NotApplicable, match="nuisance"):
        DPWhitening().prepare(metadata, unreadable, NO_PRIVACY, 0)


def test_finite_pure_budget_refuses_before_averaging(
    monkeypatch: pytest.MonkeyPatch, centered_problem: Problem
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Gaussian accounting must precede private averaging.")

    monkeypatch.setattr(np, "mean", forbidden)
    metadata, data = centered_problem
    with pytest.raises(InvalidRegime):
        DPWhitening().prepare(metadata, data, PureDPBudget(eps=1), 0)
