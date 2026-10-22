"""Tests for Google/Wilson ApproxBounds clipping transformations."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from dpstats.algorithm import Transformation
from dpstats.exceptions import InvalidRegime, NoRelease
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
    EuclideanSpace,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Array1D
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations import (
    ApproxBounds as LRApproxBounds,
    ApproxBoundsAll as LRApproxBoundsAll,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.mean_estimation.transformations import (
    ApproxBounds as MeanApproxBounds,
    Standardize as MeanStandardize,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ApproxBounds as SCEApproxBounds,
    ApproxBoundsAll as SCEApproxBoundsAll,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy


class _NegativeLaplaceNoise:
    def laplace(self, *, scale: float, size: int) -> Array1D:
        del scale
        return np.full(size, -100.0)


class _DisjointLaplaceNoise:
    def __init__(self) -> None:
        self._is_positive_count = True

    def laplace(self, *, scale: float, size: int) -> Array1D:
        del scale
        noise = np.full(size, -100.0)
        if self._is_positive_count:
            noise[3] = 100.0
            self._is_positive_count = False
        return noise


class _RecordedLaplaceNoise:
    def __init__(self, noise: Array1D) -> None:
        self._noise = noise
        self._offset = 0
        self.scales: list[float] = []
        self.sizes: list[int] = []

    def laplace(self, *, scale: float, size: int) -> Array1D:
        end = self._offset + size
        if end > self._noise.size:
            raise AssertionError("ApproxBounds redrew its noisy count vector.")
        self.scales.append(scale)
        self.sizes.append(size)
        result = self._noise[self._offset : end]
        self._offset = end
        return result


@pytest.mark.parametrize(
    "transformation",
    [
        MeanApproxBounds,
        LRApproxBounds,
        SCEApproxBounds,
        LRApproxBoundsAll,
        SCEApproxBoundsAll,
    ],
)
def test_transformations_reject_log_grid_with_nonfinite_final_edge(
    transformation: Callable[..., Transformation],
) -> None:
    with pytest.raises(ValueError, match="nonfinite grid edge"):
        transformation(scale=1.0, num_bins=1025)


def test_mean_transformation_uses_configured_log_bin_endpoints() -> None:
    metadata = MeanEstimationMetadata(n_samples=5, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[-9.0], [-3.0], [0.0], [2.0], [17.0]]))

    prepared_metadata, _, _, _, _ = MeanApproxBounds(scale=1.0, num_bins=6).prepare(
        metadata, task_input, NO_PRIVACY, seed=0
    )

    assert isinstance(prepared_metadata.domain, BoxDomain)
    np.testing.assert_array_equal(prepared_metadata.domain.bounds, [[-16.0, 32.0]])


def test_mean_transformation_uses_a_finite_highest_log_bin_edge() -> None:
    highest_finite_bin_edge = float.fromhex("0x1p1023")
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[highest_finite_bin_edge]]))

    prepared_metadata, prepared_input, _, _, _ = MeanApproxBounds(
        scale=0.5, num_bins=1025
    ).prepare(metadata, task_input, NO_PRIVACY, seed=0)

    assert isinstance(prepared_metadata.domain, BoxDomain)
    assert np.all(np.isfinite(prepared_metadata.domain.bounds))
    assert np.all(prepared_metadata.domain.contains(prepared_input.X))


def _no_bin_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    def negative_rng(_seed: int) -> _NegativeLaplaceNoise:
        return _NegativeLaplaceNoise()

    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        negative_rng,
    )


def test_mean_transformation_keeps_declared_bounds_when_no_bin_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_bin_survives(monkeypatch)

    metadata = MeanEstimationMetadata(
        n_samples=2,
        domain=BoxDomain(bounds=np.array([[-math.inf, math.inf], [0.0, math.inf]])),
    )
    task_input = MeanEstimationInput(X=np.array([[0.0, 1.0], [3.0, 9.0]]))
    prepared_metadata, prepared_input, _, _, _ = MeanApproxBounds(
        scale=1.0, num_bins=3
    ).prepare(metadata, task_input, PureDPBudget(eps=10.0), seed=0)

    assert isinstance(prepared_metadata.domain, BoxDomain)
    np.testing.assert_array_equal(
        prepared_metadata.domain.bounds,
        [[-math.inf, math.inf], [0.0, math.inf]],
    )
    np.testing.assert_array_equal(prepared_input.X, task_input.X)


def test_standardize_refuses_a_coordinate_approx_bounds_left_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_bin_survives(monkeypatch)

    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))
    prepared_metadata, prepared_input, child_budget, child_seed, _ = MeanApproxBounds(
        scale=1.0, num_bins=3
    ).prepare(metadata, task_input, PureDPBudget(eps=10.0), seed=0)

    with pytest.raises(InvalidRegime, match="finite public domain bounds"):
        MeanStandardize().prepare(
            prepared_metadata, prepared_input, child_budget, child_seed
        )


def test_public_and_private_sides_intersect() -> None:
    domain = BoxDomain(
        bounds=np.array(
            [
                [-10.0, math.inf],
                [-math.inf, 10.0],
                [0.0, math.inf],
                [-math.inf, 0.0],
                [0.0, 1.0],
            ]
        )
    )
    metadata = MeanEstimationMetadata(n_samples=3, domain=domain)
    task_input = MeanEstimationInput(
        X=np.array(
            [
                [-8.0, -9.0, 0.0, -8.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.5],
                [7.0, 8.0, 2.0, 0.0, 1.0],
            ]
        )
    )

    prepared_metadata, prepared_input, _, _, _ = MeanApproxBounds().prepare(
        metadata,
        task_input,
        NO_PRIVACY,
        seed=0,
    )

    assert isinstance(prepared_metadata.domain, BoxDomain)
    np.testing.assert_array_equal(
        prepared_metadata.domain.bounds,
        [
            [-8.0, 8.0],
            [-16.0, 8.0],
            [0.0, 2.0],
            [-8.0, 0.0],
            [0.0, 1.0],
        ],
    )
    np.testing.assert_array_equal(
        prepared_input.X,
        [
            [-8.0, -9.0, 0.0, -8.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.5],
            [7.0, 8.0, 2.0, 0.0, 1.0],
        ],
    )
    assert np.all(prepared_metadata.domain.contains(prepared_input.X))


def test_mean_transformation_preserves_floating_input_dtype_through_box_projection() -> (
    None
):
    metadata = MeanEstimationMetadata(
        n_samples=2,
        domain=BoxDomain(bounds=np.array([[-math.inf, math.inf]], dtype=np.float32)),
    )
    task_input = MeanEstimationInput(X=np.array([[-2.0], [2.0]], dtype=np.float32))

    _, prepared_input, _, _, _ = MeanApproxBounds().prepare(
        metadata, task_input, NO_PRIVACY, seed=0
    )

    assert prepared_input.X.dtype == np.dtype(np.float32)


def test_disjoint_private_and_public_intervals_return_no_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disjoint_rng(_seed: int) -> _DisjointLaplaceNoise:
        return _DisjointLaplaceNoise()

    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        disjoint_rng,
    )
    metadata = MeanEstimationMetadata(
        n_samples=2,
        domain=BoxDomain(bounds=np.array([[10.0, math.inf]])),
    )
    task_input = MeanEstimationInput(X=np.array([[10.0], [11.0]]))

    with pytest.raises(NoRelease, match="declared public domain"):
        MeanApproxBounds().prepare(
            metadata,
            task_input,
            PureDPBudget(eps=10.0),
            seed=0,
        )


def test_public_coordinates_do_not_dilute_the_bounds_budget(
    accounting_spy: AccountingSpy,
) -> None:
    domain = ProductDomain(
        parts=(
            InterceptDomain(),
            OneHotDomain(dim=2, allow_zero=True),
            BoxDomain(bounds=np.array([[0.0, 1.0], [-math.inf, math.inf]])),
        )
    )
    metadata = MeanEstimationMetadata(n_samples=2, domain=domain)
    task_input = MeanEstimationInput(
        X=np.array(
            [
                [1.0, 1.0, 0.0, 0.0, -2.0],
                [1.0, 0.0, 1.0, 1.0, 2.0],
            ]
        )
    )

    prepared_metadata, prepared_input, child_budget, _, _ = MeanApproxBounds(
        bounds_share=0.5,
        scale=0.25,
        num_bins=8,
    ).prepare(metadata, task_input, PureDPBudget(eps=8.0), seed=4)

    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", PureDPBudget(eps=4.0)),
        AccountingCall("spend_as_zcdp", PureDPBudget(eps=4.0)),
    )
    assert isinstance(child_budget, PureDPBudget)
    assert child_budget.eps == pytest.approx(4.0)
    assert isinstance(prepared_metadata.domain, ProductDomain)
    assert prepared_metadata.domain.parts[:2] == domain.parts[:2]
    assert np.all(prepared_metadata.domain.contains(prepared_input.X))


def test_private_coordinates_split_the_bounds_budget_equally(
    accounting_spy: AccountingSpy,
) -> None:
    metadata = MeanEstimationMetadata(n_samples=2, domain=EuclideanSpace(dim=2))
    task_input = MeanEstimationInput(X=np.array([[-1.0, 1.0], [1.0, -1.0]]))

    _, _, child_budget, _, _ = MeanApproxBounds(bounds_share=0.5).prepare(
        metadata, task_input, PureDPBudget(eps=8.0), seed=3
    )

    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", PureDPBudget(eps=2.0)),
        AccountingCall("spend_as_zcdp", PureDPBudget(eps=4.0)),
    )
    assert isinstance(child_budget, PureDPBudget)
    assert child_budget.eps == pytest.approx(4.0)


def test_mean_transformation_reuses_one_noisy_vector_for_late_relaxation(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
) -> None:
    noise = _RecordedLaplaceNoise(np.array([3.0, -100.0, -100.0, -100.0]))

    def recorded_rng(_seed: int) -> _RecordedLaplaceNoise:
        return noise

    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        recorded_rng,
    )
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))

    prepared_metadata, _, child_budget, _, _ = MeanApproxBounds(
        bounds_share=0.5, scale=1.0, num_bins=2
    ).prepare(metadata, task_input, PureDPBudget(eps=2.0), seed=7)

    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", PureDPBudget(eps=1.0)),
        AccountingCall("spend_as_zcdp", PureDPBudget(eps=1.0)),
    )
    assert noise.sizes == [2, 2]
    assert noise.scales == pytest.approx([1.0, 1.0])
    assert isinstance(prepared_metadata.domain, BoxDomain)
    np.testing.assert_array_equal(prepared_metadata.domain.bounds, [[0.0, 1.0]])
    assert isinstance(child_budget, PureDPBudget)
    assert child_budget.eps == pytest.approx(1.0)


def test_fully_public_mean_domain_spends_nothing() -> None:
    domain = BoxDomain(bounds=np.array([[0.0, 1.0], [-2.0, 2.0]]))
    metadata = MeanEstimationMetadata(n_samples=2, domain=domain)
    task_input = MeanEstimationInput(X=np.array([[0.0, -1.0], [1.0, 2.0]]))
    offered = ApproxDPBudget(eps=3.0, delta=1e-5)

    prepared_metadata, prepared_input, child_budget, child_seed, restoration = (
        MeanApproxBounds().prepare(metadata, task_input, offered, seed=17)
    )

    assert prepared_metadata.domain is domain
    np.testing.assert_array_equal(prepared_input.X, task_input.X)
    assert child_budget == offered
    assert child_seed == 17
    assert restoration is None


def test_unbounded_mean_domain_spends_default_share() -> None:
    n = 20
    metadata = MeanEstimationMetadata(
        n_samples=n,
        domain=EuclideanSpace(dim=1),
    )
    task_input = MeanEstimationInput(X=np.ones((n, 1)))
    offered = ApproxDPBudget(eps=20.0, delta=1e-5)

    prepared_metadata, prepared_input, child_budget, child_seed, _ = (
        MeanApproxBounds().prepare(metadata, task_input, offered, seed=21)
    )

    assert isinstance(prepared_metadata.domain, BoxDomain)
    assert isinstance(child_budget, ApproxDPBudget)
    assert child_budget.eps == pytest.approx(18.0)
    assert child_budget.delta == pytest.approx(9e-6)
    assert child_seed != 21
    assert np.all(prepared_metadata.domain.contains(prepared_input.X))


def test_finite_approx_dp_direct_bounds_plan_wins_for_one_coordinate(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
) -> None:
    noise = _RecordedLaplaceNoise(np.zeros(2))
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: noise,
    )
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))
    offered = ApproxDPBudget(eps=2.0, delta=2e-6)

    _, _, child_budget, child_seed, _ = MeanApproxBounds(
        bounds_share=0.5, num_bins=1
    ).prepare(metadata, task_input, offered, seed=59)

    accounting_calls = accounting_spy.calls
    assert [call.entry_point for call in accounting_calls] == [
        "spend_as_pure",
        "spend_as_zcdp",
        "spend_as_pure",
    ]
    assert accounting_calls[0].budget == ApproxDPBudget(eps=1.0, delta=1e-6)
    assert accounting_calls[1].budget == ApproxDPBudget(eps=1.0, delta=1e-6)
    assert isinstance(accounting_calls[2].budget, ZCDPBudget)
    # An independent accountant oracle gives rho=0.0243559703595 for
    # (eps, delta)=(1, 1e-6). The pure zCDP leaf solves
    # eps * tanh(eps / 2) = rho / m, yielding 0.221156781763 at m=1.
    assert accounting_calls[2].budget.rho == pytest.approx(0.0243559703595)
    assert noise.scales == pytest.approx([1.0, 1.0])
    assert child_budget == ApproxDPBudget(eps=1.0, delta=1e-6)
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: _NegativeLaplaceNoise(),
    )
    _, _, _, ordinary_child_seed, _ = MeanApproxBounds(
        bounds_share=0.5, num_bins=1
    ).prepare(metadata, task_input, NO_PRIVACY, seed=59)
    assert child_seed == ordinary_child_seed


def test_finite_approx_dp_zcdp_bounds_plan_wins_for_many_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
) -> None:
    num_coordinates = 100
    noise = _RecordedLaplaceNoise(np.zeros(2 * num_coordinates))
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: noise,
    )
    metadata = MeanEstimationMetadata(
        n_samples=1,
        domain=EuclideanSpace(dim=num_coordinates),
    )
    task_input = MeanEstimationInput(X=np.zeros((1, num_coordinates)))
    offered = ApproxDPBudget(eps=2.0, delta=2e-6)

    _, _, child_budget, child_seed, _ = MeanApproxBounds(
        bounds_share=0.5, num_bins=1
    ).prepare(metadata, task_input, offered, seed=61)

    accounting_calls = accounting_spy.calls
    assert [call.entry_point for call in accounting_calls] == [
        "spend_as_pure",
        "spend_as_zcdp",
        "spend_as_pure",
    ]
    assert accounting_calls[0].budget == ApproxDPBudget(eps=0.01, delta=1e-8)
    zcdp_leaf = accounting_calls[2].budget
    assert isinstance(zcdp_leaf, ZCDPBudget)
    assert zcdp_leaf.rho == pytest.approx(0.0243559703595 / num_coordinates)
    # The same relation yields 0.0220712297258 after 100 zCDP leaves, while
    # the direct typed split is 0.01 per leaf.
    assert noise.scales == pytest.approx(
        [1.0 / 0.0220712297258] * (2 * num_coordinates)
    )
    assert child_budget == ApproxDPBudget(eps=1.0, delta=1e-6)
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: _NegativeLaplaceNoise(),
    )
    _, _, _, ordinary_child_seed, _ = MeanApproxBounds(
        bounds_share=0.5, num_bins=1
    ).prepare(metadata, task_input, NO_PRIVACY, seed=61)
    assert child_seed == ordinary_child_seed


def test_native_zcdp_bounds_plans_tie_at_one_common_epsilon(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
) -> None:
    noise = _RecordedLaplaceNoise(np.zeros(2))
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: noise,
    )
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))

    _, _, child_budget, _, _ = MeanApproxBounds(bounds_share=0.5, num_bins=1).prepare(
        metadata, task_input, ZCDPBudget(rho=2.0), seed=71
    )

    bounds_budget = ZCDPBudget(rho=1.0)
    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", bounds_budget),
        AccountingCall("spend_as_zcdp", bounds_budget),
        AccountingCall("spend_as_pure", bounds_budget),
    )
    # The independently solved equality eps * tanh(eps / 2) = 1 gives the
    # common direct and zCDP-plan epsilon 1.543404638418209.
    assert noise.scales == pytest.approx([1.0 / 1.543404638418209] * 2)
    assert child_budget == ZCDPBudget(rho=1.0)


def test_finite_gdp_bounds_fall_back_to_the_direct_plan(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
) -> None:
    noise = _RecordedLaplaceNoise(np.zeros(2))
    monkeypatch.setattr(
        "dpstats.domains.approx_bounds.np.random.default_rng",
        lambda _seed: noise,
    )
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))

    _, _, child_budget, _, _ = MeanApproxBounds(bounds_share=0.5, num_bins=1).prepare(
        metadata, task_input, GDPBudget(mu=2.0), seed=73
    )

    bounds_budget = GDPBudget(mu=math.sqrt(2.0))
    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", bounds_budget),
        AccountingCall("spend_as_zcdp", bounds_budget),
    )
    # The direct GDP leaf has epsilon 1.154050277612644, independently from
    # log(Phi(mu / 2)) - log(Phi(-mu / 2)) at mu = sqrt(2).
    assert noise.scales == pytest.approx([1.0 / 1.154050277612644] * 2)
    assert child_budget == GDPBudget(mu=math.sqrt(2.0))


@pytest.mark.parametrize(
    "offered",
    [
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-6),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
        NO_PRIVACY,
    ],
)
def test_unconstrained_typed_offers_match_direct_and_zcdp_plans(
    accounting_spy: AccountingSpy,
    offered: DPBudget,
) -> None:
    metadata = MeanEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    task_input = MeanEstimationInput(X=np.array([[0.0]]))

    _, _, child_budget, _, _ = MeanApproxBounds(bounds_share=0.5, num_bins=1).prepare(
        metadata, task_input, offered, seed=79
    )

    assert [call.entry_point for call in accounting_spy.calls] == [
        "spend_as_pure",
        "spend_as_zcdp",
        "spend_as_pure",
    ]
    assert all(call.budget.is_unconstrained for call in accounting_spy.calls)
    assert child_budget.is_unconstrained
    assert type(child_budget) is type(offered)


def test_linear_regression_preserves_structural_and_bounded_blocks() -> None:
    intercept = InterceptDomain()
    category = OneHotDomain(dim=2, allow_zero=True)
    numeric = BoxDomain(
        bounds=np.array(
            [
                [-10.0, math.inf],
                [-math.inf, 10.0],
                [0.0, 1.0],
            ]
        )
    )
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=ProductDomain(parts=(intercept, category, numeric)),
        response_domain=EuclideanSpace(dim=1),
    )
    task_input = LinearRegressionInput(
        X=np.array(
            [
                [1.0, 1.0, 0.0, -8.0, -9.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [1.0, 0.0, 1.0, 7.0, 8.0, 1.0],
            ]
        ),
        y=np.array([-3.0, 0.0, 5.0]),
    )

    prepared_metadata, prepared_input, child_budget, child_seed, _ = (
        LRApproxBounds().prepare(metadata, task_input, NO_PRIVACY, seed=9)
    )

    assert isinstance(prepared_metadata.design_domain, ProductDomain)
    prepared_intercept, prepared_category, prepared_numeric = (
        prepared_metadata.design_domain.parts
    )
    assert prepared_intercept is intercept
    assert prepared_category is category
    assert isinstance(prepared_numeric, BoxDomain)
    np.testing.assert_array_equal(prepared_numeric.bounds[0], [-8.0, 8.0])
    np.testing.assert_array_equal(prepared_numeric.bounds[1], [-16.0, 8.0])
    np.testing.assert_array_equal(prepared_numeric.bounds[2], [0.0, 1.0])
    assert isinstance(prepared_metadata.response_domain, BoxDomain)
    np.testing.assert_array_equal(
        prepared_metadata.response_domain.bounds,
        [[-4.0, 8.0]],
    )
    assert np.all(prepared_metadata.design_domain.contains(prepared_input.X))
    assert np.all(prepared_metadata.response_domain.contains(prepared_input.y[:, None]))
    assert child_budget == NO_PRIVACY
    assert child_seed != 9


def test_sce_adapter_retains_coefficient_of_interest() -> None:
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=3,
        coefficient_of_interest_index=1,
        design_domain=ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=1))),
        response_domain=EuclideanSpace(dim=1),
    )
    task_input = SingleCoefficientEstimationInput(
        X=np.array([[1.0, -2.0], [1.0, 0.0], [1.0, 3.0]]),
        y=np.array([-1.0, 0.0, 4.0]),
    )

    prepared_metadata, prepared_input, _, _, _ = SCEApproxBounds().prepare(
        metadata, task_input, NO_PRIVACY, seed=5
    )

    assert prepared_metadata.coefficient_of_interest_index == 1
    assert np.all(prepared_metadata.design_domain.contains(prepared_input.X))
    assert np.all(prepared_metadata.response_domain.contains(prepared_input.y[:, None]))


def test_estimate_all_matches_approx_bounds_when_nothing_is_public() -> None:
    """On a fully unbounded instance, estimating every coordinate changes nothing.

    Every coordinate already lacks a finite public side, so "estimate every
    coordinate" and "estimate only the coordinates missing a bound" select the
    same targets in the same order and draw the same seeds and budgets.
    """
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=EuclideanSpace(dim=2),
        response_domain=EuclideanSpace(dim=1),
    )
    task_input = LinearRegressionInput(
        X=np.array([[-8.0, -9.0], [0.0, 0.0], [7.0, 8.0], [2.0, -3.0]]),
        y=np.array([-3.0, 0.0, 5.0, 1.0]),
    )
    offered = ApproxDPBudget(eps=6.0, delta=1e-6)

    all_metadata, all_input, all_budget, all_seed, _ = LRApproxBoundsAll().prepare(
        metadata, task_input, offered, seed=13
    )
    (
        missing_only_metadata,
        missing_only_input,
        missing_only_budget,
        missing_only_seed,
        _,
    ) = LRApproxBounds().prepare(metadata, task_input, offered, seed=13)

    assert isinstance(all_metadata.design_domain, BoxDomain)
    assert isinstance(missing_only_metadata.design_domain, BoxDomain)
    np.testing.assert_array_equal(
        all_metadata.design_domain.bounds, missing_only_metadata.design_domain.bounds
    )
    assert isinstance(all_metadata.response_domain, BoxDomain)
    assert isinstance(missing_only_metadata.response_domain, BoxDomain)
    np.testing.assert_array_equal(
        all_metadata.response_domain.bounds,
        missing_only_metadata.response_domain.bounds,
    )
    np.testing.assert_array_equal(all_input.X, missing_only_input.X)
    np.testing.assert_array_equal(all_input.y, missing_only_input.y)
    assert all_budget == missing_only_budget
    assert all_seed == missing_only_seed


def test_estimate_all_spends_and_stays_inside_the_public_box_when_fully_bounded() -> (
    None
):
    """ApproxBoundsAll pays privacy even where ApproxBounds would spend nothing.

    The public sides only ever narrow the released interval (the completion
    step takes the ``max`` of the two lower sides and the ``min`` of the two
    upper sides), so the released box sits inside the declared public box by
    construction whenever the call returns instead of raising ``NoRelease``.
    """
    intercept = InterceptDomain()
    numeric = BoxDomain(bounds=np.array([[-10.0, 10.0], [-1.0, 1.0]]))
    response = BoxDomain(bounds=np.array([[-5.0, 5.0]]))
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=ProductDomain(parts=(intercept, numeric)),
        response_domain=response,
    )
    task_input = LinearRegressionInput(
        X=np.array(
            [
                [1.0, -8.0, -0.5],
                [1.0, 0.0, 0.0],
                [1.0, 7.0, 0.5],
            ]
        ),
        y=np.array([-3.0, 0.0, 4.0]),
    )
    offered = PureDPBudget(eps=10.0)

    prepared_metadata, prepared_input, child_budget, child_seed, _ = (
        LRApproxBoundsAll().prepare(metadata, task_input, offered, seed=6)
    )

    assert isinstance(child_budget, PureDPBudget)
    assert child_budget.eps < offered.eps
    assert child_seed != 6
    assert isinstance(prepared_metadata.design_domain, ProductDomain)
    prepared_intercept, prepared_numeric = prepared_metadata.design_domain.parts
    assert prepared_intercept is intercept
    assert isinstance(prepared_numeric, BoxDomain)
    assert np.all(prepared_numeric.bounds[:, 0] >= numeric.bounds[:, 0])
    assert np.all(prepared_numeric.bounds[:, 1] <= numeric.bounds[:, 1])
    assert isinstance(prepared_metadata.response_domain, BoxDomain)
    assert np.all(
        prepared_metadata.response_domain.bounds[:, 0] >= response.bounds[:, 0]
    )
    assert np.all(
        prepared_metadata.response_domain.bounds[:, 1] <= response.bounds[:, 1]
    )
    assert np.all(prepared_metadata.design_domain.contains(prepared_input.X))
    assert np.all(prepared_metadata.response_domain.contains(prepared_input.y[:, None]))


def test_sce_all_adapter_spends_and_retains_coefficient_of_interest() -> None:
    """The SCE adapter forwards ``estimate_all`` and keeps the target index.

    A fully public design and response distinguish this from the plain SCE
    ``ApproxBounds`` adapter: only forwarding ``estimate_all=True`` makes this
    call spend a strictly smaller child budget than offered.
    """
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=3,
        coefficient_of_interest_index=1,
        design_domain=ProductDomain(
            parts=(InterceptDomain(), BoxDomain(bounds=np.array([[-5.0, 5.0]])))
        ),
        response_domain=BoxDomain(bounds=np.array([[-5.0, 5.0]])),
    )
    task_input = SingleCoefficientEstimationInput(
        X=np.array([[1.0, -2.0], [1.0, 0.0], [1.0, 3.0]]),
        y=np.array([-1.0, 0.0, 4.0]),
    )
    offered = ApproxDPBudget(eps=10.0, delta=1e-6)

    prepared_metadata, prepared_input, child_budget, _, _ = (
        SCEApproxBoundsAll().prepare(metadata, task_input, offered, seed=5)
    )

    assert prepared_metadata.coefficient_of_interest_index == 1
    assert isinstance(child_budget, ApproxDPBudget)
    assert child_budget.eps < offered.eps
    assert np.all(prepared_metadata.design_domain.contains(prepared_input.X))
    assert np.all(prepared_metadata.response_domain.contains(prepared_input.y[:, None]))
