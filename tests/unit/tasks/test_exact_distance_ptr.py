"""Tests for the exact-distance PTR median mechanism."""

import math
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from dpstats.domains import EuclideanSpace
from dpstats.budgets import ApproxDPBudget, PureDPBudget, ZCDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.exceptions import NoRelease
from dpstats.utils import split_seed
from dpstats.tasks.median_estimation.algorithms import (
    exact_distance_ptr as exact_distance_ptr_module,
)
from dpstats.tasks.median_estimation.algorithms.exact_distance_ptr import (
    ExactDistancePTR,
    _canonical_candidate,
    _distance_to_unsafety,
    _log_score_integral,
    _sample_truncated_laplace,
    _sample_truncated_mechanism,
    _truncated_laplace_bound,
    _unsafe_left,
)
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness


def _metadata(n_samples: int = 8) -> MedianEstimationMetadata:
    return MedianEstimationMetadata(
        n_samples=n_samples,
        domain=EuclideanSpace(dim=1),
    )


@pytest.mark.parametrize("seed", [0, 6])
def test_safe_input_passes_the_test_and_releases_near_the_median(seed: int) -> None:
    # Forty evenly spaced points are far from unsafety. The estimate spends
    # eps=5 on interval scores eps*min(k, n-k), so each unit step away from
    # the central interval [19, 20] costs a factor e**5 in weight, and two
    # steps cost e**10; the support stops at the truncation window.
    output = ExactDistancePTR().run(
        _metadata(40),
        MedianEstimationInput(X=np.arange(40.0)),
        budget=ApproxDPBudget(eps=10.0, delta=1e-3),
        seed=seed,
    )

    assert 18.0 <= output.median_est <= 21.0


@pytest.mark.parametrize("eps,delta", [(0.5, 1e-5), (1.0, 1e-6), (4.0, 0.01)])
def test_truncated_laplace_stays_inside_positive_threshold(
    eps: float, delta: float
) -> None:
    phase_budget = ApproxDPBudget(eps=eps / 2, delta=delta / 2)
    bound = _truncated_laplace_bound(phase_budget)
    seeds = split_seed(7, 10_000)
    draws = np.array(
        [_sample_truncated_laplace(seed, phase_budget, bound) for seed in seeds]
    )
    assert np.all(draws >= -bound)
    assert np.all(draws <= bound)
    assert np.all(bound + draws >= 0)
    assert np.all(bound + draws <= 2 * bound)


def test_piecewise_integral_matches_direct_formula() -> None:
    x = np.arange(1.0, 9.0)
    eps = 0.7
    # t=2 support intervals have scores 2,3,4,3,2.
    expected = sum(math.exp(eps * score) for score in (2, 3, 4, 3, 2))
    actual = math.exp(_log_score_integral(x, 2, eps, increment_left=False))
    assert actual == pytest.approx(expected)


def test_canonical_candidate_applies_four_edit_types() -> None:
    x = np.arange(10.0)
    candidate = _canonical_candidate(x, t=3, condition=1, k1=1, k2=1, k3=1, k4=1)
    assert candidate is not None
    assert candidate[0] == -np.inf
    assert candidate[-1] == np.inf
    assert candidate.size == x.size
    assert 1.0 not in candidate
    assert 8.0 not in candidate


def test_distance_zero_exactly_when_input_is_unsafe() -> None:
    x = np.array([0.0, 1.0, 2.0, 1000.0, 1001.0, 1002.0])
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    assert _unsafe_left(x, budget, 2, 3)
    distance = _distance_to_unsafety(x, budget, 2, max_distance=3)
    assert distance == 0


def test_sampling_uses_paper_interval_score(
    fake_randomness: FakeRandomness,
) -> None:
    # Selecting the first interval and returning its midpoint makes the
    # released value a fixed function of the interval grid, so the assertion
    # is about the selection weights rather than the draw.
    fake_randomness.script("choice", lambda _parameters: 0)
    fake_randomness.script(
        "uniform", lambda parameters: (parameters["low"] + parameters["high"]) / 2
    )
    result = _sample_truncated_mechanism(
        np.arange(1.0, 9.0), ApproxDPBudget(eps=1.0, delta=1e-5), 2, 0
    )
    assert result == pytest.approx(2.5)
    (probabilities,) = fake_randomness.parameters_of("choice", "p")
    expected = np.exp(np.array([2, 3, 4, 3, 2], dtype=float))
    expected /= expected.sum()
    assert probabilities == pytest.approx(expected)


def test_native_approx_budget_composes_without_lattice_conversion(
    monkeypatch: pytest.MonkeyPatch, accounting_spy: AccountingSpy
) -> None:
    outer_budget = ApproxDPBudget(eps=2.0, delta=2e-5)
    distance_budgets: list[ApproxDPBudget] = []
    observed_budgets: list[ApproxDPBudget] = []

    def record_distance(
        x_sorted: np.ndarray,
        budget: ApproxDPBudget,
        t: int,
        *,
        max_distance: int,
    ) -> int:
        del x_sorted, t
        distance_budgets.append(budget)
        return max_distance + 1

    observed_seeds: list[int] = []

    def record_threshold_noise(
        seed: int, budget: ApproxDPBudget, bound: float
    ) -> float:
        del bound
        observed_budgets.append(budget)
        observed_seeds.append(seed)
        return 0.0

    def record_estimate(
        x_sorted: np.ndarray, budget: ApproxDPBudget, t: int, seed: int
    ) -> float:
        del x_sorted, t
        observed_budgets.append(budget)
        observed_seeds.append(seed)
        return 3.0

    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_distance_to_unsafety",
        record_distance,
    )
    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_sample_truncated_laplace",
        record_threshold_noise,
    )
    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_sample_truncated_mechanism",
        record_estimate,
    )

    output = ExactDistancePTR().run(
        _metadata(),
        MedianEstimationInput(X=np.arange(8.0)),
        budget=outer_budget,
        seed=7,
    )

    assert output.median_est == 3.0
    assert distance_budgets == [outer_budget]
    assert len(observed_budgets) == 2
    for budget in observed_budgets:
        assert_budget_close(budget, ApproxDPBudget(eps=1.0, delta=1e-5))
    # The threshold and the estimate are separately accounted releases, so
    # each draws from its own child stream rather than the offered seed.
    assert len(set(observed_seeds)) == 2
    assert 7 not in observed_seeds
    assert accounting_spy.calls == ()


def test_safety_distance_uses_full_offered_budget_and_fails_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_budget = ApproxDPBudget(eps=2.0, delta=2e-5)
    observed: list[ApproxDPBudget] = []

    def budget_sensitive_distance(
        x_sorted: np.ndarray,
        budget: ApproxDPBudget,
        t: int,
        *,
        max_distance: int,
    ) -> int:
        del x_sorted, t
        observed.append(budget)
        return 0 if budget == outer_budget else max_distance + 1

    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_distance_to_unsafety",
        budget_sensitive_distance,
    )
    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_sample_truncated_laplace",
        lambda _seed, _budget, _bound: 0.0,
    )
    monkeypatch.setattr(
        exact_distance_ptr_module,
        "_sample_truncated_mechanism",
        lambda *_args, **_kwargs: pytest.fail("unsafe release must not run"),
    )

    with pytest.raises(NoRelease, match="too close to unsafety"):
        ExactDistancePTR().run(
            _metadata(),
            MedianEstimationInput(X=np.arange(8.0)),
            budget=outer_budget,
            seed=7,
        )

    assert observed == [outer_budget]


def test_public_sample_count_refuses_before_reading_the_input() -> None:
    # n_samples < 1 is already unconstructable metadata, so the algorithm's own
    # precondition is exercised at the smallest constructable count.
    unreadable = MagicMock(spec=MedianEstimationInput)
    type(unreadable).X = PropertyMock(side_effect=AssertionError("read X"))

    with pytest.raises(InvalidRegime, match="n_samples"):
        ExactDistancePTR().run(
            _metadata(1),
            unreadable,
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )


def test_empty_support_is_the_accounted_bottom() -> None:
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    with pytest.raises(NoRelease, match="empty support"):
        _sample_truncated_mechanism(np.array([0.0, 1.0]), budget, t=2, seed=0)


@pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
def test_refuses_non_approx_families_without_accounting_calls(
    accounting_spy: AccountingSpy, budget: PureDPBudget | ZCDPBudget
) -> None:
    with pytest.raises(InvalidRegime, match="Cannot spend"):
        ExactDistancePTR().run(
            _metadata(),
            MedianEstimationInput(X=np.arange(8.0)),
            budget=budget,
            seed=0,
        )
    assert accounting_spy.calls == ()


@pytest.mark.parametrize("delta", [1e-12, 1e-9, 1e-6, 1e-3, 1e-1])
@pytest.mark.parametrize("eps", [1e-5, 1e-3, 1e-1, 10.0, 100.0])
def test_truncation_bound_stays_finite_across_the_privacy_parameter_envelope(
    eps: float, delta: float
) -> None:
    # The truncated-Laplace ratio overflows only near eps = 709, far outside
    # the supported envelope eps in [1e-5, 100], delta in [1e-12, 0.1].
    bound = _truncated_laplace_bound(ApproxDPBudget(eps=eps, delta=delta))
    assert math.isfinite(bound)
    assert bound >= 1.0


@pytest.mark.parametrize("seed", [0, 6])
def test_tied_data_releases_from_positive_width_intervals(seed: int) -> None:
    # Every value appears twice, so half of the order-statistic gaps have zero
    # width and carry no mass under the interval score. The release still lands
    # within two unit steps of the central interval [9, 10], as for untied data.
    output = ExactDistancePTR().run(
        _metadata(40),
        MedianEstimationInput(X=np.repeat(np.arange(20.0), 2)),
        budget=ApproxDPBudget(eps=10.0, delta=1e-3),
        seed=seed,
    )

    assert 8.0 <= output.median_est <= 11.0
