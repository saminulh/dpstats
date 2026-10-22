"""Tests for the PLAN binary-search Gaussian mean estimator."""

from __future__ import annotations

# pyright: reportPrivateUsage=false

from typing import cast

import numpy as np
import pytest

from dpstats.domains import BoxDomain
from dpstats.budgets import (
    GDPBudget,
    NO_PRIVACY,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D
from dpstats.tasks.mean_estimation.algorithms.plan import (
    PLANBinarySearchGaussian,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from tests.unit.tasks.task_builders import Unreadable, box, mean_metadata


def test_plan_calibrates_each_twenty_round_private_quantile(
    accounting_spy: AccountingSpy,
):
    rho = 1e8
    metadata = mean_metadata(2, box((-4.0, 4.0)))
    input_data = MeanEstimationInput(X=np.array([[1.0], [3.0]]))

    PLANBinarySearchGaussian().run(metadata, input_data, ZCDPBudget(rho=rho), seed=7)

    calls = accounting_spy.calls
    assert calls[0] == AccountingCall("spend_as_zcdp", ZCDPBudget(rho=rho))
    assert len(calls) == 5
    expected_count_rhos = [3 * rho / 320, rho / 320, 3 * rho / 320]
    for call, expected_rho in zip(calls[1:4], expected_count_rhos):
        assert call.entry_point == "calibrate_gaussian_scale"
        assert call.sensitivity == 1.0
        assert isinstance(call.budget, ZCDPBudget)
        assert call.budget.rho == pytest.approx(expected_rho)


def test_plan_wires_source_defaults_pairing_and_budget_shares(
    accounting_spy: AccountingSpy,
    monkeypatch: pytest.MonkeyPatch,
):
    import dpstats.tasks.mean_estimation.algorithms.plan as plan

    n = 2
    d = 3
    rho = 1e8
    X = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    metadata = mean_metadata(n, box(*[(-10.0, 10.0)] * d))
    input_data = MeanEstimationInput(X=X)

    correction = (7 / 9) ** 3
    private_quantile_results = iter(
        [4.0 * correction, 9.0 * correction, 16.0 * correction, 0.0, 0.0, 0.0, 2.0]
    )
    calls: list[tuple[Array1D, float, float, float, ZCDPBudget, int, int]] = []

    def record_quantile(
        values: Array1D,
        quantile: float,
        lower: float,
        upper: float,
        budget: ZCDPBudget,
        seed: int,
        *,
        num_rounds: int,
        population: int,
    ) -> float:
        assert num_rounds == 20
        calls.append((values.copy(), quantile, lower, upper, budget, seed, population))
        return next(private_quantile_results)

    monkeypatch.setattr(plan, "_binary_private_quantile", record_quantile)
    output = PLANBinarySearchGaussian().run(
        metadata, input_data, ZCDPBudget(rho=rho), seed=11
    )

    assert output.mean_est.shape == (d,)
    assert len(calls) == 7
    expected_pair_statistics = [2.0, 8.0, 18.0]
    for call, expected in zip(calls[:3], expected_pair_statistics):
        values, quantile, lower, upper, budget, _, population = call
        assert population == n // 2
        np.testing.assert_allclose(values, [expected])
        assert quantile == 0.5
        assert lower == 0.0
        assert upper == 200.0
        assert budget.rho == pytest.approx(rho / 16)

    for values, quantile, lower, upper, budget, _, population in calls[3:6]:
        assert population == n
        assert values.shape == (n,)
        assert quantile == 0.5
        assert lower == -10.0
        assert upper == 10.0
        assert budget.rho == pytest.approx(rho / 48)

    scale = np.array([5.0, 6.0, 7.0])
    expected_norm_upper = np.sqrt(np.log(n) * np.log(10.0) * np.sum(scale))
    expected_norms = np.linalg.norm(X / np.sqrt(scale), axis=1)
    threshold_values, threshold_quantile, lower, upper, budget, _, population = calls[
        -1
    ]
    assert population == n
    np.testing.assert_allclose(
        threshold_values, np.minimum(expected_norms, expected_norm_upper)
    )
    expected_rank_error = np.sqrt(n) + np.sqrt(
        20 * np.log(600.0) / (2 * (3 * rho / 16))
    )
    assert threshold_quantile == pytest.approx(1 - expected_rank_error / n)
    assert lower == 0.0
    assert upper == pytest.approx(expected_norm_upper)
    assert budget.rho == pytest.approx(3 * rho / 16)

    assert accounting_spy.calls[0] == AccountingCall(
        "spend_as_zcdp", ZCDPBudget(rho=rho)
    )
    final_call = accounting_spy.calls[-1]
    assert final_call.entry_point == "calibrate_gaussian_scale"
    assert isinstance(final_call.budget, ZCDPBudget)
    assert final_call.budget.rho == pytest.approx(9 * rho / 16)
    assert final_call.sensitivity == pytest.approx(4.0)


def test_plan_wires_configured_grouping_quantiles_and_budget_shares(
    accounting_spy: AccountingSpy,
    monkeypatch: pytest.MonkeyPatch,
):
    import dpstats.tasks.mean_estimation.algorithms.plan as plan

    n = 100
    rho = 1e8
    group_size = 4
    beta = 0.2
    correction = (1 - 2 / (9 * group_size)) ** 3
    quantile_results = iter([9.0 * correction, 0.0, 2.0])
    calls: list[tuple[Array1D, float, float, float, ZCDPBudget, int]] = []

    def record_quantile(
        values: Array1D,
        quantile: float,
        lower: float,
        upper: float,
        budget: ZCDPBudget,
        seed: int,
        *,
        num_rounds: int,
        population: int,
    ) -> float:
        assert num_rounds == 10
        calls.append((values.copy(), quantile, lower, upper, budget, population))
        return next(quantile_results)

    monkeypatch.setattr(plan, "_binary_private_quantile", record_quantile)
    algorithm = PLANBinarySearchGaussian(
        variance_group_size=group_size,
        num_rounds=10,
        failure_probability=beta,
        preprocessing_budget_fraction=0.2,
        centering_budget_fraction=0.5,
        threshold_budget_fraction=0.4,
    )
    metadata = mean_metadata(n, box((-10.0, 10.0)))
    output = algorithm.run(
        metadata,
        MeanEstimationInput(X=np.linspace(-1.0, 1.0, n).reshape(n, 1)),
        ZCDPBudget(rho=rho),
        seed=17,
    )

    assert output.mean_est.shape == (1,)
    assert len(calls) == 3
    variance_values, _, _, _, variance_budget, variance_population = calls[0]
    assert variance_values.shape == (n // (2 * group_size),)
    assert variance_population == n // (2 * group_size)
    assert variance_budget.rho == pytest.approx(0.1 * rho)

    _, _, _, _, center_budget, center_population = calls[1]
    assert center_population == n
    assert center_budget.rho == pytest.approx(0.1 * rho)

    _, threshold_quantile, _, threshold_upper, threshold_budget, population = calls[2]
    assert population == n
    assert threshold_budget.rho == pytest.approx(0.32 * rho)
    expected_tail_count = np.sqrt(n) + np.sqrt(
        10 * np.log(30 / beta) / (2 * 0.32 * rho)
    )
    assert threshold_quantile == pytest.approx(1 - expected_tail_count / n)
    assert threshold_upper == pytest.approx(np.sqrt(np.log(n) * np.log(1 / beta) * 6))

    final_call = accounting_spy.calls[-1]
    assert final_call.entry_point == "calibrate_gaussian_scale"
    assert isinstance(final_call.budget, ZCDPBudget)
    assert final_call.budget.rho == pytest.approx(0.48 * rho)
    assert final_call.sensitivity == pytest.approx(4.0)


def test_plan_singleton_domain_returns_public_point_before_accounting(
    accounting_spy: AccountingSpy,
):
    point = np.array([2.0, -1.0])
    metadata = mean_metadata(2, BoxDomain(bounds=np.column_stack((point, point))))
    input_data = MeanEstimationInput(X=np.tile(point, (2, 1)))

    output = PLANBinarySearchGaussian().run(
        metadata, input_data, PureDPBudget(eps=1.0), seed=0
    )

    np.testing.assert_array_equal(output.mean_est, point)
    assert accounting_spy.calls == ()


def test_plan_no_privacy_runs_the_same_private_stage_structure(
    accounting_spy: AccountingSpy,
):
    n = 100
    d = 2
    X = np.zeros((n, d))
    X[-1] = 6.0
    metadata = mean_metadata(n, box(*[(-6.0, 6.0)] * d))
    input_data = MeanEstimationInput(X=X)

    output = PLANBinarySearchGaussian().run(metadata, input_data, NO_PRIVACY, seed=31)

    assert np.all(np.isfinite(output.mean_est))
    assert not np.allclose(output.mean_est, np.mean(X, axis=0))
    assert accounting_spy.calls[0] == AccountingCall("spend_as_zcdp", NO_PRIVACY)
    calibration_calls = accounting_spy.calls[1:]
    assert len(calibration_calls) == 2 * d + 2
    for call in calibration_calls:
        assert call.entry_point == "calibrate_gaussian_scale"
        assert isinstance(call.budget, ZCDPBudget)
        assert call.budget.rho == np.inf
    assert all(call.sensitivity == 1.0 for call in calibration_calls[:-1])


def test_plan_unbounded_domain_refusal_precedes_input_access():
    inaccessible_input = cast(MeanEstimationInput, Unreadable("Input"))
    metadata = mean_metadata(100, 2)
    with pytest.raises(InvalidRegime, match="finite"):
        PLANBinarySearchGaussian().run(
            metadata, inaccessible_input, ZCDPBudget(rho=1.0), seed=0
        )


def test_plan_small_sample_refusal_precedes_input_access():
    inaccessible_input = cast(MeanEstimationInput, Unreadable("Input"))
    metadata = mean_metadata(1, box(*[(-1.0, 1.0)] * 2))
    with pytest.raises(InvalidRegime, match="n_samples >="):
        PLANBinarySearchGaussian().run(
            metadata, inaccessible_input, ZCDPBudget(rho=1.0), seed=0
        )


def test_plan_nonpositive_quantile_refusal_precedes_input_access():
    inaccessible_input = cast(MeanEstimationInput, Unreadable("Input"))
    metadata = mean_metadata(2, box(*[(-1.0, 1.0)] * 2))
    with pytest.raises(InvalidRegime, match="quantile"):
        PLANBinarySearchGaussian().run(
            metadata, inaccessible_input, ZCDPBudget(rho=1.0), seed=0
        )


def test_binary_quantile_matches_the_hand_derived_bisection() -> None:
    """Independent oracle: at infinite budget the mechanism is the exact
    noiseless bisection, whose fixpoint is hand-derivable from the branch
    invariant. For values (1, 2, 3) on [0, 4] with target rank
    0.5 * population = 1.5, every midpoint below 2 has count 1 <= 1.5 and
    every midpoint above 2 has count 2 > 1.5, so the interval contracts to 2.
    The two sentinel values above the search interval prove the rank comes
    from the public population, not the realized array length."""
    from dpstats.tasks.mean_estimation.algorithms.plan import (
        _binary_private_quantile,
    )

    result = _binary_private_quantile(
        np.array([1.0, 2.0, 3.0, 100.0, 100.0]),
        0.5,
        0.0,
        4.0,
        ZCDPBudget(rho=np.inf),
        seed=5,
        num_rounds=20,
        population=3,
    )

    assert result == pytest.approx(2.0, abs=4.0 / 2**19)


@pytest.mark.parametrize("refused", [PureDPBudget(eps=1.0), GDPBudget(mu=1.0)])
def test_finite_pure_and_gdp_offers_are_refused_by_the_seam(refused: DPBudget) -> None:
    metadata = mean_metadata(4, box((-2.0, 2.0)))
    input_data = MeanEstimationInput(X=np.array([[0.5], [1.0], [-1.0], [0.0]]))

    with pytest.raises(InvalidRegime):
        PLANBinarySearchGaussian().run(metadata, input_data, refused, seed=0)
