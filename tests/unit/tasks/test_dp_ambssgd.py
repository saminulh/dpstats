# pyright: reportPrivateUsage=false
"""Tests for Varshney--Thakurta--Jain DP-AMBSSGD."""

from __future__ import annotations

from dpstats.domains import EuclideanSpace

import math

import numpy as np
import pytest

from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms import (
    dp_ambssgd as dp_ambssgd_module,
)
from dpstats.tasks.linear_regression.algorithms.dp_ambssgd import DPAMBSSGD
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw


def _instance(
    n: int = 220, d: int = 2
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, d))
    beta = np.arange(1, d + 1, dtype=np.float64) / d
    y = X @ beta + rng.normal(scale=0.2, size=n)
    return (
        LinearRegressionMetadata(
            n_samples=n,
            design_domain=EuclideanSpace(dim=d),
            response_domain=EuclideanSpace(dim=1),
        ),
        LinearRegressionInput(X=X, y=y),
    )


def test_default_schedule_uses_odd_final_complete_block(
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
) -> None:
    metadata, input = _instance(n=33, d=1)
    monkeypatch.setattr(dp_ambssgd_module, "_dp_stat", lambda *_args: 1.0)
    monkeypatch.setattr(
        dp_ambssgd_module,
        "clip_rows",
        lambda X, _radius: np.ones_like(X),
    )
    fake_randomness.script("permutation", lambda parameters: np.arange(parameters["x"]))
    fake_randomness.script("normal", zero_draw)
    monkeypatch.setattr(
        dp_ambssgd_module.accounting,
        "calibrate_gaussian_scale",
        lambda *_args: 0.0,
    )
    fake_randomness.forget()

    output = DPAMBSSGD(learning_rate_multiplier=4.0).run(
        metadata,
        input,
        ZCDPBudget(rho=0.2),
        seed=0,
    )

    # Three complete blocks produce iterates -1, -2, -3. The last
    # ceil(3 / 2) iterates average to -2.5.
    assert output.beta == pytest.approx(np.array([-2.5]))
    # The shuffle and every per-step gradient release draw from their own
    # child stream, never the offered seed and never each other's.
    seeds = fake_randomness.seeds
    assert len(seeds) > 1
    assert len(set(seeds)) == len(seeds)
    assert 0 not in seeds


def test_dp_stat_uses_the_proof_compatible_doubling_grid(
    fake_randomness: FakeRandomness,
) -> None:
    X = np.zeros((3, 1), dtype=np.float64)
    y = np.full(3, 8.0)
    w = np.zeros(1, dtype=np.float64)
    fake_randomness.script("normal", zero_draw)
    budget = ZCDPBudget(rho=1.0 / (2.0 * 3.0**2))
    gamma = dp_ambssgd_module._dp_stat(
        X,
        y,
        w,
        domain_size=8.0,
        failure_probability=0.125,
        budget=budget,
        seed=0,
    )
    # beta=1/8 gives Delta=1 and q=3, so the tested grid is 2, 4, 8.
    scales = fake_randomness.parameters_of("normal", "scale")
    assert len(scales) == 3
    assert scales[0] == pytest.approx(3.0 * math.sqrt(3.0))
    assert gamma == 8.0


def test_failure_probability_reaches_each_dp_stat_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input = _instance(n=22, d=1)
    observed: list[float] = []

    def fake_dp_stat(
        _X: Array2D,
        _y: Array1D,
        _w: Array1D,
        _domain_size: float,
        failure_probability: float,
        _budget: ZCDPBudget,
        _seed: int,
    ) -> float:
        observed.append(failure_probability)
        return 1.0

    monkeypatch.setattr(dp_ambssgd_module, "_dp_stat", fake_dp_stat)

    DPAMBSSGD(failure_probability=0.125).run(
        metadata,
        input,
        ZCDPBudget(rho=0.2),
        seed=0,
    )

    assert observed == [0.125, 0.125]


def _run_two_step_gradient_fixture(
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
    second_X: float,
    second_y: float,
    second_radius: float,
) -> float:
    n = 22
    X = np.zeros((n, 1))
    y = np.zeros(n)
    X[1:11, 0] = 1.0
    y[1:11] = 4.0
    X[12:22, 0] = second_X
    y[12:22] = second_y
    metadata = LinearRegressionMetadata(
        n_samples=n,
        design_domain=EuclideanSpace(dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=X, y=y)
    radii = iter((4.0, second_radius))
    log_factor = math.log(n) ** 0.5
    monkeypatch.setattr(
        dp_ambssgd_module,
        "_dp_stat",
        lambda *_args, **_kwargs: next(radii) / log_factor,
    )
    fake_randomness.script("permutation", lambda parameters: np.arange(parameters["x"]))
    fake_randomness.script("normal", zero_draw)
    monkeypatch.setattr(
        dp_ambssgd_module.accounting,
        "calibrate_gaussian_scale",
        lambda *_args, **_kwargs: 0.0,
    )
    return float(DPAMBSSGD().run(metadata, input, ZCDPBudget(rho=1.0), seed=0).beta[0])


def test_public_run_matches_ordinary_source_gradient(
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
) -> None:
    beta = _run_two_step_gradient_fixture(
        monkeypatch,
        fake_randomness,
        second_X=2.0,
        second_y=1.0,
        second_radius=10.0,
    )
    assert beta == pytest.approx(0.5)


def test_pure_dp_is_refused_through_zcdp_entry(
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _instance()
    budget = PureDPBudget(eps=1.0)

    with pytest.raises(InvalidRegime, match="PureDPBudget"):
        DPAMBSSGD().run(metadata, input, budget, seed=0)

    assert len(accounting_spy.calls) == 1
    assert accounting_spy.calls[0].entry_point == "spend_as_zcdp"


def test_incomplete_paper_schedule_rejected() -> None:
    metadata, input = _instance(n=21)
    with pytest.raises(InvalidRegime, match="two complete iteration blocks"):
        DPAMBSSGD().run(metadata, input, ZCDPBudget(rho=0.2), seed=0)


@pytest.mark.parametrize("n", [200, 240, 256, 404])
def test_public_run_handles_ordinary_sample_sizes(n: int) -> None:
    metadata, input = _instance(n=n, d=2)

    output = DPAMBSSGD().run(
        metadata, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
    )

    assert output.beta.shape == (2,)
    assert np.isfinite(output.beta).all()


@pytest.mark.parametrize(
    "budget",
    [ApproxDPBudget(eps=1.0, delta=1e-5), ZCDPBudget(rho=0.2)],
)
def test_accounting_conformance_uses_parallel_full_rho_blocks(
    budget: DPBudget,
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _instance(n=22, d=1)

    DPAMBSSGD().run(metadata, input, budget, seed=0)

    assert accounting_spy.calls[0].entry_point == "spend_as_zcdp"
    gaussian_calls = accounting_spy.calls[1:]
    assert len(gaussian_calls) == 4
    assert all(
        call.entry_point == "calibrate_gaussian_scale" for call in gaussian_calls
    )
    stat_calls = gaussian_calls[::2]
    gradient_calls = gaussian_calls[1::2]
    assert all(isinstance(call.budget, ZCDPBudget) for call in gaussian_calls)
    # DP-STAT and the gradient child use disjoint slices, and both iterations
    # use disjoint blocks, so every child receives the full offered rho.
    observed_rhos: list[float] = []
    for stat_call, gradient_call in zip(stat_calls, gradient_calls):
        assert isinstance(stat_call.budget, ZCDPBudget)
        assert isinstance(gradient_call.budget, ZCDPBudget)
        # The default beta=0.01 gives q=ceil(log2(100))=7 queries.
        stat_total = stat_call.budget.rho * 7
        assert stat_total == pytest.approx(gradient_call.budget.rho)
        observed_rhos.append(max(stat_total, gradient_call.budget.rho))
    # Each child spends exactly the zCDP budget the offered budget grants.
    granted = accounting.spend_as_zcdp(budget).rho
    assert observed_rhos == pytest.approx([granted, granted])
