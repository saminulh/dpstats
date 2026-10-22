from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.tasks.linear_regression.algorithms.boosted_adassp import (
    BoostedAdaSSP,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy


def test_large_budget_run_converges() -> None:
    x = np.linspace(-0.5, 0.5, 200)
    X = x[:, None]
    y = 2.0 * x
    metadata = LinearRegressionMetadata(
        n_samples=x.size,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=X, y=y)
    algorithm = BoostedAdaSSP(
        num_rounds=30,
        tau=10.0,
    )

    budget = ApproxDPBudget(eps=1e6, delta=0.1)
    output = algorithm.run(metadata, input, budget=budget, seed=9)

    np.testing.assert_allclose(output.beta, np.array([2.0]), atol=5e-3)


def test_author_equal_split_reaches_each_accounted_release(
    accounting_spy: AccountingSpy,
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.zeros((2, 1)), y=np.zeros(2))

    budget = GDPBudget(mu=math.sqrt(3.0))
    BoostedAdaSSP(num_rounds=4, tau=2.0).run(
        metadata,
        input,
        budget,
        seed=7,
    )

    spend_call, gram_call, eigenvalue_call, gradient_call = accounting_spy.calls
    assert spend_call == AccountingCall("spend_as_gdp", budget)
    assert isinstance(gram_call.budget, GDPBudget)
    assert gram_call.budget.mu == pytest.approx(1.0)
    assert gram_call.entry_point == "calibrate_gaussian_scale"
    assert gram_call.sensitivity == pytest.approx(1.0)
    assert isinstance(eigenvalue_call.budget, GDPBudget)
    assert eigenvalue_call.budget.mu == pytest.approx(1.0)
    assert eigenvalue_call.entry_point == "calibrate_gaussian_scale"
    assert eigenvalue_call.sensitivity == pytest.approx(1.0)
    assert isinstance(gradient_call.budget, GDPBudget)
    assert gradient_call.budget.mu == pytest.approx(1.0)
    assert gradient_call.entry_point == "compute_noise_multiplier_for_composed_gaussian"
    assert gradient_call.num_steps == 4


def test_approximate_dp_uses_shared_gdp_spend(
    accounting_spy: AccountingSpy,
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.zeros((2, 1)), y=np.zeros(2))

    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    BoostedAdaSSP(num_rounds=1).run(
        metadata,
        input,
        budget,
        seed=7,
    )

    assert accounting_spy.calls[0] == AccountingCall("spend_as_gdp", budget)


def test_refuses_pure_dp() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.ones((2, 1)), y=np.ones(2))

    with pytest.raises(InvalidRegime, match="PureDPBudget"):
        BoostedAdaSSP(num_rounds=1).run(metadata, input, PureDPBudget(eps=1.0), seed=7)


def test_requires_finite_feature_bound() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=EuclideanSpace(dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.zeros((2, 1)), y=np.zeros(2))

    with pytest.raises(InvalidRegime, match="finite covariate"):
        BoostedAdaSSP(num_rounds=1).run(metadata, input, NO_PRIVACY, seed=7)


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-5),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
def test_accepts_every_unconstrained_budget(budget: DPBudget) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.array([[0.0], [1.0]]), y=np.array([0.0, 1.0]))

    output = BoostedAdaSSP(num_rounds=1).run(metadata, input, budget, seed=7)

    assert np.all(np.isfinite(output.beta))


def test_appendix_learning_rate_schedule_changes_the_update() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(
        X=np.array([[-1.0], [0.0], [1.0]]),
        y=np.array([-1.0, 0.0, 1.0]),
    )

    constant = BoostedAdaSSP(
        num_rounds=3,
        learning_rate_schedule="constant",
    ).run(metadata, input, NO_PRIVACY, seed=4)
    inverse = BoostedAdaSSP(
        num_rounds=3,
        learning_rate_schedule="inverse",
    ).run(metadata, input, NO_PRIVACY, seed=4)

    assert not np.allclose(constant.beta, inverse.beta)


def test_public_feature_bound_transport_preserves_predictions() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=LpBallDomain(p=2, radius=3.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(
        X=np.array([[-3.0], [0.0], [2.0]]),
        y=np.array([-1.0, 0.0, 1.0]),
    )
    unit_metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=LpBallDomain(p=2, radius=1.0, center=np.zeros(1)),
        response_domain=EuclideanSpace(dim=1),
    )
    unit_input = LinearRegressionInput(
        X=input.X / 3.0,
        y=input.y,
    )

    original = BoostedAdaSSP(num_rounds=3).run(metadata, input, NO_PRIVACY, seed=4)
    unit = BoostedAdaSSP(num_rounds=3).run(
        unit_metadata,
        unit_input,
        NO_PRIVACY,
        seed=4,
    )

    np.testing.assert_allclose(input.X @ original.beta, unit_input.X @ unit.beta)
