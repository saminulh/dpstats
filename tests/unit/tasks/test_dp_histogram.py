"""Public release-law tests for a complete categorical mean block."""

import numpy as np
import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.budgets.accounting import spend_as_pure
from dpstats.domains import OneHotDomain, ProductDomain
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import list_built_in_algorithms
from tests.registry_components import ExternalBudget
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order
from tests.unit.tasks.task_builders import Unreadable, box, mean_metadata


@pytest.mark.parametrize("allow_zero", [False, True])
def test_whole_declared_block_uses_one_pure_vector_laplace_release(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
    allow_zero,
):
    fake_randomness.script("laplace", draws_in_order([np.array([1.0, -2.0, 3.0])]))
    offer = ApproxDPBudget(eps=2.0, delta=0.2)
    output = list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
        mean_metadata(4, OneHotDomain(dim=3, allow_zero=allow_zero)),
        mean_estimation_task.Input(
            X=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0 if allow_zero else 1, 0], [1, 0, 0]]
            )
        ),
        budget=offer,
        seed=17,
    )

    pure, laplace = accounting_spy.calls
    assert pure.entry_point == "spend_as_pure"
    assert_budget_close(pure.budget, offer)
    assert laplace.entry_point == "calibrate_laplace_scale"
    assert_budget_close(laplace.budget, PureDPBudget(eps=2.0))
    assert laplace.sensitivity == pytest.approx(0.25)
    assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx([0.125])
    assert fake_randomness.parameters_of("laplace", "size") == [(3,)]
    assert fake_randomness.seeds == (17,)
    # The unobserved third code remains in the release; nothing is projected.
    np.testing.assert_allclose(
        output.mean_est, [1.5, -1.75 if allow_zero else -1.5, 3.0]
    )


@pytest.mark.parametrize("budget", [ZCDPBudget(rho=1.0), GDPBudget(mu=1.0)])
def test_finite_concentrated_offers_delegate_to_shared_pure_accounting(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
    budget,
):
    fake_randomness.script("laplace", draws_in_order([np.array([1.0, -2.0])]))
    output = list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
        mean_metadata(4, OneHotDomain(dim=2)),
        mean_estimation_task.Input(X=np.array([[1, 0], [0, 1], [1, 0], [0, 1]])),
        budget=budget,
        seed=17,
    )

    pure, laplace = accounting_spy.calls
    assert pure.entry_point == "spend_as_pure"
    assert_budget_close(pure.budget, budget)
    assert laplace.entry_point == "calibrate_laplace_scale"
    assert_budget_close(laplace.budget, spend_as_pure(budget))
    assert laplace.sensitivity == pytest.approx(0.25)
    np.testing.assert_allclose(output.mean_est, [1.5, -1.5])


def test_unknown_budget_refusal_comes_from_accounting_before_input_or_randomness(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
):
    budget = ExternalBudget(credits=1.0)
    with pytest.raises(InvalidRegime) as accounting_refusal:
        spend_as_pure(budget)
    with pytest.raises(InvalidRegime) as raised:
        list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
            mean_metadata(4, OneHotDomain(dim=2)),
            Unreadable("Input"),
            budget=budget,
            seed=0,
        )
    assert str(raised.value) == str(accounting_refusal.value)
    (pure,) = accounting_spy.calls
    assert pure.entry_point == "spend_as_pure"
    assert pure.budget is budget
    assert fake_randomness.seeds == ()


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=1.0),
        ApproxDPBudget(eps=1.0, delta=1e-5),
        ZCDPBudget(rho=1.0),
        GDPBudget(mu=1.0),
        ExternalBudget(credits=1.0),
    ],
)
def test_full_coded_singleton_is_a_public_constant(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
    budget,
):
    output = list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
        mean_metadata(7, OneHotDomain(dim=1)),
        Unreadable("Input"),
        budget=budget,
        seed=0,
    )
    np.testing.assert_array_equal(output.mean_est, [1.0])
    assert accounting_spy.calls == ()
    assert fake_randomness.seeds == ()


@pytest.mark.parametrize(
    "domain",
    [
        box((0.0, 1.0)),
        ProductDomain(parts=(OneHotDomain(dim=2),)),
        ProductDomain(parts=(OneHotDomain(dim=2), OneHotDomain(dim=2))),
    ],
)
def test_non_one_hot_domain_refuses_before_input_accounting_or_randomness(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
    domain,
):
    with pytest.raises(InvalidRegime, match="top-level OneHotDomain"):
        list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
            mean_metadata(4, domain),
            Unreadable("Input"),
            budget=NO_PRIVACY,
            seed=0,
        )
    assert accounting_spy.calls == ()
    assert fake_randomness.seeds == ()


def test_reference_coded_singleton_still_needs_a_release(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
):
    fake_randomness.script("laplace", draws_in_order([np.array([-1.0])]))
    output = list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
        mean_metadata(4, OneHotDomain(dim=1, allow_zero=True)),
        mean_estimation_task.Input(X=np.array([[1], [0], [1], [0]])),
        budget=PureDPBudget(eps=2.0),
        seed=0,
    )
    assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx([0.125])
    np.testing.assert_allclose(output.mean_est, [-0.5])


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=np.inf),
        ApproxDPBudget(eps=np.inf, delta=1e-5),
        ApproxDPBudget(eps=2.0, delta=1.0),
        ZCDPBudget(rho=np.inf),
        GDPBudget(mu=np.inf),
        ExternalBudget(credits=np.inf),
    ],
)
@pytest.mark.parametrize("seed", [0, 73])
def test_unconstrained_offers_have_the_exact_mean_limit(
    mean_estimation_task,
    budget,
    seed,
):
    output = list_built_in_algorithms(mean_estimation_task)["DPHistogram"]().run(
        mean_metadata(4, OneHotDomain(dim=3, allow_zero=True)),
        mean_estimation_task.Input(
            X=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
        ),
        budget=budget,
        seed=seed,
    )
    np.testing.assert_array_equal(output.mean_est, [0.5, 0.25, 0.0])
