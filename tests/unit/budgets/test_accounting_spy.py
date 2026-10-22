"""Tests for the shared accounting-conformance spy fixture."""

import pytest

import dpstats.budgets.accounting as accounting
from dpstats.budgets import (
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
)


def test_spy_records_outermost_routes_in_order_and_delegates(
    accounting_spy: AccountingSpy,
) -> None:
    pure = PureDPBudget(eps=1.25)
    zcdp = ZCDPBudget(rho=0.5)
    gdp = GDPBudget(mu=0.75)

    assert accounting.spend_as_pure(pure) is pure
    assert accounting.spend_as_zcdp(zcdp) is zcdp
    assert accounting.spend_as_gdp(gdp) is gdp
    assert accounting.calibrate_gaussian_scale(zcdp, sensitivity=2.0) == pytest.approx(
        2.0
    )
    assert accounting.compute_noise_multiplier_for_composed_gaussian(
        zcdp, 4
    ) == pytest.approx(2.0)
    assert accounting.compute_noise_multiplier_for_subsampled_gaussian(
        zcdp, 0.5, 4
    ) == pytest.approx(2.0)
    assert accounting_spy.calls == (
        AccountingCall("spend_as_pure", pure),
        AccountingCall("spend_as_zcdp", zcdp),
        AccountingCall("spend_as_gdp", gdp),
        AccountingCall("calibrate_gaussian_scale", zcdp, sensitivity=2.0),
        AccountingCall(
            "compute_noise_multiplier_for_composed_gaussian",
            zcdp,
            num_steps=4,
        ),
        AccountingCall(
            "compute_noise_multiplier_for_subsampled_gaussian",
            zcdp,
            sampling_rate=0.5,
            num_steps=4,
        ),
    )


def test_spy_records_refusal_before_real_helper_raises(
    accounting_spy: AccountingSpy,
) -> None:
    budget = PureDPBudget(eps=1.0)

    with pytest.raises(ValueError, match="PureDPBudget"):
        accounting.spend_as_zcdp(budget)

    assert accounting_spy.calls == (AccountingCall("spend_as_zcdp", budget),)
