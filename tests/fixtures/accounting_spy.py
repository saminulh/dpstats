"""Shared spy for algorithm accounting-conformance tests."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, cast

import pytest

import dpstats.budgets.accounting as accounting
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)


AccountingEntryPoint = Literal[
    "spend_as_pure",
    "spend_as_zcdp",
    "spend_as_zcdp_via_rdp",
    "spend_as_gdp",
    "calibrate_laplace_scale",
    "calibrate_gaussian_scale",
    "compute_noise_multiplier_for_composed_gaussian",
    "compute_noise_multiplier_for_subsampled_gaussian",
]


@dataclass(frozen=True)
class AccountingCall:
    """One observed call through a typed-budget accounting entry point."""

    entry_point: AccountingEntryPoint
    budget: DPBudget
    sensitivity: float | None = None
    sampling_rate: float | None = None
    num_steps: int | None = None


class AccountingSpy:
    """Read-only record of typed-budget accounting calls in execution order.

    A call is recorded when it is entered, so a call that refuses its budget
    is still in ``calls``.
    """

    def __init__(self) -> None:
        self._calls: list[AccountingCall] = []

    @property
    def calls(self) -> tuple[AccountingCall, ...]:
        """Return the calls observed during this test."""
        return tuple(self._calls)

    def record(
        self,
        entry_point: AccountingEntryPoint,
        budget: DPBudget,
        sensitivity: float | None = None,
        sampling_rate: float | None = None,
        num_steps: int | None = None,
    ) -> None:
        """Record one entered call."""
        self._calls.append(
            AccountingCall(
                entry_point,
                budget,
                sensitivity=sensitivity,
                sampling_rate=sampling_rate,
                num_steps=num_steps,
            )
        )


def assert_budget_close(actual: DPBudget, expected: DPBudget) -> None:
    """Assert matching privacy families and approximately equal parameters."""
    assert type(actual) is type(expected)
    if isinstance(expected, NoPrivacy):
        assert actual is expected
    elif isinstance(expected, PureDPBudget):
        assert isinstance(actual, PureDPBudget)
        assert actual.eps == pytest.approx(expected.eps)
    elif isinstance(expected, ApproxDPBudget):
        assert isinstance(actual, ApproxDPBudget)
        assert actual.eps == pytest.approx(expected.eps)
        assert actual.delta == pytest.approx(expected.delta)
    elif isinstance(expected, ZCDPBudget):
        assert isinstance(actual, ZCDPBudget)
        assert actual.rho == pytest.approx(expected.rho)
    else:
        assert isinstance(actual, GDPBudget)
        expected_gdp = cast(GDPBudget, expected)
        assert actual.mu == pytest.approx(expected_gdp.mu)


@pytest.fixture
def accounting_spy(monkeypatch: pytest.MonkeyPatch) -> AccountingSpy:
    """Record outermost accounting calls while delegating to the real helpers."""
    spy = AccountingSpy()
    depth = 0

    @contextmanager
    def record_outermost(
        entry_point: AccountingEntryPoint,
        budget: DPBudget,
        sensitivity: float | None = None,
        sampling_rate: float | None = None,
        num_steps: int | None = None,
    ) -> Generator[None, None, None]:
        nonlocal depth
        if depth == 0:
            spy.record(
                entry_point,
                budget,
                sensitivity=sensitivity,
                sampling_rate=sampling_rate,
                num_steps=num_steps,
            )
        depth += 1
        try:
            yield
        finally:
            depth -= 1

    pure = accounting.spend_as_pure
    zcdp_rdp = accounting.spend_as_zcdp_via_rdp
    zcdp = accounting.spend_as_zcdp
    gdp = accounting.spend_as_gdp
    gaussian = accounting.calibrate_gaussian_scale
    laplace = accounting.calibrate_laplace_scale
    composed_gaussian = accounting.compute_noise_multiplier_for_composed_gaussian
    subsampled_gaussian = accounting.compute_noise_multiplier_for_subsampled_gaussian

    @wraps(pure)
    def record_pure(budget: DPBudget) -> PureDPBudget:
        with record_outermost("spend_as_pure", budget):
            return pure(budget)

    @wraps(zcdp)
    def record_zcdp(budget: DPBudget) -> ZCDPBudget:
        with record_outermost("spend_as_zcdp", budget):
            return zcdp(budget)

    @wraps(zcdp_rdp)
    def record_zcdp_rdp(budget: ApproxDPBudget) -> ZCDPBudget:
        with record_outermost("spend_as_zcdp_via_rdp", budget):
            return zcdp_rdp(budget)

    @wraps(gdp)
    def record_gdp(budget: DPBudget) -> GDPBudget:
        with record_outermost("spend_as_gdp", budget):
            return gdp(budget)

    @wraps(gaussian)
    def record_gaussian(budget: DPBudget, sensitivity: float) -> float:
        with record_outermost(
            "calibrate_gaussian_scale", budget, sensitivity=sensitivity
        ):
            return gaussian(budget, sensitivity)

    @wraps(laplace)
    def record_laplace(budget: DPBudget, sensitivity: float) -> float:
        with record_outermost(
            "calibrate_laplace_scale", budget, sensitivity=sensitivity
        ):
            return laplace(budget, sensitivity)

    @wraps(composed_gaussian)
    def record_composed_gaussian(budget: DPBudget, num_steps: int) -> float:
        with record_outermost(
            "compute_noise_multiplier_for_composed_gaussian",
            budget,
            num_steps=num_steps,
        ):
            return composed_gaussian(budget, num_steps)

    @wraps(subsampled_gaussian)
    def record_subsampled_gaussian(
        budget: DPBudget, sampling_rate: float, num_steps: int
    ) -> float:
        with record_outermost(
            "compute_noise_multiplier_for_subsampled_gaussian",
            budget,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
        ):
            return cast("Callable[[DPBudget, float, int], float]", subsampled_gaussian)(
                budget, sampling_rate, num_steps
            )

    patches: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("spend_as_pure", record_pure),
        ("spend_as_zcdp", record_zcdp),
        ("spend_as_zcdp_via_rdp", record_zcdp_rdp),
        ("spend_as_gdp", record_gdp),
        ("calibrate_laplace_scale", record_laplace),
        ("calibrate_gaussian_scale", record_gaussian),
        ("compute_noise_multiplier_for_composed_gaussian", record_composed_gaussian),
        (
            "compute_noise_multiplier_for_subsampled_gaussian",
            record_subsampled_gaussian,
        ),
    )
    for name, replacement in patches:
        monkeypatch.setattr(accounting, name, replacement)

    return spy
