"""Property coverage for the typed privacy-budget wire contracts.

Laws checked over generated budgets and weight vectors: every budget family
round-trips through its canonical JSON record, ``DPBudget.split`` parts stay
proportional to their weights (GDP in quadrature), unconstrained offers are
recognized, and invalid weight vectors are refused.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.budgets.families import canonical_budget_parameter
from tests import hypothesis_profile  # pyright: ignore[reportUnusedImport]
from tests.fixtures.accounting_spy import assert_budget_close

# Generators draw from the supported numerical envelope of each privacy
# parameter (eps and mu in [1e-5, 100], delta in [1e-12, 0.1], rho in
# [1e-10, 1e4]), plus the unconstrained infinite limit.
_weights = st.lists(st.floats(min_value=1e-3, max_value=1e3), min_size=1, max_size=6)
_eps_or_mu = st.one_of(st.floats(min_value=1e-5, max_value=100.0), st.just(math.inf))
_budgets = st.one_of(
    st.builds(PureDPBudget, eps=_eps_or_mu),
    st.builds(
        ApproxDPBudget,
        eps=_eps_or_mu,
        delta=st.floats(min_value=1e-12, max_value=0.1),
    ),
    st.builds(
        ZCDPBudget,
        rho=st.one_of(st.floats(min_value=1e-10, max_value=1e4), st.just(math.inf)),
    ),
    st.builds(GDPBudget, mu=_eps_or_mu),
    st.just(NO_PRIVACY),
)


@given(budget=_budgets)
def test_every_family_round_trips_through_canonical_wire_bytes(
    budget: DPBudget,
) -> None:
    record = budget.model_dump(mode="json")

    persisted = decode_json_document(canonical_json_bytes(record))

    assert type(budget).model_validate(persisted) == budget


@pytest.mark.parametrize(
    ("offer", "unconstrained"),
    [
        (NO_PRIVACY, True),
        (PureDPBudget(eps=math.inf), True),
        (ApproxDPBudget(eps=math.inf, delta=0.01), True),
        (ZCDPBudget(rho=math.inf), True),
        (GDPBudget(mu=math.inf), True),
        (ApproxDPBudget(eps=1.0, delta=1.0), True),
        (PureDPBudget(eps=1e-5), False),
        (ApproxDPBudget(eps=100.0, delta=0.1), False),
        (ZCDPBudget(rho=1e4), False),
        (GDPBudget(mu=100.0), False),
    ],
)
def test_unconstrained_offers_are_recognized_per_family(
    offer: DPBudget, unconstrained: bool
) -> None:
    assert offer.is_unconstrained is unconstrained


def test_a_delta_rounding_to_one_canonicalizes_to_exactly_one() -> None:
    # 0.9999999999999 rounds to 1.0 at BUDGET_IDENTITY_DIGITS (12 significant
    # digits); every parameter is canonicalized at construction, so the
    # stored delta is exactly 1.0 and the offer is unconstrained, the same
    # as any other value that rounds to 1.
    budget = ApproxDPBudget(eps=1.0, delta=0.9999999999999)

    assert budget.delta == 1.0
    assert budget.is_unconstrained is True


def test_infinite_parameters_canonicalize_to_infinity() -> None:
    assert canonical_budget_parameter(math.inf) == math.inf
    assert PureDPBudget(eps=math.inf).eps == math.inf
    assert ApproxDPBudget(eps=math.inf, delta=0.5).eps == math.inf
    assert ZCDPBudget(rho=math.inf).rho == math.inf
    assert GDPBudget(mu=math.inf).mu == math.inf


_unconstrained_budgets = st.one_of(
    st.just(NO_PRIVACY),
    st.builds(PureDPBudget, eps=st.just(math.inf)),
    st.builds(
        ApproxDPBudget,
        eps=st.just(math.inf),
        delta=st.floats(min_value=1e-12, max_value=1.0),
    ),
    st.builds(
        ApproxDPBudget,
        eps=st.floats(min_value=1e-5, max_value=100.0),
        delta=st.just(1.0),
    ),
    st.builds(ZCDPBudget, rho=st.just(math.inf)),
    st.builds(GDPBudget, mu=st.just(math.inf)),
)


@given(budget=_unconstrained_budgets, weights=_weights)
def test_split_of_an_unconstrained_budget_stays_unconstrained(
    budget: DPBudget, weights: list[float]
) -> None:
    parts = budget.split(weights)

    assert all(part.is_unconstrained for part in parts)


def _assert_split_matches_the_offer(
    budget: DPBudget, weights: list[float], parts: tuple[DPBudget, ...]
) -> None:
    """Check each part's share and the family's conservation law in one pass.

    Proportionality compares every part against ``weight / total`` of the
    offered parameter. Conservation accumulates the parts' parameters and
    compares the sum against the offer; GDP conserves in quadrature, since
    GDP composes by adding squared ``mu`` values.
    """
    total = math.fsum(weights)
    eps_sum = 0.0
    delta_sum = 0.0
    rho_sum = 0.0
    mu_square_sum = 0.0

    for part, weight in zip(parts, weights):
        share = weight / total
        match (budget, part):
            case (NoPrivacy(), NoPrivacy()):
                pass
            case (PureDPBudget(), PureDPBudget()):
                assert part.eps == pytest.approx(budget.eps * share, rel=1e-9)
                eps_sum += part.eps
            case (ApproxDPBudget(), ApproxDPBudget()):
                assert part.eps == pytest.approx(budget.eps * share, rel=1e-9)
                assert part.delta == pytest.approx(budget.delta * share, rel=1e-9)
                eps_sum += part.eps
                delta_sum += part.delta
            case (ZCDPBudget(), ZCDPBudget()):
                assert part.rho == pytest.approx(budget.rho * share, rel=1e-9)
                rho_sum += part.rho
            case (GDPBudget(), GDPBudget()):
                assert part.mu == pytest.approx(budget.mu * math.sqrt(share), rel=1e-9)
                mu_square_sum += part.mu**2
            case _:
                raise AssertionError(
                    f"split part {part!r} is not in the offered family {budget!r}"
                )

    match budget:
        case NoPrivacy():
            # Every part already matched the no-constraint offer above.
            pass
        case PureDPBudget():
            assert eps_sum == pytest.approx(budget.eps, rel=1e-9)
        case ApproxDPBudget():
            assert eps_sum == pytest.approx(budget.eps, rel=1e-9)
            assert delta_sum == pytest.approx(budget.delta, rel=1e-9)
        case ZCDPBudget():
            assert rho_sum == pytest.approx(budget.rho, rel=1e-9)
        case GDPBudget():
            assert mu_square_sum == pytest.approx(budget.mu**2, rel=1e-9)
        case _:
            raise AssertionError(f"unexpected offered family {budget!r}")


@given(budget=_budgets, weights=_weights)
def test_split_parts_stay_proportional_and_conserve_the_offer(
    budget: DPBudget, weights: list[float]
) -> None:
    parts = budget.split(weights)

    assert len(parts) == len(weights)
    _assert_split_matches_the_offer(budget, weights, parts)


def _assert_split_parts_are_canonical(budget: DPBudget, weights: list[float]) -> None:
    """A part is built through the class constructor, not model_copy.

    ``model_copy`` bypasses field validators, so a part built that way would
    carry a raw, non-canonical parameter instead of the canonicalized value
    every other construction path guarantees. Each part must equal a fresh
    budget built directly from its own fields, and each field must already
    equal its own canonical form.
    """
    for part in budget.split(weights):
        match part:
            case NoPrivacy():
                assert part == NoPrivacy()
            case PureDPBudget():
                assert part.eps == canonical_budget_parameter(part.eps)
                assert part == PureDPBudget(eps=part.eps)
            case ApproxDPBudget():
                assert part.eps == canonical_budget_parameter(part.eps)
                assert part.delta == canonical_budget_parameter(part.delta)
                if isinstance(budget, ApproxDPBudget) and budget.delta == 1.0:
                    assert part.delta == 1.0
                assert part == ApproxDPBudget(eps=part.eps, delta=part.delta)
            case ZCDPBudget():
                assert part.rho == canonical_budget_parameter(part.rho)
                assert part == ZCDPBudget(rho=part.rho)
            case GDPBudget():
                assert part.mu == canonical_budget_parameter(part.mu)
                assert part == GDPBudget(mu=part.mu)
            case _:
                raise AssertionError(f"split part {part!r} is not a known family")


@given(budget=_budgets, weights=_weights)
def test_split_parts_are_canonical_and_reconstruct_from_their_own_fields(
    budget: DPBudget, weights: list[float]
) -> None:
    _assert_split_parts_are_canonical(budget, weights)


@given(budget=_unconstrained_budgets, weights=_weights)
def test_split_parts_of_an_unconstrained_offer_are_canonical(
    budget: DPBudget, weights: list[float]
) -> None:
    """Covers the delta=1 case the general ``_budgets`` strategy never draws."""
    _assert_split_parts_are_canonical(budget, weights)


@given(
    weights=_weights,
    invalid=st.one_of(st.floats(max_value=0.0), st.just(math.nan), st.just(math.inf)),
    position=st.integers(min_value=0, max_value=6),
)
def test_nonpositive_or_nonfinite_weights_are_refused(
    weights: list[float], invalid: float, position: int
) -> None:
    index = min(position, len(weights))
    offered = weights[:index] + [invalid] + weights[index:]

    with pytest.raises(ValueError, match="weights must be"):
        PureDPBudget(eps=1.0).split(offered)


def test_empty_weights_are_refused() -> None:
    with pytest.raises(ValueError, match="weights must be nonempty"):
        PureDPBudget(eps=1.0).split([])


@pytest.mark.parametrize(
    "budget,expected",
    [
        (PureDPBudget(eps=3.0), (PureDPBudget(eps=0.75), PureDPBudget(eps=2.25))),
        (
            ApproxDPBudget(eps=3.0, delta=0.04),
            (
                ApproxDPBudget(eps=0.75, delta=0.01),
                ApproxDPBudget(eps=2.25, delta=0.03),
            ),
        ),
        (ZCDPBudget(rho=3.0), (ZCDPBudget(rho=0.75), ZCDPBudget(rho=2.25))),
        (GDPBudget(mu=3.0), (GDPBudget(mu=1.5), GDPBudget(mu=1.5 * math.sqrt(3.0)))),
        (NO_PRIVACY, (NO_PRIVACY, NO_PRIVACY)),
    ],
)
def test_split_example_keeps_the_family_and_weight_ratio(
    budget: DPBudget, expected: tuple[DPBudget, ...]
) -> None:
    parts = budget.split((1.0, 3.0))

    assert len(parts) == len(expected)
    for part, expected_part in zip(parts, expected):
        assert_budget_close(part, expected_part)
