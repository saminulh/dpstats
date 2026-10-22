"""Tests for typed privacy-budget values."""

import math
from collections.abc import Callable

import pytest

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
from tests.registry_components import ExternalBudget


def test_budgets_have_value_semantics() -> None:
    assert PureDPBudget(eps=1.0) == PureDPBudget(eps=1.0)
    assert ApproxDPBudget(eps=1.0, delta=1e-5) == ApproxDPBudget(eps=1.0, delta=1e-5)
    assert ZCDPBudget(rho=1.0) == ZCDPBudget(rho=1.0)
    assert GDPBudget(mu=1.0) == GDPBudget(mu=1.0)
    assert PureDPBudget(eps=1.0) != ZCDPBudget(rho=1.0)


# Two hosts' floating-point libraries returned the first two deltas for one
# rule; the third is the correctly rounded double of the rule's true value
# 0.0017735147777247594..., which neither host returned.
_RESOLVED_DELTAS = (
    0.0017735147777247584,
    0.0017735147777247586,
    0.0017735147777247595,
)


def test_one_parameter_resolved_on_any_host_names_one_budget() -> None:
    offers = [ApproxDPBudget(eps=1.0, delta=delta) for delta in _RESOLVED_DELTAS]

    assert len(set(offers)) == 1
    assert len({hash(offer) for offer in offers}) == 1
    for offer in offers:
        assert offer.identity_record().config == {
            "eps": 1.0,
            "delta": 0.00177351477772,
        }


def test_an_offer_stores_the_canonical_parameter_not_the_raw_construction_value() -> (
    None
):
    # Every parameter is canonicalized at construction, so the field, the
    # dumped config, and a round trip all agree on the canonical value
    # regardless of which correctly-rounded double named it.
    offered = ApproxDPBudget(eps=1.0, delta=_RESOLVED_DELTAS[1])
    canonical_delta = 0.00177351477772

    assert offered.delta == canonical_delta
    assert offered.dump_config().config == {"eps": 1.0, "delta": canonical_delta}
    restored = ApproxDPBudget.load_record(offered.dump_config())
    assert restored.delta == canonical_delta


def test_parameters_apart_above_the_identity_precision_stay_distinct() -> None:
    assert PureDPBudget(eps=1.0) != PureDPBudget(eps=1.0 + 1e-6)
    assert ApproxDPBudget(eps=1.0, delta=1e-5) != ApproxDPBudget(
        eps=1.0, delta=1.00000000001e-5
    )
    assert PureDPBudget(eps=math.inf) != PureDPBudget(eps=1e12)


_NONPOSITIVE = (0.0, -1.0, -math.inf, math.nan)
_OUTSIDE_UNIT_INTERVAL = (0.0, 1.0, -1.0, 2.0, math.inf, -math.inf, math.nan)
_OUTSIDE_HALF_OPEN_UNIT_INTERVAL = (0.0, -1.0, 2.0, math.inf, -math.inf, math.nan)


@pytest.mark.parametrize(
    ("build", "field", "invalid"),
    [
        *[(lambda v: PureDPBudget(eps=v), "eps", v) for v in _NONPOSITIVE],
        *[
            (lambda v: ApproxDPBudget(eps=v, delta=1e-5), "eps", v)
            for v in _NONPOSITIVE
        ],
        *[
            (lambda v: ApproxDPBudget(eps=1.0, delta=v), "delta", v)
            for v in _OUTSIDE_HALF_OPEN_UNIT_INTERVAL
        ],
        *[(lambda v: ZCDPBudget(rho=v), "rho", v) for v in _NONPOSITIVE],
        *[(lambda v: GDPBudget(mu=v), "mu", v) for v in _NONPOSITIVE],
    ],
)
def test_budget_refuses_a_parameter_outside_its_domain(
    build: Callable[[float], DPBudget], field: str, invalid: float
) -> None:
    with pytest.raises(ValueError, match=field):
        build(invalid)


@pytest.mark.parametrize(
    ("budget", "field"),
    [
        (PureDPBudget(eps=math.inf), "eps"),
        (ApproxDPBudget(eps=math.inf, delta=1e-5), "eps"),
        (ZCDPBudget(rho=math.inf), "rho"),
        (GDPBudget(mu=math.inf), "mu"),
    ],
)
def test_unconstrained_budgets_round_trip_through_canonical_json_transport(
    budget: DPBudget, field: str
) -> None:
    record = budget.model_dump(mode="json")
    persisted = decode_json_document(canonical_json_bytes(record))

    assert record[field] == math.inf
    assert persisted == {**record, field: math.inf}
    assert canonical_json_bytes(record).count(b'"+inf_json"') == 1
    assert type(budget).model_validate(persisted) == budget


def test_no_privacy_has_one_value() -> None:
    assert NoPrivacy() == NO_PRIVACY


def test_custom_budget_family_owns_its_allocation_and_constraint_behavior() -> None:
    budget = ExternalBudget(credits=8.0)

    parts: tuple[ExternalBudget, ...] = budget.split((1.0, 3.0))

    assert parts == (ExternalBudget(credits=2.0), ExternalBudget(credits=6.0))
    assert not budget.is_unconstrained
    assert ExternalBudget(credits=math.inf).is_unconstrained
