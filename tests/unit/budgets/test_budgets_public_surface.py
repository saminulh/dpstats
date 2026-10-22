from __future__ import annotations

import pytest

from dpstats.transport.canonical import canonical_json_bytes
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (
            NO_PRIVACY,
            b'{"config":{},"path":"dpstats.budgets.families:NoPrivacy"}',
        ),
        (
            PureDPBudget(eps=1.0),
            b'{"config":{"eps":1.0},"path":"dpstats.budgets.families:PureDPBudget"}',
        ),
        (
            ApproxDPBudget(eps=1.0, delta=1e-6),
            b'{"config":{"delta":1e-06,"eps":1.0},"path":"dpstats.budgets.families:ApproxDPBudget"}',
        ),
        (
            ZCDPBudget(rho=1.0),
            b'{"config":{"rho":1.0},"path":"dpstats.budgets.families:ZCDPBudget"}',
        ),
        (
            GDPBudget(mu=1.0),
            b'{"config":{"mu":1.0},"path":"dpstats.budgets.families:GDPBudget"}',
        ),
    ],
)
def test_budget_family_configured_identities_are_exact_and_round_trip(
    budget: DPBudget,
    expected: bytes,
) -> None:
    assert canonical_json_bytes(budget.dump_config().model_dump()) == expected
    assert DPBudget.load_record(budget.dump_config()) == budget
