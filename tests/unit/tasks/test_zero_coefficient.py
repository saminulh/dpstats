"""Tests for the constant-zero ZeroCoefficient baseline."""

from __future__ import annotations

from typing import cast

import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)
from dpstats.tasks.single_coefficient_estimation.algorithms.zero_coefficient import (
    ZeroCoefficient,
)
from tests.unit.tasks.task_builders import Unreadable


@pytest.mark.parametrize(
    "budget",
    [PureDPBudget(eps=1.0), ApproxDPBudget(eps=1.0, delta=1e-5), NO_PRIVACY],
)
def test_releases_zero_under_every_budget_family_without_reading_data(
    budget: DPBudget,
):
    metadata = cast(SingleCoefficientEstimationMetadata, Unreadable("Metadata"))
    input = cast(SingleCoefficientEstimationInput, Unreadable("Input"))
    output = ZeroCoefficient().run(metadata, input, budget, seed=0)
    assert output.coefficient == 0.0
