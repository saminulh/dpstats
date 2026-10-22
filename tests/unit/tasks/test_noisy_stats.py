"""Tests for the native NoisyStats simple-linear-regression algorithm."""

import numpy as np
import pytest

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    Domain,
    EuclideanSpace,
    LpBallDomain,
)
from dpstats.exceptions import NoRelease
from dpstats.tasks.simple_linear_regression.algorithms.noisy_stats import (
    NoisyStats,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from tests.fixtures.randomness_fake import (
    FakeRandomness,
    draws_in_order,
    zero_draw,
)
from tests.unit.tasks.task_builders import box, slr_metadata


UNIT_DOMAIN = box((0.0, 1.0))


def _unit_metadata(n_samples: int) -> SimpleLinearRegressionMetadata:
    return slr_metadata(n_samples, UNIT_DOMAIN, UNIT_DOMAIN)


def test_noisy_stats_matches_noise_free_formula(
    fake_randomness: FakeRandomness,
) -> None:
    fake_randomness.script("laplace", zero_draw)
    x = np.linspace(0.0, 1.0, 20)
    y = 0.4 * x + 0.2
    output = NoisyStats().run(
        _unit_metadata(x.size),
        SimpleLinearRegressionInput(x=x, y=y),
        budget=PureDPBudget(eps=1.0),
        seed=0,
    )
    assert output.slope == pytest.approx(0.4)
    assert output.intercept == pytest.approx(0.2)


def test_noisy_stats_public_domain_rescaling(
    fake_randomness: FakeRandomness,
) -> None:
    fake_randomness.script("laplace", zero_draw)
    x = np.linspace(10.0, 20.0, 20)
    y = -3.0 * x + 100.0
    metadata = slr_metadata(x.size, box((10.0, 20.0)), box((40.0, 70.0)))
    output = NoisyStats().run(
        metadata,
        SimpleLinearRegressionInput(x=x, y=y),
        budget=PureDPBudget(eps=1.0),
        seed=0,
    )
    assert output.slope == pytest.approx(-3.0)
    assert output.intercept == pytest.approx(100.0)


def test_noisy_stats_reads_the_interval_any_scalar_domain_encloses(
    fake_randomness: FakeRandomness,
) -> None:
    # The transport reads the domain's enclosing interval, not box endpoints,
    # so a scalar L2 ball of radius 5 about 15 behaves as the box [10, 20].
    fake_randomness.script("laplace", zero_draw)
    x = np.linspace(10.0, 20.0, 20)
    y = -3.0 * x + 100.0
    input = SimpleLinearRegressionInput(x=x, y=y)
    ball = NoisyStats().run(
        slr_metadata(
            x.size,
            LpBallDomain(p=2, radius=5.0, center=np.array([15.0])),
            LpBallDomain(p=2, radius=15.0, center=np.array([55.0])),
        ),
        input,
        budget=PureDPBudget(eps=1.0),
        seed=0,
    )
    boxed = NoisyStats().run(
        slr_metadata(x.size, box((10.0, 20.0)), box((40.0, 70.0))),
        input,
        budget=PureDPBudget(eps=1.0),
        seed=0,
    )
    assert ball.slope == pytest.approx(boxed.slope)
    assert ball.intercept == pytest.approx(boxed.intercept)


@pytest.mark.parametrize(
    "x_domain,message",
    [
        (EuclideanSpace(dim=1), "enclosing"),
        (box((0.0, np.inf)), "enclosing"),
        (box((3.0, 3.0)), "width"),
        (box((1e16, 1e16 + 2.0)), "width"),
    ],
    ids=[
        "unbounded",
        "non_finite",
        "degenerate",
        "floating_point_collapsed",
    ],
)
def test_noisy_stats_refuses_an_interval_public_metadata_cannot_supply(
    x_domain: Domain, message: str
) -> None:
    # Every refusal is decidable from public metadata, so it is an ordinary
    # ValueError rather than an accounted bottom, and it precedes any spend.
    x = np.linspace(0.0, 1.0, 8)
    metadata = slr_metadata(x.size, x_domain, UNIT_DOMAIN)
    with pytest.raises(InvalidRegime, match=message):
        NoisyStats().run(
            metadata,
            SimpleLinearRegressionInput(x=x, y=0.5 * x + 0.25),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )


def test_noisy_stats_refuses_before_spending_budget(
    accounting_spy: AccountingSpy,
) -> None:
    x = np.linspace(0.0, 1.0, 8)
    metadata = slr_metadata(x.size)
    with pytest.raises(InvalidRegime):
        NoisyStats().run(
            metadata,
            SimpleLinearRegressionInput(x=x, y=0.5 * x + 0.25),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )
    assert accounting_spy.calls == ()


def test_noisy_stats_nonpositive_noisy_variance_is_accounted_bottom(
    fake_randomness: FakeRandomness,
) -> None:
    # The second release is the noisy variance; a large negative draw drives it
    # below zero, which the mechanism must report as an accounted bottom.
    fake_randomness.script("laplace", draws_in_order((0.0, -100.0)))
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(NoRelease, match="noisy variance"):
        NoisyStats().run(
            _unit_metadata(x.size),
            SimpleLinearRegressionInput(x=x, y=x),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )


@pytest.mark.parametrize(
    "budget",
    [
        PureDPBudget(eps=6.0),
        ApproxDPBudget(eps=6.0, delta=1e-6),
        ZCDPBudget(rho=18.0),
    ],
)
def test_noisy_stats_spends_three_equal_pure_releases(
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
    budget: DPBudget,
) -> None:
    fake_randomness.script("laplace", zero_draw)
    x = np.linspace(0.0, 1.0, 10)
    y = 0.5 * x + 0.25

    NoisyStats().run(
        _unit_metadata(x.size),
        SimpleLinearRegressionInput(x=x, y=y),
        budget=budget,
        seed=0,
    )

    assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)
    statistic_sensitivity = 1 - 1 / x.size
    intercept_sensitivity = (1 + 0.5) / x.size
    scales = fake_randomness.parameters_of("laplace", "scale")
    assert scales[0] == pytest.approx(scales[1])
    assert scales[0] / scales[2] == pytest.approx(
        statistic_sensitivity / intercept_sensitivity
    )
