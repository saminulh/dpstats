"""Family-by-route lattice for typed privacy-budget accounting.

Each accounting route either realizes a guarantee that implies the offered
family, or refuses the offer. The approximate-DP inversions, whose quality is
numerical, are checked in ``test_accounting_numerics.py``.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import norm

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.budgets.accounting import (
    calibrate_gaussian_scale,
    calibrate_laplace_scale,
    compute_noise_multiplier_for_composed_gaussian,
    compute_noise_multiplier_for_subsampled_gaussian,
    spend_as_approx_dp,
    spend_as_gdp,
    spend_as_pure,
    spend_as_zcdp,
    spend_as_zcdp_via_rdp,
)
from tests.registry_components import ExternalBudget


@pytest.mark.parametrize(
    ("spend", "budget"),
    [
        (spend_as_pure, PureDPBudget(eps=1.25)),
        (spend_as_zcdp, ZCDPBudget(rho=0.72)),
        (spend_as_gdp, GDPBudget(mu=0.72)),
        (spend_as_approx_dp, ApproxDPBudget(eps=1.25, delta=1e-6)),
    ],
)
def test_native_family_passes_through(
    spend: Callable[[DPBudget], DPBudget], budget: DPBudget
) -> None:
    assert spend(budget) is budget


def test_approx_dp_spends_its_epsilon_as_pure_and_leaves_delta_unused() -> None:
    assert spend_as_pure(ApproxDPBudget(eps=1.25, delta=1e-6)) == PureDPBudget(eps=1.25)


@pytest.mark.parametrize("rho", [0.01, 1.0, 10.0])
def test_zcdp_pure_spend_charges_the_randomized_response_kl_divergence(
    rho: float,
) -> None:
    # An eps-DP mechanism is rho-zCDP for rho = eps * tanh(eps / 2), derived
    # from Bun & Steinke 2016, Lemma 8.6. That rho is the KL divergence of
    # eps-randomized response, the extremal eps-DP pair, whose ratio
    # D_alpha / alpha is largest as the Renyi order alpha approaches one.
    eps = spend_as_pure(ZCDPBudget(rho=rho)).eps
    p = math.exp(eps) / (1.0 + math.exp(eps))
    kl_divergence = p * math.log(p / (1 - p)) + (1 - p) * math.log((1 - p) / p)
    assert kl_divergence == pytest.approx(rho)


def test_gdp_pure_spend_meets_the_gaussian_tradeoff_fixed_point() -> None:
    budget = GDPBudget(mu=1.25)
    spent = spend_as_pure(budget)
    gaussian_fixed_point = 0.5 * math.erfc(budget.mu / (2.0 * math.sqrt(2.0)))
    pure_fixed_point = 1.0 / (1.0 + math.exp(spent.eps))
    assert pure_fixed_point == pytest.approx(gaussian_fixed_point)


def test_zcdp_is_spent_as_gdp_through_the_gdp_to_zcdp_implication() -> None:
    budget = ZCDPBudget(rho=0.72)
    assert spend_as_gdp(budget).mu ** 2 / 2.0 == pytest.approx(budget.rho)


@pytest.mark.parametrize(
    ("route", "budget", "family"),
    [
        (spend_as_zcdp, PureDPBudget(eps=1.25), "PureDPBudget"),
        (spend_as_zcdp, GDPBudget(mu=1.25), "GDPBudget"),
        (spend_as_gdp, PureDPBudget(eps=1.25), "PureDPBudget"),
        (
            lambda budget: calibrate_gaussian_scale(budget, 2.5),
            PureDPBudget(eps=1.25),
            "PureDPBudget",
        ),
        (
            lambda budget: compute_noise_multiplier_for_composed_gaussian(budget, 10),
            PureDPBudget(eps=1.0),
            "PureDPBudget",
        ),
        (
            lambda budget: compute_noise_multiplier_for_subsampled_gaussian(
                budget, 0.5, 10
            ),
            PureDPBudget(eps=1.0),
            "PureDPBudget",
        ),
        (spend_as_approx_dp, PureDPBudget(eps=1.25), "PureDPBudget"),
        (spend_as_approx_dp, ZCDPBudget(rho=0.72), "ZCDPBudget"),
        (spend_as_approx_dp, GDPBudget(mu=0.72), "GDPBudget"),
    ],
)
def test_route_refuses_an_offer_no_realized_guarantee_implies(
    route: Callable[[DPBudget], object], budget: DPBudget, family: str
) -> None:
    with pytest.raises(InvalidRegime, match=family):
        route(budget)


@pytest.mark.parametrize(
    "route",
    [
        spend_as_pure,
        spend_as_zcdp,
        spend_as_gdp,
        lambda budget: calibrate_gaussian_scale(budget, 2.5),
        lambda budget: calibrate_laplace_scale(budget, 2.5),
        lambda budget: compute_noise_multiplier_for_composed_gaussian(budget, 10),
        lambda budget: compute_noise_multiplier_for_subsampled_gaussian(
            budget, 0.5, 10
        ),
        lambda budget: compute_noise_multiplier_for_subsampled_gaussian(
            budget, 1.0, 10
        ),
        spend_as_approx_dp,
    ],
    ids=[
        "spend_as_pure",
        "spend_as_zcdp",
        "spend_as_gdp",
        "calibrate_gaussian_scale",
        "calibrate_laplace_scale",
        "compute_noise_multiplier_for_composed_gaussian",
        "compute_noise_multiplier_for_subsampled_gaussian",
        "noise_multiplier_for_unsampled_gaussian",
        "spend_as_approx_dp",
    ],
)
def test_route_refuses_a_constrained_offer_of_an_unknown_family(
    route: Callable[[DPBudget], object],
) -> None:
    with pytest.raises(InvalidRegime, match="ExternalBudget"):
        route(ExternalBudget(credits=1.0))


@pytest.mark.parametrize(
    ("budget", "expected_mu"),
    [
        (ZCDPBudget(rho=0.72), math.sqrt(2 * 0.72)),
        (GDPBudget(mu=0.72), 0.72),
    ],
)
def test_gaussian_scale_realizes_the_family_defining_shift(
    budget: DPBudget, expected_mu: float
) -> None:
    sensitivity = 2.5
    sigma = calibrate_gaussian_scale(budget, sensitivity)
    assert sensitivity / sigma == pytest.approx(expected_mu)


@pytest.mark.parametrize(
    "calibrate", [calibrate_gaussian_scale, calibrate_laplace_scale]
)
@pytest.mark.parametrize("sensitivity", [-1.0, math.inf, math.nan])
def test_calibration_refuses_invalid_sensitivity(
    calibrate: Callable[[DPBudget, float], float], sensitivity: float
) -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        calibrate(ApproxDPBudget(eps=1.25, delta=1e-6), sensitivity)


@pytest.mark.parametrize(
    "budget",
    [
        PureDPBudget(eps=1.0),
        ApproxDPBudget(eps=1.0, delta=1e-6),
        ZCDPBudget(rho=1.0),
        GDPBudget(mu=1.0),
        NO_PRIVACY,
        ExternalBudget(credits=1.0),
    ],
)
def test_zero_sensitivity_gaussian_needs_no_noise(budget: DPBudget) -> None:
    assert calibrate_gaussian_scale(budget, 0.0) == 0.0


def test_laplace_calibration_refuses_zero_sensitivity() -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        calibrate_laplace_scale(ApproxDPBudget(eps=1.25, delta=1e-6), 0.0)


@pytest.mark.parametrize(
    "budget,expected",
    [
        (PureDPBudget(eps=3.0), 1.0),
        (ApproxDPBudget(eps=2.0, delta=1 - math.exp(-0.5)), 1.0),
        (ZCDPBudget(rho=2.0 + math.exp(-3.0)), 1.0),
        (NO_PRIVACY, 0.0),
    ],
)
def test_laplace_scale_uses_family_specific_profile(
    budget: DPBudget, expected: float
) -> None:
    assert calibrate_laplace_scale(budget, 3.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "budget",
    [ApproxDPBudget(eps=1.0, delta=0.1), ZCDPBudget(rho=2.0), GDPBudget(mu=1.0)],
)
def test_non_pure_laplace_routes_improve_on_generic_pure_spend(
    budget: DPBudget,
) -> None:
    sensitivity = 3.0
    family_specific = calibrate_laplace_scale(budget, sensitivity)
    generic_pure = sensitivity / spend_as_pure(budget).eps
    assert family_specific < generic_pure


def test_gdp_laplace_profile_dominates_the_offered_gaussian_tradeoff() -> None:
    budget = GDPBudget(mu=1.0)
    sensitivity = 3.0
    scale = calibrate_laplace_scale(budget, sensitivity)
    theta = sensitivity / scale

    alphas = np.linspace(0.001, 0.999, 999)
    transition = math.exp(-theta) / 2.0
    laplace_tradeoff = np.where(
        alphas < transition,
        1.0 - math.exp(theta) * alphas,
        np.where(
            alphas <= 0.5,
            math.exp(-theta) / (4.0 * alphas),
            math.exp(-theta) * (1.0 - alphas),
        ),
    )
    gaussian_tradeoff = norm.cdf(norm.ppf(1.0 - alphas) - budget.mu)

    assert all(
        laplace > gaussian or laplace == pytest.approx(gaussian)
        for laplace, gaussian in zip(laplace_tradeoff, gaussian_tradeoff)
    )
    fixed_point = norm.cdf(-budget.mu / 2.0)
    fixed_point_value = math.exp(-theta) / (4.0 * fixed_point)
    assert fixed_point_value == pytest.approx(fixed_point)


@pytest.mark.parametrize("num_steps", [1, 4, 20, 100])
def test_composed_gaussian_realizes_the_offered_family_by_composition(
    num_steps: int,
) -> None:
    # T Gaussian steps at noise multiplier s compose to T / (2 s^2)-zCDP and
    # to sqrt(T) / s-GDP.
    zcdp = ZCDPBudget(rho=0.7)
    gdp = GDPBudget(mu=0.5)
    zcdp_multiplier = compute_noise_multiplier_for_composed_gaussian(zcdp, num_steps)
    gdp_multiplier = compute_noise_multiplier_for_composed_gaussian(gdp, num_steps)
    assert num_steps / (2.0 * zcdp_multiplier**2) == pytest.approx(zcdp.rho)
    assert math.sqrt(num_steps) / gdp_multiplier == pytest.approx(gdp.mu)


@pytest.mark.parametrize("budget", [ZCDPBudget(rho=0.7), GDPBudget(mu=0.5)])
@pytest.mark.parametrize("sampling_rate", [0.01, 0.25, 1.0])
def test_subsampled_gaussian_uses_unamplified_composition_for_zcdp_and_gdp(
    budget: DPBudget, sampling_rate: float
) -> None:
    num_steps = 20
    assert compute_noise_multiplier_for_subsampled_gaussian(
        budget, sampling_rate, num_steps
    ) == pytest.approx(
        compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
    )


@pytest.mark.parametrize("num_steps", [0, -1])
def test_gaussian_composition_refuses_a_nonpositive_step_count(
    num_steps: int,
) -> None:
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    with pytest.raises(ValueError, match="num_steps"):
        compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
    with pytest.raises(ValueError, match="num_steps"):
        compute_noise_multiplier_for_subsampled_gaussian(budget, 1.0, num_steps)


@pytest.mark.parametrize("sampling_rate", [0.0, -0.5, 1.5, math.inf, math.nan])
def test_subsampled_gaussian_refuses_a_rate_outside_the_unit_interval(
    sampling_rate: float,
) -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        compute_noise_multiplier_for_subsampled_gaussian(
            ApproxDPBudget(eps=1.0, delta=1e-5), sampling_rate, 10
        )


class TestUnconstrainedLimits:
    """An unconstrained offer spends as an infinite parameter or zero noise."""

    def test_spends_are_infinite(self) -> None:
        assert spend_as_pure(NO_PRIVACY) == PureDPBudget(eps=math.inf)
        assert spend_as_zcdp(NO_PRIVACY) == ZCDPBudget(rho=math.inf)
        assert spend_as_gdp(NO_PRIVACY) == GDPBudget(mu=math.inf)

    def test_gaussian_calibrations_return_zero_noise(self) -> None:
        assert calibrate_gaussian_scale(NO_PRIVACY, sensitivity=2.5) == 0.0
        assert (
            compute_noise_multiplier_for_composed_gaussian(NO_PRIVACY, num_steps=10)
            == 0.0
        )
        assert (
            compute_noise_multiplier_for_subsampled_gaussian(
                NO_PRIVACY, sampling_rate=0.01, num_steps=10
            )
            == 0.0
        )

    def test_infinite_family_budget_uses_the_same_limit(self) -> None:
        assert spend_as_zcdp(PureDPBudget(eps=math.inf)) == ZCDPBudget(rho=math.inf)

    def test_zcdp_via_rdp_spends_an_unconstrained_offer_as_infinite_rho(self) -> None:
        # log1p(-(delta*delta)) is log1p(-1) at delta=1, a math domain error
        # without the same leading is_unconstrained check the other
        # spend_as_* functions use.
        assert spend_as_zcdp_via_rdp(ApproxDPBudget(eps=1.0, delta=1.0)) == ZCDPBudget(
            rho=math.inf
        )
        assert spend_as_zcdp_via_rdp(
            ApproxDPBudget(eps=math.inf, delta=1e-5)
        ) == ZCDPBudget(rho=math.inf)


class TestSpendAsApproxDp:
    """Every unconstrained offer, of any family, spends as (eps=inf, delta=1)."""

    @pytest.mark.parametrize(
        "budget",
        [
            NO_PRIVACY,
            PureDPBudget(eps=math.inf),
            ZCDPBudget(rho=math.inf),
            GDPBudget(mu=math.inf),
            ApproxDPBudget(eps=math.inf, delta=1e-5),
            ApproxDPBudget(eps=1.0, delta=1.0),
        ],
    )
    def test_every_unconstrained_family_spends_as_infinite_eps_delta_one(
        self, budget: DPBudget
    ) -> None:
        assert spend_as_approx_dp(budget) == ApproxDPBudget(eps=math.inf, delta=1.0)
