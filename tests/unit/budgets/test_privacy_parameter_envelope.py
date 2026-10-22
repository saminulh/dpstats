"""Regression envelope for privacy-parameter numerical stability.

The library deliberately has no runtime upper or lower bounds on privacy
parameters.  This test-only envelope is therefore the supported numerical
monitoring range, not a validator contract:

* eps is [1e-5, 100].  Below about 1e-9 the Hourglass staircase parameter loses
  precision; it exceeds one near 1e-16 and becomes zero at or below 1e-17.
  ExactDistancePTR's truncated-Laplace ratio becomes infinite around
  [709, 709.78], silently degenerating to an untruncated draw, and overflows at
  about 709.79;
* delta is [1e-12, 0.1]; TukeyEM's PTR threshold changes sign at delta >= 0.5,
  silently permitting distance zero, while the zCDP conversion reaches a bare
  domain error within roughly three machine epsilons of one; and
* rho is [1e-10, 1e4]; the zCDP Laplace calibration silently under-noises below
  about 1e-33, reaching roughly 78-fold under-noising by rho=1e-36.  The mu
  envelope is [1e-5, 100].

These known corners are intentionally unsupported. They are not production
parameter limits and must not motivate numerical rescue code or source-level
corner guards.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from dpstats.budgets import (
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.budgets.accounting import (
    calibrate_gaussian_scale,
    calibrate_laplace_scale,
    compute_delta_from_gaussian_mechanism,
    compute_epsilon_for_composed_gaussian_mixing,
    compute_epsilon_from_gaussian_mechanism,
    compute_noise_multiplier_for_composed_gaussian,
    compute_noise_multiplier_for_subsampled_gaussian,
    spend_as_gdp,
    spend_as_pure,
    spend_as_zcdp,
)


EPSILONS = (1e-5, 1e-3, 1e-1, 10.0, 100.0)
DELTAS = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)
RHOS = (1e-10, 1e-6, 1e-2, 1e2, 1e4)
MUS = (1e-5, 1e-3, 1e-1, 10.0, 100.0)


def _finite_or_parameter_error(
    operation: Callable[[], float], parameter_names: tuple[str, ...]
) -> float | None:
    """Return a finite value, or accept an explicit parameter refusal."""
    try:
        value = operation()
    except ValueError as error:
        message = str(error).lower()
        assert any(name in message for name in parameter_names), message
        return None
    assert math.isfinite(value)
    return value


class TestCoreAccountingEnvelope:
    """Closed-form accounting and calibration routes stay numerically in-domain."""

    def test_approximate_budget_spends_stay_finite_across_the_envelope(self) -> None:
        admitted_zcdp: list[float] = []
        admitted_gdp: list[float] = []
        for epsilon in EPSILONS:
            for delta in DELTAS:
                budget = ApproxDPBudget(eps=epsilon, delta=delta)
                rho = _finite_or_parameter_error(
                    lambda: spend_as_zcdp(budget).rho, ("eps", "delta")
                )
                mu = _finite_or_parameter_error(
                    lambda: spend_as_gdp(budget).mu, ("eps", "delta")
                )
                if rho is not None:
                    assert rho > 0.0
                    admitted_zcdp.append(rho)
                if mu is not None:
                    assert mu > 0.0
                    admitted_gdp.append(mu)
        assert admitted_zcdp
        assert admitted_gdp

    def test_calibration_scales_are_positive_and_finite(self) -> None:
        for epsilon in EPSILONS:
            for delta in DELTAS:
                budget = ApproxDPBudget(eps=epsilon, delta=delta)
                gaussian_scale = _finite_or_parameter_error(
                    lambda: calibrate_gaussian_scale(budget, 1.0), ("eps", "delta")
                )
                laplace_scale = _finite_or_parameter_error(
                    lambda: calibrate_laplace_scale(budget, 1.0), ("eps", "delta")
                )
                assert gaussian_scale is not None
                assert laplace_scale is not None
                assert gaussian_scale > 0.0
                assert laplace_scale > 0.0
                composed_noise = compute_noise_multiplier_for_composed_gaussian(
                    budget, 3
                )
                assert math.isfinite(composed_noise) and composed_noise > 0.0
        for epsilon in EPSILONS:
            pure_scale = _finite_or_parameter_error(
                lambda: calibrate_laplace_scale(PureDPBudget(eps=epsilon), 1.0),
                ("eps",),
            )
            assert pure_scale is not None
            assert pure_scale > 0.0
        for rho in RHOS:
            gaussian_scale = _finite_or_parameter_error(
                lambda: calibrate_gaussian_scale(ZCDPBudget(rho=rho), 1.0),
                ("rho",),
            )
            laplace_scale = _finite_or_parameter_error(
                lambda: calibrate_laplace_scale(ZCDPBudget(rho=rho), 1.0),
                ("rho",),
            )
            assert gaussian_scale is not None
            assert laplace_scale is not None
            assert gaussian_scale > 0.0
            assert laplace_scale > 0.0
            composed_noise = compute_noise_multiplier_for_composed_gaussian(
                ZCDPBudget(rho=rho), 3
            )
            assert math.isfinite(composed_noise) and composed_noise > 0.0
        for mu in MUS:
            gaussian_scale = _finite_or_parameter_error(
                lambda: calibrate_gaussian_scale(GDPBudget(mu=mu), 1.0), ("mu",)
            )
            laplace_scale = _finite_or_parameter_error(
                lambda: calibrate_laplace_scale(GDPBudget(mu=mu), 1.0), ("mu",)
            )
            assert gaussian_scale is not None
            assert laplace_scale is not None
            assert gaussian_scale > 0.0
            assert laplace_scale > 0.0
            composed_noise = compute_noise_multiplier_for_composed_gaussian(
                GDPBudget(mu=mu), 3
            )
            assert math.isfinite(composed_noise) and composed_noise > 0.0

    def test_spend_as_pure_stays_finite_for_native_budget_families(self) -> None:
        for epsilon in EPSILONS:
            pure = spend_as_pure(PureDPBudget(eps=epsilon)).eps
            assert math.isfinite(pure)
            assert pure > 0.0
            for delta in DELTAS:
                approximate = spend_as_pure(
                    ApproxDPBudget(eps=epsilon, delta=delta)
                ).eps
                assert math.isfinite(approximate)
                assert approximate > 0.0
        for rho in RHOS:
            epsilon = _finite_or_parameter_error(
                lambda: spend_as_pure(ZCDPBudget(rho=rho)).eps, ("rho",)
            )
            assert epsilon is not None
            assert epsilon > 0.0
        for mu in MUS:
            epsilon = _finite_or_parameter_error(
                lambda: spend_as_pure(GDPBudget(mu=mu)).eps, ("mu",)
            )
            assert epsilon is not None
            assert epsilon > 0.0

    def test_gaussian_profiles_stay_finite_and_monotone_in_mu(self) -> None:
        for delta in DELTAS:
            gaussian_epsilons: list[float] = []
            for mu in MUS:
                scale = 1.0 / mu
                gaussian_epsilon = compute_epsilon_from_gaussian_mechanism(
                    scale, 1.0, delta
                )
                profile_delta = compute_delta_from_gaussian_mechanism(
                    scale, 1.0, gaussian_epsilon
                )
                assert math.isfinite(gaussian_epsilon) and gaussian_epsilon >= 0.0
                assert math.isfinite(profile_delta) and 0.0 <= profile_delta <= 1.0
                gaussian_epsilons.append(gaussian_epsilon)
            assert all(
                right >= left
                for left, right in zip(gaussian_epsilons, gaussian_epsilons[1:])
            )

    def test_gaussian_mixing_accounting_stays_finite(self) -> None:
        for delta in DELTAS:
            epsilon = compute_epsilon_for_composed_gaussian_mixing(
                gamma=3.0,
                sketch_size=5,
                num_steps=3,
                delta=delta,
            )
            assert math.isfinite(epsilon) and epsilon >= 0.0

    def test_approximate_gaussian_calibrations_relax_with_epsilon(self) -> None:
        for delta in DELTAS:
            composed_noise = [
                compute_noise_multiplier_for_composed_gaussian(
                    ApproxDPBudget(eps=epsilon, delta=delta), 3
                )
                for epsilon in EPSILONS
            ]
            assert all(
                right <= left for left, right in zip(composed_noise, composed_noise[1:])
            )

    @pytest.mark.parametrize(
        ("epsilon", "delta"),
        [
            # The small-epsilon, small-delta corner costs over ten seconds per
            # calibration; it stays in the serial slow gate.
            (
                pytest.param(epsilon, delta, marks=pytest.mark.slow)
                if epsilon <= 1e-3 and delta <= 1e-6
                else (epsilon, delta)
            )
            for epsilon in EPSILONS
            for delta in DELTAS
        ],
    )
    def test_subsampled_gaussian_calibration_stays_finite(
        self, epsilon: float, delta: float
    ) -> None:
        noise_multiplier = compute_noise_multiplier_for_subsampled_gaussian(
            ApproxDPBudget(  # pyright: ignore[reportArgumentType]  # frozen budgets are hashable at runtime
                eps=epsilon, delta=delta
            ),
            0.5,
            3,
        )
        assert math.isfinite(noise_multiplier)
        assert noise_multiplier > 0.0
