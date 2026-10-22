"""Numerical quality of the approximate-DP accounting inversions.

Every calibrated spend is fed back through an independent oracle in
``tests/fixtures/accounting_oracles.py``: it must stay within the offered
approximate-DP budget (sound) and nearly exhaust it (tight). Published
constants and ``dp_accounting`` calibrations supply the comparison routes.
"""

import logging
import math
from collections.abc import Callable

import dp_accounting
import numpy as np
import pytest
from dp_accounting import dp_event, pld, rdp

from dpstats.budgets import ApproxDPBudget, ZCDPBudget
from dpstats.budgets.accounting import (
    calibrate_gaussian_scale,
    compute_delta_from_gaussian_mechanism,
    compute_epsilon_for_composed_gaussian_mixing,
    compute_epsilon_from_gaussian_mechanism,
    compute_noise_multiplier_for_composed_gaussian,
    compute_noise_multiplier_for_subsampled_gaussian,
    spend_as_gdp,
    spend_as_zcdp,
    spend_as_zcdp_via_rdp,
)
from dpstats.budgets.accounting._gaussian_mixing import (
    _gaussian_mixing_rdp_epsilon,  # pyright: ignore[reportPrivateUsage]
)
from tests.fixtures.accounting_oracles import (
    delta_from_gdp,
    epsilon_for_subsampled_gaussian,
    epsilon_from_zcdp,
)

# The zCDP oracle's finite Renyi-order grid can overstate a certified epsilon
# by at most this relative amount on the grids below.
ZCDP_ORACLE_SLACK = 1e-5


def _alabi_lemma_23_rho(epsilon: float, delta: float) -> float:
    """Invert the simpler Lemma 3.6 branch used by Alabi et al. Lemma 23."""
    if delta >= math.sqrt(math.pi * epsilon):
        return epsilon

    lower = delta * delta / math.pi
    upper = epsilon
    for _ in range(100):
        midpoint = (lower + upper) / 2
        log_term = 0.5 * (math.log(math.pi) + math.log(midpoint)) - math.log(delta)
        achieved = midpoint + math.sqrt(4 * midpoint * log_term)
        if achieved <= epsilon:
            lower = midpoint
        else:
            upper = midpoint
    return lower


# (eps, delta) grid for the inverse calibration round-trip, plus extreme
# deltas and a very loose offer.
EPS_DELTA_GRID = [
    (eps, delta) for eps in (0.1, 0.5, 1.0, 2.0, 5.0) for delta in (1e-3, 1e-5, 1e-9)
] + [(0.1, 1e-300), (1.0, 1e-300), (5.0, 1e-100), (100.0, 0.9)]


class TestGaussianProfile:
    """The public Gaussian profile against the Balle--Wang delta."""

    @pytest.mark.parametrize(
        ("scale", "sensitivity", "eps"),
        [(1.0, 1.0, 1.0), (2.0, 0.5, 0.1), (0.2, 1.0, 5.0), (10.0, 1.0, 0.0)],
    )
    def test_delta_matches_the_exact_gaussian_delta(
        self, scale: float, sensitivity: float, eps: float
    ) -> None:
        assert compute_delta_from_gaussian_mechanism(
            scale=scale, sensitivity=sensitivity, eps=eps
        ) == pytest.approx(delta_from_gdp(sensitivity / scale, eps), rel=1e-9)

    def test_zero_epsilon_delta_is_the_total_variation_distance(self) -> None:
        mu = 1.25
        assert compute_delta_from_gaussian_mechanism(
            scale=1.0, sensitivity=mu, eps=0.0
        ) == pytest.approx(math.erf(mu / (2.0 * math.sqrt(2.0))))

    @pytest.mark.parametrize(
        ("scale", "delta"), [(1.0, 1e-5), (0.3, 1e-9), (5.0, 0.01)]
    )
    def test_epsilon_is_the_exact_gaussian_epsilon(
        self, scale: float, delta: float
    ) -> None:
        eps = compute_epsilon_from_gaussian_mechanism(
            scale=scale, sensitivity=1.0, delta=delta
        )
        assert delta_from_gdp(1.0 / scale, eps) == pytest.approx(delta, rel=1e-9)

    @pytest.mark.parametrize(
        ("profile", "arguments", "parameter"),
        [
            (compute_delta_from_gaussian_mechanism, (0.0, 1.0, 1.0), "scale"),
            (compute_delta_from_gaussian_mechanism, (math.inf, 1.0, 1.0), "scale"),
            (compute_delta_from_gaussian_mechanism, (1.0, -1.0, 1.0), "sensitivity"),
            (compute_delta_from_gaussian_mechanism, (1.0, 1.0, -1.0), "eps"),
            (compute_delta_from_gaussian_mechanism, (1.0, 1.0, math.nan), "eps"),
            (compute_epsilon_from_gaussian_mechanism, (1.0, 1.0, 0.0), "delta"),
            (compute_epsilon_from_gaussian_mechanism, (1.0, 1.0, 1.0), "delta"),
            (compute_epsilon_from_gaussian_mechanism, (1.0, 1.0, math.nan), "delta"),
        ],
    )
    def test_profile_refuses_invalid_arguments(
        self,
        profile: Callable[[float, float, float], float],
        arguments: tuple[float, float, float],
        parameter: str,
    ) -> None:
        with pytest.raises(ValueError, match=parameter):
            profile(*arguments)


class TestSpendAsGdp:
    """Approximate DP inverted through the exact GDP duality."""

    @pytest.mark.parametrize(
        "eps,delta", [(0.1, 1e-3), (0.5, 1e-6), (1.0, 1e-9), (5.0, 1e-5)]
    )
    def test_spend_exhausts_the_offered_delta(self, eps: float, delta: float) -> None:
        spent = spend_as_gdp(ApproxDPBudget(eps=eps, delta=delta))
        assert delta_from_gdp(spent.mu, eps) == pytest.approx(delta, rel=1e-6)

    def test_inverse_grants_more_mu_for_looser_offer(self) -> None:
        by_epsilon = [
            spend_as_gdp(ApproxDPBudget(eps=eps, delta=1e-5)).mu
            for eps in (0.1, 0.5, 2.0)
        ]
        by_delta = [
            spend_as_gdp(ApproxDPBudget(eps=1.0, delta=delta)).mu
            for delta in (1e-9, 1e-6, 1e-3)
        ]
        assert all(right > left for left, right in zip(by_epsilon, by_epsilon[1:]))
        assert all(right > left for left, right in zip(by_delta, by_delta[1:]))


class TestGaussianMixingAccounting:
    def test_gaussian_mixing_rdp_matches_source_reference_point(self) -> None:
        # Lev et al. 2025, Lemma 1: phi(2; 1, 4) = log(3/4) - log(1/2)/2.
        assert _gaussian_mixing_rdp_epsilon(
            alpha=2.0,
            sketch_size=1,
            gamma=4.0,
        ) == pytest.approx(math.log(3 / 4) - math.log(1 / 2) / 2)

    def test_composition_charges_additional_sketches(self) -> None:
        one_sketch = compute_epsilon_for_composed_gaussian_mixing(
            gamma=4.0,
            sketch_size=30,
            num_steps=1,
            delta=1e-6,
        )
        three_sketches = compute_epsilon_for_composed_gaussian_mixing(
            gamma=4.0,
            sketch_size=30,
            num_steps=3,
            delta=1e-6,
        )

        assert three_sketches > one_sketch


class TestGaussianCalibration:
    """Check the typed calibrator against the exact Balle--Wang profile."""

    # Boundary points: high-eps first-violation points for the
    # Wang/AdaSSP constant ((11, 1e-5), (10, 0.01)), plus a low-eps point and
    # an extreme high-eps point.
    BOUNDARY_POINTS = [
        (1.0, 1e-5),
        (10.0, 0.01),
        (11.0, 1e-5),
        (50.0, 1e-5),
    ]

    @pytest.mark.parametrize("eps,delta", BOUNDARY_POINTS)
    def test_sigma_is_valid_and_minimal(self, eps: float, delta: float) -> None:
        """Returned sigma reaches the boundary; 0.999*sigma does not."""
        sens = 1.7  # arbitrary positive sensitivity
        sigma = calibrate_gaussian_scale(ApproxDPBudget(eps=eps, delta=delta), sens)
        assert delta_from_gdp(sens / sigma, eps) == pytest.approx(delta, rel=1e-6)
        assert delta_from_gdp(sens / (0.999 * sigma), eps) > delta

    def test_sensitivity_scaling(self) -> None:
        """sigma scales linearly in the sensitivity."""
        budget = ApproxDPBudget(eps=2.0, delta=1e-5)
        s1 = calibrate_gaussian_scale(budget, 1.0)
        s2 = calibrate_gaussian_scale(budget, 3.5)
        assert np.isclose(s2, 3.5 * s1, rtol=1e-6)

    def test_wang_adassp_constant_violates_exact_gaussian_bound(self) -> None:
        """The looser sigma = sqrt(ln(4/delta))/(eps/2) constant (per-release
        at the k=2 split, i.e. checked at (eps/2, delta/2)) fails the exact
        characterization at high-eps points."""
        for eps, delta in [(11.0, 1e-5), (10.0, 0.01)]:
            old_sigma = np.sqrt(np.log(4 / delta)) / (eps / 2)
            assert (
                delta_from_gdp(1.0 / old_sigma, eps / 2) > delta / 2
            ), f"expected old constant to violate at ({eps}, {delta})"

    def test_classical_constant_violates_at_high_eps(self) -> None:
        """The classical sqrt(2 ln(1.25/delta))/eps constant (Dwork & Roth,
        Theorem A.1, proved only for eps < 1) fails the exact characterization
        at high eps."""
        eps, delta = 10.0, 0.01
        old_sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
        assert delta_from_gdp(1.0 / old_sigma, eps) > delta

    def test_classical_constant_safe_at_low_eps_and_we_beat_it(self) -> None:
        """At eps < 1 the classical constant is valid, and the analytic sigma
        is no larger (strictly better utility)."""
        eps, delta = 0.5, 1e-5
        classical = np.sqrt(2 * np.log(1.25 / delta)) / eps
        assert delta_from_gdp(1.0 / classical, eps) <= delta
        assert (
            calibrate_gaussian_scale(ApproxDPBudget(eps=eps, delta=delta), 1.0)
            <= classical
        )


class TestSpendAsZcdp:
    """Soundness and tightness of the (eps, delta) -> rho calibration."""

    @pytest.mark.parametrize("spend", [spend_as_zcdp, spend_as_zcdp_via_rdp])
    @pytest.mark.parametrize("eps,delta", EPS_DELTA_GRID)
    def test_spend_stays_within_and_nearly_exhausts_the_offer(
        self,
        spend: Callable[[ApproxDPBudget], ZCDPBudget],
        eps: float,
        delta: float,
    ) -> None:
        rho = spend(ApproxDPBudget(eps=eps, delta=delta)).rho
        certified = epsilon_from_zcdp(rho, delta)
        assert certified <= eps * (1 + ZCDP_ORACLE_SLACK)
        assert certified == pytest.approx(eps, rel=1e-3)

    @pytest.mark.parametrize(
        "eps,delta",
        [
            (1.0, 1e-5),
            (1e-4, 2.1544346900318777e-4),
            (100.0, 0.9),
            (0.1, 1e-300),
        ],
    )
    def test_never_grants_less_than_the_alabi_route(
        self, eps: float, delta: float
    ) -> None:
        rho = spend_as_zcdp(ApproxDPBudget(eps=eps, delta=delta)).rho
        assert rho >= _alabi_lemma_23_rho(eps, delta)

    @pytest.mark.parametrize("delta", [1e-9, 1e-5, 1e-2, 0.5])
    def test_admitted_rho_increases_with_epsilon(self, delta: float) -> None:
        rhos = [
            spend_as_zcdp(ApproxDPBudget(eps=eps, delta=delta)).rho
            for eps in (1e-4, 0.1, 1.0, 10.0)
        ]
        assert all(right >= left for left, right in zip(rhos, rhos[1:]))

    @pytest.mark.parametrize("eps", [1e-4, 0.1, 1.0, 10.0])
    def test_admitted_rho_increases_with_delta(self, eps: float) -> None:
        rhos = [
            spend_as_zcdp(ApproxDPBudget(eps=eps, delta=delta)).rho
            for delta in (1e-12, 1e-8, 1e-4, 0.1)
        ]
        assert all(right >= left for left, right in zip(rhos, rhos[1:]))


class TestComposedGaussian:
    """Composed-Gaussian calibration against the exact composed GDP delta."""

    @pytest.mark.parametrize("num_steps", [1, 3, 20, 100])
    @pytest.mark.parametrize(
        ("eps", "delta"),
        [(0.01, 1e-5), (0.5, 1e-5), (3.0, 1e-5), (5.0, 1e-8), (100.0, 0.01)],
    )
    def test_calibration_exhausts_the_offered_delta(
        self, num_steps: int, eps: float, delta: float
    ) -> None:
        # T Gaussian steps at noise multiplier s compose to sqrt(T) / s-GDP.
        noise_multiplier = compute_noise_multiplier_for_composed_gaussian(
            ApproxDPBudget(eps=eps, delta=delta), num_steps
        )
        composed_mu = math.sqrt(num_steps) / noise_multiplier
        assert delta_from_gdp(composed_mu, eps) == pytest.approx(delta, rel=1e-6)

    def test_larger_target_eps_needs_less_noise(self) -> None:
        """A looser epsilon budget calibrates to a smaller noise multiplier."""
        nm_tight = compute_noise_multiplier_for_composed_gaussian(
            ApproxDPBudget(eps=0.5, delta=1e-5), 10
        )
        nm_loose = compute_noise_multiplier_for_composed_gaussian(
            ApproxDPBudget(eps=3.0, delta=1e-5), 10
        )
        assert nm_loose < nm_tight

    @pytest.mark.parametrize(
        "eps,delta",
        [(0.1, 1e-5), (1.0, 1e-5), (10.0, 1e-5), (1.0, 10 * 1000**-1.1)],
    )
    def test_route_ordering(self, eps: float, delta: float) -> None:
        """Compare all routes at one common final approximate-DP guarantee.

        The installed exact-GDP multiplier agrees with ``dp_accounting``'s
        numerical PLD calibration and is strictly below its RDP calibration,
        which is below the paper's standard and advanced composition bounds.
        """
        num_steps = math.ceil(2 * math.log(1000))

        # Chang et al. (arXiv:2603.07505v1), Eq. (6): standard composition.
        paper_standard = (
            num_steps * math.sqrt(2 * math.log(1.25 * num_steps / delta)) / eps
        )

        installed_exact_gdp = compute_noise_multiplier_for_composed_gaussian(
            ApproxDPBudget(eps=eps, delta=delta), num_steps
        )

        def make_rdp_accountant() -> rdp.RdpAccountant:
            return rdp.RdpAccountant()

        def make_pld_accountant() -> pld.PLDAccountant:
            return pld.PLDAccountant(value_discretization_interval=1e-3)

        def make_composed_gaussian(
            noise_multiplier: float,
        ) -> dp_event.DpEvent:
            return dp_event.SelfComposedDpEvent(
                dp_event.GaussianDpEvent(noise_multiplier), num_steps
            )

        previous_rdp = float(
            dp_accounting.calibrate_dp_mechanism(
                make_rdp_accountant,
                make_composed_gaussian,
                eps,
                delta,
            )
        )
        numerical_pld = float(
            dp_accounting.calibrate_dp_mechanism(
                make_pld_accountant,
                make_composed_gaussian,
                eps,
                delta,
                tol=1e-8,
            )
        )

        assert installed_exact_gdp == pytest.approx(numerical_pld, rel=1e-6)
        assert installed_exact_gdp < previous_rdp < paper_standard

        # The paper's advanced-composition alternative is available only under
        # its Equation (1) preconditions: eps <= 1 and delta <= 0.01.
        if eps <= 1 and delta <= 0.01:
            paper_advanced = (
                math.sqrt(
                    5
                    * num_steps
                    * math.log(2 / delta)
                    * math.log(5 * num_steps / (2 * delta))
                )
                / eps
            )
            assert previous_rdp < paper_advanced


class _RecordingHandler(logging.Handler):
    """Collects every record routed to the logger it is attached to."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestSubsampledAbslWarningSuppression:
    """The subsampled-Gaussian seam must not leak dp-accounting's RDP chatter.

    At ``sampling_rate < 1`` dp-accounting's Sampled-Gaussian
    RDP routine emits, per non-converging Renyi order, an ``absl`` WARNING that it
    "failed to converge ... Excluding this order from the epsilon computation".
    That is the dependency's own conservative behavior (dropping an order only
    RAISES epsilon) and floods the logs. The seam scopes the ``absl`` logger to
    ERROR for the duration of the accounting call, then restores it. These tests
    pin (a) no WARNING record escapes during the call and (b) the prior level is
    restored afterward.
    """

    def test_noise_multiplier_emits_no_warning(self) -> None:
        absl_logger = logging.getLogger("absl")
        handler = _RecordingHandler()
        handler.setLevel(logging.WARNING)
        absl_logger.addHandler(handler)
        # Ensure WARNING records would propagate to the handler if not suppressed.
        absl_logger.setLevel(logging.WARNING)
        try:
            # Small noise multiplier at q<1 over many steps triggers the
            # non-convergence path in dp-accounting if anything does.
            compute_noise_multiplier_for_subsampled_gaussian(
                ApproxDPBudget(eps=1.0, delta=1e-5), 0.5, 50
            )
            assert handler.records == []
            # Level restored to the WARNING we set going in.
            assert absl_logger.level == logging.WARNING
        finally:
            absl_logger.removeHandler(handler)
            absl_logger.setLevel(logging.NOTSET)

    def test_level_restored_even_on_prior_debug(self) -> None:
        """The prior absl level (not a hard-coded default) is restored."""
        absl_logger = logging.getLogger("absl")
        absl_logger.setLevel(logging.DEBUG)
        try:
            compute_noise_multiplier_for_subsampled_gaussian(
                ApproxDPBudget(eps=1.0, delta=1e-5), 0.5, 20
            )
            assert absl_logger.level == logging.DEBUG
        finally:
            absl_logger.setLevel(logging.NOTSET)


class TestSubsampledGaussianBelowTheRdpEpsilonFloor:
    """Small epsilon at small delta must still calibrate, and calibrate tightly.

    The Sampled Gaussian Mechanism RDP bound is optimized over a finite Renyi
    order grid, so its RDP-to-(eps, delta) conversion carries an epsilon floor of
    roughly ``log(1/delta) / (max_order - 1)`` that no amount of noise beats.
    Inverting it below that floor either fails to bracket the target or latches
    onto the noise level where the Renyi values underflow to zero and returns a
    wildly over-noised multiplier. The calibrator therefore clamps to the
    unamplified exact-GDP composition, which has no such floor and is sound for
    the subsampled mechanism because data-independent Poisson sampling cannot
    worsen the guarantee. These tests pin (a) a finite multiplier certified by one
    of the two sound routes in that corner, (b) that the amplified route cannot
    inflate the answer past the unamplified one, and (c) that amplification
    is untouched above the floor.
    """

    CORNER = [
        (0.01, 1e-9, 1),
        (0.01, 1e-9, 10),
        (0.01, 1e-12, 100),
        (0.02, 1e-12, 10),
    ]

    @pytest.mark.parametrize("sampling_rate", [1.0, 0.1])
    @pytest.mark.parametrize("eps,delta,num_steps", CORNER)
    def test_subsampled_is_certified_by_a_sound_route(
        self, eps: float, delta: float, num_steps: int, sampling_rate: float
    ) -> None:
        nm = compute_noise_multiplier_for_subsampled_gaussian(
            ApproxDPBudget(eps=eps, delta=delta), sampling_rate, num_steps
        )
        assert math.isfinite(nm) and nm > 0
        # Either route may be the binding one, and each is sound on its own, so
        # the offer is honored as long as one of them certifies the multiplier.
        by_composed = delta_from_gdp(math.sqrt(num_steps) / nm, eps) <= delta * (
            1 + 1e-6
        )
        by_sgm = sampling_rate < 1.0 and epsilon_for_subsampled_gaussian(
            nm, sampling_rate, num_steps, delta
        ) <= eps * (1 + 1e-6)
        assert by_composed or by_sgm

    @pytest.mark.parametrize("sampling_rate", [1.0, 0.1])
    @pytest.mark.parametrize("eps,delta,num_steps", CORNER)
    def test_subsampled_is_never_looser_than_unamplified_composition(
        self, eps: float, delta: float, num_steps: int, sampling_rate: float
    ) -> None:
        budget = ApproxDPBudget(eps=eps, delta=delta)
        unamplified = compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
        amplified = compute_noise_multiplier_for_subsampled_gaussian(
            budget, sampling_rate, num_steps
        )
        assert amplified < unamplified or amplified == pytest.approx(unamplified)

    def test_subsampled_keeps_amplification_at_ordinary_budgets(self) -> None:
        """The clamp only binds below the floor; amplification is untouched above."""
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        unamplified = compute_noise_multiplier_for_composed_gaussian(budget, 10)
        amplified = compute_noise_multiplier_for_subsampled_gaussian(budget, 0.1, 10)
        assert amplified < 0.5 * unamplified
        assert epsilon_for_subsampled_gaussian(amplified, 0.1, 10, 1e-5) <= 1.0 + 1e-6


class TestSubsampledGaussianAmplification:
    """At ``q = 1`` subsampling is the composed mechanism; below it, it amplifies.

    The Sampled-Gaussian RDP (Mironov, Talwar and Zhang 2019, arXiv:1908.10530)
    gives a strictly smaller epsilon at a fixed noise multiplier as ``q``
    shrinks, so a fixed budget is met with strictly less noise.
    """

    @pytest.mark.parametrize(
        "eps,delta,num_steps", [(1.0, 1e-5, 10), (0.5, 1e-6, 50), (3.0, 1e-5, 20)]
    )
    def test_q_one_recovers_composed_calibration(
        self, eps: float, delta: float, num_steps: int
    ) -> None:
        budget = ApproxDPBudget(eps=eps, delta=delta)
        subsampled = compute_noise_multiplier_for_subsampled_gaussian(
            budget, 1.0, num_steps
        )
        composed = compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
        assert subsampled == pytest.approx(composed, rel=1e-9)

    def test_less_noise_at_smaller_q_for_fixed_budget(self) -> None:
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        composed = compute_noise_multiplier_for_composed_gaussian(budget, 10)
        multipliers = [
            compute_noise_multiplier_for_subsampled_gaussian(budget, q, 10)
            for q in (0.1, 0.5)
        ]
        assert all(multiplier < composed for multiplier in multipliers)
        assert all(a < b for a, b in zip(multipliers, multipliers[1:]))
