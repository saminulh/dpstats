"""Unit tests for dpstats.tasks.median_estimation algorithms.

Focused on UnboundedQuantileMechanism, the AboveThreshold/SVT
(Durfee 2023, arXiv:2305.01177) median estimator, and the smooth-sensitivity
Bun--Steinke (2019) smooth-sensitivity median family.
"""

import math
from typing import Literal, cast, get_args

import numpy as np
import pytest

from dpstats.domains import BoxDomain, LpBallDomain
from dpstats.algorithm import CompositeAlgorithm
from dpstats.tasks.median_estimation.transformations import (
    ToMEstimation,
)
from dpstats.tasks.m_estimation.algorithms.gradient_descent import (
    GD,
)
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.tasks.median_estimation.algorithms import (
    exponential_mechanism as _exponential_mechanism,
)
from dpstats.tasks.median_estimation.algorithms import (
    smooth_sensitivity as _smooth_sensitivity,
)
from dpstats.tasks.median_estimation.algorithms.exponential_mechanism import (
    ExponentialMechanism,
    WideExponentialMechanism,
)
from dpstats.tasks.median_estimation.algorithms.smooth_sensitivity import (
    SmoothSensitivity,
)
from dpstats.tasks.median_estimation.algorithms.unbounded_quantile_mechanism import (
    UnboundedQuantileMechanism,
)
from dpstats.simulations.median_estimation import (
    simulate_simple_gaussian,
)
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
    MedianEstimationOutput,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from tests.fixtures.randomness_fake import (
    FakeRandomness,
    GeneratorMethod,
    zero_draw,
)
from tests.unit.tasks.task_builders import Unreadable, box, median_metadata


def _ss_meta(n: int, lo: float = 0.0, hi: float = 10.0) -> MedianEstimationMetadata:
    return median_metadata(n, box((lo, hi)))


# Simple data with an unambiguous true median of 2.0.
_SS_DATA = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


class TestTypedBudgetAccounting:
    """Pin median mechanisms to pure, zCDP, or direct-Approx accounting."""

    @pytest.mark.parametrize(
        "algorithm",
        [
            ExponentialMechanism(),
            WideExponentialMechanism(),
            UnboundedQuantileMechanism(beta=2.0),
        ],
    )
    def test_pure_mechanisms_use_pure_spend(
        self, accounting_spy: AccountingSpy, algorithm
    ):
        metadata = _ss_meta(len(_SS_DATA))
        budget = ApproxDPBudget(eps=5.0, delta=1e-6)
        algorithm.run(
            metadata,
            MedianEstimationInput(X=_SS_DATA.copy()),
            budget=budget,
            seed=0,
        )
        assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)

    @pytest.mark.parametrize("noise", ["lln", "uln", "arsinh_normal"])
    def test_zcdp_mechanisms_use_zcdp_spend(self, accounting_spy: AccountingSpy, noise):
        metadata = _ss_meta(len(_SS_DATA))
        budget = ZCDPBudget(rho=8.0)
        SmoothSensitivity(noise=noise, beta=0.1).run(
            metadata,
            MedianEstimationInput(X=_SS_DATA.copy()),
            budget=budget,
            seed=0,
        )
        assert accounting_spy.calls == (AccountingCall("spend_as_zcdp", budget),)

    @pytest.mark.parametrize("noise", ["lln", "uln", "arsinh_normal"])
    def test_zcdp_mechanisms_refuse_pure_budget(
        self, accounting_spy: AccountingSpy, noise
    ):
        metadata = _ss_meta(len(_SS_DATA))
        budget = PureDPBudget(eps=5.0)
        with pytest.raises(InvalidRegime, match="PureDPBudget"):
            SmoothSensitivity(noise=noise, beta=0.1).run(
                metadata,
                MedianEstimationInput(X=_SS_DATA.copy()),
                budget=budget,
                seed=0,
            )
        assert accounting_spy.calls == (AccountingCall("spend_as_zcdp", budget),)


class TestUnboundedQuantileMechanism:
    def test_requires_domain_with_lower_bound(self):
        alg = UnboundedQuantileMechanism(beta=2.0)
        meta = median_metadata(4)
        inp = MedianEstimationInput(X=np.arange(4.0))
        with pytest.raises(InvalidRegime):
            alg.run(meta, inp, budget=PureDPBudget(eps=1.0), seed=0)


class TestMedianDomainContract:
    def test_metadata_domain_must_be_scalar(self):
        with pytest.raises(ValueError, match="scalar"):
            median_metadata(4, 2)

    @pytest.mark.parametrize(
        "alg_cls, budget",
        [
            (ExponentialMechanism, PureDPBudget(eps=1.0)),
            (WideExponentialMechanism, PureDPBudget(eps=1.0)),
            (SmoothSensitivity, ApproxDPBudget(eps=1.0, delta=1e-5)),
        ],
    )
    def test_bounded_mechanism_rejects_half_open_domain(self, alg_cls, budget):
        metadata = median_metadata(4, box((0.0, np.inf)))
        with pytest.raises(InvalidRegime, match="finite"):
            alg_cls().run(
                metadata, MedianEstimationInput(X=np.arange(4.0)), budget, seed=0
            )

    @pytest.mark.parametrize(
        "alg_cls, released_inside",
        [
            (ExponentialMechanism, True),
            (WideExponentialMechanism, True),
            (UnboundedQuantileMechanism, True),
            # Smooth-sensitivity noise addition can release outside the domain.
            (SmoothSensitivity, False),
        ],
    )
    def test_any_scalar_domain_with_a_finite_bounding_box_is_accepted(
        self, alg_cls, released_inside
    ):
        """A ball is not a box, but its bounding box is the interval these need."""
        ball = LpBallDomain.at_origin(p=2.0, radius=5.0, dim=1)
        metadata = median_metadata(5, ball)
        out = alg_cls().run(
            metadata,
            MedianEstimationInput(X=np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )
        assert math.isfinite(out.median_est)
        if released_inside:
            assert -5.0 <= out.median_est <= 5.0

    def test_known_domain_simulated_instance_publishes_scalar_box(self):
        metadata, input_data, _ = simulate_simple_gaussian(
            seed=0, n_samples=10, known_domain=True
        )

        assert isinstance(metadata.domain, BoxDomain)
        np.testing.assert_array_equal(
            metadata.domain.bounds,
            [[input_data.X.min(), input_data.X.max()]],
        )


class TestMEstimationTreatmentPath:
    """The median-estimation composition using M-estimation gradient descent.

    ``ToMEstimation -> GD`` presents ``GD`` as a
    native median-estimation algorithm: its child->parent map fixes the L1 loss
    (whose minimizer is the sample median) on ``MEstimationMetadata``, runs in
    m_estimation space, then projects the fitted parameter back to a
    ``MedianEstimationOutput``.
    """

    def test_gradient_descent_runs_on_median_instance(self):
        # Minimizing sum_i |x_i - theta| (the transformation's L1 objective)
        # has the median as its minimizer, so GD should land near it.
        data = np.arange(1.0, 101.0)
        true_median = float(np.median(data))
        # The declared interval is the parameter domain the descent projects
        # onto; on the unbounded line, L1-subgradient steps stop well short of
        # the median within the step budget.
        meta = median_metadata(data.shape[0], box((0.0, 101.0)))
        inp = MedianEstimationInput(X=data)

        out = CompositeAlgorithm(
            transformations=(ToMEstimation(),),
            algorithm=GD(num_steps=100),
        ).run(meta, inp, budget=NO_PRIVACY, seed=0)

        # Output is pushed back to the caller's (median) task format.
        assert isinstance(out, MedianEstimationOutput)
        assert np.isfinite(out.median_est)
        assert isinstance(out.median_est, float)
        # A meaningful estimate near the median: inside the data range and not
        # far from the truth. (L1-subgradient GD converges slowly, so the bound
        # is generous -- the point is that the composed path produces a real
        # estimate, not garbage.)
        assert float(data.min()) <= out.median_est <= float(data.max())
        assert abs(out.median_est - true_median) < 15.0


class TestThresholdNoiseDrawnOnce:
    """The privacy crux: threshold noise drawn ONCE, outside the candidate loop.

    AboveThreshold/SVT is only eps-DP when the threshold noise is shared across
    all comparisons (Durfee 2023, Algorithm 1, line 1). Per run the algorithm
    must make exactly ``1`` threshold draw plus ``k_halt + 1`` per-query draws,
    regardless of how many candidates are examined. If the threshold draw were
    (wrongly) inside the loop, the count would be ``2 * (k_halt + 1)`` and would
    scale with the number of candidates.
    """

    @pytest.mark.parametrize(
        ("beta", "num_query_iterations"),
        [(3.0, 3), (1.2, 10)],
    )
    def test_one_threshold_draw_independent_of_candidate_count(
        self, fake_randomness: FakeRandomness, beta, num_query_iterations
    ):
        # Increasing, evenly-spread data so the search runs several iterations.
        # Zero noise makes the noisy comparisons exact, so the iteration count
        # is fixed by the data alone.
        data = np.arange(10.0)
        lower = 0.0
        fake_randomness.script("laplace", zero_draw)

        alg = UnboundedQuantileMechanism(beta=beta)
        meta = median_metadata(data.shape[0], box((lower, np.inf)))
        alg.run(
            meta, MedianEstimationInput(X=data), budget=PureDPBudget(eps=1.0), seed=0
        )

        # 1 threshold draw + one fresh query draw per iteration.
        assert len(fake_randomness.draws_of("laplace")) == 1 + num_query_iterations


class TestNoiseScales:
    """Both the (once-drawn) threshold and per-query noise use Lap(2/eps).

    Durfee (2023) Lemma 3.1 / Lemma 4.1: the count queries are monotonic with
    sensitivity Delta = 1; with the symmetric split eps_1 = eps_2 = eps/2 both
    scales are Delta / eps_i = 2/eps, yielding eps-DP for monotonic queries.
    """

    @pytest.mark.parametrize("eps", [0.5, 4.0])
    def test_all_laplace_scales_are_two_over_eps(
        self, fake_randomness: FakeRandomness, eps
    ):
        data = np.arange(10.0)
        fake_randomness.script("laplace", zero_draw)

        alg = UnboundedQuantileMechanism(beta=2.0)
        meta = median_metadata(data.shape[0], box((0.0, np.inf)))
        alg.run(
            meta, MedianEstimationInput(X=data), budget=PureDPBudget(eps=eps), seed=0
        )

        scales = fake_randomness.parameters_of("laplace", "scale")
        assert len(scales) >= 2  # at least threshold + one query
        for scale in scales:
            assert scale == pytest.approx(2.0 / eps)


class TestCorrectness:
    """The mechanism recovers a value near the true median on simple data."""

    def test_candidate_grid_starts_at_public_lower_bound(
        self, fake_randomness: FakeRandomness
    ):
        fake_randomness.script("laplace", zero_draw)
        lower = 3.5
        data = np.full(5, lower)

        output = UnboundedQuantileMechanism(beta=2.0).run(
            median_metadata(data.size, box((lower, np.inf))),
            MedianEstimationInput(X=data),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )

        assert output.median_est == lower
        assert len(fake_randomness.draws_of("laplace")) == 2

    def test_subunit_median_releases_corrected_grid_bracket(
        self, fake_randomness: FakeRandomness
    ):
        fake_randomness.script("laplace", zero_draw)
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        output = UnboundedQuantileMechanism(beta=1.1).run(
            median_metadata(data.size, box((0.0, np.inf))),
            MedianEstimationInput(X=data),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )

        assert output.median_est == pytest.approx(1.1**3 - 1)
        assert output.median_est < 1.0

    def test_grid_tie_releases_empirical_median(self, fake_randomness: FakeRandomness):
        fake_randomness.script("laplace", zero_draw)
        data = np.array([0.0, 1.0, 1.0, 2.0, 3.0])

        output = UnboundedQuantileMechanism(beta=2.0).run(
            median_metadata(data.size, box((0.0, np.inf))),
            MedianEstimationInput(X=data),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )

        assert output.median_est == 1.0
        assert output.median_est == np.median(data)

    def test_recovers_approximate_median(self):
        # n=1000, true median ~500.5; fine grid (beta=1.05) + large eps (low noise).
        data = np.arange(1.0, 1001.0)
        true_median = float(np.median(data))
        alg = UnboundedQuantileMechanism(beta=1.05)
        meta = median_metadata(data.shape[0], box((0.0, np.inf)))
        out = alg.run(
            meta,
            MedianEstimationInput(X=data),
            budget=PureDPBudget(eps=10.0),
            seed=0,
        )

        assert 0.7 * true_median < out.median_est < 1.3 * true_median


class TestSmoothSensitivityMedian:
    @pytest.mark.parametrize("noise", ["lln", "uln", "arsinh_normal"])
    def test_no_privacy_is_exact_midpoint_median(
        self,
        fake_randomness: FakeRandomness,
        noise: Literal["lln", "uln", "arsinh_normal"],
    ):
        output = SmoothSensitivity(noise=noise, beta=0.1).run(
            _ss_meta(len(_SS_DATA)),
            MedianEstimationInput(X=_SS_DATA.copy()),
            budget=NO_PRIVACY,
            seed=7,
        )
        assert output.median_est == np.median(_SS_DATA)
        assert not any(
            fake_randomness.draws_of(method) for method in get_args(GeneratorMethod)
        )

    @pytest.mark.parametrize("noise", ["uln", "arsinh_normal"])
    def test_public_infeasibility_precedes_rng_and_data_access(
        self,
        fake_randomness: FakeRandomness,
        noise: Literal["lln", "uln", "arsinh_normal"],
    ):
        unreadable = cast(MedianEstimationInput, Unreadable("Input"))
        with pytest.raises(InvalidRegime, match="dilation"):
            SmoothSensitivity(noise=noise, beta=0.1).run(
                _ss_meta(len(_SS_DATA)), unreadable, budget=ZCDPBudget(rho=1e-6), seed=0
            )
        assert fake_randomness.seeds == ()

    @pytest.mark.parametrize(
        ("noise", "base_draw", "other_draw"),
        [("lln", "laplace", "uniform"), ("uln", "uniform", "laplace")],
    )
    def test_log_normal_variants_draw_their_own_base_noise(
        self,
        fake_randomness: FakeRandomness,
        noise: Literal["lln", "uln", "arsinh_normal"],
        base_draw: str,
        other_draw: str,
    ):
        SmoothSensitivity(noise=noise, beta=0.1).run(
            _ss_meta(len(_SS_DATA)),
            MedianEstimationInput(X=_SS_DATA.copy()),
            budget=ZCDPBudget(rho=8.0),
            seed=0,
        )

        assert len(fake_randomness.draws_of(base_draw)) == 1
        assert fake_randomness.draws_of(other_draw) == ()
        assert len(fake_randomness.draws_of("normal")) == 1

    @pytest.mark.parametrize("noise", ["lln", "uln", "arsinh_normal"])
    def test_release_scale_satisfies_source_privacy_equation(
        self,
        fake_randomness: FakeRandomness,
        noise: Literal["lln", "uln", "arsinh_normal"],
    ):
        """The dilation ``s`` solves the cited theorem's privacy equation.

        Unit noise (``Z = 1``) makes the release ``median + S / s`` with
        ``S = exp(-4 beta) * 10 / 2`` for ``_SS_DATA`` in ``[0, 10]``, as
        worked in ``test_hand_worked_windows``. Substituting the observed
        ``s`` into the theorem's ``epsilon`` must return the spent ``epsilon``.
        """
        beta = 0.1
        eps_cdp = 4.0
        sigma_arsinh = 2 / math.sqrt(3)
        fake_randomness.script("laplace", lambda parameters: 1.0)
        fake_randomness.script("uniform", lambda parameters: 1.0)
        fake_randomness.script(
            "normal",
            lambda parameters: (
                math.asinh(sigma_arsinh) / sigma_arsinh
                if noise == "arsinh_normal"
                else 0.0
            ),
        )

        release = (
            SmoothSensitivity(noise=noise, beta=beta)
            .run(
                _ss_meta(len(_SS_DATA)),
                MedianEstimationInput(X=_SS_DATA.copy()),
                budget=ZCDPBudget(rho=eps_cdp**2 / 2),
                seed=0,
            )
            .median_est
        )
        smooth_sensitivity = math.exp(-4 * beta) * (10 - 0) / 2
        dilation = smooth_sensitivity / (release - np.median(_SS_DATA))

        if noise == "lln":
            # Section 3.1.1: sigma is the one real root of
            # 5 (eps/beta) sigma^3 - 5 sigma^2 - 1, then eps = beta/sigma +
            # exp(3 sigma^2/2) s (Theorem 18).
            roots = np.roots([5 * eps_cdp / beta, -5.0, 0.0, -1.0])
            (sigma,) = [float(root.real) for root in roots if abs(root.imag) < 1e-12]
            reconstructed_eps = beta / sigma + math.exp(1.5 * sigma**2) * dilation
        elif noise == "uln":
            # Proposition 4 at sigma = sqrt(2).
            sigma = math.sqrt(2)
            reconstructed_eps = beta / sigma + (
                math.exp(1.5 * sigma**2)
                * math.sqrt(2 / (math.pi * sigma**2))
                * dilation
            )
        else:
            # Theorem 25 at sigma = 2 / sqrt(3).
            sigma = sigma_arsinh
            reconstructed_eps = math.sqrt(
                beta * (beta / sigma**2 + 1 / sigma + 2)
            ) + dilation * (2 / (3 * sigma) + sigma / 2)

        assert reconstructed_eps == pytest.approx(eps_cdp)


class TestSmoothSensitivityAddRemove:
    """Hand-worked bound values, and the local-bound and beta-smoothness properties under add/remove adjacency."""

    def test_hand_worked_windows(self):
        """The bound on two small samples, worked by hand.

        ``[0, 1, 2, 3, 4]`` in ``[0, 10]`` sorts with its sentinels to
        ``0 0 1 2 3 4 10``; the middle value is 2. The widest window around it
        is from the second 0 to 10 with four values strictly inside, worth
        ``exp(-4 beta) * 10 / 2``, which beats every narrower window at
        ``beta = 0.1``. ``[1, 9]`` in ``[0, 10]`` sorts to ``0 1 9 10`` with
        middle value 9; at ``beta = 1`` the adjacent gap ``9 - 1`` gives
        ``8 / 2 = 4``, and the wider windows ``(0, 9)`` and ``(1, 10)`` only
        ``9 exp(-1) / 2``.
        """
        assert _smooth_sensitivity._midpoint_median_smooth_sensitivity(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 0.1, 0.0, 10.0
        ) == pytest.approx(math.exp(-0.4) * 5)
        assert _smooth_sensitivity._midpoint_median_smooth_sensitivity(
            np.array([1.0, 9.0]), 1.0, 0.0, 10.0
        ) == pytest.approx(4.0)

    @pytest.mark.parametrize("beta", [0.01, 0.1, 0.7, 1.0])
    def test_random_neighbor_properties(self, beta: float):
        rng = np.random.default_rng(197)
        lower, upper = 0.0, 10.0
        support = np.array([0.0, 1.0, 5.0, 9.0, 10.0])

        for n in range(1, 16):
            for _ in range(20):
                data = rng.choice(support, size=n, replace=True)
                added = float(rng.choice(support))
                neighbor = np.append(data, added)
                sensitivity = _smooth_sensitivity._midpoint_median_smooth_sensitivity(
                    data, beta, lower, upper
                )
                neighbor_sensitivity = (
                    _smooth_sensitivity._midpoint_median_smooth_sensitivity(
                        neighbor, beta, lower, upper
                    )
                )

                median = float(np.median(data))
                neighbor_median = float(np.median(neighbor))
                observed_change = abs(median - neighbor_median)
                assert observed_change <= sensitivity + 1e-12
                assert observed_change <= neighbor_sensitivity + 1e-12
                assert sensitivity <= math.exp(beta) * neighbor_sensitivity + 1e-12
                assert neighbor_sensitivity <= math.exp(beta) * sensitivity + 1e-12


class TestExponentialMechanismDomainClip:
    """The exponential mechanisms clip their inputs to [lb, ub].

    Alabi et al. 2022 Algorithms 3-4 begin with a "clip z to the range" step.
    Without it, a record outside [lb, ub] leaves a candidate interval extending
    beyond the domain, so the mechanism can release a value outside [lb, ub] --
    which has probability 0 on the neighbor lacking that record, i.e. unbounded
    privacy loss. Inspecting the complete support handed to the release primitive
    makes the invariant deterministic: neither extreme outlier may leave any
    candidate boundary outside the public domain.
    """

    @pytest.mark.parametrize(
        "alg_cls", [ExponentialMechanism, WideExponentialMechanism]
    )
    def test_candidate_support_stays_within_domain(self, monkeypatch, alg_cls):
        lb, ub = 0.0, 10.0
        data = np.array([-1000.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0])
        captured_support = []

        def capture_release(candidates, budget, rng):
            captured_support.extend(candidates)
            return 5.0

        monkeypatch.setattr(
            _exponential_mechanism, "_release_rung_weighted_median", capture_release
        )
        alg = alg_cls()
        meta = median_metadata(data.shape[0], box((lb, ub)))
        out = alg.run(
            meta,
            MedianEstimationInput(X=data.copy()),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )
        assert out.median_est == 5.0
        assert captured_support[0] == lb
        assert captured_support[-1] == ub
        assert all(lb <= candidate <= ub for candidate in captured_support)


class TestExponentialMechanismUnconstrainedBudget:
    @pytest.mark.parametrize(
        "alg_cls, low, high",
        [
            (ExponentialMechanism, 2.0, 8.0),
            # Algorithm 5 widens three values on each side and leaves the
            # median in place, so the two gaps touching it are the lowest rung.
            (WideExponentialMechanism, 2.0 - 0.01, 8.0 + 0.01),
        ],
    )
    def test_release_uses_lowest_rung(self, alg_cls, low, high):
        data = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 9.0, 10.0])
        meta = median_metadata(data.shape[0], box((-20.0, 20.0)))
        for seed in range(6):
            out = alg_cls().run(
                meta,
                MedianEstimationInput(X=data.copy()),
                budget=NO_PRIVACY,
                seed=seed,
            )
            assert low <= out.median_est <= high

    def test_an_even_sample_releases_inside_its_middle_gap(self):
        """Algorithm 3 inserts nothing for an even sample: with the endpoints,
        the gap between the two middle values is the one gap at distance 0."""
        data = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 9.0])
        meta = median_metadata(data.shape[0], box((-20.0, 20.0)))
        for seed in range(6):
            out = ExponentialMechanism().run(
                meta,
                MedianEstimationInput(X=data.copy()),
                budget=NO_PRIVACY,
                seed=seed,
            )
            assert 2.0 <= out.median_est <= 5.0

    def test_empty_lowest_rungs_are_skipped(self):
        data = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0])
        meta = median_metadata(data.shape[0], box((-20.0, 20.0)))
        for seed in range(6):
            out = ExponentialMechanism().run(
                meta,
                MedianEstimationInput(X=data.copy()),
                budget=NO_PRIVACY,
                seed=seed,
            )
            assert 2.0 <= out.median_est <= 3.0

    def test_collapsed_domain_releases_its_only_value(self):
        data = np.array([1.0, 2.0, 3.0])
        meta = median_metadata(data.shape[0], box((2.0, 2.0)))
        out = ExponentialMechanism().run(
            meta, MedianEstimationInput(X=data), budget=NO_PRIVACY, seed=0
        )
        assert out.median_est == 2.0
