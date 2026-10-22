"""Tests for Harini and Tandon's DP-RSS simple-regression mechanism."""

from __future__ import annotations

import numpy as np
import pytest

from dpstats.tasks import list_built_in_metrics
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import BoxDomain, Domain, EuclideanSpace
from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms import DPRSS
from dpstats.tasks import (
    SimpleLinearRegression,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order

l2_error_to_ground_truth = list_built_in_metrics(SimpleLinearRegression)[
    "l2_error_to_ground_truth"
]


UNIT_DOMAIN = BoxDomain(bounds=np.array([[0.0, 1.0]]))
ORIGIN_INPUT = SimpleLinearRegressionInput(x=np.zeros(1), y=np.zeros(1))
ORIGIN_QUERY = np.array([0.0, 0.0, 1.0])


def _metadata(
    n_samples: int,
    x_domain: Domain = UNIT_DOMAIN,
    y_domain: Domain = UNIT_DOMAIN,
) -> SimpleLinearRegressionMetadata:
    return SimpleLinearRegressionMetadata(
        n_samples=n_samples,
        x_domain=x_domain,
        y_domain=y_domain,
    )


def _unit_square_instance(
    n_samples: int, seed: int
) -> tuple[SimpleLinearRegressionMetadata, SimpleLinearRegressionInput]:
    """Build a well-conditioned instance already inside the unit square."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.95, size=n_samples)
    y = np.clip(0.6 * x + 0.15 + rng.normal(0.0, 0.02, size=n_samples), 0.0, 1.0)
    return _metadata(n_samples), SimpleLinearRegressionInput(x=x, y=y)


def _least_squares_line(x: Array1D, y: Array1D) -> tuple[float, float]:
    """Independently fit slope and intercept with NumPy's least squares."""
    design = np.column_stack((x, np.ones_like(x)))
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(solution[0]), float(solution[1])


def _run_with_scripted_noise(
    fake_randomness: FakeRandomness,
    f_noise: Array1D,
    g_noise: Array1D,
    metadata: SimpleLinearRegressionMetadata,
    input: SimpleLinearRegressionInput,
    budget: DPBudget = PureDPBudget(eps=1.0),
) -> SimpleLinearRegressionOutput:
    """Run the mechanism with both vector releases' Laplace noise prepared."""
    fake_randomness.script("laplace", draws_in_order((f_noise, g_noise)))
    output = DPRSS().run(metadata, input, budget, seed=0)
    # Each prepared vector must match the release it stands in for, or the
    # transcript the assertions reason about is not the one the run used.
    assert fake_randomness.parameters_of("laplace", "size") == [
        f_noise.size,
        g_noise.size,
    ]
    return output


class TestReleaseContract:
    """Post-processing of the two vector releases reproduces least squares."""

    def test_noiseless_release_matches_ordinary_least_squares(self) -> None:
        # An unconstrained budget drives the Laplace scale to zero, so the
        # combined sufficient statistics are exact. The weighted direct and
        # redundant estimates then agree with an independent OLS fit only
        # because each query's three coordinates sum to the record count.
        metadata, input = _unit_square_instance(200, seed=11)
        output = DPRSS().run(metadata, input, NO_PRIVACY, seed=3)

        slope, intercept = _least_squares_line(input.x, input.y)
        assert output.slope == pytest.approx(slope, rel=1e-9)
        assert output.intercept == pytest.approx(intercept, rel=1e-9)

    def test_noiseless_release_matches_least_squares_on_a_wide_domain(self) -> None:
        # The same identity must survive the Appendix-A transport, so the
        # normalization and its inverse are pinned against an unnormalized fit.
        rng = np.random.default_rng(5)
        x = rng.uniform(-40.0, 40.0, size=150)
        y = np.clip(-3.0 * x + 12.0 + rng.normal(0.0, 1.0, size=150), -200.0, 200.0)
        metadata = _metadata(
            150,
            x_domain=BoxDomain(bounds=np.array([[-50.0, 50.0]])),
            y_domain=BoxDomain(bounds=np.array([[-200.0, 200.0]])),
        )
        output = DPRSS().run(
            metadata, SimpleLinearRegressionInput(x=x, y=y), NO_PRIVACY, seed=0
        )

        slope, intercept = _least_squares_line(x, y)
        assert output.slope == pytest.approx(slope, rel=1e-8)
        assert output.intercept == pytest.approx(intercept, rel=1e-8)

    def test_large_epsilon_recovers_the_generating_line(self) -> None:
        metadata, input = _unit_square_instance(500, seed=2)
        output = DPRSS().run(metadata, input, PureDPBudget(eps=500.0), seed=7)

        assert output.slope == pytest.approx(0.6, abs=0.05)
        assert output.intercept == pytest.approx(0.15, abs=0.05)


class TestSensitivity:
    """The unit L1 sensitivities that fix the mechanism's noise scale."""

    def test_query_rows_are_nonnegative_and_sum_to_one(self) -> None:
        # Lemmas 4.2 and 4.6. Adding or removing a record moves either sum
        # query by exactly that record's row, so the L1 sensitivity is the row's
        # L1 norm. On the unit square each row is nonnegative and sums to one,
        # which fixes both sensitivities at one for every record count.
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=64)
        y = rng.uniform(0.0, 1.0, size=64)
        f = np.column_stack((x**2, x - x**2, 1.0 - x))
        g = np.column_stack((x * y, (1.0 - x) * y, 1.0 - y))

        assert np.all(f >= 0.0)
        assert np.all(g >= 0.0)
        np.testing.assert_allclose(np.sum(f, axis=1), 1.0)
        np.testing.assert_allclose(np.sum(g, axis=1), 1.0)

    def test_both_releases_use_the_even_split_and_unit_sensitivity(
        self, fake_randomness: FakeRandomness
    ) -> None:
        eps = 4.0
        metadata, input = _unit_square_instance(20, seed=1)
        _run_with_scripted_noise(
            fake_randomness,
            np.zeros(3),
            np.zeros(3),
            metadata,
            input,
            budget=PureDPBudget(eps=eps),
        )

        # Sensitivity one at eps/2 per release is a Laplace scale of 2/eps.
        assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx(
            [2.0 / eps, 2.0 / eps]
        )
        assert fake_randomness.parameters_of("laplace", "size") == [3, 3]

    @pytest.mark.parametrize(
        "budget",
        [
            PureDPBudget(eps=2.0),
            ApproxDPBudget(eps=2.0, delta=1e-5),
            ZCDPBudget(rho=2.0),
        ],
    )
    def test_offered_budget_is_spent_as_pure(
        self,
        fake_randomness: FakeRandomness,
        accounting_spy: AccountingSpy,
        budget: DPBudget,
    ) -> None:
        metadata, input = _unit_square_instance(20, seed=1)
        _run_with_scripted_noise(
            fake_randomness,
            np.zeros(3),
            np.zeros(3),
            metadata,
            input,
            budget=budget,
        )

        assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)
        scales = fake_randomness.parameters_of("laplace", "scale")
        assert len(scales) == 2
        assert scales[0] == pytest.approx(scales[1])


class TestTranscriptFallbacks:
    """Both printed branches release the paper's constant normalized line."""

    def test_nonpositive_noisy_count_releases_the_midpoint_line(
        self, fake_randomness: FakeRandomness
    ) -> None:
        # One origin record contributes ORIGIN_QUERY to both releases. Offset
        # the scripted noise to obtain transcripts (-6, 6, 0) and (0, 0, -6):
        # their averaged count is -3 while the noisy Gram determinant is +17,
        # so only the count guard can produce the constant release.
        output = _run_with_scripted_noise(
            fake_randomness,
            np.array([-6.0, 6.0, 0.0]) - ORIGIN_QUERY,
            np.array([0.0, 0.0, -6.0]) - ORIGIN_QUERY,
            _metadata(1),
            ORIGIN_INPUT,
        )

        assert output.slope == 0.0
        assert output.intercept == pytest.approx(0.5)

    def test_nonpositive_noisy_determinant_releases_the_midpoint_line(
        self, fake_randomness: FakeRandomness
    ) -> None:
        # This transcript averages to a positive count of 3, so the count guard
        # passes; its noisy Gram determinant is -1 and the second guard fires.
        output = _run_with_scripted_noise(
            fake_randomness,
            np.zeros(3) - ORIGIN_QUERY,
            np.array([6.0, 0.0, 0.0]) - ORIGIN_QUERY,
            _metadata(1),
            ORIGIN_INPUT,
        )

        assert output.slope == 0.0
        assert output.intercept == pytest.approx(0.5)

    def test_admissible_transcript_releases_an_ordinary_fit(
        self, fake_randomness: FakeRandomness
    ) -> None:
        # Control: on the same one-record instance a transcript that passes both
        # guards releases a fitted line, so the midpoint assertions above are
        # not vacuous. This one is the transcript four records at the corners
        # (0,0) and (1,1) would produce without noise, whose fit is y = x.
        output = _run_with_scripted_noise(
            fake_randomness,
            np.array([2.0, 0.0, 2.0]) - ORIGIN_QUERY,
            np.array([2.0, 0.0, 2.0]) - ORIGIN_QUERY,
            _metadata(1),
            ORIGIN_INPUT,
        )

        assert output.slope == pytest.approx(1.0)
        assert output.intercept == pytest.approx(0.0)

    def test_refined_weights_match_a_fixed_transcript(
        self, fake_randomness: FakeRandomness
    ) -> None:
        # Offset the scripted draws so the complete transcript is (4, 1, 2),
        # (3, 2, 1). Direct substitution into Algorithm 3 gives slope -155/113
        # and intercept 204/113, pinning both inverse-variance weight pairs.
        output = _run_with_scripted_noise(
            fake_randomness,
            np.array([4.0, 1.0, 2.0]) - ORIGIN_QUERY,
            np.array([3.0, 2.0, 1.0]) - ORIGIN_QUERY,
            _metadata(1),
            ORIGIN_INPUT,
        )

        assert output.slope == pytest.approx(-155.0 / 113.0)
        assert output.intercept == pytest.approx(204.0 / 113.0)

    def test_fallback_maps_to_the_public_response_midpoint(
        self, fake_randomness: FakeRandomness
    ) -> None:
        # The constant normalized line is (0, 0.5), so Appendix A transports it
        # to a flat line at the midpoint of the public response domain.
        output = _run_with_scripted_noise(
            fake_randomness,
            np.zeros(3) - ORIGIN_QUERY,
            np.array([6.0, 0.0, 0.0]) - ORIGIN_QUERY,
            _metadata(
                1,
                x_domain=BoxDomain(bounds=np.array([[-4.0, 12.0]])),
                y_domain=BoxDomain(bounds=np.array([[20.0, 90.0]])),
            ),
            ORIGIN_INPUT,
        )

        assert output.slope == 0.0
        assert output.intercept == pytest.approx(55.0)


class TestPublicTransport:
    """Only public metadata shapes the mechanism's view of the data."""

    def test_public_affine_rescaling_is_equivariant(self) -> None:
        metadata, input = _unit_square_instance(120, seed=4)
        x_scale, x_shift, y_scale, y_shift = 4.0, 2.0, 8.0, -4.0
        rescaled = SimpleLinearRegressionInput(
            x=x_scale * input.x + x_shift,
            y=y_scale * input.y + y_shift,
        )
        rescaled_metadata = _metadata(
            120,
            x_domain=BoxDomain(bounds=np.array([[x_shift, x_scale + x_shift]])),
            y_domain=BoxDomain(bounds=np.array([[y_shift, y_scale + y_shift]])),
        )

        base = DPRSS().run(metadata, input, PureDPBudget(eps=3.0), seed=9)
        moved = DPRSS().run(rescaled_metadata, rescaled, PureDPBudget(eps=3.0), seed=9)

        expected_slope = y_scale * base.slope / x_scale
        assert moved.slope == pytest.approx(expected_slope, rel=1e-9)
        assert moved.intercept == pytest.approx(
            y_scale * base.intercept + y_shift - expected_slope * x_shift, rel=1e-9
        )

    def test_public_reference_count_is_never_used(self) -> None:
        # The mechanism keeps the paper's noisy count, so a different declared
        # reference n cannot change the release.
        _, input = _unit_square_instance(80, seed=6)
        truthful = DPRSS().run(_metadata(80), input, PureDPBudget(eps=1.0), seed=5)
        misdeclared = DPRSS().run(_metadata(5000), input, PureDPBudget(eps=1.0), seed=5)

        assert misdeclared.slope == pytest.approx(truthful.slope)
        assert misdeclared.intercept == pytest.approx(truthful.intercept)

    @pytest.mark.parametrize(
        "x_domain,y_domain,message",
        [
            (
                EuclideanSpace(dim=1),
                UNIT_DOMAIN,
                "The x interval requires a sound L2 enclosing ball",
            ),
            (
                BoxDomain(bounds=np.array([[3.0, 3.0]])),
                UNIT_DOMAIN,
                "x_domain must have positive width",
            ),
            (
                BoxDomain(bounds=np.array([[-np.inf, 1.0]])),
                UNIT_DOMAIN,
                "The x interval requires a sound L2 enclosing ball",
            ),
            (
                UNIT_DOMAIN,
                BoxDomain(bounds=np.array([[0.0, np.inf]])),
                "The y interval requires a sound L2 enclosing ball",
            ),
        ],
    )
    def test_unsupported_public_domains_are_refused(
        self, x_domain: Domain, y_domain: Domain, message: str
    ) -> None:
        _, input = _unit_square_instance(10, seed=0)
        with pytest.raises(InvalidRegime, match=message):
            DPRSS().run(
                _metadata(10, x_domain=x_domain, y_domain=y_domain),
                input,
                PureDPBudget(eps=1.0),
                seed=0,
            )
