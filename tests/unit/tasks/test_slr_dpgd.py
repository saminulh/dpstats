"""Tests for Alabi et al.'s private simple-regression gradient descent."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import JsonValue

from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime
from dpstats.budgets import (
    ApproxDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import BoxDomain, EuclideanSpace
from dpstats.task import Array1D
from dpstats.tasks.simple_linear_regression.algorithms import DPGD
from dpstats.tasks.simple_linear_regression.algorithms.dpgd import (
    _clipped_gradient_sum,  # pyright: ignore[reportPrivateUsage]
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
    assert_budget_close,
)
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw


UNIT_DOMAIN = BoxDomain(bounds=np.array([[0.0, 1.0]]))


def _metadata(n_samples: int) -> SimpleLinearRegressionMetadata:
    return SimpleLinearRegressionMetadata(
        n_samples=n_samples,
        x_domain=UNIT_DOMAIN,
        y_domain=UNIT_DOMAIN,
    )


def _unit_box_instance(
    seed: int,
) -> tuple[
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionInput,
    dict[str, JsonValue],
]:
    """Sixty near-linear records in the unit box."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=60)
    y = np.clip(0.4 * x + 0.2 + rng.normal(0.0, 0.03, size=60), 0.0, 1.0)
    return (
        _metadata(60),
        SimpleLinearRegressionInput(x=x, y=y),
        {"ground_truth": [0.4, 0.2]},
    )


class TestPaperBoundary:
    def test_clipped_record_norm_and_relation_conversion(self) -> None:
        # One clipped record has the add/remove sensitivities used by the two
        # budget routes. A replace-one difference is twice each value.
        tau = 0.2
        first = _clipped_gradient_sum(
            np.array([0.0]), np.array([0.0]), np.array([0.5, 0.5]), tau
        )
        second = _clipped_gradient_sum(
            np.array([0.0]), np.array([1.0]), np.array([0.5, 0.5]), tau
        )
        assert np.linalg.norm(first, ord=1) == pytest.approx(2 * tau)
        assert np.linalg.norm(first) == pytest.approx(math.sqrt(2) * tau)
        assert np.linalg.norm(first - second, ord=1) == pytest.approx(4 * tau)
        assert np.linalg.norm(first - second) == pytest.approx(2 * math.sqrt(2) * tau)

    def test_run_follows_clipped_descent_step(self) -> None:
        # At x=.25/.75 the paper basis is the identity. Starting from
        # (.5, .5), y=(0, 1), and tau=.25 gives first clipped gradient
        # (.25, -.25). The exact noiseless two-step release is therefore
        # slope=1.0240264607, intercept=-.0120132304; the reversed appendix
        # sign drives the slope negative instead. A huge rho makes noise
        # negligible while exercising the public run() wiring.
        input = SimpleLinearRegressionInput(
            x=np.array([0.25, 0.75]),
            y=np.array([0.0, 1.0]),
        )
        output = DPGD(tau=0.25, num_steps=2).run(
            _metadata(2),
            input,
            ZCDPBudget(rho=1e10),
            0,
        )
        assert output.slope == pytest.approx(1.0240264607, abs=1e-3)
        assert output.intercept == pytest.approx(-0.0120132304, abs=1e-3)


class TestCalibration:
    def test_pure_budget_uses_add_remove_laplace_scale(
        self,
        fake_randomness: FakeRandomness,
        accounting_spy: AccountingSpy,
    ) -> None:
        metadata, input, _ = _unit_box_instance(0)
        fake_randomness.script(
            "laplace", lambda parameters: np.ones(parameters["size"])
        )
        tau, num_steps, eps = 0.75, 20, 2.0
        budget = PureDPBudget(eps=eps)
        fake_randomness.forget()

        DPGD(tau=tau, num_steps=num_steps).run(
            metadata,
            input,
            budget,
            0,
        )

        scales = fake_randomness.parameters_of("laplace", "scale")
        assert scales
        assert all(
            scale == pytest.approx(2 * tau * num_steps / eps) for scale in scales
        )
        # A pure budget selects the Laplace route, never the Gaussian one.
        assert fake_randomness.draws_of("normal") == ()
        assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)

    def test_add_remove_sigma_is_half_the_paper_replace_one_sigma(self) -> None:
        # Add/remove sensitivity sqrt(2)*tau at rho/T gives sigma = tau*sqrt(T/rho),
        # exactly half the paper/reference replace-one sigma 2*tau*sqrt(T/rho).
        tau, num_steps, rho = 0.75, 80, 0.5
        per_step = ZCDPBudget(rho=rho / num_steps)
        sigma = accounting.calibrate_gaussian_scale(per_step, math.sqrt(2) * tau)
        assert sigma == pytest.approx(tau * math.sqrt(num_steps / rho))
        assert sigma == pytest.approx(0.5 * (2 * tau * math.sqrt(num_steps / rho)))

    def test_typed_step_budget_composes_through_native_zcdp_path(
        self, accounting_spy: AccountingSpy
    ) -> None:
        metadata, input, _ = _unit_box_instance(0)
        tau, num_steps, rho = 0.75, 20, 0.5
        DPGD(tau=tau, num_steps=num_steps).run(metadata, input, ZCDPBudget(rho=rho), 0)
        per_step = ZCDPBudget(rho=rho / num_steps)
        assert num_steps * per_step.rho == pytest.approx(rho)
        # Native path: pass the offered ZCDPBudget through the seam, then
        # calibrate one Gaussian step at the add/remove sensitivity sqrt(2)*tau.
        assert len(accounting_spy.calls) == 2
        spend, calibration = accounting_spy.calls
        assert spend == AccountingCall("spend_as_zcdp", ZCDPBudget(rho=rho))
        assert calibration.entry_point == "calibrate_gaussian_scale"
        assert_budget_close(calibration.budget, per_step)
        assert calibration.sensitivity == pytest.approx(math.sqrt(2) * tau)

    def test_approx_route_converts_then_calibrates_gaussian(
        self, accounting_spy: AccountingSpy
    ) -> None:
        metadata, input, _ = _unit_box_instance(0)
        offered = ApproxDPBudget(eps=1.0, delta=1e-5)
        tau, num_steps = 0.75, 20
        DPGD(tau=tau, num_steps=num_steps).run(metadata, input, offered, 0)
        # Approx offer: the seam converts it to zCDP, then one Gaussian step is
        # calibrated at the add/remove sensitivity sqrt(2)*tau. The accounting
        # suite owns the conversion's numeric result so this test remains valid
        # when spend_as_zcdp gains a tighter conversion.
        assert len(accounting_spy.calls) == 2
        assert accounting_spy.calls[0] == AccountingCall("spend_as_zcdp", offered)
        calibration = accounting_spy.calls[1]
        assert calibration.entry_point == "calibrate_gaussian_scale"
        assert isinstance(calibration.budget, ZCDPBudget)
        assert calibration.budget.rho > 0
        assert calibration.sensitivity == pytest.approx(math.sqrt(2) * tau)


class TestExecution:
    def test_two_step_trajectory_and_tail_release(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        gradients = iter((np.array([3.0, 4.0]), np.array([0.0, 12.0])))

        def fixed_gradient_sum(
            x: Array1D,
            y: Array1D,
            predictions: Array1D,
            tau: float,
        ) -> Array1D:
            del x, y, predictions, tau
            return next(gradients)

        monkeypatch.setattr(
            "dpstats.tasks.simple_linear_regression.algorithms."
            "dpgd._clipped_gradient_sum",
            fixed_gradient_sum,
        )
        fake_randomness.script("normal", zero_draw)
        input = SimpleLinearRegressionInput(
            x=np.array([0.25, 0.75]),
            y=np.array([0.0, 1.0]),
        )
        output = DPGD(num_steps=2).run(
            _metadata(2),
            input,
            ZCDPBudget(rho=0.5),
            19,
        )

        # The first cumulative norm is 5; after the orthogonal length-12
        # gradient, it is 13. For T=2 the release is the second post-update
        # iterate, not the initial or first iterate.
        expected_p25 = -0.1
        expected_p75 = -0.3 - 12.0 / 13.0
        expected_slope = 2 * (expected_p75 - expected_p25)
        expected_intercept = 1.5 * expected_p25 - 0.5 * expected_p75
        assert output.slope == pytest.approx(expected_slope)
        assert output.intercept == pytest.approx(expected_intercept)
        assert expected_p25 < 0.0 and expected_p75 < 0.0

    def test_budget_routes_share_the_deterministic_trajectory(
        self, fake_randomness: FakeRandomness
    ) -> None:
        fake_randomness.script("laplace", zero_draw)
        fake_randomness.script("normal", zero_draw)
        metadata = _metadata(2)
        input = SimpleLinearRegressionInput(
            x=np.array([0.25, 0.75]),
            y=np.array([0.0, 1.0]),
        )
        algorithm = DPGD(tau=0.25, num_steps=2)

        pure = algorithm.run(metadata, input, PureDPBudget(eps=1.0), 0)
        zcdp = algorithm.run(metadata, input, ZCDPBudget(rho=0.5), 0)

        assert pure.slope == pytest.approx(zcdp.slope)
        assert pure.intercept == pytest.approx(zcdp.intercept)

    def test_high_budget_follows_objective_descent_direction(self) -> None:
        x = np.linspace(0.0, 1.0, 200)
        y = 0.4 * x + 0.2
        metadata = _metadata(len(x))
        input = SimpleLinearRegressionInput(x=x, y=y)
        output = DPGD().run(metadata, input, ZCDPBudget(rho=1e12), 0)
        initial_error = np.linalg.norm(np.array([0.0, 0.5]) - [0.4, 0.2])
        final_error = np.linalg.norm(
            np.array([output.slope, output.intercept]) - [0.4, 0.2]
        )
        assert final_error < initial_error

    def test_recovers_line_on_real_scale_domain(self) -> None:
        # A bounded but non-[0,1] domain: the affine resolver normalizes, fits,
        # and back-maps. A near-noiseless line at high budget is recovered well.
        slope_src, intercept_src = 0.0009, 32.5  # y in [32.5, 77.5] over x in [0, 5e4]
        x = np.linspace(0.0, 50000.0, 400)
        y = slope_src * x + intercept_src
        metadata = SimpleLinearRegressionMetadata(
            n_samples=len(x),
            x_domain=BoxDomain(bounds=np.array([[0.0, 50000.0]])),
            y_domain=BoxDomain(bounds=np.array([[20.0, 90.0]])),
        )
        input = SimpleLinearRegressionInput(x=x, y=y)
        output = DPGD(num_steps=160).run(metadata, input, ZCDPBudget(rho=1e12), 0)
        # Predicted responses track the true line to a small fraction of y-range.
        y_range = 90.0 - 20.0
        predicted = output.slope * x + output.intercept
        assert np.max(np.abs(predicted - y)) < 0.05 * y_range
        assert output.intercept == pytest.approx(intercept_src, abs=0.05 * y_range)


class TestScopeRefusalAndValidation:
    @pytest.mark.parametrize(
        "metadata,message",
        [
            (
                SimpleLinearRegressionMetadata(
                    n_samples=4,
                    x_domain=EuclideanSpace(dim=1),
                    y_domain=EuclideanSpace(dim=1),
                ),
                "The x interval requires a sound L2 enclosing ball",
            ),
            (
                SimpleLinearRegressionMetadata(
                    n_samples=4,
                    x_domain=BoxDomain(bounds=np.array([[5.0, 5.0]])),
                    y_domain=UNIT_DOMAIN,
                ),
                "x_domain must have positive width",
            ),
            (
                SimpleLinearRegressionMetadata(
                    n_samples=4,
                    x_domain=BoxDomain(bounds=np.array([[0.0, np.inf]])),
                    y_domain=UNIT_DOMAIN,
                ),
                "The x interval requires a sound L2 enclosing ball",
            ),
        ],
        ids=["missing", "degenerate", "non_finite"],
    )
    def test_domain_scope_refusal(
        self, metadata: SimpleLinearRegressionMetadata, message: str
    ) -> None:
        input = SimpleLinearRegressionInput(x=np.linspace(0, 1, 4), y=np.ones(4))
        with pytest.raises(InvalidRegime, match=message):
            DPGD(num_steps=20).run(metadata, input, ZCDPBudget(rho=0.5), 0)

    def test_odd_step_count_is_rejected(self) -> None:
        # The tail release averages the second half of the iterates.
        with pytest.raises(ValueError, match="num_steps must be even"):
            DPGD(num_steps=3)
