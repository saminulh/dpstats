"""Focused tests for Chang et al.'s practical tail-robust DPGD estimator."""

from __future__ import annotations

from typing import Any

import math

import numpy as np
import pytest

import dpstats.tasks.linear_regression.algorithms.tail_robust_dpgd as trd
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.budgets.accounting import (
    compute_noise_multiplier_for_composed_gaussian,
)
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.tasks.linear_regression.algorithms.tail_robust_dpgd import (
    INITIALIZER_GRAD_TOL,
    TailRobustDPGD,
    _huber_score,
    _private_initializer,
    _run_algorithm_1,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
    assert_budget_close,
)
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw
from tests.unit.tasks.task_builders import lr_metadata


def _intercept_design(dim: int) -> ProductDomain:
    return ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=dim - 1)))


def _instance(
    n: int = 400,
    p: int = 4,
    seed: int = 0,
    has_intercept: bool = False,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(seed)
    if has_intercept:
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    else:
        X = rng.normal(size=(n, p))
    beta_true = rng.normal(size=p)
    y = X @ beta_true + rng.normal(size=n)
    design = _intercept_design(p) if has_intercept else p
    return lr_metadata(n, design=design), LinearRegressionInput(X=X, y=y)


def _noise_free_tau_0(y: np.ndarray) -> float:
    """The Section 5.1.1 scale estimate when both moment releases add zero.

    Responses are clipped to ``[-log n, log n]``; the estimate is the square
    root of the clipped responses' variance, or 2 when that is nonpositive.
    """
    log_n = math.log(y.size)
    clipped = np.clip(y, -log_n, log_n)
    variance = float(np.mean(clipped**2) - np.mean(clipped) ** 2)
    return math.sqrt(variance) if variance > 0.0 else 2.0


class TestAlgorithmOne:
    def test_noise_uses_add_remove_sensitivity(self, fake_randomness: FakeRandomness):
        """Each step's noise is the calibrated multiplier times ``gamma tau / n``."""
        fake_randomness.script("normal", lambda parameters: np.ones(parameters["size"]))
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)

        public_n = 10
        gamma = 2.0
        tau = 0.5
        learning_rate = 0.25
        beta = _run_algorithm_1(
            X=np.zeros((public_n, 2)),
            y=np.zeros(public_n),
            n=public_n,
            beta_init=np.zeros(2),
            tau=tau,
            gamma=gamma,
            learning_rate=learning_rate,
            num_steps=1,
            budget=budget,
            seed=0,
        )

        multiplier = compute_noise_multiplier_for_composed_gaussian(budget, 1)
        expected_coordinate = learning_rate * multiplier * gamma * tau / public_n
        np.testing.assert_allclose(beta, expected_coordinate)

    def test_noiseless_update_has_the_paper_sign(self):
        """``beta += eta * psi(y - x beta) x``: from 0, residual 1 gives 0.25."""
        beta = _run_algorithm_1(
            X=np.ones((1, 1)),
            y=np.ones(1),
            n=1,
            beta_init=np.zeros(1),
            tau=2.0,
            gamma=2.0,
            learning_rate=0.25,
            num_steps=1,
            budget=NO_PRIVACY,  # type: ignore[arg-type]
            seed=0,
        )

        np.testing.assert_allclose(beta, np.array([0.25]))

    def test_composition_is_owned_by_algorithm_boundary(
        self, accounting_spy: AccountingSpy
    ):
        metadata, data = _instance(n=200, p=3)
        budget = ApproxDPBudget(eps=1.4, delta=2e-6)
        _run_algorithm_1(
            data.X,
            data.y,
            metadata.n_samples,
            np.zeros(metadata.dim),
            tau=1.0,
            gamma=1.0,
            learning_rate=0.2,
            num_steps=7,
            budget=budget,
            seed=0,
        )
        assert accounting_spy.calls == (
            AccountingCall(
                "compute_noise_multiplier_for_composed_gaussian", budget, num_steps=7
            ),
        )


class TestTailRobustDPGD:
    def test_initializer_checks_convergence_after_final_update(self):
        beta = trd._solve_ridge_huber(
            X_tilde=np.ones((1, 1)),
            y=np.ones(1),
            tau0=10.0,
            lambda_0=0.2,
            grad_tol=0.03,
            max_iters=1,
        )
        assert beta[0] > 0.0

    @pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=0.5)])
    def test_non_approx_budget_refused(self, budget):
        metadata, data = _instance(has_intercept=True)
        with pytest.raises(InvalidRegime, match="Cannot spend"):
            TailRobustDPGD().run(metadata, data, budget, seed=0)

    def test_missing_intercept_declaration_refused_before_input(self):
        metadata, data = _instance(has_intercept=False)
        with pytest.raises(InvalidRegime, match="intercept"):
            TailRobustDPGD().run(
                metadata,
                data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    def test_half_huber_score_convention(self):
        tau = 1.5
        residual = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
        score = _huber_score(residual, tau)
        np.testing.assert_allclose(score, [-1.5, -1.5, -0.5, 0.0, 0.5, 1.5, 1.5])

    def test_moment_scales_match_the_calibrated_sensitivities(
        self, fake_randomness: FakeRandomness
    ):
        n, p = 500, 4
        metadata = lr_metadata(n, design=_intercept_design(p))
        fake_randomness.script("laplace", zero_draw)
        fake_randomness.script("normal", zero_draw)
        init_budget = ApproxDPBudget(eps=1.0 / 6.0, delta=1e-5 / 6.0)
        _private_initializer(np.ones((n, p)), np.ones(n), metadata, init_budget, seed=0)

        log_n = math.log(n)
        eps_each = init_budget.eps / 8.0
        # Clipping confines each response to [-log_n, log_n] and each squared
        # response to [0, log_n**2]. The first moment is centered at zero and
        # the second at the midpoint of its range, so one record moves their
        # public-n means by at most log_n / n and log_n**2 / (2 * n).
        first_sensitivity = log_n / n
        second_sensitivity = log_n**2 / (2.0 * n)
        assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx(
            [first_sensitivity / eps_each, second_sensitivity / eps_each]
        )
        # The moment releases and the warm-start release are separately
        # accounted, so they never share a stream.
        assert len(set(fake_randomness.seeds)) == len(fake_randomness.seeds)
        assert 0 not in fake_randomness.seeds

    def test_initializer_output_sensitivity_and_budget(
        self, fake_randomness: FakeRandomness, accounting_spy: AccountingSpy
    ):
        """The warm start is released at ``B tau_0 / (lambda_0 n) + 2 tol / lambda_0``.

        ``B = sqrt(1 + p / 36)`` bounds the clipped design and ``lambda_0 = 0.2``;
        it spends three quarters of the initializer's epsilon.
        """
        metadata, data = _instance(n=500, p=4, seed=3, has_intercept=True)
        n, p = metadata.n_samples, metadata.dim
        init_budget = ApproxDPBudget(eps=0.5, delta=1e-5 / 6.0)
        fake_randomness.script("laplace", zero_draw)
        fake_randomness.forget()

        _private_initializer(data.X, data.y, metadata, init_budget, seed=2)

        expected_sensitivity = (
            math.sqrt(1.0 + p / 36.0) * _noise_free_tau_0(data.y) / (0.2 * n)
            + 2.0 * INITIALIZER_GRAD_TOL / 0.2
        )
        (call,) = accounting_spy.calls
        assert call.entry_point == "calibrate_gaussian_scale"
        assert_budget_close(
            call.budget,
            ApproxDPBudget(eps=0.75 * init_budget.eps, delta=init_budget.delta),
        )
        assert call.sensitivity == pytest.approx(expected_sensitivity, rel=1e-12)

    def test_tau0_fallback_when_variance_is_nonpositive(
        self, fake_randomness: FakeRandomness
    ):
        n, p = 200, 3
        metadata = lr_metadata(n, design=_intercept_design(p))
        fake_randomness.script("laplace", zero_draw)
        fake_randomness.script("normal", zero_draw)
        _, tau_0 = _private_initializer(
            np.ones((n, p)),
            np.ones(n),
            metadata,
            ApproxDPBudget(eps=1.0 / 6.0, delta=1e-5 / 6.0),
            seed=0,
        )
        assert tau_0 == pytest.approx(2.0)

    def test_intercept_column_preserved_and_others_clipped(self, monkeypatch):
        n, p = 40, 4
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1)) * 50.0])
        y = rng.normal(size=n)
        metadata = lr_metadata(n, design=_intercept_design(p))
        captured: dict[str, np.ndarray] = {}
        real = trd._solve_ridge_huber

        def spy(X_tilde, *args, **kwargs):
            captured["X_tilde"] = X_tilde.copy()
            return real(X_tilde, *args, **kwargs)

        monkeypatch.setattr(trd, "_solve_ridge_huber", spy)
        _private_initializer(
            X,
            y,
            metadata,
            ApproxDPBudget(eps=1.0 / 6.0, delta=1e-5 / 6.0),
            seed=0,
        )
        X_tilde = captured["X_tilde"]
        assert np.array_equal(X_tilde[:, 0], X[:, 0])
        assert np.all(
            np.linalg.norm(X_tilde[:, 1:], axis=1) <= math.sqrt(p) / 6.0 + 1e-9
        )

    def test_tau_uses_overall_epsilon(
        self, monkeypatch, fake_randomness: FakeRandomness
    ):
        """``tau = 0.04 tau_0 sqrt(eps n / (p + log n))`` reads the whole epsilon."""
        metadata, data = _instance(n=400, p=4, has_intercept=True)
        eps, delta = 3.0, 1e-5
        fake_randomness.script("laplace", zero_draw)
        captured: dict[str, Any] = {}
        real = trd._run_algorithm_1

        def spy(X, y, n, **kwargs):
            captured.update(kwargs)
            return real(X, y, n, **kwargs)

        monkeypatch.setattr(trd, "_run_algorithm_1", spy)
        TailRobustDPGD().run(
            metadata, data, ApproxDPBudget(eps=eps, delta=delta), seed=0
        )

        n, p = metadata.n_samples, metadata.dim
        tau_0 = _noise_free_tau_0(data.y)
        expected = 0.04 * tau_0 * math.sqrt(eps * n / (p + math.log(n)))
        main_share = 0.04 * tau_0 * math.sqrt((5.0 * eps / 6.0) * n / (p + math.log(n)))
        assert captured["tau"] == pytest.approx(expected, rel=1e-12)
        assert captured["tau"] != pytest.approx(main_share, rel=1e-6)

    def test_budget_split_sums_to_total(self, accounting_spy: AccountingSpy):
        """A sixth initializes (a quarter of it on two Laplace moments); the rest runs.

        With ``T = ceil(2 log n)`` main steps, the recorded spends and the two
        moment scales account for the whole offered budget.
        """
        metadata, data = _instance(has_intercept=True)
        eps, delta = 3.0, 1e-5

        TailRobustDPGD().run(
            metadata, data, ApproxDPBudget(eps=eps, delta=delta), seed=0
        )

        output_call, main_call = accounting_spy.calls
        assert output_call.entry_point == "calibrate_gaussian_scale"
        assert main_call.entry_point == "compute_noise_multiplier_for_composed_gaussian"
        assert main_call.num_steps == math.ceil(2.0 * math.log(metadata.n_samples))
        assert isinstance(output_call.budget, ApproxDPBudget)
        assert isinstance(main_call.budget, ApproxDPBudget)
        init_eps = output_call.budget.eps / 0.75
        assert init_eps == pytest.approx(eps / 6.0)
        assert init_eps + main_call.budget.eps == pytest.approx(eps)
        assert output_call.budget.delta + main_call.budget.delta == pytest.approx(delta)
