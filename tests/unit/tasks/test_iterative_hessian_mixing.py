"""Focused tests for Lev et al.'s Iterative Hessian Mixing estimator."""

from __future__ import annotations

from typing import Any

import math

import numpy as np
import pytest

import dpstats.budgets.accounting as accounting
import dpstats.tasks.linear_regression.algorithms.iterative_hessian_mixing as ihm
from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.tasks.linear_regression.algorithms.iterative_hessian_mixing import (
    IterativeHessianMixing,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw


def _instance(
    *,
    n: int = 40,
    d: int = 3,
    seed: int = 0,
    x_bound: float = 2.0,
    y_bound: float = 3.0,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)
    X = X * (x_bound / 2.0)
    beta = rng.normal(size=d)
    y = np.clip(X @ beta + 0.1 * rng.normal(size=n), -y_bound, y_bound)
    metadata = LinearRegressionMetadata(
        n_samples=n,
        design_domain=LpBallDomain.at_origin(p=2, radius=x_bound, dim=d),
        response_domain=LpBallDomain.at_origin(p=2, radius=y_bound, dim=1),
    )
    return metadata, LinearRegressionInput(X=X, y=y)


class _NoSampleCountRead:
    """Metadata proxy that makes any sample-count access observable."""

    def __init__(self, metadata: LinearRegressionMetadata) -> None:
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        if name == "n_samples":
            raise AssertionError("IterativeHessianMixing must not read n_samples.")
        return getattr(self._metadata, name)


def _fixed_calibration(
    monkeypatch: pytest.MonkeyPatch, gamma: float, noise_multiplier: float
) -> list[int]:
    """Pin both noise calibrations and report the sketch sizes they were given."""
    sketch_sizes: list[int] = []

    def calibrate_sketch(
        eps: float, delta: float, sketch_size: int, num_iterations: int
    ) -> float:
        sketch_sizes.append(sketch_size)
        return gamma

    monkeypatch.setattr(ihm, "_calibrate_mixing_branch_gamma_cached", calibrate_sketch)
    monkeypatch.setattr(
        accounting,
        "compute_noise_multiplier_for_composed_gaussian",
        lambda budget, num_steps: noise_multiplier,
    )
    return sketch_sizes


class TestIterativeHessianMixingAccounting:
    """Pin the paper-specific forward and inverse accounting boundaries."""

    def test_calibration_reaches_budget_from_conservative_side(self) -> None:
        budget = ApproxDPBudget(eps=1.0, delta=7.5e-6)
        gamma = ihm._calibrate_mixing_branch_gamma_cached(
            budget.eps, budget.delta, sketch_size=30, num_iterations=3
        )
        achieved = ihm._epsilon_for_mixing_branch(
            gamma,
            sketch_size=30,
            num_iterations=3,
            delta=budget.delta / 3.0,
        )

        assert achieved <= budget.eps
        assert achieved == pytest.approx(budget.eps, rel=1e-10)

    def test_more_privacy_or_more_sketches_requires_more_mixing_noise(self) -> None:
        strict = ihm._calibrate_mixing_branch_gamma_cached(
            0.5, 7.5e-6, sketch_size=30, num_iterations=3
        )
        loose = ihm._calibrate_mixing_branch_gamma_cached(
            2.0, 7.5e-6, sketch_size=30, num_iterations=3
        )
        more_iterations = ihm._calibrate_mixing_branch_gamma_cached(
            2.0, 7.5e-6, sketch_size=30, num_iterations=6
        )

        assert strict > loose
        assert more_iterations > loose


class TestUnconstrainedRelease:
    """An unconstrained offer of any family releases through ordinary calibration."""

    def test_matches_a_constrained_large_eps_release_under_the_same_seed(
        self,
    ) -> None:
        # d=8 keeps the sketch size k = ceil(6*max(d, log(...))) at 6*d for
        # both offers below, so only the noise scale can differ, not k.
        metadata, input = _instance(seed=4, d=8)

        no_privacy_output = IterativeHessianMixing().run(
            metadata, input, NO_PRIVACY, seed=8
        )
        reference = IterativeHessianMixing().run(
            metadata, input, ApproxDPBudget(eps=1e8, delta=0.25), seed=8
        )

        np.testing.assert_allclose(no_privacy_output.beta, reference.beta, atol=1e-3)

    def test_gamma_matches_a_constrained_large_eps_calibration(self) -> None:
        k, num_iterations = 48, 3
        unconstrained_gamma = ihm._calibrate_mixing_branch_gamma_cached(
            math.inf, 0.75, k, num_iterations
        )
        constrained_gamma = ihm._calibrate_mixing_branch_gamma_cached(
            1e8, 0.1875, k, num_iterations
        )

        assert unconstrained_gamma == constrained_gamma
        assert unconstrained_gamma > 2.5


class TestAlgorithm2:
    def test_feature_bound_has_the_paper_scaling_equivariance(self) -> None:
        X = np.array(
            [
                [0.5, 0.0],
                [0.0, 0.25],
                [0.25, 0.25],
            ]
        )
        C_X = 3.0

        normalized = ihm._run_algorithm_2(
            X,
            1.0,
            gamma=8.0,
            tau=2.0,
            eta=1.0,
            seed=7,
        )
        scaled = ihm._run_algorithm_2(
            C_X * X,
            C_X,
            gamma=8.0,
            tau=2.0,
            eta=1.0,
            seed=7,
        )

        assert scaled == pytest.approx(C_X * normalized)


class TestIterativeHessianMixing:
    def test_neutral_row_preserves_the_complete_release(self) -> None:
        metadata, input = _instance(seed=4)
        padded_metadata = metadata.model_copy(
            update={"n_samples": metadata.n_samples + 1}
        )
        padded_input = LinearRegressionInput(
            X=np.vstack((input.X, np.zeros((1, metadata.dim)))),
            y=np.concatenate((input.y, np.zeros(1))),
        )
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)

        original = IterativeHessianMixing().run(metadata, input, budget, seed=8)
        padded = IterativeHessianMixing().run(
            padded_metadata, padded_input, budget, seed=8
        )

        np.testing.assert_allclose(original.beta, padded.beta)

    def test_public_reference_n_is_not_read(self) -> None:
        metadata, input = _instance(seed=4)
        spied_metadata = _NoSampleCountRead(metadata)

        output = IterativeHessianMixing().run(
            spied_metadata, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=8
        )

        # The sketch size, the clip, and both noise scales are functions of the
        # dimension, the domains, and the budget alone, so a complete release
        # never consults the declared sample count.
        assert np.all(np.isfinite(output.beta))

    def test_feature_rescaling_maps_coefficients_back(self) -> None:
        metadata, input = _instance(seed=2)
        scaled_metadata = metadata.model_copy(
            update={
                "design_domain": LpBallDomain.at_origin(
                    p=2, radius=4.0, dim=metadata.dim
                )
            },
        )
        scaled_input = LinearRegressionInput(X=2.0 * input.X, y=input.y)
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)

        original = IterativeHessianMixing().run(metadata, input, budget, seed=3)
        scaled = IterativeHessianMixing().run(
            scaled_metadata, scaled_input, budget, seed=3
        )

        assert np.allclose(scaled.beta, original.beta / 2.0)
        assert np.allclose(scaled_input.X @ scaled.beta, input.X @ original.beta)

    def test_sampled_sketch_law_matches_the_literal_construction(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        metadata, input = _instance(seed=1)
        _fixed_calibration(monkeypatch, gamma=8.0, noise_multiplier=0.0)
        fake_randomness.forget()

        IterativeHessianMixing().run(
            metadata, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )

        X = input.X / 2.0
        gram = X.T @ X
        sampled_covariance = fake_randomness.parameters_of(
            "multivariate_normal", "cov"
        )[0]
        eta_squared = sampled_covariance[0, 0] - gram[0, 0]
        assert eta_squared > 0.0

        # Literal Algorithm 1 sketch rows: S X + eta Xi with standard normal
        # S and Xi. Their second moment estimates the covariance the
        # implementation samples from directly.
        draws = 40_000
        source = fake_randomness.real_default_rng(11)
        S = source.normal(size=(draws, metadata.n_samples))
        Xi = source.normal(size=(draws, metadata.dim))
        literal_rows = S @ X + math.sqrt(eta_squared) * Xi
        empirical = literal_rows.T @ literal_rows / draws

        # Well above the Monte Carlo standard error of the mean and well below
        # the eta**2 ridge that separates the two candidate laws.
        tolerance = 0.05 * float(np.max(np.abs(sampled_covariance)))
        assert np.allclose(empirical, sampled_covariance, atol=tolerance)
        assert not np.allclose(empirical, gram, atol=tolerance)

    def test_unconstrained_release_keeps_the_source_floor_mixing_noise(self) -> None:
        # An unconstrained budget draws no gradient noise, but the
        # Appendix C.1.1 mixing calibration bottoms out at its own floor
        # gamma -> 5/2 rather than gamma -> 0: it is not the ordinary
        # least-squares fit (see TestUnconstrainedRelease for the exact
        # (eps=inf, delta=1) versus large-finite-eps equivalence).
        metadata, input = _instance(seed=5)
        least_squares = np.linalg.lstsq(input.X, input.y, rcond=None)[0]

        output = IterativeHessianMixing(num_iterations=30).run(
            metadata, input, ApproxDPBudget(eps=math.inf, delta=1e-5), seed=0
        )

        assert np.max(np.abs(output.beta - least_squares)) > 0.1

    def test_clipped_residuals_move_the_fixed_point_off_least_squares(self) -> None:
        # Four rows (x=1, y=1) and one row (x=-1, y=1), all at the unit bounds.
        # Least squares gives theta = sum(x*y)/sum(x**2) = 3/5. With residuals
        # clipped to C=1 the fixed point solves 4*(1 - theta) - clip(1 + theta)
        # = 0; for theta > 0 the last residual exceeds C, so theta = 3/4.
        metadata = LinearRegressionMetadata(
            n_samples=5,
            design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
            response_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        )
        input = LinearRegressionInput(
            X=np.array([[1.0], [1.0], [1.0], [1.0], [-1.0]]), y=np.ones(5)
        )

        output = IterativeHessianMixing(num_iterations=30).run(
            metadata, input, ApproxDPBudget(eps=math.inf, delta=1e-5), seed=0
        )

        assert output.beta == pytest.approx(np.array([0.75]))

    def test_accounting_boundaries_receive_their_complete_subbudgets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metadata, input = _instance(seed=5)
        budget = ApproxDPBudget(eps=2.0, delta=8e-6)
        sketch_calls: list[tuple[ApproxDPBudget, int, int]] = []
        gradient_calls: list[tuple[ApproxDPBudget, int]] = []

        def record_sketch(
            eps: float, delta: float, sketch_size: int, num_iterations: int
        ) -> float:
            sketch_calls.append(
                (ApproxDPBudget(eps=eps, delta=delta), sketch_size, num_iterations)
            )
            return 5.0

        def record_gradient(received: ApproxDPBudget, num_steps: int) -> float:
            gradient_calls.append((received, num_steps))
            return 1.0

        monkeypatch.setattr(ihm, "_calibrate_mixing_branch_gamma_cached", record_sketch)
        monkeypatch.setattr(
            accounting,
            "compute_noise_multiplier_for_composed_gaussian",
            record_gradient,
        )

        IterativeHessianMixing().run(metadata, input, budget, seed=0)

        [(sketch_budget, sketch_size, sketch_steps)] = sketch_calls
        [(gradient_budget, gradient_steps)] = gradient_calls
        assert_budget_close(sketch_budget, ApproxDPBudget(eps=1.0, delta=6e-6))
        assert_budget_close(gradient_budget, ApproxDPBudget(eps=1.0, delta=2e-6))
        assert sketch_size >= metadata.dim
        assert sketch_steps == gradient_steps == 3

    @pytest.mark.parametrize(
        ("dim", "delta", "expected_sketch_size"),
        [
            # k = 6 * max(64, log(4*3/1e-3)) = 6 * 64, the dimension branch.
            (64, 1e-2, 384),
            # k = ceil(6 * max(2, log(4*3/1e-10))) = ceil(6 * 25.5106...),
            # the failure-budget branch, rounded up.
            (2, 1e-9, 154),
        ],
    )
    def test_sketch_size_realizes_the_public_closed_form(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_randomness: FakeRandomness,
        dim: int,
        delta: float,
        expected_sketch_size: int,
    ) -> None:
        metadata, input = _instance(n=80, d=dim, seed=7)
        sketch_sizes = _fixed_calibration(monkeypatch, gamma=1.0, noise_multiplier=0.0)
        fake_randomness.forget()

        IterativeHessianMixing().run(
            metadata, input, ApproxDPBudget(eps=1.0, delta=delta), seed=0
        )

        sketches = fake_randomness.values_of("multivariate_normal")
        assert sketches[0].shape == (expected_sketch_size, dim)
        assert sketch_sizes == [expected_sketch_size]

    def test_eigenvalue_threshold_realizes_the_paper_tau(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        # An orthogonal design with X'X = 0.8 I has lambda_min = 0.8 exactly.
        row = math.sqrt(0.4)
        metadata = LinearRegressionMetadata(
            n_samples=4,
            design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
            response_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        )
        input = LinearRegressionInput(
            X=row * np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
            y=np.zeros(4),
        )
        gamma = 1.0
        _fixed_calibration(monkeypatch, gamma=gamma, noise_multiplier=0.0)
        fake_randomness.script("normal", zero_draw)
        fake_randomness.forget()

        IterativeHessianMixing(num_iterations=1).run(
            metadata, input, ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )

        # The mixing calibration and the iteration loop are separately
        # accounted releases, so neither may reuse the parent seed's stream.
        assert len(set(fake_randomness.seeds)) == 2
        assert 0 not in fake_randomness.seeds
        # Algorithm 2's threshold at delta = 1e-5 and varrho = delta / 10:
        # tau = sqrt(2 log(max(4/delta, 4/varrho))) = sqrt(2 log(4e6)).
        tau = math.sqrt(2.0 * math.log(4e6))
        (covariance,) = fake_randomness.parameters_of("multivariate_normal", "cov")
        (sketch,) = fake_randomness.values_of("multivariate_normal")
        eigen_noise = gamma / math.sqrt(sketch.shape[0])
        # With a zero eigenvalue draw the private lower bound is
        # 0.8 - eigen_noise * tau > 0, and the mixing noise tops it up to gamma.
        lambda_tilde = 0.8 - eigen_noise * tau
        assert 0.0 < lambda_tilde < gamma
        np.testing.assert_allclose(
            covariance, (0.8 + gamma - lambda_tilde) * np.eye(2), rtol=1e-12
        )

    def test_singular_solve_propagates_instead_of_a_fallback_release(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metadata, _ = _instance(n=4, d=2)
        input = LinearRegressionInput(X=np.zeros((4, 2)), y=np.zeros(4))
        monkeypatch.setattr(
            ihm,
            "_calibrate_mixing_branch_gamma_cached",
            lambda eps, delta, sketch_size, num_iterations: 0.0,
        )
        monkeypatch.setattr(
            accounting,
            "compute_noise_multiplier_for_composed_gaussian",
            lambda budget, num_steps: 0.0,
        )

        with pytest.raises(np.linalg.LinAlgError):
            IterativeHessianMixing().run(
                metadata,
                input,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    @pytest.mark.parametrize(
        "budget",
        [PureDPBudget(eps=1.0), ZCDPBudget(rho=0.5)],
    )
    def test_unsupported_budget_refused_before_domain_resolution(
        self, budget: PureDPBudget | ZCDPBudget
    ) -> None:
        metadata = LinearRegressionMetadata(
            n_samples=2,
            design_domain=EuclideanSpace(dim=2),
            response_domain=EuclideanSpace(dim=1),
        )
        input = LinearRegressionInput(X=np.zeros((2, 2)), y=np.zeros(2))

        with pytest.raises(InvalidRegime, match="Cannot spend"):
            IterativeHessianMixing().run(metadata, input, budget, seed=0)
