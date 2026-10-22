from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.exceptions import NoRelease
from dpstats.budgets import ApproxDPBudget, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.tasks.linear_regression.algorithms.efficient_ptr import (
    EfficientPTR,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close


def _identity_problem(
    repeats: int,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    X = np.tile(np.eye(2), (repeats, 1))
    y = X @ np.array([0.5, -0.25])
    return (
        LinearRegressionMetadata(
            n_samples=X.shape[0],
            design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
            response_domain=EuclideanSpace(dim=1),
        ),
        LinearRegressionInput(X=X, y=y),
    )


def test_accounted_ptr_bottom_is_no_release() -> None:
    metadata, input = _identity_problem(repeats=2)

    with pytest.raises(NoRelease, match="accounted PTR bottom"):
        EfficientPTR().run(
            metadata,
            input,
            ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )


def test_safe_problem_passes_the_test_and_releases_near_ols() -> None:
    metadata, input = _identity_problem(repeats=100)
    budget = ApproxDPBudget(eps=10.0, delta=0.1)

    output = EfficientPTR().run(metadata, input, budget, seed=7)

    # The Gaussian scale at sensitivity 4/(n*0.4)=0.05 is a few hundredths.
    np.testing.assert_allclose(output.beta, [0.5, -0.25], atol=0.25)


def test_gaussian_release_uses_paper_sensitivity_and_subbudget(
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _identity_problem(repeats=100)
    budget = ApproxDPBudget(eps=10.0, delta=0.1)

    EfficientPTR().run(metadata, input, budget, seed=7)

    assert len(accounting_spy.calls) == 1
    call = accounting_spy.calls[0]
    assert call.entry_point == "calibrate_gaussian_scale"
    assert_budget_close(call.budget, ApproxDPBudget(eps=5.0, delta=0.1))
    assert call.sensitivity == pytest.approx(4.0 / (200 * 0.4))


def test_exact_ols_is_projected_before_efficient_ptr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dpstats.tasks.linear_regression.algorithms.efficient_ptr as eptr

    metadata, input = _identity_problem(repeats=2)
    input = LinearRegressionInput(X=input.X, y=np.array([10.0, 0.0, 10.0, 0.0]))
    recorded_theta: list[np.ndarray] = []

    def record_release(
        theta_hat: np.ndarray,
        _gamma: float,
        _alpha: float,
        _budget: ApproxDPBudget,
        _seed: int,
    ) -> np.ndarray:
        recorded_theta.append(theta_hat)
        return theta_hat

    monkeypatch.setattr(eptr, "_run_algorithm_2", record_release)
    output = EfficientPTR(coefficient_radius=1.0).run(
        metadata,
        input,
        ApproxDPBudget(eps=1.0, delta=1e-5),
        seed=0,
    )

    assert len(recorded_theta) == 1
    assert np.linalg.norm(recorded_theta[0]) == pytest.approx(1.0)
    np.testing.assert_array_equal(output.beta, recorded_theta[0])


def test_singular_design_fails_instead_of_releasing_accounted_bottom() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.ones((4, 2)) / np.sqrt(2.0), y=np.ones(4))

    with pytest.raises(np.linalg.LinAlgError):
        EfficientPTR().run(
            metadata,
            input,
            ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )


def test_boundary_safety_score_releases_at_probability_half_at_every_eps() -> None:
    # Five one-dimensional unit rows with curvature_lower_bound=0.2 give
    # gamma == threshold exactly (both 1.0), so 0.5*eps*(gamma-threshold) is
    # 0.5*inf*0.0 = nan without the exact-zero-safe product, which makes the
    # release-probability comparison always false and the algorithm always
    # release. The finite limit (and the true value at every eps, since the
    # margin is exactly 0 for any eps once delta=1) is release probability 1/2.
    X = np.ones((5, 1))
    y = np.ones(5)
    metadata = LinearRegressionMetadata(
        n_samples=5,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=X, y=y)
    algorithm = EfficientPTR(curvature_lower_bound=0.2)

    def release_frequency(budget: ApproxDPBudget, num_seeds: int) -> float:
        released = 0
        for seed in range(num_seeds):
            try:
                algorithm.run(metadata, input, budget, seed=seed)
                released += 1
            except NoRelease:
                pass
        return released / num_seeds

    num_seeds = 2000
    unconstrained_frequency = release_frequency(
        ApproxDPBudget(eps=math.inf, delta=1.0), num_seeds
    )
    finite_frequency = release_frequency(ApproxDPBudget(eps=1e6, delta=1.0), num_seeds)

    # Binomial standard error at p=0.5, n=2000 is ~0.0112; a 5-sigma band.
    tolerance = 5.0 * math.sqrt(0.5 * 0.5 / num_seeds)
    assert unconstrained_frequency == pytest.approx(0.5, abs=tolerance)
    assert finite_frequency == pytest.approx(0.5, abs=tolerance)


def test_off_boundary_finite_eps_releases_are_unchanged() -> None:
    # Off the exact-zero margin, the release probability is unaffected by the
    # exact-zero-safe rewrite: a strictly positive margin releases with
    # probability near 1, and a strictly negative one with probability near 0.
    X = np.ones((5, 1))
    y = np.ones(5)
    metadata = LinearRegressionMetadata(
        n_samples=5,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=X, y=y)
    safe = EfficientPTR(curvature_lower_bound=0.1)
    unsafe = EfficientPTR(curvature_lower_bound=0.5)
    budget = ApproxDPBudget(eps=1e6, delta=1.0)

    def release_frequency(
        algorithm: EfficientPTR, budget: ApproxDPBudget, num_seeds: int
    ) -> float:
        released = 0
        for seed in range(num_seeds):
            try:
                algorithm.run(metadata, input, budget, seed=seed)
                released += 1
            except NoRelease:
                pass
        return released / num_seeds

    assert release_frequency(safe, budget, 50) > 0.9
    assert release_frequency(unsafe, budget, 50) < 0.1


def test_non_approximate_budget_is_rejected() -> None:
    metadata, input = _identity_problem(repeats=2)

    with pytest.raises(InvalidRegime, match="Cannot spend"):
        EfficientPTR().run(
            metadata,
            input,
            PureDPBudget(eps=1.0),
            seed=0,
        )
