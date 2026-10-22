"""Tests for Wang's corrected adaptive one-posterior-sample estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.domains import BoxDomain, EuclideanSpace
from dpstats.budgets import ApproxDPBudget, PureDPBudget, ZCDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.tasks.linear_regression.algorithms import (
    adaops as adaops_module,
)
from dpstats.tasks.linear_regression.algorithms.adaops import AdaOPS
from dpstats.tasks.linear_regression.algorithms._ops import (
    solve_ops_scaled_temperature,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close

minimize_equation_7 = (
    adaops_module._minimize_equation_7  # pyright: ignore[reportPrivateUsage]
)


def _bounded_instance(
    seed: int = 0,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.5, 0.5, size=(120, 3))
    y = np.clip(
        X @ np.array([0.3, -0.2, 0.1]) + rng.normal(0.0, 0.05, size=120),
        -1.0,
        1.0,
    )
    metadata = LinearRegressionMetadata(
        n_samples=120,
        design_domain=BoxDomain(bounds=np.tile(np.array([[-1.0, 1.0]]), (3, 1))),
        response_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    return metadata, LinearRegressionInput(X=X, y=y)


def test_failure_probability_changes_the_public_run() -> None:
    metadata, input = _bounded_instance()
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    strict = AdaOPS(failure_probability=0.01).run(metadata, input, budget, 19)
    loose = AdaOPS(failure_probability=0.2).run(metadata, input, budget, 19)
    assert not np.allclose(strict.beta, loose.beta)


def test_failure_probability_may_be_smaller_than_privacy_delta() -> None:
    metadata, input = _bounded_instance()
    budget = ApproxDPBudget(eps=1.0, delta=1e-2)
    result = AdaOPS(failure_probability=1e-4).run(metadata, input, budget, 19)
    assert np.all(np.isfinite(result.beta))


def test_requires_public_covariate_and_response_bounds() -> None:
    metadata, input = _bounded_instance()
    no_x_bound = LinearRegressionMetadata(
        n_samples=metadata.n_samples,
        design_domain=EuclideanSpace(dim=metadata.dim),
        response_domain=metadata.response_domain,
    )
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    with pytest.raises(InvalidRegime, match="finite covariate and response bounds"):
        AdaOPS().run(no_x_bound, input, budget, 0)

    no_y_bound = LinearRegressionMetadata(
        n_samples=metadata.n_samples,
        design_domain=metadata.design_domain,
        response_domain=EuclideanSpace(dim=1),
    )
    with pytest.raises(InvalidRegime, match="finite covariate and response bounds"):
        AdaOPS().run(no_y_bound, input, budget, 0)


def test_exact_pdp_quadratic_exhausts_ops_epsilon() -> None:
    h = 40.0
    radius = 1.5
    budget = ApproxDPBudget(eps=0.5, delta=2e-6)
    scaled_sqrt_gamma = solve_ops_scaled_temperature(h, radius, budget)
    log_ops = np.log(2.0 / budget.delta)
    achieved = (
        scaled_sqrt_gamma * np.sqrt(log_ops / h)
        + scaled_sqrt_gamma**2 / (2.0 * (h + radius**2))
        + radius**2 * (1.0 + log_ops) / (2.0 * h)
    )
    assert achieved == pytest.approx(budget.eps)


def test_equation_7_obeys_safe_curvature_lower_bound() -> None:
    eps = 1.0
    delta = 1e-5
    radius = 2.0
    lower_eigenvalue = 0.3
    log_ops = np.log(10.0 / delta)
    required_h = 2.0 * radius**2 * (1.0 + log_ops) / eps
    regularization = minimize_equation_7(
        lower_eigenvalue,
        radius,
        1.0,
        2.0,
        required_h,
    )
    assert regularization > 0.0
    assert lower_eigenvalue + regularization >= required_h


def test_equation_7_can_select_zero_regularization() -> None:
    regularization = minimize_equation_7(
        lambda_min_tilde=1_000.0,
        feature_radius=1.0,
        c1=1e-12,
        c2=1.0,
        minimum_regularized_curvature=1.0,
    )
    assert regularization == 0.0


@pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
def test_refuses_non_approx_budget(
    accounting_spy: AccountingSpy,
    budget: PureDPBudget | ZCDPBudget,
) -> None:
    metadata, input = _bounded_instance()
    with pytest.raises(InvalidRegime, match="Cannot spend"):
        AdaOPS().run(metadata, input, budget, 0)
    assert accounting_spy.calls == ()


def test_accounting_split_and_domain_exactness(
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _bounded_instance()
    offered = ApproxDPBudget(eps=2.0, delta=1e-5)
    release = ApproxDPBudget(eps=offered.eps / 4.0, delta=offered.delta / 5.0)

    AdaOPS().run(metadata, input, offered, 4)

    x_bound = metadata.design_domain.l2_norm_bound
    assert x_bound == pytest.approx(math.sqrt(3.0))
    assert len(accounting_spy.calls) == 2
    first, second = accounting_spy.calls
    assert first.entry_point == second.entry_point == "calibrate_gaussian_scale"
    assert_budget_close(first.budget, release)
    assert_budget_close(second.budget, release)
    assert first.sensitivity == pytest.approx(x_bound**2)
    assert second.sensitivity == pytest.approx(1.0)
    assert 2.0 * release.eps + offered.eps / 2.0 == pytest.approx(offered.eps)
    assert 5.0 * release.delta == pytest.approx(offered.delta)
