from __future__ import annotations

from dpstats.domains import EuclideanSpace

import math
from typing import NoReturn, cast

import numpy as np
import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.budgets import accounting
from dpstats.budgets.accounting import spend_as_gdp as _real_spend_as_gdp
from dpstats.exceptions import InvalidRegime
from dpstats.domains import BoxDomain, Domain, LpBallDomain
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms import (
    adaptive_ssp as adaptive_ssp_module,
)
from dpstats.tasks.linear_regression.algorithms.adaptive_ssp import AdaSSP
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingSpy,
    assert_budget_close,
)
from tests.fixtures.randomness_fake import (
    FakeRandomness,
    draws_in_order,
    zero_draw,
)
from tests.unit.tasks.task_builders import Unreadable


def _instance() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(72)
    design_domain = LpBallDomain.at_origin(p=2, radius=5.0, dim=3)
    response_domain = LpBallDomain.at_origin(p=2, radius=5.0, dim=1)
    X = design_domain.project(rng.normal(size=(80, 3)))
    y = X @ np.array([0.8, -0.4, 0.2]) + rng.normal(scale=0.1, size=80)
    y = response_domain.project(y[:, None])[:, 0]
    return (
        LinearRegressionMetadata(
            n_samples=80,
            design_domain=design_domain,
            response_domain=response_domain,
        ),
        LinearRegressionInput(X=X, y=y),
    )


@pytest.mark.parametrize(
    "budget",
    [
        ApproxDPBudget(eps=3.0, delta=0.03),
        ZCDPBudget(rho=3.0),
        GDPBudget(mu=3.0),
    ],
)
def test_three_releases_compose_through_gaussian_accounting(
    accounting_spy: AccountingSpy,
    budget: ApproxDPBudget | ZCDPBudget | GDPBudget,
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=BoxDomain(bounds=np.array([[-3.0, 3.0], [-3.0, 3.0]])),
        response_domain=BoxDomain(bounds=np.array([[-4.0, 4.0]])),
    )
    input_data = LinearRegressionInput(
        X=np.array([[4.0, 0.0], [0.0, 4.0], [1.0, 1.0], [-1.0, 1.0]]),
        y=np.array([3.0, -3.0, 1.0, -1.0]),
    )
    AdaSSP().run(metadata, input_data, budget=budget, seed=4)

    x_bound = math.sqrt(18.0)
    y_bound = 4.0
    assert x_bound > math.sqrt(metadata.dim)
    assert y_bound > math.sqrt(metadata.dim)
    # Exact GDP composition: the offered budget is converted to GDP once,
    # up front, and each of the three releases gets an equal third of mu.
    gdp_budget = _real_spend_as_gdp(budget)
    release_budget = gdp_budget.split((1.0, 1.0, 1.0))[0]
    assert [call.entry_point for call in accounting_spy.calls] == ["spend_as_gdp"] + [
        "calibrate_gaussian_scale"
    ] * 3
    assert_budget_close(accounting_spy.calls[0].budget, budget)
    for call in accounting_spy.calls[1:]:
        assert_budget_close(call.budget, release_budget)
    assert [call.sensitivity for call in accounting_spy.calls[1:]] == pytest.approx(
        [x_bound**2, x_bound**2, x_bound * y_bound]
    )
    assert 3 * release_budget.mu**2 == pytest.approx(gdp_budget.mu**2)


@pytest.mark.parametrize(
    "design_domain,response_domain,quantity",
    [
        (
            EuclideanSpace(dim=1),
            LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
            "covariate",
        ),
        (
            LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
            EuclideanSpace(dim=1),
            "response",
        ),
    ],
)
def test_unbounded_domain_is_rejected(
    design_domain: Domain,
    response_domain: Domain,
    quantity: str,
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=4,
        design_domain=design_domain,
        response_domain=response_domain,
    )
    input_data = LinearRegressionInput(
        X=np.array([[0.0], [0.5], [-0.5], [1.0]]),
        y=np.array([0.0, 0.5, -0.5, 1.0]),
    )

    with pytest.raises(InvalidRegime) as excinfo:
        AdaSSP().run(
            metadata,
            input_data,
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )
    assert quantity in str(excinfo.value)
    assert "finite" in str(excinfo.value)


def test_refuses_pure_budget_directly(
    accounting_spy: AccountingSpy,
) -> None:
    metadata, _ = _instance()
    unreadable = cast(LinearRegressionInput, Unreadable("Input"))
    budget = PureDPBudget(eps=1.0)
    with pytest.raises(InvalidRegime, match="PureDPBudget"):
        AdaSSP().run(metadata, unreadable, budget=budget, seed=0)
    assert [call.entry_point for call in accounting_spy.calls] == ["spend_as_gdp"]


@pytest.mark.parametrize(
    "budget", [NO_PRIVACY, ApproxDPBudget(eps=math.inf, delta=1.0)]
)
def test_unconstrained_release_matches_the_delta_one_approx_dp_offer(
    budget: DPBudget,
) -> None:
    # Every unconstrained offer, of any family, spends as (eps=inf, delta=1)
    # through ApproxDPBudget.split, so it releases exactly like an explicit
    # ApproxDPBudget(1, 1) offer, which is also unconstrained.
    metadata, input_data = _instance()
    reference = AdaSSP().run(
        metadata, input_data, budget=ApproxDPBudget(eps=1.0, delta=1.0), seed=0
    )

    output = AdaSSP().run(metadata, input_data, budget=budget, seed=0)

    np.testing.assert_allclose(output.beta, reference.beta)


def test_noisy_solve_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _instance()

    def fail_solve(_matrix: Array2D, _vector: Array1D) -> NoReturn:
        raise np.linalg.LinAlgError("private transcript is singular")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    with pytest.raises(np.linalg.LinAlgError, match="private transcript is singular"):
        AdaSSP().run(
            metadata,
            input_data,
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )


@pytest.mark.parametrize(
    "budget,failure_probability",
    [
        (ApproxDPBudget(eps=3.0, delta=0.3), 0.2),
        (ZCDPBudget(rho=3.0), 0.2),
    ],
)
def test_private_eigenvalue_tail_and_damping_reach_the_solve(
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
    budget: ApproxDPBudget | ZCDPBudget,
    failure_probability: float,
) -> None:
    """Pin the one-dimensional Algorithm 2 chain without copying the estimator."""
    eigenvalue_scale, gram_scale, cross_product_scale = 2.0, 3.0, 4.0
    scales = iter((eigenvalue_scale, gram_scale, cross_product_scale))
    monkeypatch.setattr(
        adaptive_ssp_module.accounting,
        "calibrate_gaussian_scale",
        lambda budget, sensitivity: next(scales),
    )
    # Algorithm 2 draws the eigenvalue perturbation, the Gram noise and the
    # cross-product noise, in that order.
    eigenvalue_draw = 0.0
    gram_draw = np.array([[1.0]])
    cross_product_draw = np.array([-1.0])
    fake_randomness.script(
        "normal",
        draws_in_order((eigenvalue_draw, gram_draw, cross_product_draw)),
    )

    solve_args: tuple[Array2D, Array1D] | None = None

    def record_solve(matrix: Array2D, vector: Array1D) -> Array1D:
        nonlocal solve_args
        solve_args = matrix, vector
        return np.array([42.0])

    monkeypatch.setattr(np.linalg, "solve", record_solve)
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=LpBallDomain.at_origin(p=2, radius=10.0, dim=1),
        response_domain=LpBallDomain.at_origin(p=2, radius=10.0, dim=1),
    )
    input_data = LinearRegressionInput(
        X=np.array([[1.0], [2.0]]),
        y=np.array([1.0, 2.0]),
    )

    output = AdaSSP(failure_probability=failure_probability).run(
        metadata,
        input_data,
        budget=budget,
        seed=0,
    )

    assert solve_args is not None
    matrix, vector = solve_args
    assert fake_randomness.parameters_of("normal", "size") == [None, (1, 1), 1]

    # X.T @ X and X.T @ y for X = [[1], [2]], y = [1, 2].
    gram = np.array([[5.0]])
    cross_product = np.array([5.0])
    symmetric_noise = gram_scale * gram_draw

    # Algorithm 2 solves (X.T X + E + lambda I) beta = X.T y + noise, so what
    # remains after removing the Gram release is exactly the adaptive ridge.
    damping = float((matrix - gram - symmetric_noise)[0, 0])
    # Wang's Appendix B stability condition, which the damping exists to
    # enforce for the realized Gram noise.
    assert damping >= 2.0 * float(np.linalg.norm(symmetric_noise, 2))
    assert float(np.linalg.eigvalsh(matrix)[0]) > 0.0

    assert vector == pytest.approx(
        cross_product + cross_product_scale * cross_product_draw
    )
    assert output.beta == pytest.approx(np.array([42.0]))


def test_slepian_bound_reduces_moderate_dimensional_damping(
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
) -> None:
    scales = iter((0.0, 1.0, 0.0))
    monkeypatch.setattr(
        adaptive_ssp_module.accounting,
        "calibrate_gaussian_scale",
        lambda budget, sensitivity: next(scales),
    )
    fake_randomness.script("normal", zero_draw)

    solve_args: tuple[Array2D, Array1D] | None = None

    def record_solve(matrix: Array2D, vector: Array1D) -> Array1D:
        nonlocal solve_args
        solve_args = matrix, vector
        return np.zeros(vector.shape)

    monkeypatch.setattr(np.linalg, "solve", record_solve)
    d = 16
    failure_probability = 0.05
    metadata = LinearRegressionMetadata(
        n_samples=1,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=d),
        response_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
    )
    input_data = LinearRegressionInput(X=np.zeros((1, d)), y=np.zeros(1))

    AdaSSP(failure_probability=failure_probability).run(
        metadata,
        input_data,
        budget=ZCDPBudget(rho=3.0),
        seed=0,
    )

    assert solve_args is not None
    matrix, vector = solve_args
    proof_event_failure_probability = failure_probability / 4.0
    tropp_bound = math.sqrt(
        2.0 * d * math.log(2.0 * d / proof_event_failure_probability)
    )
    slepian_bound = 2.0 * math.sqrt(d) + math.sqrt(
        2.0 * (2.0 - 1.0 / d) * math.log(2.0 / proof_event_failure_probability)
    )
    assert slepian_bound < tropp_bound
    assert matrix == pytest.approx(2.0 * slepian_bound * np.eye(d))
    assert vector == pytest.approx(np.zeros(d))


def test_failure_probability_controls_adaptive_damping() -> None:
    metadata, input_data = _instance()
    strict_tail = AdaSSP(failure_probability=0.01).run(
        metadata,
        input_data,
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        seed=4,
    )
    loose_tail = AdaSSP(failure_probability=0.5).run(
        metadata,
        input_data,
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        seed=4,
    )
    assert not np.allclose(strict_tail.beta, loose_tail.beta)


def test_gdp_offer_releases() -> None:
    metadata, input_data = _instance()

    output = AdaSSP().run(metadata, input_data, budget=GDPBudget(mu=1.0), seed=0)

    assert output.beta.shape == (3,)
    assert np.all(np.isfinite(output.beta))


def test_approx_dp_noise_scale_is_no_larger_than_basic_composition_split() -> None:
    # Old behavior (basic-composition split of eps/delta into equal thirds,
    # then per-third analytic Gaussian calibration) versus new behavior
    # (convert once to GDP, then split mu exactly into equal thirds).
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    sensitivity = 1.0

    old_third = budget.split((1.0, 1.0, 1.0))[0]
    old_scale = accounting.calibrate_gaussian_scale(old_third, sensitivity)

    new_third = _real_spend_as_gdp(budget).split((1.0, 1.0, 1.0))[0]
    new_scale = accounting.calibrate_gaussian_scale(new_third, sensitivity)

    assert new_scale <= old_scale


# Pinned against the pre-conversion AdaSSP, by running:
#   AdaSSP().run(*_instance(), budget=ZCDPBudget(rho=1.5), seed=seed).beta
# for each seed. zCDP composes identically under both the old rho/3 split
# and the new sqrt(2*rho)/sqrt(3) split, so these stay bit-identical.
_ZCDP_PINS: dict[int, list[float]] = {
    0: [0.09392221248129706, -0.0444210381844209, -0.10690953681059484],
    1: [0.12824109988627122, -0.023174082504895364, 0.010830101466983256],
    2: [0.21486821568108178, -0.06932780227995593, 0.06174227700741076],
}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_zcdp_release_matches_origin_dev_pin(seed: int) -> None:
    metadata, input_data = _instance()

    output = AdaSSP().run(metadata, input_data, budget=ZCDPBudget(rho=1.5), seed=seed)

    np.testing.assert_allclose(output.beta, _ZCDP_PINS[seed], rtol=1e-12)


def test_zcdp_split_calibrates_each_release_to_mu_over_sqrt_k() -> None:
    """Each of the 3 releases gets sqrt(2*rho)/sqrt(3), not rho/3."""
    rho = 1.5
    sensitivity = 1.0
    expected_scale = sensitivity / (math.sqrt(2.0 * rho) / math.sqrt(3.0))

    release_budget = _real_spend_as_gdp(ZCDPBudget(rho=rho)).split((1.0, 1.0, 1.0))[0]
    scale = accounting.calibrate_gaussian_scale(release_budget, sensitivity)

    assert scale == pytest.approx(expected_scale)
