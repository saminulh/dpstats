"""Tests for Asi--Duchi--Talwar local robust regression."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.exceptions import NoRelease
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms import (
    local_output_perturbation as lop_module,
)
from dpstats.tasks.linear_regression.algorithms.local_output_perturbation import (
    LocalOutputPerturbation,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingSpy,
    assert_budget_close,
)


def _perfect_fit_instance(
    n_samples: int = 200,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    """Return a one-dimensional instance whose curvature is known exactly.

    Rows alternate ``x = +-1`` on the unit design ball and ``y = 0.75 x``, so
    the robust-log minimizer is ``0.75``, every residual vanishes there, and
    the empirical Hessian is exactly ``B^2/2`` -- the largest curvature the
    robust-log score admits. Knowing ``lambda_min`` in closed form lets a test
    place it on either side of the certificate's condition threshold.
    """
    x = np.where(np.arange(n_samples) % 2 == 0, 1.0, -1.0)
    metadata = LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    return metadata, LinearRegressionInput(X=x[:, None], y=x * 0.75)


def _random_unit_ball_rows(rng: np.random.Generator, n_rows: int, dim: int) -> Array2D:
    directions = rng.normal(size=(n_rows, dim))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions * rng.uniform(0.3, 1.0, size=n_rows)[:, None]


def _releases(
    metadata: LinearRegressionMetadata,
    input_data: LinearRegressionInput,
    budget: DPBudget,
    seed: int,
) -> bool:
    """Report whether the mechanism's private curvature certificate succeeded."""
    try:
        LocalOutputPerturbation().run(metadata, input_data, budget, seed)
    except NoRelease:
        return False
    return True


def test_composes_equal_child_budgets_through_accounting(
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input_data = _perfect_fit_instance()
    LocalOutputPerturbation().run(
        metadata,
        input_data,
        ApproxDPBudget(eps=10.0, delta=0.1),
        seed=8,
    )
    assert [call.entry_point for call in accounting_spy.calls] == [
        "spend_as_pure",
        "calibrate_gaussian_scale",
    ]
    assert_budget_close(accounting_spy.calls[0].budget, PureDPBudget(eps=5.0))
    assert_budget_close(
        accounting_spy.calls[1].budget,
        ApproxDPBudget(eps=5.0, delta=0.05),
    )
    assert accounting_spy.calls[1].sensitivity is not None
    assert accounting_spy.calls[1].sensitivity > 0.0


@pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
def test_refuses_finite_budget_families_without_failure_slack(
    budget: DPBudget,
) -> None:
    metadata, input_data = _perfect_fit_instance()
    with pytest.raises(InvalidRegime, match="Cannot spend"):
        LocalOutputPerturbation().run(
            metadata,
            input_data,
            budget,
            seed=0,
        )


def test_zero_private_curvature_release_is_accounted_no_release() -> None:
    metadata = LinearRegressionMetadata(
        n_samples=20,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input_data = LinearRegressionInput(X=np.zeros((20, 1)), y=np.arange(20.0))
    with pytest.raises(NoRelease, match="private curvature lower bound was zero"):
        LocalOutputPerturbation().run(
            metadata,
            input_data,
            ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=0.1),
        ZCDPBudget(rho=math.inf),
    ],
)
def test_unconstrained_budget_families_return_exact_minimizer(
    budget: DPBudget,
) -> None:
    metadata, input_data = _perfect_fit_instance()
    output = LocalOutputPerturbation().run(
        metadata,
        input_data,
        budget,
        seed=0,
    )
    assert output.beta[0] == pytest.approx(0.75, abs=1e-8)


def test_solver_failure_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _perfect_fit_instance()
    monkeypatch.setattr(
        lop_module,
        "root",
        lambda *args, **kwargs: SimpleNamespace(x=np.array([np.nan])),
    )
    with pytest.raises(RuntimeError, match="exact-minimizer argument"):
        LocalOutputPerturbation().run(
            metadata,
            input_data,
            ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )


def test_unconstrained_release_matches_an_independent_robust_log_minimizer() -> None:
    """Compare the estimator against a derivative-free reference solve.

    The reference objective ``sum_i [log(1+e^r) + log(1+e^-r)]`` is written
    from the source and minimized by Nelder--Mead, so it shares no gradient,
    Hessian, or root-finding code with the mechanism.
    """
    rng = np.random.default_rng(11)
    X = _random_unit_ball_rows(rng, 60, 2)
    y = X @ np.array([1.5, -0.5]) + rng.normal(scale=1.2, size=60)
    metadata = LinearRegressionMetadata(
        n_samples=X.shape[0],
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        response_domain=EuclideanSpace(dim=1),
    )

    def robust_log_loss(beta: Array1D) -> float:
        residuals: Array1D = y - X @ beta
        return float(
            np.sum(np.logaddexp(0.0, residuals) + np.logaddexp(0.0, -residuals))
        )

    reference = minimize(
        robust_log_loss,
        np.zeros(2),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000},
    )
    output = LocalOutputPerturbation().run(
        metadata,
        LinearRegressionInput(X=X, y=y),
        ApproxDPBudget(eps=float("inf"), delta=0.1),
        seed=0,
    )
    assert output.beta == pytest.approx(reference.x, abs=1e-6)


@pytest.mark.parametrize(("n_samples", "releases"), [(21, False), (22, True)])
def test_condition_threshold_gates_the_add_remove_certificate(
    n_samples: int, releases: bool
) -> None:
    """Pin the add/remove condition threshold ``(8 alpha + 1) B^2 / n``.

    The instance has ``lambda_min = B^2/2`` exactly, so the curvature recursion
    survives a first step only when ``n > 2 (8 alpha + 1) = 21.7`` at the
    source's ``alpha = 1.2332``. This budget keeps the confidence offset
    ``log(1/(2 delta_cert))/eps_cert`` below one, so the released count follows
    the recursion count directly and a certificate that stops immediately
    floors to zero. The source's replace-one constants double the gradient
    term and would move the boundary to ``n > 2 (16 alpha + 1) = 41.5``, so
    both legs also separate the shipped add/remove derivation from the paper's
    fixed-size one.
    """
    metadata, input_data = _perfect_fit_instance(n_samples)
    budget = ApproxDPBudget(eps=10.0, delta=0.1)
    assert _releases(metadata, input_data, budget, seed=0) is releases


@pytest.mark.parametrize(("n_samples", "releases"), [(100, False), (400, True)])
def test_benchmark_budget_refuses_until_the_recursion_outruns_the_offset(
    n_samples: int, releases: bool
) -> None:
    """Exercise the release rate at a benchmark-scale privacy budget.

    At ``eps=1, delta=1e-6`` the confidence offset
    ``log(1/(2 delta_cert))/eps_cert`` is ``27.6``, so the mechanism releases
    only when the curvature recursion survives more than that many steps. Even
    this perfectly conditioned one-dimensional instance therefore refuses at
    ``n=100`` and releases at ``n=400``: frequent accounted refusal at
    benchmark sample sizes is a property of this comparator, not a defect.
    """
    metadata, input_data = _perfect_fit_instance(n_samples)
    budget = ApproxDPBudget(eps=1.0, delta=1e-6)
    assert _releases(metadata, input_data, budget, seed=1) is releases


def test_charged_sensitivity_dominates_add_neighbor_parameter_change(
    accounting_spy: AccountingSpy,
) -> None:
    """Check the certified stability radius on random add-neighbor pairs.

    The Gaussian seam is calibrated to the local stability radius at the
    privately certified curvature, so the privacy claim requires that radius
    to dominate the realized L2 move of the robust-log minimizer across every
    add/remove edge. Each neighbor adds one worst-radius record while holding
    the public reference ``n`` fixed, which is the convention the add/remove
    derivation is stated under and therefore an intentionally off-contract
    instance. Both minimizers come from the unconstrained route, so no
    mechanism formula is restated here.
    """
    rng = np.random.default_rng(20260729)
    n_samples = 400
    metadata = LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=LpBallDomain.at_origin(p=2, radius=1.0, dim=2),
        response_domain=EuclideanSpace(dim=1),
    )
    algorithm = LocalOutputPerturbation()
    unconstrained = ApproxDPBudget(eps=float("inf"), delta=0.1)
    ratios: list[float] = []
    for pair in range(100):
        X = _random_unit_ball_rows(rng, n_samples, 2)
        y = X @ np.array([0.5, -1.25]) + rng.normal(scale=0.8, size=n_samples)
        extra_row = rng.normal(size=2)
        extra_row /= np.linalg.norm(extra_row)
        instance = LinearRegressionInput(X=X, y=y)
        neighbor = LinearRegressionInput(
            X=np.vstack([X, extra_row]),
            y=np.append(y, rng.normal(scale=3.0)),
        )

        algorithm.run(metadata, instance, ApproxDPBudget(eps=10.0, delta=0.1), pair)
        charged = accounting_spy.calls[-1].sensitivity
        assert charged is not None
        beta = algorithm.run(metadata, instance, unconstrained, seed=0).beta
        neighbor_beta = algorithm.run(metadata, neighbor, unconstrained, seed=0).beta
        ratios.append(float(np.linalg.norm(neighbor_beta - beta)) / charged)

    assert max(ratios) < 1.0


def test_certificate_count_is_floored_to_whole_recursion_steps(
    accounting_spy: AccountingSpy,
) -> None:
    """Pin the integer floor that replaces the source's fractional exponent.

    The released count is ``m = floor(max(N_hat - q, 0))`` for the public
    offset ``q = log(1/(2 delta_cert))/eps_cert``, and the charged radius is a
    function of ``m`` alone. Holding the instance, ``eps`` and the seed fixed,
    failure probabilities whose offsets land in the same unit interval of
    ``N_hat`` must therefore charge exactly the same radius, and an offset a
    whole step further must charge a strictly larger one. Applying the
    recursion a continuous number of times, as the source pseudocode literally
    reads, would instead move the radius with every ``delta``.
    """
    metadata, input_data = _perfect_fit_instance(200)
    algorithm = LocalOutputPerturbation()
    charged: list[float] = []
    for delta in (0.2, 0.1, 0.05, 0.02):
        algorithm.run(metadata, input_data, ApproxDPBudget(eps=10.0, delta=delta), 4)
        sensitivity = accounting_spy.calls[-1].sensitivity
        assert sensitivity is not None
        charged.append(sensitivity)

    assert charged[:3] == pytest.approx([charged[0]] * 3)
    assert charged[3] > charged[2]
