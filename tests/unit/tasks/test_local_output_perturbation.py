"""Public behaviour tests for the local output perturbation COI release."""

from __future__ import annotations

from collections.abc import Callable
import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy.optimize import minimize

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    ProductDomain,
)
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.tasks import (
    SingleCoefficientEstimation,
    list_built_in_algorithms,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)
from dpstats.tasks.single_coefficient_estimation.algorithms import (
    local_output_perturbation as lop,
)
from dpstats.tasks.single_coefficient_estimation.algorithms.local_output_perturbation import (
    LocalOutputPerturbation,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.unit.tasks.task_builders import Unreadable

FINITE_OFFER = ApproxDPBudget(eps=30.0, delta=1e-6)
# At n = 20000 both certificates pass at FINITE_OFFER. On H = I/2 the modulus is
# about B ||H^-1 e_j||_2/n = 2 sqrt(2)/20000 = 1.4e-4, and the child
# (10, 2.1e-15) analytic Gaussian has mu ~ 1.22, so sigma ~ 1.2e-4 and
# PROBE_TOLERANCE is over 8 sigma on every probe fixture below.
PROBE_N = 20_000
PROBE_TOLERANCE = 1e-3


def _problem(
    n: int = 200, index: int = 0
) -> tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]:
    """Return an exactly fitted slope-plus-intercept instance.

    Rows are ``x = (+-1, 1)`` in the L2 ball of radius ``sqrt(2)`` and
    ``y = 0.75 x_0``, so the robust-log minimizer is ``(0.75, 0)``, every
    residual vanishes, and the empirical Hessian is ``X^T X/(2n) = I/2``.
    """
    x = np.where(np.arange(n) % 2, 1.0, -1.0)
    X = np.column_stack((x, np.ones(n)))
    return SingleCoefficientEstimation.Metadata(
        n_samples=n,
        design_domain=LpBallDomain.at_origin(p=2, radius=2**0.5, dim=2),
        response_domain=EuclideanSpace(dim=1),
        coefficient_of_interest_index=index,
    ), SingleCoefficientEstimation.Input(X=X, y=0.75 * x)


def _intercept_problem(
    n: int,
) -> tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]:
    X = np.ones((n, 1))
    return SingleCoefficientEstimation.Metadata(
        n_samples=n,
        design_domain=ProductDomain(parts=(InterceptDomain(),)),
        response_domain=EuclideanSpace(dim=1),
        coefficient_of_interest_index=0,
    ), SingleCoefficientEstimation.Input(X=X, y=np.full(n, 0.75))


def _contaminated_problem(
    n: int = 200, outliers: int = 1
) -> tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]:
    """Set ``y = 100`` on the first ``outliers`` rows with ``x_0 = -1``."""
    metadata, input_data = _problem(n)
    y = input_data.y.copy()
    y[: 2 * outliers : 2] = 100.0
    return metadata, SingleCoefficientEstimation.Input(X=input_data.X, y=y)


def _robust_log_oracle(
    input_data: SingleCoefficientEstimationInput, regularization: float = 0.0
) -> np.ndarray:
    """Minimize equation (1) by BFGS on the objective, not its score equation."""

    def objective(theta: np.ndarray) -> float:
        residual = input_data.y - input_data.X @ theta
        loss = np.mean(np.logaddexp(0.0, residual) + np.logaddexp(0.0, -residual))
        return float(loss + regularization / 2.0 * theta @ theta)

    oracle = minimize(objective, x0=np.zeros(input_data.X.shape[1]), method="BFGS")
    assert oracle.success
    return oracle.x


@pytest.mark.parametrize(("index", "expected"), [(0, 0.75), (1, 0.0)])
def test_registration_regularization_surface_and_coefficient_of_interest(
    index: int, expected: float
) -> None:
    """The fixture's exact minimizer is ``(0.75, 0)``; ``index`` picks one."""
    metadata, input_data = _problem(n=PROBE_N, index=index)
    assert (
        list_built_in_algorithms(SingleCoefficientEstimation)["LocalOutputPerturbation"]
        is LocalOutputPerturbation
    )
    assert set(LocalOutputPerturbation.parameter_info()) == {"regularization"}
    assert LocalOutputPerturbation().regularization == 0.0
    output = LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 1)
    assert output.coefficient == pytest.approx(expected, abs=PROBE_TOLERANCE)


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-6),
        ApproxDPBudget(eps=1.0, delta=1.0),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
def test_every_unconstrained_budget_declares_the_gap_before_reading_input(
    budget: DPBudget,
) -> None:
    metadata, input_data = _problem()
    for offered_input in (input_data, Unreadable("Input")):
        with pytest.raises(NotImplementedError, match="zero-noise limit"):
            LocalOutputPerturbation().run(
                metadata,
                cast(SingleCoefficientEstimationInput, offered_input),
                budget,
                1,
            )


def test_coefficient_of_interest_index_selects_the_existing_intercept_column() -> None:
    """``y = 0.75`` on an intercept-only design has minimizer 0.75, ``B = 1``."""
    metadata, input_data = _intercept_problem(PROBE_N)
    output = LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 1)
    assert output.coefficient == pytest.approx(0.75, abs=PROBE_TOLERANCE)


@pytest.mark.parametrize(
    "budget", [PureDPBudget(eps=1), ZCDPBudget(rho=1), GDPBudget(mu=1)]
)
def test_refuses_unsupported_finite_families(budget: DPBudget) -> None:
    metadata, input_data = _problem()
    with pytest.raises(InvalidRegime, match="ApproxDPBudget"):
        LocalOutputPerturbation().run(metadata, input_data, budget, 0)


def test_refuses_a_design_domain_without_a_finite_l2_bound() -> None:
    _, input_data = _problem()
    metadata = SingleCoefficientEstimation.Metadata(
        n_samples=200,
        design_domain=EuclideanSpace(dim=2),
        response_domain=EuclideanSpace(dim=1),
        coefficient_of_interest_index=0,
    )
    with pytest.raises(InvalidRegime, match="L2 bound"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_c1_bottom_precedes_algorithm_4(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        lop, "_algorithm_4", lambda *args: pytest.fail("Algorithm 4 ran after C1")
    )
    with pytest.raises(NoRelease, match="C1 failed"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_ratio_denominator_failure_is_accounted_bottom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    # kappa = 1e6 drives 1 - kappa (s1 + s2 r) negative.
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 1e6)
    with pytest.raises(NoRelease, match="ratio certificate"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_corrected_one_minus_gamma_factor_rejects_where_one_plus_gamma_would_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix gamma = 0.095, kappa = 1/2 and gamma' = 0 at (E, D) = (15, 1e-6).

    The child is ``e = 5``, ``d ~ 4.5e-11``, so the right-hand side of (14a)
    is ``10/(1 + 6.6^2) ~ 0.224``. With ``s1 = 1/0.905 - 1 ~ 0.105`` and
    ``s2 r < 0.008``, the corrected first term gives
    ``(1.0525/0.945)^2 - 1 ~ 0.24`` (reject), while the printed
    ``1 + kappa gamma/(1 + gamma)`` would give ``(1.0434/0.945)^2 - 1 ~ 0.22``
    (accept).
    """
    metadata, input_data = _problem()
    gamma = 0.095
    t = gamma / (1.2332 * metadata.design_domain.l2_norm_bound)
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 0.5)
    monkeypatch.setattr(lop, "_r_min", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(lop, "_t_add", lambda lam, B, n: t if lam == 1.0 else 0.0)
    with pytest.raises(NoRelease, match="ratio certificate"):
        LocalOutputPerturbation().run(
            metadata, input_data, ApproxDPBudget(eps=15.0, delta=1e-6), 0
        )


def test_gamma_margin_at_one_refuses_before_division(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 1.0)
    monkeypatch.setattr(lop, "_r_min", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        lop,
        "_t_add",
        lambda *args: 1.0 / (1.2332 * metadata.design_domain.l2_norm_bound),
    )
    with pytest.raises(NoRelease, match="ratio certificate"):
        LocalOutputPerturbation().run(
            metadata, input_data, ApproxDPBudget(eps=15.0, delta=1e-6), 0
        )


def test_lower_recursion_image_without_a_real_stability_radius_is_bottom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``t_add(R(lambda_0))`` needs ``R(lambda_0) >= 4 alpha B^2/n``."""
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 1.0)
    monkeypatch.setattr(lop, "_r_min", lambda *args, **kwargs: 1e-6)
    with pytest.raises(NoRelease, match="ratio certificate"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_empirical_curvature_below_the_modulus_domain_is_no_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reachable only when Algorithm 2's lower bound exceeds the true curvature.

    ``lambda_min = 0.01`` is below ``4 alpha B^2/n = 0.049``, where equation
    (11)'s remainder has no real ``t_add``.
    """
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 1.0)
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda matrix: np.array([0.01, 1.0]))
    with pytest.raises(NoRelease, match="confidence event failed"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_algorithm_4_uses_floored_nonnegative_count_and_upper_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 1.0)
    # The child of (30, 1e-6) is (10, d), d = 1e-6/(1 + e^10 + e^20)
    # = 2.06106004620906e-15, so k = log(1/(2d))/10 = 3.31224087783646.
    k = 3.31224087783646
    B = metadata.design_domain.l2_norm_bound
    # lambda_max at the saturated endpoint G1 = B^2/2 gives N = 0, so a Laplace
    # draw of k + 1.9 releases floor(1.9) = 1 inversion.
    monkeypatch.setattr(
        np.linalg, "eigvalsh", lambda matrix: np.array([0.5, B**2 / 2.0])
    )

    class FixedRng:
        def laplace(self, *, scale: float) -> float:
            return k + 1.9

        def normal(self, *, scale: float) -> float:
            return 0.0

    upper_inverse_calls: list[str] = []

    def observe_upper_inverse(
        predicate: object,
        lower: float,
        upper: float,
        *,
        satisfying_endpoint: str,
    ) -> float:
        assert callable(predicate)
        assert lower == 0.0
        assert upper == pytest.approx(B**2 / 2.0)
        assert predicate(upper)
        upper_inverse_calls.append(satisfying_endpoint)
        return upper

    monkeypatch.setattr(lop.np.random, "default_rng", lambda seed: FixedRng())
    monkeypatch.setattr(lop, "bisect_monotone_boundary", observe_upper_inverse)
    monkeypatch.setattr(
        lop.accounting, "calibrate_gaussian_scale", lambda *args, **kwargs: 0.0
    )
    LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)
    assert upper_inverse_calls == ["upper"]


@pytest.mark.parametrize(
    ("offer", "child"),
    [
        # d = 1e-6/(1 + e + e^2), evaluated at 30 digits.
        (ApproxDPBudget(eps=3.0, delta=1e-6), (1.0, 9.00305731703805e-8)),
        # d = 1e-6/(1 + e^10 + e^20), evaluated at 30 digits.
        (FINITE_OFFER, (10.0, 2.06106004620906e-15)),
    ],
)
def test_finite_run_spends_three_equal_approximate_children(
    accounting_spy: AccountingSpy,
    offer: ApproxDPBudget,
    child: tuple[float, float],
) -> None:
    """Corollary 3.6's split ``(E/3, D/(1 + e^e + e^{2e}))`` at moderate ``E``."""
    metadata, input_data = _problem(n=PROBE_N)
    LocalOutputPerturbation().run(metadata, input_data, offer, 0)
    assert [call.entry_point for call in accounting_spy.calls] == [
        "spend_as_pure",
        "spend_as_pure",
        "calibrate_gaussian_scale",
    ]
    for call in accounting_spy.calls:
        assert_budget_close(call.budget, ApproxDPBudget(eps=child[0], delta=child[1]))


def test_gaussian_scale_reads_the_empirical_curvature_not_its_private_bound(
    accounting_spy: AccountingSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equation (11)'s modulus depends on the data's own ``lambda_min``.

    On the fixture ``H = I/2``, so ``||H^-1 e_0||_2 = 2`` and the directional
    term is ``B * 2/n = sqrt(2)/100``; the remainder is positive.
    """
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 0.5)
    sensitivities: list[float] = []
    for lower_bound in (0.4, 0.49):
        monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: lower_bound)
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)
        (gaussian,) = [
            call
            for call in accounting_spy.calls
            if call.entry_point == "calibrate_gaussian_scale"
        ][-1:]
        assert gaussian.sensitivity is not None
        sensitivities.append(gaussian.sensitivity)
    assert sensitivities[0] == pytest.approx(sensitivities[1])
    assert sensitivities[0] > 2**0.5 / 100


def test_gaussian_scale_reads_the_regularized_modulus(
    accounting_spy: AccountingSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equation (11) with ``lambda_reg = 0.1`` on a pinned loss Hessian ``I/2``.

    ``B = sqrt(2)``, ``n = 200``: ``(H + 0.1 I)^-1 e_0 = e_0/0.6`` gives the
    directional term ``sqrt(2)/(0.6 * 200) = 0.0117851``; at
    ``lambda_min + lambda_reg = 0.6``, ``t_add = 0.0120383`` and
    ``gamma = alpha B t_add = 0.0209940``, so the remainder is
    ``sqrt(2)/120 * gamma/(1 - gamma) = 0.0002527`` and ``omega = 0.0120378``.
    """
    metadata, input_data = _problem()
    monkeypatch.setattr(
        lop,
        "_solve_robust_log",
        lambda *args: (np.zeros(metadata.dim), np.eye(metadata.dim) / 2.0),
    )
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 0.6)
    monkeypatch.setattr(lop, "_passes_ratio_certificate", lambda *args, **kwargs: True)
    LocalOutputPerturbation(regularization=0.1).run(
        metadata, input_data, FINITE_OFFER, 0
    )
    (gaussian,) = [
        call
        for call in accounting_spy.calls
        if call.entry_point == "calibrate_gaussian_scale"
    ]
    assert gaussian.sensitivity == pytest.approx(0.0120378, rel=1e-5)


def test_release_is_robust_log_not_ols() -> None:
    """One percent of responses at 100 move the OLS slope by about one."""
    metadata, contaminated = _contaminated_problem(PROBE_N, outliers=PROBE_N // 100)
    robust = LocalOutputPerturbation().run(metadata, contaminated, FINITE_OFFER, 0)
    ols = np.linalg.lstsq(contaminated.X, contaminated.y, rcond=None)[0][0]
    assert abs(robust.coefficient - ols) > 0.5


def test_release_matches_independent_robust_log_oracle() -> None:
    metadata, contaminated = _contaminated_problem(PROBE_N, outliers=PROBE_N // 100)
    oracle = _robust_log_oracle(contaminated)
    output = LocalOutputPerturbation().run(metadata, contaminated, FINITE_OFFER, 0)
    assert output.coefficient == pytest.approx(oracle[0], abs=PROBE_TOLERANCE)


def test_finite_run_passes_public_n_to_both_curvature_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()
    seen_n: list[int] = []

    def observe_algorithm_2(
        lambda_min: float,
        B: float,
        n: int,
        budget: ApproxDPBudget,
        seed: int,
        *,
        regularization: float,
    ) -> float:
        seen_n.append(n)
        return 0.5

    def observe_algorithm_4(
        lambda_max: float,
        lambda_0: float,
        B: float,
        n: int,
        budget: ApproxDPBudget,
        seed: int,
    ) -> float:
        seen_n.append(n)
        return 0.6

    monkeypatch.setattr(lop, "_algorithm_2", observe_algorithm_2)
    monkeypatch.setattr(lop, "_algorithm_4", observe_algorithm_4)
    LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)
    assert seen_n == [metadata.n_samples, metadata.n_samples]


def test_solver_nonconvergence_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()
    monkeypatch.setattr(
        lop,
        "root",
        lambda *args, **kwargs: SimpleNamespace(x=np.zeros(2), success=False),
    )
    with pytest.raises(RuntimeError, match="first-order tolerance"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_solver_exception_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, input_data = _problem()

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("bad solve")

    monkeypatch.setattr(lop, "root", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


def test_curvature_eigensolve_failure_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input_data = _problem()

    def fail(matrix: np.ndarray) -> None:
        raise np.linalg.LinAlgError("bad spectrum")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(RuntimeError, match="eigensolve"):
        LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, 0)


@pytest.mark.parametrize(
    ("problem", "seed", "expected"),
    [
        (_problem, 0, "0x1.808ce0918d42bp-1"),
        (_problem, 7, "0x1.82253023e466dp-1"),
        (_contaminated_problem, 0, "0x1.7b61a6698fcc4p-1"),
        (_contaminated_problem, 7, "0x1.7cfc36d628110p-1"),
    ],
)
def test_default_regularization_reproduces_the_unregularized_release(
    problem: Callable[
        [], tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]
    ],
    seed: int,
    expected: str,
) -> None:
    """Pinned same-seed releases of the port before ``regularization`` existed."""
    metadata, input_data = problem()
    output = LocalOutputPerturbation().run(metadata, input_data, FINITE_OFFER, seed)
    assert output.coefficient == pytest.approx(float.fromhex(expected), abs=1e-12)


def test_positive_regularization_constructs_and_releases() -> None:
    metadata, input_data = _problem()
    output = LocalOutputPerturbation(regularization=0.1).run(
        metadata, input_data, FINITE_OFFER, 0
    )
    assert math.isfinite(output.coefficient)


def test_regularized_release_matches_independent_oracle() -> None:
    """Equation (1) centred at the origin: mean robust-log loss plus ridge/2."""
    metadata, contaminated = _contaminated_problem(PROBE_N, outliers=PROBE_N // 100)
    regularization = 0.3
    oracle = _robust_log_oracle(contaminated, regularization)
    output = LocalOutputPerturbation(regularization=regularization).run(
        metadata, contaminated, FINITE_OFFER, 0
    )
    unregularized = LocalOutputPerturbation().run(
        metadata, contaminated, FINITE_OFFER, 0
    )
    assert output.coefficient == pytest.approx(oracle[0], abs=PROBE_TOLERANCE)
    assert abs(output.coefficient - unregularized.coefficient) > 0.1


def test_lower_recursion_reads_the_regularized_c1_and_stability_radius() -> None:
    """Equation (10) at ``B = 1``, ``n = 100``, ``lambda_reg = 0.02``.

    (C1) is ``lambda + 0.02/rho >= (8 alpha + 1)/100 = 0.108656``, so it holds
    from ``lambda = 0.068656``. At ``lambda = 0.07``,
    ``t_add(0.09) = [1 - sqrt(1 - 4 alpha/9)]/(2 alpha) = 0.13289`` and
    ``R = 0.07 (2 - exp(0.13289)) - 1/200 = 0.055051``.
    """
    assert getattr(lop, "_r_min")(0.07, 1.0, 100, regularization=0.02) == pytest.approx(
        0.0550514, rel=1e-5
    )
    assert getattr(lop, "_r_min")(0.06, 1.0, 100, regularization=0.02) == 0.0
    assert getattr(lop, "_r_min")(0.07, 1.0, 100, regularization=0.0) == 0.0


@pytest.mark.parametrize(("regularization", "passes_c1"), [(0.0, False), (0.02, True)])
def test_c1_adds_regularization_over_rho_to_the_released_lower_bound(
    monkeypatch: pytest.MonkeyPatch, regularization: float, passes_c1: bool
) -> None:
    """On the fixture ``(8 alpha + 1) B^2/n = 0.108656``; ``0.08 + 0.02/rho`` clears it."""
    metadata, input_data = _problem()
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 0.08)

    class ReachedAlgorithm4(Exception):
        pass

    def reach_algorithm_4(*args: object) -> float:
        raise ReachedAlgorithm4

    monkeypatch.setattr(lop, "_algorithm_4", reach_algorithm_4)
    algorithm = LocalOutputPerturbation(regularization=regularization)
    expected = ReachedAlgorithm4 if passes_c1 else NoRelease
    with pytest.raises(expected):
        algorithm.run(metadata, input_data, FINITE_OFFER, 0)


def test_ratio_certificate_reads_r_reg_at_the_unregularized_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``R_reg(lambda_0) = R_min(lambda_0 - lambda_reg) + lambda_reg``."""
    metadata, input_data = _problem()
    regularization = 0.05
    monkeypatch.setattr(lop, "_algorithm_2", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(lop, "_algorithm_4", lambda *args: 0.4)
    seen: list[tuple[float, float]] = []
    r_min = getattr(lop, "_r_min")

    def observe_r_min(lam: float, B: float, n: int, *, regularization: float) -> float:
        seen.append((lam, regularization))
        return r_min(lam, B, n, regularization=regularization)

    monkeypatch.setattr(lop, "_r_min", observe_r_min)
    LocalOutputPerturbation(regularization=regularization).run(
        metadata, input_data, FINITE_OFFER, 0
    )
    assert seen == [(pytest.approx(0.3), regularization)]


def test_ratio_certificate_uses_the_proved_beta_and_s2() -> None:
    """``B = 1``, ``n = 200``, ``lambda_0 = 0.15``, ``lambda_1 = 0.225``.

    With the proved ``[1 - alpha r t]_+^2`` in ``s2`` the certificate flips
    near ``eps = 4.876``; with the printed first power it would already pass
    at ``4.85``. Rejecting at 4.85 and accepting at 4.95 pins the square.
    """
    passes = getattr(lop, "_passes_ratio_certificate")
    below = ApproxDPBudget(eps=4.85, delta=1e-6)
    above = ApproxDPBudget(eps=4.95, delta=1e-6)
    assert not passes(0.15, 0.225, 1.0, 200, below, regularization=0.0)
    assert passes(0.15, 0.225, 1.0, 200, above, regularization=0.0)


def test_ratio_certificate_uses_lemma_8_3_hessian_term() -> None:
    """``B = 1``, ``n = 1000``, ``lambda_0 = lambda_1 = 0.1995``.

    Lemma 8.3's ``s2 kappa r^2/lambda_0`` flips the certificate near
    ``eps = 0.382``; the printed ``s2 kappa r`` would pass at ``0.33``.
    Rejecting at 0.33 and accepting at 0.40 pins the corrected term.
    """
    passes = getattr(lop, "_passes_ratio_certificate")
    below = ApproxDPBudget(eps=0.33, delta=1e-6)
    above = ApproxDPBudget(eps=0.40, delta=1e-6)
    assert not passes(0.1995, 0.1995, 1.0, 1000, below, regularization=0.0)
    assert passes(0.1995, 0.1995, 1.0, 1000, above, regularization=0.0)
