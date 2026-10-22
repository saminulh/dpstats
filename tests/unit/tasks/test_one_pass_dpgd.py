# pyright: reportPrivateUsage=false
"""Tests for the Bombari et al. one-pass DP-GD schedule family."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.configure import grid
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms import (
    one_pass_dpgd as one_pass_module,
)
from dpstats.tasks.linear_regression.algorithms.one_pass_dpgd import (
    OnePassDPGD,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order

# The default configuration (schedule="polynomial") and a representative
# harmonic configuration, reused wherever a test needs one concrete instance
# of each schedule.
POLYNOMIAL_ROSTER = OnePassDPGD()
HARMONIC_ROSTER = OnePassDPGD(schedule="harmonic", scale=10.0, tau_over_gamma=30.0)


def _instance(
    n: int = 24, d: int = 2
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, d))
    beta = np.arange(1, d + 1, dtype=np.float64) / d
    y = X @ beta + rng.normal(scale=0.2, size=n)
    metadata = LinearRegressionMetadata(
        n_samples=n,
        design_domain=EuclideanSpace(dim=d),
        response_domain=EuclideanSpace(dim=1),
    )
    return metadata, LinearRegressionInput(X=X, y=y)


# --- Equivalence with the pre-merge polynomial and harmonic implementations ---
#
# Pinned by running the pre-merge polynomial and harmonic schedule code
# directly, before it collapsed into this class, at a fixed input/budget/seed
# (n=6, d=2, rho=0.3, seed=17). Reproduced here from that captured run.

_PINNED_X = np.array(
    [
        [0.30471707975443135, -1.0399841062404955],
        [0.7504511958064572, 0.9405647163912139],
        [-1.9510351886538364, -1.302179506862318],
        [0.12784040316728537, -0.3162425923435822],
        [-0.016801157504288795, -0.85304392757358],
        [0.8793979748628286, 0.7777919354289483],
    ]
)
_PINNED_Y = np.array(
    [
        0.8313122026308007,
        0.3928929583076536,
        -1.2531945009974728,
        0.20003245305075262,
        0.4465958846907511,
        0.3946137470654545,
    ]
)
_PINNED_N, _PINNED_D = _PINNED_X.shape
_PINNED_BUDGET = ZCDPBudget(rho=0.3)
_PINNED_SEED = 17


def _pinned_instance() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    metadata = LinearRegressionMetadata(
        n_samples=_PINNED_N,
        design_domain=EuclideanSpace(dim=_PINNED_D),
        response_domain=EuclideanSpace(dim=1),
    )
    return metadata, LinearRegressionInput(X=_PINNED_X, y=_PINNED_Y)


def test_polynomial_schedule_reproduces_its_pinned_output() -> None:
    # Pinned output of the polynomial schedule at c=0.15, learning-rate
    # scale 12.0, alpha=0.5.
    metadata, input = _pinned_instance()
    algorithm = OnePassDPGD(schedule="polynomial", c=0.15, scale=12.0, alpha=0.5)

    output = algorithm.run(metadata, input, _PINNED_BUDGET, seed=_PINNED_SEED)

    np.testing.assert_array_equal(
        output.beta, np.array([-1.0526200466250433, 1.1483076033379103])
    )


def test_harmonic_schedule_reproduces_its_pinned_output_via_tau_over_gamma() -> None:
    # Pinned output of the harmonic schedule at c=0.15, beta=8.0,
    # tau=0.4. The acceptance mapping is tau_over_gamma = tau * n / d.
    metadata, input = _pinned_instance()
    tau = 0.4
    tau_over_gamma = tau * _PINNED_N / _PINNED_D
    algorithm = OnePassDPGD(
        schedule="harmonic", c=0.15, scale=8.0, tau_over_gamma=tau_over_gamma
    )

    output = algorithm.run(metadata, input, _PINNED_BUDGET, seed=_PINNED_SEED)

    np.testing.assert_array_equal(
        output.beta, np.array([-1.135922008332585, 0.7420036222251488])
    )


# --- Validator refusals ---


def test_polynomial_requires_alpha() -> None:
    # alpha defaults to 0.5; an explicit None still refuses.
    with pytest.raises(ValidationError, match="alpha is required"):
        OnePassDPGD(schedule="polynomial", scale=30.0, alpha=None)


def test_polynomial_forbids_tau_over_gamma() -> None:
    with pytest.raises(ValidationError, match="tau_over_gamma must not be set"):
        OnePassDPGD(schedule="polynomial", scale=30.0, alpha=0.5, tau_over_gamma=30.0)


def test_harmonic_requires_tau_over_gamma() -> None:
    with pytest.raises(ValidationError, match="tau_over_gamma is required"):
        OnePassDPGD(schedule="harmonic", scale=10.0)


def test_harmonic_forbids_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha must not be set"):
        OnePassDPGD(schedule="harmonic", scale=10.0, tau_over_gamma=30.0, alpha=0.5)


def test_harmonic_accepts_an_explicit_none_alpha() -> None:
    # A round trip through a harmonic instance's full field dump always
    # carries alpha=None explicitly, so that exact call must not refuse.
    algorithm = OnePassDPGD(schedule="harmonic", scale=10.0, tau_over_gamma=30.0)
    reconstructed = OnePassDPGD(**dict(algorithm))
    assert reconstructed.alpha is None
    assert reconstructed.uuid == algorithm.uuid


# --- Default construction ---


def test_default_construction_is_the_polynomial_roster_configuration() -> None:
    algorithm = OnePassDPGD()
    assert algorithm.schedule == "polynomial"
    assert algorithm.c == 0.1
    assert algorithm.scale == 30.0
    assert algorithm.alpha == 0.5
    assert algorithm.tau_over_gamma is None


def test_harmonic_roster_configuration_needs_no_alpha() -> None:
    algorithm = OnePassDPGD(schedule="harmonic", scale=10.0, tau_over_gamma=30.0)
    assert algorithm.alpha is None


# --- Default tuning grid ---


def test_default_tuning_grid_constructs_every_candidate() -> None:
    # The declared default grid is the default schedule's ("polynomial")
    # grid only, so every point it produces must construct.
    candidates = grid(OnePassDPGD, **OnePassDPGD.default_tuning_grid())
    assert candidates
    assert all(candidate.schedule == "polynomial" for candidate in candidates)


def test_harmonic_grid_constructs_explicitly() -> None:
    candidates = grid(
        OnePassDPGD,
        schedule=["harmonic"],
        scale=[10.0, 30.0],
        tau_over_gamma=[16.0, 100.0, 1000.0],
    )
    assert len(candidates) == 6
    assert all(candidate.schedule == "harmonic" for candidate in candidates)


# --- Identity ---


def test_illegal_polynomial_with_tau_over_gamma_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError):
        OnePassDPGD(schedule="polynomial", scale=30.0, alpha=0.5, tau_over_gamma=1.0)


def test_legal_polynomial_and_harmonic_configurations_have_different_uuids() -> None:
    assert POLYNOMIAL_ROSTER.uuid != HARMONIC_ROSTER.uuid


def test_clipping_constant_defaults_to_the_paper_value() -> None:
    # Bombari et al. use c=0.1 in their schedule comparison.
    assert POLYNOMIAL_ROSTER.c == HARMONIC_ROSTER.c == 0.1


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 100.0])
def test_polynomial_alpha_accepts_source_domain(alpha: float) -> None:
    assert OnePassDPGD(schedule="polynomial", scale=30.0, alpha=alpha).alpha == alpha


@pytest.mark.parametrize(
    "alpha",
    [-1.0, 0.25, 0.5000001, 0.75],
)
def test_polynomial_alpha_rejects_values_outside_source_domain(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        OnePassDPGD(schedule="polynomial", scale=30.0, alpha=alpha)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.0, [2.0, 2.0, 2.0, 2.0]),
        (0.5, [math.sqrt(3.0), math.sqrt(2.0), 1.0, 0.0]),
        (1.0, [1.5, 1.0, 0.5, 0.0]),
        (2.0, [1.125, 0.5, 0.125, 0.0]),
    ],
)
def test_polynomial_learning_rates_follow_equations_4_4_and_5_1(
    alpha: float,
    expected: list[float],
) -> None:
    rates = one_pass_module._polynomial_learning_rates(
        n=4,
        learning_rate_scale=8.0,
        alpha=alpha,
    )

    np.testing.assert_allclose(rates, np.array(expected))


def test_harmonic_learning_rates_follow_equation_5_18() -> None:
    # eta_k = scale / (k + d * tau_over_gamma).
    rates = one_pass_module._harmonic_learning_rates(
        n=7,
        d=1,
        scale=2.0,
        tau_over_gamma=2.1,
    )

    np.testing.assert_allclose(rates, 2.0 / (np.arange(1, 8) + 1 * 2.1))


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_equation_4_2_polynomial_suffix_variances(alpha: float) -> None:
    paper_rho = 0.7
    budget = ZCDPBudget(rho=paper_rho**2 / 2.0)
    rates = one_pass_module._polynomial_learning_rates(
        n=5,
        learning_rate_scale=3.0,
        alpha=alpha,
    )
    noise = one_pass_module._equation_4_2_noise_multipliers(rates, budget)

    suffix_variances = np.cumsum(noise[::-1] ** 2)[::-1]
    np.testing.assert_allclose(
        suffix_variances,
        rates**2 / paper_rho**2,
        rtol=1e-12,
        atol=1e-15,
    )
    if alpha == 0.0:
        np.testing.assert_array_equal(noise[:-1], np.zeros(4))
        assert noise[-1] == pytest.approx(3.0 / (5.0 * paper_rho))
    elif alpha == 0.5:
        np.testing.assert_allclose(
            noise[:-1],
            np.full(4, 3.0 / (5.0**1.5 * paper_rho)),
        )
        assert noise[-1] == 0.0
    else:
        assert rates[-1] == 0.0
        assert noise[-1] == 0.0


@pytest.mark.parametrize(
    ("algorithm", "n", "d", "expected_rates", "expected_clip"),
    [
        # Equation (5.18) at scale=2, tau_over_gamma=2.1, d=1: 2 / (k + 2.1).
        pytest.param(
            OnePassDPGD(schedule="harmonic", scale=2.0, tau_over_gamma=2.1),
            7,
            1,
            2.0 / (np.arange(1, 8) + 2.1),
            0.1,
            id="harmonic",
        ),
        # Equations (4.4)-(5.1) at scale=8, alpha=1, n=4: 2 * (1 - k/4).
        pytest.param(
            OnePassDPGD(schedule="polynomial", c=0.2, scale=8.0, alpha=1.0),
            4,
            2,
            np.array([1.5, 1.0, 0.5, 0.0]),
            0.2 * math.sqrt(2.0),
            id="polynomial",
        ),
    ],
)
def test_public_run_wires_rates_noise_and_clipping(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: OnePassDPGD,
    n: int,
    d: int,
    expected_rates: Array1D,
    expected_clip: float,
) -> None:
    metadata, input = _instance(n=n, d=d)
    observed: list[tuple[Array1D, Array1D, float]] = []

    def fake_algorithm_1(
        X: Array2D,
        y: Array1D,
        learning_rates: Array1D,
        noise_multipliers: Array1D,
        clipping_norm: float,
        seed: int,
    ) -> Array1D:
        del X, y, seed
        observed.append((learning_rates, noise_multipliers, clipping_norm))
        return np.zeros(d)

    monkeypatch.setattr(one_pass_module, "_run_algorithm_1", fake_algorithm_1)

    algorithm.run(metadata, input, ZCDPBudget(rho=0.7**2 / 2), seed=0)

    ((learning_rates, noise_multipliers, clipping_norm),) = observed
    np.testing.assert_allclose(learning_rates, expected_rates)
    # Equation (4.2) at the paper's rho = sqrt(2 * zCDP rho) = 0.7.
    suffix_variances = np.cumsum(noise_multipliers[::-1] ** 2)[::-1]
    np.testing.assert_allclose(suffix_variances, expected_rates**2 / 0.7**2)
    assert clipping_norm == pytest.approx(expected_clip)


def test_algorithm_1_update_and_final_release(
    fake_randomness: FakeRandomness,
) -> None:
    # A constant rate schedule (alpha=0) at scale 1 and n=2 gives rates
    # (1/2, 1/2); Equation (4.2) then puts all noise on the last step,
    # sqrt((1/2)**2 - 0) / 0.5 = 1 at the paper's rho = sqrt(2 * 0.125).
    metadata = LinearRegressionMetadata(
        n_samples=2,
        design_domain=EuclideanSpace(dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(
        X=np.array([[2.0], [1.0]]),
        y=np.array([4.0, 1.0]),
    )
    fake_randomness.script("permutation", lambda _parameters: np.array([0, 1]))
    fake_randomness.script(
        "normal", draws_in_order([np.array([7.0]), np.array([0.05])])
    )

    output = OnePassDPGD(schedule="polynomial", c=2.0, scale=1.0, alpha=0.0).run(
        metadata,
        input,
        ZCDPBudget(rho=0.125),
        seed=7,
    )

    # The first gradient 2*(0 - 4) = -8 clips to -2 and its rate caps at
    # min(1/2, 2/|x|**2) = 1/2, so theta = 1 with no first-step noise. The
    # second gradient is 1*(1 - 1) = 0, and Equation (4.1) adds
    # 2 * C_clip * 1 * 0.05 = 0.2.
    assert output.beta == pytest.approx(np.array([1.2]))
    # The projection permutation and Algorithm 1 draw from distinct child
    # streams, neither of which is the offered seed.
    assert len(set(fake_randomness.seeds)) == 2
    assert 7 not in fake_randomness.seeds


def test_zero_row_takes_no_gradient_step_and_keeps_its_scheduled_noise(
    fake_randomness: FakeRandomness,
) -> None:
    # n=1 at scale 1/2 has the single rate 1/2 and noise multiplier
    # sqrt((1/2)**2) / 0.5 = 1, so only the noise 2 * C_clip * 1 * 0.2 moves
    # theta; a zero row has no 2/|x|**2 cap to divide by.
    metadata = LinearRegressionMetadata(
        n_samples=1,
        design_domain=EuclideanSpace(dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(X=np.zeros((1, 1)), y=np.ones(1))
    fake_randomness.script("permutation", lambda _parameters: np.array([0]))
    fake_randomness.script("normal", draws_in_order([np.array([0.2])]))

    output = OnePassDPGD(schedule="polynomial", c=2.0, scale=0.5, alpha=0.0).run(
        metadata,
        input,
        ZCDPBudget(rho=0.125),
        seed=7,
    )

    assert output.beta == pytest.approx(np.array([0.8]))


@pytest.mark.parametrize(
    "algorithm",
    [HARMONIC_ROSTER, POLYNOMIAL_ROSTER],
)
def test_run_applies_one_aligned_seeded_permutation(
    algorithm: OnePassDPGD,
    monkeypatch: pytest.MonkeyPatch,
    fake_randomness: FakeRandomness,
) -> None:
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=EuclideanSpace(dim=1),
        response_domain=EuclideanSpace(dim=1),
    )
    input = LinearRegressionInput(
        X=np.array([[10.0], [20.0], [30.0]]),
        y=np.array([1.0, 2.0, 3.0]),
    )
    recorded_X: Array2D | None = None
    recorded_y: Array1D | None = None
    recorded_seed: int | None = None

    def fake_algorithm_1(
        X: Array2D,
        y: Array1D,
        learning_rates: Array1D,
        noise_multipliers: Array1D,
        clipping_norm: float,
        seed: int,
    ) -> Array1D:
        nonlocal recorded_X, recorded_y, recorded_seed
        recorded_X = X
        recorded_y = y
        recorded_seed = seed
        assert learning_rates.shape == (3,)
        assert noise_multipliers.shape == (3,)
        assert clipping_norm == pytest.approx(0.1)
        return np.zeros(1)

    fake_randomness.script(
        "permutation", lambda _parameters: np.array([2, 0, 1], dtype=np.int64)
    )
    monkeypatch.setattr(one_pass_module, "_run_algorithm_1", fake_algorithm_1)

    algorithm.run(metadata, input, ZCDPBudget(rho=0.2), seed=7)

    assert recorded_X is not None
    assert recorded_y is not None
    np.testing.assert_array_equal(recorded_X[:, 0], np.array([30.0, 10.0, 20.0]))
    np.testing.assert_array_equal(recorded_y, np.array([3.0, 1.0, 2.0]))
    # The projection permutation and Algorithm 1 are separately accounted, so
    # each runs on its own child stream rather than the offered seed.
    (permutation_seed,) = fake_randomness.seeds
    assert recorded_seed != permutation_seed
    assert 7 not in (permutation_seed, recorded_seed)


@pytest.mark.parametrize(
    "budget",
    [ApproxDPBudget(eps=1.0, delta=1e-5), ZCDPBudget(rho=0.2)],
)
@pytest.mark.parametrize(
    "algorithm",
    [HARMONIC_ROSTER, POLYNOMIAL_ROSTER],
)
def test_finite_run_spends_once_as_zcdp(
    algorithm: OnePassDPGD,
    budget: ApproxDPBudget | ZCDPBudget,
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _instance()

    algorithm.run(metadata, input, budget, seed=0)

    assert len(accounting_spy.calls) == 1
    assert accounting_spy.calls[0].entry_point == "spend_as_zcdp"
    assert accounting_spy.calls[0].budget == budget


@pytest.mark.parametrize(
    "algorithm",
    [HARMONIC_ROSTER, POLYNOMIAL_ROSTER],
)
def test_finite_pure_dp_is_refused_at_accounting_seam(
    algorithm: OnePassDPGD,
    accounting_spy: AccountingSpy,
) -> None:
    metadata, input = _instance()
    budget = PureDPBudget(eps=1.0)

    with pytest.raises(InvalidRegime, match="PureDPBudget"):
        algorithm.run(metadata, input, budget, seed=0)

    assert len(accounting_spy.calls) == 1
    assert accounting_spy.calls[0].entry_point == "spend_as_zcdp"


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        ApproxDPBudget(eps=math.inf, delta=1e-5),
        ZCDPBudget(rho=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [HARMONIC_ROSTER, POLYNOMIAL_ROSTER],
)
def test_unconstrained_budgets_execute_zero_noise_limit(
    algorithm: OnePassDPGD,
    budget: DPBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, input = _instance(n=4)

    def fake_algorithm_1(
        X: Array2D,
        y: Array1D,
        learning_rates: Array1D,
        noise_multipliers: Array1D,
        clipping_norm: float,
        seed: int,
    ) -> Array1D:
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert learning_rates.shape == (4,)
        np.testing.assert_array_equal(noise_multipliers, np.zeros(4))
        assert clipping_norm == pytest.approx(0.1 * math.sqrt(2.0))
        assert isinstance(seed, int)
        return np.zeros(2)

    monkeypatch.setattr(one_pass_module, "_run_algorithm_1", fake_algorithm_1)

    output = algorithm.run(metadata, input, budget, seed=0)

    np.testing.assert_array_equal(output.beta, np.zeros(2))
