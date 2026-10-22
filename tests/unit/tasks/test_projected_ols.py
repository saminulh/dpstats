"""Tests for Sheffet's projected OLS through its public ``run``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.algorithm import PrivacyInterpretation
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import BoxDomain
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array2D
from dpstats.tasks import LinearRegression, list_built_in_algorithms
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.algorithms import ProjectedOLS
from tests.unit.tasks.task_builders import box, lr_metadata


class _RecordingRNG:
    """A generator whose Laplace draw is fixed and whose Gaussian matrix is known."""

    def __init__(self, gate_noise: float = 0.0) -> None:
        self.gate_noise = gate_noise
        self.laplace_scales: list[float] = []
        self.normal_sizes: list[tuple[int, ...]] = []

    def laplace(self, *, scale: float) -> float:
        self.laplace_scales.append(scale)
        return self.gate_noise

    def normal(self, *, size: tuple[int, ...]) -> Array2D:
        self.normal_sizes.append(size)
        return _known_gaussian(size)


def _known_gaussian(size: tuple[int, ...]) -> Array2D:
    return np.arange(math.prod(size), dtype=float).reshape(size) / 10 + 0.1


def _use_rng(monkeypatch: pytest.MonkeyPatch, rng: _RecordingRNG) -> None:
    monkeypatch.setattr(np.random, "default_rng", lambda seed: rng)


def _bounded_instance() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(60, 2))
    y = X @ np.array([0.75, -0.4])
    metadata = lr_metadata(60, design=box((-1, 1), (-1, 1)), response=box((-2, 2)))
    return metadata, LinearRegressionInput(X=X, y=y)


def test_is_a_built_in_private_linear_regression_algorithm() -> None:
    assert list_built_in_algorithms(LinearRegression)["ProjectedOLS"] is ProjectedOLS
    assert ProjectedOLS().privacy_interpretation is PrivacyInterpretation.STANDARD


@pytest.mark.parametrize(
    "budget",
    [NO_PRIVACY, ApproxDPBudget(eps=math.inf, delta=1e-4)],
)
def test_unconstrained_budget_recovers_a_noiseless_fit_exactly(
    budget: DPBudget,
) -> None:
    # In the infinite-epsilon limit no identity block is appended, and a
    # Gaussian projection of consistent equations y = X beta keeps them
    # consistent, so least squares on the projection returns beta.
    metadata, input_data = _bounded_instance()

    output = ProjectedOLS().run(metadata, input_data, budget, seed=11)

    np.testing.assert_allclose(output.beta, [0.75, -0.4], rtol=1e-9)


def test_insufficient_conditioning_takes_the_altered_r_25_branch_with_paper_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A zero covariate column makes the smallest singular value zero, so with
    # no gate noise no unaltered size passes and the fallback appends w I.
    X = np.zeros((3, 1))
    y = np.array([0.5, -0.25, 1.0])
    metadata = lr_metadata(3, design=box((-1, 1)), response=box((-1, 1)))
    rng = _RecordingRNG()
    _use_rng(monkeypatch, rng)

    output = ProjectedOLS().run(
        metadata,
        LinearRegressionInput(X=X, y=y),
        ApproxDPBudget(eps=4.0, delta=0.1),
        seed=0,
    )

    # B^2 = 1 + 1. Algorithm 1: Laplace scale 4 B^2 / eps and
    # w^2 = (8 B^2 / eps) (sqrt(2 r log(8/delta)) + 2 log(8/delta)) at r = 25.
    log_term = math.log(80)
    w = math.sqrt(4 * (math.sqrt(50 * log_term) + 2 * log_term))
    gaussian = _known_gaussian((25, 5))
    projected_x = w * gaussian[:, 3]
    projected_y = gaussian[:, :3] @ y + w * gaussian[:, 4]
    assert rng.laplace_scales == [pytest.approx(2.0)]
    assert rng.normal_sizes == [(25, 5)]
    np.testing.assert_allclose(
        output.beta,
        [projected_x @ projected_y / (projected_x @ projected_x)],
        rtol=1e-9,
    )


def test_selects_the_largest_projection_size_that_passes_the_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Consistent equations have a zero smallest singular value, so the gate
    # noise alone sets the private value: q = 0 - (-1000).
    n = 200
    X = np.random.default_rng(3).uniform(-1, 1, size=(n, 2))
    beta = np.array([0.75, -0.4])
    metadata = lr_metadata(n, design=box((-1, 1), (-1, 1)), response=box((-1.2, 1.2)))
    epsilon, delta = 0.5, 0.5
    rng = _RecordingRNG(gate_noise=-1000.0)
    _use_rng(monkeypatch, rng)

    output = ProjectedOLS().run(
        metadata,
        LinearRegressionInput(X=X, y=X @ beta),
        ApproxDPBudget(eps=epsilon, delta=delta),
        seed=0,
    )

    # Oracle: substitute each candidate r into Algorithm 1's strict test.
    B_squared = 2 + 1.2**2
    log_term = math.log(8 / delta)

    def passes(r: int) -> bool:
        w_squared = (8 * B_squared / epsilon) * (
            math.sqrt(2 * r * log_term) + 2 * log_term
        )
        return 1000.0 > w_squared + 4 * B_squared * math.log(1 / delta) / epsilon

    expected_r = max(r for r in range(1, n + 1) if passes(r))
    assert 25 <= expected_r < n
    assert rng.normal_sizes == [(expected_r, n)]
    np.testing.assert_allclose(output.beta, beta, rtol=1e-6)


def test_size_cap_is_the_public_n_and_small_data_falls_back_to_r_25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Well-conditioned data at the infinite-epsilon limit pass at every size,
    # so the cap n = 10 binds; 10 < 25 forces the altered r = 25 branch.
    X = np.tile(np.eye(2), (5, 1))
    y = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 1.0, 0.0, 0.5, -1.0, 0.25])
    metadata = lr_metadata(10, design=box((-1, 1), (-1, 1)), response=box((-1, 1)))
    rng = _RecordingRNG()
    _use_rng(monkeypatch, rng)

    ProjectedOLS().run(metadata, LinearRegressionInput(X=X, y=y), NO_PRIVACY, seed=0)

    assert rng.normal_sizes == [(25, 13)]


def test_size_cap_is_the_declared_n_samples_for_well_conditioned_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At the infinite-epsilon limit every size passes, so the selected size is
    # the public cap: the declared n_samples.
    metadata, input_data = _bounded_instance()
    noisy = LinearRegressionInput(
        X=input_data.X, y=input_data.y + np.linspace(-0.1, 0.1, 60)
    )
    rng = _RecordingRNG()
    _use_rng(monkeypatch, rng)

    ProjectedOLS().run(metadata, noisy, NO_PRIVACY, seed=0)

    assert rng.normal_sizes == [(metadata.n_samples, 60)]


@pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
def test_requires_native_approximate_budget(
    budget: PureDPBudget | ZCDPBudget,
) -> None:
    metadata, input_data = _bounded_instance()

    with pytest.raises(InvalidRegime, match="ApproxDPBudget"):
        ProjectedOLS().run(metadata, input_data, budget, seed=0)


@pytest.mark.parametrize(
    ("design_domain", "response_domain", "message"),
    [
        (2, box((-2, 2)), "covariate"),
        (box((-1, 1), (-1, 1)), 1, "response"),
    ],
)
def test_requires_public_covariate_and_response_bounds(
    design_domain: BoxDomain | int,
    response_domain: BoxDomain | int,
    message: str,
) -> None:
    _, input_data = _bounded_instance()

    with pytest.raises(InvalidRegime, match=message):
        ProjectedOLS().run(
            lr_metadata(60, design=design_domain, response=response_domain),
            input_data,
            ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )


def test_seed_determines_the_private_output() -> None:
    metadata, input_data = _bounded_instance()
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)

    first = ProjectedOLS().run(metadata, input_data, budget, seed=7)
    repeat = ProjectedOLS().run(metadata, input_data, budget, seed=7)
    other = ProjectedOLS().run(metadata, input_data, budget, seed=8)

    np.testing.assert_array_equal(first.beta, repeat.beta)
    assert not np.array_equal(first.beta, other.beta)
