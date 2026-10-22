"""Tests for m_estimation gradient-descent algorithms."""

import math

import numpy as np
import pytest
from pydantic import PrivateAttr
from scipy.optimize import brentq
from scipy.stats import norm

from dpstats.budgets.accounting import (
    compute_noise_multiplier_for_subsampled_gaussian,
)
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import Domain
from dpstats.task import Array1D, ArrayND
from dpstats.tasks.m_estimation.algorithms.dp_gradient_descent import (
    DPSGD,
)
from dpstats.tasks.m_estimation.algorithms.erm import (
    ERM,
)
from dpstats.tasks.m_estimation.algorithms.gradient_descent import (
    GD,
)
from dpstats.tasks.m_estimation.algorithms.sgd import (
    SGD,
)
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    SquaredResidualLoss,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationLoss,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    SquaredErrorLoss,
)
from dpstats.tasks.m_estimation.losses import LossProperties
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order, zero_draw
from tests.unit.tasks.task_builders import box, m_estimation_metadata


def _gdp_noise_multiplier(eps: float, delta: float, num_steps: int) -> float:
    """Noise multiplier of ``num_steps`` composed unit-sensitivity Gaussians.

    ``T`` Gaussian releases at noise multiplier ``sigma`` are exactly
    ``sqrt(T) / sigma``-GDP (Dong, Roth and Su 2022, Cor. 3.3), and ``mu``-GDP
    is ``(eps, delta)``-DP iff ``delta >= Phi(-eps/mu + mu/2) -
    e^eps Phi(-eps/mu - mu/2)`` (Cor. 2.13); the tightest ``mu`` solves that
    with equality.
    """

    def excess_delta(mu: float) -> float:
        return (
            norm.cdf(-eps / mu + mu / 2)
            - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)
            - delta
        )

    return math.sqrt(num_steps) / brentq(excess_delta, 1e-6, 50.0)


def _ill_scaled_regression_instance(scale: float = 1e4, n: int = 200, d: int = 3):
    """Build an m-estimation instance whose curvature breaks fixed-step GD.

    Squared-residual loss with covariates of magnitude ``scale`` has a
    smoothness constant ~scale**2, so any fixed step size that is safe on
    normalized data diverges exponentially here (the GD/SGD failure mode on
    unnormalized data).
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, d)) * scale
    beta = np.array([1.5, -2.0, 0.5])
    y = X @ beta + rng.standard_normal(n)
    data = np.column_stack([X, y])
    metadata = m_estimation_metadata(n, SquaredResidualLoss(), parameter=d, data=d + 1)
    return metadata, MEstimationInput(data=data), beta


class TestDPSGDTypedBudgetAccounting:
    """Pin typed budgets through the specialized composition calls."""

    def test_full_batch_passes_exact_approx_budget_to_composition(
        self, accounting_spy: AccountingSpy
    ):
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=20, d=3)
        budget = ApproxDPBudget(eps=2.0, delta=1e-6)

        DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=3).run(
            metadata, input_data, budget=budget, seed=0
        )

        assert accounting_spy.calls == (
            AccountingCall(
                "compute_noise_multiplier_for_composed_gaussian", budget, num_steps=3
            ),
        )

    def test_subsampled_passes_exact_zcdp_budget_to_composition(
        self, accounting_spy: AccountingSpy
    ):
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=20, d=3)
        budget = ZCDPBudget(rho=0.7)

        DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=3, subsample_rate=0.5).run(
            metadata, input_data, budget=budget, seed=0
        )

        assert accounting_spy.calls == (
            AccountingCall(
                "compute_noise_multiplier_for_subsampled_gaussian",
                budget,
                sampling_rate=0.5,
                num_steps=3,
            ),
        )

    def test_refuses_pure_budget_at_gaussian_accounting_boundary(
        self, accounting_spy: AccountingSpy
    ):
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=20, d=3)
        with pytest.raises(InvalidRegime, match="PureDPBudget"):
            DPSGD(num_steps=1).run(
                metadata, input_data, budget=PureDPBudget(eps=1.0), seed=0
            )
        assert accounting_spy.calls == (
            AccountingCall(
                "compute_noise_multiplier_for_composed_gaussian",
                PureDPBudget(eps=1.0),
                num_steps=1,
            ),
        )


class TestDPSGDStep:
    """One noisy step clips each row, averages, and adds calibrated noise."""

    def test_each_row_is_clipped_before_averaging(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """Rows 4 and -0.5 clip to 1 and -0.5 at ``grad_clip = 1``.

        Their mean 0.25 takes one step of size 0.4 to ``-0.1``; clipping the
        mean 1.75 instead would give ``-0.4``.
        """
        metadata = m_estimation_metadata(2, _IdentityGradientLoss())
        fake_randomness.script("normal", zero_draw)

        output = DPSGD(C_grad_clip=1.0, eta=0.4, num_steps=1).run(
            metadata,
            MEstimationInput(data=np.array([[4.0], [-0.5]])),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )

        np.testing.assert_allclose(output.theta, [-0.1])

    def test_full_batch_noise_is_the_add_remove_composed_gaussian(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """The noise on the gradient mean has scale ``sigma * grad_clip / n``.

        ``grad_clip = C * sqrt(d)`` is the add/remove sensitivity of the
        clipped-gradient sum, and ``sigma`` meets the budget over all steps.
        """
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=200, d=3)
        fake_randomness.script("normal", zero_draw)
        fake_randomness.forget()

        DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=10).run(
            metadata, input_data, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )

        expected = _gdp_noise_multiplier(1.0, 1e-5, 10) * math.sqrt(3.0) / 200
        scales = fake_randomness.parameters_of("normal", "scale")
        assert len(scales) == 10
        assert scales == pytest.approx([expected] * 10, rel=1e-6)

    def test_subsampled_step_normalizes_by_the_expected_lot_size(
        self, fake_randomness: FakeRandomness, accounting_spy: AccountingSpy
    ) -> None:
        """The lot sum is divided by ``q * n``, not by the realized lot size.

        The scripted uniforms admit rows 0, 1 and 3 at ``q = 0.5``: their sum
        ``(1, 2)`` over ``q * n = 2`` is ``(0.5, 1)``, and a step of 0.5 lands
        at ``(-0.25, -0.5)``. The noise scale is the amplified multiplier times
        ``grad_clip / (q * n)``.
        """
        data = np.array([[0.5, 0.5], [0.25, 0.75], [9.0, 9.0], [0.25, 0.75]])
        metadata = m_estimation_metadata(4, _IdentityGradientLoss(), 2, 2)
        fake_randomness.script(
            "random", draws_in_order([np.array([0.1, 0.2, 0.9, 0.3])])
        )
        fake_randomness.script("normal", zero_draw)
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)

        output = DPSGD(C_grad_clip=10.0, eta=0.5, num_steps=1, subsample_rate=0.5).run(
            metadata, MEstimationInput(data=data), budget=budget, seed=0
        )

        np.testing.assert_allclose(output.theta, [-0.25, -0.5])
        assert accounting_spy.calls == (
            AccountingCall(
                "compute_noise_multiplier_for_subsampled_gaussian",
                budget,
                sampling_rate=0.5,
                num_steps=1,
            ),
        )
        multiplier = compute_noise_multiplier_for_subsampled_gaussian(budget, 0.5, 1)
        (scale,) = fake_randomness.parameters_of("normal", "scale")
        assert scale == pytest.approx(multiplier * 10.0 * math.sqrt(2.0) / 2.0)


class TestGDDivergenceGuard:
    """GD/SGD defaults must not produce divergent outputs on ill-scaled data.

    Fixed default step sizes make GD/SGD diverge on ill-scaled data, returning
    astronomically large or ±inf errors reported as "success"; the defaults must
    guard against this.
    """

    def test_gradient_descent_stays_finite_and_descends(self):
        from dpstats.tasks.m_estimation.algorithms.gradient_descent import (
            GD,
        )

        metadata, input, beta = _ill_scaled_regression_instance()
        out = GD().run(metadata, input, budget=NO_PRIVACY, seed=0)
        assert np.all(np.isfinite(out.theta))
        # The line search guarantees the mean loss never exceeds the start's.
        loss_at = lambda th: float(np.mean(metadata.loss(input.data, th)))
        assert loss_at(out.theta) <= loss_at(np.zeros(3))
        # And it should actually make progress on this well-posed problem.
        assert np.linalg.norm(out.theta - beta) < np.linalg.norm(beta)

    def test_sgd_stays_finite_and_improves(self):
        from dpstats.tasks.m_estimation.algorithms.sgd import (
            SGD,
        )

        metadata, input, beta = _ill_scaled_regression_instance()
        out = SGD().run(metadata, input, budget=NO_PRIVACY, seed=0)
        assert np.all(np.isfinite(out.theta))
        loss_at = lambda th: float(np.mean(metadata.loss(input.data, th)))
        assert loss_at(out.theta) <= loss_at(np.zeros(3))


class _TestLoss(MEstimationLoss):
    """Common unknown-property contract for algorithm wiring fixtures."""

    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        del data_domain, parameter_domain
        return LossProperties(
            convex=False,
            generalized_linear=False,
            continuously_twice_differentiable=False,
            loss_lipschitz_constant=math.inf,
            gradient_lipschitz_constant=math.inf,
        )


class _IdentityGradientLoss(_TestLoss):
    """Zero-valued loss whose selected gradient is the supplied record."""

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        del theta
        return np.zeros(len(data)) if reduction == "none" else 0.0

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        del theta
        if reduction == "none":
            return data
        gradient_sum = np.sum(data, axis=0)
        return gradient_sum if reduction == "sum" else gradient_sum / len(data)


class _RecordingGradientLoss(_TestLoss):
    """Zero-valued loss that records the row IDs supplied to its gradient."""

    dimension: int
    _recorded_batches: list[np.ndarray] = PrivateAttr(default_factory=list)

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        del theta
        return np.zeros(len(data)) if reduction == "none" else 0.0

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        del theta
        self._recorded_batches.append(data[:, 0].astype(int))
        if reduction == "none":
            return np.zeros((len(data), self.dimension))
        return np.zeros(self.dimension)


def _recording_instance(n: int, d: int, recorder: list):
    """An m-estimation instance whose loss records each batch's row ids.

    Row ``i`` carries its own id in column 0 (and zeros elsewhere). The gradient
    appends the integer ids of the rows it is handed to ``recorder`` and returns
    a zero gradient matrix, so theta stays at 0 (the lot composition can run for
    any ``num_steps`` without diverging). This exposes the Poisson lot the
    mechanism actually feeds the gradient oracle.
    """
    data = np.zeros((n, d))
    data[:, 0] = np.arange(n)

    loss = _RecordingGradientLoss(dimension=d)
    loss._recorded_batches = recorder
    return m_estimation_metadata(n, loss, d, d), MEstimationInput(data=data)


class TestDPSGDPoissonCorrectness:
    """The lot must be TRUE independent Poisson, not a fixed-size sample.

    Privacy-load-bearing: the amplification theorem assumes each row joins
    the lot i.i.d. with probability q. A fixed-size lot (exactly q*n rows) would
    have near-zero lot-size variance and accounting it as Poisson would be
    under-protective. This pins the Binomial(n, q) signature.
    """

    def test_lot_is_binomial_not_fixed_size(self):
        # eta=0 freezes theta, so each of the many steps draws a fresh
        # independent Poisson lot at the same theta -> num_steps independent
        # draws from one run (one calibration call, not one per draw).
        n, d, q, draws = 500, 2, 0.3, 400
        recorder: list = []
        metadata, input = _recording_instance(n, d, recorder)
        alg = DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=draws, subsample_rate=q)
        alg.run(metadata, input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0)

        lot_sizes = np.array([len(rows) for rows in recorder])
        assert len(lot_sizes) == draws
        # Mean lot size ~ q*n.
        assert abs(lot_sizes.mean() - q * n) < 0.1 * q * n
        # Variance ~ n*q*(1-q) (Binomial), NOT ~0 (which fixed-size would give).
        expected_var = n * q * (1.0 - q)
        assert lot_sizes.var() > 0.4 * expected_var
        assert lot_sizes.var() < 2.5 * expected_var

    def test_per_row_inclusion_frequency_matches_q(self):
        n, d, q, draws = 200, 2, 0.25, 600
        recorder: list = []
        metadata, input = _recording_instance(n, d, recorder)
        alg = DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=draws, subsample_rate=q)
        alg.run(metadata, input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0)

        counts = np.zeros(n)
        for rows in recorder:
            counts[rows] += 1
        freq = counts / draws
        # Every row included roughly q of the time; the mean is tight.
        assert abs(freq.mean() - q) < 0.02
        # No row is deterministically in or out (rules out a structured sample).
        assert freq.min() > 0.0
        assert freq.max() < 1.0


class TestDPSGDSubsampleMemory:
    """Small q on a large-n instance materializes only ~q*n rows per step."""

    def test_lot_path_materializes_expected_size(self):
        n, d, q = 1_000_000, 2, 0.002  # expected lot ~ 2000 rows
        recorder: list = []
        metadata, input = _recording_instance(n, d, recorder)
        alg = DPSGD(C_grad_clip=1.0, eta=0.0, num_steps=3, subsample_rate=q)
        out = alg.run(
            metadata, input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=0
        )
        assert np.all(np.isfinite(out.theta))
        assert len(recorder) == 3
        for rows in recorder:
            # The gradient matrix being clipped is ~q*n, never the full n.
            assert abs(len(rows) - q * n) < 0.5 * q * n
            assert len(rows) < n / 100


class TestDPSGDIterateAggregation:
    """iterate_aggregation post-processes the trajectory (privacy-neutral).

    The released transcript is the same ``num_steps`` noisy gradient steps for
    every mode; aggregation only selects which deterministic function of that
    trajectory is returned, so the calibration is identical across all three
    modes.
    """

    @pytest.mark.parametrize(
        "budget",
        [ApproxDPBudget(eps=1.0, delta=1e-5), ZCDPBudget(rho=0.5)],
        ids=["approx-dp", "zcdp"],
    )
    @pytest.mark.parametrize("q", [1.0, 0.5], ids=["full-batch", "poisson"])
    def test_aggregation_modes_request_identical_calibration(
        self,
        accounting_spy: AccountingSpy,
        budget: DPBudget,
        q: float,
    ) -> None:
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=20, d=3)

        for mode in ("last", "tail_average", "polyak"):
            DPSGD(
                C_grad_clip=1.0,
                eta=0.0,
                num_steps=3,
                subsample_rate=q,
                iterate_aggregation=mode,
            ).run(metadata, input_data, budget=budget, seed=0)

        expected = (
            AccountingCall(
                "compute_noise_multiplier_for_composed_gaussian", budget, num_steps=3
            )
            if q == 1.0
            else AccountingCall(
                "compute_noise_multiplier_for_subsampled_gaussian",
                budget,
                sampling_rate=q,
                num_steps=3,
            )
        )
        assert accounting_spy.calls == (expected,) * 3

    def test_aggregates_match_public_prefix_runs(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """Aggregation combines the same public trajectory without mirroring its loop.

        Scripted noise of 0.3 per coordinate, whatever scale is asked for, makes
        a run's trajectory independent of ``num_steps``, so a shorter run's
        last iterate is a prefix iterate of a longer one.
        """
        metadata, input_data, _ = _ill_scaled_regression_instance(scale=1.0, n=200, d=3)
        fake_randomness.script("normal", lambda draw: np.full(draw["size"], 0.3))

        for q in (1.0, 0.5):

            def run(num_steps: int, mode: str) -> np.ndarray:
                return (
                    DPSGD(
                        C_grad_clip=1.0,
                        eta=0.05,
                        num_steps=num_steps,
                        subsample_rate=q,
                        iterate_aggregation=mode,
                    )
                    .run(
                        metadata,
                        input_data,
                        budget=NO_PRIVACY,
                        seed=2,
                    )
                    .theta
                )

            theta_1 = run(1, "last")
            theta_2 = run(2, "last")
            theta_3 = run(3, "last")

            np.testing.assert_allclose(run(1, "tail_average"), theta_1)
            np.testing.assert_allclose(run(1, "polyak"), theta_1)
            np.testing.assert_allclose(run(2, "tail_average"), theta_2)
            np.testing.assert_allclose(
                run(2, "polyak"),
                np.mean([theta_1, theta_2], axis=0),
            )
            np.testing.assert_allclose(
                run(3, "tail_average"),
                np.mean([theta_2, theta_3], axis=0),
            )
            np.testing.assert_allclose(
                run(3, "polyak"),
                np.mean([theta_1, theta_2, theta_3], axis=0),
            )


class TestParameterDomainContract:
    """ERM refuses constrained domains; projected methods honor them."""

    def test_erm_refuses_a_constrained_parameter_domain_before_input_access(self):
        metadata = m_estimation_metadata(3, SquaredErrorLoss(), parameter=box((-1, 1)))

        with pytest.raises(InvalidRegime, match="unconstrained"):
            ERM().run(metadata, object(), budget=NO_PRIVACY, seed=0)


class _PullTowardLoss(_TestLoss):
    """One-dimensional quadratic loss with observable query coordinates.

    Records query parameters to check that projected optimizers stay in-domain.
    """

    target: float
    scale: float = 1.0
    _visited: list[float] = PrivateAttr(default_factory=list)

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        self._visited.append(float(theta[0]))
        value = self.scale * float((theta[0] - self.target) ** 2)
        if reduction == "none":
            return np.full(len(data), value)
        if reduction == "sum":
            return len(data) * value
        return value

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        self._visited.append(float(theta[0]))
        gradient = np.array([2.0 * self.scale * (theta[0] - self.target)])
        if reduction == "none":
            return np.tile(gradient, (len(data), 1))
        if reduction == "sum":
            return len(data) * gradient
        return gradient


def _boxed_problem(loss: MEstimationLoss):
    metadata = m_estimation_metadata(4, loss, parameter=box((0.0, 2.0)))
    return metadata, MEstimationInput(data=np.zeros((4, 1)))


class TestProjectedParameterDomain:
    """Every visited iterate stays in a bounded declared parameter domain."""

    def test_gd_stops_on_the_boundary_nearest_an_outside_minimizer(self):
        """The unconstrained minimizer 100 lies far outside ``[0, 2]``."""
        loss = _PullTowardLoss(target=100.0, scale=1000.0)
        metadata, input_data = _boxed_problem(loss)
        output = GD(eta=1.0, num_steps=10).run(
            metadata, input_data, budget=NO_PRIVACY, seed=0
        )

        np.testing.assert_array_equal(output.theta, [2.0])
        assert all(0.0 <= value <= 2.0 for value in loss._visited)

    def test_sgd_iterates_and_output_stay_in_domain(self):
        loss = _PullTowardLoss(target=3.0)
        metadata, input_data = _boxed_problem(loss)

        output = SGD(eta=1.0, num_steps=20).run(
            metadata, input_data, budget=NO_PRIVACY, seed=0
        )

        assert 0.0 <= float(output.theta[0]) <= 2.0
        assert all(0.0 <= value <= 2.0 for value in loss._visited)

    @pytest.mark.parametrize("aggregation", ["last", "tail_average", "polyak"])
    def test_dpsgd_iterates_and_output_stay_in_domain(self, aggregation):
        loss = _PullTowardLoss(target=3.0)
        metadata, input_data = _boxed_problem(loss)

        output = DPSGD(
            C_grad_clip=1.0,
            eta=0.5,
            num_steps=8,
            iterate_aggregation=aggregation,
        ).run(metadata, input_data, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=3)

        assert 0.0 <= float(output.theta[0]) <= 2.0
        assert all(0.0 <= value <= 2.0 for value in loss._visited)


class TestGaussianSimulatedInstanceConditioning:
    """The simulated-instance producer's conditioning knob needs a second dimension."""

    def test_one_dimensional_conditioning_is_refused(self):
        from dpstats.simulations.m_estimation import simulate_gaussian

        with pytest.raises(ValueError, match="requires dim >= 2"):
            simulate_gaussian(seed=0, dim=1, n_samples=20, condition_num=10.0)
