"""Tests for the mean-estimation Task, its algorithms and producers."""

import numpy as np
import pytest

from dpstats.tasks import list_built_in_algorithms
from dpstats.simulations.mean_estimation import (
    simulate_clipped_laplacian,
    simulate_clipped_normal,
)
from dpstats.domains import LpBallDomain, OneHotDomain
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
)
from tests.fixtures.randomness_fake import (
    FakeRandomness,
    draws_in_order,
    zero_draw,
)
from tests.unit.tasks.task_builders import box, mean_metadata


def _bounded_zero_instance(mean_estimation_task):
    metadata = mean_metadata(20, LpBallDomain.at_origin(p=2, radius=2.0, dim=2))
    input_data = mean_estimation_task.Input(X=np.zeros((20, 2)))
    return metadata, input_data


class TestSampleMeanAlgorithm:
    """Tests for SampleMean algorithm."""

    def test_returns_hand_computed_coordinate_means(self, mean_estimation_task):
        task = mean_estimation_task
        X = np.array([[1.0, 7.0], [3.0, 3.0], [8.0, 2.0]])
        metadata = mean_metadata(3, 2)
        input_data = task.Input(X=X)
        SampleMean = list_built_in_algorithms(task)["SampleMean"]
        output = SampleMean().run(metadata, input_data, budget=NO_PRIVACY, seed=42)

        np.testing.assert_allclose(output.mean_est, [4.0, 4.0])


class TestBoundedDomainAlgorithms:
    """Tests for algorithms that require bounded domains."""

    @pytest.mark.parametrize("name", ["GaussianMechanism", "CoinPress"])
    def test_an_unbounded_domain_is_refused(self, mean_estimation_task, name):
        algorithm = list_built_in_algorithms(mean_estimation_task)[name]()

        with pytest.raises(InvalidRegime, match="radius"):
            algorithm.run(
                mean_metadata(10, 5),
                mean_estimation_task.Input(X=np.zeros((10, 5))),
                budget=ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=42,
            )

    def test_gaussian_mechanism_calibrates_to_centered_radius_over_public_n(
        self, mean_estimation_task, accounting_spy: AccountingSpy
    ):
        """Noise is calibrated to the centered sensitivity ``radius / n``."""
        n = 5
        # Enclosing-ball radius sqrt(2**2 + 4**2); the box sits away from the
        # origin, so an uncentered radius would be sqrt(14**2 + 4**2) instead.
        X = np.array(
            [[10.0, -4.0], [10.0, 0.0], [14.0, 4.0], [12.5, 1.0], [11.0, -2.0]]
        )
        metadata = mean_metadata(n, box((10.0, 14.0), (-4.0, 4.0)))
        input_data = mean_estimation_task.Input(X=X)
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)

        algorithm = list_built_in_algorithms(mean_estimation_task)[
            "GaussianMechanism"
        ]()
        algorithm.run(metadata, input_data, budget=budget, seed=0)

        assert len(accounting_spy.calls) == 1
        call = accounting_spy.calls[0]
        assert call.entry_point == "calibrate_gaussian_scale"
        assert call.budget is budget
        assert call.sensitivity == pytest.approx(np.sqrt(20.0) / n)

    def test_gaussian_mechanism_adds_the_calibrated_noise(
        self, mean_estimation_task, fake_randomness: FakeRandomness
    ):
        """The release adds calibrated noise to the admitted empirical mean.

        Sensitivity ``radius / n = 3 / 4``; under ``rho = 0.5`` zCDP the
        Gaussian scale is ``sensitivity / sqrt(2 * rho) = 0.75``.
        """
        seed = 17
        X = np.array([[1.0, 2.0], [-2.0, 1.0], [0.5, -1.5], [2.0, -0.5]])
        metadata = mean_metadata(4, LpBallDomain.at_origin(p=2, radius=3.0, dim=2))
        input_data = mean_estimation_task.Input(X=X)
        budget = ZCDPBudget(rho=0.5)

        noise = np.array([0.25, -0.5])
        fake_randomness.script("normal", draws_in_order([noise]))
        algorithm = list_built_in_algorithms(mean_estimation_task)[
            "GaussianMechanism"
        ]()
        output = algorithm.run(metadata, input_data, budget=budget, seed=seed)

        assert fake_randomness.seeds == (seed,)
        (draw,) = fake_randomness.draws_of("normal")
        assert draw.parameters["loc"] == pytest.approx(0.0)
        assert draw.parameters["scale"] == pytest.approx(0.75)
        assert draw.parameters["size"] == (2,)
        np.testing.assert_allclose(output.mean_est, [0.375, 0.25] + noise)

    @pytest.mark.parametrize(
        "budget",
        [ApproxDPBudget(eps=1.0, delta=1e-6), ZCDPBudget(rho=0.5)],
    )
    def test_gaussian_mechanism_uses_offered_typed_budget(
        self, mean_estimation_task, accounting_spy: AccountingSpy, budget
    ):
        metadata, input_data = _bounded_zero_instance(mean_estimation_task)
        algorithm = list_built_in_algorithms(mean_estimation_task)[
            "GaussianMechanism"
        ]()
        algorithm.run(metadata, input_data, budget=budget, seed=0)
        assert len(accounting_spy.calls) == 1
        call = accounting_spy.calls[0]
        assert call.entry_point == "calibrate_gaussian_scale"
        assert call.budget is budget
        assert call.sensitivity == pytest.approx(0.1)

    def test_gaussian_mechanism_refuses_pure_budget(
        self, mean_estimation_task, accounting_spy: AccountingSpy
    ):
        metadata, input_data = _bounded_zero_instance(mean_estimation_task)
        budget = PureDPBudget(eps=1.0)
        algorithm = list_built_in_algorithms(mean_estimation_task)[
            "GaussianMechanism"
        ]()
        with pytest.raises(InvalidRegime, match="PureDPBudget"):
            algorithm.run(metadata, input_data, budget=budget, seed=0)
        assert len(accounting_spy.calls) == 1
        call = accounting_spy.calls[0]
        assert call.entry_point == "calibrate_gaussian_scale"
        assert call.budget is budget
        assert call.sensitivity == pytest.approx(0.1)

    def test_gaussian_mechanism_singleton_domain_needs_no_noise(
        self,
        mean_estimation_task,
        accounting_spy: AccountingSpy,
    ):
        metadata = mean_metadata(3, OneHotDomain(dim=1))
        input_data = mean_estimation_task.Input(X=np.ones((3, 1)))
        algorithm = list_built_in_algorithms(mean_estimation_task)[
            "GaussianMechanism"
        ]()

        output = algorithm.run(
            metadata,
            input_data,
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )

        np.testing.assert_allclose(output.mean_est, np.ones(1))
        assert len(accounting_spy.calls) == 1
        call = accounting_spy.calls[0]
        assert call.entry_point == "calibrate_gaussian_scale"
        assert call.budget == PureDPBudget(eps=1.0)
        assert call.sensitivity == 0.0

    @pytest.mark.parametrize(
        "num_iters, final_budget_fraction, expected_rhos, expected_betas",
        [
            (1, 0.75, [3.0], [0.01]),
            (3, 0.75, [0.375, 0.375, 2.25], [0.005, 0.005, 0.01]),
            (3, 0.5, [0.75, 0.75, 1.5], [0.005, 0.005, 0.01]),
        ],
    )
    def test_coinpress_applies_paper_practical_allocations(
        self,
        mean_estimation_task,
        accounting_spy: AccountingSpy,
        monkeypatch,
        num_iters,
        final_budget_fraction,
        expected_rhos,
        expected_betas,
    ):
        import dpstats.tasks.mean_estimation.algorithms.coinpress as coinpress
        from dpstats.tasks.mean_estimation.algorithms.coinpress import (
            CoinPress,
        )

        metadata, input_data = _bounded_zero_instance(mean_estimation_task)
        budget = ZCDPBudget(rho=3.0)
        captured_budgets: list[ZCDPBudget] = []
        captured_betas: list[float] = []

        def record_step(d, n, X, c, r, step_budget, beta, rng):
            captured_budgets.append(step_budget)
            captured_betas.append(beta)
            return c, r

        monkeypatch.setattr(coinpress, "_mvm", record_step)
        CoinPress(
            num_iters=num_iters,
            final_budget_fraction=final_budget_fraction,
            failure_probability=0.04,
        ).run(metadata, input_data, budget=budget, seed=0)
        assert accounting_spy.calls == (AccountingCall("spend_as_zcdp", budget),)
        assert [step_budget.rho for step_budget in captured_budgets] == pytest.approx(
            expected_rhos
        )
        assert captured_betas == pytest.approx(expected_betas)

    def test_coinpress_refuses_pure_budget(
        self, mean_estimation_task, accounting_spy: AccountingSpy
    ):
        metadata, input_data = _bounded_zero_instance(mean_estimation_task)
        budget = PureDPBudget(eps=1.0)
        algorithm = list_built_in_algorithms(mean_estimation_task)["CoinPress"]()
        with pytest.raises(InvalidRegime, match="PureDPBudget"):
            algorithm.run(metadata, input_data, budget=budget, seed=0)
        assert accounting_spy.calls == (AccountingCall("spend_as_zcdp", budget),)

    def test_coinpress_uses_add_remove_calibration(
        self,
        mean_estimation_task,
        fake_randomness: FakeRandomness,
        accounting_spy: AccountingSpy,
    ):
        """Per-step noise uses the centered public-n query's sensitivity."""
        task = mean_estimation_task
        d = 3
        n = 50
        eps, delta, failure_probability = 1.0, 1e-5, 0.01
        X = fake_randomness.real_default_rng(0).normal(size=(n, d))
        metadata = mean_metadata(n, LpBallDomain.at_origin(p=2, radius=6.0, dim=d))
        input_data = task.Input(X=X)

        fake_randomness.script("normal", zero_draw)

        CoinPress = list_built_in_algorithms(task)["CoinPress"]
        alg = CoinPress(
            num_iters=1,
            failure_probability=failure_probability,
        )
        alg.run(
            metadata, input_data, budget=ApproxDPBudget(eps=eps, delta=delta), seed=0
        )

        spend_call, calibration_call = accounting_spy.calls
        assert spend_call == AccountingCall(
            "spend_as_zcdp", ApproxDPBudget(eps=eps, delta=delta)
        )
        assert calibration_call.entry_point == "calibrate_gaussian_scale"
        assert isinstance(calibration_call.budget, ZCDPBudget)
        rho = calibration_call.budget.rho
        r = 6.0
        step_failure_probability = failure_probability / 4
        gamma_1 = np.sqrt(
            d
            + 2 * np.sqrt(d * np.log(n / step_failure_probability))
            + 2 * np.log(n / step_failure_probability)
        )
        expected_sigma = (r + gamma_1) / n / np.sqrt(2 * rho)
        assert fake_randomness.parameters_of("normal", "scale") == pytest.approx(
            [expected_sigma]
        )
        assert calibration_call.sensitivity == pytest.approx((r + gamma_1) / n)

    def test_coinpress_initializes_from_off_center_domain(
        self,
        mean_estimation_task,
        fake_randomness: FakeRandomness,
    ):
        """The public prior ball follows the domain instead of the origin."""
        X = np.array([[10.5, 21.0], [11.5, 23.0]])
        metadata = mean_metadata(2, box((10.0, 12.0), (20.0, 24.0)))
        input_data = mean_estimation_task.Input(X=X)
        fake_randomness.script("normal", zero_draw)

        algorithm = list_built_in_algorithms(mean_estimation_task)["CoinPress"](
            num_iters=1
        )
        output = algorithm.run(
            metadata,
            input_data,
            budget=ZCDPBudget(rho=1.0),
            seed=0,
        )

        np.testing.assert_allclose(output.mean_est, [11.0, 22.0])


class TestClippedSimulatedInstances:
    """The producers publish a bounded domain derived from parameters only."""

    def test_clipped_normal_domain_is_parameter_derived(self):
        """The radius is ``mean_radius`` plus the clipped noise's radius bound."""
        meta_a, _, _ = simulate_clipped_normal(
            seed=0,
            mean_radius=5.0,
            noise_radius_upper_bound=6.0,
            dimension=3,
            n_samples=50,
        )
        meta_b, _, _ = simulate_clipped_normal(
            seed=123,
            mean_radius=5.0,
            noise_radius_upper_bound=6.0,
            dimension=3,
            n_samples=50,
        )

        assert isinstance(meta_a.domain, LpBallDomain)
        assert meta_a.domain.dim == 3
        assert meta_a.domain.radius == 11.0
        np.testing.assert_array_equal(meta_a.domain.center, meta_b.domain.center)
        assert meta_a.domain.radius == meta_b.domain.radius

    @pytest.mark.parametrize(
        "produce", [simulate_clipped_normal, simulate_clipped_laplacian]
    )
    def test_default_instance_has_a_bounded_ball_domain(self, produce):
        metadata, _, _ = produce(seed=0)

        assert isinstance(metadata.domain, LpBallDomain)
        assert 0 < metadata.domain.l2_diameter() < np.inf


def test_metadata_dimension_comes_from_domain():
    assert mean_metadata(100, 5).dim == 5
