"""Tests for the linear-regression Task, its algorithms, producer and metrics."""

# pyright: reportPrivateUsage=false

import math
from typing import cast

import numpy as np
import pytest

from dpstats.tasks import list_built_in_algorithms, list_built_in_metrics
from dpstats.simulations.linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.budgets import accounting
from dpstats.budgets.accounting import spend_as_gdp as real_spend_as_gdp
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    InterceptDomain,
    LpBallDomain,
    ProductDomain,
    EuclideanSpace,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
    assert_budget_close,
)
from tests.unit.tasks.task_builders import Unreadable, box, lr_metadata

UNIT_BALL = LpBallDomain.at_origin(p=2, radius=1.0, dim=1)


def _small_typed_budget_instance(linear_regression_task):
    rng = np.random.default_rng(417)
    X = rng.uniform(-0.5, 0.5, size=(30, 1))
    y = 0.7 * X[:, 0] + rng.normal(0.0, 0.05, size=30)
    return (
        lr_metadata(30, design=UNIT_BALL, response=UNIT_BALL),
        linear_regression_task.Input(X=X, y=y),
    )


class TestTypedBudgetAccounting:
    """Pin each private linear-regression mechanism to its accounting route."""

    @pytest.mark.parametrize(
        "budget",
        [
            PureDPBudget(eps=1.0),
            ApproxDPBudget(eps=1.0, delta=1e-6),
            ZCDPBudget(rho=0.5),
        ],
    )
    def test_pure_mechanisms_spend_through_pure_seam(
        self,
        linear_regression_task,
        accounting_spy: AccountingSpy,
        budget,
    ):
        metadata, input_data = _small_typed_budget_instance(linear_regression_task)
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ](num_steps=4)
        algorithm.run(metadata, input_data, budget=budget, seed=0)
        assert accounting_spy.calls == (AccountingCall("spend_as_pure", budget),)

    @pytest.mark.parametrize(
        "budget",
        [
            ApproxDPBudget(eps=2.0, delta=1e-5),
            ZCDPBudget(rho=2.0),
            GDPBudget(mu=2.0),
        ],
    )
    def test_ssp_splits_typed_budget_between_releases(
        self,
        linear_regression_task,
        accounting_spy: AccountingSpy,
        budget,
    ):
        metadata, input_data = _small_typed_budget_instance(linear_regression_task)
        algorithm = list_built_in_algorithms(linear_regression_task)["SSP"]()
        algorithm.run(metadata, input_data, budget=budget, seed=0)

        # Exact GDP composition: convert once, then split mu into equal
        # halves.
        gdp_budget = real_spend_as_gdp(budget)
        release_budget = gdp_budget.split((1.0, 1.0))[0]
        assert [call.entry_point for call in accounting_spy.calls] == [
            "spend_as_gdp",
            "calibrate_gaussian_scale",
            "calibrate_gaussian_scale",
        ]
        assert_budget_close(accounting_spy.calls[0].budget, budget)
        for call in accounting_spy.calls[1:]:
            assert_budget_close(call.budget, release_budget)
            assert call.sensitivity == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "budget", [NO_PRIVACY, ApproxDPBudget(eps=math.inf, delta=1.0)]
    )
    def test_ssp_unconstrained_offers_match_the_delta_one_approx_dp_offer(
        self,
        linear_regression_task,
        budget,
    ):
        # ApproxDPBudget.split keeps delta at 1 in every part of a delta=1
        # whole, so every unconstrained offer, of any family, releases
        # identically to an explicit ApproxDPBudget(1, 1) offer.
        metadata, input_data = _small_typed_budget_instance(linear_regression_task)
        algorithm = list_built_in_algorithms(linear_regression_task)["SSP"]()
        reference = algorithm.run(
            metadata,
            input_data,
            budget=ApproxDPBudget(eps=1.0, delta=1.0),
            seed=0,
        )

        output = algorithm.run(metadata, input_data, budget=budget, seed=0)

        np.testing.assert_allclose(output.beta, reference.beta)

    def test_ssp_refuses_pure_before_releasing(
        self, linear_regression_task, accounting_spy: AccountingSpy
    ):
        metadata, _ = _small_typed_budget_instance(linear_regression_task)
        unreadable = cast(LinearRegressionInput, Unreadable("Input"))
        budget = PureDPBudget(eps=1.0)
        algorithm = list_built_in_algorithms(linear_regression_task)["SSP"]()
        with pytest.raises(InvalidRegime, match="PureDPBudget"):
            algorithm.run(metadata, unreadable, budget=budget, seed=0)
        assert [call.entry_point for call in accounting_spy.calls] == ["spend_as_gdp"]

    def test_ssp_approx_dp_noise_scale_is_no_larger_than_basic_composition_split(
        self, linear_regression_task
    ):
        # Old behavior (basic-composition split of eps/delta into equal
        # halves, then per-half analytic Gaussian calibration) versus new
        # behavior (convert once to GDP, then split mu exactly into equal
        # halves).
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        sensitivity = 1.0

        old_half = budget.split((1.0, 1.0))[0]
        old_scale = accounting.calibrate_gaussian_scale(old_half, sensitivity)

        new_half = real_spend_as_gdp(budget).split((1.0, 1.0))[0]
        new_scale = accounting.calibrate_gaussian_scale(new_half, sensitivity)

        assert new_scale <= old_scale

    # Pinned against the pre-conversion SSP, by running:
    #   SSP().run(*_small_typed_budget_instance(...),
    #                     budget=ZCDPBudget(rho=1.5), seed=seed).beta
    # for each seed. zCDP composes identically under both the old rho/2
    # split and the new sqrt(2*rho)/sqrt(2) split, so these stay
    # bit-identical.
    _VANILLA_SSP_ZCDP_PINS = {
        0: [0.6898334808558984],
        1: [0.9412174866374547],
        2: [0.5466827876047493],
    }

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_ssp_zcdp_release_matches_origin_dev_pin(
        self, linear_regression_task, seed
    ):
        metadata, input_data = _small_typed_budget_instance(linear_regression_task)

        output = list_built_in_algorithms(linear_regression_task)["SSP"]().run(
            metadata, input_data, budget=ZCDPBudget(rho=1.5), seed=seed
        )

        np.testing.assert_allclose(
            output.beta, self._VANILLA_SSP_ZCDP_PINS[seed], rtol=1e-12
        )

    def test_ssp_zcdp_split_calibrates_each_release_to_mu_over_sqrt_k(self):
        """Each of the 2 releases gets sqrt(2*rho)/sqrt(2), not rho/2."""
        rho = 1.5
        sensitivity = 1.0
        expected_scale = sensitivity / (math.sqrt(2.0 * rho) / math.sqrt(2.0))

        release_budget = real_spend_as_gdp(ZCDPBudget(rho=rho)).split((1.0, 1.0))[0]
        scale = accounting.calibrate_gaussian_scale(release_budget, sensitivity)

        assert scale == pytest.approx(expected_scale)


class TestOLSAlgorithm:
    """Tests for OLS algorithm."""

    def test_recovers_hand_computed_coefficients(self, linear_regression_task):
        input_data = linear_regression_task.Input(
            X=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            y=np.array([2.0, -1.0, 1.0]),
        )
        metadata = lr_metadata(3, design=2)
        OLS = list_built_in_algorithms(linear_regression_task)["OLS"]
        output = OLS().run(
            metadata,
            input_data,
            budget=NO_PRIVACY,
            seed=0,
        )

        np.testing.assert_allclose(output.beta, [2.0, -1.0], atol=1e-12)

    def test_ols_ignores_the_seed(self, linear_regression_task, linear_regression_data):
        X, y, _ = linear_regression_data
        metadata = lr_metadata(X.shape[0], design=X.shape[1])
        input_data = linear_regression_task.Input(X=X, y=y)
        alg = list_built_in_algorithms(linear_regression_task)["OLS"]()

        output1 = alg.run(metadata, input_data, budget=NO_PRIVACY, seed=42)
        output2 = alg.run(metadata, input_data, budget=NO_PRIVACY, seed=43)

        np.testing.assert_array_equal(output1.beta, output2.beta)

    @pytest.mark.parametrize(
        "budget",
        [
            NO_PRIVACY,
            PureDPBudget(eps=math.inf),
            ApproxDPBudget(eps=math.inf, delta=1e-5),
            ZCDPBudget(rho=math.inf),
        ],
    )
    def test_ols_accepts_any_unconstrained_budget(
        self, linear_regression_task, linear_regression_data, budget
    ):
        X, y, _ = linear_regression_data
        metadata = lr_metadata(X.shape[0], design=X.shape[1])
        output = list_built_in_algorithms(linear_regression_task)["OLS"]().run(
            metadata,
            linear_regression_task.Input(X=X, y=y),
            budget=budget,
            seed=0,
        )
        assert output.beta.shape == (X.shape[1],)

    def test_ols_rejects_a_finite_privacy_budget(
        self, linear_regression_task, linear_regression_data
    ):
        X, y, _ = linear_regression_data
        metadata = lr_metadata(X.shape[0], design=X.shape[1])
        with pytest.raises(InvalidRegime, match="unconstrained"):
            list_built_in_algorithms(linear_regression_task)["OLS"]().run(
                metadata,
                linear_regression_task.Input(X=X, y=y),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )


class TestGridDimensionGuard:
    """Pin the public dimension limit of the grid sampler."""

    def test_dimension_four_fails_before_private_input_read(
        self, linear_regression_task
    ):
        metadata = lr_metadata(1, design=box(*[(-1, 1)] * 4), response=box((-1, 1)))
        unreadable = cast(LinearRegressionInput, Unreadable("Input"))
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ](num_steps=2)

        with pytest.raises(InvalidRegime, match="dimension"):
            algorithm.run(metadata, unreadable, PureDPBudget(eps=1.0), seed=0)

    def test_dimension_three_releases(self, linear_regression_task):
        input_data = linear_regression_task.Input(
            X=np.array([[0.2, -0.4, 0.6], [-0.5, 0.3, 0.1], [0.7, 0.8, -0.2]]),
            y=np.array([0.1, -0.2, 0.4]),
        )
        metadata = lr_metadata(3, design=box(*[(-1, 1)] * 3), response=box((-1, 1)))
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ](num_steps=2)

        output = algorithm.run(metadata, input_data, PureDPBudget(eps=1.0), seed=7)

        assert output.beta.shape == (3,)
        assert np.all(np.isfinite(output.beta))


class TestGridExponentialMechanismUnconstrainedRelease:
    """The grid mechanism collapses to its non-private core at NO_PRIVACY."""

    def test_unconstrained_release_matches_the_large_budget_limit(
        self, linear_regression_task
    ):
        """The unconstrained release is the large-budget limit."""
        rng = np.random.default_rng(7)
        X = rng.normal(scale=0.5, size=(40, 2))
        y = X @ np.array([0.4, -0.7]) + rng.normal(scale=0.05, size=40)
        metadata = lr_metadata(40, design=box((-2, 2), (-2, 2)), response=box((-2, 2)))
        input_data = linear_regression_task.Input(X=X, y=y)
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ]()

        free = algorithm.run(metadata, input_data, budget=NO_PRIVACY, seed=0)
        overwhelming = algorithm.run(
            metadata, input_data, budget=PureDPBudget(eps=1e9), seed=0
        )
        np.testing.assert_allclose(free.beta, overwhelming.beta)

    def test_tied_unconstrained_release_matches_the_large_budget_limit(
        self, linear_regression_task
    ):
        """The tied-minimizer release is the large-budget limit."""
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ](num_steps=5)
        metadata = lr_metadata(4, design=box((-1, 1), (-1, 1)), response=box((-1, 1)))
        input_data = linear_regression_task.Input(X=np.zeros((4, 2)), y=np.zeros(4))

        free = algorithm.run(metadata, input_data, budget=NO_PRIVACY, seed=0).beta
        overwhelming = algorithm.run(
            metadata, input_data, budget=PureDPBudget(eps=1e9), seed=0
        ).beta

        np.testing.assert_array_equal(free, overwhelming)


class TestGridExponentialMechanismScaleEquivariance:
    """Rescaling the data and its declared domains rescales the release."""

    @staticmethod
    def _instance(seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = np.column_stack((rng.uniform(-2.0, 2.0, 40), rng.uniform(-1.0, 1.0, 40)))
        y = X @ np.array([0.6, -0.4]) + rng.normal(0.0, 0.05, 40)
        return X, y

    def test_same_seed_response_rescaling_is_exact(self, linear_regression_task):
        task = linear_regression_task
        X, y = self._instance(19)
        design = box((-2, 2), (-1, 1))
        algorithm = list_built_in_algorithms(task)["GridExponentialMechanism"](
            num_steps=7
        )
        baseline = algorithm.run(
            lr_metadata(40, design=design, response=box((-3, 3))),
            task.Input(X=X, y=y),
            budget=PureDPBudget(eps=1.0),
            seed=7,
        ).beta

        scaled = algorithm.run(
            lr_metadata(40, design=design, response=box((-30, 30))),
            task.Input(X=X, y=10.0 * y),
            budget=PureDPBudget(eps=1.0),
            seed=7,
        ).beta

        np.testing.assert_allclose(scaled, 10.0 * baseline, rtol=1e-12, atol=1e-12)

    def test_same_seed_per_column_design_rescaling_is_exact(
        self, linear_regression_task
    ):
        task = linear_regression_task
        X, y = self._instance(23)
        response = box((-3, 3))
        algorithm = list_built_in_algorithms(task)["GridExponentialMechanism"](
            num_steps=7
        )
        baseline = algorithm.run(
            lr_metadata(40, design=box((-2, 2), (-1, 1)), response=response),
            task.Input(X=X, y=y),
            budget=PureDPBudget(eps=1.0),
            seed=11,
        ).beta

        column_scales = np.array([2.0, 100.0])
        scaled = algorithm.run(
            lr_metadata(40, design=box((-4, 4), (-100, 100)), response=response),
            task.Input(X=X * column_scales, y=y),
            budget=PureDPBudget(eps=1.0),
            seed=11,
        ).beta

        np.testing.assert_allclose(
            scaled, baseline / column_scales, rtol=1e-12, atol=1e-12
        )


class TestGridExponentialMechanismCalibration:
    """The public domains fix the residual cap, score scale and grid.

    With half-widths ``h_y`` for the response and ``h_j`` for covariate ``j``,
    the residual cap is ``residual_cap_mult * h_y``, the score scale is
    ``eps / cap**2`` and slope ``j``'s grid spans ``coef_box_mult * h_y / h_j``
    about zero. An intercept's grid spans ``max|y| + sum_j |c_j| h_y / h_j``
    for covariate centers ``c_j``.
    """

    def test_single_stage_uses_resolved_cap_and_coordinate_scales(
        self, linear_regression_task, monkeypatch
    ):
        import dpstats.tasks.linear_regression.algorithms.grid_exponential_mechanism as em_module

        task = linear_regression_task
        # h_y = 4 and h = (2, 4): cap 0.25 * 4 = 1, score scale 2 / 1**2 = 2,
        # slope half-widths 1.5 * (4/2, 4/4) = (3, 1.5).
        metadata = lr_metadata(1, design=box((-2, 2), (-4, 4)), response=box((-3, 5)))
        input_data = task.Input(X=np.zeros((1, 2)), y=np.zeros(1))
        observed = {}

        def record_sampling(X, y, grids_per_dim, residual_cap, score_scale, seed):
            observed.update(
                residual_cap=residual_cap,
                score_scale=score_scale,
                grids_per_dim=grids_per_dim,
            )
            return np.zeros(2)

        monkeypatch.setattr(em_module, "_sample_grid_point", record_sampling)
        output = list_built_in_algorithms(task)["GridExponentialMechanism"](
            residual_cap_mult=0.25,
            coef_box_mult=1.5,
            num_steps=3,
        ).run(metadata, input_data, budget=PureDPBudget(eps=2.0), seed=5)

        assert observed["residual_cap"] == pytest.approx(1.0)
        assert observed["score_scale"] == pytest.approx(2.0)
        first_grid, second_grid = observed["grids_per_dim"]
        np.testing.assert_allclose(first_grid, np.linspace(-3.0, 3.0, 3))
        np.testing.assert_allclose(second_grid, np.linspace(-1.5, 1.5, 3))
        np.testing.assert_array_equal(output.beta, np.zeros(2))

    def test_intercept_grid_contains_a_large_response_offset(
        self, linear_regression_task
    ):
        """A two-point grid releases one of its endpoints on every axis.

        ``h_y = 53.55``, so the slope half-widths are ``53.55 / 2.2`` and
        ``53.55 / 170.96``; the centered covariates leave the intercept
        half-width at ``max|y| = 1048``. Each contains the truth.
        """
        metadata = lr_metadata(
            1,
            design=ProductDomain(
                parts=(InterceptDomain(), box((-2.2, 2.2), (-170.96, 170.96)))
            ),
            response=box((940.9, 1048.0)),
        )
        input_data = linear_regression_task.Input(
            X=np.array([[1.0, 0.0, 0.0]]), y=np.array([1000.0])
        )

        beta = (
            list_built_in_algorithms(linear_regression_task)[
                "GridExponentialMechanism"
            ](num_steps=2)
            .run(metadata, input_data, budget=PureDPBudget(eps=1.0), seed=0)
            .beta
        )

        np.testing.assert_allclose(
            np.abs(beta), [1048.0, 53.55 / 2.2, 53.55 / 170.96], rtol=1e-9
        )
        assert np.all(np.abs(beta) >= np.abs([1000.0, 2.0, 0.3]))

    def test_intercept_grid_contains_a_public_covariate_offset(
        self, linear_regression_task
    ):
        """The covariate center ``100.5`` widens the intercept half-width.

        ``h_y = 0.5`` and ``h = 0.5`` give slope half-width 1 and intercept
        half-width ``max|y| + 100.5 * 1 = 101.5``, which contains ``-100``.
        """
        metadata = lr_metadata(
            1,
            design=ProductDomain(parts=(InterceptDomain(), box((100.0, 101.0)))),
            response=box((0.0, 1.0)),
        )
        input_data = linear_regression_task.Input(
            X=np.array([[1.0, 100.0]]), y=np.zeros(1)
        )

        beta = (
            list_built_in_algorithms(linear_regression_task)[
                "GridExponentialMechanism"
            ](num_steps=2)
            .run(metadata, input_data, budget=PureDPBudget(eps=1.0), seed=0)
            .beta
        )

        np.testing.assert_allclose(np.abs(beta), [101.5, 1.0], rtol=1e-9)

    def test_default_release_concentrates_near_the_least_squares_fit(
        self, linear_regression_task
    ):
        """At eps = 1 on n = 200 unit-scale rows the score law is not flat.

        A flat law would spread releases over the whole coefficient box, whose
        half-widths are of the order of the coefficients themselves.
        """
        rng = np.random.default_rng(31)
        X = rng.normal(size=(200, 2))
        y = X @ np.array([1.0, -0.5]) + rng.normal(0.0, 0.1, 200)
        beta_reference = np.linalg.lstsq(X, y, rcond=None)[0]
        metadata = lr_metadata(
            200,
            design=BoxDomain(bounds=np.column_stack((X.min(axis=0), X.max(axis=0)))),
            response=box((y.min(), y.max())),
        )
        input_data = linear_regression_task.Input(X=X, y=y)
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ]()

        errors = [
            np.linalg.norm(
                algorithm.run(metadata, input_data, PureDPBudget(eps=1.0), seed).beta
                - beta_reference
            )
            for seed in range(25)
        ]

        assert np.median(errors) < 0.25 * np.linalg.norm(beta_reference)


class TestGridExponentialMechanismPublicDomainFailures:
    @pytest.mark.parametrize(
        "metadata,match",
        [
            (lr_metadata(1, design=1, response=box((-1, 1))), "finite"),
            (lr_metadata(1, design=box((-1, 1)), response=1), "finite"),
            (
                lr_metadata(1, design=box((-1, 1)), response=box((2, 2))),
                "response-domain width",
            ),
            # One zero-width covariate beside a live one: the design bound stays
            # positive, so the metadata admits it and the mechanism refuses it.
            (
                lr_metadata(1, design=box((-1, 1), (0, 0)), response=box((-1, 1))),
                "covariates",
            ),
        ],
    )
    def test_public_domain_error_precedes_private_input_read(
        self, linear_regression_task, metadata, match
    ):
        algorithm = list_built_in_algorithms(linear_regression_task)[
            "GridExponentialMechanism"
        ]()
        unreadable = cast(LinearRegressionInput, Unreadable("Input"))
        with pytest.raises(InvalidRegime, match=match):
            algorithm.run(metadata, unreadable, PureDPBudget(eps=1.0), seed=0)


class TestGaussianSimulatedInstances:
    """Tests for Gaussian simulated instances."""

    def test_ols_metric_argument_uses_canonical_fit(
        self, monkeypatch, linear_regression_task
    ):
        import dpstats.simulations.linear_regression as producer_module
        from dpstats.tasks.linear_regression.algorithms.ols import OLS

        expected_beta = np.array([7.0, -3.0])
        received = []

        def fake_run(self, metadata, input, budget, seed):
            received.append((metadata, input, budget, seed))
            return linear_regression_task.Output(beta=expected_beta)

        monkeypatch.setattr(OLS, "run", fake_run)
        metadata, input_data, metric_arguments = (
            producer_module.simulate_clipped_gaussian(seed=0, dim=2, n_samples=20)
        )

        assert received == [(metadata, input_data, NO_PRIVACY, 0)]
        l2_error_to_ols = list_built_in_metrics(linear_regression_task)[
            "l2_error_to_ols"
        ]

        assert l2_error_to_ols.calculate(
            linear_regression_task.Output(beta=expected_beta),
            ols_coef=metric_arguments["ols_coef"],
        ) == pytest.approx(0.0)

    def test_residual_baselines_stand_for_the_fit_and_the_response(
        self, linear_regression_task
    ):
        """The stored baselines are the OLS fit's own cost and var(y)."""
        metadata, input_data, references = simulate_clipped_gaussian(
            seed=5, dim=3, n_samples=60
        )
        residual_rmse = list_built_in_metrics(linear_regression_task)["residual_rmse"]
        ols_fit = linear_regression_task.Output(beta=np.asarray(references["ols_coef"]))

        assert references["ols_residual_rmse"] == pytest.approx(
            residual_rmse.calculate(ols_fit, input=input_data)
        )
        assert references["response_std"] == pytest.approx(float(np.std(input_data.y)))

    def test_one_dimensional_conditioning_is_refused(self):
        with pytest.raises(ValueError, match="requires dim >= 2"):
            simulate_clipped_gaussian(seed=0, dim=1, n_samples=20, condition_num=10.0)

    def test_data_and_metadata_share_exact_radii(self):
        metadata, input_data, _ = simulate_clipped_gaussian(
            seed=0,
            dim=3,
            n_samples=200,
            condition_num=10.0,
            covariate_radius=0.75,
            response_radius=0.5,
        )
        assert isinstance(metadata.design_domain, LpBallDomain)
        assert isinstance(metadata.response_domain, LpBallDomain)
        assert metadata.design_domain.radius == 0.75
        assert metadata.response_domain.radius == 0.5
        assert np.all(np.linalg.norm(input_data.X, axis=1) <= 0.75 + 1e-12)
        assert np.all(np.abs(input_data.y) <= 0.5 + 1e-12)


class TestMetadataWithDomains:
    """Tests for metadata domain fields."""

    def test_dimension_and_intercept_are_derived_from_the_design(self):
        metadata = lr_metadata(100, design=5)

        assert metadata.dim == 5
        assert not metadata.has_intercept

    def test_response_domain_must_be_scalar(self):
        with pytest.raises(ValueError, match="response_domain"):
            lr_metadata(100, design=5, response=2)

    def test_design_domain_must_have_a_positive_norm_bound(self):
        """A zero bound makes every design row zero, which estimates nothing."""
        with pytest.raises(ValueError, match="positive norm bound"):
            lr_metadata(100, design=box((0.0, 0.0)))

    def test_intercept_is_derived_from_leading_semantic_domain(self):
        design = ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=4)))
        metadata = lr_metadata(100, design=design)

        assert metadata.has_intercept
        assert metadata.design_domain is design

    def test_singleton_box_does_not_imply_intercept(self):
        design = ProductDomain(parts=(box((1.0, 1.0)), EuclideanSpace(dim=4)))

        assert not lr_metadata(100, design=design).has_intercept

    def test_intercept_domain_geometry(self):
        domain = InterceptDomain()

        np.testing.assert_array_equal(domain.center, [1.0])
        assert domain.l1_diameter() == pytest.approx(0.0)
        assert domain.l2_diameter() == pytest.approx(0.0)
        assert domain.l2_norm_bound == pytest.approx(1.0)
        np.testing.assert_array_equal(
            domain.bounding_box().bounds,
            np.array([[1.0, 1.0]]),
        )
        np.testing.assert_array_equal(domain.project(np.array([4.0])), [1.0])

    def test_intercept_round_trips_as_a_core_domain(self):
        """A domain annotation restores its concrete public configuration."""
        record = lr_metadata(
            1, design=ProductDomain(parts=(InterceptDomain(),))
        ).dump_config()

        design = LinearRegressionMetadata.load_record(record).design_domain

        assert design == ProductDomain(parts=(InterceptDomain(),))
        assert design.dump_config().model_dump() == record.config["design_domain"]


class TestStrictDomainBounds:
    """Bounds-assuming algorithms use domain bounds exactly or fail closed."""

    def test_ssp_calibrates_each_release_to_exact_domain_bounds(
        self, linear_regression_task, accounting_spy: AccountingSpy
    ):
        """On ``[-2, 3]^3`` rows have norm at most ``sqrt(27)``; responses 0.5.

        The second-moment release has sensitivity ``27`` and the cross-moment
        release ``0.5 * sqrt(27)``.
        """
        rng = np.random.default_rng(0)
        X = rng.uniform(-2.0, 3.0, size=(40, 3))
        y = rng.uniform(-0.5, 0.5, size=40)
        metadata = lr_metadata(
            40, design=box(*[(-2.0, 3.0)] * 3), response=box((-0.5, 0.5))
        )

        list_built_in_algorithms(linear_regression_task)["SSP"]().run(
            metadata,
            linear_regression_task.Input(X=X, y=y),
            budget=ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )

        assert [call.entry_point for call in accounting_spy.calls] == [
            "spend_as_gdp",
            "calibrate_gaussian_scale",
            "calibrate_gaussian_scale",
        ]
        cov_sens, vec_sens = (call.sensitivity for call in accounting_spy.calls[1:])
        assert cov_sens == pytest.approx(27.0)
        assert vec_sens == pytest.approx(0.5 * math.sqrt(27.0))

    @pytest.mark.parametrize(
        "design_domain,response_domain",
        [(1, UNIT_BALL), (UNIT_BALL, 1)],
    )
    def test_ssp_unbounded_domain_is_rejected(
        self, linear_regression_task, design_domain, response_domain
    ):
        rng = np.random.default_rng(1)
        input_data = linear_regression_task.Input(
            X=rng.normal(size=(40, 1)), y=rng.normal(size=40)
        )

        with pytest.raises(InvalidRegime, match="finite"):
            list_built_in_algorithms(linear_regression_task)["SSP"]().run(
                lr_metadata(40, design=design_domain, response=response_domain),
                input_data,
                budget=ApproxDPBudget(eps=1.0, delta=1e-6),
                seed=0,
            )


class TestPostRunMetrics:
    """Post-run metrics against hand-computed values."""

    def test_residual_rmse_matches_hand_computation(self, linear_regression_task):
        """Residuals ``X @ beta - y`` are ``(1, -2, -2)``: root mean square ``sqrt(3)``."""
        residual_rmse = list_built_in_metrics(linear_regression_task)["residual_rmse"]
        inp = LinearRegressionInput(
            X=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            y=np.array([1.0, 1.0, 3.0]),
        )
        out = linear_regression_task.Output(beta=np.array([2.0, -1.0]))

        assert residual_rmse.calculate(out, input=inp) == pytest.approx(np.sqrt(3.0))

    def test_prediction_rmse_to_ols_completes_the_residual_rmse_identity(
        self, linear_regression_task
    ):
        """On a hand-built design, excess risk plus OLS's own cost is exact.

        ``X`` has orthogonal columns (intercept, and a centered covariate), so
        the OLS fit is the sample mean and the covariate's OLS slope, both
        computed by hand: ``beta_ols = [2.0, 0.7]``.
        """
        metrics = list_built_in_metrics(linear_regression_task)
        residual_rmse = metrics["residual_rmse"]
        prediction_rmse_to_ols = metrics["prediction_rmse_to_ols"]

        X = np.array([[1.0, -3.0], [1.0, -1.0], [1.0, 1.0], [1.0, 3.0]])
        y = np.array([0.0, 1.0, 3.0, 4.0])
        inp = LinearRegressionInput(X=X, y=y)
        ols_coef = np.array([2.0, 0.7])
        ols_output = linear_regression_task.Output(beta=ols_coef)
        beta = np.array([1.0, 1.0])
        output = linear_regression_task.Output(beta=beta)

        assert prediction_rmse_to_ols.calculate(
            output, input=inp, ols_coef=ols_coef
        ) ** 2 + residual_rmse.calculate(ols_output, input=inp) ** 2 == pytest.approx(
            residual_rmse.calculate(output, input=inp) ** 2, abs=1e-12
        )
        assert prediction_rmse_to_ols.calculate(
            ols_output, input=inp, ols_coef=ols_coef
        ) == pytest.approx(0.0, abs=1e-12)
