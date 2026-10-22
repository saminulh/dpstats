"""Tests for the TukeyEM private linear-regression algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest

import dpstats.tasks.linear_regression.algorithms.tukey_em as tukey
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    NO_PRIVACY,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.exceptions import NoRelease
from dpstats.parameters import ValuesSearch
from dpstats.task import Array1D
from dpstats.tasks.linear_regression.algorithms.tukey_em import (
    TukeyEM,
    _distance_lower_bound,  # pyright: ignore[reportPrivateUsage]
    _log_depth_volumes,  # pyright: ignore[reportPrivateUsage]
    _ptr_check,  # pyright: ignore[reportPrivateUsage]
    _sample_depth,  # pyright: ignore[reportPrivateUsage]
    _sample_point_with_depth,  # pyright: ignore[reportPrivateUsage]
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order
from tests.unit.tasks.task_builders import lr_metadata


def _ptr_draws(fake_randomness: FakeRandomness, value: float) -> None:
    """Serve the PTR test's Laplace draw so it passes (large) or fails (negative)."""
    fake_randomness.script("laplace", lambda parameters: value)


class TestTukeyEMPaperQuantities:
    """Pin the load-bearing quantities of paper Algorithms 1--4 and Lemma 3.6."""

    def test_depth_volumes_are_products_of_order_statistic_gaps(self) -> None:
        """Algorithm 2 line 10: ``V_i`` multiplies ``S[j, m-(i-1)] - S[j, i]`` over ``j``."""
        S = np.array([[0.0, 1.0, 3.0, 7.0], [10.0, 20.0, 40.0, 80.0]])
        # Depth 1 spans the whole range, 7 by 70; depth 2 the middle gaps, 2 by 20.
        np.testing.assert_allclose(
            np.exp(_log_depth_volumes(S)), [7.0 * 70.0, 2.0 * 20.0]
        )

    def test_distance_lower_bound_follows_lemma_3_6(self) -> None:
        """The bound is one more than the largest ``k`` whose volume condition holds.

        With ``t = 6``, ``eps = 1.1`` and ``delta = 0.5`` the target is
        ``delta / (8 e^eps) = 0.0208``. For ``k = 2`` the ratio
        ``V_3 / V_(9+g) e^(-0.55 g)`` is at best ``1.01 e^(-3.85) = 0.0215`` at
        the last admissible ``g = 7``, above the target; for ``k = 1`` the
        ratio ``V_4 / V_(8+g) e^(-0.55 g)`` reaches ``1.012 e^(-4.4) = 0.0124``
        at ``g = 8``, below it. So ``k = 1`` and the bound is 2. On the sparse
        volumes no ``k`` holds, so the bound is 0.
        """
        dense = np.array(
            [
                128,
                64,
                1,
                0.999,
                0.998,
                0.997,
                0.996,
                0.995,
                0.994,
                0.993,
                0.992,
                0.991,
                0.990,
                0.989,
                0.988,
                0.987,
            ]
        )
        sparse = np.array([128, 64, 32, 16, 8, 4, 2, 1])
        budget = ApproxDPBudget(eps=1.1, delta=0.5)
        assert _distance_lower_bound(np.log(dense), 6, budget) == 2
        assert _distance_lower_bound(np.log(sparse), 4, budget) == 0

    def test_distance_lower_bound_is_one_sensitive_to_model_replacement(
        self,
    ) -> None:
        rng = np.random.default_rng(19)
        models = np.sort(rng.normal(size=64))[None, :]
        neighboring_models = models.copy()
        replaced_index = int(rng.integers(64))
        neighboring_models[0, replaced_index] = rng.normal(scale=5.0)
        neighboring_models.sort(axis=1)

        budget = ApproxDPBudget(eps=1.0, delta=0.1)
        distance = _distance_lower_bound(_log_depth_volumes(models), 16, budget)
        neighboring_distance = _distance_lower_bound(
            _log_depth_volumes(neighboring_models), 16, budget
        )

        assert abs(distance - neighboring_distance) <= 1

    def test_distance_search_reaches_the_deepest_volume(self) -> None:
        """Lemma 3.6 admits every ``g`` with ``t + k + g + 1 <= |V|``, the last included.

        With all volumes 1, ``eps = 10`` and ``delta = 0.5`` the condition
        ``-5 g <= log(0.5 / 8) - 10`` needs ``g >= 3``, which ``k = 0``
        reaches only at the eighth and last volume (``4 + 0 + 3 + 1 = 8``);
        ``k = 1`` would need a ninth. So ``k = 0`` and the bound is 1.
        """
        log_volumes = np.zeros(8)
        assert (
            _distance_lower_bound(log_volumes, 4, ApproxDPBudget(eps=10.0, delta=0.5))
            == 1
        )

    def test_ptr_uses_reciprocal_laplace_scale(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """Algorithm 1 line 3 adds ``Lap(1 / eps)`` to the distance bound."""
        epsilon = 0.4
        # A draw far above any threshold makes the test pass on the scale
        # the mechanism asked for rather than on the noise it happened to get.
        fake_randomness.script("laplace", draws_in_order([1e6]))
        passed = _ptr_check(
            np.log(np.array([16.0, 8.0, 4.0, 2.0])),
            2,
            ApproxDPBudget(eps=epsilon, delta=0.1),
            0,
        )
        assert passed
        assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx(
            [1.0 / epsilon]
        )

    def test_ptr_threshold_is_log_inverse_two_delta_over_eps(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """Algorithm 1 line 3 passes when ``k + Lap(1/eps) >= log(1/(2 delta)) / eps``.

        The sparse volumes give ``k = 0`` (the Lemma 3.6 test above), and at
        ``eps = 1`` and ``delta = 0.1`` the threshold is ``log 5 = 1.609``.
        """
        log_volumes = np.log(np.array([128.0, 64, 32, 16, 8, 4, 2, 1]))
        budget = ApproxDPBudget(eps=1.0, delta=0.1)
        fake_randomness.script("laplace", draws_in_order([1.7, 1.5]))
        assert _ptr_check(log_volumes, 4, budget, 0)
        assert not _ptr_check(log_volumes, 4, budget, 0)

    @pytest.mark.parametrize(
        ("gumbel", "expected_depth"),
        [((0.0, 0.0, 0.0), 1), ((0.0, 0.35, 0.0), 2), ((0.0, 0.0, 1.1), 3)],
    )
    def test_depth_weights_follow_displayed_algorithm_3(
        self, fake_randomness: FakeRandomness, gumbel, expected_depth
    ) -> None:
        """Depth ``i`` has log-weight ``log W_i + (eps / 2) i`` with ``W_i = V_i - V_(i+1)``.

        Volumes 10, 4, 1 at depths 1, 2, 3 leave exact volumes 6, 3, 1, so at
        ``eps = 0.8`` the log-weights are ``log 6 + 0.4 = 2.19``,
        ``log 3 + 0.8 = 1.90`` and ``1.2``; the scripted Gumbel perturbations
        decide the argmax. A coefficient of ``eps`` would pick depth 2 at zero
        perturbation.
        """
        fake_randomness.script("gumbel", draws_in_order([np.array(gumbel)]))
        log_volumes = np.log(np.array([10.0, 4.0, 1.0]))

        depth = _sample_depth(log_volumes, 1, PureDPBudget(eps=0.8), 17)

        assert depth == expected_depth

    def test_depth_sampling_starts_at_the_restriction(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """Algorithm 3 line 2 considers only depths ``t, t + 1, ...``."""
        fake_randomness.script("gumbel", draws_in_order([np.zeros(2)]))
        log_volumes = np.log(np.array([10.0, 4.0, 3.0, 1.0]))

        # Depths 3 and 4 have exact volumes 2 and 1; depth 3 wins at eps = 0.
        depth = _sample_depth(log_volumes, 3, PureDPBudget(eps=1e-9), 17)

        assert depth == 3

    def test_exact_point_sampler_has_requested_depth(self) -> None:
        """Lemma 3.9: every coordinate has depth at least ``i`` and one exactly ``i``."""
        projections = np.array(
            [[-0.5, -0.4, -0.33, -0.2, 0.03, 0.1], [-0.1, 0.4, 0.5, 1, 1.5, 1.7]]
        )
        for depth in (1, 2, 3):
            for seed in range(20):
                sample = _sample_point_with_depth(projections, depth, seed)
                coordinate_depths = []
                for projection, value in zip(projections, sample):
                    insertion = int(np.searchsorted(projection, value))
                    coordinate_depths.append(
                        min(insertion, len(projection) - insertion)
                    )
                assert min(coordinate_depths) == depth

    def test_exact_point_sampler_weights_components_by_volume(self) -> None:
        """Algorithm 4 line 7 picks dimension ``j`` with probability ``vol(C_j,i) / V_>=1,i``.

        At depth 1 with ``S`` below, dimension 1 has depth-1 end intervals of
        total length 2 beside a deeper middle of length 1, and dimension 2
        end intervals of length 2 beside a middle of length 8. By Lemma 7.4,
        ``vol(C_1,1) = W_1,1 V_2,1 = 2 * 10 = 20`` and
        ``vol(C_2,1) = V_1,2 W_2,1 = 1 * 2 = 2``, so dimension 1 is chosen
        with probability ``20 / 22``. A chosen dimension's coordinate lands in
        an end interval, so the share of samples whose first coordinate is in
        the ends estimates that probability.
        """
        S = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 9.0, 10.0]])
        draws = 4000
        first_in_ends = sum(
            not 1.0 <= _sample_point_with_depth(S, 1, seed)[0] <= 2.0
            for seed in range(draws)
        )
        assert first_in_ends / draws == pytest.approx(20 / 22, abs=0.03)


class TestTukeyEMModelCountGrid:
    def test_search_values_are_the_papers_step_250_grid(self) -> None:
        """The candidates are Section 4.3's grid ``m = 250, 500, ..., 2000``."""
        search = TukeyEM.parameter_info()["m"].search
        assert isinstance(search, ValuesSearch)
        assert search.values == (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
        # The declared default is a point of that grid.
        assert TukeyEM().m in search.values


class TestTukeyEMAlgorithm:
    @pytest.mark.parametrize(
        "budget",
        [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)],
    )
    def test_refuses_non_approx_budget(self, budget: PureDPBudget | ZCDPBudget) -> None:
        metadata = lr_metadata(4)
        input_data = LinearRegressionInput(X=np.ones((4, 1)), y=np.ones(4))
        with pytest.raises(InvalidRegime, match="Cannot spend"):
            TukeyEM(m=4).run(metadata, input_data, budget=budget, seed=0)

    def test_native_approx_phase_composition_avoids_accounting_conversion(
        self,
        accounting_spy: AccountingSpy,
        fake_randomness: FakeRandomness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PTR draws Laplace at ``1 / eps_ptr``: half the epsilon, all the delta."""
        offered = ApproxDPBudget(eps=2.0, delta=1e-5)
        _ptr_draws(fake_randomness, 1e6)
        exp_budgets: list[PureDPBudget] = []
        real_sample_depth = tukey._sample_depth  # pyright: ignore[reportPrivateUsage]

        def select_depth(
            log_volumes: Array1D,
            t: int,
            budget: PureDPBudget,
            seed: int,
        ) -> int:
            exp_budgets.append(budget)
            return real_sample_depth(log_volumes, t, budget, seed)

        monkeypatch.setattr(tukey, "_sample_depth", select_depth)
        rng = np.random.default_rng(3)
        X = rng.normal(size=(8, 1))
        y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=8)

        TukeyEM(m=4).run(
            lr_metadata(8),
            LinearRegressionInput(X=X, y=y),
            budget=offered,
            seed=7,
        )

        assert accounting_spy.calls == ()
        assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx(
            [1.0 / 1.0]
        )
        (exp_budget,) = exp_budgets
        assert_budget_close(exp_budget, PureDPBudget(eps=1.0))

    def test_partition_models_route_through_the_canonical_ols_baseline(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        """Algorithm 2 lines 2--4: every row enters one of ``m`` near-equal blocks."""
        partition_sizes: list[int] = []

        def record_ols(
            _self: tukey.OLS,
            metadata: LinearRegressionMetadata,
            _input: LinearRegressionInput,
            budget: DPBudget,
            seed: int,
        ) -> LinearRegressionOutput:
            assert budget is NO_PRIVACY
            assert seed == 0
            partition_sizes.append(metadata.n_samples)
            return LinearRegressionOutput(
                beta=np.full(metadata.dim, metadata.n_samples)
            )

        monkeypatch.setattr(tukey.OLS, "run", record_ols)
        _ptr_draws(fake_randomness, 1e6)

        # n=9, m=4, dim=2: floor(9 / 4) = 2 >= 2 satisfies the block guard
        # while keeping an uneven split (sizes 3, 2, 2, 2).
        TukeyEM(m=4).run(
            lr_metadata(9, design=2),
            LinearRegressionInput(X=np.ones((9, 2)), y=np.ones(9)),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )

        assert sorted(partition_sizes) == [2, 2, 2, 3]

    def test_ptr_failure_is_no_release(self, fake_randomness: FakeRandomness) -> None:
        _ptr_draws(fake_randomness, -1e6)
        metadata = lr_metadata(8)
        input_data = LinearRegressionInput(
            X=np.ones((8, 1)), y=np.arange(8, dtype=float)
        )
        with pytest.raises(NoRelease, match="PTR"):
            TukeyEM(m=4).run(
                metadata,
                input_data,
                budget=ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    def test_seeded_run_succeeds_without_domains(
        self, fake_randomness: FakeRandomness
    ) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(80, 2))
        beta = np.array([1.5, -0.75])
        y = X @ beta + rng.normal(scale=0.1, size=80)
        metadata = lr_metadata(80, design=2)
        input_data = LinearRegressionInput(X=X, y=y)
        alg = TukeyEM(m=20)
        budget = ApproxDPBudget(eps=10.0, delta=0.1)
        fake_randomness.forget()
        _ptr_draws(fake_randomness, 1e6)

        output = alg.run(metadata, input_data, budget=budget, seed=9)

        assert output.beta.shape == (2,)
        assert np.all(np.isfinite(output.beta))
        # The partition, the tie-breaking perturbation and the exponential
        # samplers are separately accounted stages, so each draws from its own
        # child stream rather than the offered seed.
        seeds = fake_randomness.seeds
        assert len(seeds) > 1
        assert len(set(seeds)) == len(seeds)
        assert 9 not in seeds

    def test_paper_regime_passes_ptr_and_recovers_coefficients(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        d = 2
        X = rng.normal(size=(n, d))
        beta = np.array([1.5, -0.75])
        y = X @ beta + rng.normal(scale=0.2, size=n)
        output = TukeyEM(m=1000).run(
            lr_metadata(n, design=d),
            LinearRegressionInput(X=X, y=y),
            budget=ApproxDPBudget(eps=math.log(3.0), delta=1e-5),
            seed=0,
        )
        np.testing.assert_allclose(output.beta, beta, atol=0.1)

    def test_refuses_m_exceeding_n_samples_as_a_case_of_the_block_guard(self) -> None:
        """``m > n_samples`` is the ``floor(n_samples / m) < dim`` guard at its floor of 0."""
        metadata = lr_metadata(4, design=2)
        input_data = LinearRegressionInput(X=np.ones((4, 2)), y=np.ones(4))
        with pytest.raises(InvalidRegime, match=r"floor\(n_samples / m\) >= dim"):
            TukeyEM(m=5).run(
                metadata,
                input_data,
                budget=ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    def test_refuses_underdetermined_blocks(self) -> None:
        """The authors' code refuses when a block would hold fewer rows than dim."""
        metadata = lr_metadata(9, design=3)
        input_data = LinearRegressionInput(X=np.ones((9, 3)), y=np.ones(9))
        with pytest.raises(InvalidRegime, match=r"floor\(n_samples / m\) >= dim"):
            # floor(9 / 4) = 2 < dim = 3.
            TukeyEM(m=4).run(
                metadata,
                input_data,
                budget=ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )


class _UnreadInput:
    """An Input stand-in whose every attribute read fails the test."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"the Input was read: {name}")


class TestTukeyEMNOverDRule:
    """``m="n_over_d"`` runs ``floor(n_samples / dim)`` models (DGJ Section 4.2)."""

    def test_rule_fits_floor_n_over_dim_partition_models(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        partition_sizes: list[int] = []

        def record_ols(
            _self: tukey.OLS,
            metadata: LinearRegressionMetadata,
            _input: LinearRegressionInput,
            budget: DPBudget,
            seed: int,
        ) -> LinearRegressionOutput:
            partition_sizes.append(metadata.n_samples)
            return LinearRegressionOutput(
                beta=np.full(metadata.dim, metadata.n_samples)
            )

        monkeypatch.setattr(tukey.OLS, "run", record_ols)
        _ptr_draws(fake_randomness, 1e6)

        # n=23, dim=2: floor(23 / 2) = 11 models, ten of two rows and one of three.
        TukeyEM(m="n_over_d").run(
            lr_metadata(23, design=2),
            LinearRegressionInput(X=np.ones((23, 2)), y=np.ones(23)),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )

        assert sorted(partition_sizes) == [2] * 10 + [3]

    def test_rule_restricts_depth_to_a_quarter_of_the_resolved_count(
        self, monkeypatch: pytest.MonkeyPatch, fake_randomness: FakeRandomness
    ) -> None:
        """Algorithm 2's restriction ``t = floor(m / 4)`` reads the resolved ``m``."""
        restrictions: list[int] = []
        real_sample_depth = tukey._sample_depth  # pyright: ignore[reportPrivateUsage]

        def select_depth(
            log_volumes: Array1D, t: int, budget: PureDPBudget, seed: int
        ) -> int:
            restrictions.append(t)
            return real_sample_depth(log_volumes, t, budget, seed)

        monkeypatch.setattr(tukey, "_sample_depth", select_depth)
        _ptr_draws(fake_randomness, 1e6)
        rng = np.random.default_rng(4)
        X = rng.normal(size=(60, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=60)

        # floor(60 / 3) = 20 models, so t = 5.
        TukeyEM(m="n_over_d").run(
            lr_metadata(60, design=3),
            LinearRegressionInput(X=X, y=y),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=1,
        )

        assert restrictions == [5]

    def test_rule_below_four_models_is_refused_before_the_input_is_read(
        self,
    ) -> None:
        # floor(7 / 2) = 3 models, fewer than the four the restriction needs.
        with pytest.raises(InvalidRegime, match="n_over_d"):
            TukeyEM(m="n_over_d").run(
                lr_metadata(7, design=2),
                _UnreadInput(),  # pyright: ignore[reportArgumentType]
                budget=ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    def test_rule_at_four_models_meets_the_block_guard(
        self, fake_randomness: FakeRandomness
    ) -> None:
        """``floor(n / floor(n / d)) >= d``: the resolved count is never refused by it."""
        _ptr_draws(fake_randomness, 1e6)
        rng = np.random.default_rng(5)
        X = rng.normal(size=(9, 2))
        y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.1, size=9)

        # floor(9 / 2) = 4 models; floor(9 / 4) = 2 >= dim = 2.
        output = TukeyEM(m="n_over_d").run(
            lr_metadata(9, design=2),
            LinearRegressionInput(X=X, y=y),
            budget=ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=0,
        )

        assert output.beta.shape == (2,)

    @pytest.mark.parametrize("m", ["n_over_dim", "N_OVER_D", 3])
    def test_other_names_and_counts_below_four_are_rejected(self, m: object) -> None:
        with pytest.raises(ValueError):
            TukeyEM(m=m)  # pyright: ignore[reportArgumentType]

    def test_rule_is_its_own_configuration_identity(self) -> None:
        rule = TukeyEM(m="n_over_d")
        assert rule.m == "n_over_d"
        assert rule.uuid != TukeyEM(m=1000).uuid
        assert TukeyEM.model_validate(rule.model_dump()).uuid == rule.uuid
