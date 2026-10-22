"""Focused tests for Chang et al.'s sparse Huber IHT estimator."""

from __future__ import annotations

from typing import cast

import math

import numpy as np
import pytest

import dpstats.tasks.linear_regression.algorithms.tail_robust_dpgd as trd
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D
from dpstats.tasks.linear_regression.algorithms import (
    TailRobustSparseHuberIHT,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw
from tests.unit.tasks.task_builders import Unreadable, lr_metadata


def _intercept_design(dim: int) -> ProductDomain:
    return ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=dim - 1)))


def _instance(
    n: int = 80,
    dim: int = 12,
    has_intercept: bool = True,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    X = np.zeros((n, dim), dtype=np.float64)
    if has_intercept:
        X[:, 0] = 1.0
    design = _intercept_design(dim) if has_intercept else dim
    return lr_metadata(n, design=design), LinearRegressionInput(X=X, y=np.zeros(n))


class TestPaperBoundaries:
    def test_support_recovery_keeps_intercept_and_uses_add_remove_scale(
        self, fake_randomness: FakeRandomness
    ):
        fake_randomness.script("laplace", zero_draw)
        X = np.tile(np.array([1.0, 0.2, 0.1, 0.3, 0.4]), (2, 1))
        selected = trd._private_sparse_support(  # pyright: ignore[reportPrivateUsage]
            X,
            np.ones(2),
            n=2,
            dim=5,
            sparsity=3,
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )

        observed_scales = fake_randomness.parameters_of("laplace", "scale")
        assert selected == [0, 3, 4]
        assert observed_scales
        assert all(
            scale == pytest.approx(2.0 * 2 * math.sqrt(math.log(10)) / 2)
            for scale in observed_scales
        )

    def test_algorithm_two_uses_the_paper_scale_and_exact_sparsity(
        self, fake_randomness: FakeRandomness
    ):
        fake_randomness.script("laplace", zero_draw)
        vector = np.arange(20, dtype=np.float64)
        budget = ApproxDPBudget(eps=0.1, delta=1e-4)
        output = trd._run_algorithm_2(  # pyright: ignore[reportPrivateUsage]
            vector,
            sparsity=10,
            budget=budget,
            sensitivity=0.25,
            seed=0,
        )

        expected_scale = 2.0 * 0.25 * math.sqrt(5.0 * 10 * math.log(1e4)) / 0.1
        observed_scales = fake_randomness.parameters_of("laplace", "scale")
        assert observed_scales
        assert all(scale == pytest.approx(expected_scale) for scale in observed_scales)
        assert np.array_equal(np.flatnonzero(output), np.arange(10, 20))

    def test_schedule_uses_the_lower_noise_paper_composition(self):
        offered = ApproxDPBudget(eps=2.0, delta=0.02)
        standard = trd._noisy_ht_step_budget(  # pyright: ignore[reportPrivateUsage]
            offered,
            trd._noisy_ht_step_schedules(  # pyright: ignore[reportPrivateUsage]
                offered, num_steps=10
            ),
            sparsity=10,
            sensitivity=0.25,
        )
        assert_budget_close(standard, ApproxDPBudget(eps=0.2, delta=0.002))

        offered = ApproxDPBudget(eps=1.0, delta=1e-5)
        advanced = trd._noisy_ht_step_budget(  # pyright: ignore[reportPrivateUsage]
            offered,
            trd._noisy_ht_step_schedules(  # pyright: ignore[reportPrivateUsage]
                offered, num_steps=1
            ),
            sparsity=10,
            sensitivity=0.25,
        )
        assert isinstance(advanced, ApproxDPBudget)
        assert advanced.eps == pytest.approx(math.sqrt(2.0 / (5.0 * math.log(2e5))))
        assert advanced.delta == pytest.approx(5e-6)

    def test_schedule_refuses_when_no_paper_composition_is_admissible(self):
        with pytest.raises(InvalidRegime, match="No paper-valid NoisyHT"):
            trd._noisy_ht_step_schedules(  # pyright: ignore[reportPrivateUsage]
                ApproxDPBudget(eps=2.0, delta=1e-5),
                num_steps=1,
            )

    @pytest.mark.parametrize("n", [10, 2])
    def test_algorithm_three_uses_public_n_add_remove_sensitivity(
        self, fake_randomness: FakeRandomness, n: int
    ):
        """Algorithm 2 receives ``lambda = learning_rate * gamma * tau / n``.

        Its Laplace scale is ``2 lambda sqrt(5 s log(1/delta)) / eps``; zero
        draws with ``s`` equal to the dimension keep every coordinate, so the
        released step is the noise-free update.
        """
        fake_randomness.script("laplace", zero_draw)
        offered_budget = ApproxDPBudget(eps=0.2, delta=1e-4)
        X = np.ones((1, 10))
        beta = trd._run_algorithm_3(  # pyright: ignore[reportPrivateUsage]
            X,
            np.ones(1),
            n=n,
            beta_init=np.zeros(10),
            tau=2.0,
            gamma=1.0,
            learning_rate=0.25,
            num_steps=1,
            sparsity=10,
            budget=offered_budget,
            step_schedules=(offered_budget,),
            seed=0,
        )

        # The update average and Lemma D.1's bracket both divide by the public
        # ``n``: the lemma is applied with non-strict domination, so the bracket
        # is the add/remove sensitivity ``learning_rate * gamma * tau / n``.
        np.testing.assert_allclose(beta, 0.25 / n)
        sensitivity = 0.25 * 1.0 * 2.0 / n
        expected_scale = 2.0 * sensitivity * math.sqrt(5.0 * 10 * math.log(1e4)) / 0.2
        scales = fake_randomness.parameters_of("laplace", "scale")
        assert scales
        assert scales == pytest.approx([expected_scale] * len(scales))


class TestTailRobustSparseHuberIHT:
    def test_run_owns_the_section_5_1_2_budget_split(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class Captured:
            support_budget: PureDPBudget
            initializer_budget: DPBudget
            initializer_X: Array2D
            initializer_metadata: LinearRegressionMetadata
            main_budget: DPBudget
            beta_init: Array1D
            tau: float
            gamma: float
            num_steps: int

        metadata, input_data = _instance(dim=15)
        input_data.X[:] = np.arange(metadata.dim)
        input_data.X[:, 0] = 1.0
        selected_indices = [0, *range(2, 13)]
        captured = Captured()

        def support(
            X: Array2D,
            y: Array1D,
            n: int,
            dim: int,
            sparsity: int,
            budget: PureDPBudget,
            seed: int,
        ) -> list[int]:
            captured.support_budget = budget
            return selected_indices

        def initializer(
            X: Array2D,
            y: Array1D,
            metadata: LinearRegressionMetadata,
            budget: DPBudget,
            seed: int,
        ) -> tuple[Array1D, float]:
            captured.initializer_budget = budget
            captured.initializer_X = X.copy()
            captured.initializer_metadata = metadata
            return np.arange(metadata.dim, dtype=np.float64), 1.0

        def algorithm_three(
            X: Array2D,
            y: Array1D,
            n: int,
            beta_init: Array1D,
            tau: float,
            gamma: float,
            learning_rate: float,
            num_steps: int,
            sparsity: int,
            budget: DPBudget,
            step_schedules: tuple[ApproxDPBudget, ...],
            seed: int,
        ) -> Array1D:
            captured.main_budget = budget
            captured.beta_init = beta_init.copy()
            captured.tau = tau
            captured.gamma = gamma
            captured.num_steps = num_steps
            return beta_init

        monkeypatch.setattr(trd, "_private_sparse_support", support)
        monkeypatch.setattr(trd, "_private_initializer", initializer)
        monkeypatch.setattr(trd, "_run_algorithm_3", algorithm_three)
        budget = ApproxDPBudget(eps=3.0, delta=6e-5)
        output = TailRobustSparseHuberIHT().run(metadata, input_data, budget, seed=0)

        assert_budget_close(captured.support_budget, PureDPBudget(eps=1.0))
        assert_budget_close(
            captured.initializer_budget, ApproxDPBudget(eps=1.0, delta=3e-5)
        )
        assert_budget_close(captured.main_budget, ApproxDPBudget(eps=1.0, delta=3e-5))
        selected_metadata = captured.initializer_metadata
        assert selected_metadata.has_intercept
        assert selected_metadata.dim == 12
        assert selected_metadata.response_domain is metadata.response_domain
        assert isinstance(selected_metadata.design_domain, ProductDomain)
        assert isinstance(selected_metadata.design_domain.parts[0], InterceptDomain)
        assert isinstance(selected_metadata.design_domain.parts[1], EuclideanSpace)
        initializer_X = captured.initializer_X
        assert np.array_equal(initializer_X, input_data.X[:, selected_indices])
        expected_beta = np.zeros(metadata.dim)
        expected_beta[selected_indices] = np.arange(12)
        beta_init = captured.beta_init
        assert np.array_equal(beta_init, expected_beta)
        assert np.array_equal(output.beta, expected_beta)
        assert captured.gamma == pytest.approx(
            0.5 * math.sqrt(math.log(metadata.dim * metadata.n_samples))
        )
        assert captured.num_steps == math.ceil(2.0 * math.log(metadata.n_samples))
        assert captured.tau == pytest.approx(
            0.04
            * math.sqrt(
                metadata.n_samples
                * budget.eps
                / (12 * math.log(metadata.dim) + math.log(metadata.n_samples))
            )
        )

    def test_release_has_exactly_sparsity_nonzero_coordinates(self):
        metadata, input_data = _instance()
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        output = TailRobustSparseHuberIHT().run(metadata, input_data, budget, seed=7)

        assert np.count_nonzero(output.beta) == 12

    @pytest.mark.parametrize(
        "budget",
        [
            NO_PRIVACY,
            PureDPBudget(eps=math.inf),
            ApproxDPBudget(eps=math.inf, delta=1e-5),
            ZCDPBudget(rho=math.inf),
        ],
    )
    def test_refuses_the_unconstrained_offer(self, budget: DPBudget):
        # The NoisyHT schedule refuses every large epsilon, so no limit exists.
        metadata, input_data = _instance()
        with pytest.raises(InvalidRegime, match="no unconstrained limit"):
            TailRobustSparseHuberIHT().run(metadata, input_data, budget, seed=0)

    @pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
    def test_refuses_unsupported_budget_family(self, budget: PureDPBudget | ZCDPBudget):
        metadata, input_data = _instance()
        with pytest.raises(InvalidRegime, match="Cannot spend"):
            TailRobustSparseHuberIHT().run(metadata, input_data, budget, seed=0)

    def test_refuses_publicly_unsupported_configuration(self):
        metadata, input_data = _instance(has_intercept=False)
        with pytest.raises(InvalidRegime, match="intercept"):
            TailRobustSparseHuberIHT().run(
                metadata,
                input_data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

        with pytest.raises(InvalidRegime, match="sparsity"):
            TailRobustSparseHuberIHT(sparsity=13).run(
                lr_metadata(80, design=_intercept_design(12)),
                input_data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

        single_metadata, single_input = _instance(n=1)
        with pytest.raises(InvalidRegime, match="n_samples"):
            TailRobustSparseHuberIHT().run(
                single_metadata,
                single_input,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    def test_run_refuses_a_budget_no_paper_composition_admits(self):
        metadata, _ = _instance()
        unreadable = cast(LinearRegressionInput, Unreadable("Input"))
        with pytest.raises(InvalidRegime, match="No paper-valid NoisyHT"):
            TailRobustSparseHuberIHT().run(
                metadata,
                unreadable,
                ApproxDPBudget(eps=100.0, delta=1e-5),
                seed=0,
            )
