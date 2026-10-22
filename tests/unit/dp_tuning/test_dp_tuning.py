"""Public behavior of the private hyperparameter-selection tuners."""

import math
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import NoRelease
from dpstats.budgets import ApproxDPBudget, PureDPBudget, ZCDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.dp_tuning import (
    DPExhaustiveCompose,
    DPSelectionLiuTalwar,
    DPSelectionPapernotSteinkePoisson,
    DPSelectionPapernotSteinkeTNB,
)
from dpstats.parameters import parameter_field, search_range
from dpstats.simulations.linear_regression import simulate_clipped_gaussian
from dpstats.tasks import LinearRegression, list_built_in_algorithms
from tests.fixtures.accounting_spy import assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness

APPROX = ApproxDPBudget(eps=3.0, delta=1e-5)


class EchoAlg(BaseAlgorithm):
    """A tunable candidate that releases its own configuration.

    Every run is logged in ``RUNS`` so a test sees which candidates a tuner
    trained, and a negative ``a`` declines to release.
    """

    TASK = LinearRegression
    RUNS: ClassVar[list[dict[str, Any]]] = []

    a: int = 0
    b: int = 0
    agg: str = "last"
    lr: float = parameter_field(
        default=1.0,
        search=search_range(1e-3, 1e3, scale="log"),
        description="A log-scale tuning knob.",
    )
    reg: float = parameter_field(
        default=1.0,
        search=search_range(0.0, 2.0),
        description="A linear-scale tuning knob.",
    )

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def run(self, metadata, input, budget, seed):
        released = {
            "a": self.a,
            "b": self.b,
            "agg": self.agg,
            "lr": self.lr,
            "reg": self.reg,
        }
        EchoAlg.RUNS.append(released)
        if self.a < 0:
            raise NoRelease(f"candidate a={self.a} is infeasible")
        return released


@pytest.fixture(autouse=True)
def _fresh_run_log():
    EchoAlg.RUNS.clear()
    yield
    EchoAlg.RUNS.clear()


@pytest.fixture
def instance():
    """A public Gaussian linear-regression problem of 200 rows."""
    metadata, input_data, _ = simulate_clipped_gaussian(seed=42, dim=2, n_samples=200)
    return metadata, input_data


def _recording_cost(
    calls: list[dict[str, Any]],
    costs: Mapping[int, float] | Callable[[int], float] = lambda a: 0.0,
) -> Callable[..., float]:
    """Build a selection cost that records each call and scores candidates by ``a``."""

    def selection_cost(output: Any, **arguments: Any) -> float:
        calls.append({"a": output["a"], **arguments})
        return costs(output["a"]) if callable(costs) else costs[output["a"]]

    return selection_cost


def _zero_cost(output: Any, **arguments: Any) -> float:
    return 0.0


def _fixed_selection_cost(
    output: Any,
    evaluation_metadata: Any,
    evaluation_input: Any,
    budget: Any,
    seed: int,
    offset: float,
) -> float:
    """Selection cost with an explicitly configured argument."""
    return float(output["a"] + offset)


# Each tuner family over EchoAlg; low gamma and high mu draw many candidates.
TUNERS: dict[str, Callable[..., Any]] = {
    "exhaustive": lambda **kw: DPExhaustiveCompose(alg_class=EchoAlg, **kw),
    "liu-talwar": lambda **kw: DPSelectionLiuTalwar(
        alg_class=EchoAlg, gamma=0.05, **kw
    ),
    "papernot-steinke": lambda **kw: DPSelectionPapernotSteinkePoisson(
        alg_class=EchoAlg, mu=10.0, **kw
    ),
}


class TestSelectionCostWiring:
    """The tuner hands a typed budget, seed, full data and arguments on."""

    def test_a_tuner_composes_as_the_composition_algorithm(self):
        from dpstats.tasks.linear_regression.transformations import (
            ToMEstimation,
        )
        from dpstats.tasks.m_estimation.algorithms.dp_gradient_descent import (
            DPSGD,
        )

        tuner = DPSelectionLiuTalwar(
            alg_class=DPSGD,
            alg_hyperparam_grid={"num_steps": [2, 100]},
            selection_cost=_zero_cost,
        )
        composed = CompositeAlgorithm(
            transformations=(ToMEstimation(),), algorithm=tuner
        )

        assert composed.task is LinearRegression
        assert composed.algorithm is tuner
        candidate = tuner.build_candidate({"num_steps": 2})
        assert isinstance(candidate, DPSGD)
        assert candidate.num_steps == 2

    def test_accepts_an_ordinary_selection_cost_function(self, instance):
        tuner = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2, 3]},
            selection_cost=_fixed_selection_cost,
            selection_cost_args={"offset": -2.0},
        )

        output = tuner.run(*instance, ApproxDPBudget(eps=3.0, delta=1e-5), seed=0)

        assert output["a"] == 1

    def test_a_composition_is_not_a_tuner_argument(self):
        from typing import ClassVar

        from dpstats.algorithm import Transformation
        from dpstats.budgets import DPBudget
        from dpstats.task import Task
        from dpstats.tasks.linear_regression.algorithms.ols import OLS

        class _PublicOnlyTransformation(Transformation):
            """Pass the problem through unchanged; a trivial composition partner."""

            OUTER_TASK: ClassVar[Task] = LinearRegression
            INNER_TASK: ClassVar[Task] = LinearRegression

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            def prepare(
                self, metadata: Any, input: Any, budget: DPBudget, seed: int
            ) -> tuple[Any, Any, DPBudget, int, None]:
                return metadata, input, budget, seed, None

            def restore(self, output: Any, restoration: None) -> Any:
                return output

        configured = CompositeAlgorithm(
            transformations=(_PublicOnlyTransformation(),), algorithm=OLS()
        )

        with pytest.raises(ValueError, match="subclass of BaseAlgorithm"):
            DPSelectionLiuTalwar(
                alg_class=configured,
                alg_hyperparam_grid={"num_steps": [2]},
                selection_cost=_zero_cost,
            )

    def test_kwargs_reach_selection_cost(self, instance):
        from dpstats.dp_tuning.selection_costs.linear_regression import (
            laplace_clipped_sum_of_squared_errors,
        )

        recorded = []

        def spy_fn(output, **kwargs):
            recorded.append(dict(kwargs))
            return laplace_clipped_sum_of_squared_errors(output, **kwargs)

        alg_class = list_built_in_algorithms(LinearRegression)["SSP"]
        tuner = DPSelectionLiuTalwar(
            alg_class=alg_class,
            alg_hyperparam_grid=alg_class.default_tuning_grid(),
            selection_cost=spy_fn,
            gamma=0.5,
        )

        output = tuner.run(
            *instance, budget=ApproxDPBudget(eps=2.0, delta=1e-5), seed=0
        )

        assert output.beta.shape == (2,)
        assert len(recorded) >= 1
        for kwargs in recorded:
            assert set(kwargs) == {
                "evaluation_metadata",
                "evaluation_input",
                "budget",
                "seed",
            }
            assert isinstance(kwargs["budget"], ApproxDPBudget)
            assert isinstance(kwargs["seed"], int)
            assert kwargs["evaluation_metadata"].n_samples == 200
            assert kwargs["evaluation_input"].X.shape[0] == 200

        seeds = [kwargs["seed"] for kwargs in recorded]
        assert len(set(seeds)) == len(seeds)


class TestDPSelectionLiuTalwarAccounting:
    """The Liu-Talwar Theorem 3.4 split realizes the requested (eps, delta).

    The per-run budget the selection cost receives is (eps_base, delta_base);
    the selected output is then
        eps_out   = 3*eps_base + 3*sqrt(2*delta_base)
        delta_out = sqrt(2*delta_base)*max_trials + delta_2
    which must equal the requested (eps, delta).
    """

    def test_approx_dp_composes_to_requested_budget(self, instance):
        from dpstats.dp_tuning.selection_costs.linear_regression import (
            laplace_clipped_sum_of_squared_errors,
        )

        eps, delta, gamma = 2.0, 1e-5, 0.5
        recorded = []

        def spy_fn(output, **kwargs):
            recorded.append(dict(kwargs))
            return laplace_clipped_sum_of_squared_errors(output, **kwargs)

        alg_class = list_built_in_algorithms(LinearRegression)["SSP"]
        tuner = DPSelectionLiuTalwar(
            alg_class=alg_class,
            alg_hyperparam_grid=alg_class.default_tuning_grid(),
            selection_cost=spy_fn,
            gamma=gamma,
        )
        tuner.run(*instance, budget=ApproxDPBudget(eps=eps, delta=delta), seed=0)

        assert recorded, "selection cost was never called"
        scoring_budget = recorded[0]["budget"]
        assert isinstance(scoring_budget, ApproxDPBudget)
        eps_base = scoring_budget.eps / tuner.scoring_share
        delta_base = scoring_budget.delta / tuner.scoring_share
        delta_2 = delta / 2
        max_trials = float(np.ceil(-np.log(delta_2) / gamma))
        eps_out = 3 * eps_base + 3 * np.sqrt(2 * delta_base)
        delta_out = np.sqrt(2 * delta_base) * max_trials + delta_2
        assert np.isclose(eps_out, eps)
        assert np.isclose(delta_out, delta)

    def test_approx_dp_below_the_reserve_refuses_before_any_run(self, instance):
        """delta=0.5, gamma=0.5: max_trials = ceil(log(4)/0.5) = 3, reserve 0.25."""
        calls: list[dict[str, Any]] = []
        tuner = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2]},
            selection_cost=_recording_cost(calls),
            gamma=0.5,
        )

        with pytest.raises(InvalidRegime, match="eps must be at least"):
            tuner.run(*instance, budget=ApproxDPBudget(eps=0.1, delta=0.5), seed=0)
        assert calls == []

    def test_pure_dp_gives_each_run_a_third_of_eps(self, instance):
        """Under pure DP each run spends eps/3 with no delta inflation."""
        calls: list[dict[str, Any]] = []
        tuner = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2]},
            selection_cost=_recording_cost(calls),
            gamma=0.5,
        )

        tuner.run(*instance, budget=PureDPBudget(eps=3.0), seed=0)

        assert calls, "selection cost was never called"
        assert_budget_close(calls[0]["budget"], PureDPBudget(eps=1.0 * 0.10))

    @pytest.mark.parametrize("gamma", [0.0, -0.1, 1.1, float("nan"), float("inf")])
    def test_gamma_is_validated_before_selection(self, gamma):
        with pytest.raises(ValueError, match="gamma"):
            DPSelectionLiuTalwar(
                alg_class=EchoAlg,
                alg_hyperparam_grid={},
                selection_cost=_zero_cost,
                gamma=gamma,
            )


class TestTunerConstructionValidation:
    def test_search_grid_is_detached_from_the_callers_container(self):
        candidates = [1, 2]
        tuner = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": candidates},
            selection_cost=_zero_cost,
        )
        candidates.append(3)

        assert tuner.alg_hyperparam_grid == {"a": [1, 2]}

    def test_grid_candidates_take_the_childs_field_types(self):
        tuner = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"lr": [1, 2]},
            selection_cost=_zero_cost,
        )

        candidates = tuner.alg_hyperparam_grid["lr"]
        assert [type(candidate) for candidate in candidates] == [float, float]
        assert candidates == [1.0, 2.0]

    def test_equivalent_grid_authorings_share_one_identity(self):
        integer_authored = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"lr": [1, 2]},
            selection_cost=_zero_cost,
        )
        float_authored = DPSelectionLiuTalwar(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"lr": [1.0, 2.0]},
            selection_cost=_zero_cost,
        )

        assert integer_authored.uuid == float_authored.uuid

    def test_rejects_a_candidate_the_child_would_reject(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=r"a\.1"):
            DPSelectionLiuTalwar(
                alg_class=EchoAlg,
                alg_hyperparam_grid={"a": [1, 2.5], "agg": ["last"]},
                selection_cost=_zero_cost,
            )

    def test_rejects_undeclared_search_parameter(self):
        with pytest.raises(ValueError, match="not declared.*missing"):
            DPSelectionLiuTalwar(
                alg_class=EchoAlg,
                alg_hyperparam_grid={"missing": [1, 2]},
                selection_cost=_zero_cost,
            )

    def test_rejects_noncallable_objective(self):
        with pytest.raises(ValueError, match="Configured object paths"):
            DPSelectionLiuTalwar(
                alg_class=EchoAlg,
                alg_hyperparam_grid={"a": [1, 2]},
                selection_cost="not-a-callable",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        ("alg_class", "grid"),
        [
            pytest.param(
                CompositeAlgorithm, {"algorithm": [EchoAlg()]}, id="composite"
            ),
            pytest.param(
                DPExhaustiveCompose, {"selection_cost_args": [{}]}, id="tuner"
            ),
        ],
    )
    def test_rejects_a_class_that_is_not_task_native_as_the_tuned_class(
        self, alg_class, grid
    ):
        with pytest.raises(TypeError, match="compose the tuner inside"):
            DPExhaustiveCompose(
                alg_class=alg_class,
                alg_hyperparam_grid=grid,
                selection_cost=_zero_cost,
            )


class TestPapernotSteinkeSelectors:
    """Privacy-soundness tests for the two Papernot--Steinke selectors.

    Each selector executes one repeat distribution and charges the one privacy
    argument that covers it: Corollary 4 for the truncated negative binomial,
    the corrected Theorem 6 for Poisson. Neither reads a base privacy family,
    and the budget its argument calibrates is what every candidate receives.
    """

    def _tnb_tuner(self, *, grid, alg_class=EchoAlg, selection_cost=_zero_cost, **kw):
        return DPSelectionPapernotSteinkeTNB(
            alg_class=alg_class,
            alg_hyperparam_grid=grid,
            selection_cost=selection_cost,
            **kw,
        )

    def _poisson_tuner(
        self, *, grid, alg_class=EchoAlg, selection_cost=_zero_cost, **kw
    ):
        return DPSelectionPapernotSteinkePoisson(
            alg_class=alg_class,
            alg_hyperparam_grid=grid,
            selection_cost=selection_cost,
            **kw,
        )

    def test_a_selector_refuses_the_other_routes_parameter(self):
        with pytest.raises(ValueError, match=r"\bmu\b"):
            self._tnb_tuner(grid={"a": [1]}, mu=10.0)

        with pytest.raises(ValueError, match=r"\beta\b"):
            self._poisson_tuner(grid={"a": [1]}, eta=1.0)

    @pytest.mark.parametrize("eta", [-0.5, -1.0])
    def test_unsupported_negative_eta_rejected_at_construction(self, eta):
        with pytest.raises(ValueError, match=r"\beta\b"):
            self._tnb_tuner(grid={"a": [1]}, eta=eta)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, 5.0, -0.1, float("nan"), float("inf")])
    def test_tnb_gamma_validated_at_construction(self, gamma):
        """A gamma outside (0, 1) is rejected: Corollary 4 needs log(1/gamma)."""
        with pytest.raises(ValueError, match="gamma"):
            self._tnb_tuner(grid={"a": [1]}, eta=1.0, gamma=gamma)

    @pytest.mark.parametrize("mu", [0.0, -1.0, float("nan")])
    def test_poisson_mu_validated_at_construction(self, mu):
        """A non-positive or NaN Poisson rate is not a repeat distribution."""
        with pytest.raises(ValueError, match=r"\bmu\b"):
            self._poisson_tuner(grid={"a": [1]}, mu=mu)

    def test_default_poisson_mu_is_public_fixed_value(self):
        assert self._poisson_tuner(grid={"a": [1, 2, 3]}).mu == 10.0

    # ---- The corrected Poisson bound against an independent oracle -------
    @pytest.mark.parametrize(
        ("mu", "expected_divergence", "expected_bound"),
        [
            pytest.param(0.1, 0.031222, 0.037460, id="below-one"),
            pytest.param(1.0, 0.1876281, 0.5313414, id="at-one"),
            pytest.param(10.0, 0.0726790, 2.5902706, id="above-one"),
        ],
    )
    def test_corrected_poisson_bound_covers_finite_two_point_oracle(
        self, mu, expected_divergence, expected_bound
    ):
        """The corrected bound covers a direct oracle across ``mu = 1``.

        ``Bernoulli(1/2)`` versus ``Bernoulli(1/4)`` is ``(log 2, 0)``-DP in
        both directions, so ``delta_hat = 0`` is exact.
        """
        from dpstats.dp_tuning.papernot_steinke import (
            _poisson_repeat_rdp_bound,
        )

        base_rdp = math.log(4.0 / 3.0)
        repaired = _poisson_repeat_rdp_bound(2.0, base_rdp, 0.0, mu)

        def selected_distribution(success_probability):
            # Poisson PGF: Pr(no success) = E[(1-p)^K] = exp(-mu*p).
            bottom = math.exp(-mu)
            no_success = math.exp(-mu * success_probability)
            return bottom, no_success - bottom, 1.0 - no_success

        selected = selected_distribution(0.5)
        neighboring = selected_distribution(0.25)
        assert selected[0] == neighboring[0] == math.exp(-mu)
        true_divergence = math.log(sum(p**2 / q for p, q in zip(selected, neighboring)))

        assert true_divergence == pytest.approx(expected_divergence, abs=5e-7)
        assert repaired == pytest.approx(expected_bound, abs=5e-7)
        assert repaired >= 0.0
        assert true_divergence <= repaired
        if mu == 0.1:
            published = base_rdp + math.log(mu)
            assert published == pytest.approx(-2.014903, abs=5e-7)
            assert published < true_divergence

    # ---- The calibrated budget reaches every candidate ----------------
    @pytest.mark.parametrize("eps", [0.5, 1.0, 3.0])
    @pytest.mark.parametrize("delta", [1e-5, 1e-7])
    @pytest.mark.parametrize("mu", [0.1, 1.0, 10.0])
    def test_poisson_charges_within_budget_and_hands_that_budget_on(
        self, eps, delta, mu, instance, fake_randomness: FakeRandomness
    ):
        """One zCDP budget reaches every candidate, and its charge fits eps.

        The repeat count is scripted to a fixed positive value so that a small
        rate's zero draw does not decide whether the budget is observed.
        """
        from dpstats.dp_tuning.papernot_steinke import _ps_epsilon_poisson

        calls: list[dict[str, Any]] = []
        tuner = self._poisson_tuner(
            grid={"a": [1, 2, 3]}, selection_cost=_recording_cost(calls), mu=mu
        )
        fake_randomness.script("poisson", lambda _parameters: 2)

        tuner.run(*instance, ApproxDPBudget(eps=eps, delta=delta), 0)

        handed = [call["budget"] for call in calls]
        assert handed
        assert all(budget == handed[0] for budget in handed)
        assert isinstance(handed[0], ZCDPBudget)
        assert handed[0].rho > 0.0
        assert _ps_epsilon_poisson(handed[0], mu, delta) <= eps + 1e-9

    @pytest.mark.parametrize("eps", [0.5, 3.0, 100.0])
    def test_tnb_charges_within_budget_and_respects_the_corollary_rho_cap(
        self, eps, instance
    ):
        """Corollary 4 holds only up to ``log(1/gamma)``, so the charge caps there."""
        from dpstats.dp_tuning.papernot_steinke import _ps_epsilon_tnb

        delta = 1e-6
        calls: list[dict[str, Any]] = []
        tuner = self._tnb_tuner(
            grid={"a": [7]},
            selection_cost=_recording_cost(calls),
            eta=1.0,
            gamma=0.05,
        )

        tuner.run(*instance, ApproxDPBudget(eps=eps, delta=delta), 0)

        handed = [call["budget"] for call in calls]
        assert handed
        assert all(budget == handed[0] for budget in handed)
        assert isinstance(handed[0], ZCDPBudget)
        assert 0.0 < handed[0].rho <= math.log(1.0 / tuner.gamma)
        assert _ps_epsilon_tnb(handed[0], tuner.eta, tuner.gamma, delta) <= eps + 1e-9

    def test_budget_does_not_depend_on_the_wrapped_base_class(self, instance):
        """The charge reads the budget and the repeat law, never the base.

        Two distinct base classes under the same budget and repeat rate
        receive the identical per-run budget, because no base privacy
        family is read.
        """

        class OtherEchoAlg(EchoAlg):
            """A distinct class with the same configuration."""

        handed_rhos = []
        for base in (EchoAlg, OtherEchoAlg):
            calls: list[dict[str, Any]] = []
            tuner = self._poisson_tuner(
                grid={"a": [7]},
                alg_class=base,
                selection_cost=_recording_cost(calls),
                mu=10.0,
            )
            tuner.run(*instance, ApproxDPBudget(eps=1.0, delta=1e-6), 0)
            assert calls
            handed_rhos.append(calls[0]["budget"].rho)

        assert handed_rhos[0] == pytest.approx(handed_rhos[1])

    def test_realized_k_never_released(self, instance):
        """Different repeat draws release only the selected task output."""
        outputs = [
            self._poisson_tuner(grid={"a": [7]}, mu=10.0).run(*instance, APPROX, seed)
            for seed in (0, 12345)
        ]

        assert outputs == [{"a": 7, "b": 0, "agg": "last", "lr": 1.0, "reg": 1.0}] * 2

    def test_poisson_zero_draw_releases_bottom_not_a_clamped_candidate(
        self, instance, fake_randomness: FakeRandomness
    ):
        """A Poisson K=0 draw fails selection with the fixed bottom outcome.

        Clamping K to 1 would execute a mechanism the corrected Poisson
        accounting does not cover: the K=0 outcome belongs to the charged
        mechanism's output space and must surface as a principled failure, not
        as a run of an uncharged candidate.
        """
        tuner = self._poisson_tuner(grid={"a": [1, 2, 3]}, mu=0.1)
        fake_randomness.script("poisson", lambda _parameters: 0)

        with pytest.raises(NoRelease, match="drew no candidates"):
            tuner.run(*instance, ApproxDPBudget(eps=3.0, delta=1e-6), 0)

        assert not EchoAlg.RUNS

    def test_tnb_always_trains_at_least_one_candidate(self, instance):
        """The TNB count is zero-truncated, so this route has no bottom draw."""
        budget = ApproxDPBudget(eps=3.0, delta=1e-6)
        for seed in range(20):
            EchoAlg.RUNS.clear()
            tuner = self._tnb_tuner(grid={"a": [1, 2, 3]}, eta=1.0, gamma=0.9)
            tuner.run(*instance, budget, seed)
            assert EchoAlg.RUNS

    def test_infeasible_budget_surfaces_its_numerical_failure(self, instance):
        """A budget too small to calibrate propagates its numerical failure.

        No positive per-run budget charges as little as this epsilon, so the
        calibration probes toward zero until the RDP accountant leaves its own
        numerical domain. That failure is not a publicly discernible limit this
        selector declares, so it is neither swallowed nor re-typed as one.
        """
        tuner = self._poisson_tuner(grid={"a": list(range(1, 50))})

        with pytest.raises(ValueError) as failure:
            tuner.run(*instance, ApproxDPBudget(eps=1e-6, delta=1e-6), 0)

        # Exactly ValueError: neither the declared InvalidRegime refusal nor a
        # model-validation error stands in for the accountant's own failure.
        assert type(failure.value) is ValueError
        assert not isinstance(failure.value, InvalidRegime)


class TestDPExhaustiveCompose:
    """The exhaustive tuner trains every candidate of a finite discrete grid.

    Each candidate's budget share is owned by the per-family composition test
    in ``test_typed_budgets.py``; the argmin here is post-processing of
    exactly one private cost per candidate.
    """

    @pytest.mark.parametrize(
        "grid",
        [{"a": {"lower": 0.0, "upper": 1.0}}, {"a": []}],
        ids=["continuous-range", "empty-list"],
    )
    def test_constructor_refuses_a_grid_that_is_not_finite_discrete_lists(self, grid):
        with pytest.raises(ValueError, match="discrete candidate lists"):
            DPExhaustiveCompose(
                alg_class=EchoAlg,
                alg_hyperparam_grid=grid,
                selection_cost=_zero_cost,
            )

    def test_introspection_advertises_only_discrete_candidate_lists(self):
        info = DPExhaustiveCompose.parameter_info()["alg_hyperparam_grid"]
        assert info.required is True
        assert info.search is None

    def test_every_product_candidate_is_trained_and_scored_once(self, instance):
        """Six candidates, six trainings, six costs, and the winner is not refit."""
        calls: list[dict[str, Any]] = []
        tuner = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2, 3], "b": [10, 20]},
            selection_cost=_recording_cost(calls, lambda a: abs(a - 2)),
        )

        output = tuner.run(*instance, budget=APPROX, seed=0)

        assert sorted((run["a"], run["b"]) for run in EchoAlg.RUNS) == [
            (a, b) for a in (1, 2, 3) for b in (10, 20)
        ]
        assert len(calls) == 6
        assert output["a"] == 2


class TestCategoricalHyperparameterGrids:
    """Categorical (str-valued) hyperparameters are enumerable in search grids."""

    def test_dpsgd_tuning_grid_enumerates_categorical(self):
        from dpstats.tasks.m_estimation.algorithms.dp_gradient_descent import (
            DPSGD,
        )

        grid = DPSGD.default_tuning_grid(n_points=3)
        assert grid["iterate_aggregation"] == ["last", "tail_average", "polyak"]
        assert len(grid["eta"]) == 3

    def test_a_random_tuner_draws_declared_categorical_values(self, instance):
        values = ["last", "tail_average", "polyak"]
        tuner = TUNERS["liu-talwar"](
            alg_hyperparam_grid={"agg": values}, selection_cost=_zero_cost
        )

        for seed in range(5):
            tuner.run(*instance, APPROX, seed)

        drawn = {run["agg"] for run in EchoAlg.RUNS}
        assert drawn <= set(values)
        assert len(drawn) > 1

    def test_exhaustive_compose_tunes_categorical_end_to_end(self, instance):
        from dpstats.dp_tuning.selection_costs.linear_regression import (
            laplace_clipped_sum_of_squared_errors,
        )

        tuner = DPExhaustiveCompose(
            alg_class=list_built_in_algorithms(LinearRegression)["BoostedAdaSSP"],
            alg_hyperparam_grid={
                "learning_rate_schedule": ["constant", "inverse_sqrt"],
                "num_rounds": [3],
            },
            selection_cost=laplace_clipped_sum_of_squared_errors,
        )
        output = tuner.run(
            *instance, budget=ApproxDPBudget(eps=4.0, delta=1e-5), seed=0
        )
        assert output.beta.shape == (2,)


class TestSelectionOverFailedCandidates:
    """A candidate that fails never wins, and only an all-failed pool fails.

    A selection cost is extended-real: a NaN carries no order and scores as
    +inf, and a candidate that itself releases bottom (``a = -1``) scores +inf
    without being scored at all.
    """

    FAILED = {"nan-cost": 1, "no-release": -1}

    @staticmethod
    def _cost(a: int) -> float:
        return float("nan") if a == 1 else float(a)

    @pytest.mark.parametrize("failure", list(FAILED))
    @pytest.mark.parametrize("tuner", list(TUNERS))
    def test_a_failed_candidate_never_wins(self, instance, tuner, failure):
        failed = self.FAILED[failure]
        calls: list[dict[str, Any]] = []
        tuned = TUNERS[tuner](
            alg_hyperparam_grid={"a": [failed, 2, 3]},
            selection_cost=_recording_cost(calls, self._cost),
        )

        released = tuned.run(*instance, APPROX, 0)

        trained = [run["a"] for run in EchoAlg.RUNS]
        assert failed in trained  # the failing candidate was drawn ...
        assert released["a"] == min(a for a in trained if a != failed)  # ... and lost
        assert (failed in [call["a"] for call in calls]) == (failure == "nan-cost")

    @pytest.mark.parametrize("tuner", list(TUNERS))
    def test_a_pool_releasing_only_bottom_releases_bottom(self, instance, tuner):
        tuned = TUNERS[tuner](
            alg_hyperparam_grid={"a": [-1]},
            selection_cost=_recording_cost([], self._cost),
        )

        with pytest.raises(NoRelease, match="every evaluated candidate released"):
            tuned.run(*instance, APPROX, 0)

    @pytest.mark.parametrize("cost", [math.inf, math.nan])
    def test_a_pool_scoring_only_infinite_costs_returns_a_selection(
        self, instance, cost
    ):
        """An infinite cost is a valid score, so a trained candidate is released."""
        tuned = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2]},
            selection_cost=_recording_cost([], lambda a: cost),
        )

        released = tuned.run(*instance, APPROX, 0)

        assert released["a"] == 1

    def test_a_selection_cost_releasing_bottom_scores_its_candidate_infinite(
        self, instance
    ):
        """The candidate itself released, so the tuner still has a selection."""

        def bottom_cost(output: Any, **arguments: Any) -> float:
            raise NoRelease("The selection cost releases bottom.")

        tuned = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2]},
            selection_cost=bottom_cost,
        )

        released = tuned.run(*instance, APPROX, 0)

        assert released["a"] == 1

    def test_an_infinite_cost_candidate_wins_over_a_bottom_candidate(self, instance):
        tuner = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [-1, 2]},
            selection_cost=_recording_cost([], lambda a: math.inf),
        )

        released = tuner.run(*instance, APPROX, 0)

        assert released["a"] == 2

    def test_a_negative_infinite_cost_wins_as_the_best_cost(self, instance):
        """A cost of negative infinity is a real best cost, not a failure."""
        tuner = DPExhaustiveCompose(
            alg_class=EchoAlg,
            alg_hyperparam_grid={"a": [1, 2, 3]},
            selection_cost=_recording_cost([], {1: 2.0, 2: -math.inf, 3: 3.0}),
        )

        released = tuner.run(*instance, APPROX, 0)

        assert released["a"] == 2


class RefusingEcho(EchoAlg):
    """An EchoAlg that refuses its regime once ``REFUSE_AFTER`` runs happened."""

    REFUSE_AFTER: ClassVar[int] = 0

    def run(self, metadata, input, budget, seed):
        if len(EchoAlg.RUNS) >= self.REFUSE_AFTER:
            raise InvalidRegime("RefusingEcho serves no such regime.")
        return super().run(metadata, input, budget, seed)


REFUSING_TUNERS: dict[str, Callable[..., Any]] = {
    "exhaustive": lambda **kw: DPExhaustiveCompose(alg_class=RefusingEcho, **kw),
    "liu-talwar": lambda **kw: DPSelectionLiuTalwar(
        alg_class=RefusingEcho, gamma=0.05, **kw
    ),
    "papernot-steinke": lambda **kw: DPSelectionPapernotSteinkePoisson(
        alg_class=RefusingEcho, mu=10.0, **kw
    ),
}


def _refusing_cost(output: Any, **arguments: Any) -> float:
    raise InvalidRegime("The selection cost serves no such regime.")


class TestRefusalAfterSpending:
    """A refusal is a declared limit only before any candidate has run."""

    @pytest.mark.parametrize("tuner", list(REFUSING_TUNERS))
    def test_the_first_candidate_refusing_stays_a_declared_limit(
        self, instance, tuner, monkeypatch
    ):
        monkeypatch.setattr(RefusingEcho, "REFUSE_AFTER", 0)
        tuned = REFUSING_TUNERS[tuner](
            alg_hyperparam_grid={"a": [1, 2]}, selection_cost=_zero_cost
        )

        with pytest.raises(InvalidRegime, match="RefusingEcho serves no such regime"):
            tuned.run(*instance, APPROX, 0)

    @pytest.mark.parametrize("tuner", list(REFUSING_TUNERS))
    def test_a_later_candidate_refusing_is_an_execution_failure(
        self, instance, tuner, monkeypatch
    ):
        monkeypatch.setattr(RefusingEcho, "REFUSE_AFTER", 1)
        tuned = REFUSING_TUNERS[tuner](
            alg_hyperparam_grid={"a": [1, 2]}, selection_cost=_zero_cost
        )

        with pytest.raises(RuntimeError, match="RefusingEcho") as raised:
            tuned.run(*instance, APPROX, 0)

        assert len(EchoAlg.RUNS) == 1
        assert not isinstance(raised.value, InvalidRegime)
        assert isinstance(raised.value.__cause__, InvalidRegime)

    @pytest.mark.parametrize("tuner", list(TUNERS))
    def test_a_selection_cost_refusing_after_training_is_an_execution_failure(
        self, instance, tuner
    ):
        tuned = TUNERS[tuner](
            alg_hyperparam_grid={"a": [1, 2]}, selection_cost=_refusing_cost
        )

        with pytest.raises(RuntimeError, match="selection cost") as raised:
            tuned.run(*instance, APPROX, 0)

        assert len(EchoAlg.RUNS) == 1
        assert not isinstance(raised.value, InvalidRegime)
        assert isinstance(raised.value.__cause__, InvalidRegime)


class TestSelectorConsumesDefaultSearchGrid:
    """A DP selector consumes ``default_tuning_grid`` as its search grid."""

    def test_exhaustive_compose_runs_over_default_tuning_grid(self, instance):
        from dpstats.dp_tuning.selection_costs.linear_regression import (
            laplace_clipped_sum_of_squared_errors,
        )

        alg_class = list_built_in_algorithms(LinearRegression)["SSP"]
        tuner = DPExhaustiveCompose(
            alg_class=alg_class,
            alg_hyperparam_grid=alg_class.default_tuning_grid(n_points=3),
            selection_cost=laplace_clipped_sum_of_squared_errors,
        )
        output = tuner.run(
            *instance, budget=ApproxDPBudget(eps=4.0, delta=1e-5), seed=0
        )
        assert output.beta.shape == (2,)


class TestRandomTunerProposals:
    """The random-draw tuners propose from the declared search scale.

    A magnitude-spanning parameter declared on a log scale is drawn
    log-uniformly, so each decade is proposed about equally often. The
    proposal reads no data, so the selection accounting does not depend on it.
    """

    RANDOM = ["liu-talwar", "papernot-steinke"]

    def _proposals(self, instance, tuner, grid, *, runs):
        tuned = TUNERS[tuner](alg_hyperparam_grid=grid, selection_cost=_zero_cost)
        for seed in range(runs):
            tuned.run(*instance, APPROX, seed)
        return EchoAlg.RUNS

    @pytest.mark.parametrize("tuner", RANDOM)
    def test_ranges_follow_the_declared_scale_unless_overridden(self, instance, tuner):
        """``lr`` is declared log-scale and ``reg`` linear; a range may override."""
        declared = self._proposals(
            instance,
            tuner,
            {"lr": {"lower": 1e-3, "upper": 1e3}, "reg": {"lower": 0.0, "upper": 2.0}},
            runs=30,
        )
        lr = np.array([run["lr"] for run in declared])
        reg = np.array([run["reg"] for run in declared])
        assert len(lr) > 200
        assert np.all((1e-3 <= lr) & (lr <= 1e3))
        # log10(lr) is uniform on [-3, 3]: half below 1, mean near 0.
        assert 0.35 < np.mean(lr < 1.0) < 0.65
        assert abs(np.mean(np.log10(lr))) < 0.5
        assert np.all((0.0 <= reg) & (reg <= 2.0))
        assert abs(np.mean(reg) - 1.0) < 0.15

        EchoAlg.RUNS.clear()
        overridden = self._proposals(
            instance,
            tuner,
            {"lr": {"lower": 1e-3, "upper": 1e3, "scale": "linear"}},
            runs=30,
        )
        # Linear draws centre on 500, so almost none fall in the lowest decades.
        assert np.mean([run["lr"] < 1.0 for run in overridden]) < 0.05

    def test_integer_log_scale_draws_round_within_range(self, instance):
        """An integer drawn on a log scale is rounded, and small values are common."""
        drawn = [
            run["a"]
            for run in self._proposals(
                instance,
                "papernot-steinke",
                {"a": {"lower": 1, "upper": 1000, "scale": "log"}},
                runs=30,
            )
        ]

        assert all(isinstance(value, int) and 1 <= value <= 1000 for value in drawn)
        # Uniform integers would put only about 1% at or below 10.
        assert np.mean([value <= 10 for value in drawn]) > 0.2

    def test_numeric_lists_are_discrete_candidates(self, instance):
        """A numeric list has the same discrete meaning for every DP tuner."""
        drawn = {
            run["lr"]
            for run in self._proposals(
                instance, "papernot-steinke", {"lr": [1.0, 10.0, 100.0]}, runs=5
            )
        }

        assert drawn == {1.0, 10.0, 100.0}

    def test_log_scale_requires_positive_bounds(self):
        with pytest.raises(ValueError, match="positive endpoints"):
            TUNERS["liu-talwar"](
                alg_hyperparam_grid={
                    "lr": {"lower": 0.0, "upper": 10.0, "scale": "log"}
                },
                selection_cost=_zero_cost,
            )

    @pytest.mark.parametrize("tuner", RANDOM)
    def test_random_selectors_reject_empty_candidate_lists(self, tuner):
        with pytest.raises(ValueError, match="non-empty discrete candidate list"):
            TUNERS[tuner](alg_hyperparam_grid={"a": []}, selection_cost=_zero_cost)

    @pytest.mark.parametrize("tuner", RANDOM)
    def test_ranges_are_normalized_to_their_resolved_scale(self, tuner):
        declared = TUNERS[tuner](
            alg_hyperparam_grid={
                "lr": {"lower": 1e-3, "upper": 1e3},
                "reg": {"lower": 0.0, "upper": 2.0},
            },
            selection_cost=_zero_cost,
        )
        overridden = TUNERS[tuner](
            alg_hyperparam_grid={
                "lr": {"lower": 1e-3, "upper": 1e3, "scale": "linear"},
                "reg": {"lower": 1e-3, "upper": 2.0, "scale": "log"},
            },
            selection_cost=_zero_cost,
        )

        assert declared.alg_hyperparam_grid == {
            "lr": {"lower": 1e-3, "upper": 1e3, "scale": "log"},
            "reg": {"lower": 0.0, "upper": 2.0, "scale": "linear"},
        }
        assert overridden.alg_hyperparam_grid["lr"]["scale"] == "linear"
        assert overridden.alg_hyperparam_grid["reg"]["scale"] == "log"
