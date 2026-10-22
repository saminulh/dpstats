"""Typed-budget accounting tests for private hyperparameter selectors."""

from __future__ import annotations

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime

from dpstats.domains import EuclideanSpace

# pyright: reportPrivateUsage=none

import math
from typing import Any

import numpy as np
import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.dp_tuning.base import declares_privacy_interpretation
from dpstats.dp_tuning import (
    DPExhaustiveCompose,
    DPSelectionLiuTalwar,
    DPSelectionPapernotSteinkePoisson,
    DPSelectionPapernotSteinkeTNB,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
    assert_budget_close,
)


def _instance() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    rng = np.random.default_rng(0)
    return (
        LinearRegressionMetadata(
            n_samples=100,
            design_domain=EuclideanSpace(dim=2),
            response_domain=EuclideanSpace(dim=1),
        ),
        LinearRegressionInput(X=rng.normal(size=(100, 2)), y=rng.normal(size=100)),
    )


def _zero_selection_cost(output: Any, **kwargs: Any) -> float:
    return 0.0


def _recording_components(
    observed: list[tuple[str, DPBudget, int]], *, enter_zcdp: bool = False
) -> tuple[type[BaseAlgorithm], Any]:
    """Build a candidate and a cost that record each (part, budget, seed) they see."""

    class RecordingAlgorithm(BaseAlgorithm):
        TASK = LinearRegression

        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.STANDARD

        value: int = 0

        def run(
            self,
            metadata: Any,
            input: Any,
            budget: DPBudget,
            seed: int,
        ) -> dict[str, int]:
            observed.append(("algorithm", budget, seed))
            if enter_zcdp:
                accounting.spend_as_zcdp(budget)
            return {"value": self.value}

    def selection_cost(
        output: Any, *, budget: DPBudget, seed: int, **kwargs: Any
    ) -> float:
        observed.append(("selection_cost", budget, seed))
        if enter_zcdp:
            accounting.spend_as_zcdp(budget)
        return float(output["value"])

    return RecordingAlgorithm, selection_cost


class _StandardAlgorithm(BaseAlgorithm):
    TASK = LinearRegression

    def run(self, metadata: Any, input: Any, budget: DPBudget, seed: int) -> Any:
        raise AssertionError("classification does not run the tuned algorithm")


@declares_privacy_interpretation(PrivacyInterpretation.STANDARD)
def _standard_selection_cost(output: Any, **kwargs: Any) -> float:
    return 0.0


@declares_privacy_interpretation(PrivacyInterpretation.QUALIFIED)
def _qualified_selection_cost(output: Any, **kwargs: Any) -> float:
    return 0.0


def test_a_tuner_reads_the_weakest_of_its_candidate_and_its_selection_cost() -> None:
    def tuner_class(selection_cost: Any) -> PrivacyInterpretation:
        return DPExhaustiveCompose(
            alg_class=_StandardAlgorithm,
            alg_hyperparam_grid={},
            selection_cost=selection_cost,
        ).privacy_interpretation

    assert tuner_class(_standard_selection_cost) is PrivacyInterpretation.STANDARD
    assert tuner_class(_qualified_selection_cost) is PrivacyInterpretation.QUALIFIED
    assert tuner_class(_zero_selection_cost) is PrivacyInterpretation.DIAGNOSTIC


def test_a_tuner_is_no_stronger_than_its_candidate() -> None:
    class QualifiedAlgorithm(_StandardAlgorithm):
        PRIVACY_INTERPRETATION = PrivacyInterpretation.INFORMAL

    tuner = DPSelectionLiuTalwar(
        alg_class=QualifiedAlgorithm,
        alg_hyperparam_grid={},
        selection_cost=_standard_selection_cost,
    )

    assert tuner.privacy_interpretation is PrivacyInterpretation.INFORMAL


def test_tuner_refuses_a_nonprivate_candidate() -> None:
    from dpstats.tasks.linear_regression.algorithms.ols import OLS

    nonprivate_tuner = DPExhaustiveCompose(
        alg_class=OLS,
        alg_hyperparam_grid={},
        selection_cost=_zero_selection_cost,
    )
    with pytest.raises(TypeError, match="cannot tune nonprivate candidate OLS"):
        nonprivate_tuner.build_candidate({})


def test_a_candidate_trains_and_scores_on_exactly_the_full_input() -> None:
    observed: dict[str, tuple[Any, LinearRegressionInput]] = {}

    class InnerAlgorithm(BaseAlgorithm):
        TASK = LinearRegression

        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.STANDARD

        def run(self, metadata: Any, input: Any, budget: DPBudget, seed: int) -> Any:
            observed["train"] = (metadata, input)
            return {"selected": True}

    def selection_cost(
        output: Any,
        *,
        evaluation_metadata: Any,
        evaluation_input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> float:
        observed["evaluation"] = (evaluation_metadata, evaluation_input)
        return 0.0

    tuner = DPExhaustiveCompose(
        alg_class=InnerAlgorithm,
        alg_hyperparam_grid={},
        selection_cost=selection_cost,
    )
    realized_input = LinearRegressionInput(
        X=np.arange(14, dtype=float).reshape(7, 2),
        y=np.arange(7, dtype=float),
    )
    public_metadata, _ = _instance()

    output = tuner.run(
        public_metadata,
        realized_input,
        budget=PureDPBudget(eps=1.0),
        seed=2,
    )

    assert output == {"selected": True}
    for kind in ("train", "evaluation"):
        seen_metadata, seen_input = observed[kind]
        assert seen_metadata is public_metadata
        assert seen_input is realized_input


@pytest.mark.parametrize(
    "budget, expected",
    [
        (PureDPBudget(eps=3.0), PureDPBudget(eps=1.5)),
        (ApproxDPBudget(eps=3.0, delta=1e-5), ApproxDPBudget(eps=1.5, delta=5e-6)),
        (ZCDPBudget(rho=3.0), ZCDPBudget(rho=1.5)),
        (GDPBudget(mu=3.0), GDPBudget(mu=3.0 / math.sqrt(2.0))),
    ],
)
def test_exhaustive_divides_each_family_by_its_composition_law(
    budget: DPBudget,
    expected: DPBudget,
) -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPExhaustiveCompose(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1, 2]},
        selection_cost=selection_cost,
    )
    metadata, input_data = _instance()

    tuner.run(metadata, input_data, budget, seed=1)

    assert [kind for kind, _, _ in observed] == [
        "algorithm",
        "selection_cost",
        "algorithm",
        "selection_cost",
    ]
    training, scoring = expected.split((0.90, 0.10))
    for kind, seen, _ in observed:
        assert_budget_close(seen, training if kind == "algorithm" else scoring)
    # Each candidate's training and its cost draw independent noise streams.
    seeds = [seed for _, _, seed in observed]
    assert len(set(seeds)) == len(seeds)


def test_liu_talwar_spends_zcdp_as_pure_before_its_pure_theorem(
    accounting_spy: AccountingSpy,
) -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPSelectionLiuTalwar(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
        gamma=1.0,
    )
    metadata, input_data = _instance()
    offered = ZCDPBudget(rho=2.0)

    tuner.run(metadata, input_data, offered, seed=2)

    assert accounting_spy.calls == (AccountingCall("spend_as_pure", offered),)
    assert [kind for kind, _, _ in observed] == ["algorithm", "selection_cost"]
    assert isinstance(observed[0][1], PureDPBudget)
    assert isinstance(observed[1][1], PureDPBudget)
    admitted = accounting.spend_as_pure(offered).eps
    assert observed[0][1].eps + observed[1][1].eps == pytest.approx(admitted / 3)


def test_papernot_steinke_hands_charged_zcdp_budget_to_both_parts(
    accounting_spy: AccountingSpy,
) -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed, enter_zcdp=True)
    tuner = DPSelectionPapernotSteinkeTNB(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
        eta=1.0,
        gamma=0.9,
    )
    metadata, input_data = _instance()
    offered = ApproxDPBudget(eps=3.0, delta=1e-5)

    tuner.run(metadata, input_data, offered, seed=3)

    assert observed
    (algorithm_budget, cost_budget) = (observed[0][1], observed[1][1])
    assert isinstance(algorithm_budget, ZCDPBudget)
    assert isinstance(cost_budget, ZCDPBudget)
    assert cost_budget.rho / (algorithm_budget.rho + cost_budget.rho) == (
        pytest.approx(0.10)
    )
    assert [call.budget for call in accounting_spy.calls] == [
        algorithm_budget,
        cost_budget,
    ]
    assert all(call.entry_point == "spend_as_zcdp" for call in accounting_spy.calls)


@pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
@pytest.mark.parametrize(
    "selector", [DPSelectionPapernotSteinkePoisson, DPSelectionPapernotSteinkeTNB]
)
def test_papernot_steinke_refuses_outer_non_approximate_budget(
    selector: type[DPSelectionPapernotSteinkePoisson | DPSelectionPapernotSteinkeTNB],
    budget: DPBudget,
) -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = selector(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
    )
    metadata, input_data = _instance()

    with pytest.raises(InvalidRegime, match="Cannot spend"):
        tuner.run(metadata, input_data, budget, seed=0)

    assert not observed


@pytest.mark.parametrize(
    "budget",
    [NO_PRIVACY, PureDPBudget(eps=math.inf), ApproxDPBudget(eps=1.0, delta=1.0)],
)
def test_papernot_steinke_poisson_refuses_an_unconstrained_offer(
    budget: DPBudget,
) -> None:
    """The corrected Theorem 6 charge has no per-run rho cap, so it refuses."""
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPSelectionPapernotSteinkePoisson(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
    )
    metadata, input_data = _instance()

    with pytest.raises(InvalidRegime, match="unconstrained"):
        tuner.run(metadata, input_data, budget, seed=0)

    assert not observed


@pytest.mark.parametrize(
    "budget", [NO_PRIVACY, ApproxDPBudget(eps=math.inf, delta=1.0)]
)
def test_papernot_steinke_tnb_releases_at_its_theorem_cap_for_every_unconstrained_offer(
    budget: DPBudget,
) -> None:
    """Corollary 4's rho cap is a deliberate bottom, not a crash or a refusal."""
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPSelectionPapernotSteinkeTNB(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
        eta=1.0,
        gamma=0.05,
    )
    metadata, input_data = _instance()

    tuner.run(metadata, input_data, budget, seed=3)

    assert observed
    parts = [seen for _, seen, _ in observed[:2]]
    assert all(isinstance(part, ZCDPBudget) for part in parts)
    assert sum(part.rho for part in parts if isinstance(part, ZCDPBudget)) == (
        pytest.approx(math.log(1.0 / 0.05))
    )


@pytest.mark.parametrize(
    "offered, parameters",
    [
        (PureDPBudget(eps=2.0), ("eps",)),
        (ApproxDPBudget(eps=2.0, delta=1e-5), ("eps", "delta")),
        (ZCDPBudget(rho=2.0), ("rho",)),
    ],
)
def test_the_two_shares_compose_to_the_offered_budget(
    offered: DPBudget, parameters: tuple[str, ...]
) -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPExhaustiveCompose(
        alg_class=alg_class,
        alg_hyperparam_grid={},
        selection_cost=selection_cost,
        scoring_share=0.2,
    )
    metadata, input_data = _instance()

    tuner.evaluate_candidate(
        metadata, input_data, {}, offered, 1, 2, earlier_candidate_ran=False
    )

    (_, training, _), (_, scoring, _) = observed
    for name in parameters:
        assert getattr(training, name) + getattr(scoring, name) == pytest.approx(
            getattr(offered, name)
        )
        assert getattr(scoring, name) == pytest.approx(0.2 * getattr(offered, name))


def test_a_single_row_is_trained_and_scored_without_a_release_of_bottom() -> None:
    observed: list[tuple[str, DPBudget, int]] = []
    alg_class, selection_cost = _recording_components(observed)
    tuner = DPExhaustiveCompose(
        alg_class=alg_class,
        alg_hyperparam_grid={"value": [1]},
        selection_cost=selection_cost,
    )
    metadata, input_data = _instance()
    metadata = metadata.model_copy(update={"n_samples": 1})
    single_row = LinearRegressionInput(X=input_data.X[:1], y=input_data.y[:1])

    assert tuner.run(metadata, single_row, PureDPBudget(eps=1.0), seed=0) == {
        "value": 1
    }


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, float("nan")])
def test_the_scoring_share_lies_strictly_between_zero_and_one(share: float) -> None:
    with pytest.raises(ValueError, match="scoring_share"):
        DPExhaustiveCompose(
            alg_class=_StandardAlgorithm,
            alg_hyperparam_grid={},
            selection_cost=_zero_selection_cost,
            scoring_share=share,
        )


def test_a_candidate_whose_class_diverges_from_its_declared_constant_is_refused() -> (
    None
):
    class DivergentAlgorithm(_StandardAlgorithm):
        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.INFORMAL

    tuner = DPExhaustiveCompose(
        alg_class=DivergentAlgorithm,
        alg_hyperparam_grid={},
        selection_cost=_standard_selection_cost,
    )

    with pytest.raises(TypeError, match="declared class"):
        tuner.build_candidate({})
