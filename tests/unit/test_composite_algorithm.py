"""Unit tests for the public algorithm composition surface."""

from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.algorithm import PrivacyInterpretation

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import PureDPBudget
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.algorithm import CompositeAlgorithm
from dpstats.exceptions import NotApplicable
from dpstats.algorithm import Transformation
from dpstats.task import Task
from tests.registry_components import recording_transformation, spy_algorithm


@pytest.fixture
def second_task(simple_task):
    """Build a second task sharing the first task's Input and Output."""
    _, Metadata, Input, Output = simple_task

    class SecondMetadata(Metadata):
        """Its own Metadata class, since a Task binds the class it is given."""

    return Task(
        id="second_test_task",
        name="Second Test Task",
        description="A second task for adjacency tests",
        metadata_cls=SecondMetadata,
        input_cls=Input,
        output_cls=Output,
    )


class TestCompositeConstruction:
    """Construction validates the chain before anything executes."""

    def test_empty_transformation_chain_is_rejected(self, simple_task):
        task, _, _, Output = simple_task
        algorithm = spy_algorithm(task, Output)()

        with pytest.raises(ValueError, match="at least one transformation"):
            CompositeAlgorithm(transformations=(), algorithm=algorithm)

    def test_adjacent_task_mismatch_is_rejected(self, simple_task, second_task):
        task, _, _, Output = simple_task
        first = recording_transformation(task, second_task, name="first")()
        second = recording_transformation(task, task, name="second")()
        algorithm = spy_algorithm(task, Output)()

        with pytest.raises(ValueError, match="prepares"):
            CompositeAlgorithm(transformations=(first, second), algorithm=algorithm)

    def test_algorithm_task_mismatch_is_rejected(self, simple_task, second_task):
        task, _, _, Output = simple_task
        transformation = recording_transformation(task, second_task)()
        algorithm = spy_algorithm(task, Output)()

        with pytest.raises(ValueError, match="native to"):
            CompositeAlgorithm(transformations=(transformation,), algorithm=algorithm)

    def test_task_follows_the_first_transformation(self, simple_task, second_task):
        task, _, _, Output = simple_task
        transformation = recording_transformation(second_task, task)()
        algorithm = spy_algorithm(task, Output)()

        composition = CompositeAlgorithm(
            transformations=(transformation,), algorithm=algorithm
        )

        assert composition.task is second_task

    def test_copies_answer_with_the_same_task(self, simple_task, second_task):
        task, _, _, Output = simple_task
        transformation = recording_transformation(second_task, task)()
        algorithm = spy_algorithm(task, Output)()
        composition = CompositeAlgorithm(
            transformations=(transformation,), algorithm=algorithm
        )

        for copied in (composition.model_copy(), composition.model_copy(deep=True)):
            assert copied.task is second_task


class TestCompositeExecution:
    """Execution prepares forward, runs once, and restores in reverse."""

    def test_process_and_restore_expose_the_child_call(self, simple_task):
        task, Metadata, Input, Output = simple_task
        trace: list[str] = []
        outer = recording_transformation(task, task, trace=trace, name="outer")()
        inner = recording_transformation(task, task, trace=trace, name="inner")()
        composition = CompositeAlgorithm(
            transformations=(outer, inner),
            algorithm=spy_algorithm(task, Output)(),
        )
        metadata = Metadata(n_samples=4, dim=3)
        input = Input(X=np.zeros((4, 3)))

        prepared = composition.process(
            metadata,
            input,
            budget=PureDPBudget(eps=1.0),
            seed=3,
        )

        prepared_metadata, prepared_input, budget, seed, restorations = prepared
        assert prepared_metadata is metadata
        assert prepared_input is input
        assert budget == PureDPBudget(eps=1.0)
        assert seed == 5
        assert trace == ["prepare:outer", "prepare:inner"]

        output = composition.restore(Output(result=np.zeros(3)), restorations)

        assert isinstance(output, Output)
        assert trace == [
            "prepare:outer",
            "prepare:inner",
            "restore:restoration:inner",
            "restore:restoration:outer",
        ]

    def test_algorithm_receives_the_remaining_budget(self, simple_task):
        task, Metadata, Input, Output = simple_task
        record: dict[str, Any] = {}
        transformation = recording_transformation(task, task, budget_factor=0.25)()
        algorithm = spy_algorithm(task, Output, record)()
        composition = CompositeAlgorithm(
            transformations=(transformation,), algorithm=algorithm
        )

        composition.run(
            Metadata(n_samples=4, dim=3),
            Input(X=np.zeros((4, 3))),
            budget=PureDPBudget(eps=2.0),
            seed=7,
        )

        assert record["budget"] == PureDPBudget(eps=0.5)
        assert record["seed"] == 8

    def test_invalid_transformation_output_stops_before_the_child(self, simple_task):
        task, Metadata, Input, Output = simple_task
        called = False

        class InvalidTransformation(Transformation):
            OUTER_TASK: ClassVar[Task] = task
            INNER_TASK: ClassVar[Task] = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            def prepare(self, metadata, input, budget, seed):
                return Metadata(n_samples=1, dim=3), input, budget, seed, None

            def restore(self, output, restoration):
                return output

        class Child(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            def run(self, metadata, input, budget, seed):
                nonlocal called
                called = True
                return Output(result=np.zeros(3))

        composition = CompositeAlgorithm(
            transformations=(InvalidTransformation(),), algorithm=Child()
        )
        with pytest.raises(ValueError):
            composition.run(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )
        assert not called

    def test_configuration_carries_only_declared_fields(self, simple_task):
        """A dump holds each stage's fields, never its task contract or state."""
        task, _, _, Output = simple_task
        transformation_cls = recording_transformation(task, task, name="tagged")
        algorithm = spy_algorithm(task, Output)()
        composition = CompositeAlgorithm(
            transformations=(transformation_cls(tag="configured"),), algorithm=algorithm
        )

        blob = composition.model_dump(mode="python")

        assert blob["transformations"] == ({"tag": "configured"},)
        assert "restoration" not in str(blob)
        assert recording_transformation(task, task)().model_dump(mode="python") == {
            "tag": "P"
        }

    def test_a_non_transformation_path_is_rejected(self, simple_task):
        from dpstats.tasks.linear_regression.algorithms.ols import OLS

        del simple_task
        with pytest.raises(ValidationError, match="outside the Transformation family"):
            CompositeAlgorithm.load_record(
                {
                    "path": "dpstats.algorithm:CompositeAlgorithm",
                    "config": {
                        "transformations": (
                            {
                                "path": "dpstats.budgets.families:PureDPBudget",
                                "config": {"eps": 1.0},
                            },
                        ),
                        "algorithm": OLS().dump_config(),
                    },
                }
            )


class TestNotApplicableStages:
    """A stage that does not apply structurally passes the problem through."""

    def test_a_skipped_stage_spends_nothing_and_restores_nothing(self, simple_task):
        task, Metadata, Input, Output = simple_task
        trace: list[str] = []
        record: dict[str, Any] = {}
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(
                    task, task, trace=trace, name="first", budget_factor=0.5
                )(),
                recording_transformation(
                    task, task, trace=trace, name="skipped", not_applicable="no block"
                )(),
                recording_transformation(
                    task, task, trace=trace, name="last", budget_factor=0.5
                )(),
            ),
            algorithm=spy_algorithm(task, Output, record)(),
        )

        composition.run(
            Metadata(n_samples=4, dim=3),
            Input(X=np.zeros((4, 3))),
            budget=PureDPBudget(eps=4.0),
            seed=3,
        )

        assert record["budget"] == PureDPBudget(eps=1.0)
        assert record["seed"] == 5
        assert trace == [
            "prepare:first",
            "prepare:last",
            "restore:restoration:last",
            "restore:restoration:first",
        ]

    def test_a_cross_task_stage_refuses_instead_of_passing_through(
        self, simple_task, second_task
    ):
        task, Metadata, Input, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(
                    task, second_task, name="transport", not_applicable="no transport"
                )(),
            ),
            algorithm=spy_algorithm(second_task, Output)(),
        )

        with pytest.raises(NotApplicable, match="no transport"):
            composition.run(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

    def test_a_bare_transformation_refuses_the_cell_as_a_declared_regime_limit(
        self, simple_task
    ):
        """Outside a composition nothing can pass the stage through."""
        task, Metadata, Input, _ = simple_task
        transformation = recording_transformation(
            task, task, not_applicable="no block"
        )()

        with pytest.raises(InvalidRegime, match="no block"):
            transformation.prepare(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                PureDPBudget(eps=1.0),
                0,
            )


def _refusing_algorithm(task: Task, error: type[InvalidRegime]) -> BaseAlgorithm:
    """Build a core that refuses every budget it is offered."""

    class RefusingAlgorithm(BaseAlgorithm):
        TASK = task

        def run(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
            raise error("the core refuses this budget")

    return RefusingAlgorithm()


def _refusing_stage(task: Task) -> Transformation:
    """Build a same-task stage that refuses every problem as a regime limit."""

    class RefusingStage(Transformation):
        OUTER_TASK: ClassVar[Task] = task
        INNER_TASK: ClassVar[Task] = task

        def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
            raise InvalidRegime("the stage refuses this budget")

        def restore(self, output: Any, restoration: Any) -> Any:
            return output

    return RefusingStage()


class TestRefusalAfterSpending:
    """A refusal is a declared limit only while the offered budget is intact."""

    @pytest.mark.parametrize("error", [InvalidRegime, NotApplicable])
    def test_a_core_refusing_a_reduced_budget_is_the_compositions_bottom(
        self, simple_task, error
    ):
        task, Metadata, Input, _ = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(task, task, budget_factor=0.5)(),
            ),
            algorithm=_refusing_algorithm(task, error),
        )

        with pytest.raises(NoRelease, match="RefusingAlgorithm") as raised:
            composition.run(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

        assert not isinstance(raised.value, InvalidRegime)
        assert isinstance(raised.value.__cause__, error)

    def test_a_core_refusing_the_offered_budget_stays_a_declared_limit(
        self, simple_task
    ):
        task, Metadata, Input, _ = simple_task
        composition = CompositeAlgorithm(
            transformations=(recording_transformation(task, task)(),),
            algorithm=_refusing_algorithm(task, InvalidRegime),
        )

        with pytest.raises(InvalidRegime, match="the core refuses this budget"):
            composition.run(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

    def test_public_refusal_after_standardization_is_independent_of_input(
        self, fake_randomness
    ):
        from dpstats.domains import BoxDomain
        from dpstats.tasks.mean_estimation import (
            MeanEstimationInput,
            MeanEstimationMetadata,
        )
        from dpstats.tasks.mean_estimation.algorithms import (
            GaussianMechanism,
        )
        from dpstats.tasks.mean_estimation.transformations import (
            Standardize,
        )

        metadata = MeanEstimationMetadata(
            n_samples=2, domain=BoxDomain(bounds=np.array([[2.0, 6.0]]))
        )
        budget = PureDPBudget(eps=1.0)
        composition = CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=GaussianMechanism()
        )
        messages = []
        for values, standardized in (
            ([[2.0], [4.0]], [[-1.0], [0.0]]),
            ([[3.0], [6.0]], [[-0.5], [1.0]]),
        ):
            input = MeanEstimationInput(X=np.array(values))
            prepared_metadata, prepared_input, remaining, child_seed, _ = (
                composition.process(metadata, input, budget=budget, seed=0)
            )
            np.testing.assert_allclose(prepared_input.X, standardized)
            assert remaining == budget

            with pytest.raises(InvalidRegime) as child_refusal:
                GaussianMechanism().run(
                    prepared_metadata, prepared_input, budget=remaining, seed=child_seed
                )
            with pytest.raises(InvalidRegime) as raised:
                composition.run(metadata, input, budget=budget, seed=0)
            assert str(raised.value) == str(child_refusal.value)
            messages.append(str(raised.value))

        assert messages[0] == messages[1]
        assert fake_randomness.draws_of("normal") == ()

    def test_a_stage_refusing_a_reduced_budget_is_the_compositions_bottom(
        self, simple_task
    ):
        task, Metadata, Input, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(task, task, budget_factor=0.5)(),
                _refusing_stage(task),
            ),
            algorithm=spy_algorithm(task, Output)(),
        )

        with pytest.raises(NoRelease, match="RefusingStage") as raised:
            composition.process(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

        assert not isinstance(raised.value, InvalidRegime)
        assert isinstance(raised.value.__cause__, InvalidRegime)
        assert "the stage refuses this budget" in str(raised.value)

    def test_a_stage_refusing_the_offered_budget_stays_a_declared_limit(
        self, simple_task
    ):
        task, Metadata, Input, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(task, task)(),
                _refusing_stage(task),
            ),
            algorithm=spy_algorithm(task, Output)(),
        )

        with pytest.raises(InvalidRegime, match="the stage refuses this budget"):
            composition.process(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

    def test_a_cross_task_stage_declining_a_reduced_budget_is_the_compositions_bottom(
        self, simple_task, second_task
    ):
        task, Metadata, Input, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(task, task, budget_factor=0.5)(),
                recording_transformation(
                    task, second_task, name="transport", not_applicable="no transport"
                )(),
            ),
            algorithm=spy_algorithm(second_task, Output)(),
        )

        with pytest.raises(NoRelease, match="RecordingTransformation") as raised:
            composition.process(
                Metadata(n_samples=4, dim=3),
                Input(X=np.zeros((4, 3))),
                budget=PureDPBudget(eps=1.0),
                seed=0,
            )

        assert isinstance(raised.value.__cause__, NotApplicable)


class TestCompositeInterpretation:
    """Configured compositions expose the weakest registered component semantics."""

    def test_generic_composition_uses_the_weakest_component(
        self, simple_task, tmp_path
    ):
        task, _, _, Output = simple_task

        class DiagnosticTransformation(Transformation):
            OUTER_TASK: ClassVar[Task] = task
            INNER_TASK: ClassVar[Task] = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.DIAGNOSTIC

            def prepare(self, metadata, input, budget, seed):
                return metadata, input, budget, seed, None

            def restore(self, output, restoration):
                return output

        diagnostic = CompositeAlgorithm(
            transformations=(DiagnosticTransformation(),),
            algorithm=spy_algorithm(task, Output)(),
        )
        assert diagnostic.privacy_interpretation is PrivacyInterpretation.DIAGNOSTIC

    def test_dedicated_algorithm_may_declare_a_stronger_joint_contract(
        self, simple_task
    ):
        task, _, _, Output = simple_task

        class DedicatedComposition(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            def run(self, metadata, input, budget, seed):
                return Output(result=np.zeros(3))

        assert (
            DedicatedComposition().privacy_interpretation
            is PrivacyInterpretation.STANDARD
        )

    def test_current_diagnostic_and_nonprivate_components_fold_conservatively(self):
        from dpstats.tasks.linear_regression.algorithms.ols import OLS
        from dpstats.tasks.linear_regression.algorithms.ssp import (
            SSP,
        )
        from dpstats.tasks.linear_regression.transformations.domain import (
            DataBounds,
        )
        from dpstats.tasks.single_coefficient_estimation.transformations.fwl import (
            LargestOneHotFWLDemean,
        )
        from dpstats.tasks.single_coefficient_estimation.transformations.linear_regression import (
            ToLinearRegression,
        )

        assert (
            CompositeAlgorithm(
                transformations=(DataBounds(),), algorithm=SSP()
            ).privacy_interpretation
            == "diagnostic"
        )
        assert (
            CompositeAlgorithm(
                transformations=(LargestOneHotFWLDemean(), ToLinearRegression()),
                algorithm=OLS(),
            ).privacy_interpretation
            == "nonprivate"
        )

    def test_wrapped_user_algorithm_sets_the_class_of_its_transport_chain(self):
        from dpstats.dp_tuning import DPSelectionLiuTalwar
        from dpstats.dp_tuning.selection_costs.linear_regression import (
            laplace_clipped_sum_of_squared_errors,
        )
        from dpstats.parameters import parameter_field, search_values
        from dpstats.tasks import LinearRegression
        from dpstats.tasks.linear_regression.transformations import (
            ApproxBounds,
            Standardize,
        )
        from dpstats.tasks.single_coefficient_estimation.transformations import (
            ToLinearRegression,
        )

        class WrappedUserScript(BaseAlgorithm):
            TASK = LinearRegression
            ridge: float = parameter_field(
                default=1.0,
                ge=0.0,
                search=search_values(0.1, 1.0),
                description="Ridge term.",
            )

            def run(self, metadata, input, budget, seed):
                raise NotImplementedError

        class DiagnosticUserScript(WrappedUserScript):
            PRIVACY_INTERPRETATION = PrivacyInterpretation.DIAGNOSTIC

        def chain(algorithm):
            return CompositeAlgorithm(
                transformations=(ToLinearRegression(), ApproxBounds(), Standardize()),
                algorithm=algorithm,
            ).privacy_interpretation

        def tuned(alg_class):
            return DPSelectionLiuTalwar(
                alg_class=alg_class,
                alg_hyperparam_grid=alg_class.default_tuning_grid(),
                selection_cost=laplace_clipped_sum_of_squared_errors,
                gamma=0.2,
            )

        assert chain(WrappedUserScript()) is PrivacyInterpretation.STANDARD
        assert chain(tuned(WrappedUserScript)) is PrivacyInterpretation.STANDARD
        assert chain(DiagnosticUserScript()) is PrivacyInterpretation.DIAGNOSTIC
        assert chain(tuned(DiagnosticUserScript)) is PrivacyInterpretation.DIAGNOSTIC

    def test_composition_has_no_interpretation_override(self, simple_task):
        task, _, _, Output = simple_task
        transformation = recording_transformation(task, task)()
        algorithm = spy_algorithm(task, Output)()

        with pytest.raises(ValidationError, match="privacy_interpretation"):
            CompositeAlgorithm(
                transformations=(transformation,),
                algorithm=algorithm,
                privacy_interpretation="standard",  # pyright: ignore[reportCallIssue]
            )


class TestCompositePrintedName:
    """An unlabelled composition prints as its chain and its child."""

    def test_the_label_composes_from_the_chain_and_the_child(self, simple_task):
        task, _, _, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(
                recording_transformation(task, task, name="outer")(),
                recording_transformation(task, task, name="inner")(),
            ),
            algorithm=spy_algorithm(task, Output)(label="AdaSSP"),
        )

        assert (
            str(composition)
            == "RecordingTransformation + RecordingTransformation + AdaSSP"
        )

    def test_an_authored_name_wins_over_the_chain(self, simple_task):
        task, _, _, Output = simple_task
        composition = CompositeAlgorithm(
            transformations=(recording_transformation(task, task)(),),
            algorithm=spy_algorithm(task, Output)(),
            label="Bounded baseline",
        )

        assert str(composition) == "Bounded baseline"


class TestConstructionFlattening:
    """A nested composition normalizes away, leaving one canonical chain."""

    def test_nested_composition_flattens_to_one_chain(self, simple_task):
        task, _, _, Output = simple_task
        outer = recording_transformation(task, task, name="outer")()
        inner = recording_transformation(task, task, name="inner")()
        algorithm = spy_algorithm(task, Output)()

        nested = CompositeAlgorithm(
            transformations=(outer,),
            algorithm=CompositeAlgorithm(transformations=(inner,), algorithm=algorithm),
        )

        assert nested.transformations == (outer, inner)
        assert nested.algorithm is algorithm

    def test_nested_and_flat_declarations_share_one_serialized_identity(
        self, simple_task
    ):
        task, _, _, Output = simple_task
        transformation_cls = recording_transformation(task, task, name="stage")
        algorithm = spy_algorithm(task, Output)()

        nested = CompositeAlgorithm(
            transformations=(transformation_cls(tag="a"),),
            algorithm=CompositeAlgorithm(
                transformations=(transformation_cls(tag="b"),), algorithm=algorithm
            ),
        )
        flat = CompositeAlgorithm(
            transformations=(transformation_cls(tag="a"), transformation_cls(tag="b")),
            algorithm=algorithm,
        )

        assert nested.model_dump(mode="python") == flat.model_dump(mode="python")
