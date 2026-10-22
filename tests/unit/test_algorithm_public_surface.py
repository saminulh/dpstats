"""Public seams for authored algorithms and compositions."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
    Transformation,
)
from dpstats.dp_tuning import DPExhaustiveCompose
from dpstats.dp_tuning.selection_costs.linear_regression import (
    laplace_clipped_sum_of_squared_errors,
)
from dpstats.parameters import parameter_field
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.task import Task
from dpstats.tasks import MeanEstimation
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.tasks.linear_regression.algorithms import BoostedAdaSSP
from dpstats.tasks.linear_regression.transformations import Standardize
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ToLinearRegression,
)
from tests.fixtures.external_configured_components import (
    EXTERNAL_TASK,
    ExternalAlgorithm,
    ExternalInput,
    ExternalMetadata,
    ExternalOutput,
    ExternalTransformation,
)


def _tuned_ridge() -> DPExhaustiveCompose:
    return DPExhaustiveCompose(
        alg_class=Ridge,
        alg_hyperparam_grid={"regularization_multiplier": [0.01, 0.1]},
        selection_cost=laplace_clipped_sum_of_squared_errors,
        # Explicit, so the pinned identities below track the identity format
        # rather than the default share.
        scoring_share=0.05,
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        Ridge(),
        CompositeAlgorithm(transformations=(ToLinearRegression(),), algorithm=Ridge()),
    ],
)
def test_live_relabelling_preserves_identity_equality_and_original(
    algorithm: BaseAlgorithm,
) -> None:
    uuid = algorithm.uuid
    record = algorithm.dump_config().model_dump()
    renamed = algorithm.model_copy(update={"label": "local panel"})
    assert renamed.label == "local panel"
    assert algorithm.label is None
    assert renamed.uuid == uuid
    assert renamed == algorithm and hash(renamed) == hash(algorithm)
    assert len({algorithm, renamed}) == 1
    assert algorithm.dump_config().model_dump() == record
    assert renamed.dump_config().config["label"] == "local panel"
    if isinstance(algorithm, CompositeAlgorithm):
        nested = algorithm.model_copy(
            update={
                "algorithm": algorithm.algorithm.model_copy(
                    update={"label": "inner panel"}
                )
            }
        )
        assert nested == algorithm and nested.uuid == uuid


def test_equality_separates_configurations_and_never_reaches_other_types() -> None:
    """Only the presentation label is excused; everything else tells values apart."""
    assert Ridge(regularization_multiplier=0.5) != Ridge()
    assert Ridge(label="a") == Ridge(label="b")
    assert Ridge(regularization_multiplier=0.5, label="a") != Ridge(
        regularization_multiplier=0.25, label="a"
    )
    assert Ridge() != OLS()
    assert Ridge() != CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=Ridge()
    )
    assert Ridge() != "Ridge"
    assert Ridge() != object()


PINNED_IDENTITIES = (
    (OLS(), "a3-5L6C7UXQRNBZIAAW"),
    (Ridge(), "a3-EKJTPP3PRPR4PJOE"),
    (
        CompositeAlgorithm(transformations=(ToLinearRegression(),), algorithm=OLS()),
        "a3-WJLMWDFDD5YK2L6N",
    ),
    (_tuned_ridge(), "a3-EW67OZZVIWIKHDAI"),
    (
        CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=_tuned_ridge()
        ),
        "a3-PBD7BL3YKFJFJLNA",
    ),
)


@pytest.mark.parametrize(("algorithm", "expected"), PINNED_IDENTITIES)
def test_configured_algorithm_identities_remain_pinned(
    algorithm: BaseAlgorithm, expected: str
) -> None:
    """A configuration's identity token is stable for the life of the format."""
    assert algorithm.uuid == expected


def test_a_configured_algorithm_derives_its_identity_once_per_value() -> None:
    """The cached identity never travels onto an updated copy."""
    algorithm = Ridge()
    assert algorithm.uuid is algorithm.uuid

    updated = algorithm.model_copy(update={"regularization_multiplier": 99.0})
    assert updated.uuid != algorithm.uuid


@pytest.mark.parametrize(
    "algorithm", [Ridge(), ExternalAlgorithm(), _tuned_ridge(), BoostedAdaSSP()]
)
def test_core_inspection_keeps_plain_custom_and_compound_algorithms(
    algorithm: BaseAlgorithm,
) -> None:
    assert algorithm.get_core_algorithm() is algorithm


@pytest.mark.parametrize("core", [Ridge(regularization_multiplier=0.5), _tuned_ridge()])
def test_core_inspection_preserves_the_configured_child_and_outer_identity(
    core: BaseAlgorithm,
) -> None:
    composed = CompositeAlgorithm(
        transformations=(ToLinearRegression(),),
        algorithm=CompositeAlgorithm(transformations=(Standardize(),), algorithm=core),
    )
    payload, uuid = composed.model_dump_json(), composed.uuid

    assert composed.get_core_algorithm() is core
    assert composed.get_core_algorithm().get_core_algorithm() is core
    assert composed.task is not core.task
    assert composed.model_dump_json() == payload
    assert composed.uuid == uuid


def test_extension_author_can_declare_and_run_an_algorithm() -> None:
    """The authoring seam needs no import from the library's internals."""

    class OffsetAlgorithm(BaseAlgorithm):
        TASK = EXTERNAL_TASK

        offset: int = parameter_field(
            default=2, ge=0, description="Test offset added to the result."
        )

        def run(
            self,
            metadata: ExternalMetadata,
            input: ExternalInput,
            budget: DPBudget,
            seed: int,
        ) -> ExternalOutput:
            del metadata, input, budget, seed
            return ExternalOutput(estimate=np.array([self.offset]))

    output = OffsetAlgorithm(offset=3).run(
        ExternalMetadata(n_samples=1, dimension=1),
        ExternalInput(X=np.array([[0.0]])),
        PureDPBudget(eps=1.0),
        seed=0,
    )

    assert np.array_equal(output.estimate, np.array([3]))


def test_algorithm_defaults_to_a_standard_privacy_interpretation() -> None:
    """An incomplete author declaration fails before it can be configured."""

    class MissingInterpretation(BaseAlgorithm):
        TASK = EXTERNAL_TASK

        def run(
            self,
            metadata: ExternalMetadata,
            input: ExternalInput,
            budget: DPBudget,
            seed: int,
        ) -> ExternalOutput:
            del metadata, input, budget, seed
            return ExternalOutput(estimate=np.array([0.0]))

    assert (
        MissingInterpretation().privacy_interpretation is PrivacyInterpretation.STANDARD
    )


def test_transformation_defaults_to_a_standard_privacy_interpretation() -> None:
    """A transformation author declares the class only to qualify the release."""

    class MissingInterpretation(Transformation):
        OUTER_TASK: ClassVar[Task] = EXTERNAL_TASK
        INNER_TASK: ClassVar[Task] = EXTERNAL_TASK

        def prepare(
            self,
            metadata: Any,
            input: Any,
            budget: DPBudget,
            seed: int,
        ) -> tuple[Any, Any, DPBudget, int, None]:
            return metadata, input, budget, seed, None

        def restore(self, output: Any, restoration: Any) -> Any:
            del restoration
            return output

    assert (
        MissingInterpretation().privacy_interpretation is PrivacyInterpretation.STANDARD
    )


def test_composite_algorithm_flattens_and_restores_through_public_seam() -> None:
    """Nested compositions have one normalized configured value and outcome."""
    nested = CompositeAlgorithm(
        transformations=(ExternalTransformation(label="outer"),),
        algorithm=CompositeAlgorithm(
            transformations=(ExternalTransformation(label="inner"),),
            algorithm=ExternalAlgorithm(scale=2.0),
        ),
    )

    output = nested.run(
        ExternalMetadata(n_samples=1, dimension=1),
        ExternalInput(X=np.array([[2.0]])),
        PureDPBudget(eps=1.0),
        seed=0,
    )
    assert np.array_equal(output.estimate, np.array([4.0]))
    assert nested.task is EXTERNAL_TASK
    assert nested.transformations == (
        ExternalTransformation(label="outer"),
        ExternalTransformation(label="inner"),
    )
    assert nested.algorithm == ExternalAlgorithm(scale=2.0)
    record = nested.dump_config()
    assert record.path == ("dpstats.algorithm:CompositeAlgorithm")
    assert CompositeAlgorithm.load_record(record) == nested


def test_transformation_rejects_an_input_from_the_wrong_declared_inner_task() -> None:
    """A transformation's declared transport target is checked at its seam."""

    class MisdeclaredTransformation(Transformation):
        OUTER_TASK: ClassVar[Task] = EXTERNAL_TASK
        INNER_TASK: ClassVar[Task] = MeanEstimation

        def prepare(
            self,
            metadata: Any,
            input: Any,
            budget: DPBudget,
            seed: int,
        ) -> tuple[Any, Any, DPBudget, int, None]:
            return metadata, input, budget, seed, None

        def restore(self, output: Any, restoration: Any) -> Any:
            del restoration
            return output

    composition = CompositeAlgorithm(
        transformations=(MisdeclaredTransformation(),), algorithm=SampleMean()
    )
    with pytest.raises(ValueError, match="declared inner task"):
        composition.run(
            ExternalMetadata(n_samples=1, dimension=1),
            ExternalInput(X=np.array([[0.0]])),
            PureDPBudget(eps=1.0),
            seed=0,
        )
