"""External configured components round-trip through the public base seam."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    Transformation,
)
from dpstats.domains import EuclideanSpace
from dpstats.dp_tuning.base import DPHyperparameterTuned
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.task import TaskMetadata
from dpstats.tasks.m_estimation import MEstimationLoss, MEstimationMetadata
from tests.fixtures.external_configured_components import (
    ExternalAlgorithm,
    ExternalExhaustiveTuner,
    ExternalMEstimationLoss,
    ExternalTransformation,
    external_selection_cost,
)


@pytest.mark.parametrize(
    ("family", "value", "concrete"),
    [
        (BaseAlgorithm, ExternalAlgorithm(scale=2.0), ExternalAlgorithm),
        (
            Transformation,
            ExternalTransformation(label="configured"),
            ExternalTransformation,
        ),
        (
            DPHyperparameterTuned,
            ExternalExhaustiveTuner(
                alg_class=ExternalAlgorithm,
                alg_hyperparam_grid={"scale": [1.0, 2.0]},
                selection_cost=external_selection_cost,
                selection_cost_args={"offset": 1.0},
            ),
            ExternalExhaustiveTuner,
        ),
    ],
)
def test_external_components_round_trip_through_their_declared_family(
    family: type[BaseAlgorithm | Transformation],
    value: BaseAlgorithm | Transformation,
    concrete: type[BaseAlgorithm | Transformation],
) -> None:
    persisted = decode_json_document(
        canonical_json_bytes(value.dump_config().model_dump())
    )

    restored = family.load_record(persisted)

    assert type(restored) is concrete
    assert restored == value
    if isinstance(value, DPHyperparameterTuned):
        config = persisted["config"]
        assert isinstance(config, dict)
        assert config["selection_cost"] == (
            "tests.fixtures.external_configured_components:external_selection_cost"
        )


def test_external_loss_round_trips_inside_task_metadata() -> None:
    value = MEstimationMetadata(
        parameter_domain=EuclideanSpace(dim=2),
        n_samples=3,
        loss=ExternalMEstimationLoss(weight=2.0),
        data_domain=EuclideanSpace(dim=2),
    )

    persisted = decode_json_document(
        canonical_json_bytes(value.dump_config().model_dump())
    )
    config = persisted["config"]
    assert isinstance(config, dict)
    assert config["loss"] == {
        "path": (
            "tests.fixtures.external_configured_components:" "ExternalMEstimationLoss"
        ),
        "config": {"weight": 2.0},
    }

    restored = TaskMetadata.load_record(persisted)

    assert isinstance(restored, MEstimationMetadata)
    assert isinstance(restored.loss, MEstimationLoss)
    assert type(restored.loss) is ExternalMEstimationLoss
    z = np.array([[3.0, 4.0]])
    theta = np.array([1.0, 0.5])
    np.testing.assert_allclose(restored.loss(z, theta), [25.0])
    np.testing.assert_allclose(restored.loss.gradient(z, theta), [[30.0, 40.0]])


def test_m_estimation_metadata_rejects_a_bare_callable_loss() -> None:
    with pytest.raises(ValidationError, match="loss"):
        MEstimationMetadata.model_validate(
            {
                "parameter_domain": EuclideanSpace(dim=1),
                "n_samples": 1,
                "loss": lambda data, theta: 0.0,
                "data_domain": EuclideanSpace(dim=1),
            }
        )


def test_composition_restores_external_transformation_and_tuning_wrapper() -> None:
    composition = CompositeAlgorithm(
        transformations=(ExternalTransformation(label="nested"),),
        algorithm=ExternalExhaustiveTuner(
            alg_class=ExternalAlgorithm,
            alg_hyperparam_grid={"scale": [1.0, 2.0]},
            selection_cost=external_selection_cost,
        ),
    )
    persisted = decode_json_document(
        canonical_json_bytes(composition.dump_config().model_dump())
    )

    restored = CompositeAlgorithm.load_record(persisted)

    assert type(restored.transformations[0]) is ExternalTransformation
    assert type(restored.algorithm) is ExternalExhaustiveTuner
    assert restored == composition
