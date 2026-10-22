"""Values identified by content hash by that identity, wherever they are keyed."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.dp_tuning import DPExhaustiveCompose
from dpstats.instance import Instance
from dpstats.coordinates import CellCoordinate
from dpstats.metric import Metric
from dpstats.parameters import parameter_field
from dpstats.transport.configuration import identity_payload
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
)


def _selection_cost(output: SimpleOutput, **kwargs: Any) -> float:
    """Score a candidate by the estimate it released."""
    return float(output.result[0])


def _instance(name: str) -> Instance:
    return Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(name.encode()).hexdigest(),
        metric_arguments={},
        label=name,
    )


def _coordinate(alg_seed: int) -> CellCoordinate:
    return CellCoordinate(
        instance=_instance("i1"),
        algorithm=SimpleAlgorithm(),
        budget=PureDPBudget(eps=1.0),
        alg_seed=alg_seed,
    )


def test_equal_values_hash_equal_through_their_identity() -> None:
    for first, second in (
        (SimpleAlgorithm(scale=2.0), SimpleAlgorithm(scale=2.0)),
        (_instance("i1"), _instance("i1")),
        (_coordinate(0), _coordinate(0)),
    ):
        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    assert hash(_instance("i1")) != hash(_instance("i2"))


def test_a_tuner_carrying_a_grid_still_keys_a_frame() -> None:
    """A mapping among a value's fields leaves no field-wise hash to key by."""
    tuner = DPExhaustiveCompose(
        alg_class=SimpleAlgorithm,
        alg_hyperparam_grid={"scale": [1.0, 2.0]},
        selection_cost=_selection_cost,
        label="Tuned",
    )
    plain = SimpleAlgorithm(label="Plain")
    frame = pd.DataFrame(
        [
            {"algorithm": tuner, "value": 1.0},
            {"algorithm": tuner, "value": 3.0},
            {"algorithm": plain, "value": 5.0},
        ]
    )

    grouped = frame.groupby("algorithm", sort=False)["value"].median()

    assert dict(zip([str(key) for key in grouped.index], grouped)) == {
        "Tuned": 2.0,
        "Plain": 5.0,
    }


class _UndocumentedScale(BaseAlgorithm):
    """Declare one scale field carrying no documentation metadata."""

    TASK = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    scale: float = parameter_field(default=1.0, description="Release multiplier.")

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        return SimpleOutput(result=input.X.mean(axis=0) * self.scale)


class _DocumentedScale(BaseAlgorithm):
    """Declare the same scale field with a math symbol for generated tables."""

    TASK = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    scale: float = parameter_field(
        default=1.0, description="Release multiplier.", symbol="s"
    )

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        return SimpleOutput(result=input.X.mean(axis=0) * self.scale)


def test_a_declared_symbol_leaves_the_identity_payload_untouched() -> None:
    """A field's math symbol documents it; identity covers field values only."""
    documented = identity_payload(_DocumentedScale(scale=2.0), exclude={"label"})
    undocumented = identity_payload(_UndocumentedScale(scale=2.0), exclude={"label"})

    assert documented["config"] == {"scale": 2.0}
    assert documented["config"] == undocumented["config"]


def test_a_declared_formula_leaves_a_metric_measuring_the_same_column() -> None:
    """A formula documents a Metric; its name and values are what it reports."""
    documented = Metric(
        task=SIMPLE_TASK,
        name="first_coordinate",
        calculation=_selection_cost,
        formula="y_1",
    )
    undocumented = Metric(
        task=SIMPLE_TASK, name="first_coordinate", calculation=_selection_cost
    )
    output = SimpleOutput(result=np.array([7.0]))

    assert str(documented) == str(undocumented)
    assert documented.required_names == undocumented.required_names
    assert documented.calculate(output) == undocumented.calculate(output)
