"""The hidden Pydantic base stays small and unsurprising."""

from __future__ import annotations

from typing import Any

import math
from functools import cached_property

import numpy as np
import pytest
from pydantic import ValidationError, computed_field, model_serializer

from dpstats.transport.configuration import PolymorphicModel
from dpstats.algorithm import BaseAlgorithm
from dpstats.domains import BoxDomain, Domain
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.dp_tuning import DPSelectionLiuTalwar
from dpstats.tasks.linear_regression.algorithms.ols import OLS


class ExampleValue(PolymorphicModel):
    count: int
    scale: float = 1.0


class InvalidDefaultValue(PolymorphicModel):
    count: int = "1"  # type: ignore[assignment]


class NestedValue(PolymorphicModel):
    value: ExampleValue


class OtherValue(PolymorphicModel):
    label: str


class NestedCollections(PolymorphicModel):
    direct: ExampleValue
    listed: list[ExampleValue]
    keyed: dict[str, ExampleValue]


class TupleValue(PolymorphicModel):
    counts: tuple[int, ...]
    nested: tuple[ExampleValue, ...]


class UnionValue(PolymorphicModel):
    value: ExampleValue | OtherValue


class LiteralConfigurationFields(PolymorphicModel):
    path: str
    config: dict[str, int]


class RecursiveConfigurationFields(PolymorphicModel):
    path: str
    config: RecursiveConfigurationFields | None = None


class ComputedValue(PolymorphicModel):
    count: int

    @computed_field
    @property
    def doubled(self) -> int:
        return 2 * self.count


class CachedValue(PolymorphicModel):
    count: int

    @cached_property
    def doubled(self) -> int:
        return 2 * self.count


def _selection_cost(*args: Any, **kwargs: Any) -> float:
    """Provide an importable configured callable for a tuner round trip."""
    del args, kwargs
    return 0.0


def test_numeric_fields_do_not_accept_strings_or_bools():
    value = ExampleValue(count=1, scale=2)
    assert value.scale == 2.0

    for invalid in (
        {"count": True},
        {"count": 1, "scale": True},
        {"count": 1, "scale": "2"},
    ):
        with pytest.raises(ValidationError):
            ExampleValue(**invalid)


def test_copies_do_not_carry_values_derived_from_the_original():
    original = CachedValue(count=2)
    assert original.doubled == 4

    assert original.model_copy(update={"count": 5}).doubled == 10
    assert original.model_copy().doubled == 4
    assert original.model_copy(deep=True).doubled == 4


def test_a_configuration_record_is_a_fresh_copy_each_time() -> None:
    """Mutating one returned record never changes what the value dumps next."""
    value = ExampleValue(count=1)

    first = value.dump_config()
    first.config["count"] = 9
    second = value.dump_config()

    assert second.model_dump() == {
        "path": f"{__name__}:ExampleValue",
        "config": {"count": 1, "scale": 1.0},
    }
    updated = value.model_copy(update={"count": 2})
    assert updated.dump_config().config == {"count": 2, "scale": 1.0}


def test_defaults_extras_and_assignment_are_validated():
    with pytest.raises(ValidationError):
        InvalidDefaultValue()
    with pytest.raises(ValidationError):
        ExampleValue(count=1, extra=2)

    value = ExampleValue(count=1)
    with pytest.raises(ValidationError):
        value.count = 2


def test_configuration_round_trip_restores_nested_concrete_types():
    value = NestedValue(value=ExampleValue(count=3, scale=2.5))

    record = value.dump_config()

    assert record.model_dump() == {
        "path": f"{__name__}:NestedValue",
        "config": {
            "value": {
                "path": f"{__name__}:ExampleValue",
                "config": {"count": 3, "scale": 2.5},
            }
        },
    }
    restored = PolymorphicModel.load_record(record)
    assert type(restored) is NestedValue
    assert type(restored.value) is ExampleValue
    assert restored == value


def test_configuration_dump_does_not_change_ordinary_pydantic_serialization():
    value = ExampleValue(count=3, scale=math.inf)
    ordinary = value.model_dump(mode="json")

    record = value.dump_config()

    assert ordinary == {"count": 3, "scale": math.inf}
    assert value.model_dump(mode="json") == ordinary
    assert record.config == {"count": 3, "scale": math.inf}


def test_configuration_round_trip_finds_models_in_supported_containers():
    child = ExampleValue(count=3, scale=2.5)
    value = NestedCollections(
        direct=child,
        listed=[child],
        keyed={"child": child},
    )

    restored = NestedCollections.load_record(value.dump_config())

    assert restored == value
    assert type(restored.direct) is ExampleValue
    assert type(restored.listed[0]) is ExampleValue
    assert type(restored.keyed["child"]) is ExampleValue


def test_configuration_round_trip_restores_tuples_from_json_arrays():
    value = TupleValue(
        counts=(1, 2),
        nested=(ExampleValue(count=3),),
    )

    record = value.dump_config()
    restored = TupleValue.load_record(record)

    assert record.config == {
        "counts": [1, 2],
        "nested": [
            {
                "path": f"{__name__}:ExampleValue",
                "config": {"count": 3, "scale": 1.0},
            }
        ],
    }
    assert restored == value
    assert isinstance(restored.counts, tuple)
    assert isinstance(restored.nested, tuple)
    assert (
        TupleValue.model_validate_json(
            '{"counts":[1,2],"nested":[{"count":3,"scale":1.0}]}'
        )
        == value
    )


def test_configuration_round_trip_tries_each_declared_platform_union_member():
    value = UnionValue(value=OtherValue(label="second union member"))

    restored = UnionValue.load_record(value.dump_config())

    assert restored == value
    assert type(restored.value) is OtherValue


def test_calling_class_rejects_another_platform_family():
    record = OtherValue(label="other").dump_config()

    with pytest.raises(TypeError, match="outside the ExampleValue family"):
        ExampleValue.load_record(record)


def test_qualified_paths_have_one_canonical_spelling():
    with pytest.raises(ValueError, match="use 'module:qualname'"):
        ExampleValue.load_record(
            {
                "path": f"{__name__}.ExampleValue",
                "config": {"count": 3, "scale": 2.5},
            }
        )


def test_qualified_paths_ignore_root_export_changes(monkeypatch: pytest.MonkeyPatch):
    """Configured records never derive identity from root exports."""
    import dpstats

    monkeypatch.setattr(dpstats, "__all__", [])

    budget = PureDPBudget(eps=1.0)
    domain = BoxDomain(bounds=np.array([[0.0, 1.0]]))

    assert budget.dump_config().model_dump() == {
        "path": "dpstats.budgets.families:PureDPBudget",
        "config": {"eps": 1.0},
    }
    assert domain.dump_config().model_dump() == {
        "path": "dpstats.domains:BoxDomain",
        "config": {"bounds": [[0.0, 1.0]]},
    }
    assert DPBudget.load_record(budget.dump_config()) == budget
    restored_domain = Domain.load_record(domain.dump_config())
    assert isinstance(restored_domain, BoxDomain)
    assert np.array_equal(restored_domain.bounds, domain.bounds)


def test_tuner_path_and_nested_configuration_ignore_root_export_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    """A built-in algorithm retains its qualified configured path."""
    import dpstats

    monkeypatch.setattr(dpstats, "__all__", [])
    tuner = DPSelectionLiuTalwar(
        alg_class=OLS,
        alg_hyperparam_grid={},
        selection_cost=_selection_cost,
    )

    record = tuner.dump_config()
    restored = BaseAlgorithm.load_record(record)

    assert record.path == ("dpstats.dp_tuning.liu_talwar:DPSelectionLiuTalwar")
    assert isinstance(restored, DPSelectionLiuTalwar)
    assert restored == tuner


def test_root_export_does_not_grant_a_custom_class_a_short_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """Custom classes retain qualified paths even when a caller re-exports them."""
    import dpstats

    monkeypatch.setattr(
        dpstats,
        "ExampleValue",
        ExampleValue,
        raising=False,
    )
    monkeypatch.setattr(
        dpstats,
        "__all__",
        [*dpstats.__all__, "ExampleValue"],
    )
    value = ExampleValue(count=3)

    assert value.dump_config().path == f"{__name__}:ExampleValue"
    with pytest.raises(ValueError, match="use 'module:qualname'"):
        PolymorphicModel.load_record({"path": "ExampleValue", "config": {"count": 3}})


@pytest.mark.parametrize(
    "bare_name",
    (
        "BaseAlgorithm",
        "PureDPBudget",
        "BoxDomain",
        "DPSelectionLiuTalwar",
    ),
)
def test_bare_names_are_not_configured_paths(bare_name: str):
    with pytest.raises(ValueError, match="use 'module:qualname'"):
        PolymorphicModel.load_record({"path": bare_name, "config": {}})


def test_script_local_models_are_rejected_when_persisted(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ExampleValue, "__module__", "__main__")

    with pytest.raises(ValueError, match="__main__"):
        ExampleValue(count=3).dump_config()


def test_models_cannot_declare_their_own_model_serializer():
    with pytest.raises(TypeError, match="cannot declare a model serializer"):

        class ShapeChangingValue(PolymorphicModel):
            count: int

            @model_serializer
            def rename_field(self) -> dict[str, int]:
                return {"renamed": self.count}


def test_configuration_excludes_computed_values_that_are_not_constructor_fields():
    value = ComputedValue(count=3)

    record = value.dump_config()

    assert value.model_dump(mode="python") == {"count": 3, "doubled": 6}
    assert record.config == {"count": 3}
    assert ComputedValue.load_record(record) == value


@pytest.mark.parametrize("invalid", [True, "1"])
def test_configuration_loading_preserves_strict_numeric_fields(invalid: Any):
    with pytest.raises(ValidationError):
        ExampleValue.load_record(
            {
                "path": f"{__name__}:ExampleValue",
                "config": {"count": invalid, "scale": 1.0},
            }
        )


def test_ordinary_validation_does_not_interpret_configuration_records():
    literal = LiteralConfigurationFields(path="literal", config={"count": 4})

    assert (
        LiteralConfigurationFields.model_validate(
            {"path": "literal", "config": {"count": 4}}
        )
        == literal
    )
    assert PolymorphicModel.load_record(literal.dump_config()) == literal


def test_recursive_path_and_config_fields_round_trip_as_ordinary_data():
    value = RecursiveConfigurationFields(
        path="outer",
        config=RecursiveConfigurationFields(path="inner"),
    )

    restored = PolymorphicModel.load_record(value.dump_config())

    assert restored == value
    assert isinstance(restored, RecursiveConfigurationFields)
    assert isinstance(restored.config, RecursiveConfigurationFields)
