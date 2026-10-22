from __future__ import annotations

from functools import partial
from typing import Callable

import pytest

from dpstats.parameters import Importable
from dpstats.transport.configuration import PolymorphicModel


ImportableFn = Importable[Callable[[float], float]]


def double(value: float) -> float:
    return 2 * value


def add(value: float, *, offset: float) -> float:
    return value + offset


class CallableValue(PolymorphicModel):
    fn: ImportableFn


def test_module_level_callable_round_trips_by_path():
    value = CallableValue(fn=double)
    record = value.model_dump(mode="json", serialize_as_any=False)

    assert record == {"fn": f"{__name__}:double"}
    restored = CallableValue.model_validate(record)
    assert restored.fn(3.0) == 6.0


def test_parameterized_callable_round_trips_by_path_and_kwargs():
    value = CallableValue(fn=partial(add, offset=2.5))
    record = value.model_dump(mode="json", serialize_as_any=False)

    assert record == {
        "fn": {
            "path": f"{__name__}:add",
            "kwargs": {"offset": 2.5},
        }
    }
    restored = CallableValue.model_validate(record)
    assert restored.fn(3.0) == 5.5


def test_parameterized_callable_round_trips_through_configuration():
    value = CallableValue(fn=partial(add, offset=2.5))

    record = value.dump_config()
    restored = CallableValue.load_record(record)

    assert record.model_dump() == {
        "path": f"{__name__}:CallableValue",
        "config": {
            "fn": {
                "path": f"{__name__}:add",
                "kwargs": {"offset": 2.5},
            }
        },
    }
    assert restored.fn(3.0) == 5.5


def test_script_local_callable_is_rejected_when_persisted(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(double, "__module__", "__main__")
    value = CallableValue(fn=double)

    with pytest.raises(ValueError, match="__main__"):
        value.model_dump(mode="json", serialize_as_any=False)


def test_lambda_is_rejected_when_persisted():
    value = CallableValue(fn=lambda x: x)

    with pytest.raises(ValueError, match="Cannot serialize .*lambda"):
        value.model_dump(mode="json", serialize_as_any=False)


def test_positional_partial_is_rejected_when_persisted():
    value = CallableValue(fn=partial(double, 2.0))

    with pytest.raises(ValueError, match="Only keyword-parameterized"):
        value.model_dump(mode="json", serialize_as_any=False)
