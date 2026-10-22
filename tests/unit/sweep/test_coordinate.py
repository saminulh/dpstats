"""Tests for CellCoordinate and its persistence boundary."""

from __future__ import annotations

from typing import Any

import pytest

from dpstats.instance import Instance
from dpstats.budgets import ApproxDPBudget, PureDPBudget
from dpstats.coordinates import (
    CellCoordinate,
    CoordinateRecord,
    coordinate_record_identity,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from tests.registry_components import SIMPLE_TASK, SimpleAlgorithm


def _instance(*, label: str | None = None) -> Instance:
    return Instance(
        metadata=SIMPLE_TASK.Metadata(n_samples=1, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
        label=label,
    )


def test_coordinate_holds_domain_objects_and_projects_only_stable_references() -> None:
    instance = _instance(label="presentation only")
    algorithm = SimpleAlgorithm(scale=2.0)
    budget = PureDPBudget(eps=1.0)

    coordinate = CellCoordinate(
        instance=instance, algorithm=algorithm, budget=budget, alg_seed=7
    )

    assert coordinate.task is SIMPLE_TASK
    assert coordinate.instance is instance
    assert coordinate.algorithm is algorithm
    assert coordinate.budget is budget
    assert coordinate.to_record() == {
        "instance_uuid": instance.uuid,
        "alg_uuid": algorithm.uuid,
        "budget": budget.dump_config().model_dump(),
        "alg_seed": 7,
    }


def test_coordinate_refuses_an_algorithm_native_to_another_task() -> None:
    """Construction is where a cell's Instance and algorithm must agree on task."""
    with pytest.raises(ValueError, match="belongs to task"):
        CellCoordinate(
            instance=_instance(),
            algorithm=SampleMean(),
            budget=PureDPBudget(eps=1.0),
            alg_seed=0,
        )


def test_coordinate_round_trips_through_references_without_materializing_input() -> (
    None
):
    instance = _instance()
    algorithm = SimpleAlgorithm(scale=2.0)
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=3,
    )
    record = coordinate.to_record()

    loaded = CellCoordinate.from_record(
        record,
        instances={instance.uuid: instance},
        algorithms={algorithm.uuid: algorithm},
    )

    assert loaded == coordinate
    assert loaded.instance is instance
    assert loaded.algorithm is algorithm


def test_coordinate_validation_is_code_free_but_rehydration_resolves_budget() -> None:
    instance = _instance()
    algorithm = SimpleAlgorithm()
    record = {
        "instance_uuid": instance.uuid,
        "alg_uuid": algorithm.uuid,
        "budget": {"path": "does.not.exist:Budget", "config": {}},
        "alg_seed": 0,
    }

    CoordinateRecord.model_validate(record)
    with pytest.raises((ImportError, ValueError)):
        CellCoordinate.from_record(
            record,
            instances={instance.uuid: instance},
            algorithms={algorithm.uuid: algorithm},
        )


@pytest.mark.parametrize("field", ["instance_uuid", "alg_uuid"])
@pytest.mark.parametrize("persisted", [7, None, [], {}])
def test_a_malformed_reference_is_named_rather_than_read_as_uninstalled(
    field: str, persisted: Any
) -> None:
    """A broken reference field is a broken record, not a missing component.

    Reporting it as an unknown Instance or algorithm would send a reader
    looking for something to install, and an unhashable field would surface as
    a bare lookup failure instead of a diagnosis.
    """
    instance = _instance()
    algorithm = SimpleAlgorithm()
    record = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=0,
    ).to_record()
    record[field] = persisted

    with pytest.raises(ValueError) as refusal:
        CellCoordinate.from_record(
            record,
            instances={instance.uuid: instance},
            algorithms={algorithm.uuid: algorithm},
        )

    assert field in str(refusal.value)


@pytest.mark.parametrize("persisted", [True, 3.0, "1", None])
def test_the_sweep_reader_refuses_a_persisted_seed_that_is_not_an_integer(
    persisted: Any,
) -> None:
    """The persistence boundary is where an untrusted seed is checked.

    ``True`` is the case an ordinary integer check would miss, since a boolean
    is an ``int`` that compares equal to ``1``.
    """
    instance = _instance()
    algorithm = SimpleAlgorithm()
    record = {
        "instance_uuid": instance.uuid,
        "alg_uuid": algorithm.uuid,
        "budget": PureDPBudget(eps=1.0).dump_config(),
        "alg_seed": persisted,
    }

    with pytest.raises(ValueError, match="alg_seed"):
        CoordinateRecord.model_validate(record)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_outside_the_algorithm_seed_domain_is_refused(seed: int) -> None:
    """One domain governs the runtime coordinate and its persisted record."""
    with pytest.raises(ValueError, match="alg_seed"):
        CellCoordinate(
            instance=_instance(),
            algorithm=SimpleAlgorithm(),
            budget=PureDPBudget(eps=1.0),
            alg_seed=seed,
        )


def test_a_persisted_budget_must_name_the_class_it_configures() -> None:
    """An envelope with no class path names nothing that can be rebuilt.

    Code-free validation is the boundary that judges this: rehydration would
    refuse the empty path later, but a reader holding only the bytes has to be
    able to say the row is broken without importing anything.
    """
    instance = _instance()
    record = {
        "instance_uuid": instance.uuid,
        "alg_uuid": SimpleAlgorithm().uuid,
        "budget": {"path": "", "config": {"eps": 1.0}},
        "alg_seed": 0,
    }

    with pytest.raises(ValueError) as refusal:
        CoordinateRecord.model_validate(record)

    assert "path" in str(refusal.value)


def test_coordinate_rehydration_refuses_reference_aliasing() -> None:
    instance = _instance()
    algorithm = SimpleAlgorithm(scale=1.0)
    other_algorithm = SimpleAlgorithm(scale=2.0)
    record = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=0,
    ).to_record()

    with pytest.raises(ValueError, match="does not match"):
        CellCoordinate.from_record(
            record,
            instances={instance.uuid: instance},
            algorithms={record["alg_uuid"]: other_algorithm},
        )


def test_reused_identities_equal_the_identities_of_equal_fresh_values() -> None:
    instance = _instance(label="presentation only")
    algorithm = SimpleAlgorithm(scale=2.0)
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=7,
    )

    identities = (instance.uuid, algorithm.uuid, coordinate.uuid)

    fresh_instance = _instance(label="presentation only")
    fresh_algorithm = SimpleAlgorithm(scale=2.0)
    fresh_coordinate = CellCoordinate(
        instance=fresh_instance,
        algorithm=fresh_algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=7,
    )
    assert identities == (
        fresh_instance.uuid,
        fresh_algorithm.uuid,
        fresh_coordinate.uuid,
    )


def test_an_updated_copy_addresses_its_own_cell_not_the_original_one() -> None:
    """A copy with a changed field is a different cell and needs its own token.

    Reading the identity before copying is the case that matters: the derived
    token belongs to the value it was derived from, so a copy that carried it
    would address the cell it came from in every claim, chunk, and join.
    """
    instance = _instance()
    algorithm = SimpleAlgorithm()
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=PureDPBudget(eps=1.0),
        alg_seed=0,
    )
    original = coordinate.uuid

    copied = coordinate.model_copy(update={"alg_seed": 7})

    assert copied.alg_seed == 7
    assert coordinate.uuid == original
    assert copied.uuid != original
    assert (
        copied.uuid
        == CellCoordinate(
            instance=instance,
            algorithm=algorithm,
            budget=PureDPBudget(eps=1.0),
            alg_seed=7,
        ).uuid
    )


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (PureDPBudget(eps=1.0), "c3-DNNTYIQFZ4WRGDO4"),
        (ApproxDPBudget(eps=3.0, delta=1e-6), "c3-7IAYBMD6HAEQ4TVB"),
    ],
)
def test_coordinate_identity_is_stable_and_names_its_persisted_record(
    budget: PureDPBudget | ApproxDPBudget, expected: str
) -> None:
    """Identities already persisted keep naming the same cells."""
    coordinate = CellCoordinate(
        instance=_instance(),
        algorithm=SimpleAlgorithm(scale=2.0),
        budget=budget,
        alg_seed=7,
    )

    assert coordinate.uuid == expected
    assert coordinate_record_identity(coordinate.to_record()) == expected
