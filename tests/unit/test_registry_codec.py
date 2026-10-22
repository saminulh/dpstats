"""Technical registry-surface coverage, exercised through Algorithm entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.algorithm import (
    load_algorithms,
    load_algorithms_partitioned,
    register_algorithm,
)
from dpstats.budgets import DPBudget
from dpstats.instance import Instance, identify_instance
from dpstats.task import Array1D
from dpstats.workspace import open_collection
from dpstats.workspace.layout import (
    registry_directory_name,
    task_collection_directory,
    workspace_local_root,
)
from dpstats.workspace.registries import registry_document_path
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.transport.canonical import algorithm_identity
from dpstats.transport.registry import (
    JSONRegistry,
    add_entry,
    merge_registries,
    read_registry,
)
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.m_estimation.algorithms.approximate_minima_perturbation import (
    AMP,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_redberg import (
    ObjectivePerturbationRedberg,
)
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ToLinearRegression,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks import MEstimation
from dpstats.tasks import MeanEstimation
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleInput,
    SimpleMetadata,
    SimpleOutput,
)


class _ArrayConfiguredAlgorithm(BaseAlgorithm):
    """An importable author configuration with a multi-element array setting."""

    TASK = SIMPLE_TASK

    weights: Array1D

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        del metadata, budget, seed
        return SimpleOutput(result=input.X.mean(axis=0) * self.weights[0])


def _registry(payload: dict[str, Any]) -> JSONRegistry:
    return read_registry(json.dumps(payload).encode())


def _algorithms(payload: dict[str, Any], *, task_id: str) -> dict[str, BaseAlgorithm]:
    return load_algorithms(_registry(payload).entries, task_id=task_id, strict=True)


def _merge(
    task_id: str, kind: str, *entries: dict[str, Any], reconcile: bool = True
) -> JSONRegistry:
    """Merge one registry per entry map, reconciling label-only differences.

    Reconciliation rebuilds a colliding pair through the kind's public loader;
    ``reconcile=False`` keeps the exact-bytes contract.
    """
    registries = [
        _registry({"task": task_id, "kind": kind, "entries": entry_map})
        for entry_map in entries
    ]
    if not reconcile:
        return merge_registries(registries)
    return merge_registries(
        registries,
        reconstruct=(
            Instance.load_config if kind == "instance" else BaseAlgorithm.load_record
        ),
        derive_uuid=lambda value: value.uuid,
        native_task_id=lambda value: value.task.id,
    )


def _ridge_entry(*, multiplier: float | int = 0.01) -> dict[str, Any]:
    return {
        "path": "dpstats.tasks.linear_regression.algorithms.ridge:Ridge",
        "config": {"regularization_multiplier": multiplier},
    }


def _ols_payload(
    *,
    task: str = LinearRegression.id,
    kind: str = "algorithm",
    entries: Any | None = None,
    extra_field: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": task,
        "kind": kind,
        "entries": (
            {(OLS()).uuid: OLS().dump_config().model_dump()}
            if entries is None
            else entries
        ),
    }
    if extra_field:
        payload["unexpected"] = True
    return payload


def test_registered_algorithms_round_trip_by_stable_uuid(tmp_path: Path) -> None:
    """A registered Algorithm is reloaded exactly, selected by its UUID."""
    uuid = register_algorithm(tmp_path, Ridge(label="ridge"), registry="algorithms")
    path = registry_document_path(
        task_collection_directory(tmp_path, LinearRegression.id),
        "algorithm",
        "algorithms",
    )

    loaded = load_algorithms(
        read_registry(path.read_bytes()).entries, task_id=LinearRegression.id
    )

    assert set(loaded) == {uuid}
    loaded_algorithm = loaded[uuid]
    assert isinstance(loaded_algorithm, Ridge)
    assert loaded_algorithm.regularization_multiplier == 0.01
    assert loaded_algorithm.label == "ridge"


def test_adding_one_entry_produces_the_exact_wire_record() -> None:
    """One entry is the component's own configuration record and nothing else."""
    ridge = Ridge(label="ridge")
    registry = add_entry(
        JSONRegistry(task=LinearRegression.id, kind="algorithm", entries={}),
        uuid=ridge.uuid,
        record=ridge.dump_config().model_dump(),
    )

    assert registry.model_dump() == {
        "task": "linear_regression",
        "kind": "algorithm",
        "entries": {
            ridge.uuid: {
                "path": ("dpstats.tasks.linear_regression.algorithms." "ridge:Ridge"),
                "config": {"label": "ridge", "regularization_multiplier": 0.01},
            }
        },
    }
    assert ridge.uuid == Ridge().uuid


def test_registration_is_idempotent_and_appends_a_second_configuration(
    tmp_path: Path,
) -> None:
    """Repeated calls retain one entry while distinct configurations accumulate."""
    ols_uuid = register_algorithm(tmp_path, OLS(label="ols"), registry="algorithms")
    assert (
        register_algorithm(tmp_path, OLS(label="ols"), registry="algorithms")
        == ols_uuid
    )
    ridge_uuid = register_algorithm(
        tmp_path, Ridge(label="ridge"), registry="algorithms"
    )
    path = registry_document_path(
        task_collection_directory(tmp_path, LinearRegression.id),
        "algorithm",
        "algorithms",
    )

    assert set(read_registry(path.read_bytes()).entries) == {ols_uuid, ridge_uuid}


def test_algorithm_identity_is_independent_of_its_label(tmp_path: Path) -> None:
    """A label organizes a payload without changing Algorithm identity or filed bytes."""
    assert OLS(label="ordinary").uuid == OLS(label="baseline").uuid == OLS().uuid

    local_root = workspace_local_root(tmp_path)
    register_algorithm(local_root, OLS(label="ordinary"), registry="first")
    register_algorithm(local_root, OLS(label="ordinary"), registry="second")
    paths = list(
        (
            task_collection_directory(local_root, LinearRegression.id)
            / registry_directory_name("algorithm")
        ).glob("*.json")
    )
    original = {path: path.read_bytes() for path in paths}

    loaded = open_collection(tmp_path, LinearRegression).load_algorithms()

    assert loaded[OLS().uuid].label == "ordinary"
    assert {path: path.read_bytes() for path in paths} == original


def test_loaded_algorithm_derives_its_privacy_interpretation(
    tmp_path: Path,
) -> None:
    """Registry transport does not duplicate an Algorithm object's semantics."""
    standard = AMP(
        objective_noise_ratio=1.5, gradient_tolerance=1e-6, output_budget_fraction=0.2
    )
    standard_uuid = register_algorithm(tmp_path, standard, registry="algorithms")
    theoretical_uuid = register_algorithm(
        tmp_path, ObjectivePerturbationRedberg(), registry="algorithms"
    )
    path = registry_document_path(
        task_collection_directory(tmp_path, MEstimation.id), "algorithm", "algorithms"
    )

    entries = read_registry(path.read_bytes()).entries
    assert all(
        "privacy_interpretation" not in entry["config"] for entry in entries.values()
    )
    loaded = load_algorithms(entries, task_id=MEstimation.id)
    assert loaded[standard_uuid] == standard
    assert (
        loaded[standard_uuid].privacy_interpretation is PrivacyInterpretation.STANDARD
    )
    assert (
        loaded[theoretical_uuid].privacy_interpretation
        is PrivacyInterpretation.THEORETICAL
    )


def test_merging_keeps_distinct_identities_with_the_same_label(tmp_path: Path) -> None:
    """Independently authored registries may reuse presentation labels."""
    register_algorithm(tmp_path, OLS(label="shared"), registry="first")
    register_algorithm(tmp_path, Ridge(label="shared"), registry="second")
    registries = task_collection_directory(
        tmp_path, LinearRegression.id
    ) / registry_directory_name("algorithm")

    merged = merge_registries(
        [
            read_registry((registries / "first.json").read_bytes()),
            read_registry((registries / "second.json").read_bytes()),
        ]
    )
    assert set(merged.entries) == {OLS().uuid, Ridge().uuid}


def test_merging_tolerates_a_label_only_difference() -> None:
    """A label is excluded from identity, so relabeling elsewhere is not a conflict.

    The first-seen (caller-ordered) entry's label is kept; loading the
    merged entry still yields the identity its configuration derives.
    """
    uuid = OLS().uuid
    local_entry = OLS(label="TailRobustSparseHuberIHT").dump_config().model_dump()
    release_entry = OLS().dump_config().model_dump()

    merged = _merge(
        LinearRegression.id, "algorithm", {uuid: local_entry}, {uuid: release_entry}
    )

    assert set(merged.entries) == {uuid}
    assert merged.entries[uuid]["config"]["label"] == "TailRobustSparseHuberIHT"
    loaded = load_algorithms(merged.entries, task_id=LinearRegression.id)
    assert loaded[uuid].uuid == uuid


def test_merging_tolerates_a_label_difference_with_an_array_configuration() -> None:
    """Structural reconciliation handles author-facing multi-element arrays."""
    local = _ArrayConfiguredAlgorithm(weights=np.array([1.0, 2.0]), label="local panel")
    released = _ArrayConfiguredAlgorithm(weights=np.array([1.0, 2.0]))
    assert local.uuid == released.uuid

    merged = _merge(
        SIMPLE_TASK.id,
        "algorithm",
        {local.uuid: local.dump_config().model_dump()},
        {released.uuid: released.dump_config().model_dump()},
    )

    assert merged.entries[local.uuid]["config"]["label"] == "local panel"


def test_merging_tolerates_a_label_difference_nested_inside_a_composite() -> None:
    """A label anywhere in a configured value's structure is excluded from identity.

    A byte-level, top-level-only label strip would miss this: the differing
    label sits on the composite's inner ``algorithm``, not on the composite
    itself.
    """
    composite = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=OLS()
    )
    uuid = composite.uuid
    task_id = composite.task.id
    local_entry = (
        CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=OLS(label="LocalName")
        )
        .dump_config()
        .model_dump()
    )
    release_entry = composite.dump_config().model_dump()
    assert local_entry != release_entry

    merged = _merge(task_id, "algorithm", {uuid: local_entry}, {uuid: release_entry})

    assert merged.entries[uuid] == local_entry
    loaded = load_algorithms(merged.entries, task_id=task_id)
    assert loaded[uuid].uuid == uuid


def test_merging_never_reconstructs_a_non_colliding_entry() -> None:
    """Only a colliding UUID is ever reconstructed; an unrelated broken entry is not."""
    uuid = OLS().uuid
    local_entry = OLS(label="Local").dump_config().model_dump()
    release_entry = OLS().dump_config().model_dump()
    broken_uuid = "a3-UNRELATEDBROKEN0000000"
    broken_entry: dict[str, Any] = {"path": "uninstalled_algorithm:Ghost", "config": {}}

    merged = _merge(
        LinearRegression.id,
        "algorithm",
        {uuid: local_entry, broken_uuid: broken_entry},
        {uuid: release_entry},
    )

    assert set(merged.entries) == {uuid, broken_uuid}
    assert merged.entries[uuid]["config"]["label"] == "Local"
    assert merged.entries[broken_uuid] == broken_entry


def test_merging_refuses_a_record_misfiled_under_the_wrong_uuid() -> None:
    """A colliding pair still verifies each record's filed key, not just equality."""
    wrong_uuid = Ridge().uuid
    misfiled_a = OLS().dump_config().model_dump()
    misfiled_b = OLS(label="X").dump_config().model_dump()

    with pytest.raises(ValueError, match="derived UUID"):
        _merge(
            LinearRegression.id,
            "algorithm",
            {wrong_uuid: misfiled_a},
            {wrong_uuid: misfiled_b},
        )


def test_merging_refuses_an_instance_whose_metric_arguments_differ() -> None:
    """Instance equality keeps ``metric_arguments``, unlike ``label``."""
    metadata = SimpleMetadata(n_samples=1, dim=1)
    input_data = SimpleInput(X=np.zeros((1, 1)))
    local = identify_instance(metadata, input_data, {"a": 1}, label="Local")
    release = identify_instance(metadata, input_data, {"a": 2})
    assert local.uuid == release.uuid

    with pytest.raises(ValueError, match="different entry"):
        _merge(
            SIMPLE_TASK.id,
            "instance",
            {local.uuid: local.record()},
            {local.uuid: release.record()},
        )


def test_merging_without_a_reconciler_refuses_any_byte_difference() -> None:
    """Omitting reconstruct/derive_uuid/native_task_id keeps the exact-match contract."""
    uuid = OLS().uuid
    with pytest.raises(ValueError, match="different entry"):
        _merge(
            LinearRegression.id,
            "algorithm",
            {uuid: OLS(label="Local").dump_config().model_dump()},
            {uuid: OLS().dump_config().model_dump()},
            reconcile=False,
        )


def test_merging_refuses_constructing_an_empty_registry() -> None:
    """Merging states which registries it read rather than inventing one."""
    with pytest.raises(ValueError, match="At least one registry"):
        merge_registries([])


@pytest.mark.parametrize(
    ("other", "message"),
    [
        (
            {"task": MeanEstimation.id, "kind": "algorithm", "entries": {}},
            "must declare the same task",
        ),
        (
            {"task": LinearRegression.id, "kind": "instance", "entries": {}},
            "must declare the same kind",
        ),
    ],
)
def test_merging_refuses_registries_declaring_different_things(
    other: dict[str, Any], message: str
) -> None:
    """Two registries naming different tasks or kinds are not one registry."""
    payload: dict[str, Any] = {
        "task": LinearRegression.id,
        "kind": "algorithm",
        "entries": {},
    }

    with pytest.raises(ValueError, match=message):
        merge_registries([_registry(payload), _registry(other)])


def test_loading_refuses_a_tampered_configuration() -> None:
    """A UUID commits the serialized hyperparameter values exactly.

    Merging reads an entry's shape and nothing else, so a tampered
    configuration survives it and is refused where the value is rebuilt.
    """
    tampered = {
        "task": LinearRegression.id,
        "kind": "algorithm",
        "entries": {Ridge().uuid: _ridge_entry(multiplier=99.0)},
    }

    reconciled = merge_registries([_registry(tampered)])

    with pytest.raises(ValueError, match="derived UUID"):
        load_algorithms(reconciled.entries, task_id=LinearRegression.id, strict=True)


def test_loading_refuses_an_alternate_number_uuid() -> None:
    """An integer spelling cannot introduce a second identity for a float field.

    The loader raises that identity refusal itself rather than re-typing it as a
    foreign reconstruction failure.
    """
    path = "dpstats.tasks.linear_regression.algorithms.ridge:Ridge"
    alternate = {
        "task": LinearRegression.id,
        "kind": "algorithm",
        "entries": {
            algorithm_identity(
                {"path": path, "config": {"regularization_multiplier": 2}}
            ): _ridge_entry(multiplier=2)
        },
    }

    with pytest.raises(ValueError, match="derived UUID") as refusal:
        _algorithms(alternate, task_id=LinearRegression.id)

    assert refusal.value.__cause__ is None
    assert "Could not reconstruct" not in str(refusal.value)


def test_unimportable_configuration_states_the_entry_and_keeps_its_cause() -> None:
    """A foreign reconstruction failure names its entry and keeps its cause."""
    path = "dpstats.tasks.linear_regression.algorithms.ridge:Absent"
    config: dict[str, Any] = {}
    uuid = algorithm_identity({"path": path, "config": config})
    payload: dict[str, Any] = {
        "task": LinearRegression.id,
        "kind": "algorithm",
        "entries": {uuid: {"path": path, "config": config}},
    }

    with pytest.raises(ValueError) as refusal:
        _algorithms(payload, task_id=LinearRegression.id)

    assert uuid in str(refusal.value)
    cause = refusal.value.__cause__
    assert isinstance(cause, AttributeError)
    assert "Absent" in str(cause)


def test_merging_never_imports_configured_algorithm_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry reconciliation is structural; imports occur only at loading."""
    register_algorithm(tmp_path, OLS(label="ols"), registry="algorithms")
    path = registry_document_path(
        task_collection_directory(tmp_path, LinearRegression.id),
        "algorithm",
        "algorithms",
    )
    stored = read_registry(path.read_bytes())

    def refuse(name: str) -> None:
        raise AssertionError(f"reconciliation imported {name!r}")

    monkeypatch.setattr("dpstats.transport.configuration._load_config_class", refuse)

    merged = merge_registries([stored])

    assert set(merged.entries)


def test_loading_refuses_a_configuration_native_to_another_task() -> None:
    """Only reconstruction can check a configured class's native Task.

    The entry's identity is honest: a content address names the portable
    configuration, so a registry can hold a well-formed entry that belongs to
    another task and only the rebuilt value knows the difference.
    """
    payload: dict[str, Any] = {
        "task": MeanEstimation.id,
        "kind": "algorithm",
        "entries": {(OLS()).uuid: OLS().dump_config().model_dump()},
    }

    with pytest.raises(
        ValueError, match="native to task 'linear_regression'"
    ) as refusal:
        _algorithms(payload, task_id=MeanEstimation.id)

    assert (OLS()).uuid in str(refusal.value)
    assert refusal.value.__cause__ is None


def test_loading_checks_a_composition_against_its_installed_task() -> None:
    """A composition's Task comes from its chain, not from a class attribute."""
    composite = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=OLS()
    )
    payload: dict[str, Any] = {
        "task": LinearRegression.id,
        "kind": "algorithm",
        "entries": {composite.uuid: composite.dump_config().model_dump()},
    }

    with pytest.raises(
        ValueError, match="native to task 'single_coefficient_estimation'"
    ) as refusal:
        _algorithms(payload, task_id=LinearRegression.id)

    assert composite.uuid in str(refusal.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_ols_payload(extra_field=True), "unexpected"),
        (_ols_payload(kind="roster"), "kind"),
        (_ols_payload(entries=[1]), "entries"),
        (_ols_payload(entries={(OLS()).uuid: [1]}), "entries"),
    ],
)
def test_reading_validates_the_complete_payload_before_importing(
    payload: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Envelope errors cannot reach configured-code imports."""

    def refuse(name: str) -> None:
        raise AssertionError(f"invalid payload imported {name!r}")

    monkeypatch.setattr("dpstats.transport.configuration._load_config_class", refuse)

    with pytest.raises(ValidationError, match=message):
        _registry(payload)


def _stale_and_valid_algorithms() -> tuple[dict[str, Any], str, str]:
    """One loadable Ridge entry beside one whose class does not exist."""
    path = "dpstats.tasks.linear_regression.algorithms.ridge:Absent"
    stale = algorithm_identity({"path": path, "config": {}})
    entries: dict[str, Any] = {
        Ridge().uuid: _ridge_entry(),
        stale: {"path": path, "config": {}},
    }
    return entries, Ridge().uuid, stale


def test_lenient_loading_keeps_valid_siblings_and_warns_once() -> None:
    entries, valid, stale = _stale_and_valid_algorithms()

    with pytest.warns(UserWarning) as recorded:
        loaded = load_algorithms(entries, task_id=LinearRegression.id, strict=False)

    assert list(loaded) == [valid]
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert message.startswith(
        "Skipped 1 of 2 algorithm registry entries this library cannot rebuild: "
        f"{stale} ("
    )
    assert "\n" not in message


def test_loading_warns_nothing_when_every_entry_loads(recwarn: Any) -> None:
    load_algorithms({Ridge().uuid: _ridge_entry()}, task_id=LinearRegression.id)

    assert len(recwarn) == 0


def test_loading_refuses_the_group_by_default() -> None:
    entries, _, stale = _stale_and_valid_algorithms()

    with pytest.raises(ValueError, match="Could not reconstruct") as refusal:
        load_algorithms(entries, task_id=LinearRegression.id)

    assert stale in str(refusal.value)


def test_partitioned_loading_reports_each_rejection_without_warning(
    recwarn: Any,
) -> None:
    entries, valid, stale = _stale_and_valid_algorithms()

    loaded, rejections = load_algorithms_partitioned(
        entries, task_id=LinearRegression.id
    )

    assert list(loaded) == [valid]
    assert set(rejections) == {stale}
    assert "Absent" in rejections[stale]
    assert len(recwarn) == 0
