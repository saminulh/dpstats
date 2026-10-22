"""Direct Instances, logical input storage, and task-local registries."""

from __future__ import annotations

from typing import Any

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from dpstats.transport.canonical import canonical_json_bytes
from dpstats.workspace.layout import INSTANCE_REGISTRY_NAME
from dpstats.domains import BoxDomain
from dpstats.instance import Instance
from dpstats.instance import (
    build_instance_registry,
    identify_instance,
    load_instance_input,
    load_instances,
    load_instances_partitioned,
    persist_instance,
    read_instance_registry,
    register_instance,
    verify_instance_input,
    write_instance_registry,
)
from dpstats.transport.registry import (
    JSONRegistry,
    add_entry,
    read_registry,
)
from dpstats.task import Task, TaskInput, TaskMetadata, TaskOutput
from dpstats.task import Array1D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from tests.registry_components import (
    simple_producer,
    SimpleInput,
)


def mean_metadata(n_samples: int = 2) -> MeanEstimationMetadata:
    return MeanEstimationMetadata(
        n_samples=n_samples,
        domain=BoxDomain(bounds=np.array([[0.0, 1.0], [0.0, 1.0]])),
    )


def mean_input() -> MeanEstimationInput:
    return MeanEstimationInput(X=np.array([[0.25, 0.5], [0.75, 1.0]]))


class MixedInput(TaskInput):
    X: Array1D
    note: str


class MixedMetadata(TaskMetadata):
    def validate_input(self, input: TaskInput) -> None:
        input = MixedInput.model_validate(input)
        if len(input.X) != self.n_samples:
            raise ValueError("Input length must match n_samples.")


class MixedOutput(TaskOutput):
    value: float


MixedTask = Task(
    "mixed_input",
    "Mixed input",
    "Test writer rejection of public non-array Input fields.",
    MixedMetadata,
    MixedInput,
    MixedOutput,
)


def mean_instance(root: Path, *, label: str | None = None) -> Instance:
    metadata = mean_metadata()
    return persist_instance(
        metadata,
        mean_input(),
        metric_arguments={"reference": [0.5, 0.75]},
        input_artifacts_root=root,
        label=label,
    )


class TestLogicalInputIdentity:
    def test_layout_byte_order_and_npz_repacking_do_not_change_identity(
        self, tmp_path: Path
    ) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        metadata = MeanEstimationMetadata(
            n_samples=2,
            domain=BoxDomain(bounds=np.array([[0.0, 4.0], [0.0, 4.0]])),
        )
        digest = persist_instance(
            metadata, MeanEstimationInput(X=values), {}, tmp_path
        ).input_digest
        np.savez_compressed(tmp_path / f"{digest}.npz", X=values)

        assert (
            persist_instance(
                metadata, MeanEstimationInput(X=np.asfortranarray(values)), {}, tmp_path
            ).input_digest
            == digest
        )
        assert (
            persist_instance(
                metadata,
                MeanEstimationInput(X=values.astype(values.dtype.newbyteorder(">"))),
                {},
                tmp_path,
            ).input_digest
            == digest
        )

    def test_dtype_and_values_change_the_content_address(self, tmp_path: Path) -> None:
        metadata = MeanEstimationMetadata(
            n_samples=2,
            domain=BoxDomain(bounds=np.array([[0.0, 4.0], [0.0, 4.0]])),
        )
        base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        digest = persist_instance(
            metadata, MeanEstimationInput(X=base), {}, tmp_path
        ).input_digest

        assert (
            persist_instance(
                metadata,
                MeanEstimationInput(X=base.astype(np.float32)),
                {},
                tmp_path,
            ).input_digest
            != digest
        )
        assert (
            persist_instance(
                metadata, MeanEstimationInput(X=base - 0.5), {}, tmp_path
            ).input_digest
            != digest
        )

    def test_writer_refuses_non_array_input_fields(self, tmp_path: Path) -> None:
        value = MixedInput(X=np.array([1.0]), note="not private array content")
        with pytest.raises(TypeError, match="note.*ndarray"):
            persist_instance(MixedMetadata(n_samples=1), value, {}, tmp_path)

    def test_writer_refuses_input_that_violates_metadata(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_samples=3"):
            persist_instance(mean_metadata(n_samples=3), mean_input(), {}, tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_occupied_logical_address_is_verified_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        input = mean_input()
        digest = persist_instance(mean_metadata(), input, {}, tmp_path).input_digest
        target = tmp_path / f"{digest}.npz"
        np.savez(target, X=input.X + 10)
        occupied = target.read_bytes()

        with pytest.raises(ValueError, match="logical digest"):
            persist_instance(mean_metadata(), input, {}, tmp_path)

        assert target.read_bytes() == occupied


class TestInstance:
    def test_independently_loaded_instances_are_usable_as_join_keys(
        self, tmp_path: Path
    ) -> None:
        original = mean_instance(tmp_path)
        restored = Instance.load_config(original.record())

        joined = pd.DataFrame({"instance": [original], "value": [2.0]}).merge(
            pd.DataFrame({"instance": [restored], "group": ["declared"]}),
            on="instance",
            validate="many_to_one",
        )

        assert joined["group"].tolist() == ["declared"]
        assert original == restored
        assert hash(original) == hash(restored)
        assert restored == restored.model_copy(update={"label": "another label"})
        assert restored != restored.model_copy(update={"metric_arguments": {}})
        assert restored != restored.model_copy(
            update={"metadata": mean_metadata(n_samples=3)}
        )

    def test_loaded_input_constructs_the_same_identity_without_writing(
        self, tmp_path: Path
    ) -> None:
        metadata = mean_metadata()
        input = mean_input()

        loaded = identify_instance(
            metadata,
            input,
            metric_arguments={"reference": [0.5, 0.75]},
        )

        assert list(tmp_path.iterdir()) == []
        persisted = persist_instance(
            metadata,
            input,
            metric_arguments={"reference": [0.5, 0.75]},
            input_artifacts_root=tmp_path,
        )
        assert loaded == persisted
        assert loaded.uuid == persisted.uuid

    def test_identity_covers_the_problem_but_not_label_or_references(
        self, tmp_path: Path
    ) -> None:
        base = mean_instance(tmp_path)
        assert base.uuid.startswith("i5-")
        assert base.uuid == "i5-4J7T33AH7NTLFYXK"
        assert base.uuid is base.uuid

        relabelled = base.model_copy(update={"label": "readable name"})
        new_references = base.model_copy(
            update={"metric_arguments": {"reference": [0.25, 0.75]}}
        )
        new_metadata = base.model_copy(update={"metadata": mean_metadata(n_samples=3)})
        new_input = base.model_copy(update={"input_digest": "0" * 64})

        assert relabelled.uuid == base.uuid
        assert new_references.uuid == base.uuid
        assert new_metadata.uuid != base.uuid
        assert new_input.uuid != base.uuid

    def test_an_instance_prints_as_its_label_or_its_uuid(self, tmp_path: Path) -> None:
        """``str`` is the label, else the UUID; ``repr`` stays the full record."""
        unlabelled = mean_instance(tmp_path)
        labelled = unlabelled.model_copy(update={"label": "readable name"})

        assert str(unlabelled) == unlabelled.uuid
        assert f"{labelled}" == "readable name"
        assert repr(labelled).startswith("Instance(metadata=")

    def test_metric_arguments_are_copied_and_strict_json(self, tmp_path: Path) -> None:
        authored: dict[str, list[Any]] = {"nested": [1, {"ok": True}]}
        instance = persist_instance(mean_metadata(), mean_input(), authored, tmp_path)
        authored["nested"].append(2)

        assert isinstance(instance.metric_arguments, dict)
        assert instance.model_dump(mode="python")["metric_arguments"] == {
            "nested": [1, {"ok": True}]
        }

        invalid_values: tuple[Any, ...] = (
            {"array": np.array([1.0])},
            {"set": {1, 2}},
            {1: "non-string key"},
            {"reserved": "nan_json"},
        )
        for invalid in invalid_values:
            with pytest.raises(
                ValidationError, match="valid JSON|valid string|reserved"
            ):
                Instance.model_validate(
                    {
                        "metadata": mean_metadata(),
                        "input_digest": "0" * 64,
                        "metric_arguments": invalid,
                    }
                )

    def test_metric_arguments_reserve_input_for_transient_measurement(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValidationError, match="reserved key 'input'"):
            persist_instance(
                mean_metadata(),
                mean_input(),
                {"input": "not persisted"},
                tmp_path,
            )
        assert list(tmp_path.iterdir()) == []

    def test_load_returns_the_typed_input(self, tmp_path: Path) -> None:
        instance = mean_instance(tmp_path)

        first = load_instance_input(instance, tmp_path)
        second = load_instance_input(instance, tmp_path)

        assert isinstance(first, MeanEstimationInput)
        assert isinstance(second, MeanEstimationInput)
        assert np.array_equal(first.X, mean_input().X)
        assert np.array_equal(second.X, first.X)
        assert second.X is not first.X
        assert "input" not in vars(instance)

    def test_load_takes_a_trusted_store_as_found(self, tmp_path: Path) -> None:
        """A trusted store's artifact is named by its content, so loading trusts it."""
        instance = mean_instance(tmp_path)
        substituted = mean_input().X * 0.5
        np.savez(
            tmp_path / f"{instance.input_digest}.npz",
            X=substituted,
        )

        loaded = load_instance_input(instance, tmp_path)

        assert isinstance(loaded, MeanEstimationInput)
        assert np.array_equal(loaded.X, substituted)

    def test_verification_refuses_substituted_contents(self, tmp_path: Path) -> None:
        """An untrusted artifact is refused when its bytes are not what names it."""
        instance = mean_instance(tmp_path)
        np.savez(
            tmp_path / f"{instance.input_digest}.npz",
            X=mean_input().X * 0.5,
        )

        with pytest.raises(ValueError, match="logical digest"):
            verify_instance_input(instance, tmp_path)

    def test_verification_accepts_the_exact_stored_input(self, tmp_path: Path) -> None:
        instance = mean_instance(tmp_path)

        verify_instance_input(instance, tmp_path)

    def test_verification_refuses_input_its_metadata_rejects(
        self, tmp_path: Path
    ) -> None:
        """Digest agreement alone does not admit an Input the Task contract rejects."""
        stored = mean_instance(tmp_path)
        instance = Instance(
            metadata=mean_metadata(n_samples=3),
            input_digest=stored.input_digest,
            metric_arguments={},
        )

        with pytest.raises(ValueError, match="n_samples=3"):
            verify_instance_input(instance, tmp_path)

    def test_metadata_owns_input_pairing_conditions(self, tmp_path: Path) -> None:
        instance = Instance(
            metadata=mean_metadata(n_samples=3),
            input_digest=mean_instance(tmp_path).input_digest,
            metric_arguments={},
        )

        with pytest.raises(ValueError, match="n_samples=3"):
            load_instance_input(instance, tmp_path)

    def test_producer_immediately_persists_the_exact_input(
        self, tmp_path: Path
    ) -> None:
        producer = simple_producer(n=5, d=2)
        instance = persist_instance(*producer(7), tmp_path)
        input_artifacts = tmp_path
        stored = input_artifacts / f"{instance.input_digest}.npz"

        assert stored.is_file()
        assert not list(input_artifacts.glob("*.generator.json"))

        loaded = load_instance_input(instance, input_artifacts)
        metadata, expected, references = producer(7)
        stored_instance = persist_instance(
            metadata, expected, references, tmp_path / "stored"
        )

        assert isinstance(loaded, SimpleInput)
        assert isinstance(expected, SimpleInput)
        np.testing.assert_array_equal(loaded.X, expected.X)
        assert stored.is_file()
        assert instance.uuid == stored_instance.uuid

    def test_instances_that_share_input_bytes_share_one_exact_artifact(
        self, tmp_path: Path
    ) -> None:
        first = mean_instance(tmp_path)
        # A second problem over the same data: only its declared domain differs.
        wider = MeanEstimationMetadata(
            n_samples=2, domain=BoxDomain(bounds=np.array([[0.0, 4.0], [0.0, 4.0]]))
        )
        second = persist_instance(wider, mean_input(), {}, tmp_path)
        input_artifacts = tmp_path

        assert first.input_digest == second.input_digest
        assert first.uuid != second.uuid
        assert list(input_artifacts.glob("*.npz")) == [
            input_artifacts / f"{first.input_digest}.npz"
        ]
        assert isinstance(
            load_instance_input(first, input_artifacts), MeanEstimationInput
        )
        assert isinstance(
            load_instance_input(second, input_artifacts), MeanEstimationInput
        )


class TestInstanceRegistry:
    def test_payload_has_one_task_and_direct_instance_values(
        self, tmp_path: Path
    ) -> None:
        instance = mean_instance(tmp_path, label="study row")
        payload = build_instance_registry(MeanEstimation.id, [instance])
        record = payload.entries[instance.uuid]

        assert payload.task == "mean_estimation"
        assert payload.kind == "instance"
        assert set(record) == {
            "metadata",
            "input_digest",
            "metric_arguments",
            "label",
        }
        assert "uuid" not in record
        assert "task" not in record

    def test_registry_round_trip_restores_typed_metadata_and_domain(
        self, tmp_path: Path
    ) -> None:
        instance = mean_instance(tmp_path)
        path = tmp_path / INSTANCE_REGISTRY_NAME
        write_instance_registry(path, MeanEstimation.id, [instance])

        restored = read_instance_registry(path, MeanEstimation.id)

        assert list(restored) == [instance.uuid]
        restored_instance = restored[instance.uuid]
        assert isinstance(restored_instance, Instance)
        assert isinstance(restored_instance.metadata, MeanEstimationMetadata)
        restored_metadata = restored_instance.metadata
        assert isinstance(restored_metadata.domain, BoxDomain)
        assert canonical_json_bytes(
            build_instance_registry(MeanEstimation.id, restored.values()).model_dump()
        ) == canonical_json_bytes(
            build_instance_registry(MeanEstimation.id, [instance]).model_dump()
        )

    def test_task_and_derived_key_are_relational_checks(self, tmp_path: Path) -> None:
        instance = mean_instance(tmp_path)
        document = build_instance_registry(MeanEstimation.id, [instance]).model_dump()

        with pytest.raises(ValueError, match="is native to task"):
            read_instance_registry(canonical_json_bytes(document), LinearRegression.id)

        document["entries"] = {"i2-wrong": document["entries"][instance.uuid]}
        with pytest.raises(ValueError, match="does not match derived UUID"):
            read_instance_registry(canonical_json_bytes(document), MeanEstimation.id)

    def test_entry_reader_isolates_a_malformed_sibling(self, tmp_path: Path) -> None:
        valid = mean_instance(tmp_path)
        malformed = valid.model_copy(update={"input_digest": "1" * 64})
        document = build_instance_registry(
            MeanEstimation.id, [valid, malformed]
        ).model_dump()
        document["entries"][malformed.uuid] = {"input_digest": "not-a-digest"}

        instances, rejections = load_instances_partitioned(
            read_registry(canonical_json_bytes(document)).entries,
            task_id=MeanEstimation.id,
        )

        assert list(instances) == [valid.uuid]
        assert instances[valid.uuid].uuid == valid.uuid
        assert set(rejections) == {malformed.uuid}
        assert "metadata" in rejections[malformed.uuid]
        with pytest.raises(ValueError) as refusal:
            read_instance_registry(canonical_json_bytes(document), MeanEstimation.id)
        assert malformed.uuid in str(refusal.value)
        assert rejections[malformed.uuid] == str(refusal.value.__cause__)
        assert isinstance(refusal.value.__cause__, ValidationError)

    def test_loading_keeps_a_valid_sibling_and_warns_once(self, tmp_path: Path) -> None:
        valid = mean_instance(tmp_path)
        document = build_instance_registry(MeanEstimation.id, [valid]).model_dump()
        document["entries"]["i2-wrong"] = document["entries"][valid.uuid]
        entries = read_registry(canonical_json_bytes(document)).entries

        with pytest.warns(UserWarning) as recorded:
            loaded = load_instances(entries, task_id=MeanEstimation.id, strict=False)

        assert list(loaded) == [valid.uuid]
        assert len(recorded) == 1
        assert str(recorded[0].message).startswith(
            "Skipped 1 of 2 instance registry entries this library cannot rebuild: "
            "i2-wrong ("
        )
        with pytest.raises(ValueError, match="does not match derived UUID"):
            load_instances(entries, task_id=MeanEstimation.id)

    def test_relational_rejection_states_its_cause_beside_a_valid_sibling(
        self, tmp_path: Path
    ) -> None:
        valid = mean_instance(tmp_path)
        document = build_instance_registry(MeanEstimation.id, [valid]).model_dump()
        document["entries"]["i2-wrong"] = document["entries"][valid.uuid]

        instances, rejections = load_instances_partitioned(
            read_registry(canonical_json_bytes(document)).entries,
            task_id=MeanEstimation.id,
        )

        assert list(instances) == [valid.uuid]
        assert set(rejections) == {"i2-wrong"}
        with pytest.raises(ValueError) as refusal:
            read_instance_registry(canonical_json_bytes(document), MeanEstimation.id)
        assert rejections["i2-wrong"] == str(refusal.value)
        assert "does not match derived UUID" in rejections["i2-wrong"]

    def test_entry_reader_refuses_a_malformed_envelope(self) -> None:
        payload: dict[str, Any] = {
            "task": MeanEstimation.id,
            "kind": "instance",
            "entries": [],
        }

        with pytest.raises(ValidationError, match="entries"):
            read_registry(canonical_json_bytes(payload))

    def test_metadata_from_another_task_is_refused(self, tmp_path: Path) -> None:
        metadata = LinearRegressionMetadata(
            n_samples=2,
            design_domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
            response_domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
        )
        input = LinearRegressionInput(
            X=np.array([[0.0], [1.0]]), y=np.array([0.0, 1.0])
        )
        instance = persist_instance(metadata, input, {}, tmp_path)
        document = build_instance_registry(LinearRegression.id, [instance]).model_dump()
        document["task"] = MeanEstimation.id

        with pytest.raises(ValueError, match="is native to task 'linear_regression'"):
            read_instance_registry(canonical_json_bytes(document), MeanEstimation.id)

    def test_reconciliation_deduplicates_identical_instances(
        self, tmp_path: Path
    ) -> None:
        instance = mean_instance(tmp_path)

        assert build_instance_registry(
            MeanEstimation.id, [instance, instance]
        ) == build_instance_registry(MeanEstimation.id, [instance])

    def test_reconciliation_allows_reusing_one_advisory_label(
        self, tmp_path: Path
    ) -> None:
        instance = mean_instance(tmp_path, label="shared")
        other = instance.model_copy(update={"input_digest": "1" * 64})

        registry = build_instance_registry(MeanEstimation.id, [instance, other])

        assert set(registry.entries) == {instance.uuid, other.uuid}

    @pytest.mark.parametrize(
        "route", ["build-registry", "register-instance", "add-entry"]
    )
    def test_one_identity_never_files_two_different_records(
        self, tmp_path: Path, route: str
    ) -> None:
        """Filing compares the whole record, not just the key it is filed under.

        Edited references keep the problem's identity, so they are edited in
        place rather than filed twice; an identity is a shortened digest, so
        a colliding second record under the first's key is refused too.
        """
        instance = mean_instance(tmp_path)
        corrected = instance.model_copy(
            update={"metric_arguments": {"reference": [0.5, 0.8]}}
        )
        assert corrected.uuid == instance.uuid
        colliding = instance.model_copy(update={"input_digest": "1" * 64})
        register_instance(tmp_path, instance, registry="toy")
        registry = add_entry(
            JSONRegistry(task=MeanEstimation.id, kind="instance", entries={}),
            uuid=instance.uuid,
            record=instance.record(),
        )

        with pytest.raises(ValueError, match="different"):
            if route == "build-registry":
                build_instance_registry(MeanEstimation.id, [instance, corrected])
            elif route == "register-instance":
                register_instance(tmp_path, corrected, registry="toy")
            else:
                add_entry(registry, uuid=instance.uuid, record=colliding.record())
