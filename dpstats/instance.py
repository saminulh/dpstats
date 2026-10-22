"""Portable benchmark Instances and their exact Input artifacts."""

from __future__ import annotations

from typing import Any, Generic
from functools import cached_property

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from typing_extensions import Self

from dpstats.transport.arrays import (
    Array,
    named_arrays_digest,
    read_named_arrays,
    verify_named_arrays,
    write_named_arrays,
)
from dpstats.transport.configuration import (
    DUMP_CONFIG_CONTEXT,
    LOAD_CONFIG_CONTEXT,
    evict_undeclared_copy_state,
    identity_payload,
)
from dpstats.workspace.layout import (
    input_artifact_path,
    task_collection_directory,
)
from dpstats.transport.documents import write_json_atomic
from dpstats.transport.registry import (
    JSONRegistry,
    add_entry,
    load_entries,
    partition_verified_entries,
    read_registry,
)
from dpstats.workspace.registries import file_registry_entry
from dpstats.transport.canonical import (
    canonical_json_bytes,
    instance_identity,
)
from dpstats.transport.validators import Sha256Hex
from dpstats.metric import RESERVED_REQUIREMENTS
from dpstats.task import MetadataT, Task, TaskInput, TaskMetadata


def _input_arrays(input: TaskInput) -> dict[str, Array]:
    arrays: dict[str, Array] = {}
    for name, value in input:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Stored Input field {name!r} must be an ndarray, not "
                f"{type(value).__name__}."
            )
        arrays[name] = value
    return arrays


class Instance(BaseModel, Generic[MetadataT]):
    """A task instance with metadata, input data, and metric arguments.

    The Instance object stores metadata and source-supplied metric arguments. The input data
    belongs to the instance but is stored separately; load it with
    collection.load_input(instance).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        polymorphic_serialization=True,
        strict=True,
        validate_default=True,
    )

    metadata: MetadataT = Field(description="The Task's public Instance specification.")
    input_digest: Sha256Hex = Field(
        description="Digest of this Instance's persisted private Input."
    )
    metric_arguments: dict[str, JsonValue] = Field(
        description="Stored post-run values a Metric may require, keyed by name."
    )
    label: str | None = Field(
        default=None,
        description="Optional label; never part of this Instance's identity.",
    )

    @property
    def task(self) -> Task:
        """Return the task selected by this Instance's concrete Metadata."""
        return self.metadata.task

    def __str__(self) -> str:
        """Return this Instance's label, or its UUID when unlabelled."""
        return self.label or self.uuid

    @cached_property
    def uuid(self) -> str:
        """Return this Instance's content identity.

        The identity covers the portable problem — this Metadata and Input
        digest — so neither a label a registry assigns nor the
        evaluation-side metric arguments change it. It is derived on first
        read and reused for the frozen value's life; a record mutated
        afterwards is refused at load, where the stored key does not match
        the derived identity.
        """
        return instance_identity(
            identity_payload(self, exclude={"label", "metric_arguments"})
        )

    def __eq__(self, other: object) -> bool:
        """Compare complete portable fields, including array-valued Metadata.

        Labels do not affect equality. Metric arguments remain part of
        equality even though they do not affect the problem's UUID. The record
        supplies structural values for arrays and retains each configured
        Metadata component's class.
        """
        if self is other:
            return True
        if not isinstance(other, Instance) or type(self) is not type(other):
            return NotImplemented
        return self._comparison_record == other._comparison_record

    @cached_property
    def _comparison_record(self) -> dict[str, Any]:
        """Derive this frozen value's structural comparison without labels."""
        return identity_payload(self, exclude={"label"})

    def __hash__(self) -> int:
        # Hash the content identity, not the fields: a mapping among them has
        # no field-wise hash, and equal fields derive one identity anyway, so
        # this stays consistent with field-wise equality.
        return hash(self.uuid)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy this Instance without carrying the original's cached identity."""
        copied = super().model_copy(update=update, deep=deep)
        evict_undeclared_copy_state(copied)
        return copied

    def record(self) -> dict[str, Any]:
        """Return this Instance's portable fields with configured Metadata."""
        return self.model_dump(mode="python", context={DUMP_CONFIG_CONTEXT: True})

    @classmethod
    def load_config(cls, config: dict[str, Any]) -> Self:
        """Validate one portable record, restoring its configured Metadata."""
        return cls.model_validate(config, context={LOAD_CONFIG_CONTEXT: True})

    @field_validator("metric_arguments")
    @classmethod
    def _validate_metric_arguments(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        reserved = sorted(RESERVED_REQUIREMENTS.intersection(value))
        if reserved:
            names = ", ".join(repr(name) for name in reserved)
            raise ValueError(
                f"Instance.metric_arguments cannot use the reserved key {names}."
            )
        canonical_json_bytes(value)
        return value

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Instance.label must be nonempty when supplied.")
        return value


def _materialize_input(instance: Instance, arrays: Mapping[str, Array]) -> TaskInput:
    input = instance.task.Input.model_validate(arrays)
    instance.metadata.validate_input(input)
    return input


def identify_instance(
    metadata: TaskMetadata,
    input: TaskInput,
    metric_arguments: Mapping[str, Any],
    *,
    label: str | None = None,
) -> Instance:
    """Identify an already validated, loaded Input without persisting it.

    Args:
        metadata: The Task's public Instance specification.
        input: The exact private Input this Instance names.
        metric_arguments: Stored post-run values a Metric may
            require, keyed by name.
        label: Optional label; never part of this Instance's
            identity.

    Returns:
        The constructed, data-free Instance, unwritten to any store.

    Example:
        >>> from dpstats.simulations.linear_regression import simulate_clipped_gaussian
        >>> metadata, input, refs = simulate_clipped_gaussian(seed=0)
        >>> identify_instance(metadata, input, refs).metric_arguments == refs
        True
    """
    return Instance(
        metadata=metadata,
        input_digest=named_arrays_digest(_input_arrays(input)),
        metric_arguments=dict(metric_arguments),
        label=label,
    )


def persist_instance(
    metadata: TaskMetadata,
    input: TaskInput,
    metric_arguments: Mapping[str, Any],
    input_artifacts_root: str | Path,
    *,
    label: str | None = None,
) -> Instance:
    """Persist one exact Input in the shared artifact root and construct it.

    Validates ``input`` against ``metadata``'s public contract, writes its
    arrays into ``input_artifacts_root`` keyed by their content digest, and
    returns the resulting data-free Instance. Register it in a task
    collection to make it selectable by a Sweep.

    Args:
        metadata: The Task's public Instance specification.
        input: The exact private Input this Instance names.
        metric_arguments: Stored post-run values a Metric may
            require, keyed by name.
        input_artifacts_root: Shared root to persist the Input's arrays
            under.
        label: Optional label; never part of this Instance's
            identity.

    Returns:
        The constructed, data-free Instance.

    Raises:
        ValueError: If ``input`` does not satisfy ``metadata``'s public
            contract.

    Example:
        >>> import tempfile
        >>> from dpstats.simulations.linear_regression import simulate_clipped_gaussian
        >>> metadata, input, refs = simulate_clipped_gaussian(seed=0)
        >>> with tempfile.TemporaryDirectory() as input_artifacts_root:
        ...     instance = persist_instance(
        ...         metadata, input, refs, input_artifacts_root, label="gaussian-d2"
        ...     )
        >>> instance.label
        'gaussian-d2'
    """
    metadata.validate_input(input)
    instance = identify_instance(
        metadata,
        input,
        metric_arguments,
        label=label,
    )
    arrays = _input_arrays(input)
    write_named_arrays(
        arrays,
        input_artifact_path(input_artifacts_root, instance.input_digest),
        digest=instance.input_digest,
    )
    return instance


def load_instance_input(instance: Instance, input_artifacts_root: Path) -> TaskInput:
    """Load and validate one Instance's exact private Input.

    ``input_artifacts_root`` is a trusted store whose artifacts are named by the
    logical digest of their own contents, so the stored bytes are read as
    found. Decode and Task contract failures still refuse. Call
    ``verify_instance_input`` instead for an artifact arriving from outside
    such a store.

    Args:
        instance: The data-free Instance whose persisted Input is read.
        input_artifacts_root: The trusted root the Input's arrays live under.

    Returns:
        The decoded, Task-validated Input this Instance refers to.

    Raises:
        ValueError: If the decoded Input does not satisfy this Instance's
            Task contract.

    Example:
        >>> import tempfile
        >>> from dpstats.simulations.linear_regression import simulate_clipped_gaussian
        >>> metadata, input, refs = simulate_clipped_gaussian(seed=0)
        >>> with tempfile.TemporaryDirectory() as root:
        ...     instance = persist_instance(metadata, input, refs, root)
        ...     load_instance_input(instance, root).X.shape == input.X.shape
        True
    """
    return _materialize_input(
        instance,
        read_named_arrays(
            input_artifact_path(input_artifacts_root, instance.input_digest)
        ),
    )


def verify_instance_input(instance: Instance, input_artifacts_root: Path) -> None:
    """Refuse one untrusted Input artifact that its own digest does not name.

    Contents arriving from outside a trusted store carry no guarantee that
    they are what the file name claims, so they are hashed here before this
    Instance's Task contract is applied to them.

    Args:
        instance: The data-free Instance whose Input artifact is checked.
        input_artifacts_root: The untrusted root the Input's arrays live
            under.

    Raises:
        ValueError: If the artifact's bytes do not hash to this Instance's
            digest, or the decoded Input does not satisfy its Task contract.

    Example:
        >>> import tempfile
        >>> from dpstats.simulations.linear_regression import simulate_clipped_gaussian
        >>> metadata, input, refs = simulate_clipped_gaussian(seed=0)
        >>> with tempfile.TemporaryDirectory() as root:
        ...     instance = persist_instance(metadata, input, refs, root)
        ...     verify_instance_input(instance, root)
    """
    _materialize_input(
        instance,
        verify_named_arrays(
            input_artifact_path(input_artifacts_root, instance.input_digest),
            instance.input_digest,
        ),
    )


def register_instance(
    task_collections_root: str | Path,
    instance: Instance,
    *,
    registry: str = "default",
) -> str:
    """Idempotently file one direct Instance in a named registry group.

    Args:
        task_collections_root: Exact root whose direct children are task
            collections.
        instance: The data-free Instance to register.
        registry: The named group to file it in.

    Returns:
        The registered Instance's UUID, the identifier a cell key addresses
        it by.

    Example:
        >>> import tempfile
        >>> from dpstats.simulations.linear_regression import simulate_clipped_gaussian
        >>> metadata, input, refs = simulate_clipped_gaussian(seed=0)
        >>> instance = identify_instance(metadata, input, refs)
        >>> with tempfile.TemporaryDirectory() as task_collections_root:
        ...     instance_uuid = register_instance(
        ...         task_collections_root, instance, registry="mine"
        ...     )
        >>> isinstance(instance_uuid, str)
        True
    """
    file_registry_entry(
        task_collection_directory(task_collections_root, instance.task.id),
        task_id=instance.task.id,
        kind="instance",
        uuid=instance.uuid,
        record=instance.record(),
        registry=registry,
    )
    return instance.uuid


def load_instances(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    strict: bool = True,
    sources: Mapping[str, Path] | None = None,
) -> dict[str, Instance]:
    """Rebuild the direct Instances one registry's entries name.

    An entry this library cannot rebuild refuses the group; ``strict=False``
    skips it with one warning naming it instead. ``sources`` maps an entry to
    the registry file holding it, so a refusal or warning names that file.
    """
    return load_entries(
        entries,
        kind="instance",
        task_id=task_id,
        strict=strict,
        reconstruct=Instance.load_config,
        derive_uuid=lambda instance: instance.uuid,
        native_task_id=lambda instance: instance.task.id,
        sources=sources,
    )


def load_instances_partitioned(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Instance], dict[str, str]]:
    """Load valid Instances and every rejected key's stated cause.

    The reporting counterpart of ``load_instances``.
    """
    return partition_verified_entries(
        entries,
        task_id=task_id,
        reconstruct=Instance.load_config,
        derive_uuid=lambda instance: instance.uuid,
        native_task_id=lambda instance: instance.task.id,
        sources=sources,
    )


def read_instance_registry(source: bytes | Path, task_id: str) -> dict[str, Instance]:
    """Strictly validate one task-local registry into direct Instances."""
    return load_instances(read_registry(source).entries, task_id=task_id, strict=True)


def build_instance_registry(
    task_id: str, instances: Iterable[Instance]
) -> JSONRegistry:
    """Build one task-local registry keyed by Instance UUID."""
    registry = JSONRegistry(task=task_id, kind="instance", entries={})
    for instance in instances:
        registry = add_entry(registry, uuid=instance.uuid, record=instance.record())
    return registry


def write_instance_registry(
    path: Path, task_id: str, instances: Iterable[Instance]
) -> None:
    """Write one task-local Instance registry atomically."""
    write_json_atomic(path, build_instance_registry(task_id, instances).model_dump())


__all__ = [
    "Instance",
]
