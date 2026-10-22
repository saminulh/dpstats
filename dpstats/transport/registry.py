"""Registry-document algebra: the named per-task, per-kind entry files.

A registry document records portable entries of one kind for one task, each
filed under the content identity its record derives. Everything here works on
documents and entries alone — reading, merging, selecting, adding, and
rebuilding entries verified against the key and task they are filed under. No
live class is named here: each kind binds the reconstructor and its task
accessor beside the class they serve. Which registry documents a task
collection holds, and where a named group is filed, is
``dpstats.workspace.registries``' concern, not this module's: it
knows no path, root, or directory name.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)


TEntry = TypeVar("TEntry")


class JSONRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    kind: Literal["algorithm", "instance"]
    entries: dict[str, dict[str, Any]]


def read_registry(source: bytes | Path) -> JSONRegistry:
    """Return the envelope one registry declares and the entries it holds.

    ``source`` is the bytes to decode, or a path this reads directly (byte
    I/O only: this derives no path from a task root or layout name). Bytes
    that will not decode raise the one failure that cannot say where it came
    from; a path source names itself in its own exception instead. Which task
    and kind a registry declares travels back as the data it is: an entry is
    carried as the record it is, and whether these entries are the ones a
    caller wanted is answered where they are rebuilt.
    """
    return JSONRegistry.model_validate(decode_json_document(source))


def selected_ids(
    requested: Sequence[str] | None,
    available: Collection[str],
    kind: str,
    reasons: Callable[[], Mapping[str, str]] | None = None,
) -> set[str]:
    """Return the requested installed identities, refusing unknown values.

    A refusal names any unknown value ``reasons`` says was skipped as unloadable.
    """
    if requested is None:
        return set(available)
    unique = set(requested)
    unknown = unique - set(available)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"Selected {kind} id(s) are outside the registry scope: {names}."
            + ("" if reasons is None else skipped_note(unknown, reasons))
        )
    return unique


def merge_registries(
    registries: Collection[JSONRegistry],
    *,
    reconstruct: Callable[[dict[str, Any]], TEntry] | None = None,
    derive_uuid: Callable[[TEntry], str] | None = None,
    native_task_id: Callable[[TEntry], str] | None = None,
) -> JSONRegistry:
    """Union UUID-keyed entries under one record per UUID.

    Non-colliding and byte-identical records require no reconstruction.
    With all three reconstruction/identity callables, a byte-different pair
    filed under one UUID is rebuilt, verified, and compared by value equality.
    The first record is retained whole, including its label. Without these
    callables, any byte difference under the same UUID refuses.

    Labels may repeat across distinct UUIDs. No source registry is rewritten.
    """
    if len(registries) == 0:
        raise ValueError("At least one registry must be provided to merge.")
    task_ids = {registry.task for registry in registries}
    if len(task_ids) != 1:
        raise ValueError(
            "All registries must declare the same task, but these differ: "
            f"{', '.join(sorted(task_ids))}."
        )
    task_id = next(iter(task_ids))
    kinds: set[Literal["algorithm", "instance"]] = {
        registry.kind for registry in registries
    }
    if len(kinds) != 1:
        raise ValueError(
            "All registries must declare the same kind, but these differ: "
            f"{', '.join(sorted(kinds))}."
        )
    kind = kinds.pop()
    entries: dict[str, dict[str, Any]] = {}
    for registry in registries:
        for uuid, entry in registry.entries.items():
            if uuid not in entries:
                entries[uuid] = entry
                continue
            if canonical_json_bytes(entries[uuid]) == canonical_json_bytes(entry):
                continue
            if (
                reconstruct is not None
                and derive_uuid is not None
                and native_task_id is not None
            ):
                if _records_agree(
                    uuid,
                    entries[uuid],
                    entry,
                    task_id=task_id,
                    reconstruct=reconstruct,
                    derive_uuid=derive_uuid,
                    native_task_id=native_task_id,
                ):
                    continue
            raise ValueError(
                f"The registry UUID {uuid!r} is already recorded with a "
                "different entry."
            )
    return JSONRegistry(task=task_id, kind=kind, entries=entries)


def _records_agree(
    uuid: str,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    task_id: str,
    reconstruct: Callable[[dict[str, Any]], TEntry],
    derive_uuid: Callable[[TEntry], str],
    native_task_id: Callable[[TEntry], str],
) -> bool:
    """Rebuild one colliding UUID's two records and compare them by value.

    Reconstruction failure still refuses: an unreconstructable record is not
    known to agree with anything.
    """
    try:
        first_value = reconstruct(first)
        second_value = reconstruct(second)
    except Exception as error:
        raise ValueError(
            f"Could not reconstruct the registry entry {uuid!r}."
        ) from error
    first_value = _verify_filed_entry(
        uuid,
        first_value,
        task_id=task_id,
        derive_uuid=derive_uuid,
        native_task_id=native_task_id,
    )
    second_value = _verify_filed_entry(
        uuid,
        second_value,
        task_id=task_id,
        derive_uuid=derive_uuid,
        native_task_id=native_task_id,
    )
    return bool(first_value == second_value)


def _entry_name(uuid: str, sources: Mapping[str, Path] | None) -> str:
    """Name one entry by its key and, when known, the registry file holding it."""
    source = None if sources is None else sources.get(uuid)
    return repr(uuid) if source is None else f"{uuid!r} in {source}"


def _verify_filed_entry(
    uuid: str,
    value: TEntry,
    *,
    task_id: str,
    derive_uuid: Callable[[TEntry], str],
    native_task_id: Callable[[TEntry], str],
    sources: Mapping[str, Path] | None = None,
) -> TEntry:
    """Verify the key and task one rebuilt entry is filed under.

    Every refusal names the entry it is about, and the registry file holding
    it when ``sources`` records one.
    """
    derived = derive_uuid(value)
    if derived != uuid:
        raise ValueError(
            f"The registry entry {_entry_name(uuid, sources)} does not match "
            f"derived UUID {derived!r}."
        )
    native = native_task_id(value)
    if native != task_id:
        raise ValueError(
            f"The registry entry {_entry_name(uuid, sources)} is native to task "
            f"{native!r}, not {task_id!r}."
        )
    return value


def load_verified_entries(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    reconstruct: Callable[[dict[str, Any]], TEntry],
    derive_uuid: Callable[[TEntry], str],
    native_task_id: Callable[[TEntry], str],
    sources: Mapping[str, Path] | None = None,
) -> dict[str, TEntry]:
    """Rebuild every entry in key order; one refused entry refuses the group.

    ``reconstruct`` rebuilds the value one record names; ``derive_uuid`` and
    ``native_task_id`` read the rebuilt value's own identity and native task.
    A value refusal names its own field and offending value but not the entry
    holding them, so this is the layer that names the entry, and the registry
    file ``sources`` records for it.
    """
    loaded: dict[str, TEntry] = {}
    for uuid in sorted(entries):
        try:
            value = reconstruct(entries[uuid])
        except Exception as error:
            raise ValueError(
                "Could not reconstruct the registry entry "
                f"{_entry_name(uuid, sources)}."
            ) from error
        loaded[uuid] = _verify_filed_entry(
            uuid,
            value,
            task_id=task_id,
            derive_uuid=derive_uuid,
            native_task_id=native_task_id,
            sources=sources,
        )
    return loaded


def partition_verified_entries(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    reconstruct: Callable[[dict[str, Any]], TEntry],
    derive_uuid: Callable[[TEntry], str],
    native_task_id: Callable[[TEntry], str],
    sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, TEntry], dict[str, str]]:
    """Rebuild what the entries hold and report every rejected key's cause.

    The reporting counterpart of ``load_verified_entries``: valid siblings are
    kept, and each reason is the refusal's own message. Rejections are keyed by
    the entry they concern, so a value refusal is reported unwrapped.
    """
    loaded: dict[str, TEntry] = {}
    rejections: dict[str, str] = {}
    for uuid in sorted(entries):
        try:
            value = reconstruct(entries[uuid])
        except Exception as rejection:
            rejections[uuid] = str(rejection)
            continue
        try:
            loaded[uuid] = _verify_filed_entry(
                uuid,
                value,
                task_id=task_id,
                derive_uuid=derive_uuid,
                native_task_id=native_task_id,
                sources=sources,
            )
        except ValueError as rejection:
            rejections[uuid] = str(rejection)
    return loaded, rejections


def load_entries(
    entries: Mapping[str, dict[str, Any]],
    *,
    kind: str,
    task_id: str,
    strict: bool,
    reconstruct: Callable[[dict[str, Any]], TEntry],
    derive_uuid: Callable[[TEntry], str],
    native_task_id: Callable[[TEntry], str],
    sources: Mapping[str, Path] | None = None,
) -> dict[str, TEntry]:
    """Rebuild the entries, keeping the valid ones unless ``strict``.

    ``strict`` refuses the group on a rejection; otherwise one warning names
    every rejected key and its reason. ``sources`` maps an entry to the
    registry file holding it, so a refusal or warning names that file.
    """
    if strict:
        return load_verified_entries(
            entries,
            task_id=task_id,
            reconstruct=reconstruct,
            derive_uuid=derive_uuid,
            native_task_id=native_task_id,
            sources=sources,
        )
    loaded, rejections = partition_verified_entries(
        entries,
        task_id=task_id,
        reconstruct=reconstruct,
        derive_uuid=derive_uuid,
        native_task_id=native_task_id,
        sources=sources,
    )
    if rejections:
        named = ", ".join(
            f"{uuid} ({_first_line(reason)})" for uuid, reason in rejections.items()
        )
        warnings.warn(
            f"Skipped {len(rejections)} of {len(entries)} {kind} registry entries "
            f"this library cannot rebuild: {named}",
            UserWarning,
            stacklevel=3,
        )
    return loaded


def _first_line(reason: str) -> str:
    return (reason.splitlines() or [""])[0]


def skipped_note(
    names: Collection[str], reasons: Callable[[], Mapping[str, str]]
) -> str:
    """Say which of *names* were skipped as unloadable, with each reason.

    ``reasons`` is read only when a refusal needs it.
    """
    rejected = reasons()
    hit = sorted(set(names) & set(rejected))
    if not hit:
        return ""
    named = "; ".join(f"{uuid} ({_first_line(rejected[uuid])})" for uuid in hit)
    return f" Skipped as unloadable: {named}."


def add_entry(
    registry: JSONRegistry, *, uuid: str, record: dict[str, Any]
) -> JSONRegistry:
    """Idempotently add one entry under the registry's own task and kind."""
    addition = JSONRegistry(
        task=registry.task,
        kind=registry.kind,
        entries={uuid: record},
    )
    return merge_registries([registry, addition])
