"""Which registry documents a task collection holds, and how a group is filed.

A task's ``{kind}_registries/`` directory holds one JSON document per named
group; ``layout`` supplies those directory and file names, and this module
resolves, reads, merges, and files them. The document algebra itself —
merging, verifying, rebuilding entries — is
``dpstats.transport.registry``'s concern; this module is the one
place that turns a task root and a selection into exact files on disk.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeVar

from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.documents import STAGING_PREFIX, write_json_atomic
from dpstats.transport.registry import (
    JSONRegistry,
    merge_registries,
    read_registry,
)
from dpstats.transport.validators import is_portable_name
from dpstats.workspace.layout import registry_directory_name


TEntry = TypeVar("TEntry")


def registry_document_path(task_root: Path, kind: str, name: str) -> Path:
    """Return one validated named registry document's local file path."""
    _validate_registry_name(name)
    return task_root / registry_directory_name(kind) / f"{name}.json"


def authoring_registry_path(task_root: Path, kind: str, name: str) -> Path:
    """Return one safe local target for an atomic named-group update."""
    path = registry_document_path(task_root, kind, name)
    directory = path.parent
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise ValueError(
            f"Registry directory {directory} must be a non-symlink directory."
        )
    if path.is_symlink():
        raise ValueError(f"Registry file {path} must not be a symlink.")
    return path


def select_registry_paths(
    task_root: Path,
    kind: str,
    names: str | Sequence[str] | None,
    *,
    ignore_staging: bool = False,
) -> tuple[Path, ...]:
    """Resolve the exact installed named registries for one invocation.

    ``ignore_staging`` leaves an interrupted publication's staging files out
    of an all-documents selection, for an operator that reports them itself;
    any other stray entry still refuses.
    """
    directory = task_root / registry_directory_name(kind)
    if names is None:
        return _installed_registry_paths(directory, ignore_staging=ignore_staging)
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"Registry directory {directory} is not installed.")
    if isinstance(names, bytes):
        raise TypeError(f"{kind}_registries must be a name or sequence of names.")
    requested = (names,) if isinstance(names, str) else tuple(names)
    if not requested:
        raise ValueError(f"{kind}_registries cannot be empty.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{kind}_registries cannot repeat a registry name.")
    selected: list[Path] = []
    for name in requested:
        path = registry_document_path(task_root, kind, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"The {kind} registry {name!r} is not installed for this Task."
            )
        selected.append(path)
    return tuple(selected)


def _installed_registry_paths(
    directory: Path, *, ignore_staging: bool = False
) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"Registry directory {directory} is not installed.")
    paths = tuple(
        sorted(
            path
            for path in directory.iterdir()
            if not (ignore_staging and path.name.startswith(STAGING_PREFIX))
        )
    )
    if not paths:
        raise ValueError(f"Registry directory {directory} is empty.")
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ValueError(f"Registry directory {directory} contains {path.name!r}.")
        _validate_registry_name(path.stem)
    return paths


def _validate_registry_name(name: Any) -> None:
    if not is_portable_name(name) or name.endswith(".json"):
        raise ValueError(
            "A registry name must be a portable direct name without the .json suffix."
        )


def read_selection(
    task_root: Path, kind: str, names: str | Sequence[str] | None
) -> JSONRegistry:
    """Read and merge one invocation's exact selected registry documents.

    ``names`` selects one name or an exact named subset; ``None`` selects every installed
    document of the kind. Selection fails closed: a missing, empty, or
    malformed selection refuses rather than widening.
    """
    return merge_registries(
        [read_registry(path) for path in select_registry_paths(task_root, kind, names)]
    )


def _refuse_no_installed_registries(task_roots: Sequence[Path], kind: str) -> NoReturn:
    """Refuse naming every tree checked, the Task, and the layout expected.

    ``task_roots`` are each a Task's own collection directory (``<tree>/<task>``),
    so the Task id and every checked tree are read straight off them.
    """
    if not task_roots:
        raise ValueError(f"No installed tree was given to select {kind} registries.")
    task_id = task_roots[0].name
    directory_name = registry_directory_name(kind)
    trees = ", ".join(str(root.parent) for root in task_roots)
    expected = ", ".join(str(root / directory_name) for root in task_roots)
    raise ValueError(
        f"No {directory_name} directory is installed for Task {task_id!r} in "
        f"{trees}; expected {expected}."
    )


def installed_registry_names(task_roots: Sequence[Path], kind: str) -> tuple[str, ...]:
    """Return the sorted names of every installed ``kind`` registry, or ``()``.

    Empty exactly when no tree has a ``{kind}_registries/`` directory; a
    present but malformed directory refuses as ``select_registry_paths_multi``
    does.
    """
    if not any((root / registry_directory_name(kind)).is_dir() for root in task_roots):
        return ()
    return tuple(
        sorted(
            {path.stem for path in select_registry_paths_multi(task_roots, kind, None)}
        )
    )


def select_registry_paths_multi(
    task_roots: Sequence[Path], kind: str, names: str | Sequence[str] | None
) -> tuple[Path, ...]:
    """Resolve one invocation's exact installed registries across several trees.

    A named group is filed under the same name in each tree that holds it
    (``mine``/``official``, for example), so this unions the trees rather
    than requiring every root to carry every name. A tree with no ``{kind}
    _registries/`` directory at all is skipped entirely; a name requested but
    absent from every tree that does have the directory still refuses.
    """
    existing = [
        root for root in task_roots if (root / registry_directory_name(kind)).is_dir()
    ]
    if not existing:
        _refuse_no_installed_registries(task_roots, kind)
    if names is None:
        paths: list[Path] = []
        for root in existing:
            paths.extend(
                _installed_registry_paths(root / registry_directory_name(kind))
            )
        return tuple(paths)
    if isinstance(names, bytes):
        raise TypeError(f"{kind}_registries must be a name or sequence of names.")
    requested = (names,) if isinstance(names, str) else tuple(names)
    if not requested:
        raise ValueError(f"{kind}_registries cannot be empty.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{kind}_registries cannot repeat a registry name.")
    selected: list[Path] = []
    for name in requested:
        _validate_registry_name(name)
    missing = set(requested)
    for root in existing:
        for name in requested:
            path = registry_document_path(root, kind, name)
            if path.is_symlink() or not path.is_file():
                continue
            selected.append(path)
            missing.discard(name)
    for name in requested:
        if name in missing:
            raise ValueError(
                f"The {kind} registry {name!r} is not installed for this Task."
            )
    return tuple(selected)


def read_selection_multi(
    task_roots: Sequence[Path],
    kind: str,
    names: str | Sequence[str] | None,
    *,
    reconstruct: Callable[[dict[str, Any]], TEntry] | None = None,
    derive_uuid: Callable[[TEntry], str] | None = None,
    native_task_id: Callable[[TEntry], str] | None = None,
) -> tuple[JSONRegistry, dict[str, Path]]:
    """Read and merge one invocation's exact selected registries across several trees.

    Same selection and fail-closed rules as ``read_selection``, applied to the
    union of trees. ``reconstruct``/``derive_uuid``/``native_task_id`` are
    ``merge_registries``'s own reconciling path: without them, a UUID filed
    with byte-differing content in two trees always refuses; with them, only
    a genuine (post-reconstruction) value difference does.

    Returns the merged registry beside the file each entry was read from: the
    first file filing a UUID, the same record the merge retains.
    """
    registries: list[JSONRegistry] = []
    sources: dict[str, Path] = {}
    for path in select_registry_paths_multi(task_roots, kind, names):
        registry = read_registry(path)
        registries.append(registry)
        for uuid in registry.entries:
            sources.setdefault(uuid, path)
    merged = merge_registries(
        registries,
        reconstruct=reconstruct,
        derive_uuid=derive_uuid,
        native_task_id=native_task_id,
    )
    return merged, sources


def file_registry_entry(
    task_root: str | Path,
    *,
    task_id: str,
    kind: Literal["algorithm", "instance"],
    uuid: str,
    record: dict[str, Any],
    registry: str = "default",
) -> None:
    """File one entry; an already-filed UUID must match it exactly.

    Args:
        task_root: The one Task's own collection directory (already resolved
            by the caller: this module knows registry naming, not the
            broader workspace tree shape ``layout`` owns).
        task_id: The Task the filed entry is native to.
        kind: Which family of registry this entry belongs to.
        uuid: The entry's derived identity, filed as its key.
        record: The entry's own configuration payload.
        registry: The named group it joins.
    """
    path = authoring_registry_path(Path(task_root), kind, registry)
    existing = (
        read_registry(path)
        if path.is_file()
        else JSONRegistry(task=task_id, kind=kind, entries={})
    )
    previous = existing.entries.get(uuid)
    if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(
        record
    ):
        raise ValueError(
            f"The registry UUID {uuid!r} is already recorded with a different entry."
        )
    updated = JSONRegistry(
        task=task_id, kind=kind, entries={**existing.entries, uuid: record}
    )
    write_json_atomic(path, updated.model_dump())
