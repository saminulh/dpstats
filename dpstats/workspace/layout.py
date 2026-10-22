"""Workspace and bundle path vocabulary: the on-disk shape of the world.

A workspace is any directory holding ``local/`` (the user's own store; every
write lands here) and optionally ``official-release/`` (a fetched Distribution
Release, byte for byte, never written after fetch). Both trees share the same
shape: a task id sits directly at the tree root and owns
``instance_registries/``, ``algorithm_registries/``, and ``sweeps/``; a task
directory in ``local/`` additionally owns ``attempts/`` and ``claims/``, and
one in ``official-release/`` instead owns ``results/`` and the sanctioned
views. Both trees have a sibling ``input-artifacts/`` root, so a bare
``official-release/`` directory (a Distribution Release bundle) is the same
bytes read either way.

Because a task id sits directly at a tree's root, it may not collide with a
root-level name: ``input-artifacts``, ``benchmark-info`` and ``release.json``
are reserved and refused as task ids wherever a task id is validated.

This module is the sole home of that path vocabulary; ``collection.py`` and
every application/site consumer import it, never joining directory names
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from pathlib import Path

from dpstats.transport.validators import is_portable_name

INPUT_ARTIFACTS_DIRECTORY = "input-artifacts"
LOCAL_DIRECTORY = "local"
OFFICIAL_RELEASE_DIRECTORY = "official-release"
BENCHMARK_INFO_DIRECTORY = "benchmark-info"
ALGORITHM_DECLARATIONS_NAME = "algorithm-declarations.csv"
PROVENANCE_DIRECTORY = "provenance"
ATTEMPTS_DIRECTORY = "attempts"
RESULTS_DIRECTORY = "results"
PARTITION_PREFIX = "alg_id="
PARTITION_FILE_SUFFIX = ".parquet"
INPUT_ARTIFACT_SUFFIX = ".npz"
CLAIMS_DIRECTORY = "claims"
SWEEPS_DIRECTORY = "sweeps"
WITHHELD_PROFILES_NAME = "withheld-profiles.parquet"
INSTANCE_WEIGHTS_NAME = "instance-weights.csv"
RELEASE_MANIFEST_NAME = "release.json"
BENCHMARK_PROTOCOL_NAME = "benchmark-protocol.json"
BENCHMARK_NARRATIVE_NAME = "benchmark-protocol.md"
LANE_MAP_NAME = "lanes.csv"
INSTANCE_REGISTRY_NAME = "instances.json"

# Root-level names inside a tree (``local/`` or ``official-release/``): a task
# id sitting directly at the root may not collide with one of these.
RESERVED_TASK_NAMES = frozenset(
    {
        INPUT_ARTIFACTS_DIRECTORY,
        BENCHMARK_INFO_DIRECTORY,
        RELEASE_MANIFEST_NAME,
    }
)


def is_valid_task_id(task_id: Any) -> bool:
    """Return whether *task_id* is a portable name and not a reserved root name."""
    return is_portable_name(task_id) and task_id not in RESERVED_TASK_NAMES


def init_workspace(path: str | Path) -> Path:
    """Create ``<path>/local/``, idempotently, and return *path*.

    A workspace is any directory holding ``local/``. This creates that
    directory (and *path* itself) if absent and does nothing further if a
    workspace already stands there.

    Args:
        path: The workspace root to create or reuse.

    Returns:
        *path*, unchanged.

    Raises:
        ValueError: If *path* exists and is not a directory, or is a bare
            Distribution Release bundle (``release.json`` with no ``local/``)
            rather than a workspace.

    Example:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     workspace = init_workspace(Path(root) / "ws")
        ...     workspace_local_root(workspace).is_dir()
        True
    """
    root = Path(path)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Workspace path {root} must be a non-symlink directory.")
    local, release = workspace_roots(root)
    if local is None and release is not None:
        raise ValueError(
            f"{root} is a bare Distribution Release bundle, not a workspace; "
            "open it directly with open_collection instead of init_workspace."
        )
    workspace_local_root(root).mkdir(parents=True, exist_ok=True)
    return root


def workspace_local_root(workspace: str | Path) -> Path:
    """Return a workspace's own store tree, where every write lands."""
    return Path(workspace) / LOCAL_DIRECTORY


def workspace_release_root(workspace: str | Path) -> Path:
    """Return a workspace's fetched Distribution Release tree."""
    return Path(workspace) / OFFICIAL_RELEASE_DIRECTORY


def workspace_roots(path: str | Path) -> tuple[Path | None, Path | None]:
    """Classify *path* as ``(local_root, release_root)``, each ``None`` when absent.

    A workspace holds ``local/`` and optionally ``official-release/``; a bare
    Distribution Release bundle (``release.json`` and no ``local/``) is
    ``(None, path)``; anything else is ``(None, None)``.
    """
    root = Path(path)
    local = workspace_local_root(root)
    if local.is_dir():
        release = workspace_release_root(root)
        return local, release if release.is_dir() else None
    if (root / RELEASE_MANIFEST_NAME).is_file():
        return None, root
    return None, None


@dataclass(frozen=True)
class ReleaseIdentity:
    """The fetched Distribution Release a workspace or bundle carries, if any.

    Attributes:
        version: The release's own human-readable version label.
        manifest_id: The release manifest's content identity.
        tasks: The tasks it released, in manifest order.
        path: The installed ``official-release/`` (or bare bundle) directory.
    """

    version: str
    manifest_id: str
    tasks: tuple[str, ...]
    path: Path


def read_release_identity(release_root: Path) -> ReleaseIdentity | None:
    """Return an installed release tree's identity, or ``None`` without a manifest."""
    manifest_path = release_root / RELEASE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    return ReleaseIdentity(
        version=manifest["version"],
        manifest_id=manifest["manifest_id"],
        tasks=tuple(manifest["tasks"]),
        path=release_root,
    )


def task_collection_directory(tree_root: str | Path, task_id: str) -> Path:
    """Return one Task's collection directly beneath a tree root (``local/`` or ``official-release/``)."""
    return Path(tree_root) / task_id


def input_artifact_path(input_artifacts_root: str | Path, input_digest: str) -> Path:
    """Return one logical Input artifact's path in the shared exact root."""
    return Path(input_artifacts_root) / input_artifact_name(input_digest)


def input_artifact_name(input_digest: str) -> str:
    """Return the file name storing one Input's exact logical contents."""
    return f"{input_digest}{INPUT_ARTIFACT_SUFFIX}"


def attempts_directory(task_root: Path) -> Path:
    """Return one task's durable Attempt record directory."""
    return task_root / ATTEMPTS_DIRECTORY


def results_directory(task_root: Path) -> Path:
    """Return one released task's resolved Result record directory."""
    return task_root / RESULTS_DIRECTORY


def partition_directory(store: Path, alg_uuid: str) -> Path:
    """Return one algorithm's partition inside an Attempt or Result store directory."""
    return store / f"{PARTITION_PREFIX}{alg_uuid}"


def attempt_partition_directory(task_root: Path, alg_uuid: str) -> Path:
    """Return the partition holding one algorithm's Attempts in a task's store."""
    return partition_directory(attempts_directory(task_root), alg_uuid)


def result_partition_directory(task_root: Path, alg_uuid: str) -> Path:
    """Return the partition holding one algorithm's released Results."""
    return partition_directory(results_directory(task_root), alg_uuid)


def claims_directory(task_root: Path) -> Path:
    """Return one task's in-flight attempt claim directory."""
    return task_root / CLAIMS_DIRECTORY


def sweeps_directory(task_root: Path) -> Path:
    """Return one task's installed Sweep directory."""
    return task_root / SWEEPS_DIRECTORY


def task_benchmark_info_directory(task_root: Path) -> Path:
    """Return one task's operator-only benchmark-info directory (algorithm declarations, withheld profiles)."""
    return task_root / BENCHMARK_INFO_DIRECTORY


def algorithm_declarations_path(task_root: Path) -> Path:
    """Return one task's author-owned algorithm declarations file."""
    return task_benchmark_info_directory(task_root) / ALGORITHM_DECLARATIONS_NAME


def withheld_profiles_path(task_root: Path) -> Path:
    """Return one released task's withheld-lane profile view.

    A Distribution Release's copy sits in ``benchmark-info/``; a Working store
    keeps no such file.
    """
    return task_benchmark_info_directory(task_root) / WITHHELD_PROFILES_NAME


def instance_weights_path(task_root: Path) -> Path:
    """Return one released task's per-Instance weights view."""
    return task_root / INSTANCE_WEIGHTS_NAME


def root_benchmark_info_directory(tree_root: str | Path) -> Path:
    """Return a tree's root-level benchmark-info directory (protocol, provenance)."""
    return Path(tree_root) / BENCHMARK_INFO_DIRECTORY


def provenance_tree_root(provenance_root: Path) -> Path:
    """Return the tree root whose shared provenance directory *provenance_root* is."""
    return provenance_root.parent.parent


def provenance_directory(tree_root: str | Path) -> Path:
    """Return a tree's shared provenance directory: one study document per source, naming its Instances."""
    return root_benchmark_info_directory(tree_root) / PROVENANCE_DIRECTORY


def registry_directory_name(kind: str) -> str:
    """Return the directory name holding one kind's registry documents."""
    return f"{kind}_registries"
