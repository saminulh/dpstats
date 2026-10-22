"""Prepare the one release once; render it without reading Results again.

One preparation reads the Distribution Release this build's single input names,
measures its distribution lane under the ranking that release's own protocol
states, beside the external baselines of the current protocol where the build
is given one, ingests the curves its evaluation lane publishes, and writes one
self-contained snapshot. The build then renders exactly that snapshot: no
Results or Input are read while pages are assembled, and no metric is
recomputed.

The snapshot is a cache, not a second input. It is named for the release it was
taken from and the code that took it, so the same one command chain prepares
once and renders many times::

    DOCS_LOCAL_RELEASE_ROOT=<release> python docs/site/prepare_results.py
    DOCS_LOCAL_RELEASE_ROOT=<release> mkdocs build --strict -f docs/site/mkdocs.yml

Running the first line is optional: a build with no snapshot prepares one. A
changed release, or changed measurement code, is a different name and is
prepared again, so a stale snapshot is never rendered. A local cut is private
review input, not a Distribution Release publication or authorization to
publish one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dpstats
from dpstats import open_collection
from dpstats.tasks import (
    list_built_in_metric_arguments,
    compute_metric_arguments,
)
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
    file_sha256,
)
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
    registry_directory_name,
)

from benchmark_lock import LOCK_NAME, read_benchmark_lock
from panel_data import (
    Headline,
    PanelData,
    TaskData,
    missing_results_inputs,
    protocol_conflicts,
    registered_algorithms,
    released_standings,
    released_weights,
    task_data,
)
from results_roots import (
    CURRENT_PROTOCOL_ENV,
    LOCAL_RELEASE_ENV,
    WITHHELD_PROFILES_ROLE,
    ReleaseInput,
    ReleasedTask,
    read_comparison,
    release_input,
)
from algorithm_provenance import PROVENANCE_NAME, provenance_payload
from site_copy import load_site_copy
from skipped_rows import skipped_entries, summary_line

PREPARED_NAME = "prepared.json"
BUILD_SUMMARY_NAME = "build-summary.json"
INSTANCE_ARGUMENTS_NAME = "instance-arguments.json"

# The prepared-snapshot format this module writes and reads. A snapshot naming
# any other format is refused: its stored index carries different facts.
PREPARED_VERSION = 18

# Where prepared snapshots are cached, one directory per release and per code
# that prepared it.
CACHE_DIR = Path(__file__).resolve().parent / ".prepared"


def build_release() -> ReleaseInput | None:
    """Resolve the one release this build reads from its one input."""
    return release_input(
        read_benchmark_lock(Path(__file__).resolve().parents[1] / LOCK_NAME),
        os.environ.get(LOCAL_RELEASE_ENV),
        cache_dir=CACHE_DIR / "releases",
        current_protocol=os.environ.get(CURRENT_PROTOCOL_ENV),
    )


def prepared_snapshot(release: ReleaseInput, *, cache: Path | None = None) -> Path:
    """Return *release*'s prepared snapshot, preparing it when it is absent.

    The snapshot directory is named for the release bytes and the code that
    measures them, so an already-prepared snapshot is reused unchanged and a
    changed release or changed measurement prepares a new one beside it.
    """
    root = (CACHE_DIR if cache is None else cache) / _snapshot_name(release)
    if (root / PREPARED_NAME).is_file():
        return root
    staging = root.with_name(f"{root.name}.staging-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _prepare(release, staging)
        if (root / PREPARED_NAME).is_file():
            # Another build prepared the same snapshot first; both wrote the
            # same release under the same code, so either is the snapshot.
            shutil.rmtree(staging, ignore_errors=True)
        else:
            root.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


def _measurement_code() -> dict[str, str]:
    """Digest everything whose change would measure this release differently."""
    library = Path(dpstats.__file__).parent
    code = {
        str(path.relative_to(library)): file_sha256(path)
        for path in sorted(library.rglob("*.py"))
    }
    for path in _site_modules():
        code[f"site/{path.name}"] = file_sha256(path)
    return code


def _site_modules() -> list[Path]:
    """This module and every site module it imports, directly or not.

    A hand-kept list went stale once already: the provenance writer changed the
    snapshot's bytes without changing its name.
    """
    here = Path(__file__).resolve().parent
    found: set[Path] = set()
    pending = [Path(__file__).resolve()]
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    and node.module
                    and not node.level
                    else []
                )
            )
            for name in names:
                candidate = here / f"{name.split('.')[0]}.py"
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def _snapshot_name(release: ReleaseInput) -> str:
    """Name one snapshot for the release bytes and the code that took them.

    The name is the release's own identity, never where its tree sits, so one
    release prepares once however it was named to the build. The protocol
    document the build reads is named by its bytes, so current external
    baselines that lead the release prepare a snapshot of their own; so are
    the baseline names the snapshot carries.
    """
    stated = json.dumps(
        {
            "format": PREPARED_VERSION,
            "manifest": release.manifest_id,
            "version": release.version,
            "published": release.published,
            "tasks": sorted(release.tasks),
            "protocol": hashlib.sha256(release.protocol).hexdigest(),
            "code": _measurement_code(),
            # Baseline names are written into the snapshot, so new wording
            # for them prepares a new one.
            "baseline_names": load_site_copy("baselines"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(stated.encode("utf-8")).hexdigest()[:32]


def _prepare(release: ReleaseInput, destination: Path) -> Path:
    """Write one new, self-contained snapshot of *release* at *destination*.

    Only presentation data and source notes enter it: executable registries,
    Input arrays, and Results' Outputs do not.
    """
    destination.mkdir(parents=True)
    (destination / BENCHMARK_PROTOCOL_NAME).write_bytes(release.protocol)
    narrative = release.root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_NARRATIVE_NAME
    if narrative.is_file():
        (destination / BENCHMARK_NARRATIVE_NAME).write_bytes(narrative.read_bytes())
    records = {
        task_id: _prepare_task(release, task_id, task, destination)
        for task_id, task in release.tasks.items()
    }
    files = {
        path.relative_to(destination).as_posix(): file_sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    (destination / PREPARED_NAME).write_text(
        json.dumps(
            dict(
                version=PREPARED_VERSION,
                prepared_at=datetime.now(timezone.utc).isoformat(),
                identity=release.identity,
                published=release.published,
                narrative=narrative.is_file(),
                code=_measurement_code(),
                files=files,
                tasks=records,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination


def _prepare_task(
    release: ReleaseInput, task: str, released: ReleasedTask, target: Path
) -> dict[str, Any]:
    """Prepare one task: its measured panels, and the standings it publishes.

    A release that publishes standings over Results it does not ship has them
    ingested all the same, and what is missing travels with the record rather
    than stopping the preparation, so one task's absence never takes another
    task's pages with it. The protocol and this library must agree on every
    Metric the protocol names.
    """
    root = release.root
    folder = target / task
    folder.mkdir()
    arguments: dict[str, Any] = {}
    comparison = read_comparison(release.protocol, task)
    conflicts = protocol_conflicts(task, comparison)
    if conflicts:
        raise ValueError(
            f"The protocol {release.identity} reads for task {task!r} names "
            f"{'; '.join(conflicts)}."
        )
    missing = missing_results_inputs(root, task, sweep=released.sweep)
    record: dict[str, Any] = dict(
        missing=list(missing),
        data=None,
        standings=None,
        refused=[],
        skipped=skipped_entries(root, task, published=release.published),
        removed_from_standings=0,
    )
    omit = {row["uuid"] for row in record["skipped"] if row["kind"] == "algorithm"}
    if omit and release.published:
        raise ValueError(
            f"{release.identity} is published, so it may not omit standings rows."
        )
    registered = registered_algorithms(root, task, strict=release.published)
    pending = any(
        result["state"] == "pending"
        for algorithms in released.results.values()
        for result in algorithms.values()
    )
    labels = (
        {
            uuid: str(algorithm)
            for uuid, algorithm in open_collection(root, task_id=task)
            .load_algorithms(strict=release.published)
            .items()
        }
        if pending
        else {}
    )
    record["skipped"] += [
        {
            "kind": f"pending_{lane}",
            "uuid": uuid,
            "label": labels.get(uuid, uuid),
            "reason": result["reason"],
        }
        for lane, algorithms in released.results.items()
        for uuid, result in algorithms.items()
        if result["state"] == "pending"
    ]
    if not missing and any(
        result["state"] == "ready"
        for result in released.results["distribution"].values()
    ):
        data = task_data(
            root,
            task,
            comparison=comparison,
            weights=released_weights(root, task),
            identity=release.identity,
            published=release.published,
            sweep=released.sweep,
        )
        arguments.update(data.instance_arguments)
        _write_task_data(root, task, data, folder, strict=release.published)
        record["data"] = dict(index=data.index, panels=_panels(data))
        record["refused"] = data.refused
        record["skipped"] += [
            {
                "kind": "incomplete",
                "uuid": uuid,
                "label": label,
                "reason": f"missing {lacking} of {declared} declared trials",
            }
            for uuid, (label, lacking, declared) in data.incomplete.items()
        ]
    if (root / task / registry_directory_name("instance")).is_dir():
        collection = open_collection(root, task_id=task)
        produced = {
            name
            for name, argument in list_built_in_metric_arguments(
                collection.task
            ).items()
            if argument.producer is not None
        }
        for uuid, instance in collection.load_instances(
            strict=release.published
        ).items():
            if uuid in arguments and produced <= arguments[uuid].keys():
                continue
            arguments[uuid] = {
                **{
                    name: value
                    for name, value in instance.metric_arguments.items()
                    if name not in produced
                },
                **compute_metric_arguments(
                    collection.task, instance.metadata, collection.load_input(instance)
                ),
            }
    (folder / INSTANCE_ARGUMENTS_NAME).write_bytes(canonical_json_bytes(arguments))
    standings = released_standings(
        released.views[WITHHELD_PROFILES_ROLE],
        task,
        comparison=comparison,
        registered=registered,
        identity=release.identity,
        omit=omit,
        available={
            uuid
            for uuid, result in released.results["evaluation"].items()
            if result["state"] == "ready"
        },
    )
    record["removed_from_standings"] = standings.omitted
    for name, payload in standings.files.items():
        (folder / name).write_bytes(payload)
    record["standings"] = dict(index=standings.index, panels=_panels(standings))
    return record


def _panels(data: TaskData) -> list[dict[str, Any]]:
    """Each panel's stored facts, without the matrix that ships beside it."""
    return [
        {key: value for key, value in asdict(panel).items() if key != "payload"}
        for panel in data.panels
    ]


def _write_task_data(
    root: Path, task: str, data: TaskData, folder: Path, *, strict: bool
) -> None:
    """Write one measured task's matrices, provenance and source notes."""
    provenance = provenance_payload(
        root,
        task,
        {},
        algorithm_registries=None,
        algorithm_ids=[row["uuid"] for row in data.index["algorithms"]],
        strict=strict,
    )
    for value in provenance.values():
        declaration = value.get("declaration")
        if declaration:
            ref = declaration["provenance_ref"]
            note = folder / f"sources/{ref}.txt"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_bytes((root / ref).read_bytes())
    (folder / PROVENANCE_NAME).write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    for name, payload in data.files.items():
        (folder / name).write_bytes(payload)


def read_prepared(
    root: Path, task: str
) -> tuple[TaskData | None, TaskData | None, list[str], list[tuple[str, str]]]:
    """Read one task's prepared artifacts; never repair by remeasuring.

    Returns:
        The measured panels, the standings the release's own view supplied, the
        reasons this task could not be measured, and the ``(metric, reason)``
        refusals its measurement recorded.
    """
    manifest = read_manifest(root)
    record = manifest["tasks"][task]
    return (
        _stored_data(root, task, record["data"], manifest),
        _stored_data(root, task, record["standings"], manifest),
        list(record["missing"]),
        [(metric, reason) for metric, reason in record["refused"]],
    )


def read_skipped(root: Path, task: str) -> list[dict[str, str]]:
    """The registry entries one task's preparation could not rebuild."""
    skipped: list[dict[str, str]] = read_manifest(root)["tasks"][task]["skipped"]
    return skipped


def read_explored(root: Path, task: str) -> tuple[str, ...]:
    """The Metrics one task's Explore lists, in order; empty where none was measured."""
    data = read_manifest(root)["tasks"][task]["data"]
    return () if data is None else tuple(data["index"]["metric_labels"])


def read_instance_arguments(root: Path) -> dict[str, dict[str, Any]]:
    """Read the prepared references without reopening any Input artifact."""
    return {
        task: decode_json_document(root / task / INSTANCE_ARGUMENTS_NAME)
        for task in read_manifest(root)["tasks"]
    }


def read_removed(root: Path, task: str) -> int:
    """How many skipped algorithms one task's released standings ranked."""
    removed: int = read_manifest(root)["tasks"][task]["removed_from_standings"]
    return removed


def write_build_summary(snapshot: Path) -> Path:
    """Print one line per task and write ``build-summary.json`` in *snapshot*.

    The snapshot is a build cache no site serves, so the file never deploys.
    """
    summary: dict[str, Any] = {}
    for task in read_manifest(snapshot)["tasks"]:
        skipped = read_skipped(snapshot, task)
        removed = read_removed(snapshot, task)
        print(summary_line(task, skipped, removed))
        summary[task] = {"skipped": skipped, "removed_from_standings": removed}
    path = snapshot / BUILD_SUMMARY_NAME
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_manifest(root: Path) -> dict[str, Any]:
    """Read one snapshot's manifest, refusing a format or bytes that moved on."""
    path = root / PREPARED_NAME
    if not path.is_file():
        raise ValueError(f"No prepared snapshot at {root}.")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if manifest["version"] != PREPARED_VERSION:
        raise ValueError(
            f"Prepared snapshot format {manifest['version']} is not the format "
            f"this build reads ({PREPARED_VERSION}); remove {root}."
        )
    for name, digest in manifest["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Prepared snapshot file changed: {name}")
    return manifest


def _stored_data(
    root: Path, task: str, stored: dict[str, Any] | None, manifest: dict[str, Any]
) -> TaskData | None:
    """Rebuild one stored panel set from the snapshot's own bytes."""
    if stored is None:
        return None
    files = {
        name: (root / task / name).read_bytes()
        for entry in stored["index"]["panels"]
        for name in (entry["runs"] or entry.get("curves"),)
        if name is not None
    }
    panels = [
        PanelData(
            payload=files.get(str(entry["runs"]), b""),
            **{
                **panel,
                "headline": {
                    uuid: Headline(**reading)
                    for uuid, reading in panel["headline"].items()
                },
            },
        )
        for panel, entry in zip(stored["panels"], stored["index"]["panels"])
    ]
    for name in manifest["files"]:
        if name.startswith(f"{task}/sources/") or name == f"{task}/{PROVENANCE_NAME}":
            files[name.removeprefix(f"{task}/")] = (root / name).read_bytes()
    return TaskData(index=stored["index"], files=files, panels=panels, refused=[])


if __name__ == "__main__":
    release = build_release()
    if release is None:
        raise SystemExit(
            f"This build reads no release: the committed lock pins none and "
            f"{LOCAL_RELEASE_ENV} names none."
        )
    print(prepared_snapshot(release))
