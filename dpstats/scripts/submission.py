"""Prepare and verify one portable method-submission request."""

from __future__ import annotations

import argparse
import json
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from dpstats.algorithm import BaseAlgorithm, load_algorithms
from dpstats.task import Task
from dpstats.workspace import open_collection
from dpstats.transport.configuration import ConfigurationRecord
from dpstats.transport.registry import (
    JSONRegistry,
    add_entry,
    read_registry,
)
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
    file_sha256,
)
from dpstats.transport.validators import is_sha256_hex
from dpstats.transport.documents import write_json_atomic
from dpstats.transport.validators import is_immutable_revision
from dpstats.scripts.utils import resolve_task


SUBMISSION_FILENAME = "submission.json"
REGISTRY_FILENAME = "candidate-algorithm-registry.json"
MANIFEST_FILENAME = "manifest.json"
_MAX_BUNDLE_FILE_BYTES = 1024 * 1024


def _require_exact_keys(
    value: Any,
    required: set[str],
    optional: frozenset[str] = frozenset(),
    *,
    name: str,
) -> dict[str, Any]:
    """Require a JSON object with exactly its documented keys."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    mapping = cast(dict[str, Any], value)
    keys = set(mapping)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if unknown:
            details.append(f"unknown {sorted(unknown)!r}")
        raise ValueError(f"{name} has " + " and ".join(details) + ".")
    return mapping


def _validate_source(source_url: str, source_commit: str) -> None:
    """Require an immutable, publicly interpretable source reference."""
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("--source-url must be an absolute https URL.")
    if not is_immutable_revision(source_commit):
        raise ValueError("--source-commit must be a 40-character lowercase Git commit.")


def _selected_algorithm(
    workspace: Path,
    task: Task,
    algorithm_registries: str | Sequence[str] | None,
    selector: str,
) -> tuple[str, BaseAlgorithm]:
    """Select one installed configured Algorithm by UUID."""
    algorithms = open_collection(workspace, task).load_algorithms(
        registries=algorithm_registries
    )
    if selector not in algorithms:
        raise ValueError(
            f"--algorithm {selector!r} is not an installed algorithm UUID "
            "in the selected registries. Labels are not identifiers."
        )
    return selector, algorithms[selector]


def prepare_submission(
    workspace: Path,
    task_id: str,
    selector: str,
    output_dir: Path,
    *,
    algorithm_registries: str | Sequence[str] | None,
    target_series: str,
    source_url: str,
    source_commit: str,
    resource_note: str | None,
) -> None:
    """Write the self-contained request and one-record candidate registry."""
    if not target_series:
        raise ValueError("--target-series must not be empty.")
    _validate_source(source_url, source_commit)
    task = resolve_task(task_id)
    uuid, algorithm = _selected_algorithm(
        workspace, task, algorithm_registries, selector
    )
    label = algorithm.label
    if label is None:
        raise ValueError(
            f"The installed algorithm {uuid!r} carries no label, and a request "
            "records the label an operator is asked to run. Construct the "
            "algorithm with label=, or pass label= when tuning it, and install "
            "it again."
        )
    try:
        output_mode = output_dir.lstat().st_mode
    except FileNotFoundError:
        output_dir.mkdir(parents=True)
    else:
        if stat.S_ISLNK(output_mode):
            raise ValueError(
                f"Submission output directory {output_dir} must not be a symlink."
            )
        if not stat.S_ISDIR(output_mode):
            raise ValueError(
                f"Submission output path {output_dir} must be a directory."
            )
        if any(output_dir.iterdir()):
            raise ValueError(f"Submission output directory {output_dir} must be empty.")

    # The bundle carries the record this configured algorithm derives, which
    # is the one verification rebuilds it from and compares against.
    configuration = algorithm.dump_config()
    candidate = add_entry(
        JSONRegistry(task=task.id, kind="algorithm", entries={}),
        uuid=uuid,
        record=configuration.model_dump(),
    )
    write_json_atomic(output_dir / REGISTRY_FILENAME, candidate.model_dump())

    request: dict[str, Any] = {
        "schema": 1,
        "target_series": target_series,
        "task": task.id,
        "requested_label": label,
        "algorithm": {
            "path": configuration.path,
            "hyperparams": configuration.config,
        },
        "candidate_uuid": uuid,
        "source": {"url": source_url, "commit": source_commit},
    }
    if resource_note is not None:
        request["resource_note"] = resource_note
    write_json_atomic(output_dir / SUBMISSION_FILENAME, request)
    manifest = {
        "schema": 1,
        "files": {
            SUBMISSION_FILENAME: file_sha256(output_dir / SUBMISSION_FILENAME),
            REGISTRY_FILENAME: file_sha256(output_dir / REGISTRY_FILENAME),
        },
    }
    write_json_atomic(output_dir / MANIFEST_FILENAME, manifest)


def _validated_request(value: Any) -> dict[str, Any]:
    """Validate the fixed submission request shape before reconstruction."""
    request = _require_exact_keys(
        value,
        {
            "schema",
            "target_series",
            "task",
            "requested_label",
            "algorithm",
            "candidate_uuid",
            "source",
        },
        frozenset({"resource_note"}),
        name=SUBMISSION_FILENAME,
    )
    if request["schema"] != 1:
        raise ValueError("submission.json has an unsupported schema.")
    if not all(
        isinstance(request[key], str) and request[key]
        for key in ("target_series", "task", "requested_label", "candidate_uuid")
    ):
        raise ValueError("submission.json has an empty or non-string identity field.")
    algorithm = _require_exact_keys(
        request["algorithm"], {"path", "hyperparams"}, name="submission.algorithm"
    )
    if not isinstance(algorithm["path"], str) or not isinstance(
        algorithm["hyperparams"], dict
    ):
        raise ValueError(
            "submission.algorithm requires a string path and object hyperparams."
        )
    source = _require_exact_keys(
        request["source"], {"url", "commit"}, name="submission.source"
    )
    if not isinstance(source["url"], str) or not isinstance(source["commit"], str):
        raise ValueError("submission.source fields must be strings.")
    _validate_source(source["url"], source["commit"])
    if "resource_note" in request and not isinstance(request["resource_note"], str):
        raise ValueError("submission.resource_note must be a string.")
    canonical_json_bytes(algorithm)
    return request


def _validated_manifest(value: Any) -> dict[str, str]:
    """Validate the fixed checksum manifest shape."""
    manifest = _require_exact_keys(value, {"schema", "files"}, name=MANIFEST_FILENAME)
    if manifest["schema"] != 1:
        raise ValueError("manifest.json has an unsupported schema.")
    files = _require_exact_keys(
        manifest["files"],
        {SUBMISSION_FILENAME, REGISTRY_FILENAME},
        name="manifest.files",
    )
    if not all(is_sha256_hex(digest) for digest in files.values()):
        raise ValueError("manifest.json contains an invalid SHA-256 digest.")
    return {name: digest for name, digest in files.items()}


def verify_submission(bundle_dir: Path) -> dict[str, Any]:
    """Verify checksums and reconstruct the submitted algorithm from installed code."""
    expected_paths = {
        SUBMISSION_FILENAME,
        REGISTRY_FILENAME,
        MANIFEST_FILENAME,
    }
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError(
            f"Submission bundle {bundle_dir} must be a non-symlink directory."
        )
    found_paths = {path.name for path in bundle_dir.iterdir()}
    if found_paths != expected_paths:
        raise ValueError(
            "Submission bundle must contain exactly "
            f"{sorted(expected_paths)!r}; found {sorted(found_paths)!r}."
        )
    bundle_paths = {filename: bundle_dir / filename for filename in expected_paths}
    for path in bundle_paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Submission bundle entry {path.name} must be a non-symlink "
                "regular file."
            )
        if path.stat().st_size > _MAX_BUNDLE_FILE_BYTES:
            raise ValueError(
                f"Submission bundle entry {path.name} exceeds the 1 MiB limit."
            )
    manifest = _validated_manifest(
        decode_json_document(bundle_paths[MANIFEST_FILENAME])
    )
    for filename, expected_digest in manifest.items():
        actual_digest = file_sha256(bundle_paths[filename])
        if actual_digest != expected_digest:
            raise ValueError(f"Checksum mismatch for {filename}.")
    request = _validated_request(
        decode_json_document(bundle_paths[SUBMISSION_FILENAME])
    )
    task = resolve_task(request["task"])

    entries = read_registry(bundle_paths[REGISTRY_FILENAME]).entries
    if len(entries) != 1:
        raise ValueError("Candidate registry must hold exactly one entry.")
    uuid, record = next(iter(entries.items()))
    if uuid != request["candidate_uuid"]:
        raise ValueError("Candidate registry UUID does not match submission.json.")
    entry = ConfigurationRecord.model_validate(record)
    if entry.config.get("label") != request["requested_label"]:
        raise ValueError("Candidate registry label does not match submission.json.")
    if {"path": entry.path, "hyperparams": entry.config} != request["algorithm"]:
        raise ValueError("Candidate registry algorithm does not match submission.json.")

    algorithm = load_algorithms(entries, task_id=task.id, strict=True)[uuid]
    configuration = algorithm.dump_config()
    reconstructed = {
        "path": configuration.path,
        "hyperparams": configuration.config,
    }
    if reconstructed != request["algorithm"]:
        raise ValueError("Reconstructed algorithm does not match submission.json.")
    return {
        "valid": True,
        "target_series": request["target_series"],
        "task": task.id,
        "candidate_uuid": uuid,
        "requested_label": request["requested_label"],
        "privacy_interpretation": algorithm.privacy_interpretation.value,
        "algorithm": reconstructed,
        "source": request["source"],
    }


def prepare_parser() -> argparse.ArgumentParser:
    """Build the ``prepare-submission`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Prepare one method submission bundle."
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="The workspace or bare bundle holding the installed algorithm (path, required)",
    )
    command.add_argument(
        "--task",
        required=True,
        help="Built-in task the candidate is native to (required)",
    )
    command.add_argument(
        "--algorithm-registry",
        action="append",
        help="Installed algorithm registry name; repeat to widen the selection "
        "(optional; default: every installed group)",
    )
    command.add_argument(
        "--algorithm",
        required=True,
        help="Installed algorithm's UUID (required)",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Empty bundle directory (path, required)",
    )
    command.add_argument(
        "--target-series",
        required=True,
        help="Target series the submission names, such as v1-experimental (required)",
    )
    command.add_argument(
        "--source-url",
        required=True,
        help="URL of the repository holding the candidate's source (required)",
    )
    command.add_argument(
        "--source-commit",
        required=True,
        help="Commit of that repository the candidate was built from (required)",
    )
    command.add_argument(
        "--resource-note",
        help="Free-text note on the compute the candidate needs (optional; default: none)",
    )
    return command


def prepare_main(args: list[str] | None = None) -> None:
    """Run the ``prepare-submission`` command."""
    parsed = prepare_parser().parse_args(args)
    prepare_submission(
        Path(parsed.workspace),
        parsed.task,
        parsed.algorithm,
        parsed.out,
        algorithm_registries=parsed.algorithm_registry,
        target_series=parsed.target_series,
        source_url=parsed.source_url,
        source_commit=parsed.source_commit,
        resource_note=parsed.resource_note,
    )
    print(json.dumps({"bundle": str(parsed.out), "valid": True}, sort_keys=True))


def verify_parser() -> argparse.ArgumentParser:
    """Build the ``verify-submission`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Verify one method submission bundle."
    )
    command.add_argument(
        "bundle", type=Path, help="Bundle directory to verify (path, required)"
    )
    return command


def verify_main(args: list[str] | None = None) -> None:
    """Run the ``verify-submission`` command."""
    parsed = verify_parser().parse_args(args)
    print(json.dumps(verify_submission(parsed.bundle), sort_keys=True))
