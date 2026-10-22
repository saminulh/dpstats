"""Public CLI coverage for portable method-submission bundles."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dpstats.algorithm import register_algorithm
from dpstats.transport.registry import (
    JSONRegistry,
    add_entry,
)
from dpstats.scripts.submission import (
    MANIFEST_FILENAME,
    REGISTRY_FILENAME,
    SUBMISSION_FILENAME,
    prepare_submission,
    verify_submission,
)
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.tasks.mean_estimation.algorithms.sample_mean import SampleMean
from dpstats.workspace.layout import (
    task_collection_directory,
    workspace_local_root,
)
from dpstats.workspace.registries import registry_document_path

REGISTRY_NAME = "candidates"
SOURCE_URL = "https://github.com/example/method"
SOURCE_COMMIT = "a" * 40


def _install(task_collections_root: Path, *, label: str | None = "candidate") -> str:
    """Install one configured candidate in the task-local registry group."""
    return register_algorithm(
        task_collections_root, Ridge(label=label), registry=REGISTRY_NAME
    )


def _prepare(
    task_collections_root: Path,
    bundle: Path,
    *,
    selector: str = Ridge().uuid,
    algorithm_registries: str | Sequence[str] | None = (REGISTRY_NAME,),
    resource_note: str | None = None,
) -> None:
    """Prepare one bundle from the installed task-local candidate registry."""
    prepare_submission(
        task_collections_root.parent,
        "linear_regression",
        selector,
        bundle,
        algorithm_registries=algorithm_registries,
        target_series="v1-experimental",
        source_url=SOURCE_URL,
        source_commit=SOURCE_COMMIT,
        resource_note=resource_note,
    )


def _refresh_manifest(bundle: Path) -> None:
    """Refresh checksums after intentionally tampering with a test bundle."""
    manifest_path = bundle / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    for filename in (SUBMISSION_FILENAME, REGISTRY_FILENAME):
        manifest["files"][filename] = hashlib.sha256(
            (bundle / filename).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def _assert_verification_fails(bundle: Path, *messages: str) -> None:
    """Assert that verification refuses a malformed test bundle.

    A refusal translated at an inner boundary names the file it could not read
    and leaves the specific reason on its cause, so either carries the reason.
    """
    try:
        verify_submission(bundle)
    except ValueError as error:
        reported = f"{error}\n{error.__cause__}"
        for message in messages:
            assert message in reported
    else:
        raise AssertionError(f"verification should fail with {messages!r}")


def test_prepare_and_verify_round_trip(tmp_path: Path) -> None:
    """An installed registry entry reaches the bundle and back unchanged."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle, resource_note="tested on a laptop")

    request = json.loads((bundle / SUBMISSION_FILENAME).read_text())
    registry = json.loads((bundle / REGISTRY_FILENAME).read_text())
    installed = json.loads(
        registry_document_path(
            task_collection_directory(task_collections_root, "linear_regression"),
            "algorithm",
            REGISTRY_NAME,
        ).read_text()
    )
    assert request["candidate_uuid"] == uuid
    assert request["algorithm"]["hyperparams"] == {
        "label": "candidate",
        "regularization_multiplier": 0.01,
    }
    assert registry["task"] == "linear_regression"
    assert registry["kind"] == "algorithm"
    assert registry["entries"] == {uuid: installed["entries"][uuid]}
    assert verify_submission(bundle) == {
        "valid": True,
        "target_series": "v1-experimental",
        "task": "linear_regression",
        "candidate_uuid": uuid,
        "requested_label": "candidate",
        "privacy_interpretation": "nonprivate",
        "algorithm": {
            "path": "dpstats.tasks.linear_regression.algorithms.ridge:Ridge",
            "hyperparams": {"label": "candidate", "regularization_multiplier": 0.01},
        },
        "source": {
            "url": SOURCE_URL,
            "commit": SOURCE_COMMIT,
        },
    }


def test_prepare_selects_the_same_entry_by_uuid(tmp_path: Path) -> None:
    """A UUID selects one entry from a single named registry."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(
        task_collections_root, bundle, selector=uuid, algorithm_registries=REGISTRY_NAME
    )

    report = verify_submission(bundle)
    assert report["candidate_uuid"] == uuid
    assert report["requested_label"] == "candidate"


def test_prepare_reads_every_installed_group_when_none_is_named(
    tmp_path: Path,
) -> None:
    """Omitting the group selection reads the task's installed registries."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle, algorithm_registries=None)

    assert verify_submission(bundle)["candidate_uuid"] == uuid


def test_prepare_never_selects_by_label(tmp_path: Path) -> None:
    """A UUID-looking label cannot shadow a different configured identity."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root)
    register_algorithm(
        task_collections_root,
        Ridge(label=uuid, regularization_multiplier=0.5),
        registry=REGISTRY_NAME,
    )

    for selector in ("missing", "candidate"):
        with pytest.raises(ValueError, match="not an installed algorithm UUID"):
            _prepare(task_collections_root, tmp_path / selector, selector=selector)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle, selector=uuid)
    assert verify_submission(bundle)["candidate_uuid"] == uuid


def test_prepare_refuses_an_unlabelled_entry(tmp_path: Path) -> None:
    """A request records the label an operator is asked to run."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root, label=None)

    try:
        _prepare(task_collections_root, tmp_path / "bundle", selector=uuid)
    except ValueError as error:
        assert "carries no label" in str(error)
    else:
        raise AssertionError("an unlabelled candidate must fail")


def test_a_labelled_submission_group_preserves_the_original_entry(
    tmp_path: Path,
) -> None:
    """Recover an unlabelled candidate without rewriting its existing registry."""
    task_collections_root = workspace_local_root(tmp_path)
    uuid = _install(task_collections_root, label=None)
    original = registry_document_path(
        task_collection_directory(task_collections_root, "linear_regression"),
        "algorithm",
        REGISTRY_NAME,
    )
    original_bytes = original.read_bytes()
    labelled = Ridge(label="candidate")
    assert labelled.uuid == uuid
    register_algorithm(task_collections_root, labelled, registry="submission")
    bundle = tmp_path / "bundle"

    _prepare(task_collections_root, bundle, algorithm_registries="submission")

    report = verify_submission(bundle)
    assert report["candidate_uuid"] == uuid
    assert report["requested_label"] == "candidate"
    assert original.read_bytes() == original_bytes


def test_prepare_rejects_file_as_output_directory(tmp_path: Path) -> None:
    """An existing file at ``--out`` is ordinary CLI misuse, not a traceback."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    output_path = tmp_path / "bundle"
    output_path.write_text("occupied")

    try:
        _prepare(task_collections_root, output_path)
    except ValueError as error:
        assert "must be a directory" in str(error)
    else:
        raise AssertionError("a file at --out must fail cleanly")


def test_prepare_refuses_a_symlinked_output_directory(tmp_path: Path) -> None:
    """Preparation must not write a bundle through an ambient link target."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    target = tmp_path / "target"
    target.mkdir()
    output_path = tmp_path / "bundle"
    output_path.symlink_to(target, target_is_directory=True)

    try:
        _prepare(task_collections_root, output_path)
    except ValueError as error:
        assert "must not be a symlink" in str(error)
    else:
        raise AssertionError("a symlinked --out must fail cleanly")

    assert list(target.iterdir()) == []


def test_verify_refuses_tampered_bundle(tmp_path: Path) -> None:
    """Checksums and schema both refuse an edited candidate request."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle)
    request_path = bundle / SUBMISSION_FILENAME
    request = json.loads(request_path.read_text())
    request["unexpected"] = True
    request_path.write_text(json.dumps(request))

    try:
        verify_submission(bundle)
    except ValueError as error:
        assert "Checksum mismatch" in str(error)
    else:
        raise AssertionError("tampered submission must fail")


def test_verify_allowlists_submission_schema(tmp_path: Path) -> None:
    """A checksum-valid request still cannot add undeclared semantics."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle)
    request_path = bundle / SUBMISSION_FILENAME
    request = json.loads(request_path.read_text())
    request["unexpected"] = True
    request_path.write_text(json.dumps(request))
    _refresh_manifest(bundle)

    try:
        verify_submission(bundle)
    except ValueError as error:
        assert "unknown ['unexpected']" in str(error)
    else:
        raise AssertionError("unknown submission field must fail")


def test_verify_refuses_checksum_valid_foreign_task_candidate(tmp_path: Path) -> None:
    """A bundle cannot assign a native mean algorithm to linear regression.

    Nothing here is tampered: a candidate UUID names a configuration rather
    than a task, so the entry's identity is genuine and only reconstructing the
    configured value establishes which task it belongs to.
    """
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle)
    registry_path = bundle / REGISTRY_FILENAME
    registry_path.unlink()
    algorithm = SampleMean(label="candidate")
    candidate = add_entry(
        JSONRegistry(task=algorithm.task.id, kind="algorithm", entries={}),
        uuid=algorithm.uuid,
        record=algorithm.dump_config().model_dump(),
    )
    document = candidate.model_dump()
    document["task"] = "linear_regression"
    registry_path.write_text(json.dumps(document))
    configuration = algorithm.dump_config()
    request_path = bundle / SUBMISSION_FILENAME
    request = json.loads(request_path.read_text())
    request["algorithm"] = {
        "path": configuration.path,
        "hyperparams": configuration.config,
    }
    request["candidate_uuid"] = algorithm.uuid
    request_path.write_text(json.dumps(request))
    _refresh_manifest(bundle)

    _assert_verification_fails(bundle, "native to task 'mean_estimation'")


def test_verify_refuses_checksum_valid_registry_metadata_tamper(
    tmp_path: Path,
) -> None:
    """Registry presentation fields cannot bypass the submission contract."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    cases: list[tuple[str, Any, tuple[str, ...]]] = [
        ("label", "Impostor", ("label does not match submission.json",)),
        (
            "display_name",
            "Impostor",
            ("display_name", "Extra inputs are not permitted"),
        ),
    ]
    for field, value, messages in cases:
        bundle = tmp_path / field
        _prepare(task_collections_root, bundle)
        registry_path = bundle / REGISTRY_FILENAME
        document = json.loads(registry_path.read_text())
        entry = next(iter(document["entries"].values()))
        if field == "label":
            entry["config"][field] = value
        else:
            entry[field] = value
        registry_path.write_text(json.dumps(document))
        _refresh_manifest(bundle)

        _assert_verification_fails(bundle, *messages)


def test_verify_requires_regular_bundle_files(tmp_path: Path) -> None:
    """Bundle entries cannot redirect reads or replace files with directories."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    for entry_kind in ("symlink", "directory"):
        bundle = tmp_path / entry_kind
        _prepare(task_collections_root, bundle)
        request_path = bundle / SUBMISSION_FILENAME
        request_path.unlink()
        if entry_kind == "symlink":
            outside = tmp_path / "outside.json"
            outside.write_text("{}")
            request_path.symlink_to(outside)
        else:
            request_path.mkdir()

        _assert_verification_fails(bundle, "non-symlink regular file")


def test_verify_refuses_a_symlinked_bundle_root(tmp_path: Path) -> None:
    """A portable bundle cannot derive its location from an ambient link target."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle)
    linked_bundle = tmp_path / "linked-bundle"
    linked_bundle.symlink_to(bundle, target_is_directory=True)

    _assert_verification_fails(linked_bundle, "non-symlink directory")


def test_verify_rejects_oversize_bundle_file(tmp_path: Path) -> None:
    """Small submission JSONs have a fixed pre-read size ceiling."""
    task_collections_root = workspace_local_root(tmp_path)
    _install(task_collections_root)
    bundle = tmp_path / "bundle"
    _prepare(task_collections_root, bundle)
    (bundle / SUBMISSION_FILENAME).write_bytes(b" " * (1024 * 1024 + 1))

    _assert_verification_fails(bundle, "exceeds the 1 MiB limit")


@pytest.mark.timeout(60)  # Exports and pip-installs the library, then runs three CLIs.
def test_cli_round_trip_with_normally_installed_external_package(
    committed_library_source: None,
    tmp_path: Path,
) -> None:
    """Fresh processes register, prepare, and verify from a normal install.

    The library is installed from an export of the current commit, so
    regenerable working-tree residue such as a stale setuptools ``build/``
    tree cannot enter the installed package.
    """
    repo_root = Path(__file__).resolve().parents[3]
    package_source = tmp_path / "external-source"
    package_source.mkdir()
    (package_source / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "submission-test-method"
version = "0.1.0"

[tool.setuptools]
py-modules = ["external_method"]
"""
    )
    (package_source / "external_method.py").write_text(
        """from typing import Any

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.tasks import LinearRegression


class ExternalAlgorithm(BaseAlgorithm):
    TASK = LinearRegression
    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.INFORMAL
    scale: float = 2.5

    def run(
        self, metadata: Any, input: Any, budget: Any, seed: int
    ) -> Any:
        raise NotImplementedError
"""
    )
    library_source = tmp_path / "library-source"
    library_source.mkdir()
    library_archive = tmp_path / "library-source.tar"
    exported = subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(library_archive), "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr
    with tarfile.open(library_archive) as archive:
        archive.extractall(library_source)

    installed = tmp_path / "installed"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--target",
            str(installed),
            str(library_source),
            str(package_source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = tmp_path / "bundle"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(installed)
    env["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, dpstats, external_method; "
                "print(json.dumps([dpstats.__file__, "
                "external_method.__file__]))"
            ),
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert probe.returncode == 0, probe.stderr
    assert all(
        Path(origin).resolve().is_relative_to(installed.resolve())
        for origin in json.loads(probe.stdout)
    )
    registered = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "from dpstats.algorithm import register_algorithm\n"
                "from external_method import ExternalAlgorithm\n"
                "print(register_algorithm(Path('local'), "
                "ExternalAlgorithm(label='candidate'), registry='candidates'))\n"
            ),
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert registered.returncode == 0, registered.stderr
    prepare = subprocess.run(
        [
            sys.executable,
            "-m",
            "dpstats.cli",
            "prepare-submission",
            "--workspace",
            ".",
            "--task",
            "linear_regression",
            "--algorithm-registry",
            "candidates",
            "--algorithm",
            registered.stdout.strip(),
            "--out",
            str(bundle),
            "--target-series",
            "v1-experimental",
            "--source-url",
            SOURCE_URL,
            "--source-commit",
            SOURCE_COMMIT,
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert prepare.returncode == 0, prepare.stderr
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "dpstats.cli",
            "verify-submission",
            str(bundle),
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert verified.returncode == 0, verified.stderr
    report = json.loads(verified.stdout)
    assert report["candidate_uuid"] == registered.stdout.strip()
    assert report["algorithm"] == {
        "path": "external_method:ExternalAlgorithm",
        "hyperparams": {"label": "candidate", "scale": 2.5},
    }
    assert report["privacy_interpretation"] == "informal"
    assert report["source"] == {
        "url": SOURCE_URL,
        "commit": SOURCE_COMMIT,
    }
    assert set(path.name for path in bundle.iterdir()) == {
        SUBMISSION_FILENAME,
        REGISTRY_FILENAME,
        MANIFEST_FILENAME,
    }
