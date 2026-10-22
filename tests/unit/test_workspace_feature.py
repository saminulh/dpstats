"""Seam tests for the workspace feature: init, fetch, and the dual-tree handle."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dpstats.algorithm import register_algorithm
from dpstats.analysis import compute_error_ratio
from dpstats.budgets import ApproxDPBudget
from dpstats.coordinates import CellCoordinate
from dpstats.domains import BoxDomain
from dpstats.results import ResultStatus
from dpstats.workspace import fetch
from dpstats.simulations import (
    single_coefficient_estimation as sce_simulations,
)
from dpstats.tasks import (
    SingleCoefficientEstimation,
    list_built_in_metric_arguments,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from dpstats.instance import Instance
from dpstats.workspace import (
    TaskCollection,
    ReleaseIdentity,
    fetch_benchmark,
    init_workspace,
    open_collection,
)
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    attempts_directory,
    input_artifact_path,
    instance_weights_path,
    registry_directory_name,
    results_directory,
    task_collection_directory,
    workspace_local_root,
    workspace_release_root,
)
from dpstats.workspace.fetch import (
    HUB_DOWNLOAD_METADATA_DIRECTORY,
    INSTALLED_COMMIT_RECORD,
    LATEST_RELEASE_REVISION,
)
from tests.attempts import (
    abandon_claim,
    current_result,
    write_attempt_partition,
    write_result_partition,
)
from tests.registry_components import (
    SIMPLE_TASK,
    SimpleAlgorithm,
    simple_instance,
    simple_producer,
)
from dpstats.sweep import CellScheduler
from tests.sweep_helpers import observing_schedule
from tests.sweep_helpers import add_instances, install_instances
from dpstats.transport.documents import write_json_atomic
from dpstats.workspace.registries import registry_document_path


# --------------------------------------------------------------------------
# init_workspace
# --------------------------------------------------------------------------


def test_init_workspace_creates_local(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    assert workspace_local_root(workspace).is_dir()


def test_init_workspace_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    init_workspace(root)
    (workspace_local_root(root) / "marker").write_text("kept")
    init_workspace(root)
    assert (workspace_local_root(root) / "marker").read_text() == "kept"


def test_init_workspace_refuses_a_bare_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / RELEASE_MANIFEST_NAME).write_text("{}")
    with pytest.raises(ValueError, match="not a workspace"):
        init_workspace(bundle)


def test_init_workspace_refuses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "not-a-dir"
    path.write_text("x")
    with pytest.raises(ValueError, match="non-symlink directory"):
        init_workspace(path)


# --------------------------------------------------------------------------
# open_collection resolution
# --------------------------------------------------------------------------


def _minimal_bundle(
    root: Path, *, version: str = "v1", manifest_id: str = "m1"
) -> Path:
    """Build a bare bundle directory holding one released Task's Results."""
    root.mkdir(parents=True, exist_ok=True)
    (root / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": version,
                "manifest_id": manifest_id,
                "tasks": [SIMPLE_TASK.id],
                "files": {},
            }
        )
    )
    results_directory(task_collection_directory(root, SIMPLE_TASK.id)).mkdir(
        parents=True, exist_ok=True
    )
    return root


def test_open_collection_on_a_workspace(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    collection = open_collection(workspace, SIMPLE_TASK)
    assert collection.local_root == workspace_local_root(workspace)
    assert collection.release_root is None
    assert collection.release is None


def test_open_collection_on_a_bare_bundle_is_read_only(tmp_path: Path) -> None:
    bundle = _minimal_bundle(tmp_path / "bundle")
    collection = open_collection(bundle, SIMPLE_TASK)
    assert collection.local_root is None
    assert collection.release_root == bundle
    with pytest.raises(ValueError, match="read-only"):
        collection.add_algorithm(SimpleAlgorithm())


def test_open_collection_refuses_a_plain_directory(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="neither a workspace"):
        open_collection(plain, SIMPLE_TASK)


def test_open_collection_reads_release_alongside_local(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    bundle = _minimal_bundle(tmp_path / "bundle-source")
    shutil.copytree(bundle, workspace_release_root(workspace))
    collection = open_collection(workspace, SIMPLE_TASK)
    assert collection.local_root is not None
    assert collection.release_root is not None
    identity = collection.release
    assert isinstance(identity, ReleaseIdentity)
    assert identity.version == "v1"
    assert identity.tasks == (SIMPLE_TASK.id,)


# --------------------------------------------------------------------------
# load_instances / load_algorithms source filters
# --------------------------------------------------------------------------


def test_load_algorithms_source_filter(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    mine = SimpleAlgorithm(scale=1.0)
    official = SimpleAlgorithm(scale=2.0)
    register_algorithm(local_root, mine, registry="mine")

    bundle = _minimal_bundle(tmp_path / "bundle-source")
    register_algorithm(bundle, official, registry="official")
    shutil.copytree(bundle, workspace_release_root(workspace))

    collection = open_collection(workspace, SIMPLE_TASK)
    both = {mine.uuid: mine, official.uuid: official}
    assert collection.load_algorithms(source="local") == {mine.uuid: mine}
    assert collection.load_algorithms(source="release") == {official.uuid: official}
    assert collection.load_algorithms(source="both") == both
    assert collection.load_algorithms() == both
    with pytest.raises(ValueError, match="source must be"):
        collection.load_algorithms(source="bogus")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Inputs resolve local then release
# --------------------------------------------------------------------------


def _release_only_input(tmp_path: Path) -> tuple[Path, TaskCollection, Instance]:
    """A workspace whose one Instance's Input artifact lives only in the release."""
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    (uuid,) = add_instances(
        open_collection(workspace, SIMPLE_TASK), [simple_producer()]
    )
    bundle = _minimal_bundle(tmp_path / "bundle-source")
    (bundle / INPUT_ARTIFACTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    for artifact in (local_root / INPUT_ARTIFACTS_DIRECTORY).iterdir():
        shutil.move(str(artifact), bundle / INPUT_ARTIFACTS_DIRECTORY / artifact.name)
    shutil.copytree(bundle, workspace_release_root(workspace))
    collection = open_collection(workspace, SIMPLE_TASK)
    return local_root, collection, collection.load_instances()[uuid]


def _snapshot(root: Path) -> list[tuple[str, bool]]:
    """Every path under *root*, with whether it is a symlink."""
    return sorted((str(p.relative_to(root)), p.is_symlink()) for p in root.rglob("*"))


def test_inputs_resolve_by_digest_local_then_release(tmp_path: Path) -> None:
    local_root, collection, instance = _release_only_input(tmp_path)
    before = _snapshot(local_root)
    assert collection.load_input(instance).X.shape == (100, 5)
    # Read from the release; a read never stages into local/.
    assert _snapshot(local_root) == before


def test_reads_serve_a_release_input_from_a_read_only_local_tree(
    tmp_path: Path,
) -> None:
    local_root, collection, instance = _release_only_input(tmp_path)
    before = _snapshot(local_root)
    directories = [local_root, local_root / INPUT_ARTIFACTS_DIRECTORY]
    for directory in directories:
        directory.chmod(0o555)
    try:
        assert collection.load_input(instance).X.shape == (100, 5)
        frame = collection.instance_features(
            {"rows": lambda one: float(collection.load_input(one).X.shape[0])}
        )
    finally:
        for directory in directories:
            directory.chmod(0o755)
    assert frame["rows"].tolist() == [100.0]
    assert _snapshot(local_root) == before


def test_a_dangling_staged_link_yields_the_not_installed_refusal(
    tmp_path: Path,
) -> None:
    local_root, collection, instance = _release_only_input(tmp_path)
    release_artifact = input_artifact_path(
        workspace_release_root(collection.workspace) / INPUT_ARTIFACTS_DIRECTORY,
        instance.input_digest,
    )
    link = input_artifact_path(
        local_root / INPUT_ARTIFACTS_DIRECTORY, instance.input_digest
    )
    link.symlink_to(release_artifact)
    release_artifact.unlink()
    with pytest.raises(ValueError, match="not installed"):
        collection.load_input(instance)


def test_a_missing_input_yields_the_not_installed_refusal(tmp_path: Path) -> None:
    local_root, collection, instance = _release_only_input(tmp_path)
    for artifact in (
        workspace_release_root(collection.workspace) / INPUT_ARTIFACTS_DIRECTORY
    ).iterdir():
        artifact.unlink()
    with pytest.raises(ValueError, match="not installed"):
        collection.load_input(instance)


def test_collection_roots_are_read_only(tmp_path: Path) -> None:
    collection = open_collection(init_workspace(tmp_path / "ws"), SIMPLE_TASK)
    for name in ("workspace", "local_root", "release_root"):
        with pytest.raises(AttributeError):
            setattr(collection, name, tmp_path)


# --------------------------------------------------------------------------
# load_results: source / release_version columns
# --------------------------------------------------------------------------


def test_load_results_reports_source_and_release_version(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    task_store = task_collection_directory(local_root, SIMPLE_TASK.id)
    task_store.mkdir(parents=True, exist_ok=True)
    (local_root / INPUT_ARTIFACTS_DIRECTORY).mkdir(parents=True, exist_ok=True)

    collection = open_collection(workspace, SIMPLE_TASK)
    frame = collection.load_results(algorithms=[], instances=[])
    assert list(frame.columns).count("source") == 1
    assert list(frame.columns).count("release_version") == 1
    assert frame.empty

    # A released Result row is tagged "release" with the release's own version.
    bundle = _minimal_bundle(tmp_path / "bundle-source", version="v7")
    algorithm = SimpleAlgorithm()
    register_algorithm(bundle, algorithm, registry="toy")
    bundle_task_store = task_collection_directory(bundle, SIMPLE_TASK.id)
    (instance_uuid,) = install_instances(
        bundle, SIMPLE_TASK, [simple_producer()], registry="toy"
    )
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    write_result_partition(
        bundle_task_store,
        [
            current_result(
                instance_uuid=instance_uuid,
                alg_uuid=algorithm.uuid,
                budget=budget.dump_config().model_dump(),
            )
        ],
    )
    shutil.copytree(bundle, workspace_release_root(workspace))

    collection = open_collection(workspace, SIMPLE_TASK)
    frame = collection.load_results()
    assert len(frame) == 1
    assert frame["source"].tolist() == ["release"]
    assert frame["release_version"].tolist() == ["v7"]


# --------------------------------------------------------------------------
# stored references
# --------------------------------------------------------------------------


def test_a_stored_key_outside_the_task_registry_is_still_a_baseline(
    tmp_path: Path,
) -> None:
    """A source-supplied value the library does not register stays readable."""
    collection = open_collection(
        init_workspace(tmp_path / "ws"), SingleCoefficientEstimation
    )
    metadata, input_data, metric_arguments = sce_simulations.simulate_clipped_gaussian(
        seed=1
    )
    instance = collection.add_instance(
        metadata,
        input_data,
        {**metric_arguments, "published_se": 0.25},
        registry="toy",
    )
    assert "published_se" not in list_built_in_metric_arguments(
        SingleCoefficientEstimation
    )

    frame = pd.DataFrame(
        {"task": ["single_coefficient_estimation"], "instance": [instance]}
    )

    ratios = compute_error_ratio(frame.assign(cost=0.5), "published_se", metric="cost")

    assert ratios["cost"].tolist() == [2.0]


# --------------------------------------------------------------------------
# load_weights
# --------------------------------------------------------------------------


def _release_with_weights(tmp_path: Path, view: str | None) -> TaskCollection:
    """A workspace whose release ships this weights view, or none."""
    workspace = init_workspace(tmp_path / "ws")
    bundle = _minimal_bundle(tmp_path / "bundle-source")
    task_root = task_collection_directory(bundle, SIMPLE_TASK.id)
    task_root.mkdir(parents=True, exist_ok=True)
    if view is not None:
        instance_weights_path(task_root).write_text(view)
    shutil.copytree(bundle, workspace_release_root(workspace))
    return open_collection(workspace, SIMPLE_TASK)


def test_weights_read_each_instance_weight_exactly(tmp_path: Path) -> None:
    exact = "0.33333333333333331"
    collection = _release_with_weights(
        tmp_path, f"instance,weight\nfirst,{exact}\nsecond,2\n"
    )

    weights = collection.load_weights()

    assert list(weights.columns) == ["instance_uuid", "weight"]
    assert weights["instance_uuid"].tolist() == ["first", "second"]
    assert weights["weight"].tolist() == [float(exact), 2.0]
    assert weights["weight"].dtype == np.float64


def test_weights_are_what_profile_curves_accept(tmp_path: Path) -> None:
    from dpstats.analysis import compute_profile_curves

    collection = _release_with_weights(tmp_path, "instance,weight\na,3\nb,1\n")
    rows = pd.DataFrame(
        {
            "instance_uuid": ["a", "b"],
            "instance": ["a", "b"],
            "algorithm": ["x", "x"],
            "cost": [1.0, 5.0],
        }
    )

    curve = compute_profile_curves(
        rows,
        [2.0],
        metric="cost",
        weights=collection.load_weights(),
    )

    assert curve["fraction"].tolist() == [pytest.approx(0.75)]


def test_weights_refuse_a_collection_without_a_release(tmp_path: Path) -> None:
    collection = open_collection(init_workspace(tmp_path / "ws"), SIMPLE_TASK)
    with pytest.raises(ValueError, match="no installed release"):
        collection.load_weights()


def test_weights_name_the_file_a_release_does_not_ship(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="instance-weights.csv"):
        _release_with_weights(tmp_path, None).load_weights()


@pytest.mark.parametrize(
    "view,problem",
    [
        ("instance,budget,weight\na,b,1\n", "columns"),
        ("instance,weight\na,1\na,2\n", "more than once.*'a'"),
        ("instance,weight\na,heavy\nb,1\n", r"nonnegative.*\['a'\]"),
        ("instance,weight\na,\n", r"\['a'\]"),
        ("instance,weight\na,1\nb,inf\nc,-inf\n", r"\['b', 'c'\]"),
        ("instance,weight\na,-1\nb,0\n", r"\['a'\]"),
    ],
)
def test_weights_refuse_a_malformed_view(
    tmp_path: Path, view: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        _release_with_weights(tmp_path, view).load_weights()


# --------------------------------------------------------------------------
# fetch_benchmark: no-op / install / replace against a stand-in Hub
# --------------------------------------------------------------------------


_OTHER_TASK_ID = "other_task"


def _download_into(local_dir: Path, filename: str, source: Path) -> str:
    """What a download into a directory does: the file in its place, and the
    hub's bookkeeping for it beneath ``.cache/``."""
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    metadata = (
        Path(local_dir) / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("commit\netag\n")
    return str(target)


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolve: Callable[[str], tuple[str, Path]],
    extra_listing: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], list[str]]:
    """Stand in for ``huggingface_hub`` and ``hf_xet``: ``resolve`` names each revision's tree.

    ``resolve`` maps a requested revision to the commit and the directory served
    under it; ``extra_listing`` names paths the repository lists without
    holding. Every served file except JSON and Markdown is Xet-stored, as on
    the Hub. Returns every repository lookup (revision and token) and every
    downloaded path, in order.
    """
    lookups: list[dict[str, object]] = []
    downloaded: list[str] = []
    roots: dict[str, Path] = {}

    class Api:
        def __init__(self, *, token: str | bool) -> None:
            self.token = token

        def repo_info(self, *, repo_id: str, repo_type: str, revision: str) -> Any:
            lookups.append({"revision": revision, "token": self.token})
            commit, roots[commit] = resolve(revision)
            return SimpleNamespace(sha=commit, private=False)

        def list_repo_tree(
            self, *, repo_id: str, repo_type: str, revision: str, recursive: bool
        ) -> list[Any]:
            assert recursive
            root = roots[revision]
            entries: list[Any] = [SimpleNamespace(path="benchmark-info")]  # a folder
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    xet = path.suffix not in {".json", ".md"}
                    entries.append(
                        SimpleNamespace(
                            path=relative,
                            size=path.stat().st_size,
                            xet_hash=f"{revision}:{relative}" if xet else None,
                        )
                    )
            entries += [
                SimpleNamespace(path=extra, size=0, xet_hash=None)
                for extra in extra_listing
            ]
            return entries

    def download(
        *, revision: str, filename: str, local_dir: Path, **kwargs: Any
    ) -> str:
        downloaded.append(filename)
        return _download_into(local_dir, filename, roots[revision] / filename)

    class Group:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["token_refresh_url"] == "refresh-route"

        def __enter__(self) -> Group:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def start_download_file(self, info: Any, target: str) -> None:
            commit, relative = info.hash.split(":", 1)
            downloaded.append(relative)
            shutil.copyfile(roots[commit] / relative, target)

    monkeypatch.setattr(
        fetch,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=Api,
            hf_hub_download=download,
            hf_hub_url=lambda *args, **kwargs: "file-url",
            get_hf_file_metadata=lambda url, token: SimpleNamespace(
                xet_file_data=SimpleNamespace(refresh_route="refresh-route")
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hf_xet",
        SimpleNamespace(
            XetSession=lambda: SimpleNamespace(new_file_download_group=Group),
            XetFileInfo=lambda hash, size: SimpleNamespace(hash=hash, size=size),
        ),
    )
    return lookups, downloaded


def _commit_of(tag: str) -> str:
    return hashlib.sha1(tag.encode()).hexdigest()


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Serve one small release per tag; return every repository lookup.

    Each requested tag resolves to its own commit (``_commit_of``) holding a
    manifest that names that tag as its version, and two tasks' registry
    subtrees; a requested commit of a published tag resolves to itself.
    """
    served = tmp_path / "hub"
    tags = {_commit_of(tag): tag for tag in ("rev-a", "rev-b")}

    def publish(revision: str) -> tuple[str, Path]:
        tag = tags.get(revision, revision)
        commit = _commit_of(tag)
        root = served / commit
        root.mkdir(parents=True, exist_ok=True)
        (root / RELEASE_MANIFEST_NAME).write_text(
            json.dumps({"version": tag, "manifest_id": f"manifest-{tag}", "tasks": []})
        )
        for task_id in (SIMPLE_TASK.id, _OTHER_TASK_ID):
            write_json_atomic(
                registry_document_path(
                    task_collection_directory(root, task_id), "instance", "toy"
                ),
                {"task": task_id, "kind": "instance", "entries": {}},
            )
        return commit, root

    lookups, _ = _install_fake_hub(monkeypatch, resolve=publish)
    return lookups


def _installed_version(workspace: Path) -> str:
    manifest = workspace_release_root(workspace) / RELEASE_MANIFEST_NAME
    return json.loads(manifest.read_text())["version"]


def test_fetch_benchmark_installs_the_default_revision(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    result = fetch_benchmark(workspace)
    assert result == workspace
    assert [lookup["revision"] for lookup in hub] == [LATEST_RELEASE_REVISION]
    assert _installed_version(workspace) == LATEST_RELEASE_REVISION


def test_fetch_benchmark_no_revision_is_a_noop_when_a_release_is_present(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    # Pre-installed by something else entirely (a test harness, a relayout
    # script) -- fetch_benchmark must not touch it or call the Hub.
    release_root = workspace_release_root(init_workspace(workspace))
    release_root.mkdir(parents=True)
    (release_root / RELEASE_MANIFEST_NAME).write_text("{}")

    result = fetch_benchmark(workspace)
    assert result == workspace
    assert hub == []


def test_fetch_benchmark_same_explicit_revision_is_a_noop(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision="rev-a")
    assert len(hub) == 1
    fetch_benchmark(workspace, revision="rev-a")
    assert len(hub) == 1  # no second Hub call


def test_fetch_benchmark_same_commit_is_a_noop(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    assert len(hub) == 1
    release_root = workspace_release_root(workspace)
    assert (release_root / INSTALLED_COMMIT_RECORD).read_text() == _commit_of("rev-a")
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    assert len(hub) == 1  # no second Hub call, so nothing downloaded
    assert _installed_version(workspace) == "rev-a"


def test_fetch_benchmark_the_tag_of_a_release_installed_by_commit_is_a_noop(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    fetch_benchmark(workspace, revision="rev-a")
    assert len(hub) == 1


def test_fetch_benchmark_a_different_commit_replaces_the_tree(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    marker = workspace_release_root(workspace) / "marker-from-rev-a"
    marker.write_text("x")
    fetch_benchmark(workspace, revision=_commit_of("rev-b"))
    assert len(hub) == 2
    assert not marker.exists()
    assert _installed_version(workspace) == "rev-b"
    record = workspace_release_root(workspace) / INSTALLED_COMMIT_RECORD
    assert record.read_text() == _commit_of("rev-b")


def test_fetch_benchmark_by_commit_reinstalls_a_tree_without_a_commit_record(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision="rev-a")
    (workspace_release_root(workspace) / INSTALLED_COMMIT_RECORD).unlink()
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    assert len(hub) == 2
    fetch_benchmark(workspace, revision=_commit_of("rev-a"))
    assert len(hub) == 2


def test_fetch_benchmark_a_different_revision_replaces_the_tree(
    tmp_path: Path, hub: list[dict[str, object]]
) -> None:
    workspace = tmp_path / "ws"
    fetch_benchmark(workspace, revision="rev-a")
    marker = workspace_release_root(workspace) / "marker-from-rev-a"
    marker.write_text("x")
    fetch_benchmark(workspace, revision="rev-b")
    assert len(hub) == 2
    assert not marker.exists()
    assert _installed_version(workspace) == "rev-b"


def test_fetch_benchmark_downloads_straight_into_the_tree_and_keeps_no_cache_copy(
    tmp_path: Path, hub: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    served = fetch.huggingface_hub
    calls: list[dict[str, Any]] = []

    def recording(**kwargs: Any) -> str:
        calls.append(kwargs)
        return served.hf_hub_download(**kwargs)

    monkeypatch.setattr(
        fetch,
        "huggingface_hub",
        SimpleNamespace(**{**vars(served), "hf_hub_download": recording}),
    )
    fetch_benchmark(workspace)
    release_root = workspace_release_root(workspace)
    assert calls
    for call in calls:
        # Into the staging tree beside the release, never through a download cache.
        assert "cache_dir" not in call
        assert Path(call["local_dir"]).is_relative_to(workspace)
    # The tree keeps only its commit record beneath the bookkeeping directory.
    bookkeeping = release_root / HUB_DOWNLOAD_METADATA_DIRECTORY
    assert [p.relative_to(release_root) for p in bookkeeping.rglob("*")] == [
        Path(INSTALLED_COMMIT_RECORD)
    ]
    assert {p.name for p in workspace.iterdir()} == {"local", release_root.name}
    assert _installed_version(workspace) == LATEST_RELEASE_REVISION


def test_fetch_refuses_a_released_path_beneath_the_download_bookkeeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = tmp_path / "hub"
    served.mkdir()
    (served / RELEASE_MANIFEST_NAME).write_text(
        json.dumps({"version": "v", "manifest_id": "m", "tasks": []})
    )
    _install_fake_hub(
        monkeypatch,
        resolve=lambda revision: ("0" * 40, served),
        extra_listing=(".cache/huggingface/x",),
    )
    with pytest.raises(ValueError, match="cannot install"):
        fetch_benchmark(tmp_path / "ws")


@pytest.mark.parametrize(
    "environment, token",
    [("secret-token", "secret-token"), (None, False)],
    ids=["from-environment", "anonymous"],
)
def test_fetch_benchmark_reads_its_token_only_from_the_environment(
    tmp_path: Path,
    hub: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    environment: str | None,
    token: str | bool,
) -> None:
    if environment is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", environment)
    fetch_benchmark(tmp_path / "ws")
    assert [lookup["token"] for lookup in hub] == [token]


def _two_task_tree(root: Path, digests: dict[str, str]) -> Path:
    """Write a release naming one Input artifact per task, each present."""
    (root / RELEASE_MANIFEST_NAME).parent.mkdir(parents=True, exist_ok=True)
    (root / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": "v1",
                "manifest_id": "m1",
                "tasks": {
                    task_id: {"sweep": f"{task_id}/sweeps/s.json"}
                    for task_id in digests
                },
            }
        )
    )
    for task_id, digest in digests.items():
        write_json_atomic(
            registry_document_path(
                task_collection_directory(root, task_id), "instance", "benchmark"
            ),
            {
                "task": task_id,
                "kind": "instance",
                "entries": {f"i-{task_id}": {"input_digest": digest}},
            },
        )
        artifact = input_artifact_path(root / INPUT_ARTIFACTS_DIRECTORY, digest)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(digest.encode())
    return root


def test_fetch_refuses_a_listed_xet_file_the_hub_serves_no_xet_data_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _two_task_tree(tmp_path / "served", {SIMPLE_TASK.id: "a" * 64})
    _install_fake_hub(monkeypatch, resolve=lambda _: ("c" * 40, served))
    hub = fetch.huggingface_hub
    monkeypatch.setattr(
        fetch,
        "huggingface_hub",
        SimpleNamespace(
            **{
                **vars(hub),
                "get_hf_file_metadata": lambda url, token: SimpleNamespace(
                    xet_file_data=None
                ),
            }
        ),
    )
    with pytest.raises(ValueError, match="no Xet data"):
        fetch_benchmark(tmp_path / "ws")


def test_fetch_installs_every_task_and_input_artifact_in_one_xet_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digests = {SIMPLE_TASK.id: "a" * 64, _OTHER_TASK_ID: "b" * 64}
    served = _two_task_tree(tmp_path / "served", digests)
    _, downloaded = _install_fake_hub(monkeypatch, resolve=lambda _: ("c" * 40, served))
    groups: list[str] = []
    xet = sys.modules["hf_xet"]
    session = xet.XetSession()

    def group(**kwargs: Any) -> Any:
        groups.append(kwargs["token_refresh_url"])
        return session.new_file_download_group(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hf_xet",
        SimpleNamespace(
            XetSession=lambda: SimpleNamespace(new_file_download_group=group),
            XetFileInfo=xet.XetFileInfo,
        ),
    )
    workspace = tmp_path / "ws"

    fetch_benchmark(workspace)

    release_root = workspace_release_root(workspace)
    served_files = sorted(
        p.relative_to(served).as_posix() for p in served.rglob("*") if p.is_file()
    )
    installed = sorted(
        p.relative_to(release_root).as_posix()
        for p in release_root.rglob("*")
        if p.is_file()
    )
    assert installed == sorted([*served_files, INSTALLED_COMMIT_RECORD])
    assert sorted(downloaded) == served_files
    # Every Xet-stored file shares one transfer group and so one access token.
    assert groups == ["refresh-route"]


def test_fetch_refuses_a_traversal_shaped_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _two_task_tree(tmp_path / "served", {SIMPLE_TASK.id: "a" * 64})
    _, downloaded = _install_fake_hub(
        monkeypatch,
        resolve=lambda _: ("c" * 40, served),
        extra_listing=("benchmark-info/../escape.json",),
    )
    workspace = tmp_path / "ws"

    with pytest.raises(ValueError, match="a path a fetch cannot install"):
        fetch_benchmark(workspace)

    assert downloaded == []
    assert not workspace_release_root(workspace).exists()


def test_failed_download_never_installs_a_partial_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _two_task_tree(tmp_path / "served", {SIMPLE_TASK.id: "a" * 64})
    _install_fake_hub(
        monkeypatch,
        resolve=lambda _: ("c" * 40, served),
        extra_listing=("missing-file.json",),
    )
    workspace = init_workspace(tmp_path / "ws")
    own_file = workspace / "local" / "notes.txt"
    own_file.write_text("keep my work")

    with pytest.raises(FileNotFoundError, match="missing-file.json"):
        fetch_benchmark(workspace)

    assert not workspace_release_root(workspace).exists()
    assert not list(workspace.glob(".official-release-*"))
    assert own_file.read_text() == "keep my work"


# --------------------------------------------------------------------------
# One source selector across TaskCollection, get_sweep_status vs
# load_results on a fetched workspace (official-release/ + local/)
# --------------------------------------------------------------------------


def _fetched_workspace(
    tmp_path: Path,
) -> tuple[Path, Path, str, str, ApproxDPBudget, SimpleAlgorithm]:
    """Build a workspace whose release and local trees answer different cells.

    The release covers (``instance_release``, algorithm, budget, seed 0);
    ``local/`` covers (``instance_local``, algorithm, budget, seed 0) with its
    own recorded Attempt. Neither tree answers the other's coordinate.

    Returns the workspace, the bare bundle it was cut into
    ``official-release/`` from, both instance UUIDs, the shared budget, and
    the shared algorithm.
    """
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    task_store = task_collection_directory(local_root, SIMPLE_TASK.id)
    algorithm = SimpleAlgorithm()
    register_algorithm(local_root, algorithm, registry="toy")
    (instance_local,) = install_instances(
        local_root, SIMPLE_TASK, [simple_producer()], registry="local"
    )
    budget = ApproxDPBudget(eps=1.0, delta=1e-5)
    write_attempt_partition(
        task_store,
        [
            current_result(
                instance_uuid=instance_local,
                alg_uuid=algorithm.uuid,
                budget=budget.dump_config().model_dump(),
            )
        ],
    )

    bundle = _minimal_bundle(tmp_path / "bundle-source", version="v-fixture")
    register_algorithm(bundle, algorithm, registry="toy")
    bundle_task_store = task_collection_directory(bundle, SIMPLE_TASK.id)
    (instance_release,) = install_instances(
        bundle, SIMPLE_TASK, [simple_producer()], seeds=(1,), registry="official"
    )
    write_result_partition(
        bundle_task_store,
        [
            current_result(
                instance_uuid=instance_release,
                alg_uuid=algorithm.uuid,
                budget=budget.dump_config().model_dump(),
            )
        ],
    )
    shutil.copytree(bundle, workspace_release_root(workspace))
    return workspace, bundle, instance_local, instance_release, budget, algorithm


@pytest.mark.parametrize(
    "source, expected_recorded",
    [("both", 2), ("release", 1), ("local", 1)],
)
def test_get_sweep_status_and_load_results_agree_on_a_fetched_workspace(
    tmp_path: Path, source: str, expected_recorded: int
) -> None:
    """A release row and a local row on a fetched workspace, resolved by source."""
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    # A release read resolves against the release's registries alone, so it
    # can name only what the release declares.
    instances = (
        [instance_release]
        if source == "release"
        else [instance_local, instance_release]
    )
    kwargs = dict(
        algorithms=[algorithm.uuid],
        instances=instances,
        budgets=[budget],
        seeds=[0],
    )

    frame = collection.load_results(results_source=source, **kwargs)
    status = collection.get_sweep_status(results_source=source, **kwargs)

    assert len(frame) == expected_recorded
    assert status["cells"]["recorded"] == expected_recorded
    assert status["cells"]["recorded"] == len(frame)


def test_a_release_read_cannot_name_a_local_only_declaration(tmp_path: Path) -> None:
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    kwargs = dict(
        algorithms=[algorithm.uuid],
        instances=[instance_local, instance_release],
        budgets=[budget],
        seeds=[0],
    )

    with pytest.raises(ValueError, match="unresolved instances") as refusal:
        collection.load_results(results_source="release", **kwargs)
    assert instance_local in str(refusal.value)
    status = collection.get_sweep_status(results_source="release", **kwargs)
    assert status["instances"]["unresolved"] == [instance_local]
    assert status["cells"]["recorded"] == 1


def test_bare_release_handle_local_is_empty_and_both_equals_release(
    tmp_path: Path,
) -> None:
    _, bundle, _, instance_release, budget, algorithm = _fetched_workspace(tmp_path)
    collection = open_collection(bundle, SIMPLE_TASK)
    kwargs = dict(
        algorithms=[algorithm.uuid],
        instances=[instance_release],
        budgets=[budget],
        seeds=[0],
    )

    assert collection.load_results(results_source="local", **kwargs).empty
    assert (
        collection.get_sweep_status(results_source="local", **kwargs)["cells"][
            "recorded"
        ]
        == 0
    )

    both = collection.get_sweep_status(results_source="both", **kwargs)
    release = collection.get_sweep_status(results_source="release", **kwargs)
    assert both["cells"]["recorded"] == release["cells"]["recorded"] == 1
    assert (
        collection.load_results(results_source="both", **kwargs)["status"].tolist()
        == collection.load_results(results_source="release", **kwargs)[
            "status"
        ].tolist()
    )


def _spy_schedule(executed: list[str]) -> CellScheduler:
    return observing_schedule(
        lambda context: executed.append(context.coordinate.instance.uuid)
    )


def test_run_sweep_default_source_skips_the_official_covered_cell(
    tmp_path: Path,
) -> None:
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    executed: list[str] = []

    collection.run_sweep(
        algorithms=[algorithm.uuid],
        instances=[instance_local, instance_release],
        budgets=[budget],
        seeds=[0],
        schedule_cells=_spy_schedule(executed),
    )

    assert instance_release not in executed


def test_run_sweep_source_local_runs_the_official_covered_cell(
    tmp_path: Path,
) -> None:
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    executed: list[str] = []

    collection.run_sweep(
        algorithms=[algorithm.uuid],
        instances=[instance_local, instance_release],
        budgets=[budget],
        seeds=[0],
        results_source="local",
        schedule_cells=_spy_schedule(executed),
    )

    assert instance_release in executed
    # The official cell now also has a local Attempt, so "both" and "local"
    # agree on its coverage.
    status_kwargs = dict(
        algorithms=[algorithm.uuid],
        instances=[instance_release],
        budgets=[budget],
        seeds=[0],
    )
    assert (
        collection.get_sweep_status(results_source="local", **status_kwargs)["cells"][
            "recorded"
        ]
        == 1
    )

    # instance_release is now answered under both trees: get_sweep_status
    # counts that one coordinate once (local-first precedence), while
    # load_results keeps both rows rather than merging them.
    both_status = collection.get_sweep_status(results_source="both", **status_kwargs)
    both_frame = collection.load_results(results_source="both", **status_kwargs)
    assert both_status["cells"]["recorded"] == 1
    assert len(both_frame) == 2
    assert sorted(both_frame["source"].tolist()) == ["local", "release"]


_STALE_KEY = "a3-STALEKEYFROMOLDHASH"


def _file_stale_local_registry(workspace: Path) -> Path:
    """File one local algorithm entry under a key its record no longer derives."""
    path = (
        task_collection_directory(workspace_local_root(workspace), SIMPLE_TASK.id)
        / registry_directory_name("algorithm")
        / "stale.json"
    )
    write_json_atomic(
        path,
        {
            "task": SIMPLE_TASK.id,
            "kind": "algorithm",
            "entries": {
                _STALE_KEY: SimpleAlgorithm(scale=3.0).dump_config().model_dump()
            },
        },
    )
    return path


def test_a_release_read_is_not_refused_by_a_stale_local_registry(
    tmp_path: Path,
) -> None:
    """Reading the release's own Results resolves against its registries alone."""
    workspace, _, instance_local, instance_release, budget, _ = _fetched_workspace(
        tmp_path
    )
    _file_stale_local_registry(workspace)
    collection = open_collection(workspace, SIMPLE_TASK)
    # No algorithm selection: every installed algorithm in scope is resolved,
    # which is exactly the read a stale registry used to refuse.
    kwargs = dict(instances=[instance_release], budgets=[budget], seeds=[0])

    frame = collection.load_results(results_source="release", **kwargs)
    status = collection.get_sweep_status(results_source="release", **kwargs)

    assert frame["source"].tolist() == ["release"]
    assert frame["instance_uuid"].tolist() == [instance_release]
    assert status["cells"]["recorded"] == 1
    # A local Result may reference a released declaration, so a local or dual
    # read still resolves across every installed tree and meets the stale entry.
    for source in ("local", "both"):
        with pytest.raises(ValueError, match=_STALE_KEY):
            collection.load_results(
                results_source=source,
                instances=[instance_local, instance_release],
                budgets=[budget],
                seeds=[0],
            )


def test_a_registry_refusal_names_the_file_and_the_entry(tmp_path: Path) -> None:
    workspace, *_ = _fetched_workspace(tmp_path)
    stale = _file_stale_local_registry(workspace)
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError) as refusal:
        collection.load_results()
    assert _STALE_KEY in str(refusal.value)
    assert str(stale) in str(refusal.value)

    with pytest.raises(ValueError) as refusal:
        collection.load_algorithms()
    assert _STALE_KEY in str(refusal.value)
    assert str(stale) in str(refusal.value)

    with pytest.warns(UserWarning, match=re.escape(str(stale))):
        skipped = collection.load_results(strict=False)
    assert _STALE_KEY not in set(skipped["alg_uuid"])


def test_an_unimportable_registry_entry_refusal_names_the_file(tmp_path: Path) -> None:
    workspace, *_ = _fetched_workspace(tmp_path)
    path = (
        task_collection_directory(workspace_local_root(workspace), SIMPLE_TASK.id)
        / registry_directory_name("algorithm")
        / "elsewhere.json"
    )
    write_json_atomic(
        path,
        {
            "task": SIMPLE_TASK.id,
            "kind": "algorithm",
            "entries": {_STALE_KEY: {"path": "no_such_module:Missing", "config": {}}},
        },
    )
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError, match="Could not reconstruct") as refusal:
        collection.load_results()
    assert _STALE_KEY in str(refusal.value)
    assert str(path) in str(refusal.value)


def test_a_release_read_with_no_release_installed_resolves_nothing(
    tmp_path: Path,
) -> None:
    workspace = init_workspace(tmp_path / "ws")
    algorithm = SimpleAlgorithm()
    register_algorithm(workspace_local_root(workspace), algorithm, registry="toy")
    collection = open_collection(workspace, SIMPLE_TASK)

    assert collection.load_results(results_source="release").empty
    with pytest.raises(ValueError, match="Unresolved algorithms") as refusal:
        collection.load_results(results_source="release", algorithms=[algorithm.uuid])
    assert algorithm.uuid in str(refusal.value)


def test_run_sweep_reconciles_a_release_only_selected_registry(
    tmp_path: Path,
) -> None:
    """An empty-groups reconcile must resolve the Task across both trees.

    A selected algorithm registry can be installed only in
    ``official-release/``. An abandoned local claim plus an empty instance
    selection, which resolves no coordinate to read a Task from, must still
    reconcile from that cross-tree selection.
    """
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    algorithm = SimpleAlgorithm()

    bundle = _minimal_bundle(tmp_path / "bundle-source", version="v-fixture")
    register_algorithm(bundle, algorithm, registry="release-only")
    shutil.copytree(bundle, workspace_release_root(workspace))

    collection = open_collection(workspace, SIMPLE_TASK)
    instance = collection.add_instance(*simple_instance(0))
    task_store = task_collection_directory(local_root, SIMPLE_TASK.id)
    stale_claim = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        alg_seed=0,
    )
    abandon_claim(task_store, stale_claim)

    result = collection.run_sweep(
        {"budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)], "seeds": [0]},
        instances=[],
        algorithm_registries=["release-only"],
    )

    assert result.empty
    assert collection.reconcile() == []


def test_run_sweep_default_source_returns_release_local_and_new_rows(
    tmp_path: Path,
) -> None:
    """run_sweep's return must include release-answered cells, not just local ones.

    ``results_source="both"`` skips a coordinate the release already answers, but the
    returned frame must still report it, alongside a coordinate answered
    locally before this call and one this call runs fresh -- exactly the
    frame ``load_results`` would report for the same resolved selection.
    """
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    fresh_instance = collection.add_instance(*simple_instance(2))

    both = collection.run_sweep(
        algorithms=[algorithm.uuid],
        instances=[instance_local, instance_release, fresh_instance.uuid],
        budgets=[budget],
        seeds=[0],
    )

    assert len(both) == 3
    assert list(both["status"].unique()) == [ResultStatus.SUCCESS.value]
    by_instance = dict(zip(both["instance_uuid"], both["source"]))
    assert by_instance == {
        instance_local: "local",
        instance_release: "release",
        fresh_instance.uuid: "local",
    }

    local_only = collection.run_sweep(
        results_source="local",
        algorithms=[algorithm.uuid],
        instances=[instance_local, instance_release, fresh_instance.uuid],
        budgets=[budget],
        seeds=[0],
    )
    # instance_release has no local Result, so results_source="local" excludes the
    # release row and runs that coordinate locally too.
    assert len(local_only) == 3
    assert set(local_only["source"]) == {"local"}


def test_run_sweep_source_release_does_not_reuse_a_local_only_leaf(
    tmp_path: Path,
) -> None:
    """A local-only Result must not silently answer a results_source="release" run.

    The outer pending decision already excludes a local-only Result under
    ``results_source="release"``; leaf reuse must honor the same exclusion instead of
    always trusting the local task collection.
    """
    workspace, _, instance_local, _, budget, algorithm = _fetched_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    executed: list[str] = []

    result = collection.run_sweep(
        algorithms=[algorithm.uuid],
        instances=[instance_local],
        budgets=[budget],
        seeds=[0],
        results_source="release",
        schedule_cells=_spy_schedule(executed),
    )

    assert instance_local in executed
    # results_source="release" reports exactly what load_results(results_source="release")
    # would for this selection: nothing, since instance_local has no release
    # Result. The fresh execution is still durably written to local/.
    assert result.empty
    local_view = collection.load_results(
        algorithms=[algorithm.uuid],
        instances=[instance_local],
        budgets=[budget],
        seeds=[0],
        results_source="local",
    )
    assert local_view["status"].tolist() == [ResultStatus.SUCCESS.value]
    assert local_view["source"].tolist() == ["local"]


def test_load_results_missing_coverage_counts_cells_not_rows(
    tmp_path: Path,
) -> None:
    """The finite-coverage warning must dedup a coordinate answered under both trees.

    Four instances: one answered under both trees (an overlap, two rows, one
    cell), one answered only locally (one row, one cell), and two answered
    nowhere (a two-cell gap). Counting ``frame``'s rows instead of distinct
    answered cells would report only "1 missing of 4" here -- undercounting
    by one and burying half the real gap -- instead of the correct 2, which
    ``get_sweep_status``'s own dedup reports.
    """
    workspace, _, instance_local, instance_release, budget, algorithm = (
        _fetched_workspace(tmp_path)
    )
    collection = open_collection(workspace, SIMPLE_TASK)
    # Answer instance_release locally too, so it overlaps both trees.
    collection.run_sweep(
        results_source="local",
        algorithms=[algorithm.uuid],
        instances=[instance_release],
        budgets=[budget],
        seeds=[0],
    )
    gap_instances = [
        collection.add_instance(*simple_instance(3)),
        collection.add_instance(*simple_instance(4)),
    ]
    selection = [instance_local, instance_release, *(g.uuid for g in gap_instances)]

    with pytest.warns(UserWarning, match=r"2 missing of 4"):
        frame = collection.load_results(
            algorithms=[algorithm.uuid],
            instances=selection,
            budgets=[budget],
            seeds=[0],
        )

    status = collection.get_sweep_status(
        algorithms=[algorithm.uuid],
        instances=selection,
        budgets=[budget],
        seeds=[0],
    )
    assert status["cells"]["expected"] == 4
    assert status["cells"]["recorded"] == 2
    assert status["cells"]["never_attempted"] == 2
    assert len(frame) == 3


def _relabelled_registry_workspace(tmp_path: Path) -> tuple[Path, str]:
    """A workspace whose local Roster relabels an Algorithm the release also installs.

    Both trees install the same configured Algorithm (so the same UUID)
    under the same registry name; only the label differs. Returns the
    workspace and that Algorithm's UUID.
    """
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    register_algorithm(
        local_root, SimpleAlgorithm(label="TailRobustSparseHuberIHT"), registry="roster"
    )

    bundle = _minimal_bundle(tmp_path / "bundle-source", version="v-fixture")
    register_algorithm(bundle, SimpleAlgorithm(), registry="roster")
    shutil.copytree(bundle, workspace_release_root(workspace))
    return workspace, SimpleAlgorithm().uuid


def test_get_sweep_status_tolerates_a_relabelled_shared_registry_entry(
    tmp_path: Path,
) -> None:
    """A local relabel of a released Algorithm resolves under the default source."""
    workspace, uuid = _relabelled_registry_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)

    status = collection.get_sweep_status(
        algorithm_registries=["roster"],
        budgets=[ApproxDPBudget(eps=1.0, delta=1e-5)],
        seeds=[0],
    )

    assert status["algorithms"]["resolved"] == 1
    algorithms = collection.load_algorithms(registries=["roster"])
    assert set(algorithms) == {uuid}
    assert algorithms[uuid].uuid == uuid
    assert algorithms[uuid].label == "TailRobustSparseHuberIHT"


def test_explicit_registry_selection_keeps_local_tree_precedence(
    tmp_path: Path,
) -> None:
    """Group request order cannot let a release label outrank the local tree."""
    workspace = init_workspace(tmp_path / "ws")
    local_root = workspace_local_root(workspace)
    local = SimpleAlgorithm(label="local roster")
    register_algorithm(local_root, local, registry="local-only")

    bundle = _minimal_bundle(tmp_path / "bundle")
    register_algorithm(
        bundle, SimpleAlgorithm(label="release roster"), registry="release-only"
    )
    shutil.copytree(bundle, workspace_release_root(workspace))

    algorithms = open_collection(workspace, SIMPLE_TASK).load_algorithms(
        registries=("release-only", "local-only")
    )

    assert algorithms[local.uuid].label == "local roster"


def test_reconcile_resolves_a_custom_task_across_relabelled_registries(
    tmp_path: Path,
) -> None:
    """Custom Task resolution accepts label-only differences across trees."""
    workspace, uuid = _relabelled_registry_workspace(tmp_path)
    collection = open_collection(workspace, SIMPLE_TASK)
    assert collection.reconcile() == []
    algorithms = collection.load_algorithms(registries=["roster"])
    assert algorithms[uuid].label == "TailRobustSparseHuberIHT"


def test_reconcile_clears_builtin_claims_without_loading_algorithm_registries(
    tmp_path: Path,
) -> None:
    """An unavailable released Algorithm cannot block a built-in Task's claims."""
    workspace = init_workspace(tmp_path / "ws")
    algorithm = SampleMean()
    release_root = workspace_release_root(workspace)
    register_algorithm(release_root, algorithm, registry="unavailable")
    registry_path = registry_document_path(
        task_collection_directory(release_root, algorithm.task.id),
        "algorithm",
        "unavailable",
    )
    payload = json.loads(registry_path.read_text())
    payload["entries"][algorithm.uuid]["path"] = "uninstalled_algorithm:Ghost"
    registry_path.write_text(json.dumps(payload))
    collection = open_collection(workspace, algorithm.task)
    instance = collection.add_instance(
        MeanEstimationMetadata(
            n_samples=2, domain=BoxDomain(bounds=np.array([[0.0, 1.0]]))
        ),
        MeanEstimationInput(X=np.array([[0.25], [0.75]])),
        {"ground_truth": [0.5]},
    )
    coordinate = CellCoordinate(
        instance=instance,
        algorithm=algorithm,
        budget=ApproxDPBudget(eps=1.0, delta=1e-5),
        alg_seed=0,
    )
    task_store = task_collection_directory(
        workspace_local_root(workspace), algorithm.task.id
    )
    abandon_claim(task_store, coordinate)

    assert collection.reconcile() == [coordinate.uuid]
    assert collection.reconcile() == []
    assert not list(attempts_directory(task_store).rglob("*.parquet"))


def test_get_sweep_status_still_refuses_a_genuine_cross_tree_difference(
    tmp_path: Path,
) -> None:
    """A non-label difference under one UUID remains a refused conflict.

    A genuine (non-label) content difference under one key also means that
    key does not match what the changed content derives, so this is
    reported as a misfiled entry -- the merge's own reconciling verification
    path, not a different one.
    """
    workspace, uuid = _relabelled_registry_workspace(tmp_path)
    release_registry = registry_document_path(
        task_collection_directory(workspace_release_root(workspace), SIMPLE_TASK.id),
        "algorithm",
        "roster",
    )
    payload = json.loads(release_registry.read_text())
    payload["entries"][uuid]["config"]["scale"] = 99.0
    release_registry.write_text(json.dumps(payload))
    collection = open_collection(workspace, SIMPLE_TASK)

    with pytest.raises(ValueError, match="does not match derived UUID"):
        collection.get_sweep_status(
            algorithm_registries=["roster"],
            budgets=[ApproxDPBudget(eps=1.0, delta=1e-5)],
            seeds=[0],
        )


def test_source_rejects_an_unknown_value(tmp_path: Path) -> None:
    workspace = init_workspace(tmp_path / "ws")
    collection = open_collection(workspace, SIMPLE_TASK)
    for call in (
        lambda: collection.load_results(results_source="bogus"),  # type: ignore[arg-type]
        lambda: collection.get_sweep_status(results_source="bogus"),  # type: ignore[arg-type]
        lambda: collection.list_sweeps(source="bogus"),  # type: ignore[arg-type]
        lambda: collection.list_algorithm_registries(source="bogus"),  # type: ignore[arg-type]
        lambda: collection.list_instance_registries(source="bogus"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="source must be"):
            call()
