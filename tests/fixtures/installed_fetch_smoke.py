"""Installed-wheel fetch smoke against one local immutable fake Hub."""

from __future__ import annotations

from operator import attrgetter
from typing import Any

from importlib import import_module
import json
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from types import SimpleNamespace

from dpstats import fetch_benchmark
from dpstats.workspace import TaskCollection
from dpstats.analysis import measure_results

from dpstats.workspace.fetch import (
    DISTRIBUTION_RELEASE_REPOSITORY,
    HUB_DOWNLOAD_METADATA_DIRECTORY,
    INSTALLED_COMMIT_RECORD,
)
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
    INPUT_ARTIFACTS_DIRECTORY,
    PROVENANCE_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    input_artifact_name,
    result_partition_directory,
    sweeps_directory,
    task_collection_directory,
    withheld_profiles_path,
    workspace_local_root,
    workspace_release_root,
)
from dpstats.workspace.provenance import read_provenance
from dpstats.workspace.registries import registry_document_path
from dpstats.results import result_files
from dpstats.sweep.coordinates import load_sweep_coordinates
from dpstats.tasks import list_built_in_metrics
from dpstats.tasks import MeanEstimation


STORAGE_ROOT = Path("benchmark-storage")
TASK_COLLECTIONS_ROOT = workspace_local_root(STORAGE_ROOT)
INPUT_ARTIFACTS_ROOT = TASK_COLLECTIONS_ROOT / INPUT_ARTIFACTS_DIRECTORY
TASK_STORE = task_collection_directory(TASK_COLLECTIONS_ROOT, MeanEstimation.id)
SWEEP = sweeps_directory(TASK_STORE) / "simulated.json"
RELEASE = Path("distribution-release")
PROVENANCE_REF = (
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/demonstration-generator.json"
)
PINNED_REVISION = "e" * 40
METRIC = "l2_error_to_ground_truth"

# The schema the benchmark's own standings view is stored with. A clean
# library install defines none of it, so this smoke states the shape it writes.
VIEW_SCHEMA = pa.schema(
    [
        ("task", pa.string()),
        ("algorithm", pa.string()),
        ("alg_uuid", pa.string()),
        ("budget", pa.string()),
        ("metric", pa.string()),
        ("baseline", pa.string()),
        ("tau", pa.float64()),
        ("fraction", pa.float64()),
        ("run_count", pa.int64()),
        ("measured_run_count", pa.int64()),
    ]
)


def main() -> None:
    """Build one release, serve it locally, and fetch through the root API."""
    release_task = task_collection_directory(RELEASE, MeanEstimation.id)
    for source in (
        registry_document_path(TASK_STORE, "algorithm", "simulated"),
        registry_document_path(TASK_STORE, "instance", "simulated"),
        SWEEP,
    ):
        destination = release_task / source.relative_to(TASK_STORE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    instance_payload = json.loads(
        registry_document_path(TASK_STORE, "instance", "simulated").read_text()
    )
    for record in instance_payload["entries"].values():
        name = input_artifact_name(record["input_digest"])
        destination = RELEASE / INPUT_ARTIFACTS_DIRECTORY / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(INPUT_ARTIFACTS_ROOT / name, destination)
    # A release carries one resolved Result per released cell, and no Attempt
    # history: the working store's Attempts are resolved once, here.
    files, _ = result_files(
        TASK_STORE,
        MeanEstimation,
        load_sweep_coordinates(
            SWEEP,
            TASK_COLLECTIONS_ROOT,
            MeanEstimation,
            algorithm_registries=("simulated",),
            instance_registries=("simulated",),
        ),
    )
    for alg_uuid, data in files.items():
        destination = (
            result_partition_directory(release_task, alg_uuid) / "results.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    # The example draws every Instance locally from one generator shipped under
    # this project's license, so one study document names all of them.
    provenance = RELEASE / PROVENANCE_REF
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "producer": "the example's demonstration_instance producer",
                "data_license": "MIT",
                "discipline": "simulated demonstration",
                "instances": {
                    uuid: {"sample_unit": "simulated draw", "source": {}}
                    for uuid in instance_payload["entries"]
                },
            }
        )
        + "\n"
    )

    collection = TaskCollection.from_sweep(SWEEP, STORAGE_ROOT)
    measurements = measure_results(
        collection.load_results(
            algorithm_registries=("simulated",),
            instance_registries=("simulated",),
        ),
        list_built_in_metrics(MeanEstimation)[METRIC],
    )
    # The view renders its own ruled identity columns from the configured
    # values the measurement frame carries.
    measurements = measurements.assign(
        budget=measurements["budget"].map(repr),
        alg_uuid=measurements["algorithm"].map(attrgetter("uuid")),
        algorithm=measurements["algorithm"].map(str),
    )
    counts = (
        measurements.groupby(
            ["task", "algorithm", "alg_uuid", "budget"],
            as_index=False,
            sort=True,
        )
        .agg(run_count=(METRIC, "size"))
        .assign(metric=METRIC)
    )
    # What a release's extra files mean belongs to the benchmark that defines
    # them; a clean library install carries none of that, so this smoke writes
    # them as the opaque bytes a fetch installs.
    view = pd.concat(
        [
            counts.assign(
                baseline="best_median",
                tau=tau / 100,
                fraction=1.0,
                measured_run_count=counts["run_count"],
            )
            for tau in range(1001)
        ],
        ignore_index=True,
    )
    view_path = withheld_profiles_path(release_task)
    view_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(
            view[VIEW_SCHEMA.names], schema=VIEW_SCHEMA, preserve_index=False
        ).replace_schema_metadata(None),
        view_path,
        compression="zstd",
        compression_level=9,
    )

    (RELEASE / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME).parent.mkdir(
        parents=True, exist_ok=True
    )
    (RELEASE / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME).write_text(
        json.dumps({"comparison": {MeanEstimation.id: {"headline_metric": METRIC}}}),
        encoding="utf-8",
    )
    (RELEASE / BENCHMARK_INFO_DIRECTORY / BENCHMARK_NARRATIVE_NAME).write_text(
        "# Benchmark protocol\n\nHow this simulated cut is read.\n", encoding="utf-8"
    )
    # A release's manifest names its version, a content identity, and each
    # task's Sweep; the library reads nothing else from it.
    (RELEASE / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": "simulated-v1",
                "manifest_id": "a" * 64,
                "tasks": {
                    MeanEstimation.id: {
                        "sweep": SWEEP.relative_to(TASK_COLLECTIONS_ROOT).as_posix()
                    }
                },
            }
        )
        + "\n"
    )

    # The fake Hub checks only the fetch contract: every read is anonymous and
    # addresses the official repository, and the requested tag resolves once
    # to the commit every later read is pinned to.
    downloads: list[tuple[str, str]] = []

    class Api:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["token"] is False

        def repo_info(self, **kwargs: Any) -> SimpleNamespace:
            assert kwargs["repo_id"] == DISTRIBUTION_RELEASE_REPOSITORY
            assert kwargs["revision"] == "simulated-v1"
            return SimpleNamespace(sha=PINNED_REVISION, private=False)

        def list_repo_tree(self, **kwargs: Any) -> list[SimpleNamespace]:
            assert kwargs["repo_id"] == DISTRIBUTION_RELEASE_REPOSITORY
            assert kwargs["revision"] == PINNED_REVISION
            # Every file stored outside Xet, so each takes the plain download.
            return [
                SimpleNamespace(path=relative, size=0, xet_hash=None)
                for relative in (
                    ".gitattributes",
                    *sorted(
                        path.relative_to(RELEASE).as_posix()
                        for path in RELEASE.rglob("*")
                        if path.is_file()
                    ),
                )
            ]

    def download(**kwargs: Any) -> str:
        filename = str(kwargs["filename"])
        revision = str(kwargs["revision"])
        downloads.append((filename, revision))
        assert kwargs["repo_id"] == DISTRIBUTION_RELEASE_REPOSITORY
        assert kwargs["token"] is False
        # A download into a directory: the file in place, bookkeeping under .cache/.
        target = Path(kwargs["local_dir"]) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RELEASE / filename, target)
        metadata = (
            Path(kwargs["local_dir"])
            / ".cache"
            / "huggingface"
            / f"{filename}.metadata"
        )
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("commit\n")
        return str(target)

    fetch_transport = import_module("dpstats.workspace.fetch")
    setattr(
        fetch_transport,
        "huggingface_hub",
        SimpleNamespace(HfApi=Api, hf_hub_download=download),
    )
    installed_root = Path("fetched-benchmark")
    installed = fetch_benchmark(installed_root, revision="simulated-v1")
    installed_release = workspace_release_root(installed_root)

    assert installed == installed_root
    assert all(revision == PINNED_REVISION for _, revision in downloads)
    assert set(installed_release.iterdir()) == {
        installed_release / RELEASE_MANIFEST_NAME,
        installed_release / BENCHMARK_INFO_DIRECTORY,
        installed_release / INPUT_ARTIFACTS_DIRECTORY,
        installed_release / MeanEstimation.id,
        installed_release / HUB_DOWNLOAD_METADATA_DIRECTORY,
    }
    assert (installed_release / INSTALLED_COMMIT_RECORD).read_text() == PINNED_REVISION
    # Apart from its commit record, a fetch installs the tree the release
    # repository holds and nothing else, so a fetched tree and a local cut of
    # the same release are one input.
    fetched_files = {
        path.relative_to(installed_release).as_posix()
        for path in installed_release.rglob("*")
        if path.is_file()
        and path.relative_to(installed_release).parts[0]
        != HUB_DOWNLOAD_METADATA_DIRECTORY
    }
    released_files = {
        path.relative_to(RELEASE).as_posix()
        for path in RELEASE.rglob("*")
        if path.is_file()
    }
    assert fetched_files == released_files
    differing = sorted(
        relative
        for relative in released_files
        if (installed_release / relative).read_bytes()
        != (RELEASE / relative).read_bytes()
    )
    assert differing == []

    assert (installed_release / PROVENANCE_REF).is_file()
    assert set(read_provenance(installed_release).instances) == set(
        instance_payload["entries"]
    )
    print(
        json.dumps(
            {
                "downloaded_at_pinned_revision": True,
                "installed": installed.as_posix(),
                "release_manifest_installed": True,
                "tree_matches_local_cut": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
