"""One release is prepared once; the build then renders that snapshot alone."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
import pyarrow.parquet as pq
import numpy as np
from pydantic import JsonValue

import site_copy

from dpstats.tasks import MeanEstimation
from dpstats import open_collection
from dpstats.metric import MetricArgument
from dpstats.task import TaskInput, TaskMetadata
from dpstats.workspace.layout import (
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
)

from tests.unit.docs.conftest import (
    MEASURED_METRIC,
    RELEASED_METRICS,
    WriteReleaseRoot,
    comparison_entry,
)
from release_files import (
    withheld_profiles_relative,
    write_protocol,
)
from prepare_results import (
    PREPARED_NAME,
    PREPARED_VERSION,
    prepared_snapshot,
    read_manifest,
    read_instance_arguments,
    read_prepared,
)
from results_roots import release_input
import prepare_results

TASK = MeanEstimation.id


def resolve(root: Path):
    """Resolve one local release, as a build does from its environment."""
    release = release_input(None, str(root), cache_dir=root / "cache")
    assert release is not None
    return release


@pytest.fixture(scope="session")
def snapshot(
    measurable_release_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The prepared snapshot of the measurable release, taken once."""
    cache = tmp_path_factory.mktemp("prepared-cache")
    return prepared_snapshot(resolve(measurable_release_root), cache=cache)


# ── Preparing once ───────────────────────────────────────────────────────────


def test_a_prepared_snapshot_is_reused_rather_than_measured_again(
    measurable_release_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    release = resolve(measurable_release_root)
    first = prepared_snapshot(release, cache=cache)
    import panel_data

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("A prepared snapshot was measured a second time.")

    monkeypatch.setattr(panel_data, "task_data", refuse)
    assert prepared_snapshot(release, cache=cache) == first


def test_a_different_release_prepares_a_different_snapshot(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    cache = tmp_path / "cache"
    one = write_release_root(
        tmp_path / "one", published={TASK: [MEASURED_METRIC]}, version="cut-1"
    )
    other = write_release_root(
        tmp_path / "other", published={TASK: [MEASURED_METRIC]}, version="cut-2"
    )
    assert prepared_snapshot(resolve(one), cache=cache) != prepared_snapshot(
        resolve(other), cache=cache
    )


def test_one_release_prepares_to_one_name_wherever_its_tree_sits(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    """The snapshot is named for the release bytes, never for their directory."""
    here = write_release_root(tmp_path / "here", published={TASK: [MEASURED_METRIC]})
    there = tmp_path / "there"
    shutil.copytree(here, there)
    assert (
        prepared_snapshot(resolve(here), cache=tmp_path / "one").name
        == prepared_snapshot(resolve(there), cache=tmp_path / "two").name
    )


def test_the_snapshot_name_follows_every_site_module_that_writes_it() -> None:
    """The configuration provenance writer shapes the snapshot's bytes too."""
    names = {path.name for path in prepare_results._site_modules()}
    assert {"algorithm_provenance.py", "api_reference.py", "panel_data.py"} <= names


def test_a_new_snapshot_format_prepares_under_a_new_name(
    tmp_path: Path,
    write_release_root: WriteReleaseRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = write_release_root(tmp_path / "root", published={TASK: [MEASURED_METRIC]})
    release = resolve(root)
    before = prepare_results._snapshot_name(release)
    monkeypatch.setattr(prepare_results, "PREPARED_VERSION", PREPARED_VERSION + 1)
    assert prepare_results._snapshot_name(release) != before


def test_new_baseline_names_prepare_under_a_new_name(
    tmp_path: Path,
    write_release_root: WriteReleaseRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline names are written into the snapshot, so rewording one re-prepares."""
    root = write_release_root(tmp_path / "root", published={TASK: [MEASURED_METRIC]})
    release = resolve(root)
    before = prepare_results._snapshot_name(release)
    copy = tmp_path / "site-copy.yml"
    copy.write_text(json.dumps({"baselines": {"none": "Raw cost"}}))
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", copy)
    assert prepare_results._snapshot_name(release) != before


# ── What the snapshot carries ────────────────────────────────────────────────


def test_the_snapshot_carries_measured_panels_and_the_shipped_standings(
    snapshot: Path,
) -> None:
    data, standings, missing, refused = read_prepared(snapshot, TASK)
    assert missing == []
    assert refused == []
    assert data is not None and standings is not None
    assert sorted({panel.metric for panel in data.panels}) == sorted(RELEASED_METRICS)
    assert data.index["headline_metric"] == MEASURED_METRIC
    assert {panel.metric for panel in standings.panels} == {MEASURED_METRIC}


def test_distribution_prepares_while_evaluation_is_pending(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    source = tmp_path / "release"
    shutil.copytree(measurable_release_root, source)
    release = resolve(source)
    results = release.tasks[TASK].results
    for state in results["evaluation"].values():
        state.update(
            state="pending", recorded=0, reason="awaiting validated collection"
        )
    view = source / withheld_profiles_relative(TASK)
    pq.write_table(pq.read_table(view).slice(0, 0), view)

    prepared = prepared_snapshot(release, cache=tmp_path / "cache")
    data, standings, missing, refused = read_prepared(prepared, TASK)
    assert data is not None and data.panels
    assert standings is not None and standings.panels == []
    assert missing == refused == []
    skipped = read_manifest(prepared)["tasks"][TASK]["skipped"]
    pending = [row for row in skipped if row["kind"] == "pending_evaluation"]
    assert {row["uuid"] for row in pending} == set(results["evaluation"])
    assert all(row["reason"] == "awaiting validated collection" for row in pending)


def test_evaluation_prepares_while_distribution_is_pending(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    source = tmp_path / "release"
    shutil.copytree(measurable_release_root, source)
    release = resolve(source)
    for state in release.tasks[TASK].results["distribution"].values():
        state.update(
            state="pending", recorded=0, reason="replacement collection pending"
        )
    for folder in source.glob(f"**/{TASK}/results"):
        shutil.rmtree(folder)

    prepared = prepared_snapshot(release, cache=tmp_path / "cache")
    data, standings, _, _ = read_prepared(prepared, TASK)
    assert data is None
    assert standings is not None and standings.panels


def test_the_snapshot_carries_no_narrative(snapshot: Path) -> None:
    """The site renders the library's methodology, never the release's copy."""
    assert not (snapshot / BENCHMARK_NARRATIVE_NAME).exists()
    assert "narrative" not in read_manifest(snapshot)


def test_a_snapshot_renders_without_the_release_it_was_taken_from(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    source = tmp_path / "release"
    shutil.copytree(measurable_release_root, source)
    prepared = prepared_snapshot(resolve(source), cache=tmp_path / "cache")
    collection = open_collection(source, MeanEstimation)
    instances = collection.load_instances()
    expected = {
        uuid: np.mean(collection.load_input(instance).X, axis=0)
        for uuid, instance in instances.items()
    }
    shutil.rmtree(source)
    data, standings, _, _ = read_prepared(prepared, TASK)
    assert data is not None and standings is not None
    assert data.files
    arguments = read_instance_arguments(prepared)[TASK]
    for uuid, mean in expected.items():
        assert arguments[uuid]["sample_mean"] == pytest.approx(mean)
        assert (
            arguments[uuid]["ground_truth"]
            == instances[uuid].metric_arguments["ground_truth"]
        )
        assert "sample_mahalanobis_geometry" in arguments[uuid]


def test_instance_pages_reuse_arguments_computed_for_measurements(
    measurable_release_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dpstats.metric as declarations

    produce = declarations.produce_arguments
    calls: list[TaskMetadata] = []

    def counted(
        arguments: Iterable[MetricArgument],
        metadata: TaskMetadata,
        input_data: TaskInput,
    ) -> dict[str, JsonValue]:
        calls.append(metadata)
        return produce(arguments, metadata, input_data)

    monkeypatch.setattr(declarations, "produce_arguments", counted)
    prepared = prepared_snapshot(
        resolve(measurable_release_root), cache=tmp_path / "cache"
    )
    instances = open_collection(
        measurable_release_root, MeanEstimation
    ).load_instances()
    assert len(calls) == len(instances)
    assert set(read_instance_arguments(prepared)[TASK]) == set(instances)


# ── What a snapshot refuses ──────────────────────────────────────────────────


def test_a_changed_snapshot_file_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(tmp_path / "release", published={TASK: [MEASURED_METRIC]})
    prepared = prepared_snapshot(resolve(root), cache=tmp_path / "cache")
    changed = next(prepared.glob(f"{TASK}/*.json"))
    changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        read_prepared(prepared, TASK)


def test_a_snapshot_from_another_format_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(tmp_path / "release", published={TASK: [MEASURED_METRIC]})
    prepared = prepared_snapshot(resolve(root), cache=tmp_path / "cache")
    manifest = json.loads((prepared / PREPARED_NAME).read_text(encoding="utf-8"))
    manifest["version"] = PREPARED_VERSION + 1
    (prepared / PREPARED_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the format"):
        read_prepared(prepared, TASK)


def test_a_directory_holding_no_snapshot_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No prepared snapshot"):
        read_prepared(tmp_path, TASK)


def test_a_snapshot_file_edited_by_hand_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    """Snapshot names track the manifest, so a task's own bytes still verify."""
    root = write_release_root(tmp_path / "release", published={TASK: [MEASURED_METRIC]})
    prepared = prepared_snapshot(resolve(root), cache=tmp_path / "cache")
    # The prepared snapshot is the site's own flat cache shape, not a release
    # tree, so its protocol sits bare at its root.
    (prepared / BENCHMARK_PROTOCOL_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        read_prepared(prepared, TASK)


# ── A release and a library that disagree on a Metric ─────────────────────


def _renamed_metric(root: Path) -> None:
    """State a Metric this library defines none of."""
    write_protocol(
        root,
        {TASK: comparison_entry("undefined_metric")},
    )


def test_a_release_naming_an_undefined_metric_is_refused_by_name(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    """A release and this library must agree on the Metrics it released."""
    root = tmp_path / "stale"
    shutil.copytree(measurable_release_root, root)
    _renamed_metric(root)

    with pytest.raises(ValueError, match="undefined_metric.*does not define"):
        prepared_snapshot(resolve(root), cache=tmp_path / "cache")
