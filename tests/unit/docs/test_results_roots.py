"""The one input a documentation build reads, and the routes it opens.

A build reads exactly one Distribution Release tree. The release declares the
tasks it released and ships its standings view and Instance weights for each;
its protocol states each task's ranking and external baselines, and a current
protocol may supply newer external baselines beside the shipped ranking.
Nothing here chooses a task, a metric, a baseline or a threshold.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dpstats.tasks import MeanEstimation
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_PROTOCOL_NAME,
)

from .conftest import (
    EXTERNAL_DESCRIPTION,
    EXTERNAL_REFERENCE,
    MEASURED_METRIC,
    STORED_REFERENCE,
    ServeRelease,
    SECOND_METRIC,
    VIEW_BASELINE,
    WriteReleaseRoot,
    comparison_entry,
)
from release_files import (
    instance_weights_relative,
    withheld_profiles_relative,
    write_protocol,
)


from benchmark_lock import BenchmarkPin
from results_roots import (
    BEST_MEDIAN,
    INSTANCE_WEIGHTS_ROLE,
    LEADERBOARD_SURFACE,
    NO_BASELINE,
    RESULTS_SURFACES,
    WITHHELD_PROFILES_ROLE,
    read_comparison,
    release_input,
    results_routes,
)

OTHER_TASK = "linear_regression"
PIN = BenchmarkPin(repo_id="owner/dataset", revision="a" * 40)


def read(root: Path, **kwargs: object):
    """Resolve one local release, as a build does from its environment."""
    return release_input(None, str(root), cache_dir=root / "cache", **kwargs)


# ── Which input a build reads ────────────────────────────────────────────────


def test_a_build_given_neither_input_reads_no_release(tmp_path: Path) -> None:
    assert release_input(None, None, cache_dir=tmp_path) is None


def test_a_build_given_both_inputs_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    with pytest.raises(ValueError, match="exactly one Distribution Release"):
        release_input(PIN, str(root), cache_dir=tmp_path / "cache")


def test_an_unreadable_release_stops_the_build(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        read(empty)


# ── What the release declares ────────────────────────────────────────────────


def test_the_release_declares_the_tasks_the_site_renders(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release",
        published={MeanEstimation.id: [MEASURED_METRIC], OTHER_TASK: [MEASURED_METRIC]},
    )
    release = read(root)
    assert release is not None
    assert sorted(release.tasks) == sorted([MeanEstimation.id, OTHER_TASK])


def test_a_release_without_results_availability_requires_a_new_cut(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    path = root / "release.json"
    manifest = json.loads(path.read_text())
    del manifest["tasks"][MeanEstimation.id]["results"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predates Results availability.*re-cut"):
        read(root)


def test_each_released_task_names_both_sanctioned_views(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    release = read(root)
    assert release is not None
    views = release.tasks[MeanEstimation.id].views
    assert views[WITHHELD_PROFILES_ROLE] == root / withheld_profiles_relative(
        MeanEstimation.id
    )
    assert views[INSTANCE_WEIGHTS_ROLE] == root / instance_weights_relative(
        MeanEstimation.id
    )


def test_a_task_declaring_only_one_view_role_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(tmp_path / "release", one_role=[MeanEstimation.id])
    with pytest.raises(ValueError, match="instance-weights"):
        read(root)


def test_a_declared_view_whose_bytes_are_absent_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(tmp_path / "release", unwritten=[MeanEstimation.id])
    with pytest.raises(ValueError, match="does not carry"):
        read(root)


def test_a_local_tree_names_its_benchmark_and_results_revision(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release",
        published={MeanEstimation.id: [MEASURED_METRIC]},
        version="cut-17",
    )
    release = read(root)
    assert release is not None
    assert release.identity == f"Benchmark cut-17, Results {release.manifest_id[:12]}"
    assert str(root) not in release.identity
    assert release.published is False


# ── The routes those tasks open ──────────────────────────────────────────────


def test_a_build_reading_no_release_opens_no_results_route() -> None:
    assert results_routes(None) == {}


def test_every_released_task_opens_both_surfaces_at_base_paths(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release",
        published={MeanEstimation.id: [MEASURED_METRIC], OTHER_TASK: [MEASURED_METRIC]},
    )
    release = read(root)
    routes = results_routes(release)
    assert sorted(routes) == sorted([MeanEstimation.id, OTHER_TASK])
    for task_routes in routes.values():
        assert [route.route for route in task_routes] == [
            surface for surface, _ in RESULTS_SURFACES
        ]
        assert [route.title for route in task_routes] == [
            title for _, title in RESULTS_SURFACES
        ]
        assert all(route.root == root for route in task_routes)
        assert all(route.published is False for route in task_routes)
    assert routes[MeanEstimation.id][0].surface == LEADERBOARD_SURFACE


# ── The fields the protocol states ───────────────────────────────────────────


def released_comparison(root: Path, task_id: str, current: Path | None = None):
    """Read one task's entry from the document a build of *root* reads."""
    release = read(root, current_protocol=None if current is None else str(current))
    assert release is not None
    return read_comparison(release.protocol, task_id)


def test_the_protocol_supplies_the_fields_a_page_reads(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    comparison = released_comparison(root, MeanEstimation.id)
    assert comparison.headline_metric == MEASURED_METRIC
    assert comparison.headline_baseline == VIEW_BASELINE
    assert comparison.headline_definition.kind == "best_median"
    assert comparison.headline_quantile == 0.9
    assert (comparison.taus[0], comparison.taus[-1], len(comparison.taus)) == (
        0.0,
        10.0,
        1001,
    )
    assert (comparison.weighting, comparison.eligibility, comparison.pool) == (
        "provenance_ref",
        "standard",
        "standard",
    )
    assert comparison.definitions[BEST_MEDIAN].detail == (
        ("pool", "standard"),
        ("aggregation", "median"),
    )
    external = comparison.external[EXTERNAL_REFERENCE]
    assert (external.description, external.metrics) == (
        EXTERNAL_DESCRIPTION,
        (MEASURED_METRIC,),
    )


def test_each_metric_is_offered_its_statistical_and_the_external_baselines(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    comparison = released_comparison(root, MeanEstimation.id)
    assert comparison.stored_for(MEASURED_METRIC, STORED_REFERENCE) == (
        STORED_REFERENCE,
        EXTERNAL_REFERENCE,
    )
    assert comparison.stored_for(SECOND_METRIC, STORED_REFERENCE) == (STORED_REFERENCE,)
    assert comparison.computed_for(MEASURED_METRIC) == (BEST_MEDIAN,)
    assert (
        comparison.opening_for(MEASURED_METRIC, STORED_REFERENCE, ()) == VIEW_BASELINE
    )
    assert (
        comparison.opening_for(SECOND_METRIC, STORED_REFERENCE, {STORED_REFERENCE})
        == STORED_REFERENCE
    )
    assert comparison.opening_for(SECOND_METRIC, STORED_REFERENCE, ()) == NO_BASELINE


def test_a_stored_headline_baseline_is_offered_on_the_headline_metric(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    write_protocol(
        root,
        {MeanEstimation.id: comparison_entry(MEASURED_METRIC, baseline="stored_scale")},
    )
    comparison = released_comparison(root, MeanEstimation.id)
    assert comparison.headline_definition.kind == "reference"
    assert comparison.stored_for(MEASURED_METRIC, None)[-1] == "stored_scale"
    assert "stored_scale" not in comparison.stored_for(SECOND_METRIC, None)


def test_a_log_grid_rises_by_decades(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"]["tau_grid"] = {
        "min": 0.1,
        "max": 100.0,
        "scale": "log",
        "steps": 4,
    }
    write_protocol(root, {MeanEstimation.id: entry})
    taus = released_comparison(root, MeanEstimation.id).taus
    assert taus == pytest.approx((0.1, 1.0, 10.0, 100.0))


def test_a_log1p_grid_is_even_in_log_one_plus_tau_from_zero(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    """log(1 + τ) at 0, 1, 2 and 3 times log 4, by hand: τ = 0, 3, 15, 63."""
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"]["tau_grid"] = {
        "min": 0.0,
        "max": 63.0,
        "scale": "log1p",
        "steps": 4,
    }
    write_protocol(root, {MeanEstimation.id: entry})
    comparison = released_comparison(root, MeanEstimation.id)
    assert comparison.taus == pytest.approx((0.0, 3.0, 15.0, 63.0))
    assert comparison.grid.scale == "log1p"


def _ruled(**overrides: object) -> dict[str, object]:
    """One task's entry, a ranking field replaced or dropped by ``None``."""
    entry = comparison_entry(MEASURED_METRIC)
    for name, value in overrides.items():
        if value is None:
            del entry["ranking"][name]
        else:
            entry["ranking"][name] = value
    return entry


def _external(record: object) -> dict[str, object]:
    """One task's entry declaring one external baseline as *record*."""
    entry = comparison_entry(MEASURED_METRIC)
    entry["external_baselines"] = {"declared": record}
    return entry


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (_ruled(headline_metric=None), "no headline_metric name"),
        (_ruled(headline_baseline=None), "no headline_baseline name"),
        (_ruled(headline_quantile=None), "states no headline quantile"),
        (_ruled(headline_quantile=1.0), "states no headline quantile"),
        (_ruled(headline_quantile=True), "states no headline quantile"),
        (
            _ruled(tau_grid={"min": 0.0, "max": 10.0, "scale": "cubic", "steps": 3}),
            "unknown grid scale",
        ),
        (
            _ruled(tau_grid={"min": 5.0, "max": 1.0, "scale": "linear", "steps": 3}),
            "does not rise",
        ),
        (_ruled(eligibility=None), "no eligibility name"),
        (_ruled(baselines={}), "without 'best_median'"),
        (
            _ruled(baselines={BEST_MEDIAN: {"kind": "unknown_kind"}}),
            "kind 'unknown_kind'.*none of",
        ),
        (
            _ruled(baselines={BEST_MEDIAN: {"kind": "none"}}),
            "is the pooled baseline and only that",
        ),
        (
            _ruled(headline_definition={"kind": "reference", "key": "other"}),
            "named by its key alone",
        ),
        (
            _external({"key": "other", "description": "d", "metrics": ["m"]}),
            "own stored key",
        ),
        (_external({"key": "declared", "metrics": ["m"]}), "without a description"),
        (
            _external({"key": "declared", "description": "d", "metrics": []}),
            "no list of Metrics",
        ),
        (_external({"key": "declared", "description": "d"}), "no list of Metrics"),
    ],
)
def test_a_document_this_build_cannot_show_is_refused(
    tmp_path: Path,
    write_release_root: WriteReleaseRoot,
    entry: dict[str, object],
    message: str,
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    write_protocol(root, {MeanEstimation.id: entry})
    with pytest.raises(ValueError, match=message):
        released_comparison(root, MeanEstimation.id)


def test_a_field_a_page_does_not_show_is_never_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"]["a_later_field"] = {"anything": True}
    entry["a_later_section"] = []
    write_protocol(root, {MeanEstimation.id: entry})
    assert (
        released_comparison(root, MeanEstimation.id).headline_metric == MEASURED_METRIC
    )


def test_a_document_without_a_ranking_section_is_refused_by_name(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    old = dict(comparison_entry(MEASURED_METRIC)["ranking"])
    old["metrics"] = [MEASURED_METRIC]
    write_protocol(root, {MeanEstimation.id: old})
    with pytest.raises(ValueError, match="without a ranking section"):
        released_comparison(root, MeanEstimation.id)


def test_a_released_task_the_document_does_not_rule_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    write_protocol(root, {OTHER_TASK: comparison_entry(MEASURED_METRIC)})
    with pytest.raises(ValueError, match="rules no comparison"):
        released_comparison(root, MeanEstimation.id)


def test_a_document_that_is_not_readable_json_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    (root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME).write_text(
        "{not json", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not readable JSON"):
        released_comparison(root, MeanEstimation.id)


# ── A current protocol beside the shipped ranking ────────────────────────────


def _current(tmp_path: Path, entry: dict[str, object]) -> Path:
    """Write a current protocol in its released JSON encoding."""
    path = tmp_path / "current" / BENCHMARK_PROTOCOL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"comparison": {MeanEstimation.id: entry}}))
    return path


def test_no_current_protocol_reads_the_shipped_document_unchanged(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    release = read(root)
    assert release is not None
    assert (
        release.protocol
        == (root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME).read_bytes()
    )


def test_current_external_baselines_lead_the_shipped_ranking(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["external_baselines"][EXTERNAL_REFERENCE]["metrics"] = [
        MEASURED_METRIC,
        SECOND_METRIC,
    ]
    entry["external_baselines"]["later"] = {
        "key": "later",
        "description": "A value declared after the cut.",
        "metrics": [SECOND_METRIC],
    }
    comparison = released_comparison(root, MeanEstimation.id, _current(tmp_path, entry))
    assert list(comparison.external) == [EXTERNAL_REFERENCE, "later"]
    assert comparison.stored_for(SECOND_METRIC, None) == (EXTERNAL_REFERENCE, "later")
    assert comparison.headline_baseline == VIEW_BASELINE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("headline_quantile", 0.8),
        ("eligibility", "qualified"),
        ("headline_baseline", "x"),
    ],
)
def test_a_current_protocol_with_another_ranking_is_refused(
    tmp_path: Path,
    write_release_root: WriteReleaseRoot,
    field: str,
    value: object,
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"][field] = value
    with pytest.raises(ValueError, match=f"ranking differs.*{field}.*re-cut"):
        released_comparison(root, MeanEstimation.id, _current(tmp_path, entry))


def test_a_current_protocol_ruling_other_tasks_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    path = tmp_path / "current.json"
    path.write_text(json.dumps({"comparison": {OTHER_TASK: comparison_entry("m")}}))
    with pytest.raises(ValueError, match="re-cut"):
        released_comparison(root, MeanEstimation.id, path)


def test_an_authored_toml_protocol_is_refused_with_its_encoder_named(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    authored = tmp_path / "benchmark-protocol.toml"
    authored.write_text("[comparison]\n")
    with pytest.raises(ValueError, match="protocol --encode"):
        released_comparison(root, MeanEstimation.id, authored)


def test_a_pinned_release_is_installed_and_read_as_one_tree(
    tmp_path: Path, measurable_release_root: Path, serve_release: ServeRelease
) -> None:
    """A fetched tree is read by the same reader a local cut is."""
    source = tmp_path / "published"
    shutil.copytree(measurable_release_root, source)
    serve_release(source)

    fetched = release_input(PIN, None, cache_dir=tmp_path / "cache")
    local = read(source)

    assert fetched is not None and local is not None
    assert fetched.tasks.keys() == local.tasks.keys()
    for task_id, task in fetched.tasks.items():
        assert task.sweep.relative_to(fetched.root) == local.tasks[
            task_id
        ].sweep.relative_to(local.root)
        assert {
            role: path.relative_to(fetched.root) for role, path in task.views.items()
        } == {
            role: path.relative_to(local.root)
            for role, path in local.tasks[task_id].views.items()
        }
    assert fetched.manifest_id == local.manifest_id
    assert released_comparison(fetched.root, MeanEstimation.id) == released_comparison(
        source, MeanEstimation.id
    )


def test_a_pinned_release_reports_a_published_release_and_names_its_revision(
    tmp_path: Path, measurable_release_root: Path, serve_release: ServeRelease
) -> None:
    source = tmp_path / "published"
    shutil.copytree(measurable_release_root, source)
    serve_release(source)

    release = release_input(PIN, None, cache_dir=tmp_path / "cache")

    assert release is not None
    assert release.published is True
    assert PIN.repo_id in release.identity
    assert PIN.revision in release.identity
    assert all(
        route.published
        for routes in results_routes(release).values()
        for route in routes
    )


def test_one_pinned_revision_is_fetched_once_and_reused(
    tmp_path: Path, measurable_release_root: Path, serve_release: ServeRelease
) -> None:
    """The revision names the bytes, so a second build reads the install."""
    source = tmp_path / "published"
    shutil.copytree(measurable_release_root, source)
    downloads = serve_release(source)
    cache = tmp_path / "cache"

    first = release_input(PIN, None, cache_dir=cache)
    fetched = len(downloads)
    second = release_input(PIN, None, cache_dir=cache)

    assert fetched > 0
    assert len(downloads) == fetched
    assert first is not None and second is not None
    assert first.root == second.root == cache / PIN.revision


def test_a_pinned_release_missing_what_it_declares_stops_the_build(
    tmp_path: Path, measurable_release_root: Path, serve_release: ServeRelease
) -> None:
    """A release short of a view it declares stops the build, not a page.

    The pinned commit binds the bytes, so a fetch installs whatever the
    repository holds; what the pages need is the build's own question.
    """
    source = tmp_path / "published"
    shutil.copytree(measurable_release_root, source)
    (source / instance_weights_relative(MeanEstimation.id)).unlink()
    serve_release(source)

    with pytest.raises(ValueError, match=r"declares view\(s\) it does not carry"):
        release_input(PIN, None, cache_dir=tmp_path / "cache")


def test_the_ranking_states_its_slices_in_order(
    tmp_path: Path, write_release_root: WriteReleaseRoot
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"]["slices"] = [
        {"key": "nd-le-100", "feature": "n_over_d", "at_most": 100},
        {"key": "nd-gt-100", "feature": "n_over_d", "above": 100},
    ]
    write_protocol(root, {MeanEstimation.id: entry})
    slices = released_comparison(root, MeanEstimation.id).slices
    assert [(each.key, each.feature, each.bound, each.above) for each in slices] == [
        ("nd-le-100", "n_over_d", 100.0, False),
        ("nd-gt-100", "n_over_d", 100.0, True),
    ]


@pytest.mark.parametrize(
    "stated",
    [
        {"key": "all", "feature": "d", "at_most": 10},
        {"key": "d-x", "feature": "n", "at_most": 10},
        {"key": "d-x", "feature": "d", "at_most": 10, "above": 10},
        {"key": "d-x", "feature": "d"},
    ],
)
def test_a_slice_that_is_not_one_key_feature_and_bound_is_refused(
    tmp_path: Path, write_release_root: WriteReleaseRoot, stated: dict[str, object]
) -> None:
    root = write_release_root(
        tmp_path / "release", published={MeanEstimation.id: [MEASURED_METRIC]}
    )
    entry = comparison_entry(MEASURED_METRIC)
    entry["ranking"]["slices"] = [stated]
    write_protocol(root, {MeanEstimation.id: entry})
    with pytest.raises(ValueError, match="slice"):
        released_comparison(root, MeanEstimation.id)
