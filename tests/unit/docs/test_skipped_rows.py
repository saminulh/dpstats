"""A local cut skips the registry entries this library cannot rebuild, and says so."""

import json
import shutil
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dpstats.algorithm import register_algorithm
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.transport.canonical import algorithm_identity
from dpstats.workspace.layout import task_collection_directory
from dpstats.workspace.registries import file_registry_entry

from algorithm_provenance import provenance_payload
from panel_data import registered_algorithms
from prepare_results import (
    prepared_snapshot,
    read_prepared,
    read_removed,
    read_skipped,
    write_build_summary,
)
from results_roots import release_input
from skipped_rows import banner, banner_html, skipped_entries, summary_line
from templates.filler import roster_landing_page
from templates.render import render
from templates.spec import TEMPLATES
import prepare_results
from release_files import (
    sweep_relative,
    withheld_profiles_relative,
    write_profile_view,
)
from tests.unit.docs.conftest import measured

TASK = "linear_regression"
MEAN = "mean_estimation"
ABSENT = "dpstats.tasks.linear_regression.algorithms.ridge:Absent"


@pytest.fixture
def store(tmp_path: Path) -> tuple[Path, str, str]:
    """A release-shaped tree holding one loadable and one stale algorithm."""
    register_algorithm(tmp_path, Ridge(label="ridge"), registry="roster")
    record = {"path": ABSENT, "config": {"label": "stale ridge"}}
    stale = algorithm_identity(record)
    file_registry_entry(
        task_collection_directory(tmp_path, TASK),
        task_id=TASK,
        kind="algorithm",
        uuid=stale,
        record=record,
        registry="roster",
    )
    return tmp_path, Ridge(label="ridge").uuid, stale


def test_a_local_cut_lists_the_stale_entry_with_its_label_and_reason(
    store: tuple[Path, str, str],
) -> None:
    root, _, stale = store

    (row,) = skipped_entries(root, TASK, published=False)

    assert (row["kind"], row["uuid"], row["label"]) == (
        "algorithm",
        stale,
        "stale ridge",
    )
    assert "Absent" in row["reason"]


def test_a_published_release_reports_nothing_and_is_read_strictly(
    store: tuple[Path, str, str],
) -> None:
    root, _, stale = store

    assert skipped_entries(root, TASK, published=True) == []
    with pytest.raises(ValueError, match=stale):
        registered_algorithms(root, TASK, strict=True)
    with pytest.raises(ValueError, match=stale):
        provenance_payload(root, TASK, {}, strict=True)


def test_a_local_cut_keeps_the_loadable_row_without_a_stray_warning(
    store: tuple[Path, str, str],
) -> None:
    root, valid, stale = store

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registered = registered_algorithms(root, TASK, strict=False)
        provenance: dict[str, Any] = provenance_payload(root, TASK, {}, strict=False)

    assert set(registered) == set(provenance) == {valid}
    assert stale not in registered


def test_the_banner_names_each_skipped_algorithm_by_label_or_uuid() -> None:
    skipped = [
        {"kind": "algorithm", "uuid": "a3-one", "label": "One", "reason": "x"},
        {"kind": "algorithm", "uuid": "a3-two", "label": "", "reason": "x"},
    ]

    assert banner(skipped) == (
        "This preview leaves out 2 configured algorithm(s) that this library "
        "version cannot load."
    )
    assert "<li>One</li><li>a3-two</li>" in banner_html(skipped)
    assert banner([]) == ""


def test_the_summary_line_counts_rows_and_names_the_first_reason() -> None:
    skipped = [
        {"kind": "algorithm", "uuid": "a3-one", "label": "", "reason": "bad\nmore"}
    ]

    assert summary_line("t", []) == "t: 0 rows skipped, 0 removed from standings"
    assert summary_line("t", skipped) == (
        "t: 1 row skipped, 0 removed from standings: a3-one (bad)"
    )


# ── The published-versus-local split, through prepare_results ───────────────


def _stale_release(measurable_release_root: Path, tmp_path: Path) -> tuple[Path, str]:
    """A copy of the measurable release with one unloadable, ranked algorithm."""
    root = tmp_path / "stale-release"
    shutil.copytree(measurable_release_root, root)
    record: dict[str, Any] = {
        "path": ABSENT.replace("linear_regression", "mean_estimation"),
        "config": {},
    }
    stale = algorithm_identity(record)
    file_registry_entry(
        task_collection_directory(root, MEAN),
        task_id=MEAN,
        kind="algorithm",
        uuid=stale,
        record=record,
        registry="stale",
    )
    view = root / withheld_profiles_relative(MEAN)
    body = pd.read_parquet(view)
    added = body[body["alg_uuid"] == "aaaa-1111"].assign(
        alg_uuid=stale, algorithm="Stale Alpha"
    )
    write_profile_view(view, pd.concat([body, added]).itertuples(index=False))
    _admit_added_curve(root, stale)
    return root, stale


def _admit_added_curve(root: Path, uuid: str) -> None:
    path = root / "release.json"
    manifest = json.loads(path.read_text())
    algorithms = manifest["tasks"][MEAN]["results"]["evaluation"]
    algorithms[uuid] = dict(algorithms["aaaa-1111"])
    path.write_text(json.dumps(manifest))


def _resolve(root: Path):
    release = release_input(None, str(root), cache_dir=root / "cache")
    assert release is not None
    return release


def test_a_local_prepare_drops_a_skipped_algorithm_from_the_standings(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root, stale = _stale_release(measurable_release_root, tmp_path)

    prepared = prepared_snapshot(_resolve(root), cache=tmp_path / "cache")
    _, standings, _, _ = read_prepared(prepared, MEAN)

    assert [row["uuid"] for row in read_skipped(prepared, MEAN)] == [stale]
    assert read_removed(prepared, MEAN) == 1
    assert standings is not None and standings.panels
    ranked = {row["uuid"] for row in standings.index["algorithms"]}
    assert stale not in ranked and "aaaa-1111" in ranked
    assert all(stale.encode() not in payload for payload in standings.files.values())
    assert all(stale not in panel.headline for panel in standings.panels)


def test_the_build_summary_states_removals_and_stays_out_of_the_site(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root, stale = _stale_release(measurable_release_root, tmp_path)
    prepared = prepared_snapshot(_resolve(root), cache=tmp_path / "cache")

    path = write_build_summary(prepared)

    assert path == prepared / "build-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary[MEAN]["removed_from_standings"] == 1
    (row,) = summary[MEAN]["skipped"]
    assert (row["kind"], row["uuid"]) == ("algorithm", stale)
    assert row["reason"]


def test_a_published_prepare_still_refuses_the_stale_entry(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root, stale = _stale_release(measurable_release_root, tmp_path)
    published = replace(_resolve(root), published=True)

    with pytest.raises(ValueError, match=stale):
        prepared_snapshot(published, cache=tmp_path / "cache")


def test_the_banner_names_the_standings_only_when_rows_were_removed() -> None:
    skipped = [{"kind": "algorithm", "uuid": "a3-one", "label": "One", "reason": "x"}]

    assert banner(skipped, 0).endswith("cannot load.")
    assert banner(skipped, 2).endswith(
        "cannot load. They are also left out of this preview's standings."
    )
    assert summary_line("t", skipped, 2).startswith(
        "t: 1 row skipped, 2 removed from standings: a3-one"
    )


def test_the_banner_shows_on_every_results_page_but_not_the_home_preview() -> None:
    skipped = [{"kind": "algorithm", "uuid": "a3-one", "label": "One", "reason": "x"}]

    leaderboard_or_explore = banner_html(skipped, 1)
    assert leaderboard_or_explore.startswith(
        '<div class="dp-note"><p>This preview leaves out 1'
    )
    assert "<details><summary>Show which</summary><ul><li>One</li></ul>" in (
        leaderboard_or_explore
    )
    assert banner_html(skipped, 1, home=True) == ""
    assert banner_html([], 0) == ""
    roster = render(
        TEMPLATES["T22"], replace(roster_landing_page(), notice=banner(skipped, 1))
    )
    assert 'class="dp-note"' in roster and "This preview leaves out 1" in roster
    assert "dp-note" not in render(TEMPLATES["T22"], roster_landing_page())


def test_a_sweep_naming_a_skipped_algorithm_is_refused_with_its_reason(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root, stale = _stale_release(measurable_release_root, tmp_path)
    sweep = root / sweep_relative(MEAN)
    document = json.loads(sweep.read_text(encoding="utf-8"))
    sweep.write_text(json.dumps({**document, "algorithms": [stale]}), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the registry scope") as refusal:
        measured(root)

    assert f"Skipped as unloadable: {stale} (" in str(refusal.value)


def test_a_published_prepare_refuses_to_omit_standings(
    measurable_release_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stale = _stale_release(measurable_release_root, tmp_path)
    published = replace(_resolve(root), published=True)
    monkeypatch.setattr(
        prepare_results,
        "skipped_entries",
        lambda *args, **kwargs: [{"kind": "algorithm", "uuid": stale}],
    )

    with pytest.raises(ValueError, match="may not omit"):
        prepared_snapshot(published, cache=tmp_path / "cache")


def _incomplete_release(
    measurable_release_root: Path, tmp_path: Path
) -> tuple[Path, str]:
    """A copy of the measurable release in which one ranked algorithm lacks a trial."""
    import pyarrow.parquet as pq

    from dpstats.workspace.layout import results_directory
    from tests.unit.docs.conftest import FIXTURE_ALGORITHMS, StepMean

    root = tmp_path / "incomplete-release"
    shutil.copytree(measurable_release_root, root)
    lacking = StepMean(label="Alg Gamma", error=FIXTURE_ALGORITHMS["Alg Gamma"]).uuid
    (part, *_) = sorted(
        (
            results_directory(task_collection_directory(root, MEAN))
            / f"alg_id={lacking}"
        ).glob("*.parquet")
    )
    table = pq.read_table(part)
    pq.write_table(table.slice(1), part)
    view = root / withheld_profiles_relative(MEAN)
    body = pd.read_parquet(view)
    added = body[body["alg_uuid"] == "aaaa-1111"].assign(alg_uuid=lacking)
    write_profile_view(view, pd.concat([body, added]).itertuples(index=False))
    _admit_added_curve(root, lacking)
    return root, lacking


def test_a_distribution_gap_preserves_independently_admitted_evaluation(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root, lacking = _incomplete_release(measurable_release_root, tmp_path)

    prepared = prepared_snapshot(_resolve(root), cache=tmp_path / "cache")
    data, standings, _, _ = read_prepared(prepared, MEAN)

    (row,) = [row for row in read_skipped(prepared, MEAN) if row["uuid"] == lacking]
    assert (row["kind"], row["label"]) == ("incomplete", "Alg Gamma")
    assert row["reason"].startswith("missing 1 of ")
    assert data is not None and standings is not None
    explored = {entry["uuid"] for entry in data.index["algorithms"]}
    ranked = {entry["uuid"] for entry in standings.index["algorithms"]}
    assert lacking not in explored and lacking in ranked
    assert explored and "aaaa-1111" in ranked
    assert read_removed(prepared, MEAN) == 0


def test_the_banner_names_an_incomplete_algorithm_with_its_missing_count() -> None:
    skipped = [
        {
            "kind": "incomplete",
            "uuid": "a3-one",
            "label": "One",
            "reason": "missing 2 of 40 declared trials",
        }
    ]

    html = banner_html(skipped, 1)

    assert "missing some of their trials" in banner(skipped, 1)
    assert "One (missing 2 of 40 declared trials)" in html
