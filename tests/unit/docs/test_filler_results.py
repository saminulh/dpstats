"""The results templates are the live views over an invented snapshot."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest


from leaderboard import collection_standings, leaderboard_body
from panel_data import TaskData
from playground import playground_page
from prepare_results import PREPARED_NAME, read_prepared
from templates import filler_results
from templates.spec import TEMPLATES

# The byte budget the whole generated snapshot tree is held to.
SIZE_BUDGET = 1_000_000

RESULTS_TEMPLATES = ("T1", "T4", "T5")


@pytest.fixture(scope="module")
def filler_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The filler snapshot, generated from its seeds into a throwaway root."""
    return filler_results.build_snapshot(tmp_path_factory.mktemp("filler"))


def _measured(snapshot: Path) -> TaskData:
    data, _, _, _ = read_prepared(snapshot, filler_results.FILLER_TASK.id)
    assert data is not None
    return data


def _leaderboard(
    snapshot: Path,
    *,
    absent: Sequence[tuple[str, str]] = (),
    home: bool = False,
) -> str:
    data = _measured(snapshot)
    return leaderboard_body(
        filler_results.FILLER_TASK.id,
        collection_standings(data),
        filler_results.IDENTITY,
        depth=2,
        published=False,
        provenance="provenance.json",
        absent=absent,
        policy=data.index,
        home=home,
    )


def test_the_snapshot_is_generated_within_its_size_budget(
    filler_snapshot: Path,
) -> None:
    """The whole generated tree is budgeted, not the prepared part alone."""
    root = filler_snapshot.parent.parent
    assert root.name != "prepared"
    assert (root / "store").is_dir()
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert size < SIZE_BUDGET, size


def test_the_snapshot_carries_the_roster_its_seeds_declare(
    filler_snapshot: Path,
) -> None:
    """Every Competitor, budget and threshold the page shows is invented."""
    data, standings, missing, refused = read_prepared(
        filler_snapshot, filler_results.FILLER_TASK.id
    )
    assert data is not None and standings is not None
    assert (missing, refused) == ([], [])
    labels = {str(entry["label"]) for entry in data.index["algorithms"]}
    assert labels == {*filler_results.COMPETITORS, filler_results.QUALIFIED_COMPETITOR}
    assert len(data.index["taus"]) == filler_results.TAU_GRID["steps"]
    budgets = {panel.budget for panel in data.panels}
    assert len(budgets) == len(filler_results.BUDGETS)


def test_the_roster_fills_the_rankings_table_past_its_visible_rows(
    filler_snapshot: Path,
) -> None:
    """Ten rows show and the rest are reachable, so both states are previewed."""
    data = _measured(filler_snapshot)
    eligible = {
        str(entry["uuid"])
        for entry in data.index["algorithms"]
        if not entry["diagnostic"]
    }
    assert len(eligible) > 10


def test_generating_the_snapshot_twice_measures_the_same_standings(
    tmp_path: Path, filler_snapshot: Path
) -> None:
    """The seeds decide the numbers, so a regenerated preview shows the same."""
    again = filler_results.build_snapshot(tmp_path / "again")
    first = _measured(filler_snapshot)
    second = _measured(again)
    assert first.index["headline"] == second.index["headline"]
    assert first.files == second.files


def test_a_snapshot_from_another_roster_is_refused(tmp_path: Path) -> None:
    """A stale preview would show Competitors the roster does not declare."""
    root = tmp_path / "stale"
    filler_results.build_snapshot(root)
    assert filler_results.snapshot(root).is_dir()
    (root / filler_results.STAMP_NAME).write_text(
        json.dumps({"seeds": "an earlier roster", "snapshot": "gone"}), "utf-8"
    )
    with pytest.raises(filler_results.MissingSnapshot, match="does not declare"):
        filler_results.snapshot(root)


def test_a_build_without_a_snapshot_is_refused(tmp_path: Path) -> None:
    """Preparing and rendering stay separate: a build measures nothing."""
    with pytest.raises(filler_results.MissingSnapshot, match="Generate it with"):
        filler_results.snapshot(tmp_path / "absent")


def test_a_directory_the_generator_did_not_write_is_kept(tmp_path: Path) -> None:
    """The generator replaces its own output and nothing else."""
    root = tmp_path / "theirs"
    root.mkdir()
    (root / "someone-elses-work.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="did not write"):
        filler_results.build_snapshot(root)
    assert (root / "someone-elses-work.txt").read_text(encoding="utf-8") == "keep me"


def test_the_snapshot_carries_the_invented_release_alone(
    filler_snapshot: Path,
) -> None:
    """A release the lock pins is a build input, never rendered as invented."""
    prepared = json.loads((filler_snapshot / PREPARED_NAME).read_text(encoding="utf-8"))
    assert sorted(prepared["tasks"]) == [filler_results.FILLER_TASK.id]
    assert prepared["published"] is False


# What the live renderers emit, written down here rather than read off the
# spec, so a wrong marker and a changed template are two separate failures
# instead of agreeing with each other.
EXPECTED_SECTIONS = {
    "T4": (
        "introduction",
        "controls",
        "metric",
        "budget",
        "chart",
        "plotted",
        "rankings",
        "build_source",
    ),
    "T5": (
        "loading",
        "controls",
        "empty",
        "metric",
        "budget",
        "chart",
        "plotted",
        "rankings",
        "comparisons",
    ),
}


def test_every_results_template_is_declared_once() -> None:
    for name in RESULTS_TEMPLATES:
        template = TEMPLATES[name]
        keys = [section.key for section in template.sections]
        assert len(set(keys)) == len(keys), name
        assert set(keys) == {key for key, _ in filler_results.SECTION_MARKERS[name]}


def test_the_home_tabs_marker_matches_the_tabs_markup_home_leaderboards_emits() -> None:
    """T1's ``tabs`` marker has to match the real tab strip's own markup.

    ``assemble.py`` (home to ``home_leaderboards``) cannot be imported here: it
    renders the real site as a module-level side effect. This reads its
    source text directly instead, so a real edit to the tab strip's markup is
    what this test tracks, not a hand-copied snapshot that can silently drift
    from it.
    """
    source = (
        Path(__file__).resolve().parents[3] / "docs" / "site" / "assemble.py"
    ).read_text(encoding="utf-8")
    tabs_div = re.search(r'<div class="dp-task-tabs[^>]*>', source)
    assert tabs_div, "assemble.py emits no dp-task-tabs div"
    _, marker = filler_results.SECTION_MARKERS["T1"][0]
    assert marker in tabs_div.group()


def test_the_leaderboard_renders_the_sections_its_template_declares(
    filler_snapshot: Path,
) -> None:
    """T4's declared list is what the live Leaderboard emits, in that order."""
    page = _leaderboard(filler_snapshot)
    found = filler_results.rendered_sections("T4", page)
    assert found == EXPECTED_SECTIONS["T4"]
    declared = [section.key for section in TEMPLATES["T4"].sections]
    assert list(found) == [key for key in declared if key in found]
    required = [section.key for section in TEMPLATES["T4"].sections if section.required]
    assert set(required) <= set(found)


def test_the_explore_page_renders_the_sections_its_template_declares(
    filler_snapshot: Path,
) -> None:
    """T5's declared list is what the live Explore page emits, in that order."""
    data = _measured(filler_snapshot)
    page = playground_page(
        data,
        filler_results.IDENTITY,
        depth=2,
        provenance="provenance.json",
    ).body
    found = filler_results.rendered_sections("T5", page)
    assert found == EXPECTED_SECTIONS["T5"]
    declared = [section.key for section in TEMPLATES["T5"].sections]
    assert list(found) == [key for key in declared if key in found]
    required = [section.key for section in TEMPLATES["T5"].sections if section.required]
    assert set(required) <= set(found)


def test_a_refused_metric_renders_in_its_declared_place(
    filler_snapshot: Path,
) -> None:
    """The optional absences section sits where the template says it sits."""
    page = _leaderboard(filler_snapshot, absent=[("metric_two", "no Instance has it")])
    found = filler_results.rendered_sections("T4", page)
    assert "absences" in found
    declared = [section.key for section in TEMPLATES["T4"].sections]
    assert list(found) == [key for key in declared if key in found]


def test_a_section_the_page_does_not_carry_is_left_out(
    filler_snapshot: Path,
) -> None:
    """A page reporting every metric renders no absences section at all."""
    assert "absences" not in filler_results.rendered_sections(
        "T4", _leaderboard(filler_snapshot)
    )


def test_home_and_the_leaderboard_share_one_set_of_columns(
    filler_snapshot: Path,
) -> None:
    """Both tables carry Rank, Algorithm and the headline value, and nothing else."""
    for body in (
        _leaderboard(filler_snapshot, home=True),
        _leaderboard(filler_snapshot),
    ):
        assert 'data-columns="rank,algorithm,headline"' in body
        assert "released" not in body.lower()


def test_the_snapshot_states_the_policy_the_pages_read(
    filler_snapshot: Path,
) -> None:
    """The comparison policy reaches the pages from the protocol, not the site."""
    data = _measured(filler_snapshot)
    assert data.index["headline_quantile"] == filler_results.HEADLINE_QUANTILE
    assert data.index["headline_metric"] == filler_results.METRICS[0]
    prepared = json.loads((filler_snapshot / PREPARED_NAME).read_text(encoding="utf-8"))
    assert prepared["tasks"][filler_results.FILLER_TASK.id]["missing"] == []
