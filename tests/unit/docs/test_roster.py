"""T21 configured algorithm and T22 Roster landing: data, then rendered content."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import site_copy
from leaderboard import PanelStandings, StandingRow
from panel_data import Headline
from profile_chart import share_floor
from roster_pages import _demote, roster_page, standing_href
from templates.filler import (
    roster_landing_page,
    configured_algorithm_page,
)
from templates.model import ConfiguredAlgorithm, Link
from templates.render import render
from templates.roster import (
    component_links,
    roster_entries,
    roster_rows,
    transformation_facet,
)
from templates.spec import TEMPLATES
from tests.unit.docs.conftest import WriteCopy, parse_html

# The Roster labels written as they read before they became slots, so a test
# can read the ranks, names and thresholds that fill them.
ROSTER_LABELS = {
    "roster": {
        "rank": "Rank {rank}",
        "not_ranked": "not ranked",
        "column_at_budget": "{column} at {budget}",
        "other_task_class": "{name} ({task} class)",
        "exact_configuration": "Exact configuration",
    }
}

INDEX = {
    "headline_metric": "absolute_error_to_ols",
    "algorithms": [
        {"uuid": "u1", "label": "Alpha", "class": "standard", "diagnostic": False},
        {"uuid": "u2", "label": "Beta", "class": "standard", "diagnostic": False},
        {"uuid": "u3", "label": "Gamma", "class": "informal", "diagnostic": True},
    ],
}
PROVENANCE = {
    uuid: {
        "component": {
            "path": f"pkg.{label}",
            "name": label,
            "kind": "Algorithm",
            "children": (),
        }
    }
    for uuid, label in (("u1", "Alpha"), ("u2", "Beta"), ("u3", "Gamma"))
}


def _panel(
    rows: tuple[StandingRow, ...], metric: str = "absolute_error_to_ols"
) -> PanelStandings:
    return PanelStandings(metric=metric, budget="eps=1.0", rows=rows)


def _row(uuid: str, tau: float, *, diagnostic: bool = False) -> StandingRow:
    """One standing whose headline error ratio is *tau*: lower ranks better."""
    return StandingRow(
        uuid=uuid,
        label=uuid,
        privacy_interpretation="standard",
        diagnostic=diagnostic,
        headline=Headline(tau, 1.0, tau),
        declared=10,
        measured=10,
    )


def test_a_row_missing_from_the_headline_panel_is_not_ranked() -> None:
    """u2 never appears in the headline panel, so it opens unranked."""
    headline = _panel((_row("u1", 0.1),))
    entries = roster_entries(INDEX, (headline,), PROVENANCE)
    by_uuid = {entry.uuid: entry for entry in entries}
    assert by_uuid["u1"].rank == 1
    assert by_uuid["u2"].rank is None
    assert by_uuid["u3"].rank is None


def test_a_diagnostic_entry_never_ranks_even_when_its_fraction_wins() -> None:
    """A diagnostic row is a comparison, never a ranked configured algorithm."""
    headline = _panel((_row("u1", 0.5), _row("u3", 0.01, diagnostic=True)))
    entries = roster_entries(INDEX, (headline,), PROVENANCE)
    assert {entry.uuid: entry.rank for entry in entries} == {
        "u1": 1,
        "u2": None,
        "u3": None,
    }


def test_a_tie_shares_the_better_rank_like_the_leaderboard() -> None:
    """D1: two rows sharing the winning fraction both read rank 1, not 1/2."""
    headline = _panel((_row("u1", 0.5), _row("u2", 0.5)))
    entries = roster_entries(INDEX, (headline,), PROVENANCE)
    by_uuid = {entry.uuid: entry for entry in entries}
    assert by_uuid["u1"].rank == 1
    assert by_uuid["u2"].rank == 1


def test_the_headline_metric_s_panel_ranks_even_when_listed_second() -> None:
    """D1: roster_entries reads the headline metric's opening panel, not
    ``standings[0]``."""
    other = _panel((_row("u1", 0.9), _row("u2", 0.1)), metric="prediction_rmse_to_ols")
    headline = _panel(
        (_row("u1", 0.1), _row("u2", 0.9)), metric="absolute_error_to_ols"
    )
    entries = roster_entries(INDEX, (other, headline), PROVENANCE)
    by_uuid = {entry.uuid: entry for entry in entries}
    assert by_uuid["u1"].rank == 1
    assert by_uuid["u2"].rank == 2


def _step(task: str, kind: str, name: str) -> dict[str, object]:
    return {
        "path": f"dpstats.tasks.{task}.{kind}s.module:{name}",
        "name": name,
        "kind": kind.capitalize(),
        "docs": f"{task}/{kind}s/{name}/",
        "children": (),
    }


def _composition(*steps: dict[str, object]) -> dict[str, object]:
    return {
        "component": {
            "path": "dpstats.algorithm:CompositeAlgorithm",
            "name": "CompositeAlgorithm",
            "kind": "Composition",
            "children": tuple(
                {"field": f"step[{at}]", "component": step}
                for at, step in enumerate(steps)
            ),
        }
    }


CHAINED = {
    "c1": _composition(
        _step("single_coefficient_estimation", "transformation", "ApproxBounds"),
        _step("single_coefficient_estimation", "transformation", "ToLinearRegression"),
        {
            **_step("linear_regression", "algorithm", "AdaSSP"),
            "hyperparameters": (
                {"name": "rho", "value": "0.1", "default": "0.05"},
                {"name": "bounds", "value": "(0, 1)", "default": "required"},
                {"name": "steps", "value": "10", "default": "10"},
            ),
        },
    ),
    "c2": _composition(
        _step("single_coefficient_estimation", "transformation", "ToLinearRegression"),
        _step("linear_regression", "transformation", "ApproxBounds"),
        _step("linear_regression", "algorithm", "AdaSSP"),
    ),
    "c3": {"component": _step("single_coefficient_estimation", "algorithm", "OLS")},
}
CHAINED_INDEX = {
    "headline_metric": "absolute_error_to_ols",
    "algorithms": [
        {
            "uuid": "c1",
            "label": "AdaSSP, scaled",
            "class": "standard",
            "core": "AdaSSP",
        },
        {"uuid": "c2", "label": "Shared label", "class": "qualified", "core": "AdaSSP"},
        {"uuid": "c3", "label": "OLS", "class": "nonprivate", "core": "OLS"},
    ],
}
TITLES = {
    "single_coefficient_estimation": "Single Coefficient Estimation",
    "linear_regression": "Linear Regression",
}


def _chained_rows():
    entries = [
        replace(entry, url=f"{entry.uuid}/")
        for entry in roster_entries(CHAINED_INDEX, (), CHAINED)
    ]
    return roster_rows(
        entries,
        CHAINED,
        {"c1": Headline(0.5, 1.0, 0.5)},
        "single_coefficient_estimation",
        TITLES,
        "../../",
    )


def test_a_roster_row_is_its_chain_in_application_order_core_last() -> None:
    """Each step links its page and shows the values its configuration sets
    away from their defaults; the label is never parsed."""
    rows, _ = _chained_rows()
    by_uuid = {row.algorithm.uuid: row for row in rows}
    chain = by_uuid["c1"].chain
    assert [step.component for step in chain] == [
        "ApproxBounds",
        "ToLinearRegression",
        "AdaSSP",
    ]
    assert chain[-1].hyperparameters == "rho=0.1" and not chain[-1].key
    assert (
        chain[0].url
        == "../../single_coefficient_estimation/transformations/ApproxBounds/"
    )
    assert by_uuid["c1"].headline == "0.5"
    # Whatever the label says, each step is named by its class.
    assert [step.name for step in by_uuid["c2"].chain] == [
        "ToLinearRegression",
        "ApproxBounds",
        "AdaSSP",
    ]
    assert by_uuid["c3"].headline == ""


def test_a_shared_transformation_name_keys_the_other_task_s_class(
    write_copy: WriteCopy,
) -> None:
    """The Roster task's own class keeps the plain name; the other is Name@task."""
    write_copy(ROSTER_LABELS)
    _, facet = _chained_rows()
    own = "dpstats.tasks.single_coefficient_estimation.transformations.module.ApproxBounds"
    other = "dpstats.tasks.linear_regression.transformations.module.ApproxBounds"
    assert facet[own] == ("ApproxBounds", "ApproxBounds")
    assert facet[other] == (
        "ApproxBounds@linear_regression",
        "ApproxBounds (Linear Regression class)",
    )
    lone = transformation_facet(
        [[("Standardize", "x.tasks.linear_regression.t")]],
        "single_coefficient_estimation",
        TITLES,
    )
    assert lone == {"x.tasks.linear_regression.t": ("Standardize", "Standardize")}


def test_a_component_links_the_roster_filtered_to_the_rows_it_names() -> None:
    """Each count is exactly the rows its filter shows."""
    rows, facet = _chained_rows()
    links = component_links(
        rows,
        CHAINED,
        facet,
        "Single Coefficient Estimation",
        "single_coefficient_estimation/roster/",
    )
    core = links["dpstats.tasks.linear_regression.algorithms.module.AdaSSP"]
    assert (core.count, core.url) == (
        2,
        "single_coefficient_estimation/roster/?core=AdaSSP",
    )
    other = links["dpstats.tasks.linear_regression.transformations.module.ApproxBounds"]
    assert (other.count, other.url) == (
        1,
        "single_coefficient_estimation/roster/?transformation=ApproxBounds@linear_regression",
    )
    bridge = links[
        "dpstats.tasks.single_coefficient_estimation.transformations.module.ToLinearRegression"
    ]
    assert bridge.count == 2


def test_t21_omits_standing_and_provenance_when_the_release_ships_neither(
    write_copy: WriteCopy,
) -> None:
    write_copy(ROSTER_LABELS)
    stripped = replace(configured_algorithm_page(), standings=(), provenance="")
    body = render(TEMPLATES["T21"], stripped)
    assert "ranks" not in body
    assert "## Provenance" not in body
    # Every other section still renders.
    assert "## Recipe" in body
    assert "Exact configuration" in body


def test_t21_recipe_nests_a_configured_child_under_its_owning_component() -> None:
    body = render(TEMPLATES["T21"], configured_algorithm_page())
    transformation_at = body.index("### [FillerTransformation]")
    algorithm_at = body.index("### [FillerAlgorithm]")
    loss_at = body.index("#### FillerLoss")
    assert transformation_at < algorithm_at < loss_at


def test_t21_standing_shows_rank_and_not_ranked_rows() -> None:
    body = render(TEMPLATES["T21"], configured_algorithm_page())
    assert "Rank" in body
    assert "not ranked" in body


def test_t22_is_one_table_whose_rows_carry_what_the_filters_read(
    write_copy: WriteCopy,
) -> None:
    """No Task or core column: the chain shows both, in rank order."""
    write_copy(ROSTER_LABELS)
    record = roster_landing_page()
    page = parse_html(render(TEMPLATES["T22"], record))
    (table,) = page.find_all(tag="table")
    assert [head.text for head in table.find_all(tag="th")] == [
        "Configured algorithm",
        "Privacy interpretation",
        "Rank at ε = 1.0",
        "⟨leaderboard.column_headline⟩ at ε = 1.0",
        "Page",
    ]
    rows = table.body_rows()
    assert [row.attrs["data-dp-label"] for row in rows] == [
        row.algorithm.label for row in record.rows
    ]
    chained = rows[1]
    assert chained.attrs["data-dp-transformations"] == "FillerTransformation"
    assert chained.attrs["data-dp-core"] == "FillerAlgorithm"
    steps = chained.find_all(tag="a", **{"class": "dp-chain-step"})
    assert [step.text for step in steps] == ["FillerTransformation"]
    core = chained.find(tag="a", **{"class": "dp-chain-step is-core"})
    assert core.text == "FillerAlgorithm (clip_norm=2.0)"
    unranked = rows[2]
    assert unranked.attrs["data-dp-rank"] == ""
    assert "not ranked" in unranked.text


def test_t22_controls_wait_for_the_script_and_name_every_facet_value() -> None:
    """The controls are hidden until the script that drives them runs."""
    record = roster_landing_page()
    page = parse_html(render(TEMPLATES["T22"], record))
    controls = page.find("data-dp-roster-controls")
    assert "hidden" in controls.attrs
    chips = controls.find_all("data-dp-facet")
    assert [
        (chip.attrs["data-dp-facet"], chip.attrs["data-dp-value"]) for chip in chips
    ] == [
        ("class", "standard"),
        ("class", "qualified"),
        ("transformation", "FillerTransformation"),
    ]
    options = [option.attrs["value"] for option in controls.find_all(tag="option")]
    assert options == ["rank", "headline", "name", "", "FillerAlgorithm"]


def test_t21_results_breakdown_shows_each_outcome_count() -> None:
    body = render(TEMPLATES["T21"], configured_algorithm_page())
    assert (
        "## Results breakdown on the public corpus · [Explore](../t05-explore/)" in body
    )
    assert "82 (91.1%)" in body
    assert "Measured" not in body


def test_t21_omits_outcomes_when_the_release_ships_no_distribution_lane_results() -> (
    None
):
    stripped = replace(configured_algorithm_page(), outcomes=())
    body = render(TEMPLATES["T21"], stripped)
    assert "## Outcomes" not in body


def test_a_non_zero_share_never_reads_0_0_percent() -> None:
    """D7: a tiny non-zero share reads "<0.1%", never "0.0%"."""
    assert share_floor(3 / 10_000) == "<0.1%"
    assert share_floor(0.0) == "0.0%"
    assert share_floor(0.5) == "50.0%"


def test_t21_standing_is_read_in_budget_order_not_string_order() -> None:
    """D4: the Standing table reads 1, 3, 10, 30, not string order."""
    panels = (_panel((_row("u1", 0.5),), metric="absolute_error_to_ols"),)
    budget_10 = replace(panels[0], budget="Family(eps=10.0)")
    budget_1 = replace(panels[0], budget="Family(eps=1.0)")
    budget_3 = replace(panels[0], budget="Family(eps=3.0)")
    entries = roster_entries(INDEX, (budget_1,), PROVENANCE)
    entry = next(entry for entry in entries if entry.uuid == "u1")
    described = {
        "component": {
            **PROVENANCE["u1"]["component"],
            "hyperparameters": (),
            "docs": "",
        },
        "configuration": {},
    }
    page = roster_page(
        entry,
        Link("Task Name", "../index.md"),
        described,
        (budget_10, budget_1, budget_3),
        "../leaderboard/",
        "task",
        "",
        {},
        ["Family(eps=1.0)", "Family(eps=3.0)", "Family(eps=10.0)"],
        1.0,
        "../explore/",
        (),
        "../class/",
        methodology_url="../methodology/",
    )
    body = render(TEMPLATES["T21"], page)
    one = body.index("ε = 1.0")
    three = body.index("ε = 3.0")
    ten = body.index("ε = 10.0")
    assert one < three < ten


def test_each_standing_row_links_to_the_leaderboard_at_its_own_budget() -> None:
    """E1: a Standing link names its metric and budget, not only the anchor."""
    link = standing_href(
        "../../leaderboard/", "task", "absolute_error_to_ols", "Family(eps=30.0)"
    )
    assert link == (
        "../../leaderboard/?metric=absolute_error_to_ols&budget=Family%28eps%3D30.0%29"
        "#profile-task-absolute-error-to-ols"
    )
    panels = (
        replace(_panel((_row("u1", 0.5),)), budget="Family(eps=3.0)"),
        replace(_panel((_row("u1", 0.5),)), budget="Family(eps=30.0)"),
    )
    entry = next(
        entry
        for entry in roster_entries(INDEX, panels, PROVENANCE)
        if entry.uuid == "u1"
    )
    described = {
        "component": {
            **PROVENANCE["u1"]["component"],
            "hyperparameters": (),
            "docs": "",
        },
        "configuration": {},
    }
    page = roster_page(
        entry,
        Link("Task Name", "../index.md"),
        described,
        panels,
        "../../leaderboard/",
        "task",
        "",
        {},
        ["Family(eps=3.0)", "Family(eps=30.0)"],
        1.0,
        "../explore/",
        (),
        "../class/",
        methodology_url="../methodology/",
    )
    hrefs = {row.budget: row.metric.url for row in page.standings}
    assert hrefs["ε = 3.0"].count("budget=Family%28eps%3D3.0%29") == 1
    assert "budget=Family%28eps%3D30.0%29" in hrefs["ε = 30.0"]


def _algorithm(rank: int | None) -> ConfiguredAlgorithm:
    return ConfiguredAlgorithm("u", "Alpha", "Core", (), "standard", "", rank)


def test_one_configured_algorithm_record_serves_every_roster_view(
    write_copy: WriteCopy,
) -> None:
    """The T21 page and the T22 landing share one view record."""
    write_copy(ROSTER_LABELS)
    assert _algorithm(rank=2).standing == "Rank 2"
    assert _algorithm(rank=None).standing == "not ranked"
    assert configured_algorithm_page().algorithm.label == "Filler Method A"
    assert roster_landing_page().rows[0].algorithm.label == "FillerAlgorithm"


def test_a_configured_page_links_its_results_pages_and_its_class(
    write_copy: WriteCopy,
) -> None:
    """The ranks heading reaches the Leaderboard, the headline column names its
    quantile, and the class badge reaches its rule."""
    write_copy({**ROSTER_LABELS, "leaderboard": {"column_headline": "τ at {percent}%"}})
    body = render(TEMPLATES["T21"], configured_algorithm_page())
    ranks = body.split("## [Leaderboard](../t04-leaderboard/) ranks", 1)[1].split(
        "## ", 1
    )[0]
    columns = ranks.split("<thead>", 1)[1].split("</thead>", 1)[0]
    assert (
        columns.index(">Budget<") < columns.index(">Rank<") < columns.index(">Metric<")
    )
    assert ">τ at 90%<" in columns
    assert '<a href="../t09-algorithm/"><span class="dp-badge' in body


NOTE = "Prose of the note.\n\n# A heading\n\nMore prose.\n"


def _page_with_note(note: str, description: str = ""):
    entry = ConfiguredAlgorithm("u1", "Alpha", "Core", (), "standard", "", 1)
    described = {
        "declaration": {"description": description, "provenance_ref": "note.md"},
        "component": {
            "name": "Alpha",
            "kind": "Algorithm",
            "path": "m:Alpha",
            "docs": "",
            "hyperparameters": (),
            "children": [],
            "references": [],
        },
        "configuration": {},
    }
    return roster_page(
        entry,
        Link("Task Name", "../index.md"),
        described,
        (),
        "../leaderboard/",
        "task",
        note,
        {},
        [],
        1.0,
        "",
        (),
        "../class/",
        methodology_url="../methodology/",
    )


def test_a_note_without_a_reference_block_renders_as_written() -> None:
    """A note without a Reference block renders as written."""
    assert _page_with_note(NOTE).provenance == _demote(NOTE, 2)


def test_a_notes_reference_block_renders_as_the_cited_works() -> None:
    """A note's Reference block renders as the works it cites."""
    page = _page_with_note(NOTE + "\nReference:\n    - kifer2012private: Appendix H.\n")
    assert page.provenance.startswith(_demote(NOTE.rstrip(), 2))
    assert "Reference:" not in page.provenance
    assert "Kifer" in page.provenance
    assert "Appendix H." in page.provenance
    assert "{ .dp-reference }" in page.provenance


def test_a_note_citing_an_unknown_key_fails_the_build() -> None:
    """A note citing a key outside references.bib fails the build."""
    with pytest.raises(KeyError, match="nosuchwork2099"):
        _page_with_note(NOTE + "\nReference:\n    - nosuchwork2099: Section 1.\n")


TUNING_RECORD = "# SCE tuning campaign c1\n\n| Family | eps | Campaign arm |\n"


@pytest.fixture
def maintainer_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render from the maintainer's own site-copy file, not the placeholders."""
    monkeypatch.setattr(
        site_copy,
        "SITE_COPY_PATH",
        Path(site_copy.__file__).resolve().parents[1] / "content" / "site-copy.yml",
    )


@pytest.mark.usefixtures("maintainer_copy")
def test_a_tuned_page_points_to_methodology_instead_of_its_tuning_record() -> None:
    page = _page_with_note(
        TUNING_RECORD,
        "Tuned on this benchmark's data in campaign c1; selected at eps=3.",
    )
    source = render(TEMPLATES["T21"], page).split("## Source", 1)[1]
    assert "This configuration was tuned on the public corpus" in source
    assert '<a href="../methodology/">Methodology</a>' in source
    assert "tuning campaign" not in source
    assert "Campaign arm" not in source


@pytest.mark.usefixtures("maintainer_copy")
def test_an_attribution_note_still_renders_on_its_page() -> None:
    note = "This configured algorithm appears as Algorithm 1 in a paper.\n"
    page = _page_with_note(note, "Appears as Algorithm 1 in a paper.")
    source = render(TEMPLATES["T21"], page).split("## Source", 1)[1]
    assert "appears as Algorithm 1 in a paper." in source
    assert "tuned on the public corpus" not in source


def test_a_task_without_explore_keeps_the_breakdown_heading_plain() -> None:
    page = replace(configured_algorithm_page(), explore_url="")
    body = render(TEMPLATES["T21"], page)
    assert "## Results breakdown on the public corpus" in body
    assert "[Explore]" not in body


def test_the_breakdown_ranks_each_budget_on_the_public_corpus(
    write_copy: WriteCopy,
) -> None:
    """Each breakdown row ranks this algorithm in that budget's public panel."""
    write_copy(ROSTER_LABELS)
    entry = next(
        entry for entry in roster_entries(INDEX, (), PROVENANCE) if entry.uuid == "u1"
    )
    described = {
        "component": {
            **PROVENANCE["u1"]["component"],
            "hyperparameters": (),
            "docs": "",
        },
        "configuration": {},
    }
    counts = {
        "success": 6,
        "no_release": 4,
        "invalid_regime": 0,
        "algorithm_error": 0,
        "timeout": 0,
        "memory_limit_exceeded": 0,
        "recorded": 10,
        "declared": 10,
    }
    public = (
        replace(
            _panel((_row("u2", 0.1), _row("u1", 0.5), _row("u3", 0.5))),
            budget="Family(eps=1.0)",
        ),
        replace(_panel((_row("u2", 0.8),)), budget="Family(eps=3.0)"),
    )
    page = roster_page(
        entry,
        Link("Task Name", "../index.md"),
        described,
        (),
        "../leaderboard/",
        "task",
        "",
        {"u1": {"Family(eps=1.0)": counts, "Family(eps=3.0)": counts}},
        ["Family(eps=1.0)", "Family(eps=3.0)"],
        1.0,
        "../explore/",
        public,
        "../class/",
        methodology_url="../methodology/",
    )
    ranks = {row.budget: row.rank for row in page.outcomes}
    # Second behind u2; u3's equal share ties it rather than pushing it down.
    assert ranks == {"ε = 1.0": "Rank 2", "ε = 3.0": "not ranked"}
