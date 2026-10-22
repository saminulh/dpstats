"""Tests for the Explore page shell and the datasets it names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpstats.tasks import MeanEstimation
from instance_pages import number
from panel_data import BEST_BASELINE, NO_BASELINE, TaskData
from playground import playground_page
from tests.unit.docs.conftest import Element, parse_html

SITE_DIR = Path(__file__).resolve().parents[3] / "docs" / "site"

# Where a surface's pages reach the files its root wrote once.
PROVENANCE = "../../data/working/provenance.json"


@pytest.fixture(scope="module")
def body(measured_release: TaskData) -> str:
    return playground_page(
        measured_release,
        "the fixture store",
        depth=3,
        provenance=PROVENANCE,
    ).body


@pytest.fixture(scope="module")
def page(body: str) -> Element:
    return parse_html(body)


def test_the_page_writes_no_payload_beside_itself(measured_release: TaskData) -> None:
    rendered = playground_page(
        measured_release,
        "the fixture store",
        depth=3,
        provenance=PROVENANCE,
    )
    assert rendered.sidecars == {}


def test_loading_does_not_expose_retired_rankings(body: str) -> None:
    assert 'data-explore-loading role="status"' in body
    assert "data-explore-retry hidden" in body
    assert 'class="dp-results-table" hidden' in body
    assert 'class="dp-results-table">' not in body
    assert "Without it, every metric" not in body


def test_the_page_names_its_index_and_no_data_file(
    body: str, page: Element, measured_release: TaskData
) -> None:
    root = page.find("data-index")
    assert root.attrs["data-index"] == "index.json"
    assert root.attrs["data-algorithm-provenance"] == PROVENANCE
    assert root.attrs["data-task"] == MeanEstimation.id
    listed = [
        json.loads(section.raw_text)
        for section in page.find_all("data-results-section", tag="script")
    ]
    assert listed
    assert [budget for section in listed for budget in section["budgets"]] == [
        panel.budget for panel in measured_release.panels
    ]
    # The index owns every data path, so no file name reaches the page.
    for name in measured_release.files:
        assert name not in body


def test_one_section_and_one_panel_per_measured_reading(
    page: Element, measured_release: TaskData
) -> None:
    metrics = {panel.metric for panel in measured_release.panels}
    sections = page.find_all("data-metric", tag="section")
    assert {section.attrs["data-metric"] for section in sections} == metrics
    budgets = {element.attrs["data-budget"] for element in page.find_all("data-budget")}
    assert {panel.budget for panel in measured_release.panels} <= budgets


def test_budget_controls_do_not_share_identity_across_metrics(page: Element) -> None:
    ids = [element.attrs["id"] for element in page.find_all("id")]
    assert len(ids) == len(set(ids))
    names = [
        {radio.attrs["name"] for radio in group.find_all(tag="input", type="radio")}
        for group in page.find_all("data-budget-select", role="radiogroup")
    ]
    assert len(names) > 1
    assert all(len(group) == 1 for group in names)
    assert len(set.union(*names)) == len(names)


def test_each_panel_opens_tabulated_at_the_library_reading(
    page: Element, measured_release: TaskData
) -> None:
    panel = measured_release.panels[0]
    values = [number(reading.tau) for reading in panel.headline.values()]
    assert values
    for value in values[:3]:
        assert value in page.text
    # The headline column, by the slot that writes it rather than its wording.
    slots = {
        node.attrs["data-copy-slot"]
        for header in page.find_all(tag="th", scope="col")
        for node in header.find_all("data-copy-slot")
    }
    assert "leaderboard.column_headline" in slots


def test_the_controls_offer_the_opening_metrics_own_baselines(
    page: Element, measured_release: TaskData
) -> None:
    """Every kind the opening Metric's catalogue lists, and every algorithm.

    The picker opens on the baseline the protocol names, and the roster is
    never duplicated into the control: the chosen algorithm becomes one more
    option the script names, not one this build lists in advance.
    """
    opening = measured_release.index["headline_metric"]
    catalogue = measured_release.index["baselines"][opening]
    select = page.find("data-shared-baseline", tag="select")
    offered = {
        option.attrs["value"]: option for option in select.find_all(tag="option")
    }
    for option in catalogue["options"]:
        assert option["key"] in offered
        assert offered[option["key"]].text == option["label"]
    assert NO_BASELINE in [option["key"] for option in catalogue["options"]]
    assert BEST_BASELINE in [option["key"] for option in catalogue["options"]]
    assert "selected" in offered[catalogue["default"]].attrs
    assert page.find_all("data-baseline-algorithm-option")
    assert page.find_all("data-baseline-algorithm-trigger")
    for entry in measured_release.index["algorithms"]:
        assert f"algorithm:{entry['uuid']}" not in offered
    aggregation = page.find("data-trial-aggregation", tag="select")
    assert [option.attrs["value"] for option in aggregation.find_all(tag="option")] == [
        "individual",
        "median",
    ]


def test_each_metric_has_one_comparison_panel_with_both_selections(
    page: Element,
) -> None:
    """Error ratio and Outcomes share a panel, each with its own selection."""
    features = page.find_all("data-comparison-feature")
    analyses = [
        element
        for element in page.walk()
        if "dp-comparison" in element.attrs.get("class", "").split()
    ]
    assert analyses and len(analyses) == len(features)
    for analysis in analyses:
        assert len(analysis.find_all("data-comparison-selection")) == 1
        slots = analysis.find("data-outcome-slots")
        assert "hidden" in slots.attrs
        assert [
            slot.attrs["data-outcome-slot"]
            for slot in slots.find_all("data-outcome-slot")
        ] == [
            "algorithm",
            "compare",
        ]
        for slot in slots.find_all("data-outcome-slot"):
            trigger = slot.find("data-outcome-trigger")
            assert (
                trigger.attrs["popovertarget"]
                == slot.find("data-algorithm-picker").attrs["id"]
            )
            label = slot.find(tag="label")
            assert label.attrs["for"] == trigger.attrs["id"]
            assert label.attrs["id"] in trigger.attrs["aria-labelledby"].split()
        assert [
            len(slot.find_all("data-outcome-clear"))
            for slot in slots.find_all("data-outcome-slot")
        ] == [0, 1]
    for hook in (
        "data-baseline-algorithm-option",
        "data-top-n",
        "data-comparison-feature",
    ):
        assert page.find_all(hook), hook


def test_the_page_loads_the_runtime_and_the_comparison_script(page: Element) -> None:
    scripts = {
        Path(script.attrs["src"]).name for script in page.find_all("src", tag="script")
    }
    assert {
        "panel-data.js",
        "algorithm-picker.js",
        "playground.js",
        "comparison-charts.js",
    } <= scripts
    picker = next(
        script
        for script in page.find_all("src", tag="script")
        if Path(script.attrs["src"]).name == "algorithm-picker.js"
    )
    assert picker.attrs["type"] == "module"


def test_evaluation_settings_are_shared_by_all_three_rendered_views(
    page: Element,
) -> None:
    settings = page.find_all("data-evaluation-settings", tag="div")
    assert len(settings) == 1
    assert len(settings[0].find_all("data-metric-select")) == 1
    assert len(settings[0].find_all("data-shared-baseline")) == 1
    assert len(settings[0].find_all("data-trial-aggregation")) == 1
    tabs = page.find("data-view-tabs", role="tablist").find_all(role="tab")
    assert [tab.attrs["data-view"] for tab in tabs] == [
        "profiles",
        "feature",
        "outcomes",
    ]
    for tab in tabs:
        assert page.find_all(id=tab.attrs["aria-controls"])


def test_a_store_with_no_measured_panel_is_not_a_page(
    measured_release: TaskData,
) -> None:
    empty = TaskData(index=measured_release.index, files={}, panels=[], refused=[])
    with pytest.raises(ValueError, match="panel"):
        playground_page(
            empty,
            "the fixture store",
            depth=3,
            provenance=PROVENANCE,
        )


def test_an_unreported_metric_is_named_with_its_reason(
    measured_release: TaskData,
) -> None:
    rendered = playground_page(
        measured_release,
        "the fixture store",
        depth=3,
        provenance=PROVENANCE,
        absent=[("l1_error", "no Instance carries it")],
    )
    text = parse_html(rendered.body).text
    assert "l1_error" in text
    assert "no Instance carries it" in text


def test_the_condition_number_is_named_in_the_features_a_page_offers(
    measured_release: TaskData,
) -> None:
    """The filter and the feature comparison read one catalogue, not two."""
    assert "design_condition_number" in measured_release.index["features"]


def test_each_comparison_picker_names_its_own_popup_and_selection(
    page: Element,
) -> None:
    """A combobox commits into its chart's selection, wired before scripting.

    A combobox carries no multiple selection of its own, so the chosen
    algorithms live outside the field: Error ratio's plotted list, or one
    Outcomes slot. The page names each popup and label at build time: a reader
    whose scripting has not run yet still meets a correctly described control,
    and no two pickers on one page share an element id. A native popover holds
    each search, and both selections sit above the chart they choose for.
    """
    analyses = [
        element
        for element in page.walk()
        if "dp-comparison" in element.attrs.get("class", "").split()
    ]
    assert analyses, "the page renders no comparison analysis to check"
    pickers = [
        picker
        for analysis in analyses
        for picker in analysis.find_all("data-algorithm-picker")
    ]
    popups = [picker.find(role="listbox") for picker in pickers]
    assert len(pickers) == 3 * len(analyses)
    assert len({popup.attrs["id"] for popup in popups}) == len(popups)
    assert all(picker.attrs.get("popover") == "auto" for picker in pickers)
    assert all("popover" not in popup.attrs for popup in popups)
    for analysis in analyses:
        elements = list(analysis.walk())
        chart = elements.index(analysis.find(role="img"))
        selection = analysis.find("data-comparison-selection")
        slots = analysis.find("data-outcome-slots")
        assert elements.index(selection) < chart and elements.index(slots) < chart
        assert selection.find("data-comparison-add").attrs["popovertarget"] == (
            selection.find("data-algorithm-picker").attrs["id"]
        )
    controls = [
        field.attrs["aria-controls"]
        for field in (picker.find(role="combobox") for picker in pickers)
    ]
    assert sorted(controls) == sorted(popup.attrs["id"] for popup in popups)
    plotted = page.find_all("data-comparison-algorithms")
    assert len(plotted) == len(analyses)
    for rows in plotted:
        assert page.find_all(id=rows.attrs["aria-labelledby"])


def test_the_baseline_and_comparison_searches_share_one_template(page: Element) -> None:
    pickers = page.find_all("data-algorithm-picker")
    assert len(pickers) > 1
    for picker in pickers:
        search = picker.find(role="combobox")
        assert picker.find(role="listbox").attrs["id"] == search.attrs["aria-controls"]
        assert picker.find(tag="label").attrs["for"] == search.attrs["id"]
        assert len(picker.find_all(role="status")) == 1
