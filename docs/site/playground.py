"""Render one task's Explore page over the panel datasets it reads.

The Leaderboard reports one set of standings. This page lets a reader rebuild
them over the instances they pick, against the named baseline or the baseline
algorithm they pick, from individual trials or one median per instance, and read
the same costs against an instance feature. Every one
of those is a view of the same two matrices, so the page ships the matrices and
computes the views in the browser.

What is inlined is the chrome: the controls, one section per metric, one panel
per budget, and each panel's roster at the threshold the metric opens at —
which is what a reader without scripting sees, and the only reading Python
takes, through the library's own profile. The panel list names the datasets;
nothing else about the numbers is in the page.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

from privacy_interpretations import interpretation_attribute, interpretation_badge
from chart_style import chart_style
from explore_reading import outcome_parameters, reading_parameters
from site_controls import SELECT_CLASS, setting
from site_copy import copy_attribute, copy_html, copy_span, copy_text
from templates.library_source import metric_summaries
from explore_controls import (
    algorithm_picker,
    comparison_selection,
    evaluation_settings,
    outcome_slots,
    view_tabs,
)

from leaderboard import headline_text
from panel_data import (
    INDEX_NAME,
    PanelData,
    TaskData,
)
from profile_chart import (
    privacy_interpretation_href,
    REVEALED,
    RenderedPage,
    absences_note,
    algorithm_baseline_href,
    baseline_summary,
    budget_panel_heading,
    chip_group,
    default_metric,
    element_id,
    embed,
    metrics_page_href,
    baseline_href,
    baselines_page_href,
    runtime_scripts,
    results_controls,
    results_section,
)


def _class_badge(entry: Mapping[str, Any]) -> str:
    """Mark a row whose claim the roster does not show unasked."""
    if not entry["diagnostic"]:
        return ""
    name = str(entry["class"])
    return interpretation_badge(name, f"dp-badge dp-badge--{html.escape(name)} badge")


def _table(
    panel: PanelData, algorithms: Sequence[Mapping[str, Any]], percent: str
) -> str:
    """Tabulate each algorithm's opening standing, as the library traced it."""
    rows = [
        (
            not entry["diagnostic"],
            f'<tr><td class="dp-alg">{html.escape(str(entry["label"]))}'
            f"{_class_badge(entry)}</td>"
            f'<td class="dp-value">{html.escape(str(entry["class"]))}</td>'
            f'<td class="dp-value">{html.escape(str(entry["family"]))}</td>'
            f'<td class="dp-value" title="{panel.headline[str(entry["uuid"])].tau!r}">'
            f'{html.escape(headline_text(panel.headline[str(entry["uuid"])]))}</td>'
            "</tr>",
        )
        for entry in algorithms
        if str(entry["uuid"]) in panel.headline
    ]
    headings = (
        '<thead><tr><th scope="col">'
        + copy_span("shared", "column_algorithm")
        + '</th><th scope="col">'
        + copy_span("explore", "interpretation")
        + '</th><th scope="col">'
        + copy_span("explore", "column_family")
        + '</th><th scope="col">'
        + copy_span("leaderboard", "column_headline", fields={"percent": percent})
        + "</th></tr></thead>"
    )
    # Explore's own column set, declared for the script that re-renders rows.
    declared = ' data-columns="algorithm,interpretation,family,headline"'
    standard = "".join(row for eligible, row in rows if eligible)
    comparisons = "".join(row for eligible, row in rows if not eligible)
    return (
        budget_panel_heading(panel.budget)
        + '<div class="dp-table-scroll table-responsive">'
        + f'<table class="dp-leaderboard table table-vcenter table-hover"{declared}>'
        + headings
        + f"<tbody>{standard}</tbody></table></div>"
        + (
            "<noscript><details><summary>"
            + copy_span("explore", "comparison_algorithms")
            + "</summary>"
            '<div class="dp-table-scroll table-responsive">'
            + f'<table class="dp-leaderboard table table-vcenter table-hover"{declared}>'
            + headings
            + f"<tbody>{comparisons}</tbody></table></div></details></noscript>"
            if comparisons
            else ""
        )
    )


# Where this page's own task publishes its transformations, which the picker offers
# when nothing matches what a reader typed. Explore sits beside that page in
# the task's space, so the route is the same wherever the task is.
TRANSFORMATIONS_URL = "../transformations/"


def _comparison(chart: str) -> str:
    """The controls and copy of the panel behind Error ratio and Outcomes.

    Both charts read the same panel datasets the profiles do, against one
    instance feature; which algorithms each draws is its own selection, so it
    is settled in the browser.

    Args:
        chart: The chart element this analysis sits beside, which its pickers
            build their own element ids from.
    """
    # One id per analysis on the page, so each picker names its own popup.
    picker = element_id(chart, "feature", "picker")

    def paragraph(key: str, attributes: str = "") -> str:
        return copy_html("explore", key, attributes=attributes)

    def help_label(key: str, label: str, target: str) -> str:
        """A control's label that opens its help, the site's one help idiom."""
        content = paragraph(key)
        text = copy_span("explore", label)
        label_id = element_id(chart, key, "label")
        if not content:
            return f'<label id="{label_id}" for="{target}">{text}</label>'
        identifier = element_id(chart, key)
        return (
            f'<button type="button" id="{label_id}" class="dp-help-label dp-has-help"'
            f' popovertarget="{identifier}" aria-describedby="{identifier}" data-help-trigger>{text}</button>'
            f'<div id="{identifier}">{content}</div>'
        )

    feature_controls = (
        '<div class="d-flex align-items-center flex-wrap gap-3" data-comparison-feature-controls>'
        '<div class="d-flex align-items-center flex-wrap gap-1">'
        + help_label("feature_help", "feature_label", f"{picker}-feature")
        + f'<select id="{picker}-feature" class="{SELECT_CLASS}"'
        f' aria-labelledby="{element_id(chart, "feature_help", "label")}" data-comparison-feature></select>'
        '</div><div class="d-flex align-items-center flex-wrap gap-1">'
        + help_label("feature_scale_help", "scale_label", f"{picker}-scale")
        + f'<select id="{picker}-scale" class="{SELECT_CLASS}"'
        f' aria-labelledby="{element_id(chart, "feature_scale_help", "label")}" data-comparison-scale>'
        + "".join(
            f'<option value="{scale}">'
            + html.escape(copy_text("explore", f"scale_{scale}"))
            + "</option>"
            for scale in ("log", "linear", "quantile")
        )
        + "</select></div>"
        # The error-ratio axis's own scale, on the scatter only.
        '<div class="d-flex align-items-center flex-wrap gap-1" data-comparison-scatter-only>'
        f'<label for="{picker}-yscale">'
        + copy_span("explore", "y_scale_label")
        + "</label>"
        + f'<select id="{picker}-yscale" class="{SELECT_CLASS}" data-comparison-yscale>'
        + "".join(
            f'<option value="{scale}">'
            + html.escape(copy_text("explore", f"scale_{scale}"))
            + "</option>"
            for scale in ("log", "linear")
        )
        + "</select></div></div>"
    )
    return (
        '<details class="dp-comparison dp-comparison-view"'
        f' id="{chart}-feature" role="tabpanel"'
        ' aria-labelledby="playground-view-feature" hidden>'
        "<summary hidden>"
        + copy_span("explore", "view_feature")
        + "</summary>"
        + paragraph("feature_intro", " data-comparison-scatter-only")
        + paragraph("outcomes_intro", " data-comparison-histogram-only hidden")
        + comparison_selection(picker)
        + outcome_slots(element_id(chart, "outcomes", "picker"))
        + feature_controls
        + '<p data-comparison-status role="status"></p>'
        '<p class="dp-empty" role="status" data-comparison-empty hidden></p>'
        f'<button type="button" class="dp-help-label dp-has-help" data-help-trigger'
        f' data-comparison-histogram-guide popovertarget="{element_id(chart, "outcomes_guide")}" hidden>'
        + copy_span("explore", "outcomes_guide")
        + "</button>"
        f'<div id="{element_id(chart, "outcomes_guide")}">'
        + paragraph("outcomes_glossary")
        + "</div>"
        '<div class="dp-chart-shell">'
        '<div class="dp-comparison-plot" role="img" '
        f'aria-label="{html.escape(copy_text("explore", "plot_label"), quote=True)}"></div>'
        # The bar the reader is reading sits beside the chart, never below
        # it: it always shows one bin, the tallest until a bar is chosen.
        '<aside class="dp-bar-panel card" data-comparison-histogram-explain hidden>'
        '<div class="card-header"><h3 class="card-title">'
        + copy_span("explore", "bar_heading")
        + "</h3></div>"
        '<div class="card-body">'
        + paragraph("outcomes_bar_help")
        + f'<label class="form-label" for="{picker}-bin" data-histogram-bin-label>'
        + copy_span("explore", "bin")
        + "</label>"
        f'<select id="{picker}-bin" class="{SELECT_CLASS}" data-histogram-bin></select>'
        '<table class="table table-sm table-vcenter mt-3 mb-2">'
        "<caption>"
        + copy_span("explore", "breakdown_caption")
        + '</caption><thead><tr><th scope="col">'
        + copy_span("explore", "column_outcome")
        + '</th><th scope="col">'
        + copy_span("explore", "column_this_bin")
        + '</th><th scope="col">'
        + copy_span("explore", "column_all_instances")
        + "</th></tr></thead>"
        "<tbody data-histogram-breakdown></tbody></table>"
        '<p class="dp-note mb-0" data-histogram-size></p>'
        "<div data-histogram-instances></div></div></aside>"
        # The point the reader clicked sits beside the scatter as the bar does
        # beside the histogram.
        '<aside class="dp-bar-panel card" data-comparison-point-explain data-comparison-scatter-only hidden>'
        '<div class="card-header">'
        + copy_html("explore", "point_details_heading", tag="h3", classes="card-title")
        + "</div>"
        '<div class="card-body" data-comparison-runs>'
        + copy_html("explore", "point_details_prompt", tag="p", classes="dp-note mb-0")
        + "</div></aside>"
        "</div>"
        + paragraph(
            "outcomes_empty",
            ' role="status" data-comparison-histogram-empty hidden',
        )
        + '<p class="dp-note" data-comparison-histogram-note hidden></p>'
        '<details class="dp-histogram-data" data-comparison-histogram-data hidden>'
        "<summary>" + copy_span("explore", "histogram_data") + "</summary>"
        '<div class="dp-table-scroll table-responsive"><table class="dp-table table table-vcenter table-hover">'
        "<thead data-comparison-histogram-data-head></thead>"
        "<tbody data-comparison-histogram-data-body></tbody>"
        "</table></div></details>"
        '<button type="button" class="btn btn-sm" data-comparison-retry hidden>'
        + copy_span("explore", "retry_loading")
        + "</button>"
        "<p data-comparison-note></p>"
        "</details>"
    )


def _metric_section(
    task_id: str,
    metric: str,
    panels: Sequence[PanelData],
    algorithms: Sequence[Mapping[str, Any]],
    *,
    label: str,
    active: bool,
    depth: int,
    baselines_page: str,
    baseline: str,
    percent: str,
) -> str:
    """Supply the exploratory roster and analyses to the shared layout."""
    chart = element_id("playground", task_id, metric)
    href = metrics_page_href(depth, task_id, metric)
    return results_section(
        chart,
        metric,
        [(panel.budget, _table(panel, algorithms, percent)) for panel in panels],
        label=label,
        active=active,
        comparisons=_comparison(chart),
        definition_href=href,
        baseline_definition_href=baseline_href(baselines_page, baseline),
        depth=depth,
        interactive_only=True,
    )


# The Baseline select's last option: it opens the algorithm search, and is
# never itself a baseline the page reads against.
BASELINE_ALGORITHM_SEARCH = "algorithm-search"


def _baseline_control(
    index: Mapping[str, Any], metric: str, *, depth: int, baselines_page: str
) -> str:
    """The baseline control for one Metric: named baselines plus "Algorithm…".

    The named baselines are that Metric's own, including the one the protocol
    reads it against, which opens selected. The roster itself never lists
    here: the select's last option, "Algorithm…", opens the page's shared
    algorithm search in a choose-baseline role, and the chosen algorithm
    becomes one more option the script names, cleared by the × beside it.
    """
    catalogue: Mapping[str, Any] = index["baselines"][metric]
    default = str(catalogue["default"])
    options = [
        (
            str(option["key"]),
            str(option["label"]),
        )
        for option in catalogue["options"]
    ]
    named = "".join(
        f'<option value="{html.escape(key, quote=True)}"'
        f'{" selected" if key == default else ""}>{html.escape(label)}</option>'
        for key, label in options
    )
    # The chosen algorithm, once there is one, is one more option the script
    # names; the last option opens the algorithm search rather than choosing.
    choices = (
        named
        + '<option value="" data-baseline-algorithm-option hidden></option>'
        + f'<option value="{BASELINE_ALGORITHM_SEARCH}" data-baseline-algorithm-trigger>'
        + html.escape(copy_text("explore", "baseline_algorithm_trigger"))
        + "</option>"
    )
    legend = ("shared", "baseline_label")
    href = baseline_href(baselines_page, default)
    description = next(
        str(option.get("description", ""))
        for option in catalogue["options"]
        if option["key"] == default
    )
    head = (
        '<label class="dp-chip-label" for="baseline">' + copy_span(*legend) + "</label>"
        '<span class="dp-baseline-choice">'
        f'<select class="{SELECT_CLASS}" id="baseline" '
        f'name="baseline" data-shared-baseline>{choices}</select>'
        '<button type="button" class="dp-baseline-algorithm-clear btn btn-sm btn-icon btn-ghost-secondary" '
        'data-baseline-algorithm-clear hidden aria-label="'
        + html.escape(copy_text("explore", "baseline_forget_aria"), quote=True)
        + '">&times;</button></span>'
    )
    return setting(
        head,
        baseline_summary(description, href)
        + '<p class="dp-note" role="status" data-baseline-status hidden></p>'
        + algorithm_picker("baseline-picker", "baseline_picker_label"),
        classes="dp-baseline-control",
        attributes=" data-baseline-control",
    )


def _controls(
    metrics: Sequence[str],
    opening: str,
    index: Mapping[str, Any],
    *,
    depth: int,
    notebook: bool,
    baselines_page: str,
) -> str:
    """Render the control bar: what to read, against what, and over which trials."""
    declared = sorted({str(entry["class"]) for entry in index["algorithms"]})
    baseline_control = _baseline_control(
        index, opening, depth=depth, baselines_page=baselines_page
    )
    groups = [
        chip_group(
            "class",
            ("explore", "interpretation"),
            [(name, name) for name in declared],
            None,
            hidden=True,
        ),
    ]
    return evaluation_settings(
        results_controls(
            metrics,
            opening,
            baseline_control,
            scope=str(index["task"]),
            quantile=float(index["headline_quantile"]),
            labels={
                str(key): str(text) for key, text in index["metric_labels"].items()
            },
            metric_definition_href=metrics_page_href(
                depth, str(index["task"]), opening
            ),
            summaries=metric_summaries(str(index["task"])),
            editable_trials=True,
            extra="".join(groups),
            notebook=notebook,
        )
    )


def playground_page(
    data: TaskData,
    identity: str,
    *,
    depth: int,
    provenance: str,
    absent: Sequence[tuple[str, str]] = (),
    notebooks: Mapping[str, str] | None = None,
    baselines_href: str | None = None,
    leaderboard_href: str | None = None,
) -> RenderedPage:
    """Render one task's Explore page over its emitted panel datasets.

    Args:
        data: The task's datasets, as ``panel_data.task_data`` emitted them.
        identity: The store the page reads, named for the reader.
        depth: How many path segments separate this page from the site root,
            so it can load the chart runtime that ships beside it.
        provenance: Where the configuration provenance its Details popovers
            read ships, named from this page.
        absent: One ``(metric, reason)`` pair for each metric this store does
            not report on.
        notebooks: Each view's notebook recreating this page's reading, by
            view, relative to the page, or ``None`` where it ships none. The
            one download control hands over the open view's.
        baselines_href: The page-relative link to the page carrying this
            task's baselines table, or ``None`` for the task's own Metrics page,
            which every baseline's Definition link points into.
        leaderboard_href: The page-relative link to this task's Leaderboard,
            or ``None`` for the task's own.

    Raises:
        ValueError: If there is no measured panel to report at all.
    """
    if not data.panels:
        raise ValueError(f"{identity} measured no panel for an Explore page.")
    index = data.index
    task_id = str(index["task"])
    baselines_page = baselines_href or baselines_page_href(depth, task_id)
    by_metric: dict[str, list[tuple[PanelData, Mapping[str, Any]]]] = {}
    for panel, listed in zip(data.panels, index["panels"]):
        by_metric.setdefault(panel.metric, []).append((panel, listed))
    opening = default_metric(str(index["headline_metric"]), list(by_metric))
    body = (
        copy_html(
            "explore",
            "introduction",
            classes="dp-protocol-pointer",
            pages={
                "Public corpus": f"{'../' * depth}corpus/?task={task_id}",
                "Leaderboard": leaderboard_href
                or f"{'../' * depth}{task_id}/leaderboard/",
            },
        )
        + REVEALED
        + f'<div class="dp-results dp-playground" data-playground data-index="{INDEX_NAME}"'
        + (
            f' data-notebooks="{html.escape(json.dumps(dict(notebooks)), quote=True)}"'
            if notebooks
            else ""
        )
        + f' data-algorithm-baseline-href="{html.escape(algorithm_baseline_href(depth), quote=True)}"'
        + f' data-baselines-href="{html.escape(baselines_page, quote=True)}"'
        + f' data-algorithm-provenance="{provenance}"'
        + f' data-interpretation-href="{html.escape(privacy_interpretation_href(depth), quote=True)}"'
        + f' data-task="{html.escape(task_id, quote=True)}"'
        + f' data-transformations-url="{TRANSFORMATIONS_URL}"'
        + copy_attribute("shared", "explore", "component", "leaderboard", "task")
        + interpretation_attribute()
        + ">"
        '<div data-explore-loading role="status"><p>'
        + copy_html("explore", "loading", tag="span", classes="")
        + "</p>"
        '<button type="button" class="btn btn-sm" data-explore-retry hidden>'
        + copy_span("explore", "retry_loading")
        + '</button><a href="">'
        + copy_span("explore", "reload_page")
        + "</a></div>"
        + "<noscript>"
        + copy_html("shared", "chart_noscript")
        + "</noscript>"
        + _controls(
            list(by_metric),
            opening,
            index,
            depth=depth,
            notebook=bool(notebooks),
            baselines_page=baselines_page,
        )
        + view_tabs(element_id("playground", task_id, opening))
        + absences_note(absent)
        + '<p class="dp-empty" role="status" hidden></p>'
        + "".join(
            _metric_section(
                task_id,
                metric,
                [panel for panel, _ in entries],
                index["algorithms"],
                label=str(index["metric_labels"][metric]),
                active=metric == opening,
                depth=depth,
                baselines_page=baselines_page,
                baseline=str(index["baselines"][metric]["default"]),
                percent=f"{float(index['headline_quantile']) * 100:g}",
            )
            for metric, entries in by_metric.items()
        )
        + "</div>"
        # The instance filter's range sliders: noUiSlider, themed by Tabler's own
        # vendor sheet, loaded on this page alone, ahead of the deferred controller.
        + "".join(
            f'<link rel="stylesheet" href="{"../" * depth}stylesheets/vendor/{sheet}">'
            for sheet in ("nouislider-15.8.1.min.css", "tabler-vendors-1.5.1.min.css")
        )
        + f'<script src="{"../" * depth}javascripts/vendor/nouislider-15.8.1.min.js"></script>'
        + runtime_scripts(depth)
        # The property views' reading constants and chart style, which their
        # notebooks are written with too.
        + '<script type="application/json" data-explore-chart>'
        + embed(
            {
                "reading": {**reading_parameters(), **outcome_parameters()},
                "style": chart_style(),
            }
        )
        + "</script>"
        + f'<script type="module" src="{"../" * depth}javascripts/algorithm-picker.js"></script>'
        + f'<script defer src="{"../" * depth}javascripts/playground.js"></script>'
        + f'<script src="{"../" * depth}javascripts/comparison-charts.js"></script>'
        + (
            f'<script src="{"../" * depth}javascripts/notebook-download.js"></script>'
            if notebooks
            else ""
        )
    )
    return RenderedPage(body=body, sidecars={})
