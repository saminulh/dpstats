"""Render one task's standings as its Leaderboard page.

The page ships the roster and one reading of it: every algorithm's standing at
the threshold the metric opens at, as ordinary HTML. That table is the chart's
text alternative and the page's complete roster, and it is what a reader
without scripting sees. Nothing else about the standings is inlined: the
curves, every other threshold and the ranking at a moved slider are derived
in the browser from the panel datasets ``panel_data`` emits.

Every panel reaches this page already prepared, in one shape: a measured store
ships one trial matrix per panel and a release's published view ships its curves
verbatim, and both arrive as panels with their opening standings already
traced. The comparison the page reports — its headline Metric, its baseline,
the threshold it opens at — comes from the prepared snapshot's own policy.

One section per metric, because a threshold means something different in each
metric's units; within a section, one large chart for the selected privacy
budget, because budgets are never pooled. A metric a root cannot supply
is named with the exact reason it cannot.

Only standard interpretations enter these standings and charts. Other classes
remain in the shared data for Explore, not in the official comparison.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from privacy_interpretations import interpretation_attribute
from instance_pages import number
from panel_data import ALL_SLICE, INDEX_NAME, Headline, TaskData, headline_key
from explore_controls import evaluation_settings
from site_controls import fixed_value, setting
from site_copy import copy_attribute, copy_html, copy_span, copy_text
from templates.library_source import metric_summaries
from profile_chart import (
    privacy_interpretation_href,
    REVEALED,
    absences_note,
    methodology_baseline_href,
    methodology_page_href,
    budget_panel_heading,
    default_metric,
    baseline_summary,
    element_id,
    radio_group,
    metrics_page_href,
    runtime_scripts,
    results_controls,
    results_section,
    share,
)


@dataclass(frozen=True)
class StandingRow:
    """One algorithm's standing at the threshold its page opens at.

    Attributes:
        uuid: The algorithm's identifier.
        label: Its label.
        privacy_interpretation: The privacy semantics its store registers for it,
            or ``None`` where the store registers none.
        diagnostic: Whether it is a comparison rather than a ranked entry.
        headline: Its standing under the protocol's headline quantile.
        declared: The trials the build declares for it in this panel.
        measured: Those that were measured.
    """

    uuid: str
    label: str
    privacy_interpretation: str | None
    diagnostic: bool
    headline: Headline
    declared: int
    measured: int


@dataclass(frozen=True)
class PanelStandings:
    """One metric and budget's opening standings.

    Attributes:
        metric: The Metric the standings measure.
        budget: The budget they were read under.
        rows: Every algorithm's standing, in ranked then comparison order.
    """

    metric: str
    budget: str
    rows: tuple[StandingRow, ...]


def collection_standings(data: TaskData) -> tuple[PanelStandings, ...]:
    """Read one prepared task's panels into their opening standings: the whole
    task's ("all"), which every page opens on; a page offering slices reads
    the rest in the browser."""
    labels = {
        str(entry["uuid"]): (
            str(entry["label"]),
            entry["class"],
            bool(entry["diagnostic"]),
        )
        for entry in data.index["algorithms"]
    }
    standings: list[PanelStandings] = []
    for panel in data.panels:
        if panel.slice != ALL_SLICE:
            continue
        coverage = data.index["coverage"][ALL_SLICE][f"{panel.metric}/{panel.budget}"]
        rows = [
            StandingRow(
                uuid=uuid,
                label=label,
                privacy_interpretation=declared,
                diagnostic=bool(diagnostic),
                headline=panel.headline[uuid],
                declared=coverage[uuid][0],
                measured=coverage[uuid][1],
            )
            for uuid, (label, declared, diagnostic) in labels.items()
            if uuid in panel.headline
        ]
        standings.append(
            PanelStandings(
                metric=panel.metric, budget=panel.budget, rows=_ordered(rows)
            )
        )
    return tuple(standings)


def _ordered(rows: Sequence[StandingRow]) -> tuple[StandingRow, ...]:
    """Rank one panel's standings, comparisons after them.

    The ranking rule is ``headline_order``'s; an exact tie shares the better
    rank, and tied rows are listed by algorithm so a page presents one stable
    order.
    """
    ranked = sorted(
        (row for row in rows if not row.diagnostic),
        key=lambda row: (*headline_key(row.headline), row.label),
    )
    comparisons = sorted(
        (row for row in rows if row.diagnostic), key=lambda row: row.label
    )
    return (*ranked, *comparisons)


def rank_of(row: StandingRow, ranked: Sequence[StandingRow]) -> int:
    """Position under the one ranking rule; zero for a comparison entry.

    Every page that reads a rank -- the Leaderboard table, T9 and T21/T22
    Roster pages -- calls this, so an exact tie always shares the better rank
    the same way everywhere.
    """
    if row.diagnostic:
        return 0
    key = headline_key(row.headline)
    return 1 + sum(1 for other in ranked if headline_key(other.headline) < key)


# How a slice names the Instance property it splits on: its usual symbol.
SLICE_SYMBOLS = {"d": "d", "n_over_d": "n/d"}


def slice_name(entry: Mapping[str, Any]) -> str:
    """One slice's reader name: all of the task, or the bound it splits at."""
    if entry["key"] == ALL_SLICE:
        return copy_text("leaderboard", "slice_all")
    side = "above" if "above" in entry else "at_most"
    return copy_text(
        "leaderboard",
        f"slice_{side}",
        fields={
            "feature": SLICE_SYMBOLS[entry["feature"]],
            "value": f"{float(entry[side]):g}",
        },
    )


def slice_size(entry: Mapping[str, Any]) -> str:
    """One slice's size: its problem instances and the studies they come from."""
    instances, studies = int(entry["instances"]), int(entry["studies"])
    if instances == 1:
        key = "slice_size_one_instance"
    else:
        key = "slice_size_one" if studies == 1 else "slice_size"
    return copy_text(
        "leaderboard",
        key,
        fields={"instances": f"{instances:,}", "studies": f"{studies:,}"},
    )


def _actions(selector: str) -> str:
    """The Regime, as the control bar's trailing actions, where Explore puts
    its instance filter: each narrows the problem instances counted."""
    return f'<div class="dp-control-actions">{selector}</div>' if selector else ""


def slice_selector(slices: Sequence[Mapping[str, Any]], *, scope: str) -> str:
    """The official slices a Leaderboard offers by name, "all" first, and the
    selected one's size after them, which the page script keeps current from
    every slice's size (``data-slice-sizes``); nothing where the release
    states no slice beyond "all"."""
    if len(slices) < 2:
        return ""
    sizes = {str(entry["key"]): slice_size(entry) for entry in slices}
    size = (
        '<span class="dp-slice-size" aria-live="polite" data-slice-sizes="'
        f'{html.escape(json.dumps(sizes, ensure_ascii=False), quote=True)}">'
        f"{html.escape(sizes[ALL_SLICE])}</span>"
    )
    return radio_group(
        element_id(scope, "slice"),
        ("leaderboard", "slice_label"),
        [(str(entry["key"]), slice_name(entry)) for entry in slices],
        ALL_SLICE,
        extra=size,
        marker="data-slice-select",
    )


def headline_text(reading: Headline) -> str:
    """A reading's headline cell: its error ratio, or, where it never reaches
    the quantile, the slot naming the share it does reach."""
    if math.isfinite(reading.tau):
        return number(reading.tau)
    # Half up, as the browser's Math.round rounds the same cell.
    percent = math.floor(reading.share * 100 + 0.5)
    return copy_text("leaderboard", "unreached", fields={"percent": str(percent)})


def coverage_text(declared: int, measured: int) -> str:
    """How much of its declared work one algorithm was measured on.

    An algorithm the sweep declares no trial for has no share to report.
    """
    return "—" if declared == 0 else share(measured / declared)


# The columns every rankings table carries, in order. The table declares this
# set, and every renderer of a row — this module and the script that
# re-renders after a control moves — builds from the declaration, so no row
# can carry a cell its header does not name.
COLUMNS = ("rank", "algorithm", "headline")


def _table(panel: PanelStandings, percent: str) -> str:
    """Render one panel's complete eligible roster at the opening threshold."""
    ranked = [row for row in panel.rows if not row.diagnostic]
    rows = "".join(_table_row(row, rank_of(row, ranked)) for row in ranked)
    if not rows:
        rows = (
            f'<tr><td colspan="{len(COLUMNS)}" class="dp-empty-row">'
            + copy_html("leaderboard", "empty_panel", tag="span", classes="")
            + "</td></tr>"
        )
    return (
        budget_panel_heading(panel.budget)
        + '<div class="dp-table-scroll table-responsive">'
        f'<table class="dp-leaderboard table table-vcenter table-hover" data-columns="{",".join(COLUMNS)}">'
        '<thead><tr><th scope="col">'
        + copy_span("leaderboard", "column_rank")
        + '</th><th scope="col">Algorithm</th>'
        '<th scope="col">'
        + copy_span("leaderboard", "column_headline", fields={"percent": percent})
        + "</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _table_row(row: StandingRow, rank: int) -> str:
    """Render one eligible standing as a table row."""
    return (
        "<tr>"
        f'<td class="dp-rank">{rank or "—"}</td>'
        f'<td class="dp-alg">{html.escape(row.label)}</td>'
        f'<td class="dp-value" title="{row.headline.tau!r}">'
        f"{html.escape(headline_text(row.headline))}</td>"
        "</tr>"
    )


def _metric_section(
    task_id: str,
    metric: str,
    panels: Sequence[PanelStandings],
    *,
    label: str,
    active: bool,
    baseline: str,
    baseline_key: str,
    depth: int,
    protocol: str,
    percent: str,
) -> str:
    """Supply official standings to the shared results layout."""
    href = metrics_page_href(depth, task_id, metric)
    return results_section(
        element_id("profile", task_id, metric),
        metric,
        [(panel.budget, _table(panel, percent)) for panel in panels],
        label=label,
        active=active,
        baseline_label=baseline,
        baseline_key=baseline_key,
        definition_href=href,
        baseline_definition_href=methodology_baseline_href(protocol, baseline_key),
        depth=depth,
    )


def _baseline_control(
    baselines: Mapping[str, str],
    opening: str,
    *,
    href: str,
    metric_href: str,
    description: str,
) -> str:
    """The Leaderboard's own baseline control: the protocol baseline, frozen.

    Its value as plain text, rewritten as the reader switches metrics: the
    Leaderboard reads every Metric against the one baseline its view names,
    so there is nothing to choose, and no "Algorithm…" option to open a
    picker with. Its Definition link is hidden while it would repeat the
    Metric's own.
    """
    return setting(
        fixed_value(
            ("shared", "baseline_label"),
            baselines[opening],
            attributes=" data-official-baseline",
            text_marker="data-official-baseline-label",
        ),
        baseline_summary(description, href, link=href != metric_href),
        classes="dp-baseline-control",
        attributes=" data-baseline-control",
    )


def leaderboard_body(
    task_id: str,
    standings: Sequence[PanelStandings],
    identity: str,
    *,
    depth: int,
    published: bool,
    provenance: str,
    absent: Sequence[tuple[str, str]] = (),
    policy: Mapping[str, Any],
    index: str = INDEX_NAME,
    home: bool = False,
    explore_href: str | None = None,
    roster_href: str | None = None,
    protocol_href: str | None = None,
    methodology_href: str | None = None,
) -> str:
    """Render one task's Leaderboard, one section per metric it reports.

    Args:
        task_id: The task whose standings these are.
        standings: The task's panels, metric-major and then in the order the
            panels are read.
        identity: The root the page renders, named for the reader in the page's
            own build-provenance line.
        depth: How many path segments separate this page from the site root,
            so it can load the plotting runtime that ships beside it.
        published: Whether the root published these standings.
        provenance: Where the configuration provenance its Details popovers
            read ships, named from this page.
        absent: One ``(metric, reason)`` pair for each metric this root does
            not report on.
        policy: The prepared snapshot's index, whose comparison policy names
            the headline Metric, the baseline each Metric is read against, the
            threshold the standings open at, and the grid the slider spans.
        index: The panel index URL relative to the containing page.
        home: Render the compact landing layout; its parent loads the runtime.
        explore_href: The page-relative link to this task's Explore page, or
            ``None`` where the build publishes none, which leaves the
            introduction's bare ``[Explore]`` as plain text.
        roster_href: The page-relative link to this task's Roster page, which
            the introduction's bare ``[Roster]`` names.
        protocol_href: The page-relative link to this task's methodology page,
            or ``None`` for the task's own, which every baseline's Definition
            link points into.
        methodology_href: The page-relative link to the site-wide Methodology
            page, which the introduction's bare ``[Methodology]`` names.

    Raises:
        ValueError: If there is nothing to report, which is not a page.
    """
    protocol = protocol_href or methodology_page_href(depth, task_id)
    by_metric: dict[str, list[PanelStandings]] = {}
    for panel in standings:
        by_metric.setdefault(panel.metric, []).append(panel)
    if not by_metric:
        raise ValueError(f"{identity} reports no metric for a Leaderboard page.")
    unknown = sorted(
        {
            row.label
            for panel in standings
            for row in panel.rows
            if row.privacy_interpretation is None
        }
    )
    if unknown:
        absent = [
            *absent,
            (
                copy_text("protocol", "rule_eligibility"),
                copy_text(
                    "leaderboard",
                    "missing_interpretation",
                    fields={"names": ", ".join(unknown)},
                ),
            ),
        ]
    # A Leaderboard's baseline is fixed, so each Metric's catalogue holds
    # exactly the one option the protocol reads that Metric against, already
    # labelled by the protocol's own name for it.
    baselines: Mapping[str, str] = {
        metric: str(entry["options"][0]["label"])
        for metric, entry in policy["baselines"].items()
    }
    baseline_keys: Mapping[str, str] = {
        metric: str(entry["options"][0]["key"])
        for metric, entry in policy["baselines"].items()
    }
    opening = default_metric(str(policy["headline_metric"]), list(by_metric))
    labels: Mapping[str, str] = {
        metric: str(label) for metric, label in policy["metric_labels"].items()
    }
    status = (
        ""
        if published
        else copy_html(
            "shared", "preview_badge", tag="span", classes="badge dp-preview-badge"
        )
    )
    introduction = status + (
        ""
        if home
        else copy_html(
            "leaderboard",
            "intro",
            classes="dp-protocol-pointer",
            pages={
                "Roster": roster_href,
                "Methodology": methodology_href,
                "Submit": f"{'../' * depth}submit/",
            },
        )
        + copy_html(
            "leaderboard",
            "intro_corpus",
            classes="dp-protocol-pointer",
            pages={
                "Public corpus": f"{'../' * depth}corpus/?task={task_id}",
                "Explore": explore_href,
            },
        )
    )

    # Explore's control bar, with the Regime where Explore's instance filter
    # sits; Home keeps its own compact layout.
    def bar(*arguments: Any, **keywords: Any) -> str:
        controls = results_controls(*arguments, **keywords)
        return controls if home else evaluation_settings(controls)

    if home and introduction:
        introduction = (
            '<details class="dp-home-source"><summary>'
            + copy_span("home", "source_summary")
            + "</summary>"
            + introduction
            + "</details>"
        )
    return (
        introduction
        + REVEALED
        + f'<div class="dp-results" data-leaderboard data-index="{index}" '
        f'data-algorithm-provenance="{provenance}" '
        f'data-interpretation-href="{html.escape(privacy_interpretation_href(depth), quote=True)}" '
        f'data-task="{html.escape(task_id, quote=True)}"'
        + copy_attribute("shared", "leaderboard", "component", "task")
        + interpretation_attribute()
        + ">"
        + bar(
            list(by_metric),
            opening,
            _baseline_control(
                baselines,
                opening,
                href=methodology_baseline_href(protocol, baseline_keys[opening]),
                metric_href=metrics_page_href(depth, task_id, opening),
                description=str(
                    policy["baselines"][opening]["options"][0].get("description", "")
                ),
            ),
            scope=task_id,
            quantile=float(policy["headline_quantile"]),
            extra=_actions(
                "" if home else slice_selector(policy.get("slices", []), scope=task_id)
            ),
            labels=labels,
            metric_definition_href=metrics_page_href(depth, task_id, opening),
            summaries=metric_summaries(task_id),
        )
        + absences_note(absent)
        + "".join(
            _metric_section(
                task_id,
                metric,
                panels,
                label=labels[metric],
                active=metric == opening,
                baseline=baselines[metric],
                baseline_key=baseline_keys[metric],
                depth=depth,
                protocol=protocol,
                percent=f"{float(policy['headline_quantile']) * 100:g}",
            )
            for metric, panels in by_metric.items()
        )
        + "</div>"
        + copy_html(
            "shared",
            "built_from",
            classes="dp-build-source",
            fields={"identity": identity},
        )
        + ("" if home else runtime_scripts(depth, "javascripts/leaderboard.js"))
    )
