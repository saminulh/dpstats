"""Build the T21 configured-algorithm page record.

It reads the same three release-sourced inputs every results page
already reads once: the Explore index's Roster, the released evaluation
view's panels, and the measured provenance. Nothing here re-reads the release;
:mod:`assemble` supplies each of these already loaded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from bibliography import cited_by
from leaderboard import PanelStandings, coverage_text, headline_text, rank_of
from profile_chart import (
    budget_short_label,
    element_id,
    share_floor,
)
from site_copy import copy_html
from templates.blocks import reference_paragraphs
from templates.model import (
    ConfiguredAlgorithm,
    Link,
    OutcomeRow,
    RecipeComponent,
    RosterPage,
    StandingRow,
    rank_label,
)

_HEADING = re.compile(r"^(#+)(\s)", re.M)
_UNRANKED_PANEL = "—"
# How a release marks a tuned configured algorithm: its declared description
# opens with these words, which an attribution declaration never does.
TUNED_DESCRIPTION = "Tuned on this benchmark's data"


def _demote(text: str, levels: int = 2) -> str:
    """Push every Markdown ATX heading in *text* down by *levels*, for nesting."""
    return _HEADING.sub(lambda match: "#" * (len(match[1]) + levels) + match[2], text)


def _provenance_markdown(note: str) -> str:
    """The note's prose, then the works its ``Reference:`` block cites.

    The block is rendered through the bibliography, as a class's own is; a note
    without one is its prose unchanged.

    Raises:
        KeyError: The block cites a key the bibliography does not hold.
    """
    works = cited_by(note)
    if not works:
        return note
    prose = note.splitlines()[: note.splitlines().index("Reference:")]
    return "\n".join(prose).rstrip() + "\n\n" + reference_paragraphs(works)


def _ordered_panels(
    panels: Sequence[PanelStandings], budget_order: Sequence[str]
) -> list[PanelStandings]:
    """*panels*, metric-major in first-seen order, budget-minor in
    *budget_order* -- the same order Outcomes and the Leaderboard chips use."""
    position = {budget: index for index, budget in enumerate(budget_order)}
    metrics: list[str] = []
    for panel in panels:
        if panel.metric not in metrics:
            metrics.append(panel.metric)
    metric_rank = {metric: index for index, metric in enumerate(metrics)}
    return sorted(
        panels,
        key=lambda panel: (
            metric_rank[panel.metric],
            position.get(panel.budget, len(position)),
        ),
    )


def standing_href(leaderboard_url: str, task_id: str, metric: str, budget: str) -> str:
    """The Leaderboard URL that opens on one metric and budget."""
    query = urlencode({"metric": metric, "budget": budget})
    return f"{leaderboard_url}?{query}#{element_id('profile', task_id, metric)}"


def _standing_rows(
    entry: ConfiguredAlgorithm,
    panels: Sequence[PanelStandings],
    leaderboard_url: str,
    task_id: str,
    budget_order: Sequence[str],
) -> tuple[StandingRow, ...]:
    """Every published panel's rank, headline error ratio and coverage for one row."""
    rows: list[StandingRow] = []
    for panel in _ordered_panels(panels, budget_order):
        row = next((item for item in panel.rows if item.uuid == entry.uuid), None)
        metric_link = Link(
            panel.metric,
            standing_href(leaderboard_url, task_id, panel.metric, panel.budget),
        )
        if row is None or row.diagnostic:
            rank_text = rank_label(None)
            within, coverage = _UNRANKED_PANEL, _UNRANKED_PANEL
        else:
            ranked = [item for item in panel.rows if not item.diagnostic]
            rank_text = rank_label(rank_of(row, ranked))
            within = headline_text(row.headline)
            coverage = coverage_text(row.declared, row.measured)
        rows.append(
            StandingRow(
                metric=metric_link,
                budget=budget_short_label(panel.budget),
                rank=rank_text,
                headline=within,
                coverage=coverage,
            )
        )
    return tuple(rows)


def _outcome_cell(counts: Mapping[str, int], status: str, recorded: int) -> str:
    """One outcome's count, with its share of recorded trials where any ran."""
    count = counts[status]
    return f"{count} ({share_floor(count / recorded)})" if recorded else str(count)


def _public_rank(entry: ConfiguredAlgorithm, panel: PanelStandings | None) -> str:
    """This row's rank in one public-corpus panel, under the one ranking rule."""
    row = (
        next((item for item in panel.rows if item.uuid == entry.uuid), None)
        if panel is not None
        else None
    )
    if row is None or row.diagnostic:
        return rank_label(None)
    return rank_label(
        rank_of(row, [item for item in panel.rows if not item.diagnostic])
    )


def _outcome_rows(
    entry: ConfiguredAlgorithm,
    outcomes: Mapping[str, Mapping[str, dict[str, int]]],
    budget_order: Sequence[str],
    public_panels: Sequence[PanelStandings],
) -> tuple[OutcomeRow, ...]:
    """One row per budget the release declares trials for this row under."""
    per_budget = outcomes.get(entry.uuid, {})
    by_budget = {panel.budget: panel for panel in public_panels}
    rows: list[OutcomeRow] = []
    for budget in budget_order:
        counts = per_budget.get(budget)
        if counts is None:
            continue
        recorded = counts["recorded"]
        rows.append(
            OutcomeRow(
                budget=budget_short_label(budget),
                rank=_public_rank(entry, by_budget.get(budget)),
                success=_outcome_cell(counts, "success", recorded),
                no_release=_outcome_cell(counts, "no_release", recorded),
                invalid_regime=_outcome_cell(counts, "invalid_regime", recorded),
                algorithm_error=_outcome_cell(counts, "algorithm_error", recorded),
                timeout=_outcome_cell(counts, "timeout", recorded),
                memory_limit_exceeded=_outcome_cell(
                    counts, "memory_limit_exceeded", recorded
                ),
            )
        )
    return tuple(rows)


def _recipe_component(component: Mapping[str, Any]) -> RecipeComponent:
    """One recipe entry: its own configured values, then its named children."""
    return RecipeComponent(
        name=Link(component["name"], component.get("docs") or ""),
        values=tuple(
            (item["name"], item["value"], item["description"])
            for item in component["hyperparameters"]
        ),
        children=tuple(
            _recipe_component(child["component"]) for child in component["children"]
        ),
    )


def _chain(component: Mapping[str, Any]) -> tuple[RecipeComponent, ...]:
    """The recipe in order: an algorithm alone, or its transformations then core."""
    if component["kind"] == "Composition":
        return tuple(
            _recipe_component(child["component"]) for child in component["children"]
        )
    return (_recipe_component(component),)


def roster_page(
    entry: ConfiguredAlgorithm,
    task_link: Link,
    described: Mapping[str, Any],
    panels: Sequence[PanelStandings],
    leaderboard_url: str,
    task_id: str,
    provenance_doc: str,
    outcomes: Mapping[str, Mapping[str, dict[str, int]]],
    budget_order: Sequence[str],
    quantile: float,
    explore_url: str,
    public_panels: Sequence[PanelStandings],
    class_url: str,
    *,
    methodology_url: str,
) -> RosterPage:
    """T21: the one page record for one configured algorithm.

    *quantile* is the share the headline error ratios are read at; *explore_url* is the
    task's Explore page, empty where the task has none; *public_panels* are
    the public corpus's headline-metric panels, one per budget, that the
    breakdown's ranks read; *class_url* is the privacy-interpretation definition
    the badge links to; *methodology_url* is where a tuned configured
    algorithm's Source points in place of its tuning record.
    """
    declaration = described.get("declaration") or {}
    standings = _standing_rows(entry, panels, leaderboard_url, task_id, budget_order)
    # One "not ranked" line stands in only for a row no released panel ranks.
    unranked = all(row.rank == rank_label(None) for row in standings)
    if declaration.get("description", "").startswith(TUNED_DESCRIPTION):
        provenance = copy_html(
            "roster", "tuned_source", pages={"Methodology": methodology_url}
        )
    elif provenance_doc:
        provenance = _demote(_provenance_markdown(provenance_doc), 2)
    else:
        provenance = ""
    return RosterPage(
        algorithm=entry,
        task_link=task_link,
        description=declaration.get("description", ""),
        standings=() if unranked else standings,
        outcomes=_outcome_rows(entry, outcomes, budget_order, public_panels),
        recipe=_chain(described["component"]),
        configuration_json=json.dumps(
            described["configuration"], indent=2, sort_keys=True
        ),
        percent=f"{quantile * 100:g}",
        leaderboard_url=leaderboard_url,
        explore_url=explore_url,
        class_url=class_url,
        provenance=provenance,
        unranked_note=unranked,
    )
