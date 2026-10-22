"""Join this build's standings to the component classes each configured algorithm runs.

The class a configured algorithm is built from has its own page. The join is
by configuration identity, which the prepared snapshot's provenance carries,
never by a label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from interpretation import shown_by_default
from leaderboard import PanelStandings, headline_text, rank_of
from panel_data import Headline
from profile_chart import default_metric
from site_copy import copy_text

from .model import ChainStep, ConfiguredAlgorithm, RosterLink, RosterRow


def eligible_rows(index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The registered entries the protocol ranks, by the leaderboard's own rule.

    A page counting registered algorithms counts ranked rows, not raw
    registrations: an entry whose privacy interpretation official standings do
    not rank is a comparison, not a ranked entry.

    Args:
        index: One prepared task's index, as the leaderboard reads it.

    Returns:
        The entries that open visible and ranked, in registration order.
    """
    (eligibility,) = index["visible_classes"]
    return [
        entry
        for entry in index["algorithms"]
        if shown_by_default(entry.get("class"), str(eligibility))
    ]


def _paths(component: Mapping[str, Any]) -> list[str]:
    """Every class path in one configuration, so each page finds its own."""
    paths = [str(component["path"]).replace(":", ".")]
    for child in component.get("children", ()):
        paths.extend(_paths(child["component"]))
    return paths


def headline_panel(
    index: Mapping[str, Any], standings: Sequence[PanelStandings]
) -> PanelStandings | None:
    """The headline metric's opening panel, the same one Explore opens on.

    Not simply ``standings[0]``: the panels a page carries are not
    necessarily headline-metric-first, so this finds the one the protocol's
    own ``headline_metric`` names, falling back the way ``default_metric``
    does when this root cannot supply it.
    """
    if not standings:
        return None
    metric = default_metric(
        str(index["headline_metric"]), [panel.metric for panel in standings]
    )
    return next(panel for panel in standings if panel.metric == metric)


def roster_entries(
    index: Mapping[str, Any],
    standings: Sequence[PanelStandings],
    provenance: Mapping[str, dict[str, Any]],
) -> list[ConfiguredAlgorithm]:
    """Every algorithm the release's Roster carries for one task, ranked or not.

    Args:
        index: The task's prepared index (the Explore index's ``algorithms``).
        standings: The task's panels, as the leaderboard opens them; the
            headline metric's own opening panel is the comparison a row's
            standing is read from.
        provenance: Each row's configuration, keyed by its UUID.

    Returns:
        Every configured algorithm, in the index's own order.

    Raises:
        RuntimeError: A row names a configuration the snapshot's provenance
            does not describe, so no page could name it.
    """
    headline = headline_panel(index, standings)
    ranked_rows = (
        [row for row in headline.rows if not row.diagnostic]
        if headline is not None
        else []
    )
    by_uuid = {row.uuid: row for row in ranked_rows}
    entries: list[ConfiguredAlgorithm] = []
    for entry in index["algorithms"]:
        uuid = str(entry["uuid"])
        described = provenance.get(uuid)
        if described is None:
            raise RuntimeError(
                f"The Roster names {entry.get('label')!r} ({uuid}), which this "
                "snapshot's configuration provenance does not describe."
            )
        row = by_uuid.get(uuid)
        entries.append(
            ConfiguredAlgorithm(
                uuid=uuid,
                label=str(entry["label"]),
                core=str(entry.get("core") or ""),
                transformations=tuple(entry.get("transformations") or ()),
                privacy_interpretation=str(entry.get("class") or ""),
                marker=str(entry.get("marker") or ""),
                rank=rank_of(row, ranked_rows) if row is not None else None,
            )
        )
    return entries


# The task package a component class is declared in.
_OWNER = re.compile(r"\.tasks\.(?P<task>[a-z_]+)\.")


def owner_task(path: str) -> str:
    """The slug of the task whose package declares the class at *path*."""
    match = _OWNER.search(path)
    return match["task"] if match else ""


def _steps(component: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A configured component tree as its chain: transformations, then the core."""
    if component["kind"] != "Composition":
        return [component]
    steps: list[Mapping[str, Any]] = []
    for child in component["children"]:
        steps.extend(_steps(child["component"]))
    return steps


def transformation_facet(
    chains: Sequence[Sequence[tuple[str, str]]],
    task: str,
    task_titles: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Each transformation class path on one task's Roster, keyed for its filter.

    A class name only one class on this Roster carries is its own key, and so
    is the Roster task's own class of a shared name. Every other class of a
    shared name, which a chain reaches after handing the problem to another
    task, is keyed ``Name@task_slug`` and labelled "Name (Task class)".

    Args:
        chains: Each row's transformation steps, as ``(class name, path)``.
        task: The Roster's own task slug.
        task_titles: Each task slug's title.

    Returns:
        Each class path mapped to its ``(key, label)``.
    """
    paths: dict[str, set[str]] = {}
    for chain in chains:
        for name, path in chain:
            paths.setdefault(name, set()).add(path)
    facet: dict[str, tuple[str, str]] = {}
    for name in sorted(paths, key=str.lower):
        shared = len(paths[name]) > 1
        for path in sorted(
            paths[name], key=lambda item: (owner_task(item) != task, item)
        ):
            owner = owner_task(path)
            if not shared or owner == task:
                facet[path] = (name, name)
            else:
                title = task_titles.get(owner, owner)
                facet[path] = (
                    f"{name}@{owner}",
                    copy_text(
                        "roster",
                        "other_task_class",
                        fields={"name": name, "task": title},
                    ),
                )
    return facet


def _departures(step: Mapping[str, Any]) -> str:
    """The hyperparameters a step sets away from their defaults, as ``a=1, b=2``."""
    return ", ".join(
        f"{value['name']}={value['value']}"
        for value in step.get("hyperparameters", ())
        if value["default"] not in ("required", value["value"])
    )


def roster_rows(
    entries: Sequence[ConfiguredAlgorithm],
    provenance: Mapping[str, Mapping[str, Any]],
    headlines: Mapping[str, Headline],
    task: str,
    task_titles: Mapping[str, str],
    climb: str,
) -> tuple[list[RosterRow], dict[str, tuple[str, str]]]:
    """Every configured algorithm on one task's Roster as its chain of steps.

    Each step is named by its class, with the hyperparameters its
    configuration sets away from their defaults; the label is never parsed.

    Args:
        entries: The Roster, from :func:`roster_entries`, each with its URL.
        provenance: Each row's configuration, keyed by its UUID.
        headlines: The headline panel's error ratio at the headline quantile,
            by UUID, for the rows it ranks.
        task: The Roster's own task slug.
        task_titles: Each task slug's title, for a transformation filter label.
        climb: The way from the Roster page back to the site root.

    Returns:
        The rows, in rank order with every unranked row after them, and the
        transformation filter: each transformation class path's key and label.
    """
    chains = {
        entry.uuid: _steps(provenance[entry.uuid]["component"]) for entry in entries
    }
    facet = transformation_facet(
        [
            [(step["name"], str(step["path"]).replace(":", ".")) for step in chain[:-1]]
            for chain in chains.values()
        ],
        task,
        task_titles,
    )
    rows: list[RosterRow] = []
    for entry in entries:
        chain = chains[entry.uuid]
        steps = []
        for position, step in enumerate(chain):
            path = str(step["path"]).replace(":", ".")
            core = position == len(chain) - 1
            steps.append(
                ChainStep(
                    name=step["name"],
                    component=step["name"],
                    url=climb + step["docs"] if step.get("docs") else "",
                    hyperparameters=_departures(step),
                    key="" if core else facet[path][0],
                )
            )
        reading = headlines.get(entry.uuid)
        rows.append(
            RosterRow(
                algorithm=entry,
                chain=tuple(steps),
                headline=headline_text(reading) if reading is not None else "",
                tau=reading.tau if reading is not None else None,
            )
        )
    rows.sort(key=lambda row: (row.algorithm.rank or 10**9, row.algorithm.label))
    return rows, facet


def headline_readings(
    index: Mapping[str, Any], standings: Sequence[PanelStandings]
) -> dict[str, float]:
    """Each ranked row's headline reading on the headline panel."""
    headline = headline_panel(index, standings)
    if headline is None:
        return {}
    return {row.uuid: row.headline for row in headline.rows if not row.diagnostic}


def component_links(
    rows: Sequence[RosterRow],
    provenance: Mapping[str, Mapping[str, Any]],
    facet: Mapping[str, tuple[str, str]],
    task_title: str,
    roster_url: str,
) -> dict[str, RosterLink]:
    """Each component class's link into this Roster, filtered to the rows using it.

    An Algorithm counts the rows it is the core of and filters on that core;
    a Transformation counts the rows whose chain includes it and filters on its
    key. Each count is exactly the rows the filter it links shows.

    Args:
        rows: This task's Roster rows, from :func:`roster_rows`.
        provenance: Each row's configuration, keyed by its UUID.
        facet: The transformation filter :func:`roster_rows` returned.
        task_title: The Roster's task.
        roster_url: The Roster's URL from the site root.
    """
    cores: dict[str, tuple[str, int]] = {}
    transformations: dict[str, int] = {}
    for row in rows:
        chain = _steps(provenance[row.algorithm.uuid]["component"])
        core = str(chain[-1]["path"]).replace(":", ".")
        name, count = cores.get(core, (row.chain[-1].component, 0))
        cores[core] = (name, count + 1)
        for path in {str(step["path"]).replace(":", ".") for step in chain[:-1]}:
            transformations[path] = transformations.get(path, 0) + 1
    links = {
        path: RosterLink(task_title, count, f"{roster_url}?{urlencode({'core': name})}")
        for path, (name, count) in cores.items()
    }
    for path, count in transformations.items():
        query = urlencode({"transformation": facet[path][0]}, safe="@")
        links[path] = RosterLink(task_title, count, f"{roster_url}?{query}")
    return links
