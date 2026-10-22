"""Build the T23 "Methodology" page record for one Benchmarked task.

The page states the release's own comparison policy, which Metric the official
standings rank against which baseline at which threshold and grid, and then
catalogues every baseline the task offers. Two sides declare baselines: the
library declares each Metric's statistical baseline, a value it computes from
an instance's own data, and the protocol declares
external baselines, values the source supplies, with the Metrics each is
offered for, and the computed ones its ranking defines. What the site offers
is the protocol's policy; a notebook may divide anything by anything. The
ranking comes from the released document; the library side from the
library's own declarations. Nothing here resolves a baseline or computes a
value.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from dpstats.task import Task
from dpstats.tasks import (
    list_built_in_metric_arguments,
    list_built_in_metrics,
)

from profile_chart import metrics_page_href
from site_copy import copy_html, copy_text
from results_roots import (
    BEST_MEDIAN,
    NO_BASELINE,
    REFERENCE_BASELINE_KIND,
    ReleasedBaseline,
    ReleasedComparison,
)
from templates.blocks import code
from templates.library_source import inline
from templates.model import Link, Protocol, ProtocolBaseline

# The slot naming which side declares a baseline, as a reader reads it.
LIBRARY = "declared_library"
BENCHMARK_PROTOCOL = "declared_protocol"


def _number(value: float) -> str:
    """A threshold or bound as the reader reads it, without trailing zeros."""
    return f"{value:g}"


def _computed_definition(baseline: ReleasedBaseline) -> str:
    """What one computed baseline divides a cost by, as inner HTML."""
    detail = dict(baseline.detail)
    slot, named = {
        "best_median": ("baseline_best_median", "pool"),
        "best_of": ("baseline_best_of", "algorithms"),
        "algorithm": ("baseline_algorithm", "uuid"),
    }.get(baseline.kind, ("", ""))
    if not slot:
        return ""
    fields = {"aggregation": str(detail["aggregation"]), named: str(detail[named])}
    return copy_html("protocol", slot, tag="span", classes="", fields=fields)


def baseline_definition(task: Task, comparison: ReleasedComparison, key: str) -> str:
    """How one baseline is computed or where it comes from, as inner HTML.

    A library value is described by the library's own declaration, an
    external baseline by the protocol's description, and a computed baseline
    by the fields its declaration states.
    """
    if key == NO_BASELINE:
        return _computed_definition(ReleasedBaseline(NO_BASELINE, "none"))
    if key in comparison.external:
        return escape(comparison.external[key].description)
    if key in comparison.definitions:
        return _computed_definition(comparison.definitions[key])
    if (
        key == comparison.headline_baseline
        and comparison.headline_definition.kind != REFERENCE_BASELINE_KIND
    ):
        return _computed_definition(comparison.headline_definition)
    argument = list_built_in_metric_arguments(task).get(key)
    if argument is None:
        raise ValueError(
            f"Task {task.id!r} offers baseline {key!r}, which neither the library "
            "nor the protocol declares."
        )
    return inline(argument.description)


def _links(metrics: Sequence[str], metric_links: dict[str, Link]) -> str:
    """Links to Metric sections; a cell of them reads as a sentence."""
    if not metrics:
        return ""
    return (
        '<span class="dp-sentence">'
        + ", ".join(
            f'<a href="{escape(metric_links[metric].url, quote=True)}">'
            f"{escape(metric_links[metric].text)}</a>"
            for metric in metrics
        )
        + "</span>"
    )


def _standings(comparison: ReleasedComparison) -> tuple[tuple[str, str], ...]:
    """The rules the standings follow, each as the protocol names it.

    The site-wide Methodology page explains each rule; this page states the
    values this task's ranking takes.
    """
    return (
        (copy_text("protocol", "rule_weighting"), code(comparison.weighting)),
        (copy_text("protocol", "rule_eligibility"), code(comparison.eligibility)),
    )


def protocol_page(
    task: Task,
    title: str,
    comparison: ReleasedComparison,
    identity: str,
    *,
    methodology: Link,
    depth: int,
    explored: Sequence[str] | None = None,
) -> Protocol:
    """The T23 record for one task, from its comparison policy and the library.

    Args:
        task: The Benchmarked task.
        title: The page title.
        comparison: The task's entry in the protocol document the build reads.
        identity: The release the page was built from, as the reader is told.
        methodology: The site-wide Methodology page, which the lead links.
        depth: How many ``../`` reach the site root from the page.
        explored: The Metrics Explore lists, in its order, or ``None`` for
            every library Metric. "Applies to" names only those.

    Raises:
        ValueError: If the protocol names a Metric the library does not define.
    """
    defined = list_built_in_metrics(task)
    named = [
        comparison.headline_metric,
        *(m for entry in comparison.external.values() for m in entry.metrics),
    ]
    undefined = sorted({metric for metric in named if metric not in defined})
    if undefined:
        raise ValueError(
            f"The protocol names Metric(s) {undefined} for task {task.id!r}, "
            "which the library does not define."
        )
    # A reader reads the Metric's label; the link reaches its section by key.
    metric_links = {
        metric: Link(defined[metric].label, metrics_page_href(depth, task.id, metric))
        for metric in defined
    }
    headline = comparison.headline_baseline
    shown = list(defined) if explored is None else list(explored)

    def entry(name: str, declarer: str, applies: Sequence[str]) -> ProtocolBaseline:
        return ProtocolBaseline(
            name=name,
            declarer=copy_text("protocol", declarer),
            definition=baseline_definition(task, comparison, name),
            applies=_links([m for m in applies if m in shown], metric_links),
            role=copy_text(
                "protocol", "role_headline" if name == headline else "role_offered"
            ),
        )

    # The computed baselines a reader is offered: the headline's own where it
    # is computed, the pooled one and none.
    computed = [*comparison.computed_for(comparison.headline_metric), NO_BASELINE]
    rows: dict[str, ProtocolBaseline] = {}
    for name, external in comparison.external.items():
        rows[name] = entry(name, BENCHMARK_PROTOCOL, external.metrics)
    # A library row is a value some Metric names as its statistical baseline;
    # it applies to those Metrics. Other declared values are never offered.
    statistical: dict[str, list[str]] = {}
    for name, metric in defined.items():
        if metric.baseline_name is not None:
            statistical.setdefault(metric.baseline_name, []).append(name)
    for name, metrics in statistical.items():
        rows.setdefault(name, entry(name, LIBRARY, metrics))
    for name in computed:
        # Every Metric Explore lists is offered the pooled baseline and its
        # own units.
        applies = (
            [comparison.headline_metric]
            if name == headline and name not in (BEST_MEDIAN, NO_BASELINE)
            else shown
        )
        rows.setdefault(name, entry(name, BENCHMARK_PROTOCOL, applies))
    ordered = [headline, *(name for name in rows if name != headline)]

    grid = comparison.grid
    return Protocol(
        title=title,
        methodology=methodology,
        headline=metric_links[comparison.headline_metric],
        baseline=Link(headline, f"#baseline-{headline}"),
        percent=f"{comparison.headline_quantile * 100:g}",
        baselines=tuple(rows[name] for name in ordered),
        grid=(
            _number(grid.min),
            _number(grid.max),
            grid.scale,
            str(grid.steps),
        ),
        identity=identity,
        standings=_standings(comparison),
    )
