"""The chrome both results surfaces are built out of.

A Leaderboard and an Explore page plot the same thing — what share of a group's
runs stay within a threshold — so they share one large chart, one series
colouring, one legend, and one threshold guide. Only where the curves come
from differs: a released view publishes them, a local collection is measured
for them.

One large chart for the selected privacy budget: a threshold means the same
thing in every budget and budgets are never pooled, so each keeps its own
curves and is read on its own rather than overlaid or faceted against another.
A selector switches which budget the chart and the table beneath it show;
every budget's payload and table still ship, so scripting is what shows one
at a time rather than what is available at all. The chart element itself
persists across a switch, so a reader's legend choice carries over from one
budget to the next wherever the same algorithm appears in both. Every
algorithm is a curve in every budget's chart and an entry in the legend,
always. A real roster reaches eighty, so the leading few open drawn and the
rest wait in the legend, one click from the plot.

Colour identifies the algorithm rather than the slot it happens to occupy:
``series_style`` keys it on the core estimator, so a control that reorders the
curves never re-deals the palette, and the table and the plotted-set strip
carry the same swatch. Dash says how a curve's claim is read, so an algorithm
with a nonstandard interpretation is drawn non-solid and is ranked nowhere.

Nothing here decides a comparison. The headline Metric, the baseline each
cost is read against and the threshold the standings open at all arrive from
the prepared snapshot, which was measured under one release's Benchmark
protocol; this module only says how they are shown.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module

from reader_terms import load_reader_terms
from results_roots import NO_BASELINE
from series_style import series_style
from site_controls import SELECT_CLASS, fixed_value, setting
from site_copy import copy_html, copy_span, copy_text

# The heading a chart opens with; profile-charts.js swaps it with the aggregation.


def metric_anchor(metric: str) -> str:
    """The Metrics page anchor for one metric key.

    Matches ``MetricSection.anchor`` in ``templates/model.py`` exactly: both
    derive from the same lowercase key, so a Definition link always lands on
    the right section without either side slugifying a label at
    render time.
    """
    return metric.replace("_", "-")


def metrics_page_href(depth: int, task_id: str, metric: str) -> str:
    """The link to one metric's own section on its task's Metrics page.

    Always addressed from the site root, regardless of where the linking
    page itself lives: *depth* climbs ``../`` that many times to reach the
    root, then the link descends by the task's own id. That is the one route
    every caller uses — a page inside the task's own directory tree, the
    site root Home, or a design preview living elsewhere entirely (under
    ``templates/``) — so no caller needs to know which of those it is.

    The build serves directory URLs (``metrics/`` rather than ``metrics.md``);
    a raw HTML link, unlike a Markdown one, is never rewritten for that, so it
    is spelled the served way here.

    Args:
        depth: How many ``../`` reach the site root from the linking page.
        task_id: The task whose Metrics page the link targets.
        metric: The metric key the link's section belongs to.
    """
    return f'{"../" * depth}{task_id}/metrics/#{metric_anchor(metric)}'


def methodology_page_href(depth: int, task_id: str) -> str:
    """The link to a task's methodology page, addressed from the site root."""
    return f'{"../" * depth}{task_id}/methodology/'


def methodology_baseline_href(page: str, baseline: str) -> str:
    """The link to one baseline's entry on a methodology page at *page*."""
    return f"{page}#baseline-{baseline}"


# The privacy interpretations' definitions, from the site root.
PRIVACY_INTERPRETATION_ROUTE = (
    "library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation/"
)


def privacy_interpretation_href(depth: int) -> str:
    """The page-relative link to what each privacy interpretation means."""
    return "../" * depth + PRIVACY_INTERPRETATION_ROUTE


def algorithm_baseline_href(depth: int) -> str:
    """The page-relative link to the library's baseline computation API."""
    return (
        f'{"../" * depth}library/reference/analysis/'
        "dpstats.analysis.compute_baseline/"
    )


# Mechanism groups in the panel index. ``series_style`` draws its hue groups
# over these, so a family named here is what a shared hue stands for.
ALGORITHM_FAMILIES = (
    "Exponential mechanism",
    "Objective / minimizer perturbation",
    "Gradient methods",
    "Sufficient statistics",
    "Posterior sampling",
)

# Do not infer a mechanism family for an unlisted algorithm class. A mechanism
# with one algorithm on the benchmark stays here too rather than taking a
# family of its own.
OTHER_FAMILY = "Other"

# The group for nonprivate and informal comparison curves.
DIAGNOSTIC_FAMILY = "Nonprivate / diagnostic"

# The family each configured algorithm's class belongs to, keyed on the class
# defining the estimator rather than on a configured label, so the same
# mechanism reads as one family however it was wrapped or named.
FAMILY_BY_CLASS = {
    "GridExponentialMechanism": "Exponential mechanism",
    "TukeyEM": "Exponential mechanism",
    "AMP": "Objective / minimizer perturbation",
    "ObjectivePerturbationKifer": "Objective / minimizer perturbation",
    "ObjectivePerturbationRedberg": "Objective / minimizer perturbation",
    "TailRobustSparseHuberIHT": "Gradient methods",
    "TailRobustDPGD": "Gradient methods",
    "OnePassDPGD": "Gradient methods",
    "DPSGD": "Gradient methods",
    "DPGD": "Gradient methods",
    "DPAMBSSGD": "Gradient methods",
    "SSP": "Sufficient statistics",
    "AdaSSP": "Sufficient statistics",
    "BoostedAdaSSP": "Sufficient statistics",
    "AdaOPS": "Posterior sampling",
    "OPSPTR": "Posterior sampling",
    "OLS": DIAGNOSTIC_FAMILY,
    "Ridge": DIAGNOSTIC_FAMILY,
    "ERM": DIAGNOSTIC_FAMILY,
    "GD": DIAGNOSTIC_FAMILY,
    "SGD": DIAGNOSTIC_FAMILY,
}

# Default display limit, shared by Home, Leaderboard and Explore.
LEADING_CURVES = 10

# Site-root-relative runtime every results page loads before its own
# controller: the vendored plotting bundle and the initializer that draws each
# embedded chart payload. Nothing is fetched from another origin at run time.
CHART_RUNTIME = (
    "javascripts/vendor/plotly-basic-3.7.0.min.js",
    "javascripts/panel-data.js",
    "javascripts/profile-charts.js",
)


@dataclass(frozen=True)
class RenderedPage:
    """One rendered results page and the payloads it reads beside itself.

    Attributes:
        body: The page body, as Markdown-embedded HTML.
        sidecars: The payloads to write beside the page, by file name. Empty
            for a page that embeds everything it draws.
        shared: The payloads written once for the whole root, by file name,
            which every surface of that root reads from one location. A trial
            matrix is bytes; every other payload is text.
    """

    body: str
    sidecars: dict[str, str]
    shared: dict[str, str | bytes] = field(default_factory=dict)


def default_metric(headline: str, supplied: Sequence[str]) -> str:
    """Return the metric a task's results pages open on.

    Args:
        headline: The Metric the protocol's official standings rank.
        supplied: The metrics this root supplies, in the order the selector
            offers them. A root that cannot supply the headline one opens on
            its first instead; the selector still names every metric.

    Raises:
        ValueError: If the root supplies no metric, which leaves no results
            page to open at all.
    """
    if not supplied:
        raise ValueError("No metric is supplied for these results.")
    return headline if headline in supplied else supplied[0]


# The error ratio Explore's curves and outcomes open at: the baseline itself.
# An exploration reading only; the standings rank at the protocol's quantile.
OPENING_TAU = 1.0


def algorithm_family(class_name: str) -> str:
    """Return the mechanism family one configured algorithm's class belongs to."""
    return FAMILY_BY_CLASS.get(class_name, OTHER_FAMILY)


def family_slot(family: str) -> int:
    """Return the known family's one-based index, or zero for other groups."""
    if family in ALGORITHM_FAMILIES:
        return 1 + ALGORITHM_FAMILIES.index(family)
    return 0


def element_id(*parts: str) -> str:
    """A stable, DOM-safe identifier built from the parts naming one element."""
    raw = "-".join(parts).lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", raw)).strip("-")


def embed(payload: object) -> str:
    """Serialize one payload for an inline JSON block.

    Sorted keys and no generated identifier or timestamp, so rebuilding the
    same results writes the same bytes; ``</`` is escaped so the payload
    cannot close the block that carries it.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")


def share(fraction: float) -> str:
    """Show one fraction of trials as the percentage it stands for."""
    return f"{fraction * 100:.1f}%"


def share_floor(fraction: float) -> str:
    """One fraction as a percentage; a non-zero fraction never reads 0.0%."""
    if 0 < fraction * 100 < 0.1:
        return "<0.1%"
    return share(fraction)


def results_controls(
    metrics: Sequence[str],
    opening: str,
    baseline_control_html: str,
    *,
    scope: str,
    quantile: float,
    labels: Mapping[str, str],
    metric_definition_href: str = "",
    summaries: Mapping[str, str] | None = None,
    editable_trials: bool = False,
    extra: str = "",
    notebook: bool = False,
) -> str:
    """The same comparison bar, with an editable or fixed baseline.

    The metric and baseline controls ship as one group, because a baseline
    only means something for the metric it is read against. A page whose
    reading a notebook can take again offers that notebook here, beside the
    controls whose selection it recreates.

    Args:
        baseline_control_html: The caller's complete baseline control — a
            segmented choice of named baselines plus, on Explore, the
            "Algorithm…" picker; frozen and disabled on the Leaderboard. The
            caller builds it because only it knows the named baselines a
            Metric offers.
        labels: Each metric's own short text, which the Metric control
            prints in place of its key.
        metric_definition_href: The opening metric's own Definition link
            target; unused where *frozen* is true, since a Leaderboard reads
            it from each metric's own section instead.
        summaries: Each metric's one-sentence definition as inner HTML,
            the Metric box's description beside its Details link; a metric
            without one shows the link alone.
        editable_trials: Whether the reader chooses the trial aggregation
            (Explore); otherwise it reads every trial, shown as text.
    """
    return (
        '<div class="dp-filters dp-comparison-controls pb-3 border-bottom">'
        '<div class="dp-comparison-pair">'
        + metric_control(
            metrics,
            opening,
            metric_definition_href,
            scope=scope,
            labels=labels,
            summaries=summaries or {},
        )
        + baseline_control_html
        + trials_control(editable=editable_trials)
        + "</div>"
        + extra
        + '<div class="dp-chip-group dp-threshold-control"><span class="dp-chip-label">'
        + copy_span("shared", "quantile_label")
        + "</span>"
        # The standings' quantile level p, in percent; the page reads p / 100.
        '<input class="form-range" type="range" min="1" max="99" step="1" '
        f'value="{round(quantile * 100)}" '
        f'aria-label="{html.escape(copy_text("shared", "quantile_aria"), quote=True)}">'
        '<output class="dp-value"></output></div>'
        + (notebook_control() if notebook else "")
        + "</div>"
    )


# Tabler Icons 3.35.0 (MIT) "download", as the notebook control shows it.
DOWNLOAD_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round" class="icon" aria-hidden="true">'
    '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2"/><path d="M7 11l5 5l5 -5"/>'
    '<path d="M12 4l0 12"/></svg>'
)


def notebook_control() -> str:
    """The control handing a reader this view as a runnable notebook.

    Inert without scripting, because the file is written from the selection
    the controls hold, so it stays hidden until a controller wires it.
    """
    return (
        '<div class="dp-chip-group dp-notebook-control" hidden>'
        '<button type="button" class="dp-notebook-download btn btn-sm" data-notebook-download>'
        + DOWNLOAD_ICON
        + html.escape(copy_text("explore", "download_notebook"))
        + "</button>"
        '<span class="dp-note" data-notebook-status role="status"></span></div>'
    )


def results_section(
    chart: str,
    metric: str,
    panels: Sequence[tuple[str, str]],
    *,
    label: str,
    active: bool,
    baseline_label: str = "",
    baseline_key: str = "",
    comparisons: str = "",
    definition_href: str = "",
    baseline_definition_href: str = "",
    interactive_only: bool = False,
    depth: int = 0,
    terms: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """One shared results layout; callers supply the table and optional analyses.

    *label* is the Metric's own short text, printed wherever the section
    names it for a reader; *metric* stays the key the scripts read. *depth*
    reaches the site root for the budget family's API link; *terms* defaults
    to the maintainer's reader-term file.
    """
    terms = {
        term: {
            key: f'{"../" * depth}{value}' if key == "link" else value
            for key, value in entry.items()
        }
        for term, entry in (load_reader_terms() if terms is None else terms).items()
    }
    panels = sorted(panels, key=lambda pair: budget_sort_key(pair[0]))
    budgets = [budget for budget, _ in panels]
    default = budgets[0]
    # The question and formula before the page script swaps in its own: the
    # reading's terms as plain text.
    reading = {
        "success_rate": copy_text("shared", "term_success_rate"),
        "trial": copy_text("shared", "term_trial"),
        "trials": copy_text("shared", "term_trials"),
        "instances": copy_text("shared", "term_instances"),
        "metric": label,
        "tau": "τ",
        # Reading against no baseline divides by nothing: decided by the
        # baseline's key, never by how its name is worded.
        "baseline": (
            f" × {baseline_label}"
            if baseline_label and baseline_key != NO_BASELINE
            else ""
        ),
    }
    return (
        f'<section class="dp-metric" data-metric="{html.escape(metric, quote=True)}"'
        f' data-baseline-label="{html.escape(baseline_label, quote=True)}"'
        f' data-definition-href="{html.escape(definition_href, quote=True)}"'
        f' data-baseline-definition-href="{html.escape(baseline_definition_href, quote=True)}"'
        f'{"" if active else " hidden"}>'
        + budget_selector(budgets, default, scope=chart, depth=depth)
        + (
            "<noscript>"
            + copy_html(
                "shared",
                "baseline_noscript",
                classes="",
                fields={"baseline": baseline_label},
            )
            + "</noscript>"
            if baseline_label
            else ""
        )
        + f'<div class="dp-profile-view" id="{chart}-profiles">'
        f'<section class="dp-results-profile" data-reader-terms="{html.escape(json.dumps(terms, ensure_ascii=False), quote=True)}">'
        + copy_html(
            "shared",
            "profile_question_individual",
            tag="h3",
            classes="dp-profile-heading",
            fields=reading,
        )
        + copy_html(
            "shared",
            "profile_formula_individual",
            classes="dp-profile-formula",
            fields=reading,
        )
        + f'<div class="dp-chart" id="{chart}" data-budget="{html.escape(default, quote=True)}"></div>'
        + plotted_strip(chart, top_n=active)
        + '</section><section class="dp-results-table"'
        + (" hidden" if interactive_only else "")
        + '><h3 class="dp-rankings-heading">'
        + copy_span("shared", "rankings_heading")
        + "</h3>"
        + "".join(
            budget_panel(budget, table, default=budget == default)
            for budget, table in panels
        )
        + "</section>"
        + (
            ""
            if interactive_only
            else "<noscript>" + copy_html("shared", "chart_noscript") + "</noscript>"
        )
        + "</div>"
        + comparisons
        + '<script type="application/json" data-results-section>'
        + embed({"chart": chart, "metric": metric, "budgets": budgets})
        + "</script></section>"
    )


def plotted_strip(chart: str, *, top_n: bool = False) -> str:
    """Name the curves one chart is drawing, beside it.

    The rows are filled by the controller that knows what is drawn now, so the
    strip ships empty and stays hidden until that controller is listening:
    without scripting there is no chart to name curves for, and the standings
    beneath already carry every one of them in text.

    ``top_n`` renders the profile chart's own "Show top" seed control in this
    strip's head, beside its "N plotted" count. Only the success rate view
    reads it; the comparison views keep their own selection. It is one
    control shared across every metric's panel (profile-charts.js moves the
    same node as the reader switches metrics), so only the opening metric's
    strip carries it at first render.
    """
    return (
        '<aside class="dp-plotted" data-plotted-strip hidden>'
        '<div class="dp-plotted-head">'
        f'<h4 class="dp-plotted-heading" id="{chart}-plotted">'
        + copy_span("shared", "plotted_heading")
        + "</h4>"
        + (top_n_control() if top_n else "")
        + "</div>"
        '<details class="dp-plotted-box" open><summary></summary>'
        f'<ul class="dp-plotted-rows list-group list-group-flush" aria-labelledby="{chart}-plotted"></ul>'
        '<button type="button" class="dp-plotted-more btn btn-sm" hidden></button>'
        "</details></aside>"
    )


def top_n_control() -> str:
    """The profile chart's "Show top" seed-count control.

    Lives in the success rate view's own plotted-strip head; the
    comparison views keep their own selection and never read it.
    """
    return (
        '<label class="dp-shared-baseline">'
        + copy_span("shared", "show_top")
        + f' <select class="{SELECT_CLASS}" data-top-n>'
        + "".join(
            f'<option value="{value}"{" selected" if value == str(LEADING_CURVES) else ""}>'
            + html.escape(text)
            + "</option>"
            for value, text in (
                ("5", "5"),
                ("10", "10"),
                ("20", "20"),
                ("all", copy_text("shared", "top_all")),
            )
        )
        + '<option value="custom">'
        + html.escape(copy_text("shared", "top_custom"))
        + "</option></select></label>"
    )


# One rule for the whole control bar: an exclusive choice of at most this many
# short options (a token such as a budget or a slice bound, not a phrase) is
# native radios styled as a segmented control; past either threshold it is the
# shared select, so a row of long names never outgrows the bar.
SEGMENTED_MAX_OPTIONS = 5
SEGMENTED_MAX_LABEL = 12


def fits_segmented(options: Sequence[tuple[str, str]]) -> bool:
    """Whether *options* (value, label) pairs fit the segmented-radio rule."""
    return len(options) <= SEGMENTED_MAX_OPTIONS and all(
        len(label) <= SEGMENTED_MAX_LABEL for _, label in options
    )


def radio_group(
    name: str,
    legend: tuple[str, str],
    options: Sequence[tuple[str, str]],
    selected: str,
    *,
    disabled: bool = False,
    extra: str = "",
    marker: str = "",
) -> str:
    """One exclusive choice as native radios styled as a segmented control.

    A definition link or another trailing control is a sibling of the group,
    never inside an option's own label, so it stays reachable and never joins
    a radio's accessible name (``extra``).

    Args:
        name: The ``name`` every radio in the group shares.
        legend: The site-copy slot, as ``(group, key)``, of the group's own
            plain-text legend; never a help trigger.
        options: Each option's value and reader-facing label, in order.
        selected: The value checked at first render.
        disabled: Disable every radio, for a group with nothing to choose
            (one option) or a Leaderboard's frozen baseline.
        extra: Markup appended inside the fieldset, after the options —
            an "Algorithm…" trigger or a dynamically added option, never
            inside an existing option's label.
        marker: An extra boolean attribute stamped on the fieldset (for
            example ``data-metric-select``), the stable hook a controller
            reads this group's value through.
    """
    inputs = "".join(
        f'<span class="dp-segmented-option"><input class="btn-check" type="radio" name="{html.escape(name, quote=True)}" '
        f'id="{element_id(name, value)}" value="{html.escape(value, quote=True)}"'
        f'{" checked" if value == selected else ""}{" disabled" if disabled else ""}>'
        f'<label class="btn btn-sm" for="{element_id(name, value)}">{html.escape(label)}</label></span>'
        for value, label in options
    )
    marker_attr = f" {marker}" if marker else ""
    legend_id = element_id(name, "legend")
    # A plain div with role="radiogroup" rather than a <fieldset>/<legend>: a
    # fieldset does not reliably report a wrapped row's own height in every
    # browser, which can overlap whatever follows it.
    return (
        f'<div class="dp-segmented" role="radiogroup" aria-labelledby="{legend_id}"{marker_attr}>'
        f'<span class="dp-chip-label" id="{legend_id}">{copy_span(*legend)}</span>'
        f'<div class="dp-segmented-row">' + inputs + extra + "</div></div>"
    )


def definition_link(href: str, *, attr: str = "data-definition-link") -> str:
    """The one "Full definition ↗" link, which the page moves into its control
    label's help.

    Opens in a new tab: this page's own selection state is not yet carried in
    the URL, so a same-tab navigation would lose it on Back.
    """
    return (
        f'<a class="dp-definition-link" {attr} href="{html.escape(href, quote=True)}" '
        'target="_blank" rel="noopener">'
        + copy_span("shared", "full_definition")
        + ' <span aria-hidden="true">&#8599;</span></a>'
    )


def metric_summary(
    metrics: Sequence[str], opening: str, href: str, summaries: Mapping[str, str]
) -> str:
    """The selected metric's one-sentence definition and its Details link, the
    Metric box's description; the page script swaps both as the metric
    changes, from every metric's sentence (``data-summaries``)."""
    offered = {metric: summaries[metric] for metric in metrics if metric in summaries}
    return (
        '<p class="dp-setting-desc" data-metric-summary data-summaries="'
        f'{html.escape(json.dumps(offered, sort_keys=True), quote=True)}">'
        f'<span data-metric-summary-text>{offered.get(opening, "")}</span> '
        + definition_link(href, attr="data-metric-summary-link")
        + "</p>"
    )


def baseline_summary(description: str, href: str, *, link: bool = True) -> str:
    """The Baseline box's description: what the chosen baseline is, as inner
    HTML, and its Details link (hidden where it would repeat the Metric's);
    the page script swaps both as it changes."""
    return (
        '<p class="dp-setting-desc" data-baseline-summary>'
        f"<span data-baseline-summary-text>{description}</span> "
        + definition_link(
            href, attr="data-baseline-summary-link" + ("" if link else " hidden")
        )
        + "</p>"
    )


def metric_control(
    metrics: Sequence[str],
    opening: str,
    href: str,
    *,
    scope: str,
    labels: Mapping[str, str],
    summaries: Mapping[str, str],
) -> str:
    """The Metric box: a segmented choice (or select, past the rule, or the
    value as text where there is one metric), then the selected metric's
    one-sentence definition and its Details link.

    Args:
        metrics: The metrics this page reports, in offered order.
        opening: The metric checked at first render.
        href: The opening metric's own Definition link target; a metric
            change rewrites it from the metric's own section (never
            slugified in the browser).
        labels: Each metric's own short text, printed as its option; the
            option's value stays the key the scripts read.
        summaries: Each metric's one-sentence definition as inner HTML.
    """
    options = [(metric, labels[metric]) for metric in metrics]
    name = element_id(scope, "metric")
    legend = ("shared", "metric_label")
    if len(metrics) == 1:
        # One metric is nothing to choose: its value, as text (W24). The
        # hidden radio carries the value the page scripts read.
        body = fixed_value(
            legend,
            labels[opening],
            attributes=" data-metric-select",
            extra=f'<input type="radio" value="{html.escape(opening, quote=True)}" checked hidden>',
        )
    elif fits_segmented(options):
        body = radio_group(name, legend, options, opening, marker="data-metric-select")
    else:
        rendered = "".join(
            f'<option value="{html.escape(metric, quote=True)}"'
            f'{" selected" if metric == opening else ""}>'
            f"{html.escape(labels[metric])}</option>"
            for metric in metrics
        )
        body = (
            f'<label class="dp-chip-label" for="{name}">'
            + copy_span(*legend)
            + "</label>"
            f'<select class="{SELECT_CLASS}" id="{name}" data-metric-select '
            f'aria-label="{html.escape(copy_text(*legend), quote=True)}">{rendered}</select>'
        )
    return setting(
        body,
        metric_summary(metrics, opening, href, summaries),
        classes="dp-metric-control",
        attributes=" data-metric-control",
    )


def trials_control(*, editable: bool) -> str:
    """The Trial aggregation box: Explore's choice of what the success rate
    counts, or, on the Leaderboard, which reads every trial, that value as
    text; then what the current choice means."""
    legend = ("explore", "trials_label")
    if editable:
        head = (
            f'<label class="dp-chip-label" for="trial-aggregation">'
            + copy_span(*legend)
            + "</label>"
            f'<select class="{SELECT_CLASS}" id="trial-aggregation" data-trial-aggregation>'
            '<option value="individual">'
            + html.escape(copy_text("explore", "each_trial"))
            + '</option><option value="median">'
            + html.escape(copy_text("explore", "median_per_instance"))
            + "</option></select>"
        )
    else:
        head = fixed_value(legend, copy_text("explore", "each_trial"))
    return setting(
        head,
        copy_html(
            "explore",
            "each_trial_description",
            classes="dp-setting-desc",
            attributes=" data-trials-summary",
        ),
        classes="dp-trials-control",
    )


def chip_group(
    name: str,
    label: tuple[str, str],
    options: Sequence[tuple[str, str]],
    selected: str | None,
    *,
    hidden: bool = False,
) -> str:
    """One control group, its options each pressed or not.

    A group with nothing to choose between says nothing at all, so a store
    whose instances all declare one bundle offers no bundle filter, and a task
    reporting one metric offers no metric selector.

    Args:
        name: The axis the group selects on, as the payload spells it.
        label: The site-copy slot, as ``(group, key)``, naming the group for
            a reader.
        options: Each option's payload value and what it is called for a
            reader, which differ where the payload value is a schema name.
        selected: The single option to start pressed, or ``None`` where the
            group is a multiple selection and starts with all of them pressed.
        hidden: Render the group hidden, for a page script that shows it
            where it places it, so the page does not shift as it moves.
    """
    if not options or (len(options) < 2 and name != "class"):
        return ""
    buttons = "".join(
        f'<button type="button" class="dp-chip btn btn-sm" data-filter="{name}" '
        f'data-value="{html.escape(value, quote=True)}" '
        f'aria-pressed="{"true" if selected in (None, value) else "false"}">'
        f"{html.escape(text)}</button>"
        for value, text in options
    )
    return (
        f'<div class="dp-chip-group"{" hidden" if hidden else ""}><span class="dp-chip-label">'
        f"{copy_span(*label)}</span>{buttons}</div>"
    )


# The Greek letter one budget parameter reads as, keyed on the field name a
# DPBudget or BudgetPolicy declares it under. A parameter this table does not
# name keeps its own field name in the short reading.
_BUDGET_PARAM_SYMBOLS = {"eps": "ε", "delta": "δ", "rho": "ρ", "mu": "μ"}

# A released budget label reprs the way a configured BudgetPolicy or DPBudget
# reprs itself: a family name applied to its declared fields, for example
# "SubinverseSampleSizeDelta(eps=1.0)".
_BUDGET_LABEL = re.compile(r"^(?P<family>[A-Za-z_][A-Za-z0-9_]*)\((?P<params>.*)\)$")
_BUDGET_PARAM = re.compile(r"^(?P<key>[a-z_][a-z0-9_]*)=(?P<value>[^,()=]+)$")


def _budget_parts(label: str) -> tuple[str, str] | None:
    """Split one released budget label into its family name and short reading.

    Returns ``None`` where *label* does not carry a ``Name(k=v, ...)`` shape
    naming an epsilon, so nothing here invents a reading for a value it
    cannot recognize; every caller falls back to *label* unchanged.
    """
    match = _BUDGET_LABEL.match(label.strip())
    if match is None:
        return None
    raw = match.group("params")
    params: list[tuple[str, str]] = []
    for part in raw.split(", ") if raw else ():
        param_match = _BUDGET_PARAM.match(part.strip())
        if param_match is None:
            return None
        params.append((param_match.group("key"), param_match.group("value")))
    values = dict(params)
    if "eps" not in values:
        return None
    ordered = ["eps"] + [key for key, _ in params if key != "eps"]
    short = ", ".join(
        f"{_BUDGET_PARAM_SYMBOLS.get(key, key)} = {values[key]}" for key in ordered
    )
    return match.group("family"), short


def budget_sort_key(label: str) -> tuple[str, float, str]:
    """Order released budgets by family, then epsilon as a number, then label.

    The view files name budgets alphabetically, which puts ``eps=10`` before
    ``eps=3``; every selector and panel list reads in numeric order instead.
    A label that does not parse sorts after the parsed ones, by its text.
    """
    parts = _budget_parts(label)
    if parts is None:
        return ("~", float("inf"), label)
    family, short = parts
    try:
        eps = float(short.split(",")[0].split("=")[1])
    except (IndexError, ValueError):
        eps = float("inf")
    return (family, eps, label)


def budget_family(label: str) -> str:
    """The family one released budget label names, or ``""`` where it does
    not parse."""
    parts = _budget_parts(label)
    return parts[0] if parts is not None else ""


def budget_short_label(label: str) -> str:
    """The short reading one released budget label carries, or *label*
    unchanged where it does not name an epsilon this way."""
    parts = _budget_parts(label)
    return parts[1] if parts is not None else label


def budget_choice_labels(budgets: Sequence[str]) -> tuple[dict[str, str], str]:
    """Each budget's button text and the parameters every budget shares.

    A parameter with one value across *budgets* (a fixed delta) is stated
    once, beside the choice, rather than on every button; epsilon always stays
    on the buttons. A label that does not parse keeps its short reading and
    shares nothing.
    """
    parsed = {}
    for budget in budgets:
        parts = _budget_parts(budget)
        if parts is None:
            return {budget: budget_short_label(budget) for budget in budgets}, ""
        parsed[budget] = [item.split(" = ", 1) for item in parts[1].split(", ")]
    first = parsed[budgets[0]]
    shared = [
        (symbol, value)
        for index, (symbol, value) in enumerate(first)
        if index > 0
        and all(
            len(params) > index and params[index] == [symbol, value]
            for params in parsed.values()
        )
    ]
    labels = {
        budget: ", ".join(
            f"{symbol} = {value}"
            for symbol, value in params
            if (symbol, value) not in shared
        )
        for budget, params in parsed.items()
    }
    return labels, ", ".join(f"{symbol} = {value}" for symbol, value in shared)


def budget_family_heading(budgets: Sequence[str]) -> str:
    """The one family name every *budgets* label shares, or ``""`` where they
    differ or none parses."""
    families = {budget_family(budget) for budget in budgets}
    return families.pop() if len(families) == 1 else ""


def budget_family_href(depth: int, family: str) -> str:
    """The page-relative link to a budget family's API page, or ``""`` if it has none."""
    for package in ("budgets", "sweep"):
        module = f"dpstats.{package}"
        if hasattr(import_module(module), family):
            return f'{"../" * depth}library/reference/{package}/{module}.{family}/'
    return ""


def budget_chip_group(
    budgets: Sequence[str], selected: str, *, scope: str, depth: int = 0
) -> str:
    """The control that shows one budget's chart and table at a time.

    A segmented radio past the shared rule (at most five short options)
    falls back to the shared select rather than truncating a budget's short
    reading; the family every budget shares is named once, above the row. A
    root reporting one budget renders nothing, matching ``chip_group``: there
    is nothing to switch between, and the sole panel already opens shown.
    """
    if len(budgets) < 2:
        return ""
    family = budget_family_heading(budgets)
    href = budget_family_href(depth, family) if family else ""
    name_html = (
        f'<a href="{html.escape(href, quote=True)}">{html.escape(family)}</a>'
        if href
        else html.escape(family)
    )
    heading = f'<span class="dp-budget-family">{name_html}</span>' if family else ""
    labels, shared = budget_choice_labels(budgets)
    options = [(budget, labels[budget]) for budget in budgets]
    name = element_id(scope, "budget")
    legend = ("shared", "privacy_budget_label")
    # What every budget shares, once, beside the choice.
    common = f'<p class="dp-budget-shared">{html.escape(shared)}</p>' if shared else ""
    if fits_segmented(options):
        return (
            heading
            + radio_group(name, legend, options, selected, marker="data-budget-select")
            + common
        )
    rendered = "".join(
        f'<option value="{html.escape(budget, quote=True)}"'
        f'{" selected" if budget == selected else ""}>'
        f"{html.escape(labels[budget])}</option>"
        for budget in budgets
    )
    return (
        heading
        + f'<label class="dp-chip-label" for="{name}">'
        + copy_span(*legend)
        + "</label>"
        f'<select class="{SELECT_CLASS}" id="{name}" data-budget-select '
        f'aria-label="{html.escape(copy_text(*legend), quote=True)}">{rendered}</select>'
        + common
    )


def budget_selector(
    budgets: Sequence[str], default: str, *, scope: str, depth: int = 0
) -> str:
    """Wrap ``budget_chip_group`` behind the page's filter chrome.

    Hidden until its page's controller marks the results interactive, like
    every other filter, so it never appears clickable where nothing is
    listening.
    """
    chips = budget_chip_group(budgets, default, scope=scope, depth=depth)
    return f'<div class="dp-filters">{chips}</div>' if chips else ""


def budget_panel_heading(budget: str) -> str:
    """The heading naming one budget's panel, its full label kept as a
    tooltip."""
    return (
        f'<h4 class="dp-panel-heading" title="{html.escape(budget, quote=True)}">'
        f"{html.escape(budget_short_label(budget))}</h4>"
    )


def budget_panel(budget: str, body: str, *, default: bool) -> str:
    """Wrap one budget's chart-adjacent content behind a hideable panel.

    Every budget's content ships; the default one opens visible and the rest
    wait behind ``hidden``, which ``REVEALED`` reverses where scripting is
    off.
    """
    return (
        f'<div class="dp-budget" data-budget="{html.escape(budget, quote=True)}"'
        f'{"" if default else " hidden"}>{body}</div>'
    )


def absences_note(absent: Sequence[tuple[str, str]]) -> str:
    """Name each metric a root cannot report on, and exactly why.

    A metric that cannot be reported is stated rather than left off, so a
    reader looking for it learns what is missing instead of nothing.
    """
    if not absent:
        return ""
    named = "".join(
        f"<li><code>{html.escape(metric)}</code> — {html.escape(reason)}</li>"
        for metric, reason in absent
    )
    return (
        '<div class="dp-note dp-absences">'
        + copy_html("shared", "not_reported", classes="")
        + f"<ul>{named}</ul></div>"
    )


# With scripting off nothing hides a metric's section or a budget's panel, so
# every metric and every budget a page reports is shown at once. The selectors
# are what show one at a time, and they need scripting, so the rule that
# reverses their hiding ships beside them.
REVEALED = (
    "<noscript><style>.dp-metric[hidden],.dp-budget[hidden]"
    "{display:block !important;}</style></noscript>"
)


def absence_body(identity: str, task_id: str, details_url: str) -> str:
    """Link complete missing-input or comparison-refusal reasons without ranking."""
    return (
        copy_html(
            "shared",
            "results_unavailable_lead",
            classes="",
            fields={"task": task_id, "identity": identity},
        )
        + copy_html("shared", "results_unavailable_note", classes="")
        + f'<p><a href="{html.escape(details_url, quote=True)}">'
        + copy_span("shared", "unavailability_link")
        + "</a></p>"
    )


def runtime_scripts(
    depth: int, controller: str | None = None, *, charts: bool = True
) -> str:
    """Load the shared chart runtime and one page's own controller.

    The series rule ships ahead of them, so a curve, its table marker and its
    plotted-set swatch all read one table rather than each deciding a colour.

    Args:
        depth: How many path segments separate the page from the site root, so
            it can reach the runtime that ships beside it.
        controller: The page's own script, named from the site root, or ``None``
            for a page whose charts need no controls of its own.
        charts: ``False`` for a page that draws no chart and only paints
            series swatches: it loads the series rule's script alone, not the
            plotting bundle.
    """
    prefix = "../" * depth
    runtime = CHART_RUNTIME if charts else CHART_RUNTIME[-1:]
    sources = runtime if controller is None else (*runtime, controller)
    return (
        '<script type="application/json" data-series-style>'
        + embed(series_style())
        + "</script>"
    ) + "".join(f'<script defer src="{prefix}{source}"></script>' for source in sources)
