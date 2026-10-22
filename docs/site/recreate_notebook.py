"""Emit the notebook that recreates one Explore view through the public API.

An Explore panel is a reading of one Distribution Release: a task, a Metric, a
declared budget, a threshold, a baseline, a trial aggregation, the algorithms
picked and the instance filters. The notebook reaches the same numbers by
calling the library, so the page's own arithmetic is never the authority for
what a reader can reproduce.

Each view ships one notebook. Its parameter cells are tagged: the first selects
the data, and a view with plot settings opens its plot cell with them. A build
writes the panel's opening values into them, and the page rewrites them with
the reader's selection before handing over the file.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from chart_style import chart_style
from explore_properties import PROPERTIES
from explore_reading import outcome_parameters, reading_parameters
from panel_data import FEATURE_NAMES, baseline_label
from site_copy import copy_markdown, copy_text

# The tag on each parameter cell a page rewrites.
PARAMETERS_TAG = "parameters"

# Where a parameter cell carries its names and build-time values, so the page
# rewriting them needs no list of its own.
PARAMETERS_KEY = "dp_parameters"

# The views of an Explore page, each recreated by its own notebook.
PROFILES_VIEW = "profiles"
FEATURE_VIEW = "feature"
OUTCOMES_VIEW = "outcomes"

# The file each view's notebook ships as, beside its page.
NOTEBOOK_NAMES = {
    PROFILES_VIEW: "recreate.ipynb",
    FEATURE_VIEW: "recreate-feature.ipynb",
    OUTCOMES_VIEW: "recreate-outcomes.ipynb",
}

# What one parameter may be: scalars, lists and string-keyed mappings of them.
ParameterValue: TypeAlias = (
    "str | int | float | bool | None | list[ParameterValue] | dict[str, ParameterValue]"
)


@dataclass(frozen=True)
class ReleaseIdentity:
    """Where the notebook gets the release this page was built from.

    Attributes:
        repo_id: The repository a pinned release is fetched from, or ``""``.
        revision: The immutable revision that fetch resolves to, or ``""``.
        local_root: The local tree a preview build read, or ``""``.
        version: The version the release names itself by.
    """

    repo_id: str
    revision: str
    local_root: str
    version: str


@dataclass(frozen=True)
class Selection:
    """One panel's reading, as its page holds it.

    Attributes:
        task: The task whose results are read.
        metric: The Metric the costs are measured in.
        budget: The declared budget, as the recipe states it.
        tau: The threshold the curves' guide and the outcomes split are read at.
        quantile: The share P each algorithm's error ratio is read at in the
            standings, which rank it lower first.
        baseline: ``none`` for the Metric's own units, ``best_median`` for the
            pool's best median, an instance reference's key, or
            ``algorithm:<uuid>``.
        pool: The privacy interpretation the pooled baseline ranks.
        aggregation: ``individual`` for every trial, or ``median`` for one
            summary per instance.
        algorithms: The algorithms in view, by UUID; ``None`` reads every one.
        instance_filters: Inclusive ``[low, high]`` bounds per instance
            property, ``None`` for an open end.
        plotted_algorithms: The curves shown, by UUID; ``None`` plots every
            compared algorithm.
        view: The success rate curves, the error ratio by instance property,
            or the trial outcomes by instance property.
        feature: The instance property the two property views read.
        scale: ``log``, ``linear`` or ``quantile`` placement of the property.
        y_scale: ``log`` or ``linear`` placement of an error ratio.
        algorithm: The trial outcomes view's algorithm, by UUID.
        compare_with: The algorithm it is compared with, or ``None``.
        algorithm_labels: Each algorithm's label, by UUID, as the page names it.
        series: Each algorithm's success-rate curve, by UUID, as the page
            draws it (``chart_style.profile_series``).
    """

    task: str
    metric: str
    budget: str
    tau: float
    quantile: float
    baseline: str
    pool: str
    taus: list[float]
    aggregation: str = "individual"
    algorithms: list[str] | None = None
    instance_filters: dict[str, list[float | None]] = field(
        default_factory=dict[str, list[float | None]]
    )
    plotted_algorithms: list[str] | None = None
    view: str = PROFILES_VIEW
    feature: str = "n"
    scale: str = "log"
    y_scale: str = "log"
    algorithm: str | None = None
    compare_with: str | None = None
    algorithm_labels: dict[str, str] = field(default_factory=dict[str, str])
    series: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])


def _feature_label(feature: str) -> str:
    spec = FEATURE_NAMES[feature]
    if "symbol" in spec:
        return spec["symbol"]
    group, key = spec["slot"].split(".")
    return copy_text(group, key)


def error_ratio_labels(selection: Selection) -> dict[str, str]:
    """The error ratio chart's texts, from the slots the page reads them from.

    Axis titles are written out for the selection; the hover and count texts
    keep the ``{name}`` fields the figure fills.
    """
    feature = _feature_label(selection.feature)
    counted = (
        "median_point_count" if selection.aggregation == "median" else "trial_count"
    )
    return {
        "feature": feature,
        "xlabel": (
            copy_text("explore", "scatter_x_quantile", fields={"feature": feature})
            if selection.scale == "quantile"
            else feature
        ),
        "ylabel": (
            copy_text(
                "explore",
                "scatter_y_axis",
                fields={"baseline": baseline_label(selection.baseline)},
            )
            if selection.baseline != "none"
            else selection.metric.replace("_", " ")
        ),
        "point_hover": copy_text(
            "explore",
            "point_hover",
            fields={"property": "{property}", "x": "{x}", "y": "{y}"},
        ),
        "point_beyond_view": copy_text("explore", "point_beyond_view"),
        "count_one": copy_text(
            "explore", f"{counted}_one", fields={"count": "{count}"}
        ),
        "count": copy_text("explore", counted, fields={"count": "{count}"}),
        "running_median": copy_text(
            "explore", "running_median", fields={"algorithm": "{algorithm}"}
        ),
    }


def outcome_categories() -> dict[str, str]:
    """The trial outcomes chart's categories, bottom to top, by their names.

    One algorithm's outcomes come first, then two compared algorithms' pair
    categories; the chart draws those its bins hold.
    """
    outcomes = chart_style()["layout"]["meta"]["outcomes"]
    return {
        entry["key"]: (
            copy_text(*entry["slot"].split(".")) if "slot" in entry else entry["text"]
        )
        for entry in [*outcomes["stack"], *outcomes["pairs"]]
    }


def outcome_labels(selection: Selection) -> dict[str, str]:
    """The trial outcomes chart's texts, from the slots the page reads them from.

    Axis titles and the two compared algorithms' names in ``pair_hover`` are
    written out for the selection; the hover texts keep the ``{share}``,
    ``{a}`` and ``{b}`` fields the figure fills.
    """
    feature = _feature_label(selection.feature)
    a, b = (
        selection.algorithm_labels.get(uuid, uuid or "")
        for uuid in (selection.algorithm, selection.compare_with)
    )
    return {
        "xlabel": (
            copy_text("explore", "histogram_x_quantile", fields={"feature": feature})
            if selection.scale == "quantile"
            else feature
        ),
        "ylabel": copy_text("explore", "histogram_y_axis"),
        "success_hover": copy_text(
            "explore", "bar_hover_success", fields={"share": "{share}"}
        ),
        "stack_hover": copy_text(
            "explore", "bar_hover_stack", fields={"share": "{share}"}
        ),
        "pair_hover": f"{a} ≤ τ: {{a}} · {b} ≤ τ: {{b}}",
    }


def data_parameters(selection: Selection, release: ReleaseIdentity) -> dict[str, Any]:
    """The opening cell's names: which release, panel, algorithms and instances."""
    algorithms: dict[str, Any] = (
        {"ALGORITHM": selection.algorithm, "COMPARE_WITH": selection.compare_with}
        if selection.view == OUTCOMES_VIEW
        else {"ALGORITHMS": selection.algorithms}
    )
    feature = {} if selection.view == PROFILES_VIEW else {"FEATURE": selection.feature}
    return {
        "RELEASE_REPO": release.repo_id,
        "RELEASE_REVISION": release.revision,
        "LOCAL_RELEASE_ROOT": release.local_root,
        "TASK": selection.task,
        "METRIC": selection.metric,
        "BUDGET": selection.budget,
        "TAU": selection.tau,
        "P": selection.quantile,
        **algorithms,
        "BASELINE": selection.baseline,
        "POOL": selection.pool,
        "AGGREGATION": selection.aggregation,
        "INSTANCE_FILTERS": selection.instance_filters,
        **feature,
    }


def parameter_literal(value: ParameterValue) -> str:
    """One parameter value as Python source.

    JSON spells a string, a number, a list and a mapping as Python does, and
    differs on exactly the three values below.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "[" + ", ".join(parameter_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {parameter_literal(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    return json.dumps(value, ensure_ascii=False)


def _parameter_cell(
    comment: str, values: Mapping[str, Any], body: str = "", *, hidden: bool = False
) -> dict[str, Any]:
    """A tagged cell: one comment line, one assignment per name, then ``body``.

    The page rewrites the assignments from the names and values the cell
    carries, and keeps the comment and the body.
    """
    metadata: dict[str, Any] = {"tags": [PARAMETERS_TAG], PARAMETERS_KEY: dict(values)}
    if hidden:
        metadata["jupyter"] = {"source_hidden": True}
    lines = [
        f"# {comment}\n",
        *[f"{name} = {parameter_literal(value)}\n" for name, value in values.items()],
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": lines + body.splitlines(keepends=True),
    }


_OPEN_RELEASE = """\
import json
import os
from getpass import getpass
from pathlib import Path

import pandas as pd
from huggingface_hub import get_token

from dpstats import fetch_benchmark, open_collection
from dpstats.tasks import BUILT_IN_TASKS
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    task_collection_directory,
)

if LOCAL_RELEASE_ROOT:
    WORKSPACE = Path(LOCAL_RELEASE_ROOT)
else:
    if get_token() is None:
        os.environ["HF_TOKEN"] = getpass("Hugging Face access token: ")
    WORKSPACE = Path(f"benchmark-workspace-{RELEASE_REVISION[:12]}")
    try:
        fetch_benchmark(WORKSPACE, revision=RELEASE_REVISION)
    except Exception:
        # Rerunning the cell asks for the token again.
        os.environ.pop("HF_TOKEN", None)
        raise

collection = open_collection(WORKSPACE, BUILT_IN_TASKS[TASK])
RELEASE_ROOT = collection.release.path
TASK_ROOT = task_collection_directory(RELEASE_ROOT, TASK)
manifest = json.loads((RELEASE_ROOT / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8"))
print(f"release {collection.release.version} at {RELEASE_ROOT}")
"""

_FILTER = """
properties = collection.instance_features(
    {prop.__name__: partial(prop, collection=collection) for prop in PROPERTIES}
)
kept = properties
# A range keeps the instances inside it; an open range keeps every instance,
# including one whose property is undefined (NaN).
for name, (low, high) in INSTANCE_FILTERS.items():
    if low is not None or high is not None:
        kept = kept[kept[name].between(-math.inf if low is None else low, math.inf if high is None else high)]
if ALGORITHMS == [] or kept.empty:
    raise ValueError("Nothing was selected: no algorithm, or no instance the filters keep.")
print(f"{len(kept):,} of {len(properties):,} instances")
"""


def _filter_source(task: str) -> str:
    """The task's property functions as ``explore_properties`` writes them,
    then the filter over them."""
    listed = PROPERTIES[task]
    return (
        "import math\nfrom functools import partial\n\nimport numpy as np\n\n\n"
        + "\n\n".join(inspect.getsource(prop) for prop in listed)
        + f"\n\nPROPERTIES = [{', '.join(prop.__name__ for prop in listed)}]\n"
        + _FILTER
    )


_READ = """\
from dpstats.algorithm import PrivacyInterpretation
from dpstats.analysis import (
    aggregate_trials,
    compute_baseline,
    measure_results,
    compute_error_ratio,
)
from dpstats.tasks import list_built_in_metrics

AGAINST = BASELINE.removeprefix("algorithm:") if BASELINE.startswith("algorithm:") else ""
settings = collection.load_sweep(RELEASE_ROOT / manifest["tasks"][TASK]["sweep"])
# A panel is one declared budget; the recipe's own policy selects its trials,
# and a trial outside it carries no budget group.
policy = next(budget for budget in settings["budgets"] if repr(budget) == BUDGET)
metric = list_built_in_metrics(BUILT_IN_TASKS[TASK])[METRIC]

# An algorithm read against is loaded whether or not it is shown, and the
# pooled baseline ranks every algorithm of its class.
loaded = (
    settings["algorithms"]
    if ALGORITHMS is None or BASELINE == "best_median"
    else sorted({*ALGORITHMS, AGAINST} - {""})
)
results = collection.load_results(
    settings, algorithms=loaded, instances=list(kept.index), budgets=[policy]
)
measurements = measure_results(results, [metric], collection=collection)
measurements = measurements[measurements["budget_group"].notna()]
print(f"{len(measurements):,} measured trials")
"""

_BASELINE = """\
# Each instance's share of this task's comparison, as the release ships it.
shipped = collection.load_weights()
weight = dict(zip(shipped["instance_uuid"], shipped["weight"]))

costs = measurements if AGGREGATION == "individual" else aggregate_trials(
    measurements, metric=METRIC
)
if AGAINST:
    # One algorithm's own median cost on each instance.
    against = next(algorithm for algorithm in costs["algorithm"] if algorithm.uuid == AGAINST)
    reference = compute_baseline(
        costs, metric=METRIC, baseline=against, per_budget=False, aggregation="median"
    )
    costs = costs[[algorithm.uuid != AGAINST for algorithm in costs["algorithm"]]]
elif BASELINE == "none":
    reference = costs[["task", "instance"]].drop_duplicates().assign(baseline=1.0)
elif BASELINE == "best_median":
    # The best median in the pool's privacy interpretation, over its rows alone.
    reference = compute_baseline(
        costs, metric=METRIC, baseline=PrivacyInterpretation(POOL), aggregation="median"
    )
else:
    # A value each instance stores, read by its key.
    reference = BASELINE
if ALGORITHMS is not None:
    costs = costs[[algorithm.uuid in ALGORITHMS for algorithm in costs["algorithm"]]]
if not isinstance(reference, str):
    # An instance the reference does not cover has no baseline, and is left out.
    covered = set(reference["instance"])
    costs = costs[[instance in covered for instance in costs["instance"]]]

ratios = compute_error_ratio(costs, reference, metric=METRIC)
weights = pd.DataFrame(
    {"instance": value, "weight": weight[value.uuid]}
    for value in costs["instance"].drop_duplicates()
)
"""

_STANDINGS = """\
from dpstats.analysis import compute_profile_curves

# Each algorithm's success curve, evaluated at every error ratio it reaches:
# its error ratio at P is the first at which the curve reaches P. One that
# never does ranks after every one that does, by the share it reaches, then by
# the smallest ratio at which it reaches that share.
standings = []
listed = list(ratios["instance"].drop_duplicates())
for algorithm, trials in ratios.groupby("algorithm", sort=False):
    reached = np.unique(trials[METRIC].dropna().to_numpy(dtype=float))
    reached = reached[np.isfinite(reached)]
    if not reached.size:
        standings.append((math.inf, 0.0, math.inf, algorithm))
        continue
    shares = compute_profile_curves(
        trials, reached, metric=METRIC, instances=listed, weights=weights
    )["fraction"].to_numpy()
    share = shares[-1]
    at_p = reached[shares >= P - 1e-12]
    at_share = reached[shares >= share - 1e-12][0] if share > 0 else math.inf
    standings.append((at_p[0] if at_p.size else math.inf, share, at_share, algorithm))
standings.sort(key=lambda row: (row[0], -row[1], row[2], str(row[3])))
print(f"{TASK} - {metric.label} under {BUDGET}, error ratio at P = {P}")
print("Method | Interpretation | Error ratio at P")
for at_p, _, _, algorithm in standings:
    print(f"{algorithm} | {algorithm.privacy_interpretation.value} | {at_p:.4g}")
print("Local comparison on selected Distribution instances; no official ranks.")
print("Diagnostic methods have no supported DP guarantee for the full method.")
"""

_CURVES_PLOT = """
from dpstats.analysis import plot_profiles

curves = compute_profile_curves(ratios, TAU_GRID, metric=METRIC, weights=weights)
if PLOTTED_ALGORITHMS is not None:
    curves = curves[[algorithm.uuid in PLOTTED_ALGORITHMS for algorithm in curves["algorithm"]]]
# Named as the page names them, and in its order, so equal shares rank alike.
curves = curves.assign(
    algorithm_label=[
        f"{algorithm} [{algorithm.privacy_interpretation.value}]" for algorithm in curves["algorithm"]
    ]
).sort_values("algorithm_label", kind="stable")
figure = plot_profiles(curves, xscale="linear", tau=TAU, template=CHART_STYLE, **LABELS)
figure.show()
"""

_FEATURE_POINTS = """\
from dpstats.analysis import place_error_ratio_points

# One point per trial, or per instance under the median, at its instance's
# property value and weighted rank among every instance the release weighs.
FEATURE_VALUES = properties[FEATURE]
RELEASE_WEIGHTS = pd.Series(weight)
points = place_error_ratio_points(
    ratios, metric=METRIC, feature=FEATURE_VALUES, weights=RELEASE_WEIGHTS, aggregation=AGGREGATION
)
print(f"{len(points):,} points over {points['instance'].nunique():,} instances")
"""

_FEATURE_PLOT = """
from dpstats.analysis import plot_error_ratios, trace_running_median

line = trace_running_median(
    points, feature=FEATURE_VALUES, weights=RELEASE_WEIGHTS, window=RUNNING_WINDOW, positions=RUNNING_POSITIONS
)
figure = plot_error_ratios(
    points, line, x_scale=SCALE, y_scale=Y_SCALE, fence=RANGE_FENCE, padding=RANGE_PADDING,
    template=CHART_STYLE, **LABELS,
)
figure.show()
"""

_OUTCOMES_PLOT = """
from dpstats.analysis import bin_outcomes, plot_outcomes

bins = bin_outcomes(
    measurements, reference, metric=METRIC, tau=TAU, algorithm=ALGORITHM, compare_with=COMPARE_WITH,
    feature=properties[FEATURE], weights=pd.Series(weight), instances=list(kept.index),
    scale=SCALE, bins=QUANTILE_BINS if SCALE == "quantile" else OUTCOME_BINS, aggregation=AGGREGATION,
)
print(bins.drop(columns="instances").to_string())
figure = plot_outcomes(
    bins, scale=SCALE, padding=OUTCOME_PADDING, categories=CATEGORIES, template=CHART_STYLE, **LABELS
)
figure.show()
"""

# The trial outcomes view names one algorithm and the one it is compared with;
# the cells that read the panel take them as the algorithms in view.
_OUTCOMES_ALGORITHMS = """\
if not ALGORITHM:
    raise ValueError("Nothing was selected: ALGORITHM names no algorithm.")
ALGORITHMS = [ALGORITHM] + ([COMPARE_WITH] if COMPARE_WITH else [])
"""

_DATA_COMMENT = "The data this notebook reads. Edit these and rerun."
_PLOT_COMMENT = "How the chart draws. Edit these and rerun this cell."
_STYLE_COMMENT = "The chart's texts and style, as the page draws them."


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _prose(
    heading: str,
    body: str = "",
    *,
    level: int = 2,
    fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One Markdown cell of the maintainer's text: a heading slot, then a body slot.

    A blank slot leaves its part out.
    """
    title = copy_markdown("notebook", heading)
    parts = [f"{'#' * level} {title}"] if title else []
    if body:
        text = copy_markdown("notebook", body, fields=fields)
        if text:
            parts.append(text)
    return _markdown("\n\n".join(parts))


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _view_cells(selection: Selection) -> list[dict[str, Any]]:
    """The cells that read and draw one view, after the shared reading."""
    if selection.view == FEATURE_VIEW:
        return [
            _prose("feature_heading", "feature_note"),
            _code(_FEATURE_POINTS),
            _parameter_cell(
                _STYLE_COMMENT,
                {
                    "CHART_STYLE": chart_style(),
                    "LABELS": error_ratio_labels(selection),
                },
                hidden=True,
            ),
            _parameter_cell(
                _PLOT_COMMENT,
                {
                    "SCALE": selection.scale,
                    "Y_SCALE": selection.y_scale,
                    **reading_parameters(),
                },
                _FEATURE_PLOT,
            ),
        ]
    if selection.view == OUTCOMES_VIEW:
        return [
            _prose("outcomes_heading", "outcomes_note"),
            _parameter_cell(
                _STYLE_COMMENT,
                {
                    "CHART_STYLE": chart_style(),
                    "CATEGORIES": outcome_categories(),
                    "LABELS": outcome_labels(selection),
                },
                hidden=True,
            ),
            _parameter_cell(
                _PLOT_COMMENT,
                {"SCALE": selection.scale, **outcome_parameters()},
                _OUTCOMES_PLOT,
            ),
        ]
    style = chart_style()
    style["layout"]["meta"]["series"] = selection.series
    return [
        _prose("compare_heading", "compare_note"),
        _code(_STANDINGS),
        _prose("curves_heading"),
        _parameter_cell(
            _STYLE_COMMENT,
            {
                "CHART_STYLE": style,
                "LABELS": {
                    "xlabel": copy_text("shared", "threshold_label"),
                    "ylabel": copy_text("shared", "term_success_rate"),
                    "hover": copy_text(
                        "shared",
                        "curve_hover_reading",
                        fields={"tau": "{tau}", "share": "{share}"},
                    ),
                    "guide": copy_text(
                        "shared", "threshold_guide", fields={"tau": "{tau}"}
                    ),
                },
            },
            hidden=True,
        ),
        _parameter_cell(
            _PLOT_COMMENT,
            {
                "PLOTTED_ALGORITHMS": selection.plotted_algorithms,
                "TAU_GRID": selection.taus,
            },
            _CURVES_PLOT,
        ),
    ]


def notebook_document(selection: Selection, release: ReleaseIdentity) -> dict[str, Any]:
    """Return the notebook recreating one Explore view through the public API.

    Args:
        selection: The reading the notebook opens with, which the page rewrites
            into its parameter cells before offering the file.
        release: Where the notebook gets the release this page was built from.
    """
    source = (
        copy_markdown("notebook", "source_local")
        if release.local_root
        else copy_markdown(
            "notebook",
            "source_hosted",
            fields={"repository": release.repo_id, "revision": release.revision},
        )
    )
    return {
        "cells": [
            _prose(
                "title",
                "intro",
                level=1,
                fields={"version": release.version, "source": source},
            ),
            _parameter_cell(_DATA_COMMENT, data_parameters(selection, release)),
            _prose("open_heading", "open_token"),
            _code(_OPEN_RELEASE),
            _prose("filter_heading", "condition_note"),
            _code(
                (_OUTCOMES_ALGORITHMS if selection.view == OUTCOMES_VIEW else "")
                + _filter_source(selection.task)
            ),
            _prose("read_heading", "read_lead"),
            _code(_READ),
            _prose("baseline_heading"),
            _code(_BASELINE),
            *_view_cells(selection),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_bytes(selection: Selection, release: ReleaseIdentity) -> str:
    """The notebook as the file a page hands a reader."""
    return json.dumps(notebook_document(selection, release), indent=1) + "\n"
