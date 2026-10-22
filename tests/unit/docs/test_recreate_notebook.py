"""The notebook an Explore page hands a reader reaches that page's own numbers.

Each check generates the notebook over the fixture release, runs its code cells
in order, and compares what it printed, or the tables a reader's cells hold, with
what the build computed for the same reading. A page is only worth offering if the file it hands over says the same
thing.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dpstats.analysis import compute_robust_range
from dpstats.tasks import MeanEstimation
from dpstats.workspace import TaskCollection
from panel_data import TaskData, released_weights
from recreate_notebook import (
    FEATURE_VIEW,
    PROFILES_VIEW,
    NOTEBOOK_NAMES,
    OUTCOMES_VIEW,
    PARAMETERS_KEY,
    PARAMETERS_TAG,
    ReleaseIdentity,
    Selection,
    error_ratio_labels,
    notebook_document,
    outcome_labels,
    parameter_literal,
)

from tests.unit.docs.conftest import (
    FAILING_ALGORITHM,
    FIXTURE_INSTANCES,
    FIXTURE_TAUS,
    MEASURED_METRIC,
    STORED_REFERENCE,
    VIEW_BASELINE,
    budget_labels,
)

# The budget the fixture store's own runs were measured under.
MEASURED_BUDGET = budget_labels()[0]

# The threshold the fixture reading opens at, and the share it ranks at.
OPENING_TAU = 1.0
HEADLINE_QUANTILE = 0.9


def _selection(budget: str, **overrides: Any) -> Selection:
    """One reading of the fixture release, as the page would hold it.

    A page opens the headline Metric on its headline baseline, so that is
    what a selection carries unless the reader picked another.
    """
    chosen: dict[str, Any] = {
        "baseline": VIEW_BASELINE,
        "pool": "standard",
        **overrides,
    }
    tau = chosen.pop("tau", OPENING_TAU)
    quantile = chosen.pop("quantile", HEADLINE_QUANTILE)
    return Selection(
        task=MeanEstimation.id,
        metric=MEASURED_METRIC,
        budget=budget,
        tau=tau,
        quantile=quantile,
        taus=[float(value) for value in FIXTURE_TAUS],
        **chosen,
    )


def _run(root: Path, selection: Selection) -> tuple[str, dict[str, Any]]:
    """Run one generated notebook's code cells, returning its output and state."""
    document = notebook_document(
        selection, ReleaseIdentity("", "", str(root), "fixture-release")
    )
    namespace: dict[str, Any] = {"__name__": "__main__"}
    printed = io.StringIO()
    for position, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = compile("".join(cell["source"]), f"cell-{position}", "exec")
        # A figure is kept in the cells' state rather than opened.
        with contextlib.redirect_stdout(printed), mock.patch.object(go.Figure, "show"):
            exec(source, namespace)  # noqa: S102 - the cells are this build's own
    return printed.getvalue(), namespace


def _parameter_cells(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in document["cells"]
        if PARAMETERS_TAG in cell["metadata"].get("tags", [])
    ]


def _parameters(document: dict[str, Any]) -> dict[str, Any]:
    """Every name the page rewrites, across the notebook's parameter cells."""
    return {
        name: value
        for cell in _parameter_cells(document)
        for name, value in cell["metadata"][PARAMETERS_KEY].items()
    }


def _rendered(output: str) -> dict[str, float]:
    """Each error ratio at P the notebook printed, by its printed label."""
    return {
        label.strip(): float(value)
        for label, _, value in re.findall(
            r"^(.+?) \| (\w+) \| ([-+.e0-9]+|inf)$", output, flags=re.M
        )
    }


def _labels(data: TaskData) -> dict[str, str]:
    return {
        str(entry["uuid"]): str(entry["label"]) for entry in data.index["algorithms"]
    }


@pytest.fixture(scope="module")
def opening_run(measurable_release_root: Path) -> tuple[str, dict[str, Any]]:
    """The notebook a page opens on, run once: its output and its cells' state."""
    return _run(measurable_release_root, _selection(MEASURED_BUDGET))


def test_the_explore_notebook_prints_the_shares_that_page_shows(
    measured_release: TaskData, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """Measuring the release again prints the build's own headline readings."""
    (shipped,) = [
        panel.headline
        for panel in measured_release.panels
        if (panel.metric, panel.budget) == (MEASURED_METRIC, MEASURED_BUDGET)
    ]
    labels = _labels(measured_release)
    output, _ = opening_run
    assert shipped

    # The notebook prints each error ratio at P to four significant figures.
    assert _rendered(output) == pytest.approx(
        {labels[uuid]: reading.tau for uuid, reading in shipped.items()}, rel=1e-3
    )


def test_the_selection_decides_which_algorithms_the_notebook_reads(
    measurable_release_root: Path, measured_release: TaskData
) -> None:
    """A reader's roster is the notebook's roster, not every registered entry."""
    labels = _labels(measured_release)
    picked = list(labels)[:2]
    output, _ = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, algorithms=picked),
    )

    assert set(_rendered(output)) == {labels[uuid] for uuid in picked}


def test_the_selection_decides_which_instances_the_notebook_reads(
    measurable_release_root: Path,
    measured_release: TaskData,
    opening_run: tuple[str, dict[str, Any]],
) -> None:
    """A sample-size range is a different reading, and the notebook takes it."""
    small = {
        str(entry["uuid"])
        for entry in measured_release.index["instances"]
        if entry["n"] < 10
    }
    _, whole = opening_run
    _, narrowed = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, instance_filters={"n": [None, 10.0]}),
    )

    assert 0 < len(narrowed["ratios"]) < len(whole["ratios"])
    assert {value.uuid for value in narrowed["ratios"]["instance"]} == small


def test_a_median_over_trials_summarizes_each_trial_group_before_it_is_read(
    measurable_release_root: Path, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """The aggregation a reader picks reaches the runs the profile is traced over."""
    _, individual = opening_run
    _, median = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, aggregation="median"),
    )

    assert len(median["costs"]) < len(individual["costs"])
    assert all(0.0 <= share <= 1.0 for _, share, _, _ in median["standings"])


def test_an_algorithm_baseline_is_read_against_rather_than_ranked(
    measurable_release_root: Path, measured_release: TaskData
) -> None:
    """Choosing an algorithm as the baseline takes it out of the standings."""
    labels = _labels(measured_release)
    against = next(iter(labels))
    output, _ = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, baseline=f"algorithm:{against}"),
    )

    assert set(_rendered(output)) == set(labels.values()) - {labels[against]}


def test_an_instance_the_algorithm_baseline_misses_is_left_out(
    measurable_release_root: Path, measured_release: TaskData
) -> None:
    """An Instance the baseline algorithm has no trial on has no denominator:
    the notebook reads every other Instance, as the page does."""
    against = next(iter(_labels(measured_release)))
    load_results = TaskCollection.load_results
    missing: list[str] = []

    def without_one(self: TaskCollection, *args: Any, **kwargs: Any) -> pd.DataFrame:
        results = load_results(self, *args, **kwargs)
        missing[:] = missing or [results["instance"].iloc[0].uuid]
        return results[
            [
                not (algorithm.uuid == against and instance.uuid == missing[0])
                for algorithm, instance in zip(
                    results["algorithm"], results["instance"]
                )
            ]
        ]

    with mock.patch.object(TaskCollection, "load_results", without_one):
        _, state = _run(
            measurable_release_root,
            _selection(MEASURED_BUDGET, baseline=f"algorithm:{against}"),
        )

    read = {instance.uuid for instance in state["ratios"]["instance"]}
    assert missing[0] not in read
    assert len(read) == len(FIXTURE_INSTANCES) - 1
    assert isinstance(state["figure"], go.Figure)


def test_an_outcomes_notebook_reads_against_an_algorithm_baseline(
    measurable_release_root: Path, measured_release: TaskData
) -> None:
    """Read against a third algorithm, the outcomes notebook bins and draws."""
    first, second, against = _algorithms(measured_release)[:3]
    _, state = _run(
        measurable_release_root,
        _selection(
            MEASURED_BUDGET,
            view=OUTCOMES_VIEW,
            baseline=f"algorithm:{against}",
            algorithm=first,
            compare_with=second,
        ),
    )
    bins = state["bins"]
    assert sum(len(members) for members in bins["instances"]) > 0
    assert isinstance(state["figure"], go.Figure)


def test_the_pair_hover_names_the_two_selected_algorithms() -> None:
    """The notebook's pair hover names the algorithms as the page does."""
    document = notebook_document(
        _selection(
            MEASURED_BUDGET,
            view=OUTCOMES_VIEW,
            algorithm="a",
            compare_with="b",
            algorithm_labels={"a": "Alpha", "b": "Beta"},
        ),
        ReleaseIdentity("", "", "/release", "fixture-release"),
    )
    assert _parameters(document)["LABELS"]["pair_hover"] == (
        "Alpha ≤ τ: {a} · Beta ≤ τ: {b}"
    )


@pytest.mark.parametrize("view", [PROFILES_VIEW, FEATURE_VIEW, OUTCOMES_VIEW])
def test_each_parameter_cell_carries_the_names_a_page_rewrites(view: str) -> None:
    """Each tagged cell opens with a comment, then one assignment per name it
    carries, in order; the page rewrites those lines and keeps the rest."""
    document = notebook_document(
        _selection(MEASURED_BUDGET, view=view),
        ReleaseIdentity("owner/set", "a" * 40, "", "v1"),
    )

    for cell in _parameter_cells(document):
        declared = cell["metadata"][PARAMETERS_KEY]
        head = cell["source"][1 : 1 + len(declared)]
        assert cell["source"][0].startswith("# ")
        assert [line.split(" = ")[0] for line in head] == list(declared)
        assert head == [
            f"{name} = {parameter_literal(value)}\n" for name, value in declared.items()
        ]
    opening = _parameter_cells(document)[0]["metadata"][PARAMETERS_KEY]
    assert opening["LOCAL_RELEASE_ROOT"] == ""
    assert opening["RELEASE_REVISION"] == "a" * 40


def test_the_error_ratio_plot_cell_opens_with_its_settings() -> None:
    """The data cell selects the data; the style cell sits collapsed above the
    plot cell, which holds the chart's own settings."""
    document = notebook_document(
        _selection(MEASURED_BUDGET, view=FEATURE_VIEW),
        ReleaseIdentity("", "", "/cut", "v1"),
    )
    data, style, plot = _parameter_cells(document)

    assert "SCALE" not in data["metadata"][PARAMETERS_KEY]
    assert list(style["metadata"][PARAMETERS_KEY]) == ["CHART_STYLE", "LABELS"]
    assert style["metadata"]["jupyter"] == {"source_hidden": True}
    assert list(plot["metadata"][PARAMETERS_KEY]) == [
        "SCALE",
        "Y_SCALE",
        "RUNNING_WINDOW",
        "RUNNING_POSITIONS",
        "RANGE_FENCE",
        "RANGE_PADDING",
    ]
    body = "".join(plot["source"])
    assert "trace_running_median(" in body and "plot_error_ratios(" in body
    assert document["cells"].index(style) + 1 == document["cells"].index(plot)


def test_the_curves_plot_cell_draws_with_the_series_the_page_draws() -> None:
    """The collapsed style cell carries each algorithm's curve look; the plot
    cell opens with the plotted set and draws through plot_profiles."""
    look = {"a": {"color": "#6929c4", "symbol": "circle", "dash": "solid"}}
    document = notebook_document(
        _selection(MEASURED_BUDGET, series=look),
        ReleaseIdentity("", "", "/cut", "v1"),
    )
    _, style, plot = _parameter_cells(document)

    assert list(style["metadata"][PARAMETERS_KEY]) == ["CHART_STYLE", "LABELS"]
    assert style["metadata"]["jupyter"] == {"source_hidden": True}
    chart = style["metadata"][PARAMETERS_KEY]["CHART_STYLE"]
    assert chart["layout"]["meta"]["series"] == look
    assert list(plot["metadata"][PARAMETERS_KEY]) == ["PLOTTED_ALGORITHMS", "TAU_GRID"]
    assert "tau=TAU" in "".join(plot["source"])
    assert "template=CHART_STYLE" in "".join(plot["source"])
    assert document["cells"].index(style) + 1 == document["cells"].index(plot)


def test_a_pinned_release_is_fetched_and_a_preview_is_named(
    measurable_release_root: Path,
) -> None:
    """A reader of a published page gets the release; a preview names the cut."""
    pinned = notebook_document(
        _selection(MEASURED_BUDGET),
        ReleaseIdentity("owner/set", "a" * 40, "", "v1"),
    )
    preview = notebook_document(
        _selection(MEASURED_BUDGET),
        ReleaseIdentity("", "", "/cut", "v1"),
    )
    code = "".join(
        "".join(cell["source"])
        for cell in pinned["cells"]
        if cell["cell_type"] == "code"
    )

    assert "fetch_benchmark(" in code
    assert _parameters(pinned)["RELEASE_REPO"] == "owner/set"
    assert _parameters(preview)["RELEASE_REPO"] == ""
    assert _parameters(preview)["LOCAL_RELEASE_ROOT"] == "/cut"


def test_the_weights_the_notebook_reads_are_the_shipped_ones(
    measurable_release_root: Path, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """The Instance weights come from the view, so the site chooses none."""
    shipped = released_weights(measurable_release_root, MeanEstimation.id)
    _, state = opening_run

    assert dict(
        zip(
            state["weights"]["instance"].map(lambda value: value.uuid),
            state["weights"]["weight"],
        )
    ) == pytest.approx(shipped)


def test_the_measured_budget_is_the_declared_budget_the_panel_names(
    opening_run: tuple[str, dict[str, Any]],
) -> None:
    """A panel is one declared budget, and the recipe's own policy selects it."""
    _, state = opening_run

    assert repr(state["policy"]) == MEASURED_BUDGET


def test_a_computed_reference_baseline_divides_by_each_instances_own_value(
    measurable_release_root: Path,
) -> None:
    """Each fixture's sample-mean standard error changes only the division."""
    _, raw = _run(measurable_release_root, _selection(MEASURED_BUDGET, baseline="none"))
    _, stored = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, baseline=STORED_REFERENCE),
    )
    assert [value.uuid for value in stored["ratios"]["instance"]] == [
        value.uuid for value in raw["ratios"]["instance"]
    ]

    observed: list[float] = []
    expected: list[float] = []
    for instance, against, cost in zip(
        stored["ratios"]["instance"],
        stored["ratios"][MEASURED_METRIC],
        raw["ratios"][MEASURED_METRIC],
    ):
        # A refused run is infinite against any denominator and carries no ratio.
        if not (math.isfinite(float(against)) and math.isfinite(float(cost))):
            continue
        observed.append(float(against) / float(cost))
        samples, dim = instance.metadata.n_samples, instance.metadata.dim
        X = np.vander(np.linspace(0.2, 1.0, samples), dim, increasing=True)
        standard_error = np.sqrt(np.var(X, axis=0, ddof=1).sum() / samples)
        expected.append(1.0 / standard_error)
    assert observed
    assert observed == pytest.approx(expected)


def test_the_pooled_baseline_ranks_the_pool_the_selection_names(
    measurable_release_root: Path, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """The pooled class is a parameter, read from the protocol by the page."""
    _, standard = opening_run
    _, qualified = _run(
        measurable_release_root, _selection(MEASURED_BUDGET, pool="qualified")
    )
    assert not standard["reference"]["baseline"].equals(
        qualified["reference"]["baseline"]
    )


def test_the_notebook_reads_the_metrics_own_units_when_no_baseline_is_chosen(
    measurable_release_root: Path,
) -> None:
    """Choosing no baseline divides by one, leaving the measured cost itself."""
    _, raw = _run(measurable_release_root, _selection(MEASURED_BUDGET, baseline="none"))
    costs = list(raw["costs"][MEASURED_METRIC])
    assert list(raw["ratios"][MEASURED_METRIC]) == pytest.approx(costs)


def test_the_notebook_recomputes_the_condition_number_it_filters_on(
    measurable_release_root: Path, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """The bounds are a parameter, and the feature behind them is recomputed.

    The fixture designs are Vandermonde matrices: the two one-column designs
    have a single singular value and so a condition number of one, and the two
    wider ones are worse conditioned. A ceiling between them therefore keeps
    exactly the Instances with the narrow designs, which the notebook has to
    reach by measuring rather than by replaying a list.
    """
    _, whole = opening_run
    _, narrowed = _run(
        measurable_release_root,
        _selection(
            MEASURED_BUDGET, instance_filters={"design_condition_number": [None, 10.0]}
        ),
    )

    kappa = dict(whole["properties"]["design_condition_number"])
    assert len(kappa) == 4
    kept = {uuid for uuid, value in kappa.items() if value <= 10.0}
    assert 0 < len(kept) < len(kappa)
    assert {value.uuid for value in narrowed["ratios"]["instance"]} == kept
    assert len(narrowed["measurements"]) < len(whole["measurements"])


def test_an_unbounded_condition_range_narrows_nothing(
    opening_run: tuple[str, dict[str, Any]],
) -> None:
    """The page opens unbounded, so its notebook opens on every Instance."""
    _, whole = opening_run

    assert len(whole["kept"]) == len(whole["properties"])
    assert len(whole["measurements"]) == sum(
        repr(group) == whole["BUDGET"] for group in whole["results"]["budget_group"]
    )


def test_the_notebook_declares_the_instance_filters_as_parameters(
    measurable_release_root: Path,
) -> None:
    """The reader edits the ranges in the first cell, like every other choice."""
    document = notebook_document(
        _selection(
            MEASURED_BUDGET,
            instance_filters={"n": [2, None], "design_condition_number": [2.0, 500.0]},
        ),
        ReleaseIdentity("", "", str(measurable_release_root), "fixture-release"),
    )
    opening = _parameter_cells(document)[0]

    assert opening["metadata"][PARAMETERS_KEY]["INSTANCE_FILTERS"] == {
        "n": [2, None],
        "design_condition_number": [2.0, 500.0],
    }
    source = "".join(opening["source"])
    assert (
        'INSTANCE_FILTERS = {"n": [2, None], "design_condition_number": [2.0, 500.0]}\n'
        in source
    )


def test_every_algorithm_and_no_algorithm_are_written_differently() -> None:
    """``None`` is every algorithm and an empty list is none, in the cell itself."""
    release = ReleaseIdentity("", "", "/cut", "v1")

    def opening(**overrides: Any) -> str:
        document = notebook_document(_selection(MEASURED_BUDGET, **overrides), release)
        return "".join(_parameter_cells(document)[0]["source"])

    assert "ALGORITHMS = None\n" in opening()
    assert "ALGORITHMS = []\n" in opening(algorithms=[])
    assert 'ALGORITHMS = ["a"]\n' in opening(algorithms=["a"])


def test_an_empty_selection_is_refused_rather_than_read_as_every_algorithm(
    measurable_release_root: Path,
) -> None:
    """Nothing selected exports a notebook that stops and says so."""
    for overrides in (
        {"algorithms": []},
        {"instance_filters": {"n": [1e9, None]}},
    ):
        with pytest.raises(ValueError, match="Nothing was selected"):
            _run(measurable_release_root, _selection(MEASURED_BUDGET, **overrides))


def test_the_notebook_plots_every_compared_algorithm_with_its_interpretation(
    measured_release: TaskData, opening_run: tuple[str, dict[str, Any]]
) -> None:
    """Every selected method keeps its curve and privacy class in the export."""
    output, state = opening_run

    compared = {str(row[-1]) for row in state["standings"]}
    assert compared == set(_labels(measured_release).values())
    assert {str(value) for value in state["curves"]["algorithm"]} == compared
    assert "Method | Interpretation | Error ratio at P" in output
    assert "Local comparison" in output
    assert "no official ranks" in output
    table = output.split("Method | Interpretation | Error ratio at P", 1)[1].split(
        "Local comparison", 1
    )[0]
    assert not re.search(r"^\s*\d+\s+\S", table, re.M)
    for *_, algorithm in state["standings"]:
        declared = algorithm.privacy_interpretation.value
        assert f"{algorithm} | {declared} |" in output
        assert f"{algorithm} [{declared}]" in {
            trace.name for trace in state["figure"].data
        }


@pytest.mark.parametrize("count", [0, 1])
def test_the_plotted_subset_preserves_standings_and_baselines(
    measurable_release_root: Path,
    measured_release: TaskData,
    opening_run: tuple[str, dict[str, Any]],
    count: int,
) -> None:
    """Hiding curves changes only the figure, including when none are shown."""
    picked = list(_labels(measured_release))[:count]
    output, state = _run(
        measurable_release_root,
        _selection(MEASURED_BUDGET, plotted_algorithms=picked),
    )
    original_output, original = opening_run

    assert _rendered(output) == _rendered(original_output)
    pd.testing.assert_frame_equal(state["reference"], original["reference"])
    pd.testing.assert_series_equal(
        state["ratios"][MEASURED_METRIC], original["ratios"][MEASURED_METRIC]
    )
    assert {algorithm.uuid for algorithm in state["curves"]["algorithm"]} == set(picked)
    assert len(state["figure"].data) == count


def test_the_page_writes_the_notebook_texts_under_the_names_the_plots_take() -> None:
    """A download rewrites ``LABELS`` from the page's own object, whose keys
    must be the keyword names the plot calls unpack them into."""
    script = (
        Path(__file__).resolve().parents[3]
        / "docs/site/docs/javascripts/comparison-charts.js"
    ).read_text(encoding="utf-8")
    outcomes, feature = [
        set(re.findall(r"^\s*(\w+):", block, re.M))
        for block in re.findall(r"LABELS: \{(.*?)\n\s*\},", script, re.S)
    ]
    assert outcomes == set(
        outcome_labels(_selection(MEASURED_BUDGET, view=OUTCOMES_VIEW, feature="n"))
    )
    assert feature == set(
        error_ratio_labels(_selection(MEASURED_BUDGET, view=FEATURE_VIEW, feature="n"))
    )


def test_the_download_refuses_an_empty_selection_and_keeps_all_distinct() -> None:
    """The page's download runtime, run under node against stub controls."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the download runtime cannot run.")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("notebook_download.test.cjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ── The two property views, checked against the page's own computation ─────

# The browser test that recomputes each view from the panel matrix.
VIEWS_TEST = Path(__file__).with_name("notebook_views.test.cjs")


def _number(value: Any) -> float | str | None:
    """A JSON-safe number: non-finite as the string JS reads back."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value = float(value)
    return value if math.isfinite(value) else json.dumps(value)


def _view_case(root: Path, data: TaskData, **chosen: Any) -> dict[str, Any]:
    """Run one property view's notebook and keep what it plots."""
    selection = _selection(MEASURED_BUDGET, **chosen)
    _, state = _run(root, selection)
    case: dict[str, Any] = {
        "view": selection.view,
        "feature": selection.feature,
        "scale": selection.scale,
        "aggregation": selection.aggregation,
        "baseline": selection.baseline,
        "tau": selection.tau,
    }
    if selection.view == FEATURE_VIEW:
        points, line = state["points"], state["line"]
        algorithms = [algorithm.uuid for algorithm in points["algorithm"]]
        case["points"] = [
            [uuid, instance.uuid, _number(feature), _number(ratio)]
            for uuid, instance, feature, ratio in zip(
                algorithms,
                points["instance"],
                points["feature"],
                points["ratio"],
            )
        ]
        owners = [algorithm.uuid for algorithm in line["algorithm"]]
        case["running"] = {
            uuid: {
                column: [
                    _number(value)
                    for owner, value in zip(owners, line[column])
                    if owner == uuid
                ]
                for column in ("position", "median", "lower", "upper")
            }
            for uuid in dict.fromkeys(owners)
        }
        case["algorithms"] = sorted(set(algorithms))
        # The opening ratio range's fences, over the ratios the axis places.
        ratios = points["ratio"].to_numpy(dtype=float)
        logarithmic = selection.y_scale == "log"
        placed = (
            (ratios > 0) & np.isfinite(ratios)
            if logarithmic
            else np.isfinite(ratios) & (ratios >= 0)
        )
        case["y_scale"] = selection.y_scale
        case["fence"] = state["RANGE_FENCE"]
        case["window"] = state["RUNNING_WINDOW"]
        case["positions"] = state["RUNNING_POSITIONS"]
        case["robust_range"] = list(
            compute_robust_range(
                (np.log10(ratios[placed]) if logarithmic else ratios[placed]).tolist(),
                points["weight"].to_numpy(dtype=float)[placed].tolist(),
                fence=state["RANGE_FENCE"],
            )
        )
        case["figure"] = isinstance(state["figure"], go.Figure)
        # The view the figure opens on, and the padding it was written with.
        case["x_range"] = list(state["figure"].layout.xaxis.range)
        case["y_range"] = list(state["figure"].layout.yaxis.range)
        case["padding"] = list(state["RANGE_PADDING"])
        # What the markers stand for, per algorithm: points, and trials.
        drawn: dict[str, list[int]] = {}
        for trace in state["figure"].data:
            if trace.mode != "markers" or trace.customdata is None:
                continue
            total = drawn.setdefault(str(trace.legendgroup), [0, 0])
            for points_there, trials in trace.customdata:
                total[0] += int(points_there)
                total[1] += int(trials)
        case["drawn"] = drawn
        return case
    bins = state["bins"]
    keys = [
        column
        for column in bins.columns
        if column not in ("start", "stop", "lower", "upper", "instances")
    ]
    starts = bins["start"].to_numpy(dtype=float)
    edges = [*starts, float(bins["stop"].iloc[-1])]
    case.update(
        algorithm=selection.algorithm,
        compare_with=selection.compare_with,
        bins=len(bins),
        keys=keys,
        shares={key: [float(value) for value in bins[key]] for key in keys},
        excluded=1 - float(bins[keys].to_numpy().sum()),
        members=[sorted(members) for members in bins["instances"]],
        # In axis coordinates, as the page keeps them: log10 on a log scale.
        edges=[
            math.log10(edge) if selection.scale == "log" else edge for edge in edges
        ],
        figure=isinstance(state["figure"], go.Figure),
    )
    return case


# The locale the parity run's page reads in, and the separators its
# Intl.NumberFormat writes there.
READER_LOCALE = "de_DE.UTF-8"
READER_SEPARATORS = {"decimal": ",", "group": "."}


def _locale_case(root: Path, *algorithms: str) -> dict[str, Any]:
    """The error ratio figure for a German reader, as the page writes its style.

    The notebook draws again with the separators the page writes into
    CHART_STYLE on download; each marker inside the view keeps its value,
    the trials it covers and its hover.
    """
    _, state = _run(
        root,
        _selection(
            MEASURED_BUDGET,
            view=FEATURE_VIEW,
            feature="n",
            scale="log",
            algorithms=list(algorithms),
        ),
    )
    # Texts with every field, since the fixture's copy is placeholders.
    labels = {
        **state["LABELS"],
        "feature": "n",
        "point_hover": "{property} {x} · {y}",
        "count_one": "{count} trial",
        "count": "{count} trials",
    }
    style = json.loads(json.dumps(state["CHART_STYLE"]))
    style["layout"]["meta"]["number_separators"] = READER_SEPARATORS
    figure = state["plot_error_ratios"](
        state["points"],
        state["line"],
        x_scale=state["SCALE"],
        y_scale=state["Y_SCALE"],
        fence=state["RANGE_FENCE"],
        padding=state["RANGE_PADDING"],
        template=style,
        **labels,
    )
    # A hover counts every algorithm's trials at its spot.
    spots: dict[tuple[float, float], list[Any]] = {}
    for trace in figure.data:
        if trace.mode != "markers" or not trace.customdata or trace.yaxis != "y":
            continue
        for x, y, data, text in zip(
            trace.x, trace.y, trace.customdata, trace.hovertext
        ):
            spot = spots.setdefault((float(x), float(y)), [0, text])
            spot[0] += int(data[1])
    markers = [[x, y, trials, text] for (x, y), (trials, text) in spots.items()]
    assert markers
    return {
        "separators": READER_SEPARATORS,
        "labels": labels,
        "markers": markers,
    }


def _algorithms(data: TaskData) -> list[str]:
    return [str(entry["uuid"]) for entry in data.index["algorithms"]]


def test_each_view_ships_its_own_notebook() -> None:
    """The curves keep their file; each property view adds its own."""
    assert NOTEBOOK_NAMES == {
        "profiles": "recreate.ipynb",
        FEATURE_VIEW: "recreate-feature.ipynb",
        OUTCOMES_VIEW: "recreate-outcomes.ipynb",
    }


def test_a_property_view_declares_its_property_and_scale() -> None:
    """The page rewrites the property, the scale and the view's algorithms."""
    for view, names in (
        (FEATURE_VIEW, {"ALGORITHMS", "FEATURE", "SCALE"}),
        (OUTCOMES_VIEW, {"ALGORITHM", "COMPARE_WITH", "FEATURE", "SCALE"}),
    ):
        document = notebook_document(
            _selection(MEASURED_BUDGET, view=view),
            ReleaseIdentity("", "", "/release", "fixture-release"),
        )
        declared = set(_parameters(document))
        assert names <= declared, view
        assert "PLOTTED_ALGORITHMS" not in declared


def test_an_outcomes_notebook_refuses_no_algorithm(
    measurable_release_root: Path,
) -> None:
    """The outcome view draws one chosen algorithm; none is no reading."""
    with pytest.raises(ValueError, match="ALGORITHM"):
        _run(
            measurable_release_root,
            _selection(MEASURED_BUDGET, view=OUTCOMES_VIEW, algorithm=None),
        )


def test_the_property_notebooks_plot_what_the_page_plots(
    measured_release: TaskData, measurable_release_root: Path, tmp_path: Path
) -> None:
    """Each property view's notebook reaches the points, running medians and
    outcome shares the page's own dpPanel computes from the same matrix."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the page's computation cannot run.")
    named = {
        str(entry["label"]): str(entry["uuid"])
        for entry in measured_release.index["algorithms"]
    }
    # The failing algorithm refuses the smallest instances, so its readings
    # carry failures and failure-dominated medians; its successes fall within
    # a threshold of ten and beyond one of a thousandth. Paired with a ranked
    # algorithm it shares every seed.
    failing, alpha = named[FAILING_ALGORITHM], named["Alg Alpha"]
    root = measurable_release_root
    cases = [
        _view_case(
            root,
            measured_release,
            view=FEATURE_VIEW,
            feature="n_over_d",
            scale="log",
            algorithms=[failing, alpha],
        ),
        _view_case(
            root,
            measured_release,
            view=FEATURE_VIEW,
            feature="n_over_d",
            scale="quantile",
            aggregation="median",
            y_scale="linear",
            algorithms=[failing, alpha],
        ),
        _view_case(
            root,
            measured_release,
            view=FEATURE_VIEW,
            feature="d",
            scale="linear",
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n_over_d",
            scale="linear",
            algorithm=failing,
            tau=10.0,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n_over_d",
            scale="quantile",
            aggregation="median",
            algorithm=failing,
            tau=0.001,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n",
            scale="log",
            algorithm=failing,
            compare_with=alpha,
            tau=10.0,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n",
            scale="log",
            aggregation="median",
            algorithm=failing,
            compare_with=alpha,
            tau=10.0,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n",
            scale="log",
            aggregation="median",
            algorithm=failing,
            tau=10.0,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="n_over_d",
            scale="quantile",
            algorithm=failing,
            compare_with=alpha,
            tau=10.0,
        ),
        _view_case(
            root,
            measured_release,
            view=OUTCOMES_VIEW,
            feature="d",
            scale="linear",
            aggregation="median",
            algorithm=alpha,
            compare_with=failing,
            tau=0.001,
        ),
    ]
    # The failing algorithm's ratio is the same on every seed of an instance,
    # so its trials collapse to fewer points than trials.
    points_drawn, trials_drawn = cases[0]["drawn"][failing]
    assert points_drawn < trials_drawn
    # The cases reach every kind of outcome the views split by.
    reached = {
        key
        for case in cases
        if case["view"] == OUTCOMES_VIEW
        for key, shares in case["shares"].items()
        if any(shares)
    }
    assert {"success_within", "success_beyond", "both"} <= reached
    assert reached & {"invalid_regime", "no_release", "algorithm_error"}
    position = next(
        index
        for index, entry in enumerate(measured_release.index["panels"])
        if entry["metric"] == MEASURED_METRIC and entry["budget"] == MEASURED_BUDGET
    )
    entry = measured_release.index["panels"][position]
    expectations = tmp_path / "notebook-views.json"
    expectations.write_text(
        json.dumps(
            {
                "index": measured_release.index,
                "position": position,
                "runs": base64.b64encode(measured_release.files[entry["runs"]]).decode(
                    "ascii"
                ),
                "cases": cases,
                "locale": _locale_case(root, failing, alpha),
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, "--test", "--test-reporter=tap", str(VIEWS_TEST)],
        # A German reader: the page's own number format writes in this locale.
        env={
            **os.environ,
            "DP_NOTEBOOK_VIEWS": str(expectations),
            "LANG": READER_LOCALE,
            "LC_ALL": READER_LOCALE,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert re.search(r"^# skip(ped)? 0$", completed.stdout, re.M)


def test_a_mapping_parameter_is_written_as_the_page_writes_it() -> None:
    """notebook_download.test.cjs writes this same mapping with the page's writer."""
    style = {
        "layout": {"showlegend": False, "meta": {"series": {}, "glyph": "∞", "x": None}}
    }
    assert parameter_literal(style) == (
        '{"layout": {"showlegend": False, "meta": {"series": {}, "glyph": "∞", "x": None}}}'
    )
