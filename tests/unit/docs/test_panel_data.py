"""Check the emitted matrix against the library, and the browser against both.

The site computes no analysis of its own: the Python side measures the
release's distribution lane through the library, divides by the denominators
the release ships, and serializes one binary matrix per panel; the browser
derives what a control chooses. Those derivations live in
``docs/site/docs/javascripts/panel-data.js``, and ``panel_data.test.cjs`` runs
them over the emitted matrix against the library's own ``aggregate_trials``
and ``profile_curves`` readings of the same store, to the double precision the
matrix ships in.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import math
from dataclasses import replace
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dpstats.analysis import (
    aggregate_trials,
    compute_profile_curves,
    measure_results,
    compute_error_ratio,
)
from dpstats.tasks import MeanEstimation, list_built_in_metrics
from dpstats.transport.arrays import write_named_arrays
from dpstats.workspace import open_collection
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    input_artifact_path,
)
from panel_data import (
    BEST_BASELINE,
    DESIGN_CONDITION,
    NO_BASELINE,
    TaskData,
    disambiguated_labels,
    listed_metrics,
)
from results_roots import ReleasedBaseline, ReleasedComparison
from site_copy import SITE_COPY_KEYS, copy_text
from tests.unit.docs.conftest import (
    COMPARISON,
    EXTERNAL_DESCRIPTION,
    EXTERNAL_REFERENCE,
    FAILING_ALGORITHM,
    FIXTURE_ALGORITHMS,
    FIXTURE_INSTANCES,
    MAHALANOBIS_METRIC,
    MAHALANOBIS_REFERENCE,
    MEASURED_METRIC,
    POOLED,
    REFERENCE_ERROR,
    RELEASED_METRICS,
    SECOND_METRIC,
    STORED_REFERENCE,
    UNDECLARED_REFERENCE,
    VIEW_BASELINE,
    measured,
    write_fixture_weights,
)
from release_files import sweep_relative, write_weights_view


# ── What the release decides, and the site does not ──────────────────────────


def test_the_shares_a_page_reads_are_the_ones_the_release_ships(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    """The weights reach the page from the view, never from a site rule."""
    even = measured(measurable_release_root)
    assert even.index["weights"] == [0.25, 0.25, 0.25, 0.25]

    uneven = tmp_path / "uneven"
    shutil.copytree(measurable_release_root, uneven)
    instances = [entry["uuid"] for entry in even.index["instances"]]
    write_weights_view(
        uneven,
        MeanEstimation.id,
        {
            instance: 0.7 if position == 0 else 0.1
            for position, instance in enumerate(instances)
        },
    )

    weighted = measured(uneven)
    assert weighted.index["weights"] == [0.7, 0.1, 0.1, 0.1]
    assert [panel.headline for panel in weighted.panels] != [
        panel.headline for panel in even.panels
    ]


def test_each_metric_opens_on_the_protocols_opening_rule(
    measurable_release_root: Path,
) -> None:
    """The headline Metric opens on its headline baseline; another Metric on
    its statistical baseline, never on the pooled or an external baseline."""
    data = measured(measurable_release_root)
    assert {
        metric: entry["default"] for metric, entry in data.index["baselines"].items()
    } == {
        MEASURED_METRIC: VIEW_BASELINE,
        SECOND_METRIC: STORED_REFERENCE,
        MAHALANOBIS_METRIC: MAHALANOBIS_REFERENCE,
    }


def test_a_metric_with_no_stored_value_opens_on_its_own_units(
    measurable_release_root: Path,
) -> None:
    """With no reference or external baseline carried, the pooled baseline is
    offered but the Metric opens on none."""
    data = measured(
        measurable_release_root,
        comparison=replace(COMPARISON, external={}),
    )
    assert COMPARISON.opening_for(SECOND_METRIC, None, ()) == NO_BASELINE
    assert data.index["baselines"][MEASURED_METRIC]["default"] == BEST_BASELINE


def test_the_panels_are_every_metric_the_instances_can_be_measured_on(
    measurable_release_root: Path,
) -> None:
    data = measured(measurable_release_root)
    assert sorted({panel.metric for panel in data.panels}) == sorted(RELEASED_METRICS)

    installed = open_collection(
        measurable_release_root, MeanEstimation
    ).load_instances()
    without_truth = [
        instance.model_copy(
            update={
                "metric_arguments": {
                    key: value
                    for key, value in instance.metric_arguments.items()
                    if key != "ground_truth"
                }
            }
        )
        for instance in installed.values()
    ]
    assert listed_metrics(MeanEstimation.id, without_truth, inputs=True) == (
        SECOND_METRIC,
        MAHALANOBIS_METRIC,
    )


def test_a_headline_metric_the_instances_cannot_measure_is_refused(
    measurable_release_root: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot measure its headline Metric"):
        measured(
            measurable_release_root,
            comparison=replace(COMPARISON, headline_metric="invented_metric"),
        )


def test_an_instance_the_release_does_not_weigh_is_refused(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "short-weights"
    shutil.copytree(measurable_release_root, root)
    write_fixture_weights(root, MeanEstimation.id, ["i5-not-in-this-release"])
    with pytest.raises(ValueError, match="do not weigh"):
        measured(root)


def test_a_published_release_missing_a_declared_run_is_refused(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    from dpstats.algorithm import register_algorithm
    from tests.unit.docs.conftest import StepMean

    root = tmp_path / "incomplete"
    shutil.copytree(measurable_release_root, root)
    register_algorithm(
        root,
        StepMean(label="Uncollected", error=1.0),
        registry="benchmark",
    )
    with pytest.raises(ValueError, match="is missing"):
        measured(root, published=True)


def test_a_local_cut_leaves_out_an_algorithm_it_never_ran(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    """An algorithm lacking a declared trial is left out whole, never shown on part.

    The build names it with the trials it lacks; the coverage the preview
    discloses still counts them.
    """
    from dpstats.algorithm import register_algorithm
    from tests.unit.docs.conftest import StepMean

    root = tmp_path / "preview"
    shutil.copytree(measurable_release_root, root)
    uncollected = StepMean(label="Uncollected", error=1.0)
    register_algorithm(root, uncollected, registry="benchmark")
    data = measured(root, published=False)

    coverage = data.index["preview_coverage"]
    assert coverage["expected"] > coverage["missing"] > 0
    label, lacking, declared = data.incomplete[uncollected.uuid]
    assert (label, lacking) == ("Uncollected", declared)
    assert uncollected.uuid not in {row["uuid"] for row in data.index["algorithms"]}
    assert uncollected.uuid not in data.index["outcomes"]


def test_outcomes_count_every_declared_run_once_by_algorithm_and_budget(
    measurable_release_root: Path, tmp_path: Path
) -> None:
    """Declared counts every cell; recorded and each status count only attempted ones.

    An algorithm never run is left out, so it has no counts. A failing
    algorithm's invalid-regime runs and its successes both count, and every
    recorded run is exactly one status.
    """
    from dpstats.algorithm import register_algorithm
    from tests.unit.docs.conftest import FAILING_ALGORITHM, StepMean

    root = tmp_path / "preview"
    shutil.copytree(measurable_release_root, root)
    uncollected = StepMean(label="Uncollected", error=1.0)
    register_algorithm(root, uncollected, registry="benchmark")
    data = measured(root, published=False)
    outcomes = data.index["outcomes"]

    assert uncollected.uuid not in outcomes

    failing_uuid = next(
        row["uuid"]
        for row in data.index["algorithms"]
        if row["label"] == FAILING_ALGORITHM
    )
    failing = outcomes[failing_uuid]
    assert sum(entry["invalid_regime"] for entry in failing.values()) > 0
    assert sum(entry["success"] for entry in failing.values()) > 0
    for entry in failing.values():
        assert entry["recorded"] == entry["declared"]
        assert entry["recorded"] == sum(
            entry[status]
            for status in (
                "success",
                "no_release",
                "invalid_regime",
                "algorithm_error",
                "timeout",
                "memory_limit_exceeded",
            )
        )


def test_panel_preserves_large_finite_metric_costs(
    tmp_path: Path, measurable_release_root: Path
) -> None:
    from dpstats.results import result_files
    from dpstats.sweep.coordinates import load_sweep_coordinates
    from dpstats.workspace import init_workspace, open_collection
    from dpstats.workspace.layout import (
        result_partition_directory,
        registry_directory_name,
        sweeps_directory,
        task_collection_directory,
        workspace_local_root,
    )
    from tests.unit.docs.conftest import StepMean

    root = tmp_path / "large-costs"
    shutil.copytree(measurable_release_root, root)
    added = StepMean(label="Large finite cost", error=1e40)
    # A release is read-only, so the added algorithm runs in its own store and
    # only its own resolved partition is cut into the release beside the rest.
    workspace = init_workspace(tmp_path / "workspace")
    store = workspace_local_root(workspace)
    source_task = task_collection_directory(root, MeanEstimation.id)
    store_task = task_collection_directory(store, MeanEstimation.id)
    for kind in ("algorithm", "instance"):
        directory = registry_directory_name(kind)
        shutil.copytree(source_task / directory, store_task / directory)
    shutil.copytree(sweeps_directory(source_task), sweeps_directory(store_task))
    shutil.copytree(root / INPUT_ARTIFACTS_DIRECTORY, store / INPUT_ARTIFACTS_DIRECTORY)
    collection = open_collection(workspace, MeanEstimation)
    collection.add_algorithm(added, registry="benchmark")
    shutil.copy2(
        store_task / registry_directory_name("algorithm") / "benchmark.json",
        source_task / registry_directory_name("algorithm") / "benchmark.json",
    )
    sweep = store / sweep_relative(MeanEstimation.id)
    collection.run_sweep(sweep, algorithms=[added])
    files, _ = result_files(
        task_collection_directory(store, MeanEstimation.id),
        MeanEstimation,
        load_sweep_coordinates(sweep, store, MeanEstimation),
    )
    installed = (
        result_partition_directory(
            task_collection_directory(root, MeanEstimation.id), added.uuid
        )
        / "results.parquet"
    )
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(files[added.uuid])
    data = measured(root)
    assert data.panels and not data.refused
    values, _ = _matrix(data, 0)
    assert any(value == 1e40 for value in values)


def test_a_release_reads_its_declared_sweep_not_directory_discovery(
    tmp_path: Path, measurable_release_root: Path
) -> None:
    root = tmp_path / "release"
    shutil.copytree(measurable_release_root, root)
    before = measured(root, published=True)
    extra = root / "mean_estimation/sweeps/not-published.json"
    extra.write_text("not a Sweep the release declares")
    after = measured(root, published=True)
    assert after.index == before.index
    assert after.files == before.files


# A short grid: the matrix does not depend on it, and a curve is a curve at four
# thresholds.
TAUS = (0.5, 1.0, 2.0, 10.0)

# The browser and library retain the same double-precision metric values.
PRECISION = 1e-12

# The browser loader's own test, which reads the expectations dumped below.
BROWSER_TEST = Path(__file__).with_name("panel_data.test.cjs")


def test_an_infinite_denominator_survives_browser_json(
    measurable_release_root: Path,
) -> None:
    """A baseline may be infinite on an Instance; JSON has no infinity.

    Read against the algorithm that returns no estimate outside its regime,
    a denominator is infinite there and finite elsewhere.
    """
    data = measured(measurable_release_root)
    failing = next(
        entry["uuid"]
        for entry in data.index["algorithms"]
        if entry["label"] == FAILING_ALGORITHM
    )
    against = ReleasedBaseline(
        "failing", "algorithm", (("uuid", str(failing)), ("aggregation", "median"))
    )
    comparison = replace(
        COMPARISON, headline_baseline="failing", headline_definition=against
    )
    decoded = json.loads(
        json.dumps(
            measured(measurable_release_root, comparison=comparison).index,
            allow_nan=False,
        )
    )
    vector = decoded["panels"][0]["baselines"]["failing"]
    assert any(float(value) == math.inf for value in vector)
    assert any(math.isfinite(float(value)) for value in vector)


def test_preparation_measures_the_whole_roster_once(
    measurable_release_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One measurement over the task's Results loads each Instance's Input once;
    measured algorithm by algorithm, every Input was loaded again per algorithm."""
    import panel_data

    measure = panel_data.measure_results
    cohorts = []

    def record_cohort(results: Any, metrics: Any, **kwargs: Any) -> Any:
        cohorts.append({algorithm.uuid for algorithm in results["algorithm"]})
        return measure(results, metrics, **kwargs)

    monkeypatch.setattr(panel_data, "measure_results", record_cohort)
    data = measured(measurable_release_root)
    assert len(cohorts) == 1
    assert {entry["uuid"] for entry in data.index["algorithms"]} <= cohorts[0]


def _panel_index(datasets: TaskData, metric: str) -> int:
    return next(
        index
        for index, entry in enumerate(datasets.index["panels"])
        if entry["metric"] == metric
    )


def _matrix(datasets: TaskData, position: int) -> tuple[list[float], dict[str, Any]]:
    """Decode one panel's binary matrix: little-endian doubles under gzip."""
    entry = datasets.index["panels"][position]
    payload = gzip.decompress(datasets.files[entry["runs"]])
    values = list(struct.unpack(f"<{len(payload) // 8}d", payload))
    return values, entry


# ── The library, run independently on the same store ─────────────────────────


def _measured(root: Path, metric: str, budget_label: str):
    collection = open_collection(root, MeanEstimation)
    measured = measure_results(
        collection.load_results(),
        list_built_in_metrics(MeanEstimation)[metric],
        collection=collection,
    )
    # The fixture Sweep declares its budgets directly, so one resolved
    # budget is one panel and its repr is the panel's own label.
    rows = measured[measured["budget"].map(repr) == budget_label]
    assert not rows.empty
    return rows


def _library_curves(root: Path, metric: str, budget: str, *, aggregation: str):
    rows = _measured(root, metric, budget)
    cells = (
        rows if aggregation == "individual" else aggregate_trials(rows, metric=metric)
    )
    ones = cells[["task", "instance"]].drop_duplicates().assign(baseline=1.0)
    normalized = compute_error_ratio(cells, ones, metric=metric)
    curves = compute_profile_curves(normalized, list(TAUS), metric=metric)
    traced: dict[str, list[float]] = {}
    for algorithm, fraction in zip(
        curves.sort_values("tau")["algorithm"], curves.sort_values("tau")["fraction"]
    ):
        traced.setdefault(algorithm.uuid, []).append(float(fraction))
    return traced


def _json_number(value: float) -> float | str:
    """A double JSON can carry: non-finite values as the string JS reads back."""
    return value if math.isfinite(value) else json.dumps(value)


def _library_expectations(datasets: TaskData, root: Path) -> dict[str, Any]:
    """What the library reads from the headline panel's run matrix.

    The curves and seed medians come from the library's own analysis over the
    store; the opening standings are the ones the index ships. The browser
    test reproduces each from the matrix alone.
    """
    position = _panel_index(datasets, MEASURED_METRIC)
    entry = datasets.index["panels"][position]
    metric, budget = entry["metric"], entry["budget"]
    rows = {
        algorithm["uuid"]: row
        for row, algorithm in enumerate(datasets.index["algorithms"])
    }
    columns = {
        instance["uuid"]: column
        for column, instance in enumerate(datasets.index["instances"])
    }
    curves: dict[str, list[list[float]]] = {}
    for aggregation in ("individual", "median"):
        traced = _library_curves(root, metric, budget, aggregation=aggregation)
        assert traced
        # An algorithm the panel carries no cost for draws no curve at all.
        curves[aggregation] = [traced.get(uuid, [0.0] * len(TAUS)) for uuid in rows]
    aggregated = aggregate_trials(_measured(root, metric, budget), metric=metric)
    medians = [
        [rows[algorithm.uuid], columns[instance.uuid], _json_number(float(value))]
        for algorithm, instance, value in zip(
            aggregated["algorithm"], aggregated["instance"], aggregated[metric]
        )
    ]
    assert medians
    # Each standing is read at the protocol's quantile, against the baseline
    # the protocol names for this metric.
    standings = datasets.index["headline"]["all"][f"{metric}/{budget}"]
    assert standings
    return {
        "index": datasets.index,
        "position": position,
        "runs": base64.b64encode(datasets.files[entry["runs"]]).decode("ascii"),
        "taus": list(TAUS),
        "precision": PRECISION,
        "curves": curves,
        "medians": medians,
        "headline": {
            "quantile": datasets.index["headline_quantile"],
            "baseline": datasets.index["baselines"][metric]["default"],
            "standings": {
                str(rows[uuid]): standing for uuid, standing in standings.items()
            },
        },
    }


def test_the_browser_reads_the_matrix_as_the_library_does(
    measured_release: TaskData, measurable_release_root: Path, tmp_path: Path
) -> None:
    """The real ``dpPanel`` reproduces the library's curves, medians and standings.

    Every view a page draws is a derivation of the emitted matrix, so the
    browser's derivations are checked against the library's readings of the
    same store rather than against a port of the browser's own arithmetic.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the browser loader cannot run.")
    expectations = tmp_path / "panel-expectations.json"
    expectations.write_text(
        json.dumps(
            _library_expectations(measured_release, measurable_release_root),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, "--test", "--test-reporter=tap", str(BROWSER_TEST)],
        env={**os.environ, "DP_PANEL_EXPECTATIONS": str(expectations)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert re.search(r"^# skip(ped)? 0$", completed.stdout, re.M)


def test_the_index_carries_what_the_release_stated(measured_release: TaskData) -> None:
    """A page decides no comparison: the headline reading, the one shipped
    baseline, the Instances' shares and the ranked class all travel with the
    data they were measured with."""
    index = measured_release.index
    assert index["headline_metric"] == MEASURED_METRIC
    assert index["headline_quantile"] == 0.9
    assert index["visible_classes"] == ["standard"]
    assert index["baselines"][MEASURED_METRIC]["default"] == VIEW_BASELINE
    assert index["pool"] == "standard"
    assert index["metric_labels"][MEASURED_METRIC] == "L2 error vs ground truth"
    assert index["weights"] == [0.25] * len(index["instances"])


def test_the_matrix_reserves_one_slot_per_algorithm_instance_and_seed(
    measured_release: TaskData,
) -> None:
    for position, entry in enumerate(measured_release.index["panels"]):
        values, _ = _matrix(measured_release, position)
        assert len(values) == (
            len(measured_release.index["algorithms"])
            * len(measured_release.index["instances"])
            * int(entry["seeds"])
        )
        assert entry["seeds"] == len(entry["seed_ids"])


def test_the_matrix_holds_exactly_the_runs_that_were_measured(
    measured_release: TaskData,
) -> None:
    """Every recorded run fills one slot; a failed one stays a failure.

    NaN pads a slot no run fills, and a non-success run stores a negative status
    code, so it is never a plain finite cost and never absent. Coverage counts
    successes alone, so it undercounts every present, non-negative slot.
    """
    values, entry = _matrix(measured_release, 0)
    present = [value for value in values if not math.isnan(value)]
    successes = [value for value in present if value >= 0]
    coverage = measured_release.index["coverage"]["all"][
        f"{entry['metric']}/{entry['budget']}"
    ]
    assert len(successes) == sum(counts[1] for counts in coverage.values())
    assert any(value < 0 for value in present)


def test_a_known_invalid_regime_codes_negative_in_the_runs_matrix(
    measured_release: TaskData,
) -> None:
    """FailingMean's own regime refusal on its smallest Instance stores the
    invalid_regime status code directly in the runs matrix, never a plain
    +inf cost and never a code any other status or success also uses."""
    values, entry = _matrix(measured_release, 0)
    table = measured_release.index["status_codes"]
    assert table["-2"] == "invalid_regime"
    assert len(set(table.values())) == len(table)
    row = next(
        index
        for index, algorithm in enumerate(measured_release.index["algorithms"])
        if algorithm["label"] == FAILING_ALGORITHM
    )
    column = next(
        index
        for index, instance in enumerate(measured_release.index["instances"])
        if instance["label"] == "narrow-small"
    )
    seeds = int(entry["seeds"])
    width = len(measured_release.index["instances"])
    start = (row * width + column) * seeds
    cell_values = values[start : start + seeds]
    assert cell_values and all(value == -2.0 for value in cell_values)


# ── The baselines a reader may divide by ─────────────────────────────────────


def test_statistical_and_external_baselines_are_offered_per_metric(
    measurable_release_root: Path,
) -> None:
    """Computed and source-supplied denominators keep Instance index order.

    Two Metrics share one statistical baseline; the protocol offers its
    external baseline for the headline Metric alone. The opening baseline
    leads, then none, the statistical baseline, the external baselines and
    the pooled one.
    """
    data = measured(measurable_release_root)
    labels = [str(entry["label"]) for entry in data.index["instances"]]
    reference = []
    mahalanobis = []
    for label in labels:
        samples, dim, _ = FIXTURE_INSTANCES[label]
        x = np.vander(np.linspace(0.2, 1.0, samples), dim, increasing=True)
        centered = x - x.sum(axis=0) / samples
        reference.append(
            math.sqrt(float((centered**2).sum()) / (samples * (samples - 1)))
        )
        # Centering removes the constant column; Vandermonde's other columns
        # span min(n - 1, d - 1) directions on these distinct nodes.
        mahalanobis.append(math.sqrt(min(samples - 1, dim - 1) / samples))
    external = [
        float(20 * FIXTURE_INSTANCES[label][0]) + FIXTURE_INSTANCES[label][1]
        for label in labels
    ]
    for panel in data.panels:
        if panel.metric == MAHALANOBIS_METRIC:
            assert panel.baselines[MAHALANOBIS_REFERENCE] == pytest.approx(mahalanobis)
        else:
            assert panel.baselines[STORED_REFERENCE] == pytest.approx(reference)
        if panel.metric == MEASURED_METRIC:
            assert panel.baselines[EXTERNAL_REFERENCE] == pytest.approx(external)
        else:
            assert EXTERNAL_REFERENCE not in panel.baselines
    keys = {
        metric: [option["key"] for option in entry["options"]]
        for metric, entry in data.index["baselines"].items()
    }
    assert keys == {
        MEASURED_METRIC: [
            BEST_BASELINE,
            NO_BASELINE,
            STORED_REFERENCE,
            EXTERNAL_REFERENCE,
        ],
        SECOND_METRIC: [STORED_REFERENCE, NO_BASELINE, BEST_BASELINE],
        MAHALANOBIS_METRIC: [MAHALANOBIS_REFERENCE, NO_BASELINE, BEST_BASELINE],
    }
    described = {
        option["key"]: option["description"]
        for option in data.index["baselines"][MEASURED_METRIC]["options"]
    }
    assert "standard error" in described[STORED_REFERENCE].lower()
    # A description is the inner HTML of its cell.
    assert described[EXTERNAL_REFERENCE] == html.escape(EXTERNAL_DESCRIPTION)


def test_a_scalar_reference_the_protocol_declares_no_baseline_for_is_absent(
    measurable_release_root: Path,
) -> None:
    """A stored number is offered only as a statistical or external baseline.

    The Instances carry this value exactly as they carry the declared ones;
    what keeps it out of the picker is that no Metric names it as its
    baseline and no protocol entry offers it, not its type or absence.
    """
    data = measured(measurable_release_root)
    offered = {
        option["key"]
        for entry in data.index["baselines"].values()
        for option in entry["options"]
    }
    assert UNDECLARED_REFERENCE not in offered
    assert "ground_truth" not in offered
    for panel in data.panels:
        assert UNDECLARED_REFERENCE not in panel.baselines


def test_reading_in_the_metrics_own_units_is_offered_and_ships_no_vector(
    measurable_release_root: Path,
) -> None:
    """Dividing by one everywhere needs no denominator to travel at all."""
    data = measured(measurable_release_root)
    for entry in data.index["baselines"].values():
        # The opening baseline leads; the Metric's own units follow it.
        assert entry["options"][0]["key"] == entry["default"]
        assert entry["options"][1]["key"] == NO_BASELINE
        assert entry["options"][1]["label"] == copy_text("baselines", NO_BASELINE)
    for panel in data.panels:
        assert NO_BASELINE not in panel.baselines


def test_the_best_median_is_the_eligible_pools_own(
    measurable_release_root: Path,
) -> None:
    """The cheapest standard algorithm's median cost on each Instance.

    A ranked fixture run costs ``error * 10 ** seed * (n / 2) * d / eps``, so
    the median of its two seeds is 5.5 times its cost at seed zero. The pool
    also holds the stand-in that refuses the smallest Instances: it costs the
    reference error wherever its regime holds, and an infinite cost where it
    does not, which is a cost no minimum ever picks.
    """
    data = measured(measurable_release_root)
    best_error = min(FIXTURE_ALGORITHMS.values())
    pool = {
        entry["uuid"]
        for entry in data.index["algorithms"]
        if entry["class"] == "standard"
    }
    for panel in data.panels:
        assert panel.epsilon is not None
        costs = _measured(measurable_release_root, panel.metric, panel.budget)
        costs = costs[costs["algorithm"].map(lambda algorithm: algorithm.uuid in pool)]
        expected: list[float] = []
        for entry, epsilon in zip(data.index["instances"], panel.epsilon):
            by_algorithm = costs[
                costs["instance"].map(lambda instance: instance.uuid == entry["uuid"])
            ].groupby("algorithm", sort=False)
            minimum = min(
                float(np.median(group[panel.metric].to_numpy()))
                for _, group in by_algorithm
            )
            samples, dim, _ = FIXTURE_INSTANCES[str(entry["label"])]
            ranked = 5.5 * best_error * (samples / 2) * dim / epsilon
            # The refusing stand-in is outside its regime at the base size.
            refusing = math.inf if samples == 2 else REFERENCE_ERROR
            if panel.metric == MEASURED_METRIC:
                assert minimum == pytest.approx(min(ranked, refusing))
            expected.append(minimum)
        assert panel.baselines[BEST_BASELINE] == pytest.approx(expected)


def test_the_pooled_baseline_ranks_the_protocols_eligibility_class(
    measurable_release_root: Path,
) -> None:
    """A pool of another class ranks that class, never the standard one."""
    qualified = ReleasedBaseline(
        POOLED.name, "best_median", (("pool", "qualified"), ("aggregation", "median"))
    )
    standard = measured(measurable_release_root)
    other = measured(
        measurable_release_root,
        comparison=replace(
            COMPARISON,
            definitions={BEST_BASELINE: qualified},
            headline_definition=qualified,
            eligibility="qualified",
        ),
    )
    assert other.index["visible_classes"] == ["qualified"]
    assert other.index["pool"] == "qualified"
    assert [panel.baselines[BEST_BASELINE] for panel in other.panels] != [
        panel.baselines[BEST_BASELINE] for panel in standard.panels
    ]
    shown = {
        entry["class"] for entry in other.index["algorithms"] if not entry["diagnostic"]
    }
    assert shown == {"qualified"}


def test_only_the_headline_baseline_on_the_headline_metric_is_marked(
    measurable_release_root: Path,
) -> None:
    data = measured(measurable_release_root)
    marked = {
        metric: [option["key"] for option in entry["options"] if option["protocol"]]
        for metric, entry in data.index["baselines"].items()
    }
    assert marked == {
        MEASURED_METRIC: [VIEW_BASELINE],
        SECOND_METRIC: [],
        MAHALANOBIS_METRIC: [],
    }
    for entry in data.index["baselines"].values():
        # The marker is a fact beside the name, never folded into it.
        assert all("(protocol)" not in option["label"] for option in entry["options"])


# ── The design feature a reader filters Instances by ─────────────────────────


def test_each_instance_carries_the_condition_number_of_its_own_design(
    measurable_release_root: Path,
) -> None:
    """Prepare reads each Instance's Input once and carries kappa_2 per Instance.

    The oracle is the shape of the fixture designs rather than the formula that
    measured them: an n-by-1 design has exactly one singular value, so its
    condition number is one; the wider designs span several directions and are
    worse conditioned the taller they get.
    """
    data = measured(measurable_release_root)
    kappa = {
        str(entry["label"]): entry[DESIGN_CONDITION]
        for entry in data.index["instances"]
    }

    assert set(kappa) == set(FIXTURE_INSTANCES)
    assert not [label for label, value in kappa.items() if value is None]
    single_column = [
        label for label, (_, dim, _) in FIXTURE_INSTANCES.items() if dim == 1
    ]
    assert [kappa[label] for label in single_column] == pytest.approx([1.0, 1.0])
    assert kappa["wide-large"] > kappa["wide-small"] > 1.0
    assert math.isfinite(float(kappa["wide-large"]))


def test_the_condition_number_is_offered_as_an_instance_feature(
    measurable_release_root: Path,
) -> None:
    """The key a filter reads is listed with the slot naming it for a reader."""
    data = measured(measurable_release_root)

    assert data.index["features"][DESIGN_CONDITION] == {
        "slot": "site.fact_condition_number"
    }


def test_every_instance_feature_is_named_by_a_slot_or_a_symbol(
    measurable_release_root: Path,
) -> None:
    """A feature's name is read from the site copy when the page draws it."""
    features = measured(measurable_release_root).index["features"]

    for spec in features.values():
        if "slot" in spec:
            group, key = spec["slot"].split(".")
            assert key in SITE_COPY_KEYS[group]
        else:
            assert spec == {"symbol": "ε"}


def test_a_singular_design_travels_as_the_string_json_can_carry(
    measurable_release_root: Path,
    tmp_path: Path,
) -> None:
    """An unbounded condition number is a value, and the payload keeps it.

    The index is JSON, which has no infinities, so the build writes the numeric
    string the browser restores rather than dropping the Instance or rounding
    its value down to a finite one.
    """
    root = tmp_path / "singular"
    shutil.copytree(measurable_release_root, root)
    collection = open_collection(root, MeanEstimation)
    # Overwrite one stored Input with a design whose coordinates never vary:
    # it spans one direction, so its smaller singular value is exactly zero.
    instances = collection.load_instances()
    singular = next(
        instance for instance in instances.values() if instance.label == "wide-large"
    )
    samples, dim, _ = FIXTURE_INSTANCES["wide-large"]
    flattened = np.zeros((samples, dim))
    flattened[:, 0] = 1.0
    artifact = input_artifact_path(
        root / INPUT_ARTIFACTS_DIRECTORY, singular.input_digest
    )
    artifact.unlink()
    write_named_arrays({"X": flattened}, artifact, digest=singular.input_digest)

    data = measured(root)

    carried = {
        str(entry["label"]): entry[DESIGN_CONDITION]
        for entry in data.index["instances"]
    }
    assert carried["wide-large"] == json.dumps(math.inf)
    assert float(carried["wide-large"]) == math.inf


# ── Indicator Metrics, reported as rates ─────────────────────────────────────


def _above_reference(output: Any, *, sample_mean: Any) -> float:
    """1 where the estimate's first coordinate exceeds the reference error."""
    return float(output.mean_est[0] - sample_mean[0] > REFERENCE_ERROR)


def test_an_indicator_metric_is_left_off_explore(
    measurable_release_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Metric scoring only 0 and 1 ships no panel and nothing else."""
    import panel_data
    from dpstats.metric import Metric

    indicator = Metric(
        task=MeanEstimation,
        name="above_reference",
        calculation=_above_reference,
        requires=("sample_mean",),
        label="Above the reference",
    )
    defined = {**list_built_in_metrics(MeanEstimation), indicator.name: indicator}
    monkeypatch.setattr(panel_data, "list_built_in_metrics", lambda task: defined)
    data = measured(measurable_release_root)
    assert indicator.name not in {panel.metric for panel in data.panels}
    assert indicator.name not in data.index["metric_labels"]
    assert indicator.name not in data.index["baselines"]
    assert indicator.name not in json.dumps(data.index)


def test_a_name_shared_across_classes_stays_plain() -> None:
    labels = {"a3-AAAAAAAA11111111": "Mean only", "a3-BBBBBBBB22222222": "Mean only"}
    assert (
        disambiguated_labels(
            labels,
            {"a3-AAAAAAAA11111111": "standard", "a3-BBBBBBBB22222222": "diagnostic"},
        )
        == labels
    )
    assert disambiguated_labels(
        labels, {"a3-AAAAAAAA11111111": "standard", "a3-BBBBBBBB22222222": "standard"}
    ) == {
        "a3-AAAAAAAA11111111": "Mean only (11111111)",
        "a3-BBBBBBBB22222222": "Mean only (22222222)",
    }
