"""Emit one compact dataset per panel; the browser computes the views.

A results page is a view over one task's results, and a reader moves through
many of them: a threshold, a metric, a budget, an instance subset, a
baseline algorithm, individual trials or one median per instance. Precomputing
each of those states is exhaustive for the build and the disk, so this module
emits the data they are all computed from and nothing else.

Per task and results surface:

- ``index.json``: the algorithms, the instances, the panels, and the coverage
  counts, each in one fixed order the panel files index into.
- ``runs-<metric>-<budget>.f64.gz``: one gzip-compressed matrix of every trial's
  cost, a little-endian ``Float64Array`` in row-major ``[algorithm][instance][seed]``
  order. Every view — share within a threshold, a profile curve, one median per
  instance, a ratio to a chosen baseline, the pairwise scatter and its count
  strip, the feature comparison — is a derivation of this one matrix.

The two lanes reach a page differently. Over the distribution lane the release
ships its raw Results, so they are measured here through the library's generic
analysis, under the instance weights the release ships, for every Metric the
library defines whose arguments every shipped instance supplies or can compute. Over the evaluation
lane the release ships aggregated curves and nothing per-instance, so that view
is ingested as panels of the same shape carrying no trial matrix: their
Leaderboard ships ``curves-<metric>-<budget>.json``, the released fractions
verbatim, and every reading of them is the published number.

A panel carries one denominator vector per baseline a reader may divide by:
its Metric's statistical baseline, each external baseline the protocol
offers for it, the headline baseline on the headline Metric, and the best
median in the protocol's eligibility class, each taken once per Metric, budget
and instance through the library's own baseline functions. Reading in the
Metric's own units divides by one and carries no vector, and reading against a
chosen algorithm is derived from the trial matrix itself. The browser divides;
this module applies the protocol's opening rule and no weighting or threshold
of its own.

A Metric whose every estimate is 0 or 1 is an indicator, such as a sign
disagreement: a share of trials within a threshold of it says nothing, so it
ships no panel and no page lists it.

Costs are extended-real: an infinity is stored natively, because a failed trial
is a real cost no threshold covers, and ``NaN`` marks a seed a cell has no
Attempt for — throughout a cell no Attempt answered at all. A panel is one
**declared** budget rather than one resolved budget: a Sweep may state a
budget as a rule that reads an instance's sample size, so the declared entry is
what every instance's trials share.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
    load_algorithms,
)
from dpstats.analysis import (
    compute_baseline,
    compute_profile_curves,
    get_metric_availability,
    compute_error_ratio,
    summarize_outcomes,
)
from dpstats.budgets import ApproxDPBudget, PureDPBudget
from dpstats.results import ResultStatus
from dpstats.workspace import open_collection
from dpstats.analysis import measure_results
from dpstats.instance import Instance
from dpstats.tasks import BUILT_IN_TASKS, list_built_in_metrics
from dpstats.transport.documents import STAGING_PREFIX
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    PARTITION_PREFIX,
    results_directory,
    sweeps_directory,
    task_collection_directory,
)
from dpstats.workspace.layout import registry_directory_name
from dpstats.transport.registry import selected_ids
from dpstats.workspace.provenance import read_provenance
from dpstats.workspace.registries import read_selection
from dpstats.sweep.coordinates import load_sweep_coordinates
from dpstats.sweep._sweep import Sweep

from explore_properties import instance_properties
from interpretation import shown_by_default
from results_roots import (
    BEST_MEDIAN,
    NO_BASELINE,
    ReleasedBaseline,
    ReleasedComparison,
)
from protocol_page import baseline_definition
from profile_chart import (
    LEADING_CURVES,
    OPENING_TAU,
    algorithm_family,
    element_id,
    embed,
    family_slot,
)
from series_style import marker_shapes_by_uuid
from site_copy import SITE_COPY_KEYS, copy_text
from skipped_rows import reported_separately, skipped_reasons

# The baselines a page offers. Each is one denominator per instance, so
# choosing between them changes only the division: no key here selects a
# different pool, weighting, grid or eligibility.
#
# ``none`` divides by one, leaving costs in the Metric's own units, and so
# ships no vector at all. ``best_median`` divides by the best median in the
# protocol's eligibility class, which the library takes over that class's rows
# alone. A stored instance value is offered under its key where the library
# declares it the Metric's statistical baseline or the protocol offers it as
# an external baseline for the Metric; an instance stores other scalars too,
# which a reader's own notebook may divide by.
BEST_BASELINE = BEST_MEDIAN

# The two values an indicator Metric's estimates take.
INDICATOR_VALUES = frozenset({0.0, 1.0})

DESIGN_CONDITION = "design_condition_number"

# A non-success trial's status, stored directly in the trial matrix as a negative
# code beside real costs, which are always zero, positive, or +inf. The
# browser's one raw-cost accessor maps a negative value back to +Infinity, so
# every existing derivation over the matrix is unchanged; a second, small
# accessor reads the code itself for the histogram. This table, shipped once
# in the index as status_codes, is the only place the codes are named.
_STATUS_CODES: dict[ResultStatus, int] = {
    ResultStatus.NO_RELEASE: -1,
    ResultStatus.INVALID_REGIME: -2,
    ResultStatus.TIMEOUT: -3,
    ResultStatus.MEMORY_LIMIT_EXCEEDED: -4,
    ResultStatus.ALGORITHM_ERROR: -5,
}
_STATUS_CODE_BY_NAME = {status.value: code for status, code in _STATUS_CODES.items()}
# A declared cell no Attempt ever answered: not a recorded terminal outcome,
# so it is not in ResultStatus, but still needs its own code.
_NOT_RUN_CODE = -6

# How a reader sees each feature named: the site-copy slot holding its name, or
# the bare symbol it is written as. A Task declares which Instance properties
# it has; epsilon comes from the budget a panel declares.
FEATURE_NAMES: dict[str, dict[str, str]] = {
    "n": {"slot": "explore.feature_sample_size"},
    "d": {"slot": "site.card_dimension"},
    "n_over_d": {"slot": "explore.feature_samples_per_dimension"},
    DESIGN_CONDITION: {"slot": "site.fact_condition_number"},
    "vif": {"slot": "explore.feature_vif"},
    "epsilon": {"symbol": "ε"},
}

INDEX_NAME = "index.json"

# The columns a released profile view carries, in order, each with the one
# type it is stored as. A release states them; this build reads them and
# refuses a view that does not carry exactly these.
PROFILE_VIEW_SCHEMA = pa.schema(
    [
        ("task", pa.string()),
        ("algorithm", pa.string()),
        ("alg_uuid", pa.string()),
        ("budget", pa.string()),
        ("metric", pa.string()),
        ("baseline", pa.string()),
        ("slice", pa.string()),
        ("tau", pa.float64()),
        ("fraction", pa.float64()),
        ("run_count", pa.int64()),
        ("measured_run_count", pa.int64()),
        ("tau_at_headline", pa.float64()),
        ("share_reached", pa.float64()),
        ("tau_at_share_reached", pa.float64()),
        ("slice_instance_count", pa.int64()),
        ("slice_study_count", pa.int64()),
    ]
)
PROFILE_VIEW_COLUMNS = tuple(PROFILE_VIEW_SCHEMA.names)
# The headline reading's two columns that may be empty: null is an infinite τ.
_NULLABLE_VIEW_COLUMNS = ("tau_at_headline", "tau_at_share_reached")

# One released profile row: the seven names it is keyed by (its slice last),
# then its threshold, the fraction reached there, the run counts that fraction
# was taken over, its curve's headline reading (constant across the curve's
# rows) and its slice's Instance and study counts (constant across the slice).
ProfileRow = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    float,
    float,
    int,
    int,
    float,
    float,
    float,
    int,
    int,
]
# The slice every release ranks: all of the task's Instances.
ALL_SLICE = "all"


@dataclass(frozen=True)
class Headline:
    """One curve's standing under the protocol's headline quantile.

    Attributes:
        tau: The error ratio at which its weighted share first reaches the
            headline quantile; ``inf`` where it never does.
        share: The weighted share it reaches at all (its success rate at
            τ = ∞).
        tau_at_share: The smallest error ratio at which it reaches ``share``;
            ``inf`` where ``share`` is 0.
    """

    tau: float
    share: float
    tau_at_share: float

    def json(self) -> list[float | str]:
        """The reading as the browser index carries it, an infinity as text."""
        return [
            value if math.isfinite(value) else json.dumps(value)
            for value in (self.tau, self.share, self.tau_at_share)
        ]


def headline_key(reading: Headline) -> tuple[float, float, float]:
    """What the ranking compares, smaller first; equal keys share a rank."""
    return reading.tau, -reading.share, reading.tau_at_share


def headline_order(readings: Mapping[str, Headline]) -> list[str]:
    """The one ranking rule: algorithm identifiers, best first.

    A finite headline τ ranks above every infinite one, lower first; among the
    infinite ones a larger share reached ranks first, then the smaller τ at
    which that share is reached. An exact tie keeps the identifiers' order.
    """
    return sorted(readings, key=lambda uuid: (*headline_key(readings[uuid]), uuid))


@dataclass(frozen=True)
class RegisteredAlgorithm:
    """What a store's registry says about one algorithm a page draws.

    Attributes:
        estimator: The class defining the estimator it runs, which is what its
            curve's family colour is keyed on.
        privacy_interpretation: The privacy semantics its claim is read under.
        transformations: The transformation classes wrapped around that estimator, outer
            first.
        data_bounds: Whether its configured chain uses empirical DataBounds.
        fwl_demean: Whether its configured chain uses LargestOneHotFWLDemean.
    """

    estimator: str
    privacy_interpretation: str
    transformations: tuple[str, ...]
    data_bounds: bool
    fwl_demean: bool


@dataclass
class PanelData:
    """One metric and declared budget, as its dataset files carry it.

    Attributes:
        metric: The Metric the costs measure.
        budget: The declared budget, as the Sweep states it.
        payload: Gzip-compressed little-endian float64 costs, row-major over algorithms,
            instances and seeds, with ``NaN`` for an unobserved seed. A non-success
            trial stores its negative status code (see the index's ``status_codes``)
            in place of the extended-real cost it would otherwise carry.
        seeds: The algorithm seeds the matrix reserves a slot for, ascending.
        baselines: Each named baseline's per-instance denominator.
        epsilon: The budget's epsilon on each instance, or ``None`` where it
            supplies none.
        declared_runs: The trials the Sweep declares in this panel for one
            algorithm.
        measured: How many trials each algorithm returned an estimate on here
            (recorded with status ``success``).
        headline: Each algorithm's standing under the protocol's headline
            quantile, which ``headline_order`` ranks.
        slice: The official slice of Instances the panel reads.
    """

    metric: str
    budget: str
    payload: bytes
    seeds: list[int]
    baselines: dict[str, list[float]]
    epsilon: list[float] | None
    declared_runs: int
    measured: dict[str, int] = field(default_factory=dict)
    headline: dict[str, Headline] = field(default_factory=dict)
    slice: str = ALL_SLICE


@dataclass
class TaskData:
    """One results surface's complete dataset, ready to be written.

    Attributes:
        index: The ``index.json`` payload.
        files: The trial matrix of each panel, by file name.
        panels: The panels the index lists, in the same order.
        refused: One ``(metric, reason)`` pair per Metric whose costs could not
            be read over one denominator.
        omitted: How many algorithms a released view ranked that were left out
            of these panels because the build could not load them.
        incomplete: Each configured algorithm a local cut left out because it
            lacks a declared trial, keyed by uuid: its label, the trials it
            lacks and the trials declared for it.
    """

    index: dict[str, Any]
    files: dict[str, bytes]
    panels: list[PanelData]
    refused: list[tuple[str, str]]
    omitted: int = 0
    incomplete: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    instance_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)


# ── Reading what a store carries ─────────────────────────────────────────────


def _task_root(root: Path, task_id: str) -> Path:
    return task_collection_directory(root, task_id)


def _has_results(task_root: Path) -> bool:
    """Whether the release holds at least one sealed Result file.

    A release's Results are these sealed partition files; only their names are
    read here, through the layout's own directory and partition-prefix
    constants, so no partition's parquet content is opened. A staging name is a
    writer's in-flight file, not a sealed Result yet.
    """
    directory = results_directory(task_root)
    if not directory.is_dir():
        return False
    return any(
        entry.is_file() and not entry.name.startswith(STAGING_PREFIX)
        for partition in directory.glob(f"{PARTITION_PREFIX}*")
        if partition.is_dir()
        for entry in partition.iterdir()
    )


def missing_results_inputs(
    root: Path, task_id: str, *, sweep: Path | None = None, require_results: bool = True
) -> tuple[str, ...]:
    """Name what stands between *root* and measuring its own Results at all.

    A store may ship standings without shipping what they were computed from.
    That absence belongs on the pages that cannot be built rather than to the
    build, so it is reported rather than raised.

    Returns:
        One phrase per absence, each naming what is missing and where it would
        be, or empty where the Results can be measured.
    """
    task_root = _task_root(root, task_id)
    required = {
        "the Sweep declaring its trials": (
            sweep if sweep is not None else sweeps_directory(task_root)
        ),
        "the algorithm registry those trials name": (
            task_root / registry_directory_name("algorithm")
        ),
        "the instance registry those trials name": (
            task_root / registry_directory_name("instance")
        ),
    }
    missing = [
        f"{description} ({path})"
        for description, path in required.items()
        if not path.exists() or (path.is_dir() and not any(path.iterdir()))
    ]
    if require_results and not _has_results(task_root):
        missing.append(
            "the Results themselves "
            f"({results_directory(task_root).relative_to(root).as_posix()})"
        )
    return tuple(missing)


def _estimator_class(algorithm: BaseAlgorithm) -> str:
    """Name the class defining the estimator one configured algorithm runs.

    A configured algorithm may be a transformation chain around the estimator that
    does the work, and the mechanism family belongs to that estimator: the same
    mechanism reads as one family however it was wrapped, standardized, or
    relabelled.
    """
    inner = algorithm
    while isinstance(inner, CompositeAlgorithm):
        inner = inner.algorithm
    return type(inner).__name__


def _transformation_chain(algorithm: BaseAlgorithm) -> tuple[str, ...]:
    """Name the transformation classes wrapped around the estimator, outer first."""
    chain: list[str] = []
    inner = algorithm
    while isinstance(inner, CompositeAlgorithm):
        chain.extend(
            type(transformation).__name__ for transformation in inner.transformations
        )
        inner = inner.algorithm
    return tuple(chain)


def registered_algorithms(
    root: Path,
    task_id: str,
    *,
    algorithm_registries: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, RegisteredAlgorithm]:
    """Return what *root* registered for each algorithm, by identifier.

    A store that registers no algorithm for the task says nothing about any,
    and a page reading it draws every curve in the unmapped family rather than
    guessing a mechanism from a label.
    """
    task_root = _task_root(root, task_id)
    if not (task_root / registry_directory_name("algorithm")).is_dir():
        return {}
    with reported_separately():
        registered = load_algorithms(
            read_selection(task_root, "algorithm", algorithm_registries).entries,
            task_id=task_id,
            strict=strict,
        )
    facts: dict[str, RegisteredAlgorithm] = {}
    for uuid, value in registered.items():
        transformations = _transformation_chain(value)
        facts[uuid] = RegisteredAlgorithm(
            estimator=_estimator_class(value),
            privacy_interpretation=value.privacy_interpretation.value,
            transformations=transformations,
            data_bounds="DataBounds" in transformations,
            fwl_demean="LargestOneHotFWLDemean" in transformations,
        )
    return facts


# ── Measuring one store into panel datasets ──────────────────────────────────


def released_weights(root: Path, task_id: str) -> dict[str, float]:
    """Each instance's share of one task's comparison, as the release ships it.

    The library's own reader parses the weights view, so nothing is chosen,
    scaled or substituted here.
    """
    weights = open_collection(root, BUILT_IN_TASKS[task_id]).load_weights()
    return {
        str(uuid): float(weight)
        for uuid, weight in zip(weights["instance_uuid"], weights["weight"])
    }


def _resolved_budget_groups(
    settings: Mapping[str, Any], *, run_count: int
) -> pd.DataFrame:
    """Relate each selected instance's concrete budgets to the Sweep entries declaring them.

    Delegates to ``Sweep.budget_groups_for``, the sole per-instance
    resolution a Sweep defines. ``run_count`` is constant across a
    Sweep's declared seeds, unlike a store's own recorded ``budget_group``.
    """
    sweep = Sweep(task=settings["task"], budgets=settings["budgets"] or None)
    rows = [
        (instance, budget, declared, run_count)
        for instance in settings["instances"]
        for budget, declared in sweep.budget_groups_for(instance)
    ]
    return pd.DataFrame(
        rows, columns=["instance", "budget", "budget_group", "run_count"]
    )


def _panel_labels(
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], tuple[str, ...]]:
    """Label the library's resolved budget groups for this results surface.

    A panel is the budget the Sweep declares, whatever store supplied the
    trials: a published view carries its own labels and is read on its own.
    """
    selected = groups.copy()
    selected["panel"] = selected["budget_group"].map(repr)
    runs = selected.groupby("panel", sort=False)["run_count"].sum()
    return (
        selected[["instance", "budget", "panel"]],
        {str(name): int(count) for name, count in runs.items()},
        tuple(str(name) for name in runs.index),
    )


def _payload_number(value: float) -> float | str | None:
    """One property value as JSON carries it: ``None`` where it is undefined
    (NaN), and an infinity as the numeric string the browser restores."""
    if math.isnan(value):
        return None
    return value if math.isfinite(value) else json.dumps(value)


def _instance_payload(
    instance: Instance, ref: str | None, properties: pd.Series
) -> dict[str, Any]:
    """Describe one instance by the properties a reader selects on."""
    return {
        "uuid": instance.uuid,
        "label": str(instance),
        "ref": ref,
        **{str(name): _payload_number(value) for name, value in properties.items()},
    }


def _vector(
    denominators: Mapping[str, float], instances: Sequence[dict[str, Any]]
) -> list[float] | None:
    """One denominator per instance in index order, or ``None`` if incomplete."""
    if any(entry["uuid"] not in denominators for entry in instances):
        return None
    return [denominators[entry["uuid"]] for entry in instances]


def _reference_vector(
    cells: pd.DataFrame,
    payloads: Sequence[dict[str, Any]],
    reference: str,
) -> list[float] | None:
    """One resolved reference as a denominator per instance, or ``None``.

    Every drawn instance must carry the named reference as a number; one that
    does not leaves a denominator missing, and a reading over an incomplete
    denominator is not one this page draws.
    """
    values: dict[str, float] = {}
    for row in cells.drop_duplicates("instance").itertuples():
        instance = row.instance
        held = row.metric_arguments.get(reference)
        if type(held) is int or type(held) is float:
            values[instance.uuid] = float(held)
    return _vector(values, payloads)


def _computed_vector(
    cells: pd.DataFrame,
    metric: str,
    payloads: Sequence[dict[str, Any]],
    definition: ReleasedBaseline,
) -> list[float] | None:
    """One computed baseline on each instance, or ``None``.

    The library takes it: the best aggregated cost in the pool's class, among
    a named roster, or one algorithm's own. A pool with no algorithm in this
    panel offers no pooled baseline, and a panel whose baseline leaves an
    instance unmeasured offers none rather than a denominator taken over
    fewer algorithms.
    """
    detail = dict(definition.detail)
    baseline: PrivacyInterpretation | BaseAlgorithm | list[BaseAlgorithm]
    if definition.kind == "best_median":
        baseline = PrivacyInterpretation(detail["pool"])
        if not any(
            algorithm.privacy_interpretation is baseline
            for algorithm in cells["algorithm"]
        ):
            return None
    else:
        carried = {algorithm.uuid: algorithm for algorithm in cells["algorithm"]}
        uuids = (
            [uuid.strip() for uuid in detail["algorithms"].split(",")]
            if definition.kind == "best_of"
            else [detail["uuid"]]
        )
        absent = [uuid for uuid in uuids if uuid not in carried]
        if absent:
            raise ValueError(
                f"the {definition.kind} baseline {definition.name!r} names "
                f"algorithms this panel does not carry: {absent}"
            )
        named = [carried[uuid] for uuid in uuids]
        baseline = named if definition.kind == "best_of" else named[0]
    best = compute_baseline(
        cells, metric=metric, baseline=baseline, aggregation=detail["aggregation"]
    )
    return _vector(
        {
            instance.uuid: float(value)
            for instance, value in zip(best["instance"], best["baseline"])
        },
        payloads,
    )


def _offered_baselines(
    cells: pd.DataFrame,
    *,
    metric: str,
    comparison: ReleasedComparison,
    statistical: str | None,
    payloads: Sequence[dict[str, Any]],
) -> dict[str, list[float]]:
    """Every denominator vector one panel offers, by the name a reader picks.

    The Metric's statistical baseline is offered where every drawn instance
    carries it as a number and left out otherwise. An external baseline the
    protocol offers for this Metric must be available, because its check
    confirmed the store carries it on every instance. Reading against no baseline divides
    by one everywhere and so carries no vector at all.
    """
    offered: dict[str, list[float]] = {}
    for name in comparison.stored_for(metric, statistical):
        vector = _reference_vector(cells, payloads, name)
        if vector is not None:
            offered[name] = vector
        elif name != statistical or (
            metric == comparison.headline_metric
            and name == comparison.headline_baseline
        ):
            raise ValueError(
                f"the protocol offers the stored value {name!r}, which not "
                "every drawn instance carries as a number"
            )
    for name in comparison.computed_for(metric):
        definition = (
            comparison.definitions[name]
            if name in comparison.definitions
            else comparison.headline_definition
        )
        if definition.kind == NO_BASELINE:
            continue
        vector = _computed_vector(cells, metric, payloads, definition)
        if vector is not None:
            offered[name] = vector
        elif (
            name == comparison.headline_baseline
            and metric == comparison.headline_metric
        ):
            raise ValueError(
                f"the headline baseline {name!r} leaves a drawn instance without "
                "a denominator"
            )
    return offered


def listed_metrics(
    task_id: str, instances: Sequence[Instance], *, inputs: bool
) -> tuple[str, ...]:
    """The library Metrics every drawn instance can be measured on, in its order.

    A Metric is listed where every instance stores what it requires, which is
    what keeps one needing a simulated ground truth off a corpus release
    without any list naming it. One needing the raw Input is listed only
    where the release ships its Inputs.
    """
    defined = list_built_in_metrics(BUILT_IN_TASKS[task_id])
    if not defined:
        return ()
    available = get_metric_availability(list(instances), defined.values())
    usable = {"available", "available_with_input"} if inputs else {"available"}
    return tuple(name for name in defined if available[name].isin(usable).all())


def _is_indicator(cells: pd.DataFrame, metric: str) -> bool:
    """Whether every estimate one Metric scored is 0 or 1."""
    succeeded = cells[
        cells["recorded"].eq(True) & cells["status"].eq(ResultStatus.SUCCESS.value)
    ]
    values = {float(value) for value in succeeded[metric]}
    return bool(values) and values <= INDICATOR_VALUES


def protocol_conflicts(task_id: str, comparison: ReleasedComparison) -> list[str]:
    """Name what the protocol states that this build's library cannot show.

    The protocol names Metrics, the headline and those each external baseline
    is offered for, and a page links each to its definition, so a Metric the
    library does not define has nowhere to link.
    """
    defined = list_built_in_metrics(BUILT_IN_TASKS[task_id])
    named = [
        comparison.headline_metric,
        *(metric for entry in comparison.external.values() for metric in entry.metrics),
    ]
    return [
        f"Metric {name!r}, which the library does not define"
        for name in dict.fromkeys(named)
        if name not in defined
    ]


def task_data(
    root: Path,
    task_id: str,
    *,
    comparison: ReleasedComparison,
    weights: Mapping[str, float],
    identity: str,
    published: bool,
    sweep: Path,
    algorithm_registries: Sequence[str] | None = None,
    instance_registries: Sequence[str] | None = None,
) -> TaskData:
    """Measure one release's distribution lane once and lay it out as panels.

    Each task's Results are decoded once and grouped in memory, and every
    Metric the drawn instances can be measured on is measured over that one
    decode.

    Args:
        root: The release tree whose Results are read.
        task_id: The task to report on.
        comparison: The fields the protocol states for this task: the
            headline Metric and baseline, the threshold the standings open at,
            the grid the curves span, the eligibility class and the external
            baselines.
        weights: The instance weights the release ships for this task.
        identity: The release as the reader is told about it.
        published: Whether these pages report a published release. A published
            release carries every trial it declares; a local cut may not, and
            an algorithm lacking any declared trial is left out whole, never
            shown on part of its trials.
        sweep: The Sweep declaring the trials this surface reports.

    Raises:
        ValueError: If the drawn instances cannot be measured on the headline
            Metric, a drawn instance is unweighted, no trial falls under a
            declared budget, or, when published, a declared trial is missing.
    """
    defined = list_built_in_metrics(BUILT_IN_TASKS[task_id])
    taus = comparison.taus
    task_collections_root = root
    input_store = root / INPUT_ARTIFACTS_DIRECTORY
    # A release's handle: it reads the shipped Results and Inputs and writes
    # nothing.
    collection = open_collection(root, BUILT_IN_TASKS[task_id])
    has_input_store = input_store.is_dir()
    settings = collection.load_sweep(sweep)
    with reported_separately():
        configured = collection.load_algorithms(
            registries=algorithm_registries, strict=published
        )
        installed = collection.load_instances(
            registries=instance_registries, strict=published
        )
    algorithm_ids = selected_ids(
        settings["algorithms"],
        configured,
        "algorithm",
        lambda: skipped_reasons(root, task_id, "algorithm"),
    )
    instance_ids = selected_ids(
        settings["instances"],
        installed,
        "instance",
        lambda: skipped_reasons(root, task_id, "instance"),
    )
    settings["instances"] = [
        value for uuid, value in installed.items() if uuid in instance_ids
    ]
    metrics = list(
        listed_metrics(task_id, settings["instances"], inputs=has_input_store)
    )
    if comparison.headline_metric not in metrics:
        raise ValueError(
            f"{identity} cannot measure its headline Metric "
            f"{comparison.headline_metric!r} on every instance it draws."
        )
    # The headline Metric leads, as every page opens on it; the rest keep the
    # library's order.
    metrics.sort(key=lambda name: name != comparison.headline_metric)
    selected = [defined[name] for name in metrics]
    groups = _resolved_budget_groups(
        settings, run_count=len(set(settings["seeds"] or ()))
    )
    labels, declared_runs, order = _panel_labels(groups)
    coverage = collection.get_sweep_status(
        settings,
        algorithms=[configured[uuid] for uuid in algorithm_ids],
    )["cells"]
    missing_runs = coverage["never_attempted"]
    if missing_runs and published:
        raise ValueError(
            f"{identity} is missing {missing_runs} of {coverage['expected']} "
            "trials it declares for this task."
        )
    fill = bool(missing_runs)
    instance_order = {
        instance.uuid: index for index, instance in enumerate(settings["instances"])
    }
    # Keep every trial of an (Instance, Algorithm, budget) group together:
    # a Metric requiring the group's outputs depends on the complete cohort.
    # Outputs live for one Algorithm only; the site retains just scalar costs.
    columns = [
        "task",
        "instance",
        "metric_arguments",
        "algorithm",
        "budget",
        "alg_seed",
        *metrics,
        "status",
        "recorded",
    ]
    chunks = []
    # One decode of the task's Results serves the whole roster, and one
    # measurement over it loads each instance's Input once: measured algorithm
    # by algorithm, every Input loaded again for every algorithm. The Outputs
    # leave as scalar costs straight after.
    observed = measure_results(
        collection.load_results(
            settings, algorithms=[configured[uuid] for uuid in algorithm_ids]
        ),
        selected,
        collection=collection,
    ).drop(columns="payload")
    resolved = observed.drop_duplicates("instance_uuid")
    instance_arguments = dict(
        zip(resolved["instance_uuid"], resolved["metric_arguments"])
    )
    del resolved
    by_algorithm = {
        str(alg_uuid): rows.reset_index(drop=True)
        for alg_uuid, rows in observed.groupby("alg_uuid", sort=False)
    }
    unobserved = observed.iloc[:0].copy()
    del observed
    incomplete: dict[str, tuple[str, int, int]] = {}
    for uuid, algorithm in configured.items():
        if uuid not in algorithm_ids:
            continue
        measured_chunk = by_algorithm.pop(uuid, unobserved).copy()
        measured_chunk["recorded"] = True
        if fill:
            # Complete display costs only after measuring observed trial groups.
            # The library resolves the declared cells; no Attempt is invented.
            coordinates = load_sweep_coordinates(
                sweep,
                task_collections_root,
                collection.task,
                algorithm_registries=algorithm_registries,
                instance_registries=instance_registries,
                algorithm_ids=[uuid],
                instance_ids=list(instance_ids),
            )
            keys = ["instance", "budget", "alg_seed"]
            index = pd.MultiIndex.from_tuples(
                [(cell.instance, cell.budget, cell.alg_seed) for cell in coordinates],
                names=keys,
            )
            measured_chunk = measured_chunk.set_index(keys).reindex(index)
            lacking = int(measured_chunk["recorded"].ne(True).sum())
            if lacking:
                # Ranking on part of an algorithm's trials biases the
                # standings, so an incomplete one is left out whole.
                incomplete[uuid] = (algorithm.label or "", lacking, len(index))
                del measured_chunk, coordinates, index
                continue
            measured_chunk = measured_chunk.reset_index()
            del coordinates, index
        if not measured_chunk.empty:
            chunks.append(
                (
                    instance_order[measured_chunk.iloc[0]["instance"].uuid],
                    measured_chunk[columns].copy(),
                )
            )
        del measured_chunk
    del by_algorithm, unobserved
    if not chunks and incomplete:
        raise ValueError(
            f"{identity} carries no configured algorithm with every declared "
            f"trial for task {task_id!r}."
        )
    measured = (
        pd.concat(
            [chunk for _, chunk in sorted(chunks, key=lambda pair: pair[0])],
            ignore_index=True,
        )
        if chunks
        else pd.DataFrame(columns=columns)
    )
    del chunks
    # _resolved_budget_groups already establishes unique (Instance, budget) keys.
    # The join keys on each instance's uuid: matching a million instance objects
    # hashes every one of them, which costs seconds on a large task.
    keyed = labels.assign(
        instance_uuid=[instance.uuid for instance in labels["instance"]]
    ).drop(columns="instance")
    named = (
        measured.assign(
            instance_uuid=[instance.uuid for instance in measured["instance"]]
        )
        .merge(keyed, on=["instance_uuid", "budget"], how="left")
        .drop(columns="instance_uuid")
    )
    drawn: pd.DataFrame = named[named["panel"].notna()]
    if drawn.empty:
        raise ValueError(
            f"{identity} carries no trial under a budget the Sweep for task "
            f"{task_id!r} declares."
        )

    # The study each instance came from is an opaque grouping key here:
    # instances from one study came from one source, which is what a reader
    # groups by. A store that records no provenance offers no such grouping.
    refs = read_provenance(root).study_of
    properties = instance_properties(collection, registries=instance_registries)
    instances = sorted(
        (
            _instance_payload(
                instance, refs.get(instance.uuid), properties.loc[instance.uuid]
            )
            for instance in dict.fromkeys(drawn["instance"])
        ),
        key=lambda entry: str(entry["label"]),
    )
    positions = {str(entry["uuid"]): index for index, entry in enumerate(instances)}
    unweighted = sorted(uuid for uuid in positions if uuid not in weights)
    if unweighted:
        raise ValueError(
            f"{identity} draws instance(s) its released weights do not weigh: "
            f"{unweighted}."
        )
    weights = {uuid: weights[uuid] for uuid in positions}
    registered = registered_algorithms(
        root, task_id, algorithm_registries=algorithm_registries, strict=published
    )
    algorithms = _algorithm_payloads(drawn, registered, comparison.eligibility)
    rows = {entry["uuid"]: index for index, entry in enumerate(algorithms)}

    panels: list[PanelData] = []
    refused: list[tuple[str, str]] = []
    for metric in metrics:
        if metric != comparison.headline_metric and _is_indicator(drawn, metric):
            # A 0/1 indicator has no reading a profile shows; Explore leaves
            # it out.
            continue
        metric_panels: list[PanelData] = []
        try:
            for panel in order:
                cells = drawn[drawn["panel"] == panel]
                if cells.empty:
                    continue
                statistical = defined[metric].baseline_name
                offered = _offered_baselines(
                    cells,
                    metric=metric,
                    comparison=comparison,
                    statistical=statistical,
                    payloads=instances,
                )
                opening = comparison.opening_for(metric, statistical, offered)
                denominators = (
                    dict(zip(positions, offered[opening]))
                    if opening in offered
                    else dict.fromkeys(positions, 1.0)
                )
                metric_panels.append(
                    _panel(
                        cells,
                        metric=metric,
                        budget=panel,
                        rows=rows,
                        positions=positions,
                        instances=instances,
                        denominators=denominators,
                        offered=offered,
                        declared_runs=declared_runs.get(panel, 0),
                        quantile=comparison.headline_quantile,
                        weights=weights,
                    )
                )
        except ValueError as error:
            # A release fills one cell at a time reaches a state where one
            # algorithm covers instances another does not. The library refuses a
            # reading over different denominators; that refusal belongs to the
            # Metric it was raised for, and the other Metrics still report what
            # they can.
            refused.append((metric, str(error)))
            continue
        panels.extend(metric_panels)

    files: dict[str, bytes] = {}
    listed: list[dict[str, Any]] = []
    for panel in panels:
        name = runs_name(panel)
        files[name] = panel.payload
        listed.append(
            {
                "metric": panel.metric,
                "budget": panel.budget,
                "slice": panel.slice,
                "runs": name,
                "seeds": len(panel.seeds),
                "seed_ids": list(panel.seeds),
                # JSON has no infinities; the browser restores these numeric
                # strings at the panel-loading boundary. Binary costs stay native.
                "baselines": {
                    key: [
                        value if math.isfinite(value) else json.dumps(value)
                        for value in values
                    ]
                    for key, values in panel.baselines.items()
                },
                "epsilon": panel.epsilon,
            }
        )
    index = _index(
        task_id=task_id,
        identity=identity,
        published=published,
        comparison=comparison,
        taus=taus,
        instances=instances,
        weights=weights,
        algorithms=algorithms,
        panels=panels,
        listed=listed,
        features=list(properties.columns),
    )
    index["outcomes"] = _outcome_counts(drawn)
    if fill:
        # The counts a preview discloses; the page owns how it reads them.
        index["preview_coverage"] = {
            "missing": missing_runs,
            "expected": coverage["expected"],
        }
    return TaskData(
        index=index,
        files=files,
        panels=panels,
        refused=refused,
        incomplete=incomplete,
        instance_arguments=instance_arguments,
    )


def disambiguated_labels(
    labels: Mapping[str, str], classes: Mapping[str, str]
) -> dict[str, str]:
    """Keep a label shared within one privacy interpretation distinguishable by
    its configuration UUID; a page tells classes apart by their tag."""
    counts: dict[tuple[str, str], int] = {}
    for uuid, label in labels.items():
        key = (label, classes.get(uuid, ""))
        counts[key] = counts.get(key, 0) + 1
    return {
        uuid: (
            label
            if counts[(label, classes.get(uuid, ""))] == 1
            else f"{label} ({uuid[-8:]})"
        )
        for uuid, label in labels.items()
    }


def _algorithm_payloads(
    drawn: pd.DataFrame,
    registered: Mapping[str, RegisteredAlgorithm],
    eligibility: str,
) -> list[dict[str, Any]]:
    """Describe every drawn algorithm by what a page colours, ranks and filters on."""
    payloads: list[dict[str, Any]] = []
    algorithms = dict.fromkeys(drawn["algorithm"])
    labels = disambiguated_labels(
        {algorithm.uuid: str(algorithm) for algorithm in algorithms},
        {
            algorithm.uuid: algorithm.privacy_interpretation.value
            for algorithm in algorithms
        },
    )
    for algorithm in algorithms:
        facts = registered.get(algorithm.uuid)
        estimator = facts.estimator if facts else _estimator_class(algorithm)
        declared = algorithm.privacy_interpretation.value
        family = algorithm_family(estimator)
        payloads.append(
            {
                "uuid": algorithm.uuid,
                "label": labels[algorithm.uuid],
                "class": declared,
                "diagnostic": not shown_by_default(declared, eligibility),
                "family": family,
                "slot": family_slot(family),
                "core": estimator,
                "transformations": list(facts.transformations if facts else ()),
                "data_bounds": bool(facts.data_bounds) if facts else False,
                "fwl_demean": bool(facts.fwl_demean) if facts else False,
            }
        )
    _apply_markers(payloads)
    return payloads


def _apply_markers(payloads: list[dict[str, Any]]) -> None:
    """Give each configuration the marker separating it within its core."""
    shapes = marker_shapes_by_uuid(payloads)
    for payload in payloads:
        payload["marker"] = shapes[str(payload["uuid"])]


def _panel(
    cells: pd.DataFrame,
    *,
    metric: str,
    budget: str,
    rows: Mapping[str, int],
    positions: Mapping[str, int],
    instances: Sequence[dict[str, Any]],
    denominators: Mapping[str, float],
    offered: Mapping[str, Sequence[float]],
    declared_runs: int,
    quantile: float,
    weights: Mapping[str, float],
) -> PanelData:
    """Lay one metric and budget's trials out as one dense matrix.

    The matrix reserves one slot per algorithm, instance and declared seed, in
    that order, so a reader's own derivations index into it arithmetically. A
    slot no Attempt answered stays ``NaN``. A recorded non-success trial stores
    its negative status code (see ``_STATUS_CODES``); a declared cell no
    Attempt answered at all stores ``_NOT_RUN_CODE`` instead of the ordinary
    +inf ``cells`` carries for it, so the scientific reading over ``cells``
    itself (below, and this panel's ``measured`` counts) is untouched by this
    matrix's own encoding.
    """
    width = len(instances)
    seeds = sorted({int(seed) for seed in cells["alg_seed"]})
    slots = {seed: index for index, seed in enumerate(seeds)}
    stride = len(seeds)
    values = [math.nan] * (len(rows) * width * stride)
    measured: dict[str, int] = {}
    for algorithm, instance, seed, value, recorded, status in zip(
        cells["algorithm"],
        cells["instance"],
        cells["alg_seed"],
        cells[metric],
        cells["recorded"],
        cells["status"],
    ):
        uuid = algorithm.uuid
        succeeded = recorded and status == ResultStatus.SUCCESS.value
        measured[uuid] = measured.get(uuid, 0) + int(succeeded)
        cell = (rows[uuid] * width + positions[instance.uuid]) * stride
        code = None if recorded else _NOT_RUN_CODE
        if code is None:
            code = _STATUS_CODE_BY_NAME.get(status)
        values[cell + slots[int(seed)]] = float(value) if code is None else float(code)
    return PanelData(
        metric=metric,
        budget=budget,
        payload=gzip.compress(struct.pack(f"<{len(values)}d", *values), mtime=0),
        seeds=seeds,
        baselines={name: list(values) for name, values in offered.items()},
        epsilon=_epsilons(cells, positions, width),
        declared_runs=declared_runs,
        measured=measured,
        headline=_headlines(cells, quantile, metric, denominators, weights),
    )


def _headlines(
    cells: pd.DataFrame,
    quantile: float,
    metric: str,
    denominators: Mapping[str, float],
    weights: Mapping[str, float],
) -> dict[str, Headline]:
    """Read each algorithm's standing under the headline quantile, exactly.

    The library owns the weighting end to end: ``compute_error_ratio``
    divides each cost by the denominator the release shipped, and
    ``compute_profile_curves`` takes the share inside each instance before
    averaging it across them under the release's weights. Evaluated at every
    distinct finite ratio an algorithm reaches, that curve's first threshold at
    or above the quantile is its headline τ, never a point of the drawing grid.
    """
    reference = cells[["task", "instance"]].drop_duplicates()
    reference = reference.assign(
        baseline=reference["instance"].map(lambda instance: denominators[instance.uuid])
    )
    normalized = compute_error_ratio(cells, reference, metric=metric)
    shares = pd.DataFrame(
        [
            {"instance": instance, "weight": weights[instance.uuid]}
            for instance in cells["instance"].drop_duplicates()
        ],
        columns=["instance", "weight"],
    )
    # Every curve averages over the same Instances, as the drawn curves do.
    listed = list(normalized["instance"].drop_duplicates())
    readings: dict[str, Headline] = {}
    for algorithm, trials in normalized.groupby("algorithm", sort=False):
        ratios = np.unique(trials[metric].dropna().to_numpy(dtype=float))
        reached = ratios[np.isfinite(ratios)]
        if not reached.size:
            readings[algorithm.uuid] = Headline(math.inf, 0.0, math.inf)
            continue
        curve = compute_profile_curves(
            trials, reached, metric=metric, instances=listed, weights=shares
        )
        fractions = curve["fraction"].to_numpy(dtype=float)
        # A share summed in floating point may fall a rounding short of the
        # quantile it exactly reaches.
        within = np.flatnonzero(fractions >= quantile - 1e-12)
        share = float(fractions[-1])
        readings[algorithm.uuid] = Headline(
            float(reached[within[0]]) if within.size else math.inf,
            share,
            # A curve reaching no share has no τ that reaches it.
            (
                float(reached[np.flatnonzero(fractions >= share - 1e-12)[0]])
                if share > 0
                else math.inf
            ),
        )
    return readings


def _epsilons(
    cells: pd.DataFrame, positions: Mapping[str, int], width: int
) -> list[float] | None:
    """The budget's epsilon on each instance, where it supplies one for all."""
    found: list[float | None] = [None] * width
    for instance, budget in zip(cells["instance"], cells["budget"]):
        if not isinstance(budget, (PureDPBudget, ApproxDPBudget)):
            return None
        if not math.isfinite(budget.eps):
            return None
        found[positions[instance.uuid]] = float(budget.eps)
    if any(value is None for value in found):
        return None
    return [value for value in found if value is not None]


def _outcome_counts(
    drawn: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count each declared trial's terminal outcome once, by algorithm and budget.

    ``drawn`` carries one row per trial regardless of how many Metrics measure
    it, so grouping it directly counts every trial exactly once. ``panel`` is
    the declared budget's own display label (``budget`` carries the typed
    object, unordered and ungroupable). A group with no recorded trial reports
    zero for every outcome without calling :func:`summarize_outcomes`,
    which refuses an empty frame.
    """
    frame = drawn.assign(uuid=[algorithm.uuid for algorithm in drawn["algorithm"]])
    declared = frame.groupby(["uuid", "panel"]).size()
    recorded_frame = frame[frame["recorded"]]
    by_key: dict[tuple[str, str], dict[str, int]] = {}
    if not recorded_frame.empty:
        shares = summarize_outcomes(recorded_frame, by=["uuid", "panel"])
        for uuid, panel, status, count in zip(
            shares["uuid"], shares["panel"], shares["status"], shares["count"]
        ):
            by_key.setdefault((str(uuid), str(panel)), {})[str(status)] = int(count)
    recorded_totals = recorded_frame.groupby(["uuid", "panel"]).size()
    outcomes: dict[str, dict[str, dict[str, int]]] = {}
    for (uuid, panel), total in declared.items():
        key = (str(uuid), str(panel))
        by_status = by_key.get(key, {})
        outcomes.setdefault(key[0], {})[key[1]] = {
            **{status.value: by_status.get(status.value, 0) for status in ResultStatus},
            "declared": int(total),
            "recorded": int(recorded_totals.get((uuid, panel), 0)),
        }
    return outcomes


def runs_name(panel: PanelData) -> str:
    """The file one panel's trial matrix ships in."""
    return f"{element_id('runs', panel.metric, panel.budget)}.f64.gz"


def _index(
    *,
    task_id: str,
    identity: str,
    published: bool,
    comparison: ReleasedComparison,
    taus: Sequence[float],
    instances: Sequence[dict[str, Any]],
    weights: Mapping[str, float],
    algorithms: Sequence[dict[str, Any]],
    panels: Sequence[PanelData],
    listed: Sequence[dict[str, Any]],
    features: Sequence[str],
) -> dict[str, Any]:
    """Assemble the one index every panel file is read through."""
    coverage: dict[str, dict[str, list[int]]] = {}
    for panel in panels:
        key = f"{panel.metric}/{panel.budget}"
        coverage[key] = {
            entry["uuid"]: [
                panel.declared_runs,
                panel.measured.get(str(entry["uuid"]), 0),
            ]
            for entry in algorithms
        }
    return {
        "task": task_id,
        "identity": identity,
        "published": published,
        "taus": list(taus),
        "headline_metric": comparison.headline_metric,
        "headline_quantile": comparison.headline_quantile,
        # The scale the τ grid is spaced on, which the charts draw τ on.
        "tau_scale": comparison.grid.scale,
        # The τ Explore's outcomes view and notebook open at; not a ranking.
        "opening_tau": OPENING_TAU,
        "leading": LEADING_CURVES,
        "baselines": _baseline_catalogue(task_id, comparison, panels),
        "pool": comparison.pool,
        "metric_labels": {
            metric: metric_label(task_id, metric)
            for metric in dict.fromkeys(panel.metric for panel in panels)
        },
        "weights": [weights[str(entry["uuid"])] for entry in instances],
        "status_codes": {
            **{str(code): status.value for status, code in _STATUS_CODES.items()},
            str(_NOT_RUN_CODE): "not_run",
        },
        "visible_classes": [comparison.eligibility],
        "features": {name: FEATURE_NAMES[name] for name in [*features, "epsilon"]},
        "notes": {},
        "instances": list(instances),
        "algorithms": list(algorithms),
        "panels": list(listed),
        "coverage": {ALL_SLICE: coverage},
        "headline": _headline_index(panels),
    }


def _headline_index(
    panels: Sequence[PanelData],
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Each panel's headline readings, by slice, panel and algorithm, as JSON
    carries them."""
    index: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for panel in panels:
        index.setdefault(panel.slice, {})[f"{panel.metric}/{panel.budget}"] = {
            uuid: reading.json() for uuid, reading in panel.headline.items()
        }
    return index


def _slice_index(
    comparison: ReleasedComparison, sizes: Mapping[str, tuple[int, int]]
) -> list[dict[str, Any]]:
    """The slices a Leaderboard offers, "all" first, each with its size."""
    entries: list[dict[str, Any]] = [{"key": ALL_SLICE}]
    for entry in comparison.slices:
        entries.append(
            {
                "key": entry.key,
                "feature": entry.feature,
                "above" if entry.above else "at_most": entry.bound,
            }
        )
    for entry in entries:
        instances, studies = sizes[entry["key"]]
        entry.update(instances=instances, studies=studies)
    return entries


def metric_label(task_id: str, metric: str) -> str:
    """What one Metric is called for a reader: the library's own label for it.

    A chart axis, a picker option and a formula line print this; the key
    stays what the scripts and the notebook read.
    """
    return list_built_in_metrics(BUILT_IN_TASKS[task_id])[metric].label


def baseline_label(key: str) -> str:
    """What one baseline is called for a reader: its name in the site copy.

    The name carries no qualifier; the baseline's full definition is on the
    task's Methodology page. A key the copy names nothing for is shown as the
    key the release gives it.
    """
    if key not in SITE_COPY_KEYS["baselines"]:
        return key
    return copy_text("baselines", key) or key


def _baseline_catalogue(
    task_id: str, comparison: ReleasedComparison, panels: Sequence[PanelData]
) -> dict[str, dict[str, Any]]:
    """What each Metric may be read against, and which one it opens on.

    A baseline is offered for a Metric only where every one of that Metric's
    panels carries a denominator under its name, so changing the budget
    never silently drops the reader's choice. The opening baseline leads and
    the Metric's own units follow; the rest keep the order they are offered
    in: the Metric's statistical baseline, the external baselines offered for
    it, then the computed ones. Only the headline baseline on the headline Metric is
    marked as the protocol's.
    """
    task = BUILT_IN_TASKS[task_id]
    defined = list_built_in_metrics(task)
    catalogue: dict[str, dict[str, Any]] = {}
    for metric in dict.fromkeys(panel.metric for panel in panels):
        shared: set[str] | None = None
        for panel in panels:
            if panel.metric == metric:
                keys = set(panel.baselines)
                shared = keys if shared is None else shared & keys
        available = shared or set()
        statistical = defined[metric].baseline_name
        stored = [
            name
            for name in comparison.stored_for(metric, statistical)
            if name in available
        ]
        computed = [
            name for name in comparison.computed_for(metric) if name in available
        ]
        opening = comparison.opening_for(metric, statistical, available)
        # Reading against no baseline needs no shipped vector, so it is offered
        # for every Metric whatever the panels carry.
        ordered = list(dict.fromkeys((opening, NO_BASELINE, *stored, *computed)))
        catalogue[metric] = {
            "default": opening,
            "options": [
                {
                    "key": key,
                    "label": baseline_label(key),
                    "description": baseline_definition(task, comparison, key),
                    "protocol": metric == comparison.headline_metric
                    and key == comparison.headline_baseline,
                }
                for key in ordered
            ],
        }
    return catalogue


def released_standings(
    view: Path,
    task_id: str,
    *,
    comparison: ReleasedComparison,
    registered: Mapping[str, RegisteredAlgorithm],
    identity: str,
    available: Collection[str],
    omit: Collection[str] = (),
) -> TaskData:
    """Ingest one released profile view as curves-only panels.

    The view publishes standings over the evaluation lane, which the release
    does not ship, so there is no trial matrix to derive them from: each panel
    carries the published curves verbatim, on the thresholds the view itself
    publishes, and the standing each one reaches at the threshold the protocol
    reads them at. The baseline each row names is shown as the release states
    it; nothing here checks or recomputes it.

    Args:
        view: The released view to read.
        task_id: The task it publishes.
        comparison: The scalars the release's protocol states for this task.
        registered: What the release's registries say about each algorithm.
        identity: The release as the reader is told about it.
        omit: Algorithm identifiers whose rows are dropped before anything is
            grouped; ``TaskData.omitted`` counts those the view ranked.

    Raises:
        ValueError: If the view publishes another task, repeats a row, names
            two baselines for one Metric, or leaves a curve short of a
            threshold its section publishes.
    """
    grouped: dict[
        tuple[str, str, str], dict[str, dict[float, tuple[float, int, int]]]
    ] = {}
    labels: dict[str, str] = {}
    shipped_baseline: dict[str, str] = {}
    readings: dict[tuple[str, str, str], dict[str, Headline]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    dropped: set[str] = set()
    for (
        row_task,
        algorithm,
        uuid,
        budget,
        metric,
        baseline,
        slice_key,
        tau,
        fraction,
        run_count,
        measured_run_count,
        tau_at_headline,
        share_reached,
        tau_at_share_reached,
        slice_instances,
        slice_studies,
    ) in _shipped_profile_rows(view):
        if uuid not in available:
            raise ValueError(
                f"{identity} publishes a curve without admitted Evaluation evidence: {uuid}."
            )
        if uuid in omit:
            dropped.add(uuid)
            continue
        if row_task != task_id:
            raise ValueError(
                f"Profile view for task {task_id!r} publishes rows for "
                f"task {row_task!r}: {view}."
            )
        named = shipped_baseline.setdefault(metric, baseline)
        if named != baseline:
            raise ValueError(
                f"Profile view reads {metric!r} against both {named!r} and "
                f"{baseline!r}: {view}."
            )
        if sizes.setdefault(slice_key, (slice_instances, slice_studies)) != (
            slice_instances,
            slice_studies,
        ):
            raise ValueError(
                f"Profile view states two sizes for slice {slice_key!r}: {view}."
            )
        labels[uuid] = algorithm
        points = grouped.setdefault((slice_key, metric, budget), {}).setdefault(
            uuid, {}
        )
        if tau in points:
            raise ValueError(
                f"Profile view repeats {algorithm!r} at threshold "
                f"{tau!r} under budget {budget!r}: {view}."
            )
        points[tau] = (fraction, run_count, measured_run_count)
        reading = Headline(tau_at_headline, share_reached, tau_at_share_reached)
        stated = readings.setdefault((slice_key, metric, budget), {}).setdefault(
            uuid, reading
        )
        if stated != reading:
            raise ValueError(
                f"Profile view states two headline readings for {algorithm!r} "
                f"under {metric!r} {budget!r} in slice {slice_key!r}: {view}."
            )
    ruled = [ALL_SLICE, *(entry.key for entry in comparison.slices)]
    if sizes and set(sizes) != set(ruled):
        raise ValueError(
            f"Profile view's slices {sorted(sizes)} are not the protocol's "
            f"{ruled}: {view}."
        )
    if set(labels) != set(available) - set(omit):
        raise ValueError(
            f"{identity} publishes curves for a different set of algorithms than its admitted Evaluation evidence."
        )
    taus = tuple(
        sorted(
            {
                tau
                for points in grouped.values()
                for curve in points.values()
                for tau in curve
            }
        )
    )
    labels = disambiguated_labels(
        labels,
        {
            uuid: registered[uuid].privacy_interpretation
            for uuid in labels
            if uuid in registered
        },
    )
    panels: list[PanelData] = []
    files: dict[str, bytes] = {}
    listed: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, dict[str, list[int]]]] = {}
    for (slice_key, metric, budget), points in grouped.items():
        if set(points) != set(available) - set(omit):
            raise ValueError(
                f"{identity} omits admitted Evaluation curves under {metric} "
                f"{budget} in slice {slice_key!r}."
            )
        curves: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        for uuid, published in points.items():
            missing = [tau for tau in taus if tau not in published]
            if missing:
                raise ValueError(
                    f"Profile view leaves {labels[uuid]!r} without thresholds "
                    f"{missing} its {budget} curves publish: {view}."
                )
            cells = [published[tau] for tau in taus]
            curves[uuid] = [fraction for fraction, _, _ in cells]
            # Every threshold of one curve counts the same trials.
            _, declared, measured = cells[-1]
            counts[uuid] = [declared, measured]
        name = released_curves_name(slice_key, metric, budget)
        files[name] = released_payload(metric, budget, taus, curves).encode("utf-8")
        panels.append(
            PanelData(
                metric=metric,
                budget=budget,
                payload=b"",
                seeds=[],
                baselines={},
                epsilon=None,
                declared_runs=max(pair[0] for pair in counts.values()),
                measured={uuid: pair[1] for uuid, pair in counts.items()},
                headline=readings[(slice_key, metric, budget)],
                slice=slice_key,
            )
        )
        listed.append(
            {
                "metric": metric,
                "budget": budget,
                "slice": slice_key,
                "runs": None,
                "curves": name,
            }
        )
        coverage.setdefault(slice_key, {})[f"{metric}/{budget}"] = counts
    index = {
        "task": task_id,
        "identity": identity,
        "published": True,
        "taus": list(taus),
        "headline_metric": comparison.headline_metric,
        "headline_quantile": comparison.headline_quantile,
        # The scale the τ grid is spaced on, which the charts draw τ on.
        "tau_scale": comparison.grid.scale,
        "leading": LEADING_CURVES,
        # A Leaderboard reads each Metric against the one baseline its view
        # names, and offers no other: the choice belongs to Explore.
        "baselines": {
            metric: {
                "default": name,
                "options": [
                    {
                        "key": name,
                        "label": baseline_label(name),
                        "description": baseline_definition(
                            BUILT_IN_TASKS[task_id], comparison, name
                        ),
                        "protocol": True,
                    }
                ],
            }
            for metric, name in shipped_baseline.items()
        },
        "metric_labels": {
            metric: metric_label(task_id, metric) for _, metric, _ in grouped
        },
        "weights": [],
        "visible_classes": [comparison.eligibility],
        "features": {},
        "notes": {},
        "instances": [],
        "algorithms": _view_algorithms(labels, registered, comparison.eligibility),
        "panels": listed,
        "coverage": coverage,
        "headline": _headline_index(panels),
        "slices": _slice_index(comparison, sizes) if sizes else [],
    }
    return TaskData(
        index=index, files=files, panels=panels, refused=[], omitted=len(dropped)
    )


def _shipped_profile_rows(view: Path) -> list[ProfileRow]:
    """Read one released profile view as its typed rows, in published order."""
    try:
        schema = pq.read_schema(view)
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"Released view is not readable Parquet: {view}.") from error
    if not schema.remove_metadata().equals(PROFILE_VIEW_SCHEMA):
        raise ValueError(
            f"Released view does not carry the columns and types this build "
            f"reads {PROFILE_VIEW_SCHEMA.to_string(show_schema_metadata=False)}: "
            f"{view}."
        )
    frame = pd.read_parquet(view)
    required = [
        name for name in PROFILE_VIEW_COLUMNS if name not in _NULLABLE_VIEW_COLUMNS
    ]
    if frame[required].isna().any().any():
        raise ValueError(f"Released view has an empty field: {view}.")
    frame[list(_NULLABLE_VIEW_COLUMNS)] = frame[list(_NULLABLE_VIEW_COLUMNS)].fillna(
        math.inf
    )
    return list(zip(*(frame[column].tolist() for column in PROFILE_VIEW_COLUMNS)))


def _view_algorithms(
    labels: Mapping[str, str],
    registered: Mapping[str, RegisteredAlgorithm],
    eligibility: str,
) -> list[dict[str, Any]]:
    """Describe each published curve's algorithm from the release's registry."""
    payloads: list[dict[str, Any]] = []
    for uuid, label in labels.items():
        facts = registered.get(uuid)
        declared = facts.privacy_interpretation if facts else None
        family = algorithm_family(facts.estimator if facts else "")
        payloads.append(
            {
                "uuid": uuid,
                "label": label,
                "class": declared,
                "diagnostic": not shown_by_default(declared, eligibility),
                "family": family,
                "slot": family_slot(family),
                "core": facts.estimator if facts else "",
                "transformations": list(facts.transformations if facts else ()),
                "data_bounds": bool(facts.data_bounds) if facts else False,
                "fwl_demean": bool(facts.fwl_demean) if facts else False,
            }
        )
    _apply_markers(payloads)
    return payloads


def released_curves_name(slice_key: str, metric: str, budget: str) -> str:
    """The file one released panel's published curves ship in."""
    return f"{element_id('curves', slice_key, metric, budget)}.json"


def released_payload(
    metric: str,
    budget: str,
    taus: Sequence[float],
    curves: Mapping[str, Sequence[float]],
) -> str:
    """Serialize one released panel's own fractions, exactly as published."""
    return embed(
        {
            "metric": metric,
            "budget": budget,
            "taus": [float(tau) for tau in taus],
            "curves": {
                uuid: [float(value) for value in row] for uuid, row in curves.items()
            },
        }
    )
