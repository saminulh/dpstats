"""Baseline comparisons and performance profiles over measurement frames."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from operator import attrgetter
from typing import Any

import numpy as np
import pandas as pd

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.analysis._measurements import ArgumentResolver
from dpstats.instance import Instance
from dpstats.metric import Metric

_TRIAL_GROUP_KEYS = ("task", "algorithm", "budget", "instance")
_LABEL_COLUMNS = ("algorithm_label", "instance_label")
_INSTANCE_KEYS = ("task", "instance")


def measured_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Select rows with a measured value for one explicitly named Metric."""
    if metric not in frame.columns:
        raise ValueError(f"Metric column {metric!r} is absent.")
    missing = frame[metric].isna()
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} rows have missing {metric!r} measurements and are excluded from this analysis.",
            stacklevel=3,
        )
    return frame.loc[~missing]


def summarize_by(
    frame: pd.DataFrame,
    keys: Sequence[str],
    aggregation: str | Callable[..., float],
    metric: str,
) -> pd.DataFrame:
    """Summarize each group's costs with one pandas aggregation.

    Whatever a grouped column of costs accepts is accepted here, so an
    aggregation is refused only where pandas cannot apply it. Groups keep the
    order they first appear in, since a grouping column may hold configured
    values, which answer equality but not order.
    """
    frame = measured_rows(frame, metric)
    grouped = frame.groupby(list(keys), dropna=False, sort=False)[metric]
    try:
        summarized = grouped.agg(aggregation)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            f"{aggregation!r} is not an aggregation pandas can apply to a "
            "column of costs."
        ) from exc
    return summarized.reset_index()


def aggregate_trials(
    frame: pd.DataFrame,
    *,
    metric: Metric | str,
    aggregation: str | Callable[..., float] = "median",
) -> pd.DataFrame:
    """Summarize the trials of each trial group with one aggregation.

    A trial group is all trials of one algorithm on one Instance at one
    budget, so its trials differ only by algorithm seed. Costs are
    extended-real, and an aggregation over them keeps an infinite cost in the
    ordering rather than discarding it, so a trial group whose trials mostly
    failed summarizes as infinite.

    Args:
        metric: The Metric or its named measurement column.
        frame: Measurement rows carrying the trial-group columns and their
            named Metric column.
        aggregation: Any aggregation a grouped column of costs accepts, by
            name or as a callable. Defaults to the median.

    Returns:
        One row per trial group: its key columns, the label columns the
        frame carries, and the summarized Metric column.

    Raises:
        ValueError: If pandas cannot apply ``aggregation`` to a cost column.
    """
    metric = str(metric)
    labels = [column for column in _LABEL_COLUMNS if column in frame.columns]
    summarized = summarize_by(frame, [*_TRIAL_GROUP_KEYS, *labels], aggregation, metric)
    if "metric_arguments" in frame.columns:
        arguments = frame.drop_duplicates(list(_INSTANCE_KEYS))[
            [*_INSTANCE_KEYS, "metric_arguments"]
        ]
        summarized = summarized.merge(
            arguments, on=list(_INSTANCE_KEYS), validate="many_to_one"
        )
    return summarized


def _algorithm_rows(frame: pd.DataFrame, algorithm: BaseAlgorithm) -> pd.DataFrame:
    """Select one configured identity without conflating equal display labels."""
    selected = frame.loc[frame["algorithm"].map(attrgetter("uuid")) == algorithm.uuid]
    if selected.empty:
        raise ValueError(
            f"Frame carries no rows for reference algorithm {algorithm!r}."
        )
    return selected


def _ranked_baseline(
    pool: pd.DataFrame,
    metric: str,
    *,
    per_budget: bool,
    aggregation: str | Callable[..., float],
) -> pd.DataFrame:
    """The pool's minimum aggregated cost, per Instance and, if asked, budget."""
    if pool.empty:
        raise ValueError("The baseline pool has no rows to rank.")
    pool = pool.assign(algorithm=pool["algorithm"].map(attrgetter("uuid")))
    if per_budget:
        aggregated = aggregate_trials(pool, metric=metric, aggregation=aggregation)
        group_keys = [*_INSTANCE_KEYS, "budget"]
    else:
        aggregated = summarize_by(
            pool, [*_INSTANCE_KEYS, "algorithm"], aggregation, metric
        )
        group_keys = list(_INSTANCE_KEYS)
    values = (
        aggregated.groupby(group_keys, dropna=False, sort=False)[metric]
        .min()
        .reset_index()
    )
    return values.rename(columns={metric: "baseline"})


def _stored_baseline(
    frame: pd.DataFrame, name: str, metric: Metric | str
) -> pd.DataFrame:
    """A resolved or source-supplied value as a fixed per-Instance baseline.

    An unmeasured frame carries no resolved column; its Instances' stored
    values are read, less any name the Metric's Task produces.
    """
    instances = frame.drop_duplicates(list(_INSTANCE_KEYS))
    if "metric_arguments" in instances.columns:
        arguments: list[dict[str, Any]] = instances["metric_arguments"].tolist()
    else:
        resolver = ArgumentResolver((metric,) if isinstance(metric, Metric) else ())
        arguments = [
            resolver.source_arguments(instance) for instance in instances["instance"]
        ]
    missing = [
        str(instance)
        for instance, values in zip(instances["instance"], arguments)
        if name not in values
    ]
    if missing:
        raise ValueError(
            f"Instances {missing} have no value named {name!r}; "
            "measure with collection= to compute produced values from Inputs."
        )
    return instances[list(_INSTANCE_KEYS)].assign(
        baseline=[float(values[name]) for values in arguments]
    )


def _class_rows(frame: pd.DataFrame, wanted: PrivacyInterpretation) -> pd.DataFrame:
    """The rows whose algorithm declares this privacy interpretation."""
    # The class belongs to the configured algorithm, so it is derived once per
    # algorithm rather than once per row.
    classes: dict[str, str] = {}
    keep: list[bool] = []
    for algorithm in frame["algorithm"]:
        declared = classes.get(algorithm.uuid)
        if declared is None:
            declared = classes[algorithm.uuid] = algorithm.privacy_interpretation.value
        keep.append(declared == wanted.value)
    return frame.loc[keep]


def _best_of_rows(
    frame: pd.DataFrame, algorithms: Sequence[BaseAlgorithm]
) -> pd.DataFrame:
    """The rows of a named roster, refusing a member the frame does not carry."""
    if not algorithms:
        raise ValueError("A best-of baseline needs at least one algorithm.")
    named: dict[str, BaseAlgorithm] = {}
    for algorithm in algorithms:
        # The annotation cannot stop a caller passing a uuid string.
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            algorithm, BaseAlgorithm
        ):
            raise TypeError(
                "A best-of baseline lists configured algorithms; one member is "
                f"a {type(algorithm).__name__!r}."
            )
        named.setdefault(algorithm.uuid, algorithm)
    return pd.concat(
        [_algorithm_rows(frame, algorithm) for algorithm in named.values()]
    )


def compute_baseline(
    frame: pd.DataFrame,
    *,
    metric: Metric | str,
    baseline: BaseAlgorithm | Sequence[BaseAlgorithm] | PrivacyInterpretation | None,
    per_budget: bool = True,
    aggregation: str | Callable[..., float] = "median",
) -> pd.DataFrame:
    """Derive the baseline table a Metric's costs are divided by.

    The kind of baseline follows the type of ``baseline``. A stored
    per-Instance value needs no table: pass its key straight to
    :func:`compute_error_ratio`.

    Args:
        frame: Measurement rows carrying the trial-group columns.
        metric: The Metric or its named measurement column.
        baseline: A configured algorithm gives that algorithm's aggregated
            cost, a sequence of them the best among those, and an
            privacy interpretation the best among the frame's algorithms of
            that class. ``None`` gives one everywhere, leaving costs in their
            own units.
        per_budget: Key an algorithm, sequence or class baseline by budget as
            well as Instance. ``None`` is per-Instance regardless.
        aggregation: The aggregation over trials to take before ranking or
            reading an algorithm's own cost, by name or as a callable.

    Returns:
        A baseline table: one ``baseline`` row per covered Instance, keyed by
        ``task`` and ``instance``, and by ``budget`` too when ``per_budget``
        applies.

    Raises:
        ValueError: If a named algorithm or class has no row to rank, or
            pandas cannot apply the aggregation.
        TypeError: If ``baseline`` is a string, or a sequence lists something
            other than configured algorithms.
    """
    metric = str(metric)
    if baseline is None:
        instances = measured_rows(frame, metric)[list(_INSTANCE_KEYS)]
        return instances.drop_duplicates().assign(baseline=1.0)
    if isinstance(baseline, PrivacyInterpretation):
        rows = _class_rows(frame, baseline)
    elif isinstance(baseline, str):
        raise TypeError(
            f"compute_baseline takes no stored key {baseline!r}; "
            "pass the key to compute_error_ratio."
        )
    elif isinstance(baseline, BaseAlgorithm):
        rows = _algorithm_rows(frame, baseline)
    else:
        rows = _best_of_rows(frame, baseline)
    return _ranked_baseline(
        rows, metric, per_budget=per_budget, aggregation=aggregation
    )


def _broadcast(rows: pd.DataFrame, keyed: pd.DataFrame, column: str) -> pd.DataFrame:
    """Join one caller-supplied keyed frame onto rows it may cover broadly.

    The keyed frame's own columns are the join: a key it omits broadcasts
    across every value that key takes among the rows. A row the frame does not
    cover has no honest value to stand in for, so it is refused rather than
    dropped or filled.
    """
    if column not in keyed.columns:
        raise ValueError(f"A {column} frame needs a {column!r} column beside its keys.")
    keys = [name for name in keyed.columns if name != column]
    if not keys:
        raise ValueError(f"A {column} frame needs at least one key column.")
    if keyed.duplicated(subset=keys).any():
        raise ValueError(
            f"A {column} frame carries more than one {column} for the same "
            f"{', '.join(keys)}."
        )
    merged = rows.merge(keyed, on=keys, how="left")
    uncovered = merged[column].isna()
    if bool(uncovered.any()):
        sample = merged.loc[uncovered, keys].drop_duplicates().head(3)
        raise ValueError(
            f"{int(uncovered.sum())} row(s) have no {column}, including "
            f"{sample.to_dict(orient='records')}."
        )
    return merged


def compute_error_ratio(
    frame: pd.DataFrame,
    baseline: pd.DataFrame | str | None = None,
    *,
    metric: Metric | str,
) -> pd.DataFrame:
    """Read each cost as its error ratio, a multiple of its baseline.

    Division on extended-real costs is total and deterministic: a cost matching
    an exact zero baseline is one, anything worse against that
    baseline to infinity, a finite cost against an infinite baseline to zero,
    within every profile threshold because it answered where the baseline
    failed, and a quotient no order can rank takes that same worst score.
    Missing measurements are excluded before division, with a warning.

    A baseline table is a ``baseline`` column beside its key columns, and its
    keys are whichever other columns it carries: each row joins on those, and
    a key column the table omits broadcasts across every value it takes.
    :func:`compute_baseline` keys by ``task`` and ``instance``, and by
    ``budget`` when per budget; a table built elsewhere may key by
    ``instance_uuid`` instead. Every row must be covered exactly once.

    A string names a per-Instance value, such as ``"published_se"``, read
    from the frame's resolved ``metric_arguments`` column or from
    source-supplied Instance arguments; any value may be named for any
    Metric. Measure with ``collection=`` to compute library-produced values
    before dividing. Omitted, the baseline is the Metric's own statistical
    baseline.

    Args:
        frame: Trial or :func:`aggregate_trials` rows carrying the named
            Metric column.
        baseline: A baseline table, from :func:`compute_baseline` or built
            on the same contract, the name of a per-Instance value, or
            ``None`` for ``metric.baseline``.
        metric: The Metric or its named measurement column.

    Returns:
        The rows with the selected Metric column replaced by its
        baseline-relative ratio and the matched ``baseline`` beside it.

    Raises:
        ValueError: If the frame already carries a ``baseline`` column, the
            baseline is omitted for a column name or a Metric with no
            statistical baseline, an Instance stores no such value, or the
            baseline table is shaped wrongly, keys a value more than once, or
            does not cover every row.
    """
    column = str(metric)
    if baseline is None:
        if not isinstance(metric, Metric):
            raise ValueError(
                f"The error ratio of column {column!r} with no baseline needs the "
                "Metric, whose statistical baseline is read; pass baseline= or the Metric."
            )
        if metric.baseline_name is None:
            raise ValueError(
                f"Metric {metric.name!r} declares no statistical baseline; pass "
                "baseline=."
            )
        baseline = metric.baseline_name
    if "baseline" in frame.columns:
        raise ValueError(
            "The frame already carries a baseline column; pass rows that have "
            "not been divided."
        )
    rows = measured_rows(frame, column)
    table = (
        _stored_baseline(rows, baseline, metric)
        if isinstance(baseline, str)
        else baseline
    )
    merged = _broadcast(rows, table, "baseline")
    value = merged[column].to_numpy(dtype=float)
    divisor = merged["baseline"].to_numpy(dtype=float)
    ratio = np.full(value.shape, math.inf)
    with np.errstate(invalid="ignore"):
        np.divide(value, divisor, out=ratio, where=divisor != 0)
    ratio[(divisor == 0) & (value == 0)] = 1.0
    # A measured negative infinity is a legitimate best cost and divides
    # ordinarily; an infinite cost and an unrankable quotient carry no ratio.
    ratio[np.isposinf(value) | np.isnan(ratio)] = math.inf
    merged[column] = ratio
    return merged


def compute_profile_curves(
    frame: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    metric: Metric | str,
    instances: Sequence[Instance] | None = None,
    weights: pd.DataFrame | None = None,
    baseline: pd.DataFrame | str | None = None,
) -> pd.DataFrame:
    """Trace what share of trials each algorithm keeps within each threshold.

    At threshold ``tau`` an algorithm's curve gives the share of its trials
    costing at most ``tau``, taken inside each Instance first and then
    averaged across Instances, so an Instance carrying many trials does not
    outvote the rest. A trial with a positive-infinite cost is within no
    threshold. Every row passed counts toward its algorithm's curve whatever
    budget it ran under, so select the budgets to compare first.

    A supplied baseline normalizes the measured values using
    ``compute_error_ratio`` before constructing the curves. This reads no
    Inputs, evaluates no Metrics, and leaves the supplied frame unchanged.
    With no baseline, values are used as supplied, including values already
    normalized by an explicit earlier call.

    Args:
        frame: Trial or :func:`aggregate_trials` rows, raw or from
            :func:`compute_error_ratio`, carrying ``algorithm``,
            ``instance``, and the named Metric column.
        thresholds: The thresholds to evaluate, in the units of the supplied
            Metric column, or as multiples of the supplied baseline.
        metric: The Metric or its named measurement column.
        instances: The Instances every curve averages over, defaulting to
            every Instance the frame carries. Rows on other Instances are left
            out.
        weights: An optional per-Instance frame carrying a ``weight`` column
            beside the keys it is keyed by. Instances weigh equally by default.
        baseline: A per-Instance value's name or a keyed denominator
            table accepted by ``compute_error_ratio``. ``None`` performs no
            normalization. Construct algorithm-derived tables explicitly with
            ``compute_baseline``; algorithm objects are not accepted here.

    Returns:
        One row per algorithm and threshold: ``algorithm``, its
        ``algorithm_label`` when the frame carries one, the ``metric`` name,
        ``tau``, and the weighted ``fraction`` within it. When normalized, the
        thresholds are baseline multiples; the ``metric`` column still names
        the selected raw metric. Plot labels remain an explicit caller choice.

    Raises:
        ValueError: If no threshold is given or one of them is NaN, if a curve
            has no trial on a listed Instance, if an Instance carries no
            weight, or if a curve's total weight is zero. Baseline errors,
            including missing coverage and double normalization, propagate
            from ``compute_error_ratio`` without silently dropping instances.
    """
    if baseline is not None:
        frame = compute_error_ratio(frame, baseline, metric=metric)
    metric = str(metric)
    grid = np.asarray(thresholds, dtype=float)
    if not grid.size:
        raise ValueError("Profile curves need at least one threshold.")
    if bool(np.isnan(grid).any()):
        raise ValueError(
            "A profile threshold must be a number; NaN names no threshold to "
            "take a share within."
        )
    frame = measured_rows(frame, metric)
    listed = list(dict.fromkeys(frame["instance"] if instances is None else instances))
    frame = frame.loc[frame["instance"].isin(listed)]
    keys = ["algorithm", *(["algorithm_label"] if "algorithm_label" in frame else [])]
    for _, group in frame.groupby(keys, dropna=False, sort=False):
        present = set(group["instance"])
        missing = [str(instance) for instance in listed if instance not in present]
        if missing:
            raise ValueError(
                f"The {group.iloc[0]['algorithm']} curve has no trial on "
                f"Instance(s) {', '.join(missing)}; pass instances= naming the "
                "Instances every curve covers."
            )

    prepared = (
        frame.assign(weight=1.0)
        if weights is None
        else _broadcast(frame, weights, "weight")
    )
    # Broadcasting preserves row order but replaces the index; align these
    # counts by position.
    runs = frame.groupby([*keys, "instance"], dropna=False, sort=False)[
        metric
    ].transform("size")
    # One trial's share of its own Instance, so summing a curve's trials is
    # already the mean over Instances of their within-Instance shares.
    prepared = prepared.assign(contribution=prepared["weight"] / runs.to_numpy())
    curves: list[pd.DataFrame] = []
    for _, group in prepared.groupby(keys, dropna=False, sort=False):
        ordered = group.sort_values(metric, kind="stable")
        values = ordered[metric].to_numpy(dtype=float)
        shares = ordered["contribution"].to_numpy(dtype=float)
        # The weighted CDF is a prefix sum; each threshold selects the prefix
        # ending after its ties rather than scanning and regrouping every trial.
        cumulative = np.concatenate(([0.0], np.cumsum(shares)))
        if cumulative[-1] == 0:
            raise ValueError(
                f"The {group.iloc[0]['algorithm']} curve has zero total " "weight."
            )
        # A positive-infinite cost lies outside every threshold, an infinite
        # threshold included; any other cost lies within each threshold it is <=.
        positions = np.searchsorted(values[values < np.inf], grid, side="right")
        curve = group[keys].iloc[np.zeros(len(grid), dtype=int)].reset_index(drop=True)
        curve["metric"] = metric
        curve["tau"] = grid
        curve["fraction"] = cumulative[positions] / cumulative[-1]
        curve["_threshold_order"] = np.arange(len(grid))
        curves.append(curve)
    if not curves:
        return prepared[keys].assign(
            metric=pd.Series(dtype=str),
            tau=pd.Series(dtype=float),
            fraction=pd.Series(dtype=float),
        )
    return (
        pd.concat(curves, ignore_index=True)
        .sort_values("_threshold_order", kind="stable")
        .drop(columns="_threshold_order")
        .reset_index(drop=True)
    )
