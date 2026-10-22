"""Outcome counts over a Results frame, whole or binned by an Instance property."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from dpstats.analysis._error_ratios import PropertyRanks
from dpstats.analysis._profiles import (
    aggregate_trials,
    compute_error_ratio,
)
from dpstats.metric import Metric
from dpstats.results import ResultStatus


def summarize_outcomes(results: pd.DataFrame, *, by: Sequence[str]) -> pd.DataFrame:
    """Report what share of Results each outcome took, group by group.

    Every recorded outcome appears in every group, with a zero count where no
    Result took it, and each share is over every Result in its group.

    Args:
        results: Results rows carrying ``status`` and the ``by`` columns, as
            a task collection returns them.
        by: Columns to break the counts down by; an empty sequence counts
            every Result together.

    Returns:
        One row per group and outcome: the ``by`` columns, ``status``,
        ``count``, the group's ``denominator``, and ``share``.

    Raises:
        ValueError: If there are no Results, a column is absent or named
            twice, or a row carries an unknown status.

    Example:
        >>> results = pd.DataFrame(
        ...     {"algorithm_label": ["a", "a"], "status": ["success", "algorithm_error"]}
        ... )
        >>> outcomes = summarize_outcomes(results, by=["algorithm_label"])
        >>> outcomes.loc[outcomes["status"] == "success", "share"].item()
        0.5
    """
    columns = list(by)
    if len(set(columns)) < len(columns):
        raise ValueError("Outcome groups name a column more than once.")
    missing = [
        column for column in ("status", *columns) if column not in results.columns
    ]
    if missing:
        raise ValueError(f"Results frame is missing column(s): {', '.join(missing)}.")
    if results.empty:
        raise ValueError("Outcome counts need at least one Result.")
    vocabulary = [status.value for status in ResultStatus]
    unrecorded = sorted(set(results["status"]) - set(vocabulary))
    if unrecorded:
        raise ValueError(f"Unrecorded Result status(es): {', '.join(unrecorded)}.")

    # Values rank by first appearance in the whole frame, nested by column, so
    # they need not be orderable.
    ranks = pd.DataFrame(
        {
            column: pd.factorize(results[column], use_na_sentinel=False)[0]
            for column in columns
        },
        index=results.index,
    )
    recorded = results.loc[:, columns].assign(
        _group=ranks.groupby(columns, sort=True).ngroup() if columns else 0,
        status=results["status"].astype(pd.CategoricalDtype(vocabulary)),
    )
    counts = (
        recorded.groupby(["_group", "status"], observed=False, sort=True)
        .size()
        .reset_index(name="count")
    )
    counts["denominator"] = counts.groupby("_group")["count"].transform("sum")
    counts["share"] = counts["count"] / counts["denominator"]
    keys = (
        recorded.drop_duplicates("_group")
        .set_index("_group")
        .sort_index()
        .loc[:, columns]
    )
    return keys.join(counts.set_index("_group")).reset_index(drop=True)


# One algorithm's outcome categories: its successes within and beyond the
# threshold, then every other recorded outcome.
_OUTCOMES = [
    "success_within",
    "success_beyond",
    *(status.value for status in ResultStatus if status is not ResultStatus.SUCCESS),
]

# Two algorithms' pair categories: both within the threshold, each alone, and
# neither.
_PAIRS = ["both", "a_only", "b_only", "neither"]


def bin_outcomes(
    trials: pd.DataFrame,
    baseline: pd.DataFrame | str,
    *,
    metric: Metric | str,
    tau: float,
    algorithm: str,
    compare_with: str | None,
    feature: pd.Series,
    weights: pd.Series,
    instances: Sequence[str],
    scale: Literal["log", "linear", "quantile"],
    bins: int,
    aggregation: Literal["individual", "median"],
) -> pd.DataFrame:
    """Split each selected Instance's weight into outcomes, binned by a property.

    A ratio is within the threshold when ``tau`` is positive and finite and
    the ratio is at most ``tau``. One algorithm's Instance weight is split
    evenly across its trials, each taking its own outcome; under ``median``
    it goes whole to within or beyond by the median ratio, unless the median
    cost is infinite, when it is split by its trials' outcomes, a success
    counting as beyond. Compared with a second algorithm, an Instance is
    split by the two algorithms' own success rates ``p_a`` and ``p_b`` (the
    share of trials within, or 1 or 0 as the median is within under
    ``median``), never matching one's trials to the other's: ``both`` takes
    ``min(p_a, p_b)``, ``a_only`` and ``b_only`` each one's excess over that,
    and ``neither`` the rest.

    Bins are ``bins`` equal spans of the placed values on a ``linear`` or
    ``log`` scale, or ``bins`` equal shares of the weight in the property's
    weighted mid-rank space on a ``quantile`` scale, each inner edge moved to
    the nearest tie group's upper end so tied values share a bin. Bins are
    right-open; the last also holds its upper edge.

    Args:
        trials: Measurement rows carrying ``algorithm``, ``instance``,
            ``status`` and the named Metric column, as costs.
        baseline: What each cost is read against, as
            :func:`compute_error_ratio` takes it. A trial on an Instance a
            baseline table does not cover is left out.
        metric: The Metric or its named measurement column.
        tau: The threshold a ratio is read at.
        algorithm: The algorithm read, by UUID.
        compare_with: The algorithm it is compared with, by UUID, or ``None``.
        feature: The property's value per Instance UUID, NaN where an
            Instance has none.
        weights: The share each Instance carries, per Instance UUID, over
            every Instance a ``quantile`` scale ranks.
        instances: The selected Instances, by UUID.
        scale: How the property is placed: by its logarithm (positive values
            only), its value, or its weighted mid-rank position.
        bins: How many bins.
        aggregation: ``individual`` to read every trial, ``median`` to read
            each algorithm's median on each Instance.

    Returns:
        One row per bin: ``start`` and ``stop``, its edges where the axis
        places them (the property value, or the position on a ``quantile``
        scale); ``lower`` and ``upper``, the property values there; one
        column per category (the outcomes, or ``both``, ``a_only``,
        ``b_only`` and ``neither`` when compared) holding its share of the
        selected Instances' weight; and ``instances``, the UUIDs it holds.
        The weight no bin holds (an Instance with no placeable value, no
        baseline, or without a trial of each algorithm read) is ``1`` minus
        every share.
    """
    metric = str(metric)
    read = [algorithm] if compare_with is None else [algorithm, compare_with]
    rows = trials[[entry.uuid in read for entry in trials["algorithm"]]]
    if isinstance(baseline, pd.DataFrame) and "instance" in baseline.columns:
        covered = {instance.uuid for instance in baseline["instance"]}
        rows = rows[[instance.uuid in covered for instance in rows["instance"]]]
    ratios = compute_error_ratio(rows, baseline, metric=metric)
    trials_of: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for entry, instance, status, ratio in zip(
        ratios["algorithm"], ratios["instance"], ratios["status"], ratios[metric]
    ):
        key = (entry.uuid, instance.uuid)
        trials_of.setdefault(key, []).append((str(status), float(ratio)))
    middles = aggregate_trials(rows, metric=metric)
    # The median ratio, or None where the median cost is infinite.
    median_of = {
        (entry.uuid, instance.uuid): (
            float(ratio) if math.isfinite(float(cost)) else None
        )
        for entry, instance, cost, ratio in zip(
            middles["algorithm"],
            middles["instance"],
            middles[metric],
            compute_error_ratio(middles, baseline, metric=metric)[metric],
        )
    }

    def within(ratio: float | None) -> bool:
        return ratio is not None and 0 < tau < math.inf and ratio <= tau

    def outcomes(uuid: str) -> dict[str, float] | None:
        own = trials_of.get((algorithm, uuid))
        if own is None:
            return None
        if aggregation == "median" and median_of[(algorithm, uuid)] is not None:
            middle = median_of[(algorithm, uuid)]
            return {"success_within" if within(middle) else "success_beyond": 1.0}
        split: dict[str, float] = {}
        for status, ratio in own:
            if status != ResultStatus.SUCCESS.value:
                key = status
            elif aggregation == "individual" and within(ratio):
                key = "success_within"
            else:
                key = "success_beyond"
            split[key] = split.get(key, 0.0) + 1 / len(own)
        return split

    def success_rate(owner: str, uuid: str) -> float | None:
        own = trials_of.get((owner, uuid))
        if own is None:
            return None
        if aggregation == "median":
            return float(within(median_of[(owner, uuid)]))
        hits = [
            status == ResultStatus.SUCCESS.value and within(ratio)
            for status, ratio in own
        ]
        return sum(hits) / len(hits)

    def pairs(uuid: str) -> dict[str, float] | None:
        rate_a = success_rate(algorithm, uuid)
        rate_b = success_rate(str(compare_with), uuid)
        if rate_a is None or rate_b is None:
            return None
        both = min(rate_a, rate_b)
        return {
            "both": both,
            "a_only": rate_a - both,
            "b_only": rate_b - both,
            "neither": 1 - max(rate_a, rate_b),
        }

    selected = list(instances)
    values = feature.reindex(selected).to_numpy(dtype=float)
    if scale == "quantile":
        ranks = PropertyRanks(feature, weights)
        coordinates = np.array([ranks.position(value) for value in values])
        placed = ~np.isnan(coordinates)
        nearest = [
            float(ranks.upper[np.argmin(np.abs(ranks.upper - k / bins))])
            for k in range(1, bins)
        ]
        edges = np.array([0.0, *nearest, 1.0])
        bounds = np.array([ranks.value_at(float(edge)) for edge in edges])
        axis = edges
    else:
        if scale == "log":
            placed = (values > 0) & np.isfinite(values)
            coordinates = np.log10(
                values, where=placed, out=np.full(values.shape, np.nan)
            )
        else:
            placed = np.isfinite(values)
            coordinates = values
        low, high = (
            (coordinates[placed].min(), coordinates[placed].max())
            if placed.any()
            else (0.0, 1.0)
        )
        edges = low + np.arange(bins + 1) * ((high - low) / bins or 1.0)
        bounds = 10**edges if scale == "log" else edges
        axis = bounds

    keys = _OUTCOMES if compare_with is None else _PAIRS
    total = float(weights.reindex(selected).sum()) or 1.0
    shares = np.zeros((bins, len(keys)))
    members: list[list[str]] = [[] for _ in range(bins)]
    for uuid, coordinate, weight in zip(
        np.asarray(selected)[placed],
        coordinates[placed],
        weights.reindex(selected).to_numpy(dtype=float)[placed],
    ):
        split = outcomes(uuid) if compare_with is None else pairs(uuid)
        if split is None:
            continue
        at = int(np.searchsorted(edges[1:-1], coordinate, side="right"))
        members[at].append(str(uuid))
        for key, share in split.items():
            shares[at, keys.index(key)] += share * weight / total

    frame = pd.DataFrame(shares, columns=keys)
    frame.insert(0, "start", axis[:-1])
    frame.insert(1, "stop", axis[1:])
    frame.insert(2, "lower", bounds[:-1])
    frame.insert(3, "upper", bounds[1:])
    frame["instances"] = members
    return frame
