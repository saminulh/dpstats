"""Error ratios read against one public property of the Instance they ran on."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from dpstats.analysis._profiles import aggregate_trials, measured_rows
from dpstats.metric import Metric


def _weighted_quantile(values: ArrayLike, weights: ArrayLike, target: float) -> float:
    """The first value whose cumulative weight reaches ``target`` of the total.

    Values are read in ascending order; rounding that leaves no value
    reaching it gives the largest.
    """
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    ordered = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    reached = np.flatnonzero(cumulative >= target * cumulative[-1])
    return float(ordered[reached[0]] if reached.size else ordered[-1])


def compute_robust_range(
    values: Sequence[float], weights: Sequence[float], *, fence: float
) -> tuple[float, float]:
    """Tukey's fences around the weighted quartiles, cut to the data.

    The range runs from ``q1 - fence * (q3 - q1)`` to ``q3 + fence * (q3 - q1)``,
    narrowed to the smallest and largest value, so a few extreme values do not
    set it. A quartile is the first value whose cumulative weight reaches that
    share of the total.

    Args:
        values: Finite values, on the scale the range is read on.
        weights: Each value's nonnegative weight.
        fence: How many interquartile ranges each fence lies beyond its
            quartile; 3 gives Tukey's far-out fences.

    Returns:
        The lower and upper end of the range.

    Raises:
        ValueError: If there is no value.
    """
    if not len(values):
        raise ValueError("A robust range has no value to read.")
    lower = _weighted_quantile(values, weights, 0.25)
    upper = _weighted_quantile(values, weights, 0.75)
    spread = upper - lower
    return (
        max(lower - fence * spread, float(np.min(values))),
        min(upper + fence * spread, float(np.max(values))),
    )


class PropertyRanks:
    """Each property value's weighted mid-rank over every weighted Instance.

    For value v, its position is ``(W(< v) + W(= v) / 2) / W``: tied Instances
    share one position, and each tie group spans ``[W(< v), W(<= v)) / W``.
    """

    def __init__(self, feature: pd.Series, weights: pd.Series) -> None:
        weight_of: dict[object, float] = {
            uuid: float(share) for uuid, share in weights.items()
        }
        shares = [
            (float(value), weight_of[uuid])
            for uuid, value in feature.items()
            if not math.isnan(value) and uuid in weight_of
        ]
        total = float(np.sum([share for _, share in shares]))
        if total <= 0:
            raise ValueError("The weighted Instances carry no property value.")
        shares.sort(key=lambda entry: entry[0])
        values: list[float] = []
        positions: list[float] = []
        upper: list[float] = []
        held = 0.0
        start = 0
        while start < len(shares):
            stop, group = start, 0.0
            while stop < len(shares) and shares[stop][0] == shares[start][0]:
                group += shares[stop][1]
                stop += 1
            values.append(shares[start][0])
            positions.append((held + group / 2) / total)
            upper.append((held + group) / total)
            held += group
            start = stop
        self.values = np.asarray(values)
        self.upper = np.asarray(upper)
        self._position = dict(zip(values, positions))

    def position(self, value: float) -> float:
        """The position one value sits at, or NaN for a value no Instance has."""
        return self._position.get(value, math.nan)

    def value_at(self, target: float) -> float:
        """The value whose tie group's span holds ``target``; the last at 1."""
        found = np.flatnonzero(target < self.upper)
        return float(self.values[found[0] if found.size else -1])


def place_error_ratio_points(
    ratios: pd.DataFrame,
    *,
    metric: Metric | str,
    feature: pd.Series,
    weights: pd.Series,
    aggregation: Literal["individual", "median"],
) -> pd.DataFrame:
    """Place each error ratio beside one public property of its Instance.

    Args:
        ratios: :func:`compute_error_ratio` rows carrying ``algorithm``,
            ``instance`` and the named Metric column.
        metric: The Metric or its named measurement column.
        feature: The property's value per Instance UUID, NaN where an
            Instance has none. A point on such an Instance is left out.
        weights: The share each Instance carries, per Instance UUID, over
            every Instance the positions rank, not only those with points.
        aggregation: ``individual`` for one point per trial, ``median`` for
            one per algorithm and Instance, the median of its trials.

    Returns:
        One row per point, in Instance name then seed order: ``algorithm``,
        ``instance``, the property value
        ``feature``, its weighted mid-rank ``position`` in ``[0, 1]`` among
        every weighted Instance, its ``weight`` (its Instance's weight split
        evenly across that algorithm's points there) and its ``ratio``.

    Raises:
        ValueError: If ``aggregation`` is neither ``individual`` nor
            ``median``, or a point's Instance carries no weight.
    """
    metric = str(metric)
    if aggregation == "median":
        rows = aggregate_trials(ratios, metric=metric)
    elif aggregation == "individual":
        rows = measured_rows(ratios, metric)
    else:
        raise ValueError(
            f"aggregation must be 'individual' or 'median', not {aggregation!r}."
        )
    # A stable order, by Instance name then seed, fixes the order tied ratios
    # accumulate their weight in, so a weighted quantile whose cumulative
    # weight lands exactly on its target resolves deterministically.
    order = [str(instance) for instance in rows["instance"]]
    seeds = rows["alg_seed"].tolist() if "alg_seed" in rows.columns else [0] * len(rows)
    rows = rows.iloc[sorted(range(len(rows)), key=lambda at: (order[at], seeds[at]))]
    uuids = [instance.uuid for instance in rows["instance"]]
    values = feature.reindex(uuids).to_numpy(dtype=float)
    kept = ~np.isnan(values)
    rows, uuids, values = rows.loc[kept], np.asarray(uuids)[kept], values[kept]
    unweighted = sorted(set(uuids) - set(weights.index))
    if unweighted:
        raise ValueError(
            f"Instance(s) {', '.join(unweighted[:3])} carry no weight to place "
            "their points by."
        )
    ranks = PropertyRanks(feature, weights)
    algorithms = [algorithm.uuid for algorithm in rows["algorithm"]]
    shared = pd.Series(list(zip(algorithms, uuids))).map(
        pd.Series(list(zip(algorithms, uuids))).value_counts()
    )
    points = pd.DataFrame(
        {
            "algorithm": rows["algorithm"].to_numpy(),
            "instance": rows["instance"].to_numpy(),
            "feature": values,
            "position": [ranks.position(value) for value in values],
            "weight": weights.reindex(uuids).to_numpy(dtype=float)
            / shared.to_numpy(dtype=float),
            "ratio": rows[metric].to_numpy(dtype=float),
        }
    )
    if "algorithm_label" in rows.columns:
        points.insert(1, "algorithm_label", rows["algorithm_label"].to_numpy())
    return points


def trace_running_median(
    points: pd.DataFrame,
    *,
    feature: pd.Series,
    weights: pd.Series,
    window: float,
    positions: int,
) -> pd.DataFrame:
    """Trace each algorithm's weighted median and quartiles along the property.

    The positions are evenly spaced over ``[0, 1]`` in the property's
    weighted mid-rank space. Each reads the points within ``window / 2`` of it
    (one-sided at the ends), weighted as the points are, and draws nothing
    where those hold less than ``window / 2`` of the algorithm's weight. A
    quantile is the first ratio whose cumulative weight reaches that share,
    so a window dominated by failures reads infinite.

    Args:
        points: :func:`place_error_ratio_points` output.
        feature: The property's value per Instance UUID, as given to
            :func:`place_error_ratio_points`.
        weights: The share each Instance carries, per Instance UUID, as
            given to :func:`place_error_ratio_points`.
        window: The share of the weight each position reads, in ``(0, 1]``.
        positions: How many positions to read, at least two.

    Returns:
        One row per algorithm and position: ``algorithm``, its ``position``,
        the property value ``feature`` whose tie group holds it, and the
        ``median``, ``lower`` and ``upper`` quartile there, NaN where the
        window holds too little weight.

    Raises:
        ValueError: If ``positions`` is below two or ``window`` outside
            ``(0, 1]``.
    """
    if positions < 2:
        raise ValueError("A running median needs at least two positions.")
    if not 0 < window <= 1:
        raise ValueError("A running median's window must lie in (0, 1].")
    ranks = PropertyRanks(feature, weights)
    half = window / 2
    stops = np.arange(positions) / (positions - 1)
    rows: list[dict[str, object]] = []
    keys = [algorithm.uuid for algorithm in points["algorithm"]]
    for uuid in dict.fromkeys(keys):
        own = points.loc[[key == uuid for key in keys]]
        algorithm = own["algorithm"].iloc[0]
        at = own["position"].to_numpy(dtype=float)
        ratio = own["ratio"].to_numpy(dtype=float)
        weight = own["weight"].to_numpy(dtype=float)
        total = float(np.sum(weight))
        for stop in stops:
            inside = (at >= max(0.0, stop - half)) & (at <= min(1.0, stop + half))
            held = float(np.sum(weight[inside]))
            drawn = total > 0 and held / total >= half
            quartiles = (
                [
                    _weighted_quantile(ratio[inside], weight[inside], target)
                    for target in (0.5, 0.25, 0.75)
                ]
                if drawn
                else [math.nan] * 3
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "position": float(stop),
                    "feature": ranks.value_at(float(stop)),
                    "median": quartiles[0],
                    "lower": quartiles[1],
                    "upper": quartiles[2],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["algorithm", "position", "feature", "median", "lower", "upper"],
    )
