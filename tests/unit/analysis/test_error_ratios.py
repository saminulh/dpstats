"""Public behavior of the error-ratio reading, against hand-computed oracles."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd
import pytest

from dpstats.analysis import (
    place_error_ratio_points,
    plot_error_ratios,
    compute_robust_range,
    trace_running_median,
)
from dpstats.budgets import PureDPBudget
from dpstats.instance import Instance
from tests.registry_components import SimpleAlgorithm, SimpleMetadata


def test_the_robust_range_is_the_far_out_fences_cut_to_the_data() -> None:
    """Quartiles 1 and 4 put the k = 1 fences at -2 and 7; the data start at
    0, so the range is [0, 7] and 100 lies beyond it."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0]
    weights = [1.0] * 6
    # Cumulative weight 1..6 of 6: the first value reaching a quarter is the
    # second (2/6 >= 1/4), and three quarters the fifth (5/6 >= 3/4).
    assert compute_robust_range(values, weights, fence=1.0) == pytest.approx((0.0, 7.0))


def test_the_robust_range_never_reaches_past_the_data() -> None:
    """Wide fences stop at the smallest and largest value."""
    assert compute_robust_range([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], fence=3.0) == (
        1.0,
        3.0,
    )


def test_weight_moves_the_quartiles() -> None:
    """A value carrying most of the weight is both quartiles, so a zero
    interquartile range keeps nothing but that value."""
    assert compute_robust_range([1.0, 5.0, 9.0], [1.0, 10.0, 1.0], fence=3.0) == (
        5.0,
        5.0,
    )


def test_an_empty_sample_has_no_range() -> None:
    with pytest.raises(ValueError, match="no value"):
        compute_robust_range([], [], fence=3.0)


# ── Points beside an instance property ──────────────────────────────────────


def _instance(name: str) -> Instance:
    return Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(name.encode()).hexdigest(),
        metric_arguments={},
        label=name,
    )


ALPHA = SimpleAlgorithm(label="Alpha")
BETA = SimpleAlgorithm(label="Beta", scale=2.0)
SMALL, LARGE, UNMEASURED = _instance("small"), _instance("large"), _instance("none")
# The property each Instance sits at, and the share it carries: the small one
# a quarter of the weight, the large three quarters, the third none of either
# since it has no value.
FEATURE = pd.Series({SMALL.uuid: 10.0, LARGE.uuid: 30.0, UNMEASURED.uuid: math.nan})
WEIGHTS = pd.Series({SMALL.uuid: 1.0, LARGE.uuid: 3.0, UNMEASURED.uuid: 1.0})


def _ratios(*rows: tuple[SimpleAlgorithm, Instance, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "toy",
                "algorithm": algorithm,
                "budget": PureDPBudget(eps=1.0),
                "instance": instance,
                "alg_seed": seed,
                "cost": ratio,
            }
            for seed, (algorithm, instance, ratio) in enumerate(rows)
        ]
    )


RATIOS = _ratios(
    (ALPHA, SMALL, 0.5),
    (ALPHA, SMALL, 2.0),
    (ALPHA, LARGE, 4.0),
    (ALPHA, UNMEASURED, 1.0),
    (BETA, SMALL, math.inf),
)


def _by_point(
    points: pd.DataFrame,
) -> list[tuple[str, str, float, float, float, float]]:
    return sorted(
        (str(a), str(i), f, q, w, r)
        for a, i, f, q, w, r in zip(
            points["algorithm"],
            points["instance"],
            points["feature"],
            points["position"],
            points["weight"],
            points["ratio"],
        )
    )


def test_each_trial_is_one_point_at_its_instances_property() -> None:
    """Weighted mid-ranks: the small value holds [0, 1/4) of the weight and
    sits at 1/8, the large [1/4, 1) at 5/8. An Instance's weight splits evenly
    across one algorithm's points on it; an Instance with no value is left out."""
    points = place_error_ratio_points(
        RATIOS,
        metric="cost",
        feature=FEATURE,
        weights=WEIGHTS,
        aggregation="individual",
    )

    assert _by_point(points) == [
        ("Alpha", "large", 30.0, 0.625, 3.0, 4.0),
        ("Alpha", "small", 10.0, 0.125, 0.5, 0.5),
        ("Alpha", "small", 10.0, 0.125, 0.5, 2.0),
        ("Beta", "small", 10.0, 0.125, 1.0, math.inf),
    ]


def test_the_median_is_one_point_per_algorithm_and_instance() -> None:
    """The middle of 0.5 and 2 is 1.25, and it carries the Instance's whole weight."""
    points = place_error_ratio_points(
        RATIOS, metric="cost", feature=FEATURE, weights=WEIGHTS, aggregation="median"
    )

    assert _by_point(points) == [
        ("Alpha", "large", 30.0, 0.625, 3.0, 4.0),
        ("Alpha", "small", 10.0, 0.125, 1.0, 1.25),
        ("Beta", "small", 10.0, 0.125, 1.0, math.inf),
    ]


def test_a_point_on_an_unweighted_instance_is_refused() -> None:
    with pytest.raises(ValueError, match="no weight"):
        place_error_ratio_points(
            RATIOS,
            metric="cost",
            feature=FEATURE,
            weights=WEIGHTS.drop(LARGE.uuid),
            aggregation="individual",
        )


# ── The running median ──────────────────────────────────────────────────────


def _line(algorithm: SimpleAlgorithm, line: pd.DataFrame) -> list[Any]:
    """One algorithm's rows, each value approximate, NaN matching NaN."""
    own = line.loc[[value.uuid == algorithm.uuid for value in line["algorithm"]]]
    columns = ["position", "feature", "median", "lower", "upper"]
    return [
        [pytest.approx(value, nan_ok=True) for value in row]
        for row in own[columns].itertuples(index=False)
    ]


def test_the_running_median_reads_each_window_of_weight() -> None:
    """Three positions, each over half the weight: at 0 the small Instance's
    two points (a quarter of the weight, the least a window may hold), at 1/2
    the large one's, and at 1 nothing, which draws no value. The median of 0.5
    and 2, each carrying half, is the first to reach half: 0.5."""
    points = place_error_ratio_points(
        RATIOS,
        metric="cost",
        feature=FEATURE,
        weights=WEIGHTS,
        aggregation="individual",
    )

    line = trace_running_median(
        points, feature=FEATURE, weights=WEIGHTS, window=0.5, positions=3
    )

    nan = math.nan
    assert [
        [0.0, 10.0, 0.5, 0.5, 2.0],
        [0.5, 30.0, 4.0, 4.0, 4.0],
        [1.0, 30.0, nan, nan, nan],
    ] == _line(ALPHA, line)
    # A failure is an infinite ratio, so a window of failures reads infinite.
    infinite = math.inf
    assert [
        [0.0, 10.0, infinite, infinite, infinite],
        [0.5, 30.0, nan, nan, nan],
        [1.0, 30.0, nan, nan, nan],
    ] == _line(BETA, line)


# ── The figure ──────────────────────────────────────────────────────────────

# The figure's texts, as a caller writes them.
TEXTS = {
    "feature": "n",
    "xlabel": "n",
    "ylabel": "Error / baseline",
    "point_hover": "{property} {x} · ratio {y}",
    "running_median": "{algorithm}: running median",
}
STYLE = {
    "layout": {
        "font": {"color": "#222222"},
        "meta": {
            "ink": "#222222",
            "series": {
                ALPHA.uuid: {"color": "#0085da", "symbol": "square", "name": "Alpha"}
            },
            "marker": {
                "size": 6,
                "size_span": 20,
                "opacity": 0.3,
                "edge_opacity": 0.65,
                "ring_width": 2,
            },
            "running_median": {"width": 2.5, "band_opacity": 0.25},
            "significant_digits": 3,
            "reference_line": {"width": 1, "dash": "dash"},
            "edge_count": {"at": 0.98, "shift": 15},
        },
    }
}


def _point(
    instance: Instance, feature: float, position: float, ratio: float
) -> dict[str, Any]:
    return {
        "algorithm": ALPHA,
        "instance": instance,
        "feature": feature,
        "position": position,
        "weight": 1.0,
        "ratio": ratio,
    }


# Two trials with one ratio on one Instance and a third on another Instance at
# the same spot, one point inside the view, one far beyond its top and one
# failure; every point weighs the same.
POINTS = pd.DataFrame(
    [
        _point(SMALL, 10.0, 0.125, 1.0),
        _point(SMALL, 10.0, 0.125, 1.0),
        _point(UNMEASURED, 10.0, 0.125, 1.0),
        _point(LARGE, 30.0, 0.625, 2.0),
        _point(LARGE, 30.0, 0.625, 1000.0),
        _point(LARGE, 30.0, 0.625, math.inf),
    ]
)
LINE = pd.DataFrame(
    {
        "algorithm": [ALPHA] * 3,
        "position": [0.0, 0.5, 1.0],
        "feature": [10.0, 30.0, 30.0],
        "median": [1.0, 2.0, math.nan],
        "lower": [1.0, 1.0, math.nan],
        "upper": [2.0, math.inf, math.nan],
    }
)


def _figure(**overrides: Any) -> Any:
    return plot_error_ratios(POINTS, LINE, **{**TEXTS, "template": STYLE, **overrides})


def _markers(figure: Any) -> dict[str, Any]:
    return {
        trace.marker.symbol: trace
        for trace in figure.data
        if trace.mode == "markers" and trace.marker.symbol is not None
    }


def test_the_opening_view_holds_the_robust_range_of_the_ratios() -> None:
    """On log10 scale the ratios are 0, 0, 0, 0.301 and 3 (the failure has no
    place): quartiles 0 and 0.301, so the top fence 0.301 + 3 * 0.301 = 1.204
    leaves out 1000. Padding is 5% of the span; the property axis fits log10
    10 and 30 with 8% padding."""
    figure = _figure()
    top = 4 * math.log10(2)
    assert figure.layout.yaxis.range == pytest.approx((-0.05 * top, 1.05 * top))
    low, high = 1.0, math.log10(30)
    pad = 0.08 * (high - low)
    assert figure.layout.xaxis.range == pytest.approx((low - pad, high + pad))
    assert (figure.layout.xaxis.type, figure.layout.yaxis.type) == ("log", "log")


def test_trials_with_one_ratio_on_one_instance_are_one_point() -> None:
    """The two equal trials on the small Instance are one point; with the
    third Instance's at the same spot, that marker stands for two points, the
    busiest, and covers three trials."""
    inside = _markers(_figure())["square"]
    assert list(inside.x) == [10.0, 30.0] and list(inside.y) == [1.0, 2.0]
    assert [list(data) for data in inside.customdata] == [[2, 3], [1, 1]]
    assert list(inside.marker.size) == [26.0, 6.0]
    assert inside.marker.color == "#0085da" and inside.marker.opacity == 0.3
    assert inside.hovertext[0] == "n 10 · ratio 1<br>3 trials"


def test_ratios_beyond_the_view_sit_on_its_edge_and_are_counted() -> None:
    """1000 is a filled dot on the top edge, the failure an open ring there,
    and the edge counts both trials in the algorithm's colour."""
    figure = _figure()
    markers = _markers(figure)
    above = markers["circle"]
    assert (list(above.x), list(above.y), above.yaxis) == ([30.0], [1.0], "y2")
    assert above.marker.opacity == 0.65
    assert above.hovertext[0] == "n 30 · ratio 1,000<br>beyond the view<br>1 trial"
    failed = markers["circle-open"]
    assert (list(failed.x), list(failed.y), failed.yaxis) == ([30.0], [1.0], "y2")
    assert failed.hovertext[0] == "n 30 · ratio ∞<br>1 trial"
    counts = {
        (note.x, note.y, note.yshift): note.text for note in figure.layout.annotations
    }
    assert counts == {(0.98, 1, -15): '<span style="color:#0085da">2</span>'}
    reference = figure.data[0]
    assert (reference.line.dash, reference.line.width) == ("dash", 1)


def test_the_running_median_draws_where_its_window_holds_weight() -> None:
    """The line follows the property value at each position; a position with
    no value is a gap, and a quartile beyond the view sits on its edge."""
    figure = _figure()
    named = [trace for trace in figure.data if trace.name == "Alpha: running median"]
    assert len(named) == 1
    assert list(named[0].x) == [10.0, 30.0, None]
    assert list(named[0].y) == [1.0, 2.0, None]
    upper = figure.data[figure.data.index(named[0]) - 2]
    assert list(upper.y)[1] == pytest.approx(10 ** (1.05 * 4 * math.log10(2)))
    band = figure.data[figure.data.index(named[0]) - 1]
    assert band.fillcolor == "rgba(0, 133, 218, 0.25)"


def test_the_axes_take_their_titles() -> None:
    figure = _figure()
    assert figure.layout.xaxis.title.text == "n"
    assert figure.layout.yaxis.title.text == "Error / baseline"
    assert _figure(xlabel="n (weighted)").layout.xaxis.title.text == "n (weighted)"


def test_the_figure_reads_without_any_text_given() -> None:
    """Every text has a default."""
    figure = plot_error_ratios(POINTS, LINE)
    assert figure.layout.xaxis.title.text == "property"
    assert figure.layout.yaxis.title.text == "error ratio"
    assert figure.layout.xaxis.type == figure.layout.yaxis.type == "log"
    markers = next(trace for trace in figure.data if trace.mode == "markers")
    assert markers.hovertext[0] == "property 10 · error ratio 1<br>3 trials"
    medians = [
        trace.name for trace in figure.data if trace.name and trace.mode == "lines"
    ]
    assert medians and all(name.endswith(" running median") for name in medians)


def test_a_quantile_property_axis_places_points_by_position() -> None:
    figure = _figure(x_scale="quantile")
    assert list(_markers(figure)["square"].x) == [0.125, 0.625]
    assert figure.layout.xaxis.type == "linear"
    assert figure.layout.xaxis.ticktext[0] == "10"


def test_a_linear_ratio_axis_starts_at_zero() -> None:
    """Linear ratios 1, 1, 2, 1000: quartiles 1 and 2, so the top fence is 5."""
    figure = _figure(y_scale="linear")
    assert figure.layout.yaxis.range == pytest.approx((-0.25, 5.25))


def test_without_a_template_the_figure_is_plain_plotly() -> None:
    figure = _figure(template=None)
    assert len([trace for trace in figure.data if trace.mode == "markers"]) == 3
    assert figure.data[-1].hovertext[0] == "n 30 · ratio ∞<br>1 trial"


def test_a_hover_rounds_to_the_templates_significant_digits() -> None:
    """Three digits: 1025 is a tie and rounds away from zero to 1,030, and
    328.701... reads 329."""
    one = pd.DataFrame([{**_point(SMALL, 1025.0, 0.5, 328.70106315668295)}])
    line = LINE.assign(feature=[1025.0] * 3)
    figure = plot_error_ratios(one, line, **TEXTS, template=STYLE)
    assert _markers(figure)["square"].hovertext[0] == "n 1,030 · ratio 329<br>1 trial"


def test_a_hover_writes_the_templates_separators() -> None:
    """A German reader's separators: 1,030 reads 1.030 and 0.125 reads 0,125."""
    one = pd.DataFrame([{**_point(SMALL, 1025.0, 0.5, 0.125)}])
    german = {
        "layout": {
            "meta": {
                **STYLE["layout"]["meta"],
                "number_separators": {"decimal": ",", "group": "."},
            }
        }
    }
    figure = plot_error_ratios(
        one, LINE.assign(feature=[1025.0] * 3), **TEXTS, template=german
    )
    assert _markers(figure)["square"].hovertext[0] == "n 1.030 · ratio 0,125<br>1 trial"
