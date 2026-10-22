"""Public behavior of the profile plot."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from dpstats.analysis import plot_profiles
from tests.registry_components import SimpleAlgorithm

ORDINARY = SimpleAlgorithm(label="OLS")
ADAPTIVE = SimpleAlgorithm(label="AdaSSP", scale=2.0)
REFERENCE = SimpleAlgorithm(label="Ref", scale=3.0)


def _curves() -> pd.DataFrame:
    """OLS within 1 on 20% of trials, AdaSSP on 10%; within 10, 80% and 40%."""
    return pd.DataFrame(
        {
            "algorithm": [ORDINARY, ORDINARY, ADAPTIVE, ADAPTIVE],
            "algorithm_label": ["OLS", "OLS", "AdaSSP", "AdaSSP"],
            "tau": [1.0, 10.0, 1.0, 10.0],
            "fraction": [0.2, 0.8, 0.1, 0.4],
        }
    )


def _template(leading: int) -> dict[str, Any]:
    return {
        "layout": {
            "meta": {
                "series": {
                    ORDINARY.uuid: {"color": "#111111", "symbol": "square"},
                    ADAPTIVE.uuid: {"color": "#222222", "dash": "10 4"},
                    REFERENCE.uuid: {"color": "#000000", "muted": "#000000"},
                },
                "profiles": {
                    "leading": leading,
                    "lead": {"width": 2.5, "opacity": 1.0},
                    "context": {"color": "#888888", "width": 1.5, "opacity": 0.6},
                    "marker_size": 8,
                    "markers_shown": 6,
                    "x_range": [0, 5],
                    "pad": 1.05,
                    "guide": {"color": "#2c5d8f", "width": 2},
                },
            }
        }
    }


def test_plot_profiles_draws_one_line_per_algorithm_labelled_by_its_label() -> None:
    figure = plot_profiles(
        _curves(),
        xlabel="error / baseline",
        ylabel="share of trials",
        hover="τ {tau} · {share} within",
    )

    assert [trace.name for trace in figure.data] == ["OLS", "AdaSSP"]
    assert [list(trace.x) for trace in figure.data] == [[1.0, 10.0]] * 2
    assert [list(trace.y) for trace in figure.data] == [[0.2, 0.8], [0.1, 0.4]]
    assert [trace.mode for trace in figure.data] == ["lines", "lines"]
    assert figure.layout.yaxis.range is None
    assert figure.layout.xaxis.title.text == "error / baseline"
    assert figure.layout.yaxis.title.text == "share of trials"
    assert figure.data[0].hovertemplate == (
        "<b>%{fullData.name}</b><br>τ %{x:.3~g} · %{y:.1%} within<extra></extra>"
    )
    assert figure.layout.showlegend is True
    assert not figure.layout.shapes


def test_plot_profiles_names_its_axes_and_readings_without_being_told() -> None:
    """The curves alone give a log τ axis, a share axis, a reading and a guide."""
    figure = plot_profiles(_curves(), tau=1.0)

    assert figure.layout.xaxis.type == "log"
    assert figure.layout.xaxis.title.text == "τ"
    assert figure.layout.yaxis.title.text == "share of trials within τ"
    assert figure.data[0].hovertemplate == (
        "<b>%{fullData.name}</b><br>τ %{x:.3~g} · %{y:.1%} of trials<extra></extra>"
    )
    assert figure.layout.shapes[0].label.text == "τ = 1"
    assert not plot_profiles(_curves()).layout.shapes


def test_the_curves_leading_at_tau_are_drawn_in_their_own_look_above_the_rest() -> None:
    """At τ = 1 OLS (20%) leads AdaSSP (10%); with one leading curve, AdaSSP
    takes the context ink and is drawn first, beneath OLS."""
    figure = plot_profiles(_curves(), tau=1.0, template=_template(1))

    behind, ahead = figure.data
    assert (ahead.name, behind.name) == ("OLS", "AdaSSP")
    assert (ahead.legendrank, behind.legendrank) == (0, 1)
    assert ahead.line.color == ahead.marker.color == "#111111"
    assert (ahead.line.width, ahead.opacity, ahead.marker.symbol) == (
        2.5,
        1.0,
        "square",
    )
    assert behind.line.color == behind.marker.color == "#888888"
    assert (behind.line.width, behind.opacity, behind.line.dash) == (1.5, 0.6, "10 4")
    assert ahead.mode == "lines+markers"
    assert (ahead.marker.size, ahead.marker.maxdisplayed) == (8, 6)
    assert figure.layout.yaxis.range == (0, 1.05)


def test_the_ranking_reads_each_curve_at_its_last_point_within_tau() -> None:
    """At τ = 12 the shares are 80% and 40%, so OLS leads either way round;
    at τ = 0.5 both are 0, and the curves keep the order they came in."""
    reversed_order = _curves().iloc[::-1]
    at_twelve = plot_profiles(reversed_order, tau=12.0)
    below_grid = plot_profiles(_curves(), tau=0.5)
    below_grid_reversed = plot_profiles(reversed_order, tau=0.5)

    assert [trace.legendrank for trace in at_twelve.data] == [0, 1]
    assert [trace.name for trace in at_twelve.data] == ["OLS", "AdaSSP"]
    assert [trace.name for trace in below_grid.data] == ["OLS", "AdaSSP"]
    assert [trace.name for trace in below_grid_reversed.data] == ["AdaSSP", "OLS"]


def test_a_reference_keeps_its_own_ink_when_it_does_not_lead() -> None:
    curves = pd.concat(
        [
            _curves(),
            pd.DataFrame(
                {
                    "algorithm": [REFERENCE],
                    "algorithm_label": ["Ref"],
                    "tau": [1.0],
                    "fraction": [0.0],
                }
            ),
        ]
    )
    figure = plot_profiles(curves, tau=1.0, template=_template(1))

    reference = next(trace for trace in figure.data if trace.name == "Ref")
    assert reference.line.color == "#000000"
    assert reference.line.width == 1.5


def test_the_threshold_axis_opens_on_its_range_widened_to_reach_tau() -> None:
    """A linear axis opens on [0, 5]; τ = 10 widens it to 10 × 1.05."""
    within = plot_profiles(_curves(), xscale="linear", tau=1.0, template=_template(2))
    beyond = plot_profiles(_curves(), xscale="linear", tau=10.0, template=_template(2))

    assert within.layout.xaxis.range == (0, 5)
    assert beyond.layout.xaxis.range == (0, 10.5)
    (guide,) = beyond.layout.shapes
    assert (guide.x0, guide.x1, guide.yref, guide.y0, guide.y1) == (
        10.0,
        10.0,
        "paper",
        0,
        1,
    )
    assert (guide.line.color, guide.line.width) == ("#2c5d8f", 2)
    assert guide.label.text == "τ = 10"
    assert within.layout.shapes[0].label.text == "τ = 1"


def test_plot_profiles_keeps_equally_labelled_algorithms_apart() -> None:
    first = SimpleAlgorithm(label="same")
    second = SimpleAlgorithm(label="same", scale=2.0)
    curves = pd.DataFrame(
        {
            "algorithm": [first, second],
            "algorithm_label": ["same", "same"],
            "tau": [1.0, 1.0],
            "fraction": [0.5, 1.0],
        }
    )

    figure = plot_profiles(curves)

    assert [list(trace.y) for trace in figure.data] == [[0.5], [1.0]]


def test_raw_profile_keeps_zero_and_caller_units() -> None:
    curves = _curves().assign(tau=[0.0, 1.0, 0.0, 1.0])
    figure = plot_profiles(curves, xscale="linear", xlabel="sign disagreement")

    assert figure.layout.xaxis.type == "linear"
    assert figure.layout.xaxis.title.text == "sign disagreement"
    assert list(figure.data[0].x) == [0.0, 1.0]
    assert list(figure.data[0].y) == [0.2, 0.8]


def test_profile_plot_refuses_what_it_cannot_draw() -> None:
    with pytest.raises(ValueError, match="positive thresholds"):
        plot_profiles(_curves().assign(tau=0.0))
    with pytest.raises(ValueError, match="algorithm_label"):
        plot_profiles(_curves().drop(columns="algorithm_label"))


def test_without_a_series_any_value_names_a_curve() -> None:
    """Curves grouped under plain names draw without a template."""
    curves = _curves().assign(algorithm=["a", "a", "b", "b"])

    figure = plot_profiles(curves, tau=1.0)

    assert [trace.name for trace in figure.data] == ["OLS", "AdaSSP"]
