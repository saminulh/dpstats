"""Notebook plots for analysis outputs."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Literal, NamedTuple

import pandas as pd
import plotly.colors
import plotly.graph_objects as go

from dpstats.analysis._error_ratios import compute_robust_range


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True)
class NumberStyle:
    """How numbers are written: rounding and the reader's separators.

    Attributes:
        significant: Digits a value is rounded to, or ``None`` for the
            shortest form that reads back as it.
        decimal: The decimal separator of a rounded value.
        group: The thousands separator of a rounded value or a count.
    """

    significant: int | None = None
    decimal: str = "."
    group: str = ","

    def grouped(self, text: str) -> str:
        """``text`` written with ``,`` grouping and a ``.`` decimal, in this style."""
        return text.translate(str.maketrans({",": self.group, ".": self.decimal}))


def _rounded(value: float, significant: int) -> float:
    """``value`` to ``significant`` digits, a tie rounding away from zero."""
    exact = Decimal(value)
    place = exact.adjusted() - significant + 1
    return float(exact.quantize(Decimal(1).scaleb(place), rounding=ROUND_HALF_UP))


def _shortest(value: float) -> str:
    """The shortest form that reads back as ``value``."""
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    mantissa, _, exponent = repr(value).partition("e")
    return mantissa + ("e" + str(int(exponent)) if exponent else "")


def number_text(value: float, style: NumberStyle) -> str:
    """One value, ``∞`` for infinity.

    With significant digits it is rounded to that many, grouped by thousands
    with a true minus sign, and from a million up written as a power of ten
    whose mantissa keeps a ``.``; without, in the shortest form that reads
    back as it.
    """
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    significant = style.significant
    if significant is None:
        return _shortest(value)
    rounded = _rounded(value, significant)
    if abs(rounded) < 1e6:
        whole, _, fraction = format(Decimal(repr(rounded)), "f").partition(".")
        fraction = fraction.rstrip("0")
        text = f"{abs(int(whole)):,}" + ("." + fraction if fraction else "")
        return ("−" if whole.startswith("-") else "") + style.grouped(text)
    exponent = math.floor(math.log10(abs(rounded)))
    mantissa = _rounded(rounded / 10**exponent, significant)
    if abs(mantissa) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    power = "10" + str(exponent).translate(_SUPERSCRIPTS)
    if mantissa == 1:
        return power
    return _shortest(mantissa).replace("-", "−") + " × " + power


def span_text(values: list[float], style: NumberStyle) -> str:
    """One value, or the span from the smallest to the largest of several."""
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values) or not finite:
        return number_text(values[0], style)
    low, high = min(finite), max(finite)
    if low == high:
        return number_text(low, style)
    return f"{number_text(low, style)} – {number_text(high, style)}"


def count_text(count: int, style: NumberStyle) -> str:
    """A count, grouped by thousands."""
    return style.grouped(f"{count:,}")


def tick_text(value: float, style: NumberStyle) -> str:
    """An axis tick's value: a whole number below a million in full."""
    if math.isfinite(value) and value == int(value) and abs(value) < 1e6:
        return count_text(int(value), style)
    return number_text(value, style)


def _share_within(curve: pd.DataFrame, tau: float) -> float:
    """A curve's share within ``tau``: its last point at or below it."""
    within = curve["fraction"][curve["tau"] <= tau]
    return float(within.iloc[-1]) if len(within) else 0.0


def plot_profiles(
    curves: pd.DataFrame,
    *,
    xlabel: str = "τ",
    ylabel: str = "share of trials within τ",
    xscale: Literal["linear", "log"] = "log",
    hover: str = "τ {tau} · {share} of trials",
    guide: str = "τ = {tau}",
    tau: float | None = None,
    template: Mapping[str, Any] | None = None,
) -> Any:
    """Draw one line per algorithm, labelled by its ``algorithm_label``.

    Given ``tau``, a guide marks that threshold and the curves are ranked by
    their share within it; equal shares keep the order the curves come in.
    The leading curves are drawn at full strength above the rest, and the
    legend lists every curve in rank order.

    A template's ``layout.meta`` may carry the look. ``series`` gives each
    algorithm, by UUID, its ``color``, ``symbol`` and ``dash``, and
    ``muted``, the colour it keeps when it does not lead. ``profiles`` gives:

    - ``leading``: how many curves lead; every curve leads when it is absent;
    - ``lead`` and ``context``: the ``width`` and ``opacity`` of a leading
      curve and of any other, and ``color``, the other curves' colour;
    - ``marker_size`` and ``markers_shown``: each curve's marker size, and
      how many of its markers show;
    - ``x_range``: the range a linear threshold axis opens on;
    - ``pad``: how far each axis reaches past what it must show, the share
      axis past 1 and the threshold axis past ``tau``;
    - ``guide``: the guide's line.

    With a ``profiles`` look the curves are drawn as lines with markers,
    without one as plain lines. What the template leaves out is Plotly's own.

    Args:
        curves: :func:`~dpstats.analysis.compute_profile_curves`
            output carrying ``algorithm``, ``algorithm_label``, ``tau`` and
            ``fraction``. Each distinct ``algorithm`` value is one curve; a
            template's ``series`` finds it by its ``uuid``.
        xlabel: The threshold axis's title. Name the Metric or
            baseline-relative units the thresholds are in; the default names
            only τ.
        ylabel: The share axis's title: what the share counts.
        xscale: Linear or logarithmic threshold axis. Logarithmic axes require
            strictly positive thresholds; use linear for thresholds including zero.
        hover: A curve's reading under the pointer, whose ``{tau}`` and
            ``{share}`` fields the figure fills.
        guide: The guide's label, whose ``{tau}`` field the figure fills.
        tau: The threshold the curves are read at, or ``None`` for none.
        template: A Plotly template, as a mapping, applied to the figure;
            ``None`` keeps Plotly's default template.

    Returns:
        The ``plotly.graph_objects.Figure``, one trace per algorithm.

    Raises:
        ValueError: If the curves carry no ``algorithm_label``, or a
            logarithmic axis would hide nonpositive thresholds.
    """
    if "algorithm_label" not in curves.columns:
        raise ValueError(
            "Profile curves need an algorithm_label column; trace them from "
            "rows TaskCollection.load_results returned."
        )
    if xscale == "log" and not bool((curves["tau"] > 0).all()):
        raise ValueError(
            "A log axis needs positive thresholds; use xscale='linear' for zero."
        )
    meta: Mapping[str, Any] = (template or {}).get("layout", {}).get("meta", {})
    series: Mapping[str, Mapping[str, str]] = meta.get("series", {})
    profiles: Mapping[str, Any] = meta.get("profiles", {})
    # Each curve's label, algorithm and points.
    ranked: list[tuple[str, Any, pd.DataFrame]] = [
        (str(curve["algorithm_label"].iloc[0]), algorithm, curve.sort_values("tau"))
        for algorithm, curve in curves.groupby("algorithm", dropna=False, sort=False)
    ]
    if tau is not None:
        ranked.sort(key=lambda entry: -_share_within(entry[2], tau))
    leading = profiles.get("leading", len(ranked))

    # The reading under the pointer, with Plotly's own fields in its slots.
    reading = _fill(
        html.escape(hover).replace("%{", "%\u200b{"),
        tau="%{x:.3~g}",
        share="%{y:.1%}",
    )
    # The leading curves are drawn last, above the rest.
    context: list[Any] = []
    above: list[Any] = []
    for position, (label, algorithm, ordered) in enumerate(ranked):
        # Without a series, a curve's algorithm may be any value, a name say.
        look: Mapping[str, str] = series.get(algorithm.uuid, {}) if series else {}
        lead = position < leading
        line = profiles.get("lead" if lead else "context", {})
        colour = look.get("color") if lead else look.get("muted", line.get("color"))
        (above if lead else context).append(
            go.Scatter(
                x=ordered["tau"],
                y=ordered["fraction"],
                mode="lines+markers" if profiles else "lines",
                name=html.escape(label),
                legendrank=position,
                opacity=line.get("opacity"),
                line={
                    "color": colour,
                    "width": line.get("width"),
                    "dash": look.get("dash"),
                },
                marker={
                    "color": colour,
                    "symbol": look.get("symbol"),
                    "size": profiles.get("marker_size"),
                    "maxdisplayed": profiles.get("markers_shown"),
                },
                hovertemplate=f"<b>%{{fullData.name}}</b><br>{reading}<extra></extra>",
            )
        )

    x_axis: dict[str, Any] = {"type": xscale, "title": {"text": html.escape(xlabel)}}
    if xscale == "linear" and "x_range" in profiles:
        low, high = profiles["x_range"]
        x_axis["range"] = [
            low,
            high if tau is None else max(high, tau * profiles.get("pad", 1)),
        ]
    fig = go.Figure(context + above)
    fig.update_layout(
        template=dict(template) if template is not None else None,
        showlegend=True,
        xaxis=x_axis,
        yaxis={
            "range": [0, profiles["pad"]] if "pad" in profiles else None,
            "tickformat": ".0%",
            "title": {"text": html.escape(ylabel)},
        },
    )
    if tau is not None:
        fig.add_shape(
            type="line",
            x0=tau,
            x1=tau,
            yref="paper",
            y0=0,
            y1=1,
            line=profiles.get("guide", {}),
            label={
                "text": html.escape(
                    _fill(guide, tau=number_text(tau, _Look.of(template).numbers))
                ),
                "textposition": "end",
                "textangle": 0,
                "xanchor": "left",
                "yanchor": "top",
            },
        )
    return fig


class _Side(Enum):
    """Where a ratio sits against the view, and the edge share it is drawn at.

    A ratio beyond the view lies past its bottom or top edge; one the axis
    cannot place is zero on a log axis, or infinite.
    """

    IN = "in"
    BELOW = "below"
    ABOVE = "above"
    ZERO = "zero"
    INF = "inf"

    @property
    def edge(self) -> float:
        """The share of the ratio axis this side's markers sit at."""
        return 0.0 if self in (_Side.BELOW, _Side.ZERO) else 1.0

    @property
    def beyond(self) -> bool:
        """Whether the ratio is finite and outside the view."""
        return self in (_Side.BELOW, _Side.ABOVE)


class _Trial(NamedTuple):
    """One trial as the figure reads it."""

    algorithm: str
    instance: str
    feature: float
    position: float
    weight: float
    ratio: float


class _Spot(NamedTuple):
    """Where a marker is drawn: its side, then its x and y on their axes."""

    side: _Side
    x: float
    y: float


@dataclass(frozen=True)
class _Look:
    """What a template's ``layout.meta`` sets; ``None`` leaves Plotly's own."""

    ink: str
    series: Mapping[str, Mapping[str, str]]
    slots: Sequence[Mapping[str, str]]
    size: float | None
    size_span: float
    opacity: float | None
    edge_opacity: float | None
    ring_width: float | None
    median_width: float | None
    band_opacity: float | None
    numbers: NumberStyle
    reference_line: Mapping[str, Any]
    count_at: float
    count_shift: float

    @classmethod
    def of(cls, template: Mapping[str, Any] | None) -> _Look:
        meta: Mapping[str, Any] = (template or {}).get("layout", {}).get("meta", {})
        marker: Mapping[str, Any] = meta.get("marker", {})
        running: Mapping[str, Any] = meta.get("running_median", {})
        return cls(
            # Plotly's own default ink.
            ink=str(meta.get("ink", "#444444")),
            series=meta.get("series", {}),
            slots=meta.get("slots", ()),
            size=marker.get("size"),
            size_span=float(marker.get("size_span", 0)),
            opacity=marker.get("opacity"),
            edge_opacity=marker.get("edge_opacity"),
            ring_width=marker.get("ring_width"),
            median_width=running.get("width"),
            band_opacity=running.get("band_opacity"),
            numbers=NumberStyle(
                meta.get("significant_digits"),
                **meta.get("number_separators", {}),
            ),
            reference_line=meta.get("reference_line", {}),
            # Without a template the counts sit at the right end of each edge.
            count_at=float(meta.get("edge_count", {}).get("at", 1.0)),
            count_shift=float(meta.get("edge_count", {}).get("shift", 0)),
        )


@dataclass(frozen=True)
class _Axes:
    """How the two axes place a value, and the view they open on."""

    x_scale: str
    log_y: bool
    x_range: list[float]
    y_range: list[float]

    def placed(self, value: float, *, ratio: bool) -> bool:
        """Whether an axis has a place for a value: a log one has none for 0."""
        if ratio and not self.log_y:
            return math.isfinite(value) and value >= 0
        return value > 0 and math.isfinite(value)

    def x_at(self, trial: _Trial) -> float:
        if self.x_scale == "quantile":
            return trial.position
        return math.log10(trial.feature) if self.x_scale == "log" else trial.feature

    def y_at(self, ratio: float) -> float:
        return math.log10(ratio) if self.log_y else ratio

    def side(self, ratio: float) -> _Side:
        if not self.placed(ratio, ratio=True):
            return _Side.INF if ratio == math.inf or not self.log_y else _Side.ZERO
        at = self.y_at(ratio)
        if at < self.y_range[0]:
            return _Side.BELOW
        return _Side.ABOVE if at > self.y_range[1] else _Side.IN

    def spot(self, trial: _Trial) -> _Spot | None:
        """Where a trial is drawn, or ``None`` outside the property axis's view."""
        if not self.placed(trial.feature, ratio=False):
            return None
        at = self.x_at(trial)
        if at < self.x_range[0] or at > self.x_range[1]:
            return None
        side = self.side(trial.ratio)
        x = trial.position if self.x_scale == "quantile" else trial.feature
        return _Spot(side, x, trial.ratio if side is _Side.IN else side.edge)


def _opening_axes(
    trials: list[_Trial],
    x_scale: str,
    log_y: bool,
    fence: float,
    padding: tuple[float, float],
) -> _Axes:
    """The view a figure opens on.

    The property axis fits every placed value. The ratio axis holds the
    robust range of the placed ratios, always a ratio of one, and from zero
    on a linear axis.
    """
    probe = _Axes(x_scale, log_y, [], [])
    across = [
        probe.x_at(trial)
        for trial in trials
        if probe.placed(trial.feature, ratio=False)
    ]
    if across:
        low, high = min(across), max(across)
        pad = (high - low) * padding[0] or 0.5
        x_range = [low - pad, high + pad]
    else:
        x_range = [-1.0, 1.0]
    sample = [
        (probe.y_at(trial.ratio), trial.weight)
        for trial in trials
        if probe.placed(trial.ratio, ratio=True)
    ]
    low = high = probe.y_at(1.0)
    if sample:
        fenced = compute_robust_range(
            [value for value, _ in sample],
            [weight for _, weight in sample],
            fence=fence,
        )
        low, high = min(low, fenced[0]), max(high, fenced[1])
    if not log_y:
        low = 0.0
    pad = (high - low) * padding[1] or 0.5
    return _Axes(x_scale, log_y, x_range, [low - pad, high + pad])


def _series_looks(
    order: list[str], names: Mapping[str, str], look: _Look
) -> dict[str, dict[str, str]]:
    """Each algorithm's colour, marker and name.

    Each takes the template's series entry, else the next of its slots,
    else Plotly's own colourway.
    """
    plain = plotly.colors.qualitative.Plotly
    looks: dict[str, dict[str, str]] = {}
    unlisted = 0
    for uuid in order:
        given: Mapping[str, str]
        if uuid in look.series:
            given = look.series[uuid]
        else:
            given = look.slots[unlisted % len(look.slots)] if look.slots else {}
            unlisted += 1
        looks[uuid] = {
            "color": given.get("color", plain[len(looks) % len(plain)]),
            "symbol": given.get("symbol", "circle"),
            "name": given.get("name", names[uuid]),
        }
    return looks


def _fill(text: str, **fields: str) -> str:
    """One label with each of its ``{name}`` fields filled."""
    for name, value in fields.items():
        text = text.replace("{" + name + "}", value)
    return text


@dataclass(frozen=True)
class _MarkerTexts:
    """The texts a marker's hover is written from."""

    feature: str
    point_hover: str
    point_beyond_view: str
    count_one: str
    count: str


def _hover(
    spot: _Spot, group: list[_Trial], texts: _MarkerTexts, numbers: NumberStyle
) -> str:
    """A marker's hover.

    It names the values, whether they lie beyond the view, and how many
    trials the marker covers.
    """
    count = len(group)
    lines = [
        _fill(
            texts.point_hover,
            property=texts.feature,
            x=span_text([trial.feature for trial in group], numbers),
            y=span_text([trial.ratio for trial in group], numbers),
        ),
        *([texts.point_beyond_view] if spot.side.beyond else []),
        _fill(
            texts.count_one if count == 1 else texts.count,
            count=count_text(count, numbers),
        ),
    ]
    return "<br>".join(html.escape(text) for text in lines)


def _points_in(group: list[_Trial], uuid: str) -> int:
    """One algorithm's points in a group: its distinct Instance ratios."""
    return len({(t.instance, t.ratio) for t in group if t.algorithm == uuid})


def _running_median_traces(
    go: Any,
    line: pd.DataFrame,
    order: list[str],
    looks: Mapping[str, Mapping[str, str]],
    axes: _Axes,
    running_median: str,
    look: _Look,
) -> list[Any]:
    """Each algorithm's quartile band and running median.

    A quartile the ratio axis cannot place sits on the edge of the view.
    """
    top = 10 ** axes.y_range[1] if axes.log_y else axes.y_range[1]
    bottom = 10 ** axes.y_range[0] if axes.log_y else axes.y_range[0]

    def on_axis(value: float) -> float | None:
        if math.isnan(value):
            return None
        if axes.placed(value, ratio=True):
            return value
        return top if value == math.inf else bottom

    traces: list[Any] = []
    owners = [algorithm.uuid for algorithm in line["algorithm"]]
    for uuid in order:
        own = line.loc[[owner == uuid for owner in owners]]
        stops = own["position"] if axes.x_scale == "quantile" else own["feature"]
        xs = [
            float(x) if drawn else None
            for x, drawn in zip(stops, own["median"].notna())
        ]
        colour = looks[uuid]["color"]
        common: dict[str, Any] = {
            "mode": "lines",
            "showlegend": False,
            "legendgroup": uuid,
            "x": xs,
            "connectgaps": False,
        }
        band: dict[str, str] = {}
        if look.band_opacity is not None and colour.startswith("#"):
            red, green, blue = (int(colour[at : at + 2], 16) for at in (1, 3, 5))
            band = {"fillcolor": f"rgba({red}, {green}, {blue}, {look.band_opacity:g})"}
        name = html.escape(_fill(running_median, algorithm=looks[uuid]["name"]))
        traces += [
            go.Scatter(
                **common,
                hoverinfo="skip",
                y=[on_axis(float(v)) for v in own["upper"]],
                line={"width": 0},
            ),
            go.Scatter(
                **common,
                hoverinfo="skip",
                y=[on_axis(float(v)) for v in own["lower"]],
                fill="tonexty",
                line={"width": 0},
                **band,
            ),
            go.Scatter(
                **common,
                name=name,
                y=[on_axis(float(v)) for v in own["median"]],
                line={"color": colour, "width": look.median_width},
                hovertemplate=f"{name}<extra></extra>",
            ),
        ]
    return traces


def _marker_traces(
    go: Any,
    groups: Mapping[_Spot, list[_Trial]],
    order: list[str],
    looks: Mapping[str, Mapping[str, str]],
    texts: _MarkerTexts,
    look: _Look,
) -> list[Any]:
    """One trace per algorithm and marker kind.

    A marker's area grows with the points it stands for, on one scale for
    the whole figure. Inside the view it takes the algorithm's own symbol; a
    ratio beyond the view is a filled dot on the edge, and one the axis
    cannot place an open ring there.
    """
    busiest = max(
        [1, *(_points_in(group, uuid) for group in groups.values() for uuid in order)]
    )

    def diameter(count: int) -> float:
        if look.size is None or busiest <= 1:
            return look.size or 0.0
        return look.size + look.size_span * math.sqrt((count - 1) / (busiest - 1))

    traces: list[Any] = []
    for uuid in order:
        kinds: dict[tuple[bool, str], list[tuple[_Spot, list[_Trial]]]] = {}
        for spot, group in groups.items():
            if not any(trial.algorithm == uuid for trial in group):
                continue
            inside = spot.side is _Side.IN
            symbol = (
                looks[uuid]["symbol"]
                if inside
                else "circle" if spot.side.beyond else "circle-open"
            )
            kinds.setdefault((inside, symbol), []).append((spot, group))
        for (inside, symbol), marked in kinds.items():
            counts = [
                (_points_in(group, uuid), sum(t.algorithm == uuid for t in group))
                for _, group in marked
            ]
            ring = symbol == "circle-open"
            marker: dict[str, Any] = {
                "color": looks[uuid]["color"],
                "symbol": symbol,
                "sizemode": "diameter",
                "line": {"width": 0} if not ring else {"width": look.ring_width},
                "opacity": look.opacity if inside else look.edge_opacity,
            }
            if look.size is not None:
                marker["size"] = [diameter(points) for points, _ in counts]
            traces.append(
                go.Scatter(
                    mode="markers",
                    showlegend=False,
                    legendgroup=uuid,
                    name=html.escape(looks[uuid]["name"]),
                    yaxis="y" if inside else "y2",
                    cliponaxis=False,
                    x=[spot.x for spot, _ in marked],
                    y=[spot.y for spot, _ in marked],
                    marker={
                        key: value for key, value in marker.items() if value is not None
                    },
                    customdata=[list(pair) for pair in counts],
                    hovertext=[
                        _hover(spot, group, texts, look.numbers)
                        for spot, group in marked
                    ],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
    return traces


def _edge_counts(
    groups: Mapping[_Spot, list[_Trial]],
    order: list[str],
    looks: Mapping[str, Mapping[str, str]],
    look: _Look,
) -> list[dict[str, Any]]:
    """One count per edge, per algorithm in its colour."""
    edges: dict[float, dict[str, int]] = {}
    for spot, group in groups.items():
        if spot.side is _Side.IN:
            continue
        tally = edges.setdefault(spot.side.edge, {})
        for trial in group:
            tally[trial.algorithm] = tally.get(trial.algorithm, 0) + 1
    return [
        {
            "xref": "paper",
            "yref": "paper",
            "x": look.count_at,
            "y": at,
            "xanchor": "right",
            "yanchor": "bottom" if at < 0.5 else "top",
            "yshift": look.count_shift if at < 0.5 else -look.count_shift,
            "showarrow": False,
            "text": " · ".join(
                f'<span style="color:{looks[uuid]["color"]}">{tally[uuid]}</span>'
                for uuid in order
                if uuid in tally
            ),
            "font": {"color": look.ink},
        }
        for at, tally in edges.items()
    ]


def _quantile_ticks(line: pd.DataFrame, numbers: NumberStyle) -> dict[str, Any]:
    """Ticks at each tenth of the weight, named by the property value there."""
    stops = line.drop_duplicates("position")
    if not len(stops):
        return {}
    nearest = [
        stops.iloc[int((stops["position"] - tenth / 10).abs().to_numpy().argmin())]
        for tenth in range(11)
    ]
    return {
        "tickmode": "array",
        "tickvals": [float(stop["position"]) for stop in nearest],
        "ticktext": [tick_text(float(stop["feature"]), numbers) for stop in nearest],
    }


def plot_error_ratios(
    points: pd.DataFrame,
    line: pd.DataFrame,
    *,
    x_scale: Literal["log", "linear", "quantile"] = "log",
    y_scale: Literal["log", "linear"] = "log",
    fence: float = 3.0,
    padding: tuple[float, float] = (0.08, 0.05),
    feature: str = "property",
    xlabel: str = "property",
    ylabel: str = "error ratio",
    point_hover: str = "{property} {x} · error ratio {y}",
    point_beyond_view: str = "beyond the view",
    count_one: str = "{count} trial",
    count: str = "{count} trials",
    running_median: str = "{algorithm} running median",
    template: Mapping[str, Any] | None = None,
) -> Any:
    """Draw each error ratio against its Instance's property, with running medians.

    One algorithm's trials on one Instance with the same ratio are one point,
    and coincident points merge into one marker whose area grows with the
    points it stands for; its hover states how many trials it covers. The
    view opens on every placed property value and on the :func:`compute_robust_range`
    of the placed ratios on the ratio axis's own scale, widened to hold a
    ratio of one, and from zero on a linear axis. A ratio beyond the view
    sits on the edge it lies past as a filled dot, and one the axis cannot
    place (an infinite one on the top edge, zero on a log axis on the bottom)
    as an open ring; each edge names how many trials of each algorithm it
    holds.

    A template's ``layout.meta`` may carry the look: ``ink`` (lines and
    counts), ``series`` (per algorithm UUID its ``color``, ``symbol`` and
    ``name``), ``slots`` (the ``color`` and ``symbol`` an algorithm the series
    omits takes, in order), ``marker`` (``size``, ``size_span``, ``opacity``,
    ``edge_opacity``, ``ring_width``), ``running_median`` (``width``,
    ``band_opacity``), ``reference_line`` (the line at a ratio of one),
    ``edge_count`` (``at``, how far along an edge its counts sit, and
    ``shift``, in pixels), ``significant_digits`` (how many a hover rounds a
    value to; unrounded without it) and ``number_separators`` (``decimal``
    and ``group``; ``.`` and ``,`` without them). What it leaves out is
    Plotly's own.

    Args:
        points: :func:`place_error_ratio_points` output.
        line: :func:`trace_running_median` output for the same points.
        x_scale: How the property axis places a value: by its logarithm, its
            value, or its weighted mid-rank ``position``.
        y_scale: Whether the ratio axis is logarithmic or linear.
        fence: The interquartile ranges the opening ratio range reaches
            beyond each quartile.
        padding: The share of each axis's span, property then ratio, left
            empty beyond the opening range.
        feature: The property's name, filling ``{property}`` in a hover.
        xlabel: The property axis's title.
        ylabel: The ratio axis's title.
        point_hover: A marker's first hover line, whose ``{property}``,
            ``{x}`` and ``{y}`` fields the figure fills.
        point_beyond_view: The hover line under the first when the marker
            stands for values beyond the view.
        count_one: The hover's count line for one trial, whose ``{count}``
            field the figure fills.
        count: The hover's count line for several trials, whose ``{count}``
            field the figure fills.
        running_median: A running median's hover, whose ``{algorithm}``
            field the figure fills.
        template: A Plotly template, as a mapping, applied to the figure;
            ``None`` keeps Plotly's default template.

    Returns:
        The ``plotly.graph_objects.Figure``. Each marker's ``customdata`` is
        the number of points it stands for and the trials they cover.
    """
    texts = _MarkerTexts(feature, point_hover, point_beyond_view, count_one, count)
    look = _Look.of(template)
    trials = [
        _Trial(a.uuid, i.uuid, float(f), float(q), float(w), float(r))
        for a, i, f, q, w, r in zip(
            points["algorithm"],
            points["instance"],
            points["feature"],
            points["position"],
            points["weight"],
            points["ratio"],
        )
    ]
    axes = _opening_axes(trials, x_scale, y_scale == "log", fence, padding)
    groups: dict[_Spot, list[_Trial]] = {}
    for trial in trials:
        spot = axes.spot(trial)
        if spot is not None:
            groups.setdefault(spot, []).append(trial)
    order = list(dict.fromkeys(trial.algorithm for trial in trials))
    names = (
        dict(zip((a.uuid for a in points["algorithm"]), points["algorithm_label"]))
        if "algorithm_label" in points.columns
        else {a.uuid: str(a) for a in points["algorithm"]}
    )
    looks = _series_looks(order, names, look)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[10**v for v in axes.x_range] if x_scale == "log" else axes.x_range,
            y=[1, 1],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line={**look.reference_line, "color": look.ink},
        )
    )
    fig.add_traces(
        _running_median_traces(go, line, order, looks, axes, running_median, look)
    )
    fig.add_traces(_marker_traces(go, groups, order, looks, texts, look))
    fig.update_layout(
        template=dict(template) if template is not None else None,
        xaxis={
            "type": "linear" if x_scale == "quantile" else x_scale,
            "range": axes.x_range,
            "title": {"text": html.escape(xlabel)},
            **(_quantile_ticks(line, look.numbers) if x_scale == "quantile" else {}),
        },
        yaxis={
            "type": y_scale,
            "range": axes.y_range,
            "title": {"text": html.escape(ylabel)},
        },
        yaxis2={
            "overlaying": "y",
            "range": [0, 1],
            "visible": False,
            "fixedrange": True,
        },
        annotations=_edge_counts(groups, order, looks, look),
        showlegend=False,
    )
    return fig


def _percent(share: float) -> str:
    return f"{share * 100:.1f}%"


def plot_outcomes(
    bins: pd.DataFrame,
    *,
    scale: Literal["log", "linear", "quantile"] = "log",
    padding: float = 0.05,
    categories: Mapping[str, str],
    xlabel: str = "property",
    ylabel: str = "share of problem instances",
    success_hover: str = "success rate in this bin: {share}",
    stack_hover: str = "this stack: {share} of problem instances",
    pair_hover: str = "success rate in this bin: {a} · compared algorithm: {b}",
    template: Mapping[str, Any] | None = None,
) -> Any:
    """Draw each bin's outcome shares as one stacked bar.

    Each category the bins hold a share of is one bar segment, stacked bottom
    to top in the order ``categories`` names them. A bar's hover names its
    property range, the segment's share, the bin's share within the
    threshold (each algorithm's, when two are compared) and the segment's
    share of the bin.

    A template's ``layout.meta`` may carry the look: ``ink`` (a hatch's
    lines), ``significant_digits`` and ``number_separators`` (as
    :func:`plot_error_ratios` reads them), and ``outcomes``: ``colors`` (per
    category), ``hatched`` (the categories drawn hatched), ``hatch`` (the
    pattern's ``shape``, ``size`` and ``solidity``), ``bargap`` and
    ``legend``. What it leaves out is Plotly's own.

    Args:
        bins: :func:`bin_outcomes` output.
        scale: The scale the bins were cut on.
        padding: The share of the property axis's span left empty beyond the
            bins on a ``log`` or ``linear`` scale.
        categories: Each category's name, by its column, bottom to top.
        xlabel: The property axis's title.
        ylabel: The share axis's title.
        success_hover: A bar's hover line for one algorithm's share within
            the threshold, whose ``{share}`` field the figure fills.
        stack_hover: A bar's last hover line, the segment's share of the
            shown instances, whose ``{share}`` field the figure fills.
        pair_hover: A bar's hover line for two compared algorithms' shares
            within the threshold, whose ``{a}`` and ``{b}`` fields the
            figure fills.
        template: A Plotly template, as a mapping, applied to the figure;
            ``None`` keeps Plotly's default template.

    Returns:
        The ``plotly.graph_objects.Figure``.
    """
    look = _Look.of(template)
    meta: Mapping[str, Any] = (template or {}).get("layout", {}).get("meta", {})
    outcomes: Mapping[str, Any] = meta.get("outcomes", {})
    start = bins["start"].to_numpy(dtype=float)
    stop = bins["stop"].to_numpy(dtype=float)
    centres = (start * stop) ** 0.5 if scale == "log" else (start + stop) / 2
    write = tick_text if scale == "quantile" else number_text
    bounds = [
        f"{write(float(lower), look.numbers)}–{write(float(upper), look.numbers)}"
        for lower, upper in zip(bins["lower"], bins["upper"])
    ]
    present = [
        key for key in categories if key in bins.columns and bool(bins[key].gt(0).any())
    ]
    held = bins[present].sum(axis=1).to_numpy(dtype=float)

    def within(*keys: str) -> list[str]:
        shares = bins[list(keys)].sum(axis=1).to_numpy(dtype=float)
        return [
            _percent(share / total) if total else "–"
            for share, total in zip(shares, held)
        ]

    if "both" in bins.columns:
        success = [
            _fill(pair_hover, a=a, b=b)
            for a, b in zip(within("both", "a_only"), within("both", "b_only"))
        ]
    else:
        success = [
            _fill(success_hover, share=share) for share in within("success_within")
        ]
    traces: list[Any] = []
    for key in present:
        colour = outcomes.get("colors", {}).get(key)
        marker: dict[str, Any] = {"color": colour}
        if key in outcomes.get("hatched", ()):
            marker["pattern"] = {
                **outcomes.get("hatch", {}),
                "fgcolor": look.ink,
                "bgcolor": colour,
            }
        shares = bins[key].to_numpy(dtype=float)
        traces.append(
            go.Bar(
                name=html.escape(categories[key]),
                x=centres,
                y=shares,
                width=stop - start,
                marker={name: value for name, value in marker.items() if value},
                hovertext=[
                    "<br>".join(
                        html.escape(text)
                        for text in (
                            bound,
                            f"{categories[key]}: {_percent(share)}",
                            line,
                            _fill(stack_hover, share=_percent(share)),
                        )
                    )
                    for bound, share, line in zip(bounds, shares, success)
                ],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    if scale == "quantile":
        axis: dict[str, Any] = {
            "type": "linear",
            "range": [0, 1],
            "tickmode": "array",
            "tickvals": [*start, stop[-1]],
            "ticktext": [
                tick_text(float(value), look.numbers)
                for value in [*bins["lower"], bins["upper"].iloc[-1]]
            ],
        }
    else:
        ends = [start[0], stop[-1]]
        if scale == "log":
            ends = [math.log10(end) for end in ends]
        pad = (ends[1] - ends[0]) * padding or 0.5
        axis = {"type": scale, "range": [ends[0] - pad, ends[1] + pad]}
    fig = go.Figure(traces)
    fig.update_layout(
        template=dict(template) if template is not None else None,
        barmode="stack",
        bargap=outcomes.get("bargap"),
        showlegend=True,
        legend=outcomes.get("legend"),
        xaxis={**axis, "title": {"text": html.escape(xlabel)}},
        yaxis={
            "rangemode": "tozero",
            "tickformat": ".0%",
            "title": {"text": html.escape(ylabel)},
        },
    )
    return fig
