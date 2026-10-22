"""The one Plotly template Explore's property charts and their notebooks draw with.

It is built from the series rule (``series_style``) and the stylesheet's
light-theme tokens. The page applies it as its scatter's layout defaults and
paints the reader's theme over its colours; the notebook receives it as
``CHART_STYLE``, with the reader's selection's colours and markers written
into ``layout.meta.series``. What Plotly's layout has no attribute for (marker
sizes and opacities, number rounding, the comparison slots) travels in
``layout.meta``, which ``plot_error_ratios`` reads. The trial outcomes
histogram's categories, colours, hatch, bar gap and legend travel in
``layout.meta.outcomes``, which the page and ``plot_outcomes`` both read.
The success-rate curves' emphasis, markers, opening range and threshold guide
travel in ``layout.meta.profiles`` and each algorithm's curve in
``layout.meta.series`` (``profile_series``), which ``plot_profiles`` reads;
the page reads the same values from the series rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from series_style import (
    CLASS_DASH,
    COMPARISON_COLOURS,
    CORE_COLOURS,
    EMPHASISED_CURVES,
    GUIDE_WIDTH,
    MARKER_SHAPES,
    MARKER_SIZE,
    MARKERS_SHOWN,
    MUTED_OPACITY,
    MUTED_WIDTH,
    PROFILE_PAD,
    PROFILE_RANGE,
    REFERENCE_CLASS,
    REGULAR_OPACITY,
    REGULAR_WIDTH,
)

_STYLESHEETS = Path(__file__).resolve().parent / "docs" / "stylesheets"
_SITE_SHEET = _STYLESHEETS / "extra.css"
_TABLER_SHEET = _STYLESHEETS / "vendor" / "tabler-1.5.1.min.css"

# One CSS rem, in the pixels Plotly's font size is given in.
_REM = 16.0

# The figure's frame: room for a two-line log axis under its title.
MARGIN = {"l": 65, "r": 30, "t": 28, "b": 100}

# A point inside the view is drawn faint, one on its edge stronger; a marker
# standing for several points grows in area from SIZE to SIZE + SIZE_SPAN.
MARKER = {
    "size": 6,
    "size_span": 20,
    "opacity": 0.3,
    "edge_opacity": 0.65,
    "ring_width": 2,
}

# Each axis's tick count and its gridlines' layer, below the traces.
AXIS_TICKS = {"nticks": 5, "layer": "below traces"}

# The dashed line at a ratio of one.
REFERENCE_LINE = {"width": 1, "dash": "dash"}

# Each edge's counts sit this far along it, and this many pixels in from it.
EDGE_COUNT = {"at": 0.98, "shift": 15}

# The running median's line width, and its quartile band's opacity.
RUNNING_MEDIAN = {"width": 2.5, "band_opacity": 0.25}

# How many significant digits a hover rounds a value to, as the site's own
# number format does.
SIGNIFICANT_DIGITS = 3

# A drag pans the chart; zoom and the reset to the opening view stay in the
# modebar, which drops the buttons the chart has no use for.
DRAG_MODE = "pan"
MODEBAR_REMOVE = ["autoScale2d", "select2d", "lasso2d"]

# The trial outcome categories, bottom to top: the two success strengths,
# every other terminal outcome, then a declared cell nothing answered. Each
# is named by its symbol or its site-copy slot and drawn in its own
# --dp-hist-* colour, never the privacy-interpretation tokens, which keep one
# meaning site-wide; "not run" is a faint neutral, hatched.
OUTCOME_STACK = [
    {"key": "success_within", "text": "≤ τ", "token": "--dp-hist-success-strong"},
    {"key": "success_beyond", "text": "> τ", "token": "--dp-hist-success-soft"},
    {
        "key": "invalid_regime",
        "slot": "explore.outcome_invalid_regime",
        "token": "--dp-hist-fail-1",
    },
    {
        "key": "no_release",
        "slot": "explore.outcome_no_release",
        "token": "--dp-hist-fail-2",
    },
    {"key": "timeout", "slot": "explore.outcome_timeout", "token": "--dp-hist-fail-3"},
    {
        "key": "memory_limit_exceeded",
        "slot": "explore.outcome_memory_limit",
        "token": "--dp-hist-fail-4",
    },
    {
        "key": "algorithm_error",
        "slot": "explore.outcome_algorithm_error",
        "token": "--dp-hist-fail-5",
    },
    {"key": "not_run", "slot": "explore.outcome_not_run", "token": "--dp-hist-fail-6"},
]

# Two compared algorithms' categories, bottom to top. "Both" and "neither"
# are a strong and a quiet neutral, so they never read as either algorithm's
# colour; "A only" and "B only" have no token and take the two algorithms'
# own series colours.
PAIR_STACK = [
    {"key": "both", "slot": "explore.pair_both", "token": "--dp-hist-pair-both"},
    {"key": "a_only", "slot": "explore.pair_a_only", "token": None},
    {"key": "b_only", "slot": "explore.pair_b_only", "token": None},
    {
        "key": "neither",
        "slot": "explore.pair_neither",
        "token": "--dp-hist-pair-neither",
    },
]

# The categories drawn hatched, and the hatch: lines in the ink over the
# category's colour.
HATCHED = ["not_run"]
HATCH = {"shape": "/", "size": 6, "solidity": 0.35}

# The gap between bars, as a share of a bin, and the compact key above the
# chart, which names the categories and is not a control.
BAR_GAP = 0.05
LEGEND = {
    "orientation": "h",
    "traceorder": "normal",
    "x": 0,
    "xanchor": "left",
    "y": 1.02,
    "yanchor": "bottom",
    "bgcolor": "rgba(0,0,0,0)",
    "itemclick": False,
    "itemdoubleclick": False,
}


def _block(sheet: str, selector: str) -> str:
    """The declarations of the first rule whose selector list is ``selector``."""
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\}", sheet, re.S)
    if match is None:
        raise ValueError(f"The stylesheet has no {selector} rule.")
    return match.group(1)


def _token(declarations: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}:\s*([^;]+);", declarations)
    if match is None:
        raise ValueError(f"The stylesheet sets no {name}.")
    return match.group(1).strip()


def _colour(value: str, light: str) -> str:
    """A token's value as Plotly reads it.

    A ``color-mix`` of a light-theme token with ``transparent`` is that
    token's colour at the mixed share's opacity.
    """
    mixed = re.fullmatch(
        r"color-mix\(in srgb, var\((--[\w-]+)\) (\d+)%, transparent\)", value
    )
    if mixed is None:
        return value
    hex_colour = _token(light, mixed.group(1))
    red, green, blue = (int(hex_colour[at : at + 2], 16) for at in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {int(mixed.group(2)) / 100:g})"


def _font_family(stack: str) -> str:
    """A font stack as a browser serializes it: a name with a space quoted."""
    names = [name.strip().strip("'\"") for name in stack.split(",")]
    return ", ".join(f'"{name}"' if " " in name else name for name in names)


def _theme() -> tuple[str, str]:
    """The site sheet's light-theme declarations, and those every theme shares."""
    site = _SITE_SHEET.read_text(encoding="utf-8")
    return _block(site, '[data-bs-theme="light"]'), _block(
        site, ":root, [data-bs-theme]"
    )


def profile_series(algorithms: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Each algorithm's success-rate curve in the light theme, by UUID.

    It follows the series rule as the page applies it: the core's colour, the
    class's dash and the configuration's marker. A nonprivate reference keeps
    the reference ink, even among the curves drawn in context ink, and a core
    the rule does not list takes the context ink.

    Args:
        algorithms: The page index's algorithms, each with its ``uuid``,
            ``core``, ``class`` and ``marker``.
    """
    light, _ = _theme()
    reference = _token(light, "--dp-chart-diagnostic")
    context = _token(light, "--dp-chart-other")
    series: dict[str, dict[str, str]] = {}
    for entry in algorithms:
        look = {
            "color": CORE_COLOURS["default"].get(entry["core"], context),
            "symbol": entry["marker"],
            "dash": CLASS_DASH[entry["class"]],
        }
        if entry["class"] == REFERENCE_CLASS:
            look.update(color=reference, muted=reference)
        series[entry["uuid"]] = look
    return series


def chart_style() -> dict[str, Any]:
    """The light-theme template, with no selection's series written into it."""
    light, shared = _theme()
    tabler = _TABLER_SHEET.read_text(encoding="utf-8")
    ink = _token(light, "--tblr-body-color")
    size = _token(shared, "--dp-size-support")
    slots = [
        {"color": colour, "symbol": symbol}
        for colour, symbol in zip(COMPARISON_COLOURS["default"], MARKER_SHAPES)
    ]
    colours = {
        # A light-theme token, or one every theme shares.
        entry["key"]: _colour(_token(light + shared, str(entry["token"])), light)
        for entry in [*OUTCOME_STACK, *PAIR_STACK]
        if entry["token"]
    }
    colours.update(a_only=slots[0]["color"], b_only=slots[1]["color"])
    axis = {
        "showline": True,
        "mirror": True,
        "linecolor": ink,
        "gridcolor": _token(light, "--dp-chart-grid"),
        "zeroline": False,
        "automargin": True,
        **AXIS_TICKS,
    }
    return {
        "layout": {
            "font": {
                "family": _font_family(_token(tabler, "--tblr-font-monospace")),
                "size": float(size.removesuffix("rem")) * _REM,
                "color": ink,
            },
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": _token(light, "--dp-chart-plot"),
            "margin": dict(MARGIN),
            "showlegend": False,
            "dragmode": DRAG_MODE,
            "modebar": {"remove": list(MODEBAR_REMOVE)},
            "hovermode": "closest",
            "xaxis": dict(axis),
            "yaxis": dict(axis),
            "meta": {
                "ink": ink,
                "slots": slots,
                "reference": {
                    "class": REFERENCE_CLASS,
                    "color": _token(light, "--dp-chart-diagnostic"),
                },
                "series": {},
                "marker": dict(MARKER),
                "running_median": dict(RUNNING_MEDIAN),
                "significant_digits": SIGNIFICANT_DIGITS,
                "reference_line": dict(REFERENCE_LINE),
                "edge_count": dict(EDGE_COUNT),
                "profiles": {
                    "leading": EMPHASISED_CURVES,
                    "lead": {"width": REGULAR_WIDTH, "opacity": REGULAR_OPACITY},
                    "context": {
                        "color": _token(light, "--dp-chart-other"),
                        "width": MUTED_WIDTH,
                        "opacity": MUTED_OPACITY,
                    },
                    "marker_size": MARKER_SIZE,
                    "markers_shown": MARKERS_SHOWN,
                    "x_range": list(PROFILE_RANGE),
                    "pad": PROFILE_PAD,
                    # The site accent, Tabler's --tblr-primary, is this token.
                    "guide": {
                        "color": _token(light, "--tblr-primary"),
                        "width": GUIDE_WIDTH,
                    },
                },
                "outcomes": {
                    "stack": [dict(entry) for entry in OUTCOME_STACK],
                    "pairs": [dict(entry) for entry in PAIR_STACK],
                    "colors": colours,
                    "hatched": list(HATCHED),
                    "hatch": dict(HATCH),
                    "bargap": BAR_GAP,
                    "legend": dict(LEGEND),
                },
            },
        }
    }
