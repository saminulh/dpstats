"""Hold the shipped chart colours to the floors they claim.

The site ships colour as literal hex, so nothing in production recomputes
contrast or colour distance. These checks derive both from the published
formulas — WCAG relative luminance, CIE76 and CIEDE2000 over CIELAB, and the
Machado (2009) colour-vision-deficiency simulation — and read the plot surface
and histogram colours back out of the stylesheet that paints them, so a table
edited by hand cannot quietly drop below the floors it is chosen to meet.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SITE_DIR = REPO_ROOT / "docs" / "site"

from dpstats.algorithm import PrivacyInterpretation

from chart_style import OUTCOME_STACK, chart_style, profile_series
from series_style import (
    CLASS_DASH,
    CORE_COLOURS,
    CORE_ORDER,
    CVD_SEPARATED,
    MARKER_SHAPES,
    MINIMUM_CONTRAST,
    MINIMUM_CVD_DELTA_E,
    MINIMUM_DELTA_E,
    PALETTE,
    PLOT_SURFACE,
    marker_shape,
    marker_shapes_by_uuid,
    series_style,
)

STYLESHEET = SITE_DIR / "docs" / "stylesheets" / "extra.css"

# Machado, Oliveira and Fernandes (2009), severity 1.0, on linear sRGB.
DEFICIENCIES = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def rgb(value: str) -> tuple[float, float, float]:
    """One ``#rrggbb`` string as sRGB components in the unit interval."""
    text = value.lstrip("#")
    return tuple(int(text[at : at + 2], 16) / 255 for at in (0, 2, 4))  # type: ignore[return-value]


def linear(channel: float) -> float:
    """One sRGB component, linearized."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(colour: tuple[float, float, float]) -> float:
    """WCAG relative luminance."""
    red, green, blue = (linear(channel) for channel in colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(one: str, two: str) -> float:
    """The WCAG contrast ratio between two colours."""
    first, second = luminance(rgb(one)), luminance(rgb(two))
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def lab(value: str, deficiency: str | None = None) -> tuple[float, float, float]:
    """One colour in CIELAB under D65, as seen under ``deficiency`` if named."""
    matrix = (
        (0.4124564, 0.3575761, 0.1804375),
        (0.2126729, 0.7151522, 0.0721750),
        (0.0193339, 0.1191920, 0.9503041),
    )
    white = (0.95047, 1.0, 1.08883)
    components = [linear(channel) for channel in rgb(value)]
    if deficiency is not None:
        simulation = DEFICIENCIES[deficiency]
        components = [
            min(
                1.0,
                max(
                    0.0,
                    sum(
                        simulation[row][column] * components[column]
                        for column in range(3)
                    ),
                ),
            )
            for row in range(3)
        ]
    xyz = [
        sum(matrix[row][column] * components[column] for column in range(3))
        for row in range(3)
    ]

    def shape(ratio: float) -> float:
        return (
            ratio ** (1 / 3) if ratio > 216 / 24389 else (24389 / 27 * ratio + 16) / 116
        )

    fx, fy, fz = (shape(xyz[axis] / white[axis]) for axis in range(3))
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(one: str, two: str) -> float:
    """CIE76 colour difference."""
    return math.dist(lab(one), lab(two))


def delta_e_2000(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """CIEDE2000 colour difference between two CIELAB colours (Sharma 2005)."""
    (l1, a1, b1), (l2, a2, b2) = first, second
    chroma = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(chroma**7 / (chroma**7 + 25**7)))
    a1, a2 = (1 + g) * a1, (1 + g) * a2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1, h2 = (math.degrees(math.atan2(b, a)) % 360 for a, b in ((a1, b1), (a2, b2)))
    turn = h2 - h1
    if c1 * c2 == 0:
        turn = 0.0
    elif turn > 180:
        turn -= 360
    elif turn < -180:
        turn += 360
    d_hue = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(turn / 2))
    mean_l, mean_c = (l1 + l2) / 2, (c1 + c2) / 2
    if c1 * c2 == 0:
        mean_h = h1 + h2
    elif abs(h1 - h2) <= 180:
        mean_h = (h1 + h2) / 2
    else:
        mean_h = (h1 + h2 + (360 if h1 + h2 < 360 else -360)) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(mean_h - 30))
        + 0.24 * math.cos(math.radians(2 * mean_h))
        + 0.32 * math.cos(math.radians(3 * mean_h + 6))
        - 0.20 * math.cos(math.radians(4 * mean_h - 63))
    )
    rotation = (
        -2
        * math.sqrt(mean_c**7 / (mean_c**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((mean_h - 275) / 25) ** 2))))
    )
    s_l = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    s_c, s_h = 1 + 0.045 * mean_c, 1 + 0.015 * mean_c * t
    terms = ((l2 - l1) / s_l, (c2 - c1) / s_c, d_hue / s_h)
    return math.sqrt(sum(term**2 for term in terms) + rotation * terms[1] * terms[2])


def histogram_colours(scheme: str) -> list[tuple[str, str]]:
    """The histogram's hued stack, bottom to top, as the stylesheet paints it."""
    order = [str(entry["token"]) for entry in OUTCOME_STACK]
    block = re.search(
        rf'\[data-bs-theme="{scheme}"\]\s*\{{(.*?)\}}',
        STYLESHEET.read_text(encoding="utf-8"),
        re.S,
    )
    assert block is not None, scheme
    declared = dict(
        re.findall(r"(--dp-hist-[\w-]+):\s*(#[0-9a-fA-F]{6});", block.group(1))
    )
    return [(token, declared[token]) for token in order if token in declared]


SCHEMES = tuple(PLOT_SURFACE)


def test_every_listed_core_takes_its_own_colour_in_both_schemes() -> None:
    """No two cores share an entry, and the palette covers every listed core."""
    assert len(CORE_ORDER) == len(set(CORE_ORDER)), "a core is listed twice"
    for scheme in SCHEMES:
        assert len(PALETTE[scheme]) >= len(CORE_ORDER), scheme
        assert list(CORE_COLOURS[scheme]) == list(CORE_ORDER), scheme
        assert len(set(CORE_COLOURS[scheme].values())) == len(CORE_ORDER), scheme


def test_comparison_slots_use_five_distinct_shipped_colours() -> None:
    """Five selected configurations have distinct inks without changing profiles."""
    slots = series_style()["comparisonColours"]
    assert slots == {
        "default": ["#0085da", "#ed404c", "#198038", "#9f1853", "#71855f"],
        "slate": ["#33b1ff", "#fa4d56", "#6fdc8c", "#ff7eb6", "#fee296"],
    }
    for scheme, colours in slots.items():
        assert len(set(colours)) == 5
        assert set(colours) <= set(PALETTE[scheme])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_entry_stands_against_its_plot_surface(scheme: str) -> None:
    """Every drawn colour clears the contrast floor on the surface it is on."""
    surface = PLOT_SURFACE[scheme]
    for position, colour in enumerate(PALETTE[scheme]):
        assert (
            contrast(colour, surface) >= MINIMUM_CONTRAST
        ), f"entry {position} {colour} is {contrast(colour, surface):.2f}:1 on {surface} ({scheme})"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_entry_is_separable_from_every_other(scheme: str) -> None:
    """No two drawn colours in one scheme sit inside the separation floor."""
    for (one, first), (two, second) in itertools.combinations(
        enumerate(PALETTE[scheme]), 2
    ):
        assert (
            delta_e(first, second) >= MINIMUM_DELTA_E
        ), f"entries {one} and {two} are {delta_e(first, second):.1f} dE apart ({scheme})"


def test_the_colour_difference_matches_the_published_reference_pair() -> None:
    """CIEDE2000 reproduces Sharma, Wu and Dalal's first test pair."""
    assert delta_e_2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) == pytest.approx(
        2.0425, abs=1e-4
    )


@pytest.mark.parametrize("deficiency", DEFICIENCIES)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_leading_entries_stay_apart_under_colour_vision_deficiency(
    scheme: str, deficiency: str
) -> None:
    """The cores most often compared stay separable to a reader with CVD."""
    for (one, first), (two, second) in itertools.combinations(
        enumerate(PALETTE[scheme][:CVD_SEPARATED]), 2
    ):
        difference = delta_e_2000(lab(first, deficiency), lab(second, deficiency))
        assert (
            difference >= MINIMUM_CVD_DELTA_E
        ), f"entries {one} and {two} are {difference:.1f} dE2000 apart under {deficiency} ({scheme})"


@pytest.mark.parametrize("deficiency", [None, "protanopia", "deuteranopia"])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_adjacent_histogram_outcomes_stay_apart(
    scheme: str, deficiency: str | None
) -> None:
    """Each stacked outcome reads apart from the one it sits on, as the stylesheet claims."""
    stack = histogram_colours(scheme)
    assert len(stack) >= 7, stack
    for (below, first), (above, second) in itertools.pairwise(stack):
        difference = delta_e_2000(lab(first, deficiency), lab(second, deficiency))
        assert (
            difference >= 20
        ), f"{below} and {above} are {difference:.1f} dE2000 apart ({scheme}, {deficiency})"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_stylesheet_paints_the_surface_the_table_was_chosen_against(
    scheme: str,
) -> None:
    """The floors are measured against the surface the page actually draws on."""
    stylesheet = STYLESHEET.read_text(encoding="utf-8")
    # Palette keys are release data; the browser theme uses light/dark names.
    theme = "light" if scheme == "default" else "dark"
    match = re.search(rf'\[data-bs-theme="{theme}"\]\s*\{{(.*?)\}}', stylesheet, re.S)
    assert match is not None, f"no {scheme} block in the stylesheet"
    surface = re.search(r"--dp-chart-plot:\s*([^;]+);", match.group(1))
    assert surface is not None, f"no plot surface in the {scheme} block"
    assert surface.group(1).strip() == PLOT_SURFACE[scheme]


def test_every_privacy_interpretation_takes_a_dash() -> None:
    """No class a release can register is left without a line style."""
    assert sorted(CLASS_DASH) == sorted(entry.value for entry in PrivacyInterpretation)


def test_marker_shapes_stop_separating_past_the_last_shape() -> None:
    """A sixth configuration of one core takes the last shape, not a new one."""
    assert marker_shape(0) == MARKER_SHAPES[0]
    assert marker_shape(len(MARKER_SHAPES) - 1) == MARKER_SHAPES[-1]
    assert marker_shape(len(MARKER_SHAPES)) == MARKER_SHAPES[-1]
    assert marker_shape(99) == MARKER_SHAPES[-1]


def test_markers_are_assigned_within_a_core_by_label_order() -> None:
    """Two cores each start at the first shape; order inside a core is by name."""
    shapes = marker_shapes_by_uuid(
        [
            {"uuid": "b", "core": "AdaSSP", "label": "Standardize + AdaSSP"},
            {"uuid": "a", "core": "AdaSSP", "label": "ApproxBounds + AdaSSP"},
            {"uuid": "c", "core": "TukeyEM", "label": "Standardize + TukeyEM"},
        ]
    )
    assert shapes == {
        "a": MARKER_SHAPES[0],
        "b": MARKER_SHAPES[1],
        "c": MARKER_SHAPES[0],
    }


def test_the_payload_carries_both_schemes_and_the_whole_rule() -> None:
    """What the drawing surfaces read holds every part of the rule."""
    payload = series_style()
    assert set(payload["colours"]) == set(SCHEMES)  # type: ignore[arg-type]
    assert payload["dash"] == CLASS_DASH
    assert payload["surface"] == PLOT_SURFACE
    assert payload["emphasis"]["curves"] > 0  # type: ignore[index]


def test_a_notebook_curve_takes_the_look_the_page_draws_it_with() -> None:
    """The light theme: AMP's palette entry, a qualified dash, a nonprivate
    reference in reference ink even when muted, an unlisted core in context
    ink."""
    series = profile_series(
        [
            {"uuid": "a", "core": "AMP", "class": "qualified", "marker": "square"},
            {"uuid": "r", "core": "AMP", "class": "nonprivate", "marker": "circle"},
            {"uuid": "u", "core": "Unlisted", "class": "standard", "marker": "circle"},
        ]
    )
    assert series == {
        "a": {"color": "#6929c4", "symbol": "square", "dash": "10 4"},
        "r": {
            "color": "#1c1c1c",
            "symbol": "circle",
            "dash": "9 3 2 3",
            "muted": "#1c1c1c",
        },
        "u": {"color": "#848484", "symbol": "circle", "dash": "solid"},
    }
    profiles = chart_style()["layout"]["meta"]["profiles"]
    assert profiles["context"]["color"] == "#848484"
    assert profiles["guide"] == {"color": "#2c5d8f", "width": 2}
    assert profiles["x_range"] == [0.0, 5.0]
