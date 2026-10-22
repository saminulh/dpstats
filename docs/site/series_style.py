"""How a plotted curve takes its colour, its dash and its marker.

On profiles, one core algorithm is one colour on every page and reload,
never re-dealt by rank, filter, threshold or budget. Every core
takes its own entry of one categorical palette, so no two cores share a hue.
The palette opens with IBM Carbon's categorical order and continues with the
entries farthest from those already taken; cores are listed in the order they
most often lead a released reading, so the cores most often compared take the
most separable entries. Configurations of one core share its colour and are
separated by marker. Explore's five-choice comparison views instead identify
each selected configuration by a colour/marker slot retained until removal.

Two other channels carry what colour must not: the dash pattern says how the
curve's claim is to be interpreted, and the marker shape separates the
transformation configurations of one core. Each palette is checked in rather than
computed at draw time, and is held to three measured floors in its own scheme:
every entry stands at least 3:1 against the plot surface it is drawn on, at
least 20 dE (CIE76) from every other entry, and the leading entries stay apart
under simulated protanopia, deuteranopia and tritanopia.
"""

from __future__ import annotations

# The surface each scheme's curves are drawn on, keyed by the scheme the page
# carries. Both floors below are measured against these, so they are stated
# here and checked against the stylesheet that paints them.
PLOT_SURFACE = {"default": "#e6e9ee", "slate": "#17232e"}

# The contrast and separation every shipped entry clears in its own scheme.
MINIMUM_CONTRAST = 3.0
MINIMUM_DELTA_E = 20.0

# No categorical palette this long stays separable under colour-vision
# deficiency, so only the leading entries are held to it: the cores most often
# compared. The floor is CIEDE2000 under the Machado (2009) simulation at full
# severity; it only rises, towards the ~10 a CVD-safe set reaches.
CVD_SEPARATED = 8
MINIMUM_CVD_DELTA_E = 7.0

# The categorical palette in the order cores take it. Light entries of
# Carbon's categorical set are darkened in OKLab only as far as the plot
# surface requires; later entries are the farthest remaining from those before.
PALETTE: dict[str, tuple[str, ...]] = {
    "default": (
        "#6929c4",
        "#0085da",
        "#005d5d",
        "#9f1853",
        "#ed404c",
        "#570408",
        "#198038",
        "#002d9c",
        "#8a3800",
        "#a77b00",
        "#00918e",
        "#012749",
        "#e14781",
        "#9a63f3",
        "#92718e",
        "#423b11",
        "#e118b6",
        "#7b0977",
        "#521942",
        "#71855f",
        "#738800",
        "#b16559",
        "#a06cc4",
        "#db551d",
        "#1c73fd",
        "#2e88ad",
        "#505392",
        "#673d3d",
        "#c635e4",
        "#02240c",
    ),
    "slate": (
        "#8a3ffc",
        "#33b1ff",
        "#007d79",
        "#ff7eb6",
        "#fa4d56",
        "#fff1f1",
        "#6fdc8c",
        "#4589ff",
        "#ba4e00",
        "#d2a106",
        "#08bdba",
        "#bae6ff",
        "#d12771",
        "#d4bbff",
        "#52ed1c",
        "#94695a",
        "#d6f55b",
        "#507b07",
        "#c039b9",
        "#fee296",
        "#677398",
        "#95a982",
        "#8b7122",
        "#ff936d",
        "#a06cc4",
        "#9efbda",
        "#c99cb0",
        "#b84c51",
        "#02e36a",
        "#fd8ff4",
    ),
}

# The core algorithms, in the order they take palette entries: most often in a
# released reading's leading ten first (release 2026-09-24), then every other
# core the library ships. A core not listed is drawn in unmapped ink, and the
# page says so rather than borrowing another core's colour.
CORE_ORDER: tuple[str, ...] = (
    "AMP",
    "SSP",
    "GaussianMechanism",
    "CoinPress",
    "DPSGD",
    "PLANBinarySearchGaussian",
    "AdaSSP",
    "AdaOPS",
    "OnePassDPGD",
    "DPGD",
    "BoostedAdaSSP",
    "DPRSS",
    "EfficientPTR",
    "GridExponentialMechanism",
    "IterativeHessianMixing",
    "OPSPTR",
    "TailRobustDPGD",
    "TailRobustSparseHuberIHT",
    "TukeyEM",
    "ZeroCoefficient",
    "LocalOutputPerturbation",
    "ObjectivePerturbationKifer",
    "ObjectivePerturbationRedberg",
)

# Each core's colour, by scheme.
CORE_COLOURS: dict[str, dict[str, str]] = {
    scheme: dict(zip(CORE_ORDER, entries)) for scheme, entries in PALETTE.items()
}

# The dash a curve takes from the class its claim is read under, in screen
# pixels and drawn with a non-scaling stroke so the pattern survives the chart
# being scaled down. A ranked claim is solid; a claim made but not ranked is a
# long dash; a run that is not a valid algorithm is a short dash; a nonprivate
# reference is a dash-dot, and takes neutral ink rather than a core's colour
# because it is a reference and not a competitor.
CLASS_DASH = {
    "standard": "solid",
    "qualified": "10 4",
    "theoretical": "10 4",
    "informal": "3 3",
    "diagnostic": "3 3",
    "nonprivate": "9 3 2 3",
}

# The class drawn in neutral reference ink instead of its core's colour.
REFERENCE_CLASS = "nonprivate"

# The shapes separating the transformation configurations of one core, taken by the
# configuration's position among that core's labels in name order. A core with
# more configurations than shapes stops separating them on the chart; the
# rankings table and the plotted-set strip still name every curve, and the page
# says so.
MARKER_SHAPES = ("circle", "square", "triangle-up", "diamond", "cross")
MARKER_SIZE = 8
# A curve shows at most this many of its markers, spread along it.
MARKERS_SHOWN = 6

# The threshold range a profile chart opens on, widened to reach the threshold
# it is read at, and the width of the guide marking that threshold.
PROFILE_RANGE = (0.0, 5.0)
GUIDE_WIDTH = 2
# Each profile axis reaches this far past what it must show: the share axis
# past 100%, the threshold axis past the threshold it is read at.
PROFILE_PAD = 1.05

# The five comparison slots, in selection order, reuse the shipped palette.
# They name configurations rather than cores; profiles keep CORE_COLOURS.
COMPARISON_COLOURS = {
    scheme: [entries[index] for index in (1, 4, 6, 3, 19)]
    for scheme, entries in PALETTE.items()
}

# Emphasis follows the current view: the leading rows of the reading now on
# screen are drawn at full strength in their core's colour and every other
# plotted curve is drawn in neutral context ink. A leading row is never muted,
# and a muted row is never drawn heavier than a leading one.
EMPHASISED_CURVES = 10
REGULAR_WIDTH = 2.5
MUTED_WIDTH = 1.5
REGULAR_OPACITY = 1.0
MUTED_OPACITY = 0.6


def marker_shape(position: int) -> str:
    """The shape one configuration takes from its position among its core's."""
    return MARKER_SHAPES[min(max(position, 0), len(MARKER_SHAPES) - 1)]


def marker_shapes_by_uuid(
    algorithms: list[dict[str, object]],
) -> dict[str, str]:
    """Assign each configuration the marker separating it within its core.

    Args:
        algorithms: The drawn configurations, each carrying its ``uuid``, its
            ``core`` and the ``label`` it is named by.
    """
    ordered: dict[str, list[tuple[str, str]]] = {}
    for entry in algorithms:
        core = str(entry.get("core") or "")
        ordered.setdefault(core, []).append(
            (str(entry.get("label") or ""), str(entry["uuid"]))
        )
    shapes: dict[str, str] = {}
    for configurations in ordered.values():
        configurations.sort()
        for position, (_label, uuid) in enumerate(configurations):
            shapes[uuid] = marker_shape(position)
    return shapes


def series_style() -> dict[str, object]:
    """The whole rule, as one payload the drawing surfaces read it from."""
    return {
        "colours": {scheme: dict(table) for scheme, table in CORE_COLOURS.items()},
        "comparisonColours": COMPARISON_COLOURS,
        "surface": dict(PLOT_SURFACE),
        "dash": dict(CLASS_DASH),
        "referenceClass": REFERENCE_CLASS,
        "markers": list(MARKER_SHAPES),
        "markerSize": MARKER_SIZE,
        "markersShown": MARKERS_SHOWN,
        "range": list(PROFILE_RANGE),
        "guideWidth": GUIDE_WIDTH,
        "pad": PROFILE_PAD,
        "emphasis": {
            "curves": EMPHASISED_CURVES,
            "regularWidth": REGULAR_WIDTH,
            "mutedWidth": MUTED_WIDTH,
            "regularOpacity": REGULAR_OPACITY,
            "mutedOpacity": MUTED_OPACITY,
        },
    }
