"""The one treatment every choice control on the site shares.

Every select a page renders carries ``SELECT_CLASS``, whichever page or
script builds it, so no page draws a select of its own; the built-site check
refuses one that does not.
"""

from html import escape

from site_copy import copy_span

SELECT_CLASS = "dp-select form-select form-select-sm w-auto"

# A choice of several (independent filters) is a row of toggle chips while it
# stays this short, and past either bound a checkbox list; each option's count
# is supporting text beside it, never part of its name.
CHIP_MAX_OPTIONS = 6
CHIP_MAX_LABEL = 16


def fits_chips(labels: list[str]) -> bool:
    """Whether a choice of several fits the toggle-chip rule."""
    return len(labels) <= CHIP_MAX_OPTIONS and all(
        len(label) <= CHIP_MAX_LABEL for label in labels
    )


def fixed_value(
    legend: tuple[str, str],
    text: str,
    *,
    attributes: str = "",
    text_marker: str = "",
    extra: str = "",
) -> str:
    """A value the benchmark fixes, as its label over plain text in a control's
    place: never a disabled control, which reads as a choice the reader cannot
    make.

    Args:
        legend: The site-copy slot, as ``(group, key)``, of the value's label.
        text: The value as the reader reads it.
        attributes: Extra attributes on the wrapper, the hook a script reads.
        text_marker: An attribute on the text itself, for a script that
            rewrites it as the page changes.
        extra: Markup after the text, such as the state a script reads.
    """
    marker = f" {text_marker}" if text_marker else ""
    return (
        f'<div class="dp-fixed-value"{attributes}>'
        f'<span class="dp-chip-label">{copy_span(*legend)}</span>'
        f'<span class="dp-fixed-text"{marker}>{escape(text)}</span>{extra}</div>'
    )


def setting(head: str, description: str, *, classes: str, attributes: str = "") -> str:
    """One results setting as its own box: its label beside its control (or
    its fixed value), then one line saying what the current choice means.
    Explore and the Leaderboard share it; only the control differs."""
    return (
        f'<div class="dp-setting {classes}"{attributes}>'
        f'<div class="dp-setting-head">{head}</div>{description}</div>'
    )
