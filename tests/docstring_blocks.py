"""Read the scientific documentation sections used by the reference pages."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The renderer's own heading recognizer: any capitalized ``Name:`` line at the
# docstring's own indentation is a Google-style section to it, whatever the
# name, so the gate reads every heading through the same regex rather than a
# second, narrower guess at what counts as a section.
_SITE_DIR = Path(__file__).resolve().parent.parent / "docs" / "site"
if str(_SITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SITE_DIR))
from templates.library_source import (  # noqa: E402
    _BLOCK as _RENDERED_SECTION_HEADER,  # pyright: ignore[reportPrivateUsage]
)

REFERENCE_HEADER = re.compile(r"^([ \t]*)Reference:[ \t]*$", re.MULTILINE)
REQUIRES_HEADER = re.compile(r"^([ \t]*)Requires:[ \t]*$", re.MULTILINE)
PARAMETRIZATION_NOTES_HEADER = re.compile(
    r"^([ \t]*)Parametrization notes:[ \t]*$", re.MULTILINE
)
IMPLEMENTATION_NOTES_HEADER = re.compile(
    r"^([ \t]*)Implementation notes:[ \t]*$", re.MULTILINE
)


def named_block(doc: str, header: re.Pattern[str]) -> str | None:
    """Return the text of one named docstring block, or ``None`` if absent."""
    match = header.search(doc)
    if match is None:
        return None
    indent = len(match.group(1))
    lines: list[str] = []
    for line in doc[match.end() :].splitlines()[1:]:
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        if stripped and current_indent <= indent:
            break
        lines.append(line)
    return "\n".join(lines)


def named_block_headers(doc: str) -> list[str]:
    """Return every heading the renderer would treat as a named block.

    Uses the exact heading regex ``docs/site/templates/library_source.py``
    parses sections with, so a heading the gate does not recognize can never
    be one the renderer silently accepts as a block anyway.
    """
    return [match["name"] for match in _RENDERED_SECTION_HEADER.finditer(doc)]


def named_block_shape_violations(doc: str) -> list[str]:
    """Report duplicate sections that the reference renderer would overwrite."""
    names = named_block_headers(doc)
    return [
        f"{name!r} appears more than once"
        for name in dict.fromkeys(names)
        if names.count(name) > 1
    ]
