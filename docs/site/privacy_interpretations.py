"""Each privacy interpretation's reading, from the library's own documentation.

``PrivacyInterpretation`` documents its members as one bullet each in its own
docstring (``- ``NAME``: reading``). A badge's hover text is that reading as
plain text, so the site states a class exactly as the library does and holds
no wording of its own. A badge whose class is missing carries no hover text.
"""

from __future__ import annotations

import html
import json
import re
from inspect import cleandoc

from dpstats.algorithm import PrivacyInterpretation

# One member's bullet: its name, then its reading up to the next bullet or a
# blank line.
_BULLET = re.compile(r"^- ``([A-Z_]+)``:\s*(.*?)(?=^- ``|^\s*$|\Z)", re.M | re.S)
_LITERAL = re.compile(r"``([^`]+)``")


def _readings() -> dict[str, str]:
    """Every member's reading, keyed by its value, as plain text."""
    found = {
        name: " ".join(_LITERAL.sub(r"\1", reading).split())
        for name, reading in _BULLET.findall(
            cleandoc(PrivacyInterpretation.__doc__ or "")
        )
    }
    readings = {
        member.value: found.get(member.name, "") for member in PrivacyInterpretation
    }
    undocumented = sorted(value for value, reading in readings.items() if not reading)
    if undocumented:
        raise ValueError(
            "PrivacyInterpretation documents no reading for " + ", ".join(undocumented)
        )
    return readings


INTERPRETATION_TEXT = _readings()


def interpretation_text(name: str) -> str:
    """The hover text on *name*'s badge, or ``""`` for a missing class."""
    return INTERPRETATION_TEXT.get(name.lower(), "")


def interpretation_badge(name: str, kind: str) -> str:
    """*name*'s badge with classes *kind*, its reading as hover text."""
    reading = interpretation_text(name)
    title = f' title="{html.escape(reading, quote=True)}"' if reading else ""
    return f'<span class="{kind}"{title}>{html.escape(name)}</span>'


def interpretation_attribute() -> str:
    """The ``data-privacy-interpretations`` attribute a page's scripts read."""
    payload = json.dumps(INTERPRETATION_TEXT, ensure_ascii=False)
    return f' data-privacy-interpretations="{html.escape(payload, quote=True)}"'
