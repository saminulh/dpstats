"""Render the source markup a docstring carries as the page's own markup.

Docstrings are written for a reader of the source, so they carry three shapes a
page must not show literally: reStructuredText cross-reference roles
(``:class:`Thing```), double-backtick literals (``` ``value`` ```), and the
line wrapping the source's width imposes. The first two reach a page through
several routes — this build's own renderers, and the API reference the
documentation theme renders from the same docstrings — so the conversion runs
once over each finished page rather than in each renderer; the third is
undone as a renderer reads each block.

A role's target is set as code, and a ``~`` prefix keeps only the last segment,
which is the convention the docstrings are written to.
"""

from __future__ import annotations

import re
from typing import Any

# The cross-reference roles the docstrings in this package use.
ROLES = ("class", "func", "meth", "mod", "attr", "data", "exc", "obj")
_ROLE_NAMES = "|".join(ROLES)

# The same role after the page is rendered: the target is already a code span.
_ROLE_HTML = re.compile(rf":(?:{_ROLE_NAMES}):(?=<code)")
# A target that survived as plain text, with no code span around it.
_ROLE_BARE = re.compile(rf":(?:{_ROLE_NAMES}):~?([A-Za-z_][\w.]*)")
# What a ``~`` prefix leaves: the last dotted segment of the target.
_QUALIFIED = re.compile(r"<code>~([\w.]+)</code>")
# A double-backtick literal that reached the page as text.
_LITERAL = re.compile(r"``([^`]+)``")
# Blocks whose contents are shown verbatim and so are never rewritten.
_VERBATIM = re.compile(r"<pre\b.*?</pre>", re.S)


def joined_block(text: str) -> str:
    """One docstring block as the paragraphs and bullets it was written as.

    A docstring wraps its lines to the source's width, so each run of lines is
    joined back into the one item it states, and a blank line or a new bullet
    is what starts the next one. A wrap inside a ``literal`` closes up with no
    space, because the literal is one word the writer had to break.

    Example:
        >>> joined_block("A sentence that\\nwrapped.\\n\\n- One bullet.")
        'A sentence that wrapped.\\n\\n- One bullet.'
    """
    items: list[str] = []
    wrapping = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            wrapping = False
        elif wrapping and not stripped.startswith("- "):
            inside = items[-1].count("``") % 2 == 1
            items[-1] += ("" if inside else " ") + stripped
        else:
            items.append(stripped)
            wrapping = True
    written = items[:1]
    for previous, item in zip(items, items[1:]):
        listed = previous.startswith("- ") and item.startswith("- ")
        written.append(("\n" if listed else "\n\n") + item)
    return "".join(written)


def page_text(html: str) -> str:
    """One rendered page, with the source markup its docstrings carried set.

    Verbatim blocks are left exactly as they are: a code sample may legitimately
    contain backticks or a colon-delimited word.
    """
    verbatim: list[str] = []

    def hold(match: re.Match[str]) -> str:
        verbatim.append(match[0])
        return f"\x00{len(verbatim) - 1}\x00"

    held = _VERBATIM.sub(hold, html)
    held = _ROLE_HTML.sub("", held)
    held = _ROLE_BARE.sub(lambda match: f"<code>{_last(match[1])}</code>", held)
    held = _QUALIFIED.sub(
        lambda match: f"<code>{match[1].rsplit('.', 1)[-1]}</code>", held
    )
    held = _LITERAL.sub(lambda match: f"<code>{match[1]}</code>", held)
    return re.sub(r"\x00(\d+)\x00", lambda match: verbatim[int(match[1])], held)


def _last(target: str) -> str:
    """The segment a reader reads: the whole target, or its last dotted part."""
    if target.startswith("~"):
        return target.lstrip("~").rsplit(".", 1)[-1]
    return target


def on_page_content(html: str, page: Any = None, **kwargs: Any) -> str:
    """The build hook: run over every page the site renders."""
    return page_text(html)
