"""Give every table on a page its own named scroll region.

A table wider than the window scrolls inside a region of its own, never the
page (style guide W18). Generated tables arrive in the shared wrapper; a
table authored in Markdown, a notebook output or an API reference does not, so
this hook wraps it. Every wrapper is a region named by the heading it sits
under, so a reader who scrolls it knows which table it is; ``site.js`` makes a
region a keyboard stop only while its table overflows.
"""

from __future__ import annotations

import html
import re
from typing import Any

from site_copy import copy_text

WRAPPER = '<div class="dp-table-scroll table-responsive">'
_WRAPPED_BEFORE = re.compile(r'<div class="dp-table-scroll[^"]*">\s*$')
_TABLE = re.compile(r"<table\b.*?</table>", re.S)
_TAG = re.compile(r"<[^>]+>")
_PIECE = re.compile(
    r"<h(?P<level>[1-4])\b[^>]*>(?P<heading>.*?)</h(?P=level)>"
    r'|<div class="dp-table-scroll(?P<extra>[^"]*)">',
    re.S,
)


def wrap_tables(page_html: str) -> str:
    """Wrap each table that is not already in the shared scroll wrapper."""

    def wrap(match: re.Match[str]) -> str:
        before = page_html[max(0, match.start() - 200) : match.start()]
        if _WRAPPED_BEFORE.search(before):
            return match[0]
        return f"{WRAPPER}\n{match[0]}\n</div>"

    return _TABLE.sub(wrap, page_html)


def name_regions(page_html: str, title: str) -> str:
    """Mark each scroll wrapper a region named by the heading above it."""
    heading = title

    def mark(match: re.Match[str]) -> str:
        nonlocal heading
        if match["heading"] is not None:
            text = " ".join(html.unescape(_TAG.sub("", match["heading"])).split())
            heading = text.rstrip("¶ ").strip() or heading
            return match[0]
        label = html.escape(
            copy_text("site", "table_region", fields={"heading": heading}), quote=True
        )
        return f'<div class="dp-table-scroll{match["extra"]}" role="region" aria-label="{label}">'

    return _PIECE.sub(mark, page_html)


def on_page_content(html_text: str, page: Any = None, **_: Any) -> str:
    """Wrap and name the page's tables."""
    title = getattr(page, "title", "") or copy_text("site", "untitled_page")
    return name_regions(wrap_tables(html_text), str(title))
