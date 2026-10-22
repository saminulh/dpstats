"""Read an area's own pages off the navigation this build declares.

Library and Submit land on a list of the pages they carry. The
navigation already says which pages those are, so the landing is derived from
it rather than restated, and each entry carries the sentence its page opens on
rather than a sentence written for the landing.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from typing import Any

from templates.model import Link

# A block that opens with any of these is markup, not the page's first
# sentence: a heading, a list, a table row, raw HTML, an admonition, a
# definition-list term, emphasis-only copy, or a code fence.
NOT_PROSE = "#-|<!:*"

# A fenced block is one block however many blank lines it holds, so its own
# lines are never read as the page's opening sentence.
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def page_summary(text: str) -> str:
    """The sentence one page opens on, or an empty string for none.

    Args:
        text: The page's Markdown, with or without its front matter.

    Returns:
        The first sentence of the page's first prose paragraph. A page that
        opens on a heading, table, list or code block, or on a line that ends
        by introducing one, declares no sentence, and the caller leaves its
        entry's description empty rather than writing one.
    """
    body = text.split("\n---\n", 1)[-1] if text.startswith("---\n") else text
    for block in re.split(r"\n\s*\n", FENCE.sub("", body)):
        line = " ".join(block.split())
        if not line or line[0] in NOT_PROSE or line.endswith(":"):
            continue
        sentence, stop, _ = line.partition(". ")
        return sentence + "." if stop else line
    return ""


def _document(target: str) -> str:
    """The document one navigation target names, directory targets included."""
    return target if target.endswith(".md") else f"{target}index.md"


def first_page(entries: Sequence[Any]) -> str:
    """The first document one navigation branch names, depth first."""
    for entry in entries:
        for target in entry.values():
            if isinstance(target, str):
                return _document(target)
            found = first_page(target)
            if found:
                return found
    return ""


def branch_for(nav: Sequence[Any], page: str) -> list[Any]:
    """The navigation branch whose first document is *page*, or an empty list."""
    for entry in nav:
        for target in entry.values():
            if not isinstance(target, list):
                continue
            if first_page(target) == page:
                return target
            found = branch_for(target, page)
            if found:
                return found
    return []


def area_pages(branch: Sequence[Any], area: str) -> tuple[Link, ...]:
    """Every page *area*'s navigation declares, beside its own landing.

    Args:
        branch: The area's navigation branch.
        area: The area's directory, which its landing is the index of.

    Returns:
        Each entry's link, measured from the landing. A group entry names the
        first page under it, so a grouped section is one row rather than none.
    """
    pages: list[Link] = []
    for entry in branch:
        for label, target in entry.items():
            page = first_page(target) if isinstance(target, list) else _document(target)
            if not page or page == f"{area}/index.md":
                continue
            pages.append(Link(label, posixpath.relpath(page, area)))
    return tuple(pages)
