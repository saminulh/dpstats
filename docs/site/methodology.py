"""Render the site-wide Methodology page from ``docs/content/methodology.md``.

The page renders that narrative verbatim under its own title, whichever
release the build reads; a release ships a copy of it taken when it was cut.
A tuned configured algorithm's page links its tuning section.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown import Markdown

from templates.model import Methodology

# Where the page is published, from the site root.
METHODOLOGY_ROUTE = "methodology/"
# Where the narrative lives, from the library's docs directory.
METHODOLOGY_SOURCE = Path("content") / "methodology.md"


# A Markdown link in the narrative. A relative target names a file beside the
# narrative inside the release, which the site does not publish, so the link
# shows as its text and the build names it; a web address or a fragment stays
# a link.
_LINK = re.compile(r"(?<!!)\[(?P<text>[^\]\n]+)\]\((?P<url>[^)\s]+)\)")
_SCHEME = re.compile(r"^[A-Za-z][\w+.-]*:")


def release_relative_links(narrative: str) -> tuple[str, ...]:
    """The narrative's link targets that name a file inside the release."""
    return tuple(
        match["url"] for match in _LINK.finditer(narrative) if _relative(match["url"])
    )


def _relative(url: str) -> bool:
    return not url.startswith(("#", "/")) and not _SCHEME.match(url)


def _unlink(narrative: str) -> str:
    """*narrative* with each release-relative link shown as its text alone."""
    return _LINK.sub(
        lambda match: match["text"] if _relative(match["url"]) else match[0],
        narrative,
    )


# A heading's leading section number, as in "4. The algorithms roster".
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
# The sections a tuned configured algorithm's page links, most specific first.
_TUNING_SECTIONS = ("tuning", "the algorithms roster")


def tuning_fragment(narrative: str | None) -> str:
    """The Methodology page fragment a tuned configured algorithm's page links.

    The narrative's Tuning heading where it has one, else its "The algorithms
    roster" heading, else none, so the link always lands on the page. Each id
    is the one MkDocs' toc extension gives the heading on the page.
    """
    toc = Markdown(extensions=["toc"])
    toc.convert(narrative or "")
    ids: dict[str, str] = {}
    pending = list(toc.toc_tokens)
    while pending:
        heading = pending.pop(0)
        pending[0:0] = heading["children"]
        ids.setdefault(
            _SECTION_NUMBER.sub("", heading["name"]).casefold(), heading["id"]
        )
    for section in _TUNING_SECTIONS:
        if section in ids:
            return f"#{ids[section]}"
    return ""


def split_title(narrative: str) -> tuple[str, str]:
    """The narrative's own title and the text under it, unchanged.

    The narrative opens on its one ``#`` heading, which becomes the page's
    title; everything after it renders as the narrative wrote it.

    Raises:
        ValueError: If the narrative does not open on a ``#`` heading.
    """
    first, _, rest = narrative.lstrip("\ufeff").lstrip("\n").partition("\n")
    if not first.startswith("# "):
        raise ValueError(
            "The release's protocol narrative does not open on its own "
            f"'# ' title: {first[:60]!r}."
        )
    return first[2:].strip(), rest.lstrip("\n")


def methodology_page(narrative: str, identity: str, *, banner: str = "") -> Methodology:
    """The T15 record: the methodology narrative, verbatim.

    The narrative renders as written, except that a relative link shows as its
    text: the site publishes no file beside the narrative
    (:func:`release_relative_links` names them for the build's output).

    Args:
        narrative: The methodology narrative.
        identity: The release the page was built from, as the reader is told,
            or empty where the build reads none.
        banner: The status line a build reading no release shows.
    """
    title, body = split_title(narrative)
    return Methodology(
        title=title, narrative=_unlink(body), identity=identity, banner=banner
    )


def read_methodology(docs: Path) -> str:
    """The methodology narrative under the library's *docs* directory."""
    return (docs / METHODOLOGY_SOURCE).read_text(encoding="utf-8")
