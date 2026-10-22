"""Resolve every DOI and arXiv link in the bibliography, docstrings and docs.

Network, opt-in. Each DOI is looked up on Crossref (falling back to doi.org for
other registrars) and each arXiv link on its abs page. A ``references.bib``
entry's title, or a quoted title just before a link in prose, is compared with
the resolved title.

Usage:
    python tools/check_references.py [--threshold 0.6]
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from typing import Any, cast

import pybtex.database
import pybtex.richtext

REPO_ROOT = Path(__file__).parent.parent
SCAN_ROOTS = (REPO_ROOT / "dpstats", REPO_ROOT / "docs")
SCAN_SUFFIXES = {".py", ".md"}
BIBLIOGRAPHY = REPO_ROOT / "dpstats" / "references.bib"

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s)>\]\"'`,;]+)")
_ARXIV = re.compile(
    r"arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
_TITLE = re.compile(r"[\"“]([^\"”]{12,300}?)[.,]?[\"”]")
_TITLE_WINDOW = 400


@dataclass(frozen=True)
class Reference:
    kind: str  # "doi" or "arxiv"
    identifier: str
    cited_title: str | None
    location: str


def _clean(identifier: str) -> str:
    return identifier.rstrip(".")


def parse_references(text: str, location: str) -> list[Reference]:
    """Return every DOI and arXiv link in ``text`` with its nearby quoted title."""
    found: list[tuple[int, str, str]] = []
    for match in _ARXIV.finditer(text):
        found.append((match.start(), "arxiv", _clean(match.group(1))))
    for match in _DOI.finditer(text):
        found.append((match.start(), "doi", _clean(match.group(1))))
    references: list[Reference] = []
    for start, kind, identifier in sorted(found):
        if kind == "doi" and identifier.lower().startswith("10.48550/arxiv."):
            kind, identifier = "arxiv", identifier.split(".", 2)[2]
        window = text[max(0, start - _TITLE_WINDOW) : start].rsplit('"""', 1)[-1]
        titles = _TITLE.findall(re.sub(r"\s+", " ", window))
        cited = titles[-1] if titles else None
        references.append(Reference(kind, identifier, cited, location))
    return references


def title_similarity(cited: str, resolved: str) -> float:
    """Return a 0-1 similarity of two titles, ignoring case and punctuation."""

    def normal(title: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split())

    return difflib.SequenceMatcher(None, normal(cited), normal(resolved)).ratio()


def _get(url: str) -> tuple[int, str]:
    """GET with retries on rate limits, server errors and network failures."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "dpstats-reference-check"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                return error.code, ""
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5 * (attempt + 1))
    return 0, ""


def resolve_doi(doi: str) -> tuple[bool, str | None]:
    """Return (resolves, title); a non-Crossref DOI resolves without a title."""
    status, body = _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if status == 200:
        titles = json.loads(body)["message"].get("title") or [None]
        return True, titles[0]
    status, _ = _get("https://doi.org/api/handles/" + urllib.parse.quote(doi))
    return status == 200, None


def resolve_arxiv(identifier: str) -> tuple[bool, str | None]:
    """Return (resolves, title) from the arXiv abs page."""
    status, body = _get("https://arxiv.org/abs/" + identifier)
    if status != 200:
        return False, None
    match = re.search(r"<title>\s*\[[^\]]+\]\s*(.*?)\s*</title>", body, re.DOTALL)
    return True, html.unescape(" ".join(match.group(1).split())) if match else None


def collect() -> list[Reference]:
    references: list[Reference] = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in SCAN_SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[:3] == ("docs", "site", "templates"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for reference in parse_references(text, str(relative)):
                line = text.count("\n", 0, text.find(reference.identifier)) + 1
                references.append(
                    Reference(
                        reference.kind,
                        reference.identifier,
                        reference.cited_title,
                        f"{relative}:{line}",
                    )
                )
    return references


def collect_bibliography(text: str, location: str) -> list[Reference]:
    """Every DOI and arXiv identifier the bibliography's entries carry, with titles."""
    database = cast(Any, pybtex.database.parse_string(text, "bibtex"))
    entries: dict[str, Any] = dict(database.entries)
    references: list[Reference] = []
    for key, entry in entries.items():
        title = str(pybtex.richtext.Text.from_latex(entry.fields["title"]))
        if "doi" in entry.fields:
            references.append(
                Reference("doi", entry.fields["doi"], title, f"{location}:{key}")
            )
        if entry.fields.get("archivePrefix") == "arXiv":
            references.append(
                Reference("arxiv", entry.fields["eprint"], title, f"{location}:{key}")
            )
    return references


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.6)
    threshold = parser.parse_args().threshold
    references = (
        collect_bibliography(
            BIBLIOGRAPHY.read_text(encoding="utf-8"),
            str(BIBLIOGRAPHY.relative_to(REPO_ROOT)),
        )
        + collect()
    )
    cache: dict[tuple[str, str], tuple[bool, str | None]] = {}
    unresolved: list[Reference] = []
    low: list[tuple[float, Reference, str]] = []
    for reference in references:
        key = (reference.kind, reference.identifier)
        if key not in cache:
            resolver = resolve_doi if reference.kind == "doi" else resolve_arxiv
            cache[key] = resolver(reference.identifier)
            time.sleep(3.0)
        resolves, title = cache[key]
        if not resolves:
            unresolved.append(reference)
        elif title and reference.cited_title:
            score = title_similarity(reference.cited_title, title)
            if score < threshold:
                low.append((score, reference, title))
    print(f"{len(references)} references, {len(cache)} distinct")
    print(f"{len(unresolved)} unresolvable")
    for reference in unresolved:
        print(f"  {reference.location}: {reference.kind} {reference.identifier}")
    print(f"{len(low)} low title similarity (< {threshold})")
    for score, reference, title in low:
        print(
            f"  {reference.location}: {reference.kind} {reference.identifier} "
            f"score={score:.2f}\n    cited:    {reference.cited_title}\n    resolved: {title}"
        )
    return 1 if unresolved or low else 0


if __name__ == "__main__":
    sys.exit(main())
