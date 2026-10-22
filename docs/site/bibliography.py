"""The library's bibliography and the one citation style the site renders it in.

Every cited work lives once in ``dpstats/references.bib``; a class's
``Reference:`` block names works by key, each with the part of the work the
class follows::

    Reference:
        - wang2018revisiting: Algorithm 2.

How a citation reads is decided only by :class:`AuthorYearStyle`; replace
``STYLE`` to change every citation on the site at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from html import escape
from importlib.resources import files

from pybtex.database import BibliographyData, Entry, Person, parse_string
from pybtex.richtext import Text

# One cited work in a ``Reference:`` block: ``- key: locator.``, or ``- key.``
# for a work the class follows as a whole. A locator may wrap onto indented
# continuation lines.
_CITED = re.compile(r"^- (?P<key>[a-z0-9]+)(?:: (?P<locator>.+?))?\.$", re.S)


@dataclass(frozen=True)
class CitedWork:
    """One bullet of a ``Reference:`` block: a bibliography key and its locator."""

    key: str
    locator: str


def cited_works(block: str) -> list[CitedWork]:
    """The works one ``Reference:`` block cites, in order.

    Raises:
        ValueError: A bullet is not ``- key: locator.`` or ``- key.``.
    """
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped)
        elif stripped and items:
            items[-1] += " " + stripped
        elif stripped:
            raise ValueError(f"Reference: text outside a bullet: {stripped!r}")
    works = []
    for item in items:
        match = _CITED.match(item)
        if match is None:
            raise ValueError(f"Reference: bullet is not '- key: locator.': {item!r}")
        works.append(CitedWork(match["key"], match["locator"] or ""))
    return works


def cited_by(doc: str) -> list[CitedWork]:
    """The works a dedented docstring's ``Reference:`` block cites; none without one."""
    lines = doc.splitlines()
    if "Reference:" not in lines:
        return []
    body = []
    for line in lines[lines.index("Reference:") + 1 :]:
        if line.strip() and not line[0].isspace():
            break
        body.append(line)
    return cited_works("\n".join(body))


def references_bib() -> str:
    """The text of the bibliography the library ships."""
    return (files("dpstats") / "references.bib").read_text(encoding="utf-8")


@cache
def _library() -> BibliographyData:
    return parse_string(references_bib(), "bibtex")


def load(text: str | None = None) -> BibliographyData:
    """The library's bibliography, or one parsed from *text* (a test fixture)."""
    return _library() if text is None else parse_string(text, "bibtex")


def _plain(latex: str) -> str:
    """BibTeX field text as a reader sees it: accents decoded, braces dropped."""
    return str(Text.from_latex(latex))


def _surname(person: Person) -> str:
    return _plain(" ".join(person.prelast_names + person.last_names))


def _initials(person: Person) -> str:
    """``Yu-Xiang`` -> ``Y.-X.``; ``H. Brendan`` -> ``H. B.``."""
    names = [_plain(name) for name in person.first_names + person.middle_names]
    return " ".join(
        "-".join(part[0] + "." for part in name.split("-") if part) for name in names
    )


def _series(items: list[str], comma: bool) -> str:
    """``A``, ``A and B``, ``A, B and C`` (``A, B, and C`` with *comma*)."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ("," if comma else "") + " and " + items[-1]


class AuthorYearStyle:
    """The provisional citation style: author-year, full author lists.

    ``short`` is the in-text form a sentence or a table cell cites; ``full`` is
    one bibliography entry as HTML; ``link`` is the one address a bare link to
    the work uses.
    """

    # From this many authors on, the in-text form names only the first.
    ET_AL_FROM = 5

    def short(self, entry: Entry) -> str:
        """``Wang (2018)``, ``Dick, Gillenwater and Joseph (2023)``, ``Alabi et al. (2022)``."""
        people = entry.persons["author"]
        if len(people) >= self.ET_AL_FROM:
            names = f"{_surname(people[0])} et al."
        else:
            names = _series([_surname(person) for person in people], comma=False)
        return f"{names} ({entry.fields['year']})"

    def full(self, entry: Entry) -> str:
        """One entry as HTML: authors, year, title, venue, then its identifiers."""
        fields = entry.fields
        authors = _series(
            [f"{_surname(p)}, {_initials(p)}" for p in entry.persons["author"]],
            comma=True,
        )
        title = _plain(fields["title"])
        parts = [
            f"{escape(authors)} ({escape(fields['year'])}).",
            verbatim(escape(title)) + ("" if title.endswith(("?", "!", ".")) else "."),
        ]
        venue = self._venue(entry)
        if venue:
            parts.append(venue + ".")
        links = self._links(entry)
        if links:
            parts.append(", ".join(links) + ".")
        return " ".join(parts)

    def link(self, entry: Entry) -> str:
        """The work's DOI, else its publisher page, else its arXiv page."""
        fields = entry.fields
        if "doi" in fields:
            return f"https://doi.org/{fields['doi']}"
        if _publisher_url(entry):
            return _publisher_url(entry)
        if "eprint" in fields:
            return f"https://arxiv.org/abs/{fields['eprint']}"
        return fields["url"]

    def _venue(self, entry: Entry) -> str:
        fields = entry.fields
        pages = _plain(fields.get("pages", "")).replace("--", "–")
        # A registrar may type a working paper @article without a journal.
        if entry.type == "article" and "journal" in fields:
            venue = f"<em>{escape(_plain(fields['journal']))}</em>"
            if "volume" in fields:
                venue += ", " + escape(fields["volume"])
                if "number" in fields:
                    venue += f"({escape(fields['number'])})"
            return venue + (f", {escape(pages)}" if pages else "")
        if entry.type in ("inproceedings", "incollection"):
            venue = f"In <em>{escape(_plain(fields['booktitle']))}</em>"
            if "volume" in fields:
                venue += f", vol. {escape(fields['volume'])}"
            if pages:
                venue += f", pp. {escape(pages)}"
            if entry.type == "incollection" and "publisher" in fields:
                venue += f". {escape(_plain(fields['publisher']))}"
            return venue
        if "howpublished" in fields:
            return escape(_plain(fields["howpublished"]))
        if entry.type == "software" and "version" in fields:
            return f"Version {escape(fields['version'])}"
        return "Preprint" if "eprint" in fields else ""

    def _links(self, entry: Entry) -> list[str]:
        fields = entry.fields
        links = []
        if "doi" in fields:
            doi = escape(fields["doi"])
            links.append(f'<a href="https://doi.org/{doi}">doi:{doi}</a>')
        elif _publisher_url(entry):
            url = _publisher_url(entry)
            host = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
            links.append(f'<a href="{escape(url)}">{escape(host)}</a>')
        if "eprint" in fields:
            eprint = escape(fields["eprint"])
            links.append(f'<a href="https://arxiv.org/abs/{eprint}">arXiv:{eprint}</a>')
        return links


class Apa7Style(AuthorYearStyle):
    """APA 7th edition reference entries; the in-text form stays the site's.

    Entries list up to 20 authors as ``Surname, I.`` with ``&`` before the
    last, the year, the title as published (corpus titles carry no protected
    capitals, so APA's sentence case would lower-case proper nouns), the venue
    with the journal or proceedings in italics, and end with the DOI as a link.
    A work with no DOI ends with its publisher page, or its arXiv page when it
    has no published version.
    """

    def full(self, entry: Entry) -> str:
        """One entry as HTML, in APA 7 order."""
        fields = entry.fields
        people = [f"{_surname(p)}, {_initials(p)}" for p in entry.persons["author"]]
        if len(people) > 20:
            people = people[:19] + ["…", people[-1]]
            authors = ", ".join(people)
        elif len(people) == 1:
            authors = people[0]
        else:
            authors = ", ".join(people[:-1]) + ", & " + people[-1]
        title = _plain(fields["title"])
        title_html = escape(title)
        venue = self._venue(entry)
        standalone = not ("journal" in fields or "booktitle" in fields)
        if standalone and "howpublished" not in fields:
            title_html = f"<em>{title_html}</em>"
        parts = [
            f"{escape(authors)} ({escape(fields['year'])}).",
            verbatim(title_html) + ("" if title.endswith(("?", "!", ".")) else "."),
        ]
        if venue:
            parts.append(venue + ("" if venue.endswith(".") else "."))
        elif "eprint" in fields and "doi" not in fields:
            parts.append("arXiv.")
        link = self.link(entry)
        if link:
            parts.append(f'<a href="{escape(link)}">{escape(link)}</a>')
        return " ".join(parts)

    def _venue(self, entry: Entry) -> str:
        fields = entry.fields
        pages = _plain(fields.get("pages", "")).replace("--", "–")
        if entry.type == "article" and "journal" in fields:
            venue = f"<em>{escape(_plain(fields['journal']))}</em>"
            if "volume" in fields:
                venue += f", <em>{escape(fields['volume'])}</em>"
                if "number" in fields:
                    venue += f"({escape(fields['number'])})"
            return venue + (f", {escape(pages)}" if pages else "")
        if entry.type in ("inproceedings", "incollection"):
            venue = f"In <em>{escape(_plain(fields['booktitle']))}</em>"
            details = []
            if "volume" in fields:
                details.append(f"Vol. {escape(fields['volume'])}")
            if pages:
                details.append(f"pp. {escape(pages)}")
            if details:
                venue += f" ({', '.join(details)})"
            if "publisher" in fields:
                venue += f". {escape(_plain(fields['publisher']))}"
            return venue
        if "howpublished" in fields:
            return escape(_plain(fields["howpublished"]))
        if entry.type == "software" and "version" in fields:
            return f"(Version {escape(fields['version'])}) [Computer software]"
        if "publisher" in fields:
            return escape(_plain(fields["publisher"]))
        return ""


def verbatim(title_html: str) -> str:
    """A work's title as its source states it, marked so the site's own
    style checks leave its capitals alone."""
    return f"<span data-verbatim>{title_html}</span>"


def _publisher_url(entry: Entry) -> str:
    """The entry's own ``url`` when it names the published version, not arXiv."""
    url = entry.fields.get("url", "")
    return "" if "arxiv.org/" in url else url


STYLE = Apa7Style()


def short_citation(key: str, bibliography: BibliographyData | None = None) -> str:
    """The in-text citation of one work, in the site's style."""
    return STYLE.short(load_entry(key, bibliography))


def source_text(
    works: list[CitedWork], bibliography: BibliographyData | None = None
) -> str:
    """The works a class follows, as a sentence reads them, without its verb.

    Each work is ``<locator> from <citation>``, or its citation alone where
    the class follows the work as a whole. The first work with a locator is
    the one followed; the rest are joined to it with "with". Works none of
    which names a locator are listed alike::

        Algorithm 1 from Iyengar et al. (2019), with Theorems 3.2 and 4.1
        from Redberg, Koskela and Wang (2023)
        Frisch and Waugh (1933) and Lovell (1963)

    Returns:
        HTML-escaped text; empty for no works.
    """
    parts = [
        (
            f"{escape(work.locator)} from {escape(short_citation(work.key, bibliography))}"
            if work.locator
            else escape(short_citation(work.key, bibliography))
        )
        for work in works
    ]
    if len(parts) < 2:
        return "".join(parts)
    if works[0].locator:
        return parts[0] + ", with " + _series(parts[1:], comma=False)
    return _series(parts, comma=False)


def full_citation(key: str, bibliography: BibliographyData | None = None) -> str:
    """One work's bibliography entry as HTML, in the site's style."""
    return STYLE.full(load_entry(key, bibliography))


def work_link(key: str, bibliography: BibliographyData | None = None) -> str:
    """The one address a bare link to the work uses."""
    return STYLE.link(load_entry(key, bibliography))


def load_entry(key: str, bibliography: BibliographyData | None = None) -> Entry:
    """One entry of *bibliography* (default: the library's).

    Raises:
        KeyError: No entry has *key*.
    """
    entries = (bibliography or load()).entries
    if key not in entries:
        raise KeyError(f"{key!r} is not in references.bib")
    return entries[key]
