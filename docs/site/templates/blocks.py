"""The shared spellings every generated page uses for one kind of fragment.

One spelling per fragment: a code cell, an inline constraint, a citation, a
table. The installed-library source and the filler registry both build their
records through these, so a mock page and a real page differ in their values
and never in their markup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from bibliography import CitedWork, full_citation
from pybtex.database import BibliographyData


def code(value: object) -> str:
    """One value as an inline code cell."""
    return "<code>" + escape(str(value), quote=False) + "</code>"


def math(expression: str) -> str:
    """One constraint as inline LaTeX math, as the style guide sets them."""
    return '<span class="arithmatex">\\( ' + expression + " \\)</span>"


def display_math(expression: str) -> str:
    """One formula as display LaTeX math, on its own lines."""
    return "$$\n" + expression + "\n$$"


def math_name(name: str) -> str:
    """One field's name set in math, where the field declares no symbol."""
    return name.replace("_", "\\_")


def link(text: str, url: str) -> str:
    """One Markdown link, for a prose line or a Markdown table.

    An empty target is a page this build does not publish, so the name is set
    as plain text rather than as a link that would resolve to nothing.
    """
    return f"[{text}]({url})" if url else text


def anchor(text: str, url: str) -> str:
    """One link inside a generated table, whose cells are HTML.

    An empty target renders as plain text, for a page this build does not
    publish.
    """
    if not url:
        return escape(text)
    return f'<a href="{escape(url, quote=True)}">{escape(text)}</a>'


def code_anchor(text: str, url: str) -> str:
    """One code-set name inside a generated table, linked where it has a page."""
    cell = code(text)
    return f'<a href="{escape(url, quote=True)}">{cell}</a>' if url else cell


def note(text: str) -> str:
    """One supporting line: approved copy, never an explanation of the page.

    Empty *text* renders nothing; an attribute list with no paragraph to
    attach to would reach the page as text.
    """
    return f"{text}\n{{ .dp-note }}\n" if text.strip() else ""


def python_block(lines: Sequence[str]) -> str:
    """One Python code block."""
    return "```python\n" + "\n".join(lines) + "\n```\n"


def table(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    numeric: Sequence[int] = (),
    row_attributes: Sequence[Mapping[str, str]] = (),
    empty: str = "—",
    ordered: Mapping[str, Sequence[str]] | None = None,
    wide: bool | None = None,
    headings: Mapping[str, str] | None = None,
) -> str:
    """One generated table, in the shape family every page shares.

    Four columns or more carry the wide modifier, so the table scrolls inside
    its own container rather than crushing its columns on a phone. Cells hold
    rendered HTML; an empty cell renders an em dash. *numeric* names the
    column positions holding counts, which are set as values.
    *row_attributes* carries one attribute map per row, for a page whose
    controls narrow rows the table already renders in full. *empty* is what an
    absent value renders as; a table whose values are generated passes an empty
    string, so an absence is an empty cell rather than a substitute glyph.
    *ordered* maps a column to the declared vocabulary its cells hold, so
    sorting that column follows that vocabulary rather than the alphabet.
    *wide* overrides the automatic four-column modifier where a caller's own
    columns are narrow enough to fit without it. *headings* shows a column's
    heading as the given HTML, such as a text slot, in place of its name.
    """
    use_wide = wide if wide is not None else len(columns) >= 4
    wide_class = " dp-table--wide" if use_wide else ""
    declared = ordered or {}
    heads = "".join(
        f'<th scope="col"'
        + (
            f' data-dp-order="{escape(",".join(declared[name]), quote=True)}"'
            if declared.get(name)
            else ""
        )
        + f">{(headings or {}).get(name, escape(name))}</th>"
        for name in columns
    )
    blank: Mapping[str, str] = {}
    marks = [*row_attributes, *[blank] * (len(rows) - len(row_attributes))]
    body = "\n".join(
        "<tr"
        + "".join(
            f' {name}="{escape(value, quote=True)}"' for name, value in mark.items()
        )
        + ">"
        + "".join(
            (
                f'<td class="dp-value">{cell or empty}</td>'
                if position in numeric
                else f"<td>{cell or empty}</td>"
            )
            for position, cell in enumerate(row)
        )
        + "</tr>"
        for row, mark in zip(rows, marks)
    )
    return (
        '<div class="dp-table-scroll table-responsive">\n'
        f'<table class="dp-table{wide_class} table table-vcenter table-hover">\n'
        f"<thead><tr>{heads}</tr></thead>\n"
        f"<tbody>\n{body}\n</tbody>\n"
        "</table>\n"
        "</div>\n"
    )


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """One two-column table, where the page carries prose rather than data."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell or "—" for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def reference_paragraphs(
    works: Sequence[CitedWork],
    bibliography: BibliographyData | None = None,
    *,
    locators: bool = True,
) -> str:
    """One hanging-indent paragraph per cited work: its entry, then its locator.

    A page whose source line already names each locator passes
    ``locators=False``, so References lists the works alone.
    """
    paragraphs = []
    for work in works:
        entry = full_citation(work.key, bibliography)
        if locators and work.locator:
            entry += (
                '<br><span class="dp-reference-locator">'
                + escape(work.locator, quote=False)
                + ".</span>"
            )
        paragraphs.append(entry + "\n{ .dp-reference }")
    return "\n\n".join(paragraphs)
