"""The API reference renders one row per parameter and per return, not per line.

Google-style Args and Returns entries wrap onto several indented physical
lines. Griffe joins Args continuation lines by default, but its Returns
section treats every physical line as a new unnamed return value unless
``returns_multiple_items`` is disabled (Google returns are never documented
that way in this library; see the "no multi-name Returns section" scan this
test's issue was filed from). Without that option, mkdocstrings renders one
truncated table row per physical line of a wrapped Returns paragraph.
"""

from __future__ import annotations

from pathlib import Path

import griffe

MKDOCS_CONFIG = Path(__file__).resolve().parents[3] / "docs" / "site" / "mkdocs.yml"

WRAPPED_DOCSTRING = """Summarize a wrapped Args entry and a wrapped Returns entry.

Args:
    frame: Measurement rows carrying the comparison-cell columns and their
        named Metric column.

Returns:
    One row per comparison cell: its key columns and the summarized
    Metric column, without the seed or the budget document.
"""


def _configured_docstring_options() -> dict[str, object]:
    """The google ``docstring_options`` mkdocs.yml declares for mkdocstrings."""
    lines = MKDOCS_CONFIG.read_text().splitlines()
    heading = next(line for line in lines if line.strip() == "docstring_options:")
    heading_indent = len(heading) - len(heading.lstrip(" "))
    options: dict[str, object] = {}
    for line in lines[lines.index(heading) + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= heading_indent:
            break
        key, _, value = stripped.partition(":")
        value = value.strip()
        if value in ("true", "false"):
            options[key] = value == "true"
    return options


def _parse(**overrides: object) -> list[griffe.DocstringSection]:
    options = {**_configured_docstring_options(), **overrides}
    docstring = griffe.Docstring(WRAPPED_DOCSTRING)
    return griffe.parse_google(docstring, **options)


def test_configured_options_join_a_wrapped_args_entry_into_one_row():
    sections = _parse()
    (parameters,) = [
        section
        for section in sections
        if section.kind is griffe.DocstringSectionKind.parameters
    ]
    (frame,) = parameters.value
    assert frame.description == (
        "Measurement rows carrying the comparison-cell columns and their\n"
        "named Metric column."
    )


def test_configured_options_join_a_wrapped_returns_entry_into_one_row():
    sections = _parse()
    (returns,) = [
        section
        for section in sections
        if section.kind is griffe.DocstringSectionKind.returns
    ]
    (only_return,) = returns.value
    assert only_return.description == (
        "One row per comparison cell: its key columns and the summarized\n"
        "Metric column, without the seed or the budget document."
    )


def test_mkdocs_config_disables_returns_multiple_items():
    """Named to point straight at the fix: the gate a regression would trip."""
    assert _configured_docstring_options().get("returns_multiple_items") is False
