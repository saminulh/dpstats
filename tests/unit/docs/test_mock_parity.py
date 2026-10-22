"""The golden pages are the tuned mock, reproduced by the renderers."""

from __future__ import annotations

from pathlib import Path

from templates.filler import mock_pages

GOLDEN = Path(__file__).resolve().parents[3] / "docs" / "site" / "templates" / "golden"


def test_renderers_reproduce_the_golden_pages() -> None:
    """The checked-in pages are exactly what the filler registry renders."""
    rendered = mock_pages()
    assert {path.stem for path in GOLDEN.glob("*.md")} == set(rendered)
    for name, text in rendered.items():
        assert (GOLDEN / f"{name}.md").read_text(encoding="utf-8") == text, name
