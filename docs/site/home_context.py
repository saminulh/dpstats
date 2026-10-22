"""The paragraph a Home task panel opens with: the maintainer's own words."""

from __future__ import annotations

from pathlib import Path

from markdown import markdown

# One maintainer-approved paragraph per Benchmarked task, saying what the task
# is and what its headline reading counts as success, for a newcomer.
HOME_READINGS = Path(__file__).resolve().parents[1] / "content" / "tasks"


def home_context(slug: str, readings: Path = HOME_READINGS) -> str:
    """The task's Home paragraph as HTML, or ``""`` where none is written."""
    path = readings / slug / "home.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return f'<div class="dp-home-reading">{markdown(text)}</div>'
