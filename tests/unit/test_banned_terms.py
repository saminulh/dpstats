"""Retired vocabulary stays out of the repository's own text and code."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Each retired term and what to say instead.
BANNED = {
    # A privacy budget is a budget everywhere.
    re.compile(r"\ballowances?\b", re.IGNORECASE): "say 'budget'",
    # A default states where it came from or why it was chosen.
    re.compile(
        r"\b(?:platform|house)\s+(?:choices?|defaults?)\b", re.IGNORECASE
    ): "state the default's source or the reason it was chosen",
}

# The ban lists name the words they ban.
BAN_LISTS = ("tests/unit/test_banned_terms.py", "docs/site/templates/conformance.py")


def test_no_tracked_file_uses_a_banned_term() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    hits = []
    for name in tracked:
        if name in BAN_LISTS:
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        # Whole-text search, so a term wrapped across two lines still counts.
        hits += [
            f"{name}:{text.count(chr(10), 0, match.start()) + 1}: {advice}"
            for pattern, advice in BANNED.items()
            for match in pattern.finditer(text)
        ]
    assert not hits, f"retired terms: {hits}"
