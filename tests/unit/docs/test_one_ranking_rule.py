"""The site ranks algorithms by one rule, ``panel_data.headline_order``.

Every page that orders algorithms by how they stand (the Leaderboard, Home,
Explore's table, the Roster pages and their sort script) reads the one
``Headline`` reading and compares it through ``headline_key``. Each rule that
ranks by its own reading of a curve (a share within some threshold, sorted
descending) is a second ranking rule, and fails here by file and line.
"""

from __future__ import annotations

import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[3] / "docs" / "site"
SCRIPTS = SITE / "docs" / "javascripts"
# The one home of the rule.
RULE = SITE / "panel_data.py"

# A sort, and what it is keyed on, as Python writes it on one or a few lines.
SORT = re.compile(r"\b(sorted|\.sort)\(")
# A standing read as a share and ordered by it: the shape every second rule took.
SHARE_ORDER = re.compile(r"-\s*[\w.\[\]\"']*\b(fraction|share)s?\b")
# A script comparator ordering by a share, descending: right's (or b's)
# fraction or share minus left's (or a's), or a running best share.
SCRIPT_SHARE_ORDER = re.compile(
    r"\b(right|b)\.[\w.\[\]]*(fraction|share)\w*[^;\n]*-\s*(left|a)\."
    r"|\bbestShare\b|\bshareAt\("
)
# The readings the rule replaced, which nothing may read again.
RETIRED = re.compile(r"\bheadline_tau\b|\.opening\b|\[\"opening\"\]")


def _python() -> list[Path]:
    return sorted(
        path
        for path in SITE.rglob("*.py")
        if ".prepared" not in path.parts and "filler-results" not in path.parts
    )


def _window(lines: list[str], index: int) -> str:
    """One sort call, from its line to the bracket closing it."""
    start = SORT.search(lines[index])
    assert start is not None
    text, depth = "", 0
    for line in lines[index:]:
        for char in line[start.start() if not text else 0 :]:
            text += char
            depth += char == "("
            depth -= char == ")"
            if depth == 0 and char == ")":
                return text
        text += "\n"
    return text


def test_no_python_module_orders_algorithms_by_a_share() -> None:
    found = []
    for path in _python():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not SORT.search(line):
                continue
            window = _window(lines, index)
            if SHARE_ORDER.search(window):
                found.append(f"{path.relative_to(SITE)}:{index + 1}")
    assert found == [], "a second ranking rule orders by a share"


def test_every_sort_on_a_headline_reading_goes_through_the_rule() -> None:
    found = []
    for path in _python():
        if path == RULE:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not SORT.search(line):
                continue
            window = _window(lines, index)
            if "headline" in window and not re.search(
                r"\bheadline_(key|order)\b", window
            ):
                found.append(f"{path.relative_to(SITE)}:{index + 1}")
    assert found == [], "a sort on the headline reading bypasses headline_key"


def test_nothing_reads_the_retired_threshold_standing() -> None:
    found = [
        f"{path.relative_to(SITE)}:{number}"
        for path in [*_python(), *sorted(SCRIPTS.glob("*.js"))]
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if RETIRED.search(line)
    ]
    assert found == [], "a page reads the standing at a threshold"


def test_no_script_orders_algorithms_by_a_share() -> None:
    """Every page script ranks through panel-data.js's compareReadings."""
    found = [
        f"{path.relative_to(SITE)}:{number}"
        for path in sorted(SCRIPTS.glob("*.js"))
        if "vendor" not in path.parts
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if SCRIPT_SHARE_ORDER.search(line)
    ]
    assert found == [], "a script orders algorithms by a share"
