"""Execute the fenced ``python`` blocks in ``docs/*.md`` and ``docs/tasks/*.md``.

A documented example that does not run is a broken promise, and nothing else in
the suite reads the prose. Every fence in one page executes in that page's own
namespace, in order, from a temporary working directory, so a block may continue
the block above it exactly as a reader would.

One explicit escape: ``<!-- no-run: reason -->`` immediately above a fence means
the block is not executable on its own — it needs a collected task collection, it
is an illustrative fragment, or it is module source for a package the page
generates. Every such marker must be consumed by a fence.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTATION_ROOT = REPOSITORY_ROOT / "docs"
NO_RUN = re.compile(r"<!--\s*no-run:(?P<reason>[^>]*?)-->")


def _documents() -> list[Path]:
    candidates = sorted(DOCUMENTATION_ROOT.glob("*.md")) + sorted(
        (DOCUMENTATION_ROOT / "tasks").glob("*.md")
    )
    return [path for path in candidates if "```python" in path.read_text("utf-8")]


DOCUMENTS = _documents()
SYBIL = Sybil(parsers=[PythonCodeBlockParser()], patterns=["*.md"])


def _no_run_reason(text: str) -> str | None:
    """Return the reason of the last no-run marker in one slice of a document."""
    markers = NO_RUN.findall(text)
    if not markers:
        return None
    reason = markers[-1].strip()
    assert reason, "a no-run marker must state why the block cannot execute"
    return reason


@pytest.fixture(scope="module")
def fence_totals() -> Iterator[dict[str, int]]:
    """Accumulate the executed and skipped counts and report."""
    totals = {"executed": 0, "no_run": 0}
    yield totals
    print(
        f"\ndocumentation fences: {totals['executed']} executed, "
        f"{totals['no_run']} marked no-run"
    )


def test_documents_are_discovered() -> None:
    """Guard the glob: an empty list would silently skip every page below."""
    assert DOCUMENTS, f"no fenced python found under {DOCUMENTATION_ROOT}"


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: str(path.name))
def test_documentation_examples_execute(
    path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence_totals: dict[str, int],
) -> None:
    """Run every runnable fence in one page, in that page's own namespace."""
    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    document = SYBIL.parse(path)
    # A class statement needs a module name; a parsed document has no module.
    document.namespace["__name__"] = "documentation_example"
    monkeypatch.chdir(tmp_path)

    consumed = 0
    previous_end = 0
    for example in document.examples():
        reason = _no_run_reason(document.text[previous_end : example.region.start])
        previous_end = example.region.end
        if reason is not None:
            consumed += 1
            fence_totals["no_run"] += 1
            continue
        fence_totals["executed"] += 1
        example.evaluate()

    assert consumed == len(
        NO_RUN.findall(document.text)
    ), f"{relative} has a no-run marker that no fence consumed"
