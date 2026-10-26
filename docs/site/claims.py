"""Which authored files under ``docs/`` the site assembler publishes.

The assembler publishes an authored file only when navigation, a task or a
reference group claims it; the tutorials live outside ``docs/``. This module
owns those claim rules so the assembler and the inverse gate read one
definition.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

# The runnable script the API reference index shows in full.
INTERACTIVE_WORKFLOW = Path("reference") / "interactive_workflow.py"

# Where an authored page selected by navigation leaves the Library area's root.
RELOCATED_PAGES = {
    "submission.md": "submit/submission.md",
}


def navigation_targets(entries: list) -> list[str]:
    """Every string target in a nested mkdocs ``nav``."""
    targets: list[str] = []
    for entry in entries:
        for target in entry.values():
            if isinstance(target, list):
                targets.extend(navigation_targets(target))
            elif isinstance(target, str):
                targets.append(target)
    return targets


def nav_source(target: str) -> str:
    """The ``docs/``-relative source a navigation target publishes."""
    relocated = {published: source for source, published in RELOCATED_PAGES.items()}
    return relocated.get(target, target.removeprefix("library/"))


def shared_reference_groups(nav: list, docs: Path) -> dict[str, Path]:
    """Reference groups the nav names that have an authored intro page."""
    groups = {}
    for entry in nav:
        for target in entry.values():
            if isinstance(target, list):
                groups.update(shared_reference_groups(target, docs))
            elif isinstance(target, str) and target.startswith("library/reference/"):
                group = (
                    target.removeprefix("library/reference/")
                    .rstrip("/")
                    .removesuffix(".md")
                )
                source = docs / "reference" / f"{group}.md"
                if source.is_file():
                    groups[group] = source
    return groups


def documented_task_ids(docs: Path, tasks: Iterable[str]) -> set[str]:
    """Tasks with an authored concept page or problem formulation."""
    return {
        task
        for task in tasks
        if (docs / "tasks" / f"{task}.md").exists()
        or (docs / "content" / "tasks" / task / "problem-formulation.md").exists()
    }


def claimed_sources(docs: Path, nav: list[Any], task_ids: Iterable[str]) -> set[Path]:
    """Every file outside ``docs/site/`` the assembler reads, relative to *docs*."""
    claimed = {
        Path("benchmark.lock"),
        Path("content") / "reader-terms.yml",
        Path("content") / "transformation-groups.yml",
        Path("content") / "site-copy.yml",
        Path("content") / "home.md",
        Path("content") / "corpus-about.md",
        Path("content") / "methodology.md",
        INTERACTIVE_WORKFLOW,
    }
    for target in navigation_targets(nav):
        claimed.add(Path(nav_source(target)))
    for source in shared_reference_groups(nav, docs).values():
        claimed.add(source.relative_to(docs))
    for task in documented_task_ids(docs, task_ids):
        claimed.add(Path("tasks") / f"{task}.md")
        claimed.add(Path("content") / "tasks" / task / "problem-formulation.md")
    for task in task_ids:
        home = Path("content") / "tasks" / task / "home.md"
        if (docs / home).is_file():
            claimed.add(home)
    return claimed
