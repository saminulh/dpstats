"""Validate autodoc coverage of the API-reference pages.

Docstrings are the single source, rendered via mkdocstrings ``:::``
directives, so this script does not match prose. It checks **coverage** of
what ``docs/reference/*.md`` owns, under the one-place-per-class contract of
the site: a class renders in exactly one place, inside its task space or on an
API-reference page, never both.

A task's own package (its Metadata, Input and Output models) and its
``algorithms`` and ``transformations`` packages render inside the task space
that ``docs/site/assemble.py`` builds: the definition page and one component
page per Algorithm or Transformation, each carrying its own API block. They are
therefore excluded here, as is Metrics, whose task section is prose pointing at
``built_in_metrics()``. What a task owes ``docs/reference`` is its ``losses``
module when present: the loss properties every task's losses implement are
task-generic machinery on the Tasks page. A per-file loss submodule under
``transformations`` is reachable through the directive that names it and is not
separately required.

The reverse direction is checked too: a ``:::`` directive naming a module that
does not exist is a dangling entry left by a deletion, caught here rather than
only at the strict site build.

Usage:
    python tools/validate_docs.py            # check all tasks
    python tools/validate_docs.py --task m_estimation
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PACKAGE = REPO_ROOT / "dpstats"
REFERENCE_DIR = REPO_ROOT / "docs" / "reference"

_DIRECTIVE_RE = re.compile(r"^\s*:::\s+(\S+)")


def collect_directive_targets() -> set[str]:
    """Return every dotted module path targeted by a ``:::`` directive."""
    targets: set[str] = set()
    for md in REFERENCE_DIR.glob("*.md"):
        for line in md.read_text().splitlines():
            match = _DIRECTIVE_RE.match(line)
            if match:
                targets.add(match.group(1))
    return targets


def required_modules_for_task(task: str) -> list[str]:
    """Dotted module paths one task owes ``docs/reference``.

    The task package and its ``algorithms`` and ``transformations`` packages
    render in the task space, so only the ``losses`` module is required.
    """
    task_dir = PACKAGE / "tasks" / task
    if (task_dir / "losses.py").is_file():
        return [f"dpstats.tasks.{task}.losses"]
    return []


def discover_tasks() -> list[str]:
    """Task package names under ``dpstats/tasks``."""
    tasks_dir = PACKAGE / "tasks"
    return sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(("_", "."))
        and (path / "__init__.py").is_file()
    )


def dangling_targets(targets: set[str]) -> list[str]:
    """Directive targets under the package that name no existing module."""
    return [
        target
        for target in sorted(targets)
        if target.startswith("dpstats") and not _module_exists(target)
    ]


def _module_exists(target: str) -> bool:
    path = REPO_ROOT.joinpath(*target.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def main(args: list[str] | None = None) -> int:
    """Report missing and dangling coverage; exit 1 on either."""
    parser = argparse.ArgumentParser(
        description="Check autodoc coverage of docs/reference/*.md pages"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Check coverage for a single task only (e.g. 'm_estimation')",
    )
    parsed = parser.parse_args(args)

    targets = collect_directive_targets()
    tasks = [parsed.task] if parsed.task else discover_tasks()

    missing: list[str] = []
    checked = 0
    for task in tasks:
        for module in required_modules_for_task(task):
            checked += 1
            if module not in targets:
                missing.append(module)

    dangling = dangling_targets(targets)

    print(f"Checked {checked} public module(s) across {len(tasks)} task(s).")
    if dangling:
        print(f"\nDANGLING reference entries ({len(dangling)}):")
        for target in dangling:
            print(f"  - ':::' directive for deleted module {target}")
        print("\nValidation FAILED")
        return 1
    if missing:
        print(f"\nMISSING reference coverage ({len(missing)}):")
        for module in missing:
            print(f"  - no ':::' directive for {module}")
        print("\nValidation FAILED")
        return 1

    print("Every public module has a reference entry. Validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
