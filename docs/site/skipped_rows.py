"""Report what a local-cut build leaves out: unloadable entries and incomplete algorithms."""

from __future__ import annotations

import warnings
from html import escape
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from dpstats.algorithm import load_algorithms_partitioned
from dpstats.instance import load_instances_partitioned
from dpstats.workspace.layout import (
    registry_directory_name,
    task_collection_directory,
)
from dpstats.workspace.registries import read_selection

# Maintainer-approved wording.
ALGORITHM_BANNER = (
    "This preview leaves out {count} configured algorithm(s) that this library "
    "version cannot load."
)
INCOMPLETE_BANNER = (
    "This preview leaves out {count} configured algorithm(s) missing some of "
    "their trials in this cut, rather than rank them on part of them."
)
INSTANCE_BANNER = (
    "This preview leaves out {count} problem instance(s) that this library version "
    "cannot load."
)

_STANDINGS_SENTENCE = "They are also left out of this preview's standings."

_LOADERS = {
    "algorithm": load_algorithms_partitioned,
    "instance": load_instances_partitioned,
}


def skipped_entries(
    root: Path, task_id: str, *, published: bool
) -> list[dict[str, str]]:
    """Every entry of a local cut this library cannot rebuild, or none if published.

    Items carry ``kind``, ``uuid``, the raw record's ``label`` and the ``reason``.
    Preparation adds ``incomplete`` rows for algorithms lacking a declared trial.
    """
    if published:
        return []
    task_root = task_collection_directory(root, task_id)
    skipped: list[dict[str, str]] = []
    for kind, load in _LOADERS.items():
        if not (task_root / registry_directory_name(kind)).is_dir():
            continue
        entries = read_selection(task_root, kind, None).entries
        _, rejections = load(entries, task_id=task_id)
        skipped += [
            {
                "kind": kind,
                "uuid": uuid,
                "label": _label(entries[uuid]),
                "reason": reason,
            }
            for uuid, reason in rejections.items()
        ]
    return skipped


def skipped_reasons(root: Path, task_id: str, kind: str) -> dict[str, str]:
    """Each unloadable entry of one *kind* in a local cut, keyed by uuid."""
    return {
        row["uuid"]: row["reason"]
        for row in skipped_entries(root, task_id, published=False)
        if row["kind"] == kind
    }


def _label(record: Mapping[str, Any]) -> str:
    config: Any = record.get("config")
    label: Any = record.get("label")
    if not label and isinstance(config, Mapping):
        label = cast("Mapping[str, Any]", config).get("label")
    return label if isinstance(label, str) else ""


@contextmanager
def reported_separately() -> Generator[None]:
    """Silence the library's skip warning where the build reports skips itself."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Skipped \d+ of \d+ \w+ registry entries",
            category=UserWarning,
        )
        yield


def banner(skipped: Sequence[Mapping[str, str]], removed: int = 0) -> str:
    """The banner a local-cut preview shows for its skipped entries, or ``""``.

    ``removed`` is how many of them the standings also left out.
    """
    parts = []
    for kind, template in (
        ("algorithm", ALGORITHM_BANNER),
        ("incomplete", INCOMPLETE_BANNER),
        ("instance", INSTANCE_BANNER),
        ("pending_evaluation", "Evaluation pending: {count}"),
        ("pending_distribution", "Distribution pending: {count}"),
    ):
        rows = [row for row in skipped if row["kind"] == kind]
        if rows:
            text = template.format(count=len(rows))
            if kind == "algorithm" and removed:
                text += f" {_STANDINGS_SENTENCE}"
            parts.append(text)
    return " ".join(parts)


def banner_html(
    skipped: Sequence[Mapping[str, str]], removed: int = 0, *, home: bool = False
) -> str:
    """The banner as the note a Leaderboard or Explore page opens with.

    The Home preview carries none.
    """
    text = "" if home else banner(skipped, removed)
    if not text:
        return ""
    # The names sit in a closed disclosure so a long list never buries the page.
    names = "".join(
        f"<li>{escape(row['label'] or row['uuid'])}"
        + (
            f" ({escape(row['reason'])})"
            if row["kind"]
            in {"incomplete", "pending_evaluation", "pending_distribution"}
            else ""
        )
        + "</li>"
        for row in skipped
    )
    return (
        f'<div class="dp-note"><p>{escape(text)}</p>'
        f"<details><summary>Show which</summary><ul>{names}</ul></details></div>"
    )


def summary_line(
    task_id: str, skipped: Sequence[Mapping[str, str]], removed: int = 0
) -> str:
    """One build-output line: the task, its skipped rows and standings removals."""
    noun = "row" if len(skipped) == 1 else "rows"
    line = f"{task_id}: {len(skipped)} {noun} skipped, {removed} removed from standings"
    if skipped:
        line += ": " + "; ".join(
            f"{row['uuid']} ({(row['reason'].splitlines() or [''])[0]})"
            for row in skipped
        )
    return line
