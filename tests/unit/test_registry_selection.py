"""Selecting a Task's installed registry documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpstats.transport.documents import STAGING_PREFIX
from dpstats.workspace.registries import select_registry_paths


def _task_root(tmp_path: Path) -> Path:
    directory = tmp_path / "task" / "instance_registries"
    directory.mkdir(parents=True)
    (directory / "accepted.json").write_text("{}")
    (directory / f"{STAGING_PREFIX}killed").write_text("{")
    return tmp_path / "task"


def test_staging_residue_refuses_unless_ignored(tmp_path: Path) -> None:
    task_root = _task_root(tmp_path)

    with pytest.raises(ValueError, match="contains"):
        select_registry_paths(task_root, "instance", None)
    assert [
        path.name
        for path in select_registry_paths(
            task_root, "instance", None, ignore_staging=True
        )
    ] == ["accepted.json"]


def test_ignoring_staging_still_refuses_other_strays(tmp_path: Path) -> None:
    task_root = _task_root(tmp_path)
    (task_root / "instance_registries" / "notes.txt").write_text("x")

    with pytest.raises(ValueError, match="notes.txt"):
        select_registry_paths(task_root, "instance", None, ignore_staging=True)
