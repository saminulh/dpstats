"""The reserved root-level names a task id may never collide with."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    INPUT_ARTIFACTS_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    ReleaseIdentity,
    is_valid_task_id,
    read_release_identity,
    workspace_roots,
)


@pytest.mark.parametrize(
    "name",
    [
        INPUT_ARTIFACTS_DIRECTORY,
        BENCHMARK_INFO_DIRECTORY,
        RELEASE_MANIFEST_NAME,
    ],
)
def test_a_reserved_root_level_name_is_not_a_valid_task_id(name: str) -> None:
    assert is_valid_task_id(name) is False


@pytest.mark.parametrize(
    "name", ["mean_estimation", "linear_regression", "a3-VALID-TASK"]
)
def test_an_ordinary_portable_name_is_a_valid_task_id(name: str) -> None:
    assert is_valid_task_id(name) is True


@pytest.mark.parametrize("name", ["", "../escape", "a/b", None, 1])
def test_a_non_portable_value_is_not_a_valid_task_id(name: object) -> None:
    assert is_valid_task_id(name) is False


def test_read_release_identity_is_none_without_a_manifest(tmp_path: Path) -> None:
    assert read_release_identity(tmp_path) is None


def test_read_release_identity_reads_the_manifest(tmp_path: Path) -> None:
    (tmp_path / RELEASE_MANIFEST_NAME).write_text(
        json.dumps({"version": "v1", "manifest_id": "abc", "tasks": ["t1", "t2"]})
    )
    assert read_release_identity(tmp_path) == ReleaseIdentity(
        version="v1", manifest_id="abc", tasks=("t1", "t2"), path=tmp_path
    )


def test_workspace_roots_of_a_workspace_with_and_without_a_release(
    tmp_path: Path,
) -> None:
    (tmp_path / "local").mkdir()
    assert workspace_roots(tmp_path) == (tmp_path / "local", None)
    (tmp_path / "official-release").mkdir()
    assert workspace_roots(tmp_path) == (
        tmp_path / "local",
        tmp_path / "official-release",
    )


def test_workspace_roots_of_a_bare_bundle_is_release_only(tmp_path: Path) -> None:
    (tmp_path / RELEASE_MANIFEST_NAME).write_text("{}")
    assert workspace_roots(tmp_path) == (None, tmp_path)


def test_workspace_roots_of_anything_else_is_neither(tmp_path: Path) -> None:
    assert workspace_roots(tmp_path) == (None, None)
