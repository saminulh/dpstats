"""Tests for the committed release pin the documentation build reads."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmark_lock import read_benchmark_lock

REPO_ROOT = Path(__file__).resolve().parents[3]

REVISION = "0123456789abcdef0123456789abcdef01234567"


def write_lock(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_an_absent_lock_stops_the_build_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "benchmark.lock"

    with pytest.raises(ValueError, match=re.escape(str(missing))):
        read_benchmark_lock(missing)


def test_the_committed_lock_parses() -> None:
    """Whatever it pins, the lock a build reads is one the reader accepts."""
    read_benchmark_lock(REPO_ROOT / "docs" / "benchmark.lock")


def test_an_explicit_no_release_declaration_pins_nothing(tmp_path: Path) -> None:
    lock = write_lock(
        tmp_path / "benchmark.lock",
        "# The site renders no results until a release carries the view.\nrelease = none\n",
    )

    assert read_benchmark_lock(lock) is None


def test_a_named_release_carries_its_repository_and_revision(
    tmp_path: Path,
) -> None:
    lock = write_lock(
        tmp_path / "benchmark.lock",
        f"repo_id = org/dataset\nrevision = {REVISION}\n",
    )

    pin = read_benchmark_lock(lock)

    assert pin is not None
    assert pin.repo_id == "org/dataset"
    assert pin.revision == REVISION


def test_blank_lines_and_comments_are_not_settings(tmp_path: Path) -> None:
    lock = write_lock(
        tmp_path / "benchmark.lock",
        f"\n# repo_id = commented/out\n\nrepo_id = org/dataset\n"
        f"revision = {REVISION}\n\n",
    )

    pin = read_benchmark_lock(lock)

    assert pin is not None
    assert pin.repo_id == "org/dataset"


@pytest.mark.parametrize(
    ("text", "violated"),
    [
        # The release declares the tasks it released; the lock never does.
        (f"repo_id = org/dataset\nrevision = {REVISION}\ntasks = x\n", "tasks"),
        ("release none\n", "release none"),
        (f"repo_id = org/one\nrepo_id = org/two\nrevision = {REVISION}\n", "repo_id"),
        (f"release = none\nrepo_id = org/dataset\nrevision = {REVISION}\n", "release"),
        ("release = later\n", "later"),
        (f"repo_id = org/dataset\nrevision = {REVISION}\nbranch = main\n", "branch"),
        (f"revision = {REVISION}\n", "repo_id"),
        ("repo_id = org/dataset\n", "revision"),
    ],
    ids=[
        "names tasks",
        "not a setting",
        "repeated setting",
        "release beside a pin",
        "release other than none",
        "unknown setting",
        "no repository",
        "no revision",
    ],
)
def test_an_invalid_lock_is_refused_naming_what_is_wrong(
    tmp_path: Path, text: str, violated: str
) -> None:
    lock = write_lock(tmp_path / "benchmark.lock", text)

    with pytest.raises(ValueError, match=violated):
        read_benchmark_lock(lock)


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", REVISION.upper(), REVISION[:39], REVISION + "0"],
)
def test_a_revision_that_is_not_an_immutable_commit_is_refused(
    tmp_path: Path, revision: str
) -> None:
    lock = write_lock(
        tmp_path / "benchmark.lock",
        f"repo_id = org/dataset\nrevision = {revision}\n",
    )

    with pytest.raises(ValueError, match="lowercase commit"):
        read_benchmark_lock(lock)
