"""A Home panel opens with its task's paragraph, exactly as the maintainer wrote it."""

from __future__ import annotations

from pathlib import Path

from home_context import HOME_READINGS, home_context


def test_the_paragraph_is_the_file_rendered(tmp_path: Path) -> None:
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "home.md").write_text("The task is *small*.\n")

    assert home_context("toy", tmp_path) == (
        '<div class="dp-home-reading"><p>The task is <em>small</em>.</p></div>'
    )


def test_a_task_without_a_paragraph_opens_with_none(tmp_path: Path) -> None:
    assert home_context("toy", tmp_path) == ""


def test_each_written_paragraph_is_one_paragraph() -> None:
    written = sorted(HOME_READINGS.glob("*/home.md"))
    assert written
    for path in written:
        assert "\n\n" not in path.read_text(encoding="utf-8").strip(), path
