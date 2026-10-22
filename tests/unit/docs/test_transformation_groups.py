"""Tests for the Transformations group wording loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_groups import (
    TRANSFORMATION_GROUP_IDS,
    TRANSFORMATION_GROUPS_PATH,
    load_transformation_groups,
)


def test_the_default_path_is_the_maintainer_file() -> None:
    assert TRANSFORMATION_GROUPS_PATH.parts[-3:] == (
        "docs",
        "content",
        "transformation-groups.yml",
    )


def test_the_maintainer_file_names_every_group_in_page_order() -> None:
    groups = load_transformation_groups()
    assert tuple(groups) == TRANSFORMATION_GROUP_IDS
    assert all(group.title for group in groups.values())


@pytest.mark.parametrize(
    "body",
    [
        "bounds:\n  title: Bounds\n",
        "\n".join(f"{g}:\n  title: T" for g in (*TRANSFORMATION_GROUP_IDS, "extra")),
        "\n".join(f"{g}:\n  text: Only text." for g in TRANSFORMATION_GROUP_IDS),
    ],
    ids=["missing group", "unknown group", "no title"],
)
def test_a_file_that_drifts_from_the_site_s_groups_fails(
    tmp_path: Path, body: str
) -> None:
    path = tmp_path / "groups.yml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_transformation_groups(path)
