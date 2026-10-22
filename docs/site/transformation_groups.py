"""Read the maintainer-owned headings and intro lines of the Transformations groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# The groups the Transformations pages know, in page order. Which transformation sits in
# which group is the site's layout; each group's wording is the maintainer's.
TRANSFORMATION_GROUP_IDS = (
    "task_conversion",
    "bounds",
    "scaling_and_projection",
    "feature_selection",
    "other",
)
TRANSFORMATION_GROUPS_PATH = (
    Path(__file__).resolve().parents[1] / "content" / "transformation-groups.yml"
)


@dataclass(frozen=True)
class GroupWording:
    """One group's heading and the line that introduces it, if any."""

    title: str
    text: str = ""


def load_transformation_groups(
    path: Path = TRANSFORMATION_GROUPS_PATH,
) -> dict[str, GroupWording]:
    """Each group's wording in *path*, in page order.

    Raises:
        ValueError: The file is missing a group, names one the site does not
            know, or gives a group no title.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict) or set(loaded) != set(TRANSFORMATION_GROUP_IDS):
        raise ValueError(
            f"{path}: expected exactly the groups {TRANSFORMATION_GROUP_IDS}"
        )
    groups: dict[str, GroupWording] = {}
    for group in TRANSFORMATION_GROUP_IDS:
        entry = loaded[group]
        title = entry.get("title") if isinstance(entry, dict) else None
        text = entry.get("text", "") if isinstance(entry, dict) else ""
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{path}: group {group!r} needs a non-empty title")
        if not isinstance(text, str):
            raise ValueError(f"{path}: group {group!r} text must be a string")
        groups[group] = GroupWording(title.strip(), text.strip())
    return groups
