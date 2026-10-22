"""Every authored file under ``docs/`` is one the site assembler claims."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from claims import claimed_sources, navigation_targets
from dpstats.tasks import BUILT_IN_TASKS

DOCS = Path(__file__).resolve().parents[3] / "docs"


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads mkdocs' ``!ENV`` tags as their default value."""


_Loader.add_multi_constructor("!", lambda loader, suffix, node: None)


def test_no_authored_file_is_unclaimed() -> None:
    nav = yaml.load((DOCS / "site" / "mkdocs.yml").read_text(), _Loader)["nav"]
    claimed = claimed_sources(DOCS, nav, BUILT_IN_TASKS)
    # Git's own view of authored files, so ignored build output (executed
    # tutorials, caches) never counts as authored.
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "."],
        cwd=DOCS,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    authored = {
        Path(line)
        for line in listed
        if Path(line).parts[0] != "site" and (DOCS / line).is_file()
    }
    unclaimed = sorted(str(path) for path in authored - claimed)
    assert not unclaimed, f"unclaimed by the assembler: {unclaimed}"


def test_the_user_guide_lists_chapters_in_the_order_the_notebooks_assume() -> None:
    """E20: a chapter is listed after the chapters it builds on."""
    nav = yaml.load((DOCS / "site" / "mkdocs.yml").read_text(), _Loader)["nav"]
    guide = next(entry["User guide"] for entry in nav if "User guide" in entry)
    listed = [Path(target).stem for target in navigation_targets(guide)]
    order = [
        "install_and_fetch",
        "run_your_own_script",
        "compare_on_the_benchmark",
    ]
    assert [name for name in listed if name in order] == order
