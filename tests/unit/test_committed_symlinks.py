"""A committed symlink points inside the repository.

One pointing at an absolute or outside path names a single machine's layout,
and a release bundle built from the tree refuses it.
"""

import posixpath
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_every_committed_symlink_stays_inside_the_repository() -> None:
    found = []
    for line in _git("ls-files", "-s").splitlines():
        mode, blob, _, name = line.split(maxsplit=3)
        if mode != "120000":
            continue
        target = _git("cat-file", "blob", blob)
        landing = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        if posixpath.isabs(target) or landing == ".." or landing.startswith("../"):
            found.append(f"{name} -> {target}")
    assert found == [], "\n".join(found)
