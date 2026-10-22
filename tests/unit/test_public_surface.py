"""The repository's tracked text names nothing private.

This repository is published, so a tracked file that names a private
checkout, a maintainer's machine path, an internal host or an internal
tracker reaches every reader. The scan reads every tracked text file and
refuses each token below; a file that
names one on purpose (a portability guard that refuses it, say) is listed in
``ALLOWED`` with its reason.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Spellings that only a private checkout, machine or tracker would carry.
TOKENS: tuple[str, ...] = (
    "/Users/",
    "/home/",
    "dp-platform-project",
    "prokofiev",
    "dp-testing-platform#",
    "github.io/dp-testing-platform",
    "github.com/saminulh/dp-testing-platform",
    "STATUS.md",
    "_archive/",
)


# Agent instruction and runtime files: the published tree carries none.
AGENT_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    ".claude",
    ".agents",
    ".cursor",
    ".codex",
    ".github/copilot-instructions.md",
)

# path -> {token: reason}. A hit here is read and justified once.
ALLOWED: dict[str, dict[str, str]] = {
    "tests/unit/test_public_surface.py": {
        token: "this check's own token list" for token in TOKENS
    },
    "tests/integration/test_tutorials.py": {
        "/Users/": "the portability guard refuses machine paths in executed notebooks",
        "/home/": "the portability guard refuses machine paths in executed notebooks",
    },
    "tests/unit/docs/test_corpus.py": {
        "/Users/": "an assertion that no built page names a machine path",
    },
}


def _tracked_text_files() -> list[str]:
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [relative for relative in listed.split("\0") if relative]


def _findings() -> list[str]:
    found = []
    for relative in _tracked_text_files():
        path = REPO_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        lowered = text.lower()
        for token in TOKENS:
            if token.lower() in lowered and token not in ALLOWED.get(relative, {}):
                line = next(
                    number
                    for number, content in enumerate(text.splitlines(), start=1)
                    if token.lower() in content.lower()
                )
                found.append(f"{relative}:{line}: names {token!r}")
    return found


def test_no_tracked_file_names_anything_private() -> None:
    findings = _findings()
    assert not findings, "\n".join(findings)


def test_every_allowance_is_still_needed() -> None:
    stale = [
        f"{relative}: {token!r}"
        for relative, tokens in ALLOWED.items()
        for token in tokens
        if relative != "tests/unit/test_public_surface.py"
        and token.lower() not in (REPO_ROOT / relative).read_text().lower()
    ]
    assert not stale, "\n".join(stale)


def test_the_tree_carries_no_agent_files() -> None:
    tracked = _tracked_text_files()
    present = [
        name
        for name in AGENT_FILES
        if any(path == name or path.startswith(name + "/") for path in tracked)
    ]
    assert not present, present
