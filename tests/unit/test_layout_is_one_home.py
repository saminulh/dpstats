"""``workspace/layout.py`` is the sole home of the on-disk shape's names.

A directory or file name that names part of the workspace/bundle tree shape
must be spelled exactly once, as a constant in ``dpstats.workspace
.layout``, so a future rename touches one file. This walks the source tree
outside that module and fails on a literal occurrence of one of those names
as a quoted string literal (an exact match) or as a path segment inside a
longer string (bounded by ``/`` or the string's own start/end), skipping
comments and docstrings. A name that legitimately needs to appear elsewhere —
a CLI help string quoting the directory it creates, for example — is named in
``ALLOWED`` with a one-line reason instead of silently passing.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

LAYOUT_MODULE = REPO_ROOT / "dpstats" / "workspace" / "layout.py"

# The exact on-disk names workspace/layout.py owns. Kept as a literal tuple
# here (rather than imported) so this guard does not depend on the module it
# is checking, and so a name is remembered even if a future edit drops its
# constant from layout.py without updating this list.
LAYOUT_NAMES: tuple[str, ...] = (
    "local",
    "official-release",
    "benchmark-info",
    "input-artifacts",
    "instance_registries",
    "algorithm_registries",
    "attempts",
    "results",
    "claims",
    "sweeps",
    "instance-weights.csv",
    "withheld-profiles.parquet",
    "release.json",
    "algorithm-declarations.csv",
    ".npz",
)

SCAN_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "dpstats",
    REPO_ROOT / "docs" / "site",
    REPO_ROOT / "docs" / "site" / "templates",
)

# path -> {name: reason}. A hit here is a confirmed non-path-shape use of the
# word, read and justified once, not a blanket exemption for the file.
ALLOWED: dict[str, dict[str, str]] = {
    "dpstats/cli.py": {
        "official-release": "COMMAND_DESCRIPTIONS help string for `fetch`, quoting the destination for the reader",
        "local": "COMMAND_DESCRIPTIONS help string for `init`, quoting what it creates for the reader",
    },
    "dpstats/transport/arrays.py": {
        ".npz": "the array-artifact writer's own file suffix; it names the npz format, not a workspace path",
    },
    "dpstats/scripts/fetch.py": {
        "official-release": "argparse description text quoting the destination fetch ensures a release at",
    },
    "dpstats/scripts/init.py": {
        "local": "argparse description text quoting the directory init creates, for the reader",
    },
    "dpstats/scripts/get_sweep_status.py": {
        "local": "--results-source choices, the same results_source= filter vocabulary as collection.py, not a directory literal",
    },
    "dpstats/scripts/measure_sweep.py": {
        "local": "--results-source choices, the same results_source= filter vocabulary as collection.py, not a directory literal",
    },
    "docs/site/instance_pages.py": {
        "claims": "the source_facts key for the paper claims an instance page quotes, a study-document key, not the Attempt claims directory",
    },
    "docs/site/release_files.py": {
        "results": "the release manifest's availability field, not the Results directory",
    },
    "docs/site/results_roots.py": {
        "results": "the release manifest's availability field, not the Results directory",
    },
    "dpstats/workspace/collection.py": {
        # The `source` column/filter vocabulary ("local"/"release") tags a
        # loaded row or a read scope; it is coincidentally the same spelling
        # as LOCAL_DIRECTORY, but a rename of that directory would not
        # rename this vocabulary, and vice versa.
        "local": "the source= filter/column value 'local' (Literal type, comparisons, and DataFrame tags), not a directory literal",
    },
}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        pattern = "*.py" if root.name in {"site", "templates"} else "**/*.py"
        files.extend(sorted(root.glob(pattern)))
    # De-duplicate (docs/site and docs/site/templates overlap via **/*.py from
    # neither being recursive above) and drop layout.py itself.
    seen: dict[Path, None] = {}
    for path in files:
        if path.resolve() == LAYOUT_MODULE.resolve():
            continue
        if "__pycache__" in path.parts:
            continue
        seen[path] = None
    return list(seen)


def _is_docstring(tokens: list[tokenize.TokenInfo], index: int) -> bool:
    """Whether the STRING token at *index* stands alone as a docstring."""
    token = tokens[index]
    if token.type != tokenize.STRING:
        return False
    previous = next(
        (
            tokens[i]
            for i in range(index - 1, -1, -1)
            if tokens[i].type not in (tokenize.NL,)
        ),
        None,
    )
    following = next(
        (
            tokens[i]
            for i in range(index + 1, len(tokens))
            if tokens[i].type not in (tokenize.NL,)
        ),
        None,
    )
    starts_statement = previous is None or previous.type in (
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.ENCODING,
    )
    ends_statement = following is None or following.type == tokenize.NEWLINE
    return starts_statement and ends_statement


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every non-docstring STRING token's decoded value, with its line number."""
    source = path.read_bytes()
    tokens = list(tokenize.tokenize(io.BytesIO(source).readline))
    literals: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        if token.type == getattr(tokenize, "FSTRING_MIDDLE", None):
            literals.append((token.start[0], token.string))
            continue
        if token.type != tokenize.STRING:
            continue
        if _is_docstring(tokens, index):
            continue
        try:
            value = eval(token.string)  # noqa: S307 - literal string tokens only
        except Exception:
            continue
        if isinstance(value, str):
            literals.append((token.start[0], value))
    return literals


def _segment_hit(literal: str, name: str) -> bool:
    if literal == name:
        return True
    needle = f"/{name}/"
    if needle in f"/{literal}/":
        return True
    return False


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _findings() -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for path in _iter_source_files():
        rel = _relative(path)
        allowed_here = ALLOWED.get(rel, {})
        for line, literal in _string_literals(path):
            for name in LAYOUT_NAMES:
                if name in allowed_here:
                    continue
                if _segment_hit(literal, name):
                    findings.append((rel, line, name, literal))
    return findings


def test_layout_names_are_spelled_only_in_layout_py() -> None:
    """A layout name spelled outside layout.py is a second home for the shape."""
    findings = _findings()
    assert not findings, (
        "These files spell a workspace/bundle layout name outside "
        "workspace/layout.py; route them through that module's constants "
        "and path helpers instead, or add a justified entry to ALLOWED in "
        "this test:\n"
        + "\n".join(
            f"  {path}:{line}: {name!r} in {literal!r}"
            for path, line, name, literal in findings
        )
    )


def test_allowed_entries_are_still_present_in_their_files() -> None:
    """A stale ALLOWED entry (the literal it excused is gone) is dead weight."""
    stale: list[str] = []
    for rel, names in ALLOWED.items():
        path = REPO_ROOT / rel
        if not path.is_file():
            stale.append(f"{rel}: file does not exist")
            continue
        literals = [value for _, value in _string_literals(path)]
        for name in names:
            if not any(_segment_hit(value, name) for value in literals):
                stale.append(f"{rel}: {name!r} does not occur")
    assert not stale, "Stale ALLOWED entries (delete them):\n" + "\n".join(stale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
