"""The unit gate never re-enters a real site build.

The strict mkdocs build runs only through docs/site/check_build.py, never
from this gate. This is that
boundary's consumer: a file under `tests/unit` that imports `mkdocs`'s build
commands, `assemble` (`docs/site/assemble.py`, which runs a full build as an
import side effect), or a `build_site` helper has silently reintroduced it.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_UNIT_DIRECTORY = Path(__file__).resolve().parent
BANNED_MODULES = {"mkdocs", "assemble", "docs.site.assemble"}


def _is_banned_module(name: str) -> bool:
    # `mkdocs.config` and plugin modules only parse configuration; the build
    # entry points live under `mkdocs.commands`.
    return name in BANNED_MODULES or name.startswith("mkdocs.commands")


def test_no_unit_test_imports_a_site_build() -> None:
    offending: list[tuple[str, str]] = []
    for module in sorted(TESTS_UNIT_DIRECTORY.rglob("*.py")):
        tree = ast.parse(module.read_text(), filename=str(module))
        relative = module.relative_to(TESTS_UNIT_DIRECTORY.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_banned_module(alias.name):
                        offending.append((str(relative), alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and _is_banned_module(node.module):
                    offending.append((str(relative), node.module))
                elif any(alias.name == "build_site" for alias in node.names):
                    offending.append((str(relative), f"{node.module}.build_site"))
    assert offending == []
