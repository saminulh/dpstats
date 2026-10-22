"""The installable user distribution carries and needs library code only.

The wheel carries the ``dpstats`` package and nothing else; the first test
runs package discovery as the project configures it and refuses any other
package.

Package discovery is configured to ignore directories that carry no
``__init__.py``. That keeps an undeclared directory out of the wheel, but the
omission is invisible from a source checkout, where such a directory still
imports: the first failure would otherwise land on a wheel user. The second
test below is that missing signal, comparing configured discovery against the
permissive walk and naming whatever only the permissive walk can reach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from setuptools import find_namespace_packages, find_packages

REPO_ROOT = Path(__file__).parents[2]
LIBRARY_ROOT = REPO_ROOT / "dpstats"


def _project_configuration() -> dict[str, Any]:
    """The project's own ``pyproject.toml``, parsed."""
    tomllib = pytest.importorskip("tomllib", reason="pyproject parsing needs 3.11+")
    parsed: dict[str, Any] = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return parsed


def _discovery_filters(configuration: dict[str, Any]) -> dict[str, Any]:
    """The configured package-discovery filters."""
    filters: dict[str, Any] = configuration["tool"]["setuptools"]["packages"]["find"]
    return filters


def _discover(configuration: dict[str, Any], *, namespaces: bool) -> set[str]:
    """Run one discovery walk under the project's configured filters.

    ``namespaces`` selects the walk rather than reading the configured value,
    so a caller can compare the configured mode against the permissive one.
    """
    discovery = _discovery_filters(configuration)
    finder = find_namespace_packages if namespaces else find_packages
    return set(
        finder(
            where=str(REPO_ROOT / discovery["where"][0]),
            include=discovery["include"],
            exclude=discovery.get("exclude", ()),
        )
    )


def _declared_data_directories(configuration: dict[str, Any]) -> set[str]:
    """Directories that exist to hold declared package data, not modules.

    Derived from the ``package-data`` patterns themselves so the exemption
    cannot drift from what the build actually ships. Every ancestor counts:
    a data directory's parents are equally not modules.
    """
    package_data: dict[str, list[str]] = configuration["tool"]["setuptools"][
        "package-data"
    ]
    directories: set[str] = set()
    for package, patterns in package_data.items():
        root = REPO_ROOT / package.replace(".", "/")
        for pattern in patterns:
            for match in root.glob(pattern):
                if not match.is_file():
                    continue
                parts = match.parent.relative_to(root).parts
                for depth in range(1, len(parts) + 1):
                    directories.add(".".join((package, *parts[:depth])))
    return directories


def test_the_user_distribution_contains_no_application_package() -> None:
    """Package discovery, run in the mode the project actually configures."""
    configuration = _project_configuration()
    discovery = _discovery_filters(configuration)
    discovered = _discover(configuration, namespaces=discovery.get("namespaces", True))

    assert discovered, "package discovery found no library packages at all"
    outside = [name for name in discovered if not name.startswith("dpstats")]
    assert outside == []


def test_no_source_directory_is_silently_dropped_from_the_distribution() -> None:
    """A directory the build skips but a source checkout imports must be named.

    Configured discovery skips any directory without an ``__init__.py``. The
    permissive walk does not, so whatever only it can reach is a directory that
    imports here and is missing from the wheel -- unless it holds declared
    package data, which ships as data and is never imported.
    """
    configuration = _project_configuration()
    discovery = _discovery_filters(configuration)

    configured = _discover(configuration, namespaces=discovery.get("namespaces", True))
    permissive = _discover(configuration, namespaces=True)
    # A directory holding no Python source, such as the bytecode a deleted
    # package leaves behind, has nothing to import.
    undeclared = sorted(
        name
        for name in permissive - configured - _declared_data_directories(configuration)
        if any((REPO_ROOT / name.replace(".", "/")).glob("*.py"))
    )

    assert undeclared == [], (
        "these directories import from a source checkout but are absent from "
        f"the built distribution: {undeclared}. Add an __init__.py to declare "
        "each as a package, or declare its contents in package-data."
    )
