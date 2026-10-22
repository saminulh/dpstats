"""In-process doctest gate over the installed package.

Walks `dpstats` with `pkgutil`, imports every module it finds,
and runs `doctest.testmod` on each with `NORMALIZE_WHITESPACE | ELLIPSIS`.
Doctest honors a per-example `# doctest: +SKIP` natively, which is how a
data- or network-touching example (for example `fetch_benchmark`'s) opts
out. Every module is pure Python source with no I/O of its own, so the
whole gate runs in-process with no fixture, no data, and no network.

A private (leading-underscore) submodule is skipped unless some package's
`__all__` re-exports one of its names, matching `test_docstring_standard.py`'s
notion of the curated surface: implementation detail carries no doctest
obligation, but a name a user can actually import does.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import dpstats

_OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


def _walk_module_names() -> list[tuple[str, bool]]:
    """Return every (dotted name, is_package) reachable under the package."""
    return [
        (name, is_pkg)
        for _, name, is_pkg in pkgutil.walk_packages(
            dpstats.__path__, prefix="dpstats."
        )
    ]


def _reexported_module_names(all_names: list[tuple[str, bool]]) -> set[str]:
    """Return every module some package's `__all__` re-exports a name from."""
    reexported: set[str] = set()
    for name, is_pkg in all_names:
        if not is_pkg:
            continue
        package = importlib.import_module(name)
        for exported in getattr(package, "__all__", ()):
            owner = getattr(getattr(package, exported, None), "__module__", None)
            if owner:
                reexported.add(owner)
    return reexported


def _doctestable_modules() -> list[str]:
    """Return every module name this gate imports and doctests."""
    all_names = _walk_module_names()
    reexported = _reexported_module_names(all_names)
    names = ["dpstats"]
    for name, _ in all_names:
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("_") and name not in reexported:
            continue
        names.append(name)
    return names


_MODULE_NAMES = _doctestable_modules()


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_doctests_pass(module_name: str) -> None:
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, optionflags=_OPTIONFLAGS)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {module_name}"


def test_doctest_gate_covers_the_package() -> None:
    """A near-empty module list would silently stop gating almost everything."""
    assert len(_MODULE_NAMES) > 100
