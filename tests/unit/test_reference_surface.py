"""Every shared module declaring ``__all__`` is one the reference renders.

A task's own modules are rendered per class in that task's space, never as
whole modules; the built-site suite checks each of their exports has exactly
one home.

The reverse does not hold on this tree: the rendered
``dp_tuning.selection_costs`` family predates the curated-``__all__``
convention and renders its plain public surface instead, so only the
forward direction is an invariant.
"""

from __future__ import annotations

import re
from pathlib import Path

import dpstats

_LIBRARY_ROOT = Path(dpstats.__file__).parent
_REFERENCE_ROOT = _LIBRARY_ROOT.parent / "docs" / "reference"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_LIBRARY_ROOT.parent).parts)
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _declares_all(path: Path) -> bool:
    return re.search(r"^__all__\s*=", path.read_text(), re.M) is not None


def test_all_declaring_modules_are_rendered() -> None:
    declaring = {
        _module_name(path)
        for path in _LIBRARY_ROOT.rglob("*.py")
        if _declares_all(path) and not _module_name(path).startswith("dpstats.tasks.")
    }
    rendered = {
        match
        for source in _REFERENCE_ROOT.glob("*.md")
        for match in re.findall(r"^::: (\S+)$", source.read_text(), re.M)
    }
    only_declaring = sorted(declaring - rendered)
    assert not only_declaring, f"declares __all__ but unrendered: {only_declaring}"
