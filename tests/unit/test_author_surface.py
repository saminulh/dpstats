"""Names the built-in extension implementations import must be author-facing.

Every ``PolymorphicModel`` extension point is a public authoring surface, so
its built-in implementations — task types, algorithms, transformations, budgets,
budget policies, domains, losses, search specs, and tuners — are themselves
treated as user creations: every library name they import must be public,
exported in its module's declared ``__all__``, and live in a module with no
private (underscore) path segment.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import dpstats

_ROOT = Path(dpstats.__file__).parent
_PATTERNS = (
    "tasks/*/__init__.py",
    "tasks/*/_types.py",
    "tasks/*/_metrics.py",
    "tasks/*/algorithms/*.py",
    "tasks/*/transformations/*.py",
    "tasks/*/losses.py",
    "budgets/families.py",
    "sweep/budget_policies.py",
    "domains/__init__.py",
    "parameters.py",
    "dp_tuning/*.py",
    "dp_tuning/selection_costs/*.py",
)


def _is_peer_module(module: str) -> bool:
    """A `tasks.<task>.algorithms`/`.transformations` source is peer author content."""
    parts = module.split(".")
    return (
        len(parts) >= 3
        and parts[0] == "tasks"
        and parts[2] in ("algorithms", "transformations")
    )


def test_every_imported_name_is_author_facing() -> None:
    violations: list[str] = []
    for pattern in _PATTERNS:
        for path in sorted(_ROOT.glob(pattern)):
            rel = path.relative_to(_ROOT)
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if not node.module.startswith("dpstats."):
                    continue
                module = node.module.removeprefix("dpstats.")
                if _is_peer_module(module):
                    continue
                full_module = f"dpstats.{module}"
                for alias in node.names:
                    name = alias.name
                    if (
                        path.name == "__init__.py"
                        and module == f"tasks.{rel.parts[1]}._types"
                        and name
                        in importlib.import_module(
                            f"dpstats.tasks.{rel.parts[1]}"
                        ).__all__
                    ):
                        continue  # a facade re-export of its own value types
                    try:
                        is_submodule = importlib.util.find_spec(f"{full_module}.{name}")
                    except (ImportError, ModuleNotFoundError, ValueError):
                        is_submodule = None
                    if is_submodule:
                        continue  # a subpackage/submodule import, always fine
                    if any(part.startswith("_") for part in module.split(".")):
                        violations.append(
                            f"{rel}: {full_module}.{name} [private module]"
                        )
                    elif name.startswith("_"):
                        violations.append(f"{rel}: {full_module}.{name} [private name]")
                    else:
                        all_ = getattr(
                            importlib.import_module(full_module), "__all__", None
                        )
                        if all_ is not None and name not in all_:
                            violations.append(
                                f"{rel}: {full_module}.{name} [not in __all__]"
                            )
    assert not violations, "author-facing surface violations:\n" + "\n".join(violations)
