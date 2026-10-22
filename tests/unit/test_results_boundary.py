"""Results implementations import their siblings, never the package facade."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "dpstats"


def test_results_implementations_import_siblings_without_the_package_facade() -> None:
    """Internal imports must not depend on the facade's initialization order."""
    for module in (PACKAGE / "results").glob("*.py"):
        if module.name == "__init__.py":
            continue
        tree = ast.parse(module.read_text(), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "dpstats.results", module.name
