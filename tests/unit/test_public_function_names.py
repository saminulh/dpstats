"""Every exported function is named for what it does, starting with a verb.

A declaration factory, a function whose value is a declaration placed in a
class body the way Pydantic's ``Field`` is, keeps the noun of what it declares.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import dpstats

# Imperative verbs an exported function may start with; grow the list with
# the vocabulary, never with a name that merely needs to pass.
VERBS = frozenset(
    """
    aggregate bin build calibrate compute configure declare derive estimate
    fetch generate get identify init list load measure open parse persist
    place plot prepare read resolve run schedule select simulate spend split
    standardize summarize trace validate write
    """.split()
)

# Declaration factories: their value is a declaration, so they are named for it.
DECLARATION_FACTORIES = frozenset(
    {"grid", "parameter_field", "search_range", "search_values"}
)


def _exported_functions() -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    modules = [dpstats.__name__] + [
        module.name
        for module in pkgutil.walk_packages(dpstats.__path__, f"{dpstats.__name__}.")
        if ".templates" not in module.name
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            obj = inspect.unwrap(getattr(module, name, None) or (lambda: None))
            if inspect.isfunction(obj) and obj.__name__ != "<lambda>":
                found.setdefault(name, module_name)
    return sorted(found.items())


EXPORTED = _exported_functions()


@pytest.mark.parametrize(
    "name,module", EXPORTED, ids=[f"{module}.{name}" for name, module in EXPORTED]
)
def test_exported_function_starts_with_a_verb(name: str, module: str) -> None:
    if name in DECLARATION_FACTORIES:
        return
    first = name.split("_", 1)[0]
    assert first in VERBS, (
        f"{module}.{name} does not start with a verb; name it for what it does, "
        "or add its verb to VERBS, or declare it a declaration factory."
    )


def test_every_declared_factory_is_exported() -> None:
    exported = {name for name, _ in EXPORTED}
    assert DECLARATION_FACTORIES <= exported
