"""Structural checks for the documented public Python API.

Google permits a summary-only docstring when it sufficiently explains the
call, and omits Returns for None or a return value already explained by the
summary. Those are editorial decisions. These tests check presence and the
accuracy of supplied parameter sections; they impose no section, example,
or prose-length quota. Runnable examples are checked separately.
"""

from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from tests.docstring_blocks import named_block

# Every module the docstring-standard ratchet (pyproject.toml's ruff
# per-file-ignores) already holds clean; each one's `__all__` is this test's
# curated surface.
_MODULES: list[str] = [
    "dpstats.analysis",
    "dpstats.configure",
    "dpstats.parameters",
    "dpstats.instance",
    "dpstats.metric",
    "dpstats.task",
    "dpstats.budgets",
    "dpstats.budgets.accounting",
    "dpstats.dp_tuning",
]

_ARGS_HEADER = re.compile(r"^([ \t]*)Args:[ \t]*$", re.MULTILINE)


@dataclass
class _Check:
    """One name's pass/fail facts, plus human-readable failure notes."""

    name: str
    has_purpose: bool = False
    args_complete: bool = True
    fields_described: bool = True
    notes: list[str] = field(default_factory=list[str])

    @property
    def ok(self) -> bool:
        return self.has_purpose and self.args_complete and self.fields_described


def _documented_names(args_text: str) -> set[str]:
    """Return the parameter names one ``Args:`` block documents."""
    lines = [line for line in args_text.splitlines() if line.strip()]
    if not lines:
        return set()
    indent = min(len(line) - len(line.lstrip()) for line in lines)
    return set(
        re.findall(
            rf"^[ \t]{{{indent}}}\*{{0,2}}(\w+)[ \t]*(?:\(.*?\))?:",
            args_text,
            re.MULTILINE,
        )
    )


def _signature_params(func: Any) -> list[str]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [name for name in signature.parameters if name not in {"self", "cls"}]


def _own_model_fields(cls: type[BaseModel]) -> dict[str, Any]:
    """Return the Pydantic fields this class declares, excluding inherited ones."""
    inherited: set[str] = set()
    for base in cls.__bases__:
        if issubclass(base, BaseModel):
            inherited |= set(base.model_fields)
    return {
        name: info for name, info in cls.model_fields.items() if name not in inherited
    }


def _check_function(name: str, func: Any) -> _Check:
    result = _Check(name=name)
    doc = inspect.getdoc(func) or ""
    result.has_purpose = bool(doc.strip())
    params = _signature_params(func)
    args_text = named_block(doc, _ARGS_HEADER)
    if args_text is not None:
        documented = _documented_names(args_text)
        missing = [p for p in params if p not in documented]
        extra = sorted(documented - set(params))
        if missing or extra:
            result.args_complete = False
            result.notes.append(f"Args: missing {missing}; unknown {extra}")
    return result


def _check_class(name: str, cls: type) -> _Check:
    result = _Check(name=name)
    doc = inspect.getdoc(cls) or ""
    result.has_purpose = bool(doc.strip())
    if issubclass(cls, BaseModel):
        own_fields = _own_model_fields(cls)
        undescribed = [
            field_name
            for field_name, info in own_fields.items()
            if not info.description
        ]
        if undescribed:
            result.fields_described = False
            result.notes.append(f"Field(description=...) missing on {undescribed}")
    return result


def _curated_surface() -> list[tuple[str, str, Any]]:
    """Resolve one (qualified name, kind, object) triple per `__all__` entry."""
    resolved: list[tuple[str, str, Any]] = []
    for module_path in _MODULES:
        module = importlib.import_module(module_path)
        for exported in module.__all__:
            obj = inspect.unwrap(getattr(module, exported))
            if inspect.isclass(obj):
                kind = "class"
            elif inspect.isfunction(obj):
                kind = "function"
            else:
                # A type alias or module-level constant (e.g. `task.Array1D`,
                # `budgets.NO_PRIVACY`) carries no docstring of its own to
                # check; ruff's `D` rules already require its owning
                # module/class to be documented.
                continue
            resolved.append((f"{module_path}.{exported}", kind, obj))
    return resolved


def test_class_description_can_explain_use_beyond_the_summary() -> None:
    """A collection's full storage contract is valid class documentation."""
    from dpstats import TaskCollection

    assert _check_class("TaskCollection", TaskCollection).ok


_SURFACE = _curated_surface()


@pytest.mark.parametrize(
    "name,kind,target", _SURFACE, ids=[item[0] for item in _SURFACE]
)
def test_curated_surface_has_documentation_with_valid_sections(
    name: str, kind: str, target: Any
) -> None:
    check = (
        _check_class(name, target) if kind == "class" else _check_function(name, target)
    )
    assert check.ok, "; ".join(check.notes)


def _budget_methods() -> list[tuple[str, Any]]:
    """Return every public method a budget family exports or overrides."""
    budgets = importlib.import_module("dpstats.budgets")
    found: dict[str, Any] = {}
    for exported in budgets.__all__:
        cls = getattr(budgets, exported)
        if not inspect.isclass(cls):
            continue
        for name, member in vars(cls).items():
            if not name.startswith("_") and inspect.isfunction(member):
                found[f"{cls.__name__}.{name}"] = member
    return sorted(found.items())


_BUDGET_METHODS = _budget_methods()


@pytest.mark.parametrize(
    "name,method", _BUDGET_METHODS, ids=[item[0] for item in _BUDGET_METHODS]
)
def test_budget_public_methods_describe_their_parameters(
    name: str, method: Any
) -> None:
    """Documented budget parameters match the callable signature."""
    check = _check_function(name, method)
    assert check.ok, "; ".join(check.notes)
