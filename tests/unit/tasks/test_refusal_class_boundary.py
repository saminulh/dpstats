"""Task ``run`` and ``prepare`` bodies raise only the recorded outcome classes.

This test is the boundary's consumer. Execution reads a cell's outcome from
the class a task raises: ``InvalidRegime`` — including its ``NotApplicable``
subclass, which a composition instead consumes as a pass-through — and
``NoRelease`` are recorded with their authored message, and every other
exception is recorded as an execution error by type name alone.
``RuntimeError`` is the one spelling of an execution failure.
``NotImplementedError`` is the one spelling of a declared gap: a regime whose
behavior exists, such as an infinite-parameter limit, but is not yet
implemented. Its recorded type name keeps it apart from failures in results.
A body reaching for any other class either meant one of the message-keeping
outcomes and silently discarded its reason, or spelled one of these two
outcomes a second way.

The walk is lexical over ``run`` and ``prepare`` bodies. A refusal a body
delegates to a module helper is outside what this boundary can see.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dpstats.tasks as tasks

TASKS_DIRECTORY = Path(tasks.__file__).resolve().parent

REFUSAL_CLASSES = frozenset(
    {
        "InvalidRegime",
        "NotApplicable",
        "NoRelease",
        "NotImplementedError",
        "RuntimeError",
    }
)

# Refusals not swept yet, as ``<module>: <class>``. Each entry is deleted with
# the sweep of its site, which this test then requires; a second unswept
# refusal in the same module fails the comparison rather than hiding behind an
# entry already here.
UNSWEPT_REFUSALS: tuple[str, ...] = ()


def _raised_class_name(statement: ast.Raise) -> str | None:
    """Name the class a ``raise`` constructs, or ``None`` if it re-raises."""
    exception = statement.exc
    if exception is None:
        return None
    if isinstance(exception, ast.Call):
        exception = exception.func
    if isinstance(exception, ast.Attribute):
        return exception.attr
    if isinstance(exception, ast.Name):
        return exception.id
    return ast.unparse(exception)


def test_run_and_prepare_raise_only_recorded_refusal_classes() -> None:
    unswept: list[str] = []
    located: list[str] = []
    for module in sorted(TASKS_DIRECTORY.rglob("*.py")):
        relative = module.relative_to(TASKS_DIRECTORY).as_posix()
        tree = ast.parse(module.read_text(), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in {
                "run",
                "prepare",
            }:
                continue
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Raise):
                    continue
                raised = _raised_class_name(statement)
                if raised is not None and raised not in REFUSAL_CLASSES:
                    unswept.append(f"{relative}: {raised}")
                    located.append(f"{relative}:{statement.lineno} raises {raised}")

    assert sorted(unswept) == sorted(UNSWEPT_REFUSALS), located
