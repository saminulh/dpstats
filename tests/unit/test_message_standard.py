"""STYLE S26 and S27: what a refusal message and an ``InvalidRegime`` reason may say.

The checks walk every ``raise`` in ``dpstats/`` with
``ast``. A tool can decide the shape of a message, not whether it names the
right public condition, so each rule is covered only where it is decidable.

S26 checks that a raised exception carries a message that is a literal or an
f-string, that the message ends with a full stop, and that no interpolated
value is private data. Whether the message names the failed condition, the
public value, and the change to make is judged in review.

S27 checks that an ``InvalidRegime`` reason is one sentence, with no line
break.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

LIBRARY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (LIBRARY_ROOT / "dpstats",)

# Names that carry private data when interpolated as a value. A message may
# interpolate their shape, size, or declared bound, never the value itself.
PRIVATE_NAMES = frozenset(
    {"X", "y", "input", "data", "rows", "theta", "beta", "Input", "inputs"}
)
# Attributes that turn a private array into a public description of it.
PUBLIC_DESCRIPTIONS = frozenset({"shape", "ndim", "size", "dtype", "__name__"})
# Calls that turn a private value into a public description of it.
PUBLIC_CALLS = frozenset({"len", "type"})

# Deliberate exceptions to the deny list: the name is public at that site.
# Entry: (path relative to the library root, name) -> why it is public there.
PUBLIC_AT_SITE: dict[tuple[str, str], str] = {
    (
        "dpstats/tasks/median_estimation/algorithms/smooth_sensitivity.py",
        "beta",
    ): "the smoothness parameter of the mechanism, set by the configuration",
}


@dataclass(frozen=True)
class Raise:
    path: str
    line: int
    exception: str
    message: ast.expr

    @property
    def site(self) -> str:
        return f"{self.path}:{self.line}"


def _raises() -> Iterator[Raise]:
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(LIBRARY_ROOT)
            if {"tests", ".venv", "__pycache__"} & set(relative.parts):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    call = node.exc
                    name = ast.unparse(call.func)
                    if call.args:
                        yield Raise(
                            relative.as_posix(), node.lineno, name, call.args[0]
                        )
                    else:
                        yield Raise(
                            relative.as_posix(), node.lineno, name, ast.Constant("")
                        )


def _operands(message: ast.expr) -> list[ast.expr]:
    """The pieces of ``a + b + c``, in order; any other expression is one piece."""
    if isinstance(message, ast.BinOp) and isinstance(message.op, ast.Add):
        return _operands(message.left) + _operands(message.right)
    return [message]


def _literal_parts(piece: ast.expr) -> list[str] | None:
    """Literal text of a string or f-string, with ``{}`` for each field; else None."""
    if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
        return [piece.value]
    if isinstance(piece, ast.JoinedStr):
        return [
            part.value if isinstance(part, ast.Constant) else "{}"
            for part in piece.values
        ]
    return None


def _message_text(message: ast.expr) -> str | None:
    """The message with ``{}`` for every dynamic part, or None when it is not authored."""
    text = ""
    for piece in _operands(message):
        parts = _literal_parts(piece)
        text += "".join(parts) if parts is not None else "{}"
    return (
        text if any(_literal_parts(p) is not None for p in _operands(message)) else None
    )


def _private_values(expression: ast.expr) -> Iterator[str]:
    """Names in an interpolated expression that stand for private values."""
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        if expression.func.id in PUBLIC_CALLS:
            return
    if isinstance(expression, ast.Attribute) and expression.attr in PUBLIC_DESCRIPTIONS:
        return
    if isinstance(expression, ast.Name) and expression.id in PRIVATE_NAMES:
        yield expression.id
    if isinstance(expression, ast.Attribute) and expression.attr in PRIVATE_NAMES:
        yield expression.attr
    for child in ast.iter_child_nodes(expression):
        if isinstance(child, ast.expr):
            yield from _private_values(child)


def _interpolations(message: ast.expr) -> Iterator[ast.expr]:
    for piece in _operands(message):
        if isinstance(piece, ast.JoinedStr):
            for part in piece.values:
                if isinstance(part, ast.FormattedValue):
                    yield part.value


RAISES = list(_raises())


def test_the_walk_finds_the_raises_it_checks() -> None:
    assert len(RAISES) > 500
    assert any(r.exception == "InvalidRegime" for r in RAISES)


def test_every_raised_exception_carries_an_authored_message() -> None:
    """A bare ``raise ValueError()`` or a message built from a variable is refused."""
    unauthored = [
        f"{r.site}: {r.exception}"
        for r in RAISES
        if _message_text(r.message) in (None, "", "{}")
    ]
    assert not unauthored, "\n".join(unauthored)


def test_every_message_ends_with_a_full_stop() -> None:
    """The last authored piece ends with a full stop or question mark.

    A message that ends in an interpolated value passes: the public value is
    what the sentence names. One that ends in a joined list of findings passes
    for the same reason, each finding being a message of its own.
    """
    open_ended = []
    for r in RAISES:
        pieces = _operands(r.message)
        last = _literal_parts(pieces[-1])
        if last is None:
            continue
        text = "".join(last).rstrip()
        if text and text[-1] not in ".?" and not text.endswith("{}"):
            open_ended.append(f"{r.site}: ...{text[-40:]!r}")
    assert not open_ended, "\n".join(open_ended)


def test_no_message_interpolates_a_private_value() -> None:
    """Interpolating ``X``, ``y``, ``input``, ``data``, ``rows``, ``theta``, or ``beta`` is refused.

    Their ``shape``, ``ndim``, ``size``, ``dtype``, and ``len`` are public
    descriptions and pass. A site where one of those names is public by
    construction goes in ``PUBLIC_AT_SITE`` with its reason.
    """
    leaks = []
    for r in RAISES:
        for expression in _interpolations(r.message):
            for name in _private_values(expression):
                if (r.path, name) not in PUBLIC_AT_SITE:
                    leaks.append(f"{r.site}: {{{ast.unparse(expression)}}}")
    assert not leaks, "\n".join(leaks)


# --- S27 -----------------------------------------------------------------

REGIME_REASONS = [r for r in RAISES if r.exception == "InvalidRegime"]
SENTENCE_BREAK = re.compile(r"[.!?](?=\s+\S)")


def test_the_walk_finds_the_invalid_regime_reasons() -> None:
    assert len(REGIME_REASONS) > 80


def test_every_invalid_regime_reason_is_one_sentence() -> None:
    """One terminal full stop, no line break, no sentence boundary inside."""
    problems = []
    for r in REGIME_REASONS:
        text = _message_text(r.message)
        if text is None or not text.endswith("."):
            problems.append(f"{r.site}: does not end with one full stop")
        elif "\n" in text:
            problems.append(f"{r.site}: contains a line break")
        elif SENTENCE_BREAK.search(text):
            problems.append(f"{r.site}: more than one sentence: {text!r}")
    assert not problems, "\n".join(problems)
