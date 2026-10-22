"""Explicit post-run Metric declarations and the arguments they read."""

from __future__ import annotations

from typing import Annotated, Any

import keyword
from collections.abc import Callable, Iterable, Mapping
from html import escape

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from dpstats.task import Task

# Task is not a Pydantic model, so it is accepted by instance check alone.
_DECLARATION_CONFIG = ConfigDict(
    extra="forbid", strict=True, arbitrary_types_allowed=True
)

INPUT = "input"
"""The requirement naming the Instance's private Input."""

OUTPUTS = "outputs"
"""The requirement naming the successful Outputs of the measured group."""

RESERVED_REQUIREMENTS = frozenset({INPUT, OUTPUTS})
"""Requirement names measurement supplies itself; never a stored argument."""

Producer = Callable[[Any, Any], Mapping[str, Any]]
"""A function of Metadata and Input returning produced values by name."""


def _require_identifier(name: str, *, owner: str, noun: str) -> None:
    """Require a name that can be injected as a calculation keyword."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{owner} {noun} {name!r} must be a non-keyword identifier.")


def _validate_dependencies(
    *, owner: str, required: tuple[str, ...], supplied: set[str]
) -> None:
    """Require a calculation call to receive exactly its declaration's values."""
    missing = set(required) - supplied
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{owner} is missing declared dependencies: {names}.")
    undeclared = supplied - set(required)
    if undeclared:
        names = ", ".join(sorted(undeclared))
        raise ValueError(f"{owner} received undeclared dependencies: {names}.")


@dataclass(frozen=True, config=_DECLARATION_CONFIG)
class MetricArgument:
    """One named per-Instance value a Metric reads.

    ``description`` is its one-sentence meaning. ``producer`` computes it
    from the Instance's Metadata and Input, returning every value it yields
    by name, so several arguments may share one producer and one fit; it is
    ``None`` for a value the source stores on the Instance.
    """

    name: str
    description: str
    producer: Producer | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.name, owner="MetricArgument", noun="name")
        if self.name in RESERVED_REQUIREMENTS:
            raise ValueError(
                f"A MetricArgument name cannot be the reserved name {self.name!r}."
            )
        if not self.description.strip():
            raise ValueError("MetricArgument.description must be a non-blank string.")

    def __repr__(self) -> str:
        """Name the argument; its producer has no stable text."""
        return f"MetricArgument({self.name!r})"


def produce_arguments(
    arguments: Iterable[MetricArgument], metadata: Any, input: Any
) -> dict[str, Any]:
    """Compute every produced argument, calling each distinct producer once.

    Raises:
        ValueError: If a producer's result lacks an argument it was declared
            to yield.
    """
    by_producer: dict[Producer, list[str]] = {}
    for argument in arguments:
        if argument.producer is None:
            continue
        names = by_producer.setdefault(argument.producer, [])
        if argument.name not in names:
            names.append(argument.name)
    values: dict[str, Any] = {}
    for producer, names in by_producer.items():
        produced = producer(metadata, input)
        for name in names:
            if name not in produced:
                raise ValueError(
                    f"Producer of metric argument {name!r} returned no such value."
                )
            values[name] = produced[name]
    return values


@dataclass(frozen=True, config=_DECLARATION_CONFIG)
class Metric:
    """One named post-run calculation for one Task.

    Extended-real-valued: ``calculate`` may return ``±inf``, and a NaN
    scores as ``+inf``. ``requires`` declares the arguments the calculation
    receives by keyword: MetricArgument declarations, the name of a value
    the Instance stores without a declaration, ``"input"`` for the private
    Input, or ``"outputs"`` for the group's successful Outputs. ``label`` is
    the short text a page prints where a reader, not code, reads the Metric;
    it is outside the Metric's identity. ``baseline`` is the Metric's
    statistical baseline: the sampling error of the nonprivate estimator
    whose error the Metric measures, on the Metric's own scale, as a
    MetricArgument or the name of a value the Instance stores. It is the
    default a cost is read against, not a restriction: any Metric may be
    divided by any per-Instance value. A calculation and a producer read the
    Input they receive and never write to it: one loaded Input serves every
    Metric and group measuring an Instance.
    """

    task: Task
    name: Annotated[str, Field(min_length=1)]
    calculation: Callable[..., float]
    requires: tuple[str | MetricArgument, ...] = ()
    # This Metric's definition as LaTeX display math carrying no delimiters;
    # None where its calculation has no closed form worth displaying.
    formula: str | None = None
    label: str = ""
    baseline: MetricArgument | str | None = None

    def __post_init__(self) -> None:
        if self.formula is not None and not self.formula.strip():
            raise ValueError("Metric.formula must be a non-blank string or None.")
        for requirement in self.requires:
            if (
                isinstance(requirement, str)
                and requirement not in RESERVED_REQUIREMENTS
            ):
                _require_identifier(requirement, owner="Metric", noun="requirement")
        if len(set(self.required_names)) != len(self.required_names):
            raise ValueError("Metric requirements must not repeat a name.")
        if isinstance(self.baseline, str):
            _require_identifier(self.baseline, owner="Metric", noun="baseline")

    def __str__(self) -> str:
        """Name the measurement column this Metric produces."""
        return self.name

    def _repr_html_(self) -> str:
        """Display the existing declaration without evaluating or loading data."""
        details = [
            ("Name", self.name),
            ("Task", self.task.name),
            ("Requires", ", ".join(self.required_names) or "None"),
            ("Baseline", self.baseline_name or "None declared"),
        ]
        if self.formula is not None:
            details.append(("Formula (LaTeX)", self.formula))
        rows = "".join(
            f'<tr><th scope="row">{escape(name)}</th>'
            f"<td><code>{escape(value)}</code></td></tr>"
            for name, value in details
        )
        return (
            f"<div><strong>{escape(self.label or self.name)}</strong>"
            f"<table><tbody>{rows}</tbody></table></div>"
        )

    @property
    def required_names(self) -> tuple[str, ...]:
        """Return the keyword names this Metric's calculation is given."""
        return tuple(
            requirement.name if isinstance(requirement, MetricArgument) else requirement
            for requirement in self.requires
        )

    @property
    def baseline_name(self) -> str | None:
        """Return the name of this Metric's statistical baseline, if it has one."""
        if isinstance(self.baseline, MetricArgument):
            return self.baseline.name
        return self.baseline

    @property
    def arguments(self) -> tuple[MetricArgument, ...]:
        """Return the declared arguments this Metric requires or divides by."""
        declared = [item for item in self.requires if isinstance(item, MetricArgument)]
        if isinstance(self.baseline, MetricArgument) and self.baseline not in declared:
            declared.append(self.baseline)
        return tuple(declared)

    def calculate(self, output: Any, **dependencies: Any) -> float:
        """Calculate this Metric from one Output and exactly declared values."""
        _validate_dependencies(
            owner=f"Metric {self.name!r}",
            required=self.required_names,
            supplied=set(dependencies),
        )
        return self.calculation(output, **dependencies)


def declared_arguments(
    metrics: Iterable[Metric], arguments: Iterable[MetricArgument] = ()
) -> dict[str, MetricArgument]:
    """Collect the arguments these Metrics declare, keyed by name.

    ``arguments`` are declared first, so a Metric redeclaring one of them
    differently is the one named in the refusal.

    Raises:
        ValueError: If one name carries two different declarations.
    """
    collected: dict[str, MetricArgument] = {}
    declared = [
        *(("A declared argument", a) for a in arguments),
        *((f"Metric {m.name!r}", a) for m in metrics for a in m.arguments),
    ]
    for owner, argument in declared:
        previous = collected.setdefault(argument.name, argument)
        if previous != argument:
            raise ValueError(
                f"{owner} declares argument {argument.name!r} differently from "
                "another declaration."
            )
    return collected


__all__ = ["Metric", "MetricArgument", "Producer"]
