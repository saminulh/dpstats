"""Configured-parameter declarations: search policies and field introspection.

A hyperparameter is declared with :func:`parameter_field`, which requires a
one-sentence ``description`` (a field's meaning has one home) and records the
platform's tuning-search policy that Pydantic does not know about. A
structural, non-hyperparameter field uses plain :func:`pydantic.Field`. Either
kind may also declare the short math symbol a generated table shows for it:
``parameter_field(symbol=...)`` for a hyperparameter, and
``Annotated[T, FieldSymbol(...)]`` for any other field. Both spellings store
one :class:`FieldSymbol` that :func:`_field_symbol` reads back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from functools import partial
from html import escape
from inspect import formatannotation
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)
from pydantic.fields import FieldInfo

from dpstats.transport.configuration import (
    PolymorphicModel,
    get_function_path,
    load_object_from_path,
)

SearchScale = Literal["linear", "log"]


class RangeSearch(PolymorphicModel):
    """Data-independent numeric tuning range."""

    lower: int | float = Field(description="Inclusive lower endpoint.")
    upper: int | float = Field(description="Inclusive upper endpoint.")
    scale: SearchScale = Field(
        description="How to space points between the endpoints: 'linear' or 'log'."
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "RangeSearch":
        if (
            isinstance(self.lower, bool)
            or isinstance(self.upper, bool)
            or not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
        ):
            raise ValueError("search-range endpoints must be finite numbers.")
        if self.lower > self.upper:
            raise ValueError(
                "search-range lower bound must not exceed its upper bound."
            )
        if self.scale == "log" and self.lower <= 0:
            raise ValueError("a logarithmic search range requires positive endpoints.")
        return self


class ValuesSearch(PolymorphicModel):
    """Data-independent discrete tuning candidates."""

    values: tuple[Any, ...] = Field(description="The exact candidates, in order.")

    @model_validator(mode="after")
    def _validate_values(self) -> "ValuesSearch":
        if not self.values:
            raise ValueError("ValuesSearch requires at least one candidate.")
        return self


ParameterSearch = RangeSearch | ValuesSearch


@dataclass(frozen=True)
class FieldSymbol:
    """One field's short math symbol, as LaTeX carrying no delimiters.

    Declared inline by :func:`parameter_field` for a hyperparameter and as
    ``Annotated[T, FieldSymbol("n")]`` for any other field but a Task Output's,
    whose annotation maps to a stored column by identity.
    """

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("A field symbol must be a non-blank string.")


@dataclass(frozen=True)
class ParameterInfo:
    """Read-only introspection record for one configured field."""

    annotation: Any
    default: Any
    required: bool
    search: ParameterSearch | None
    description: str
    symbol: str | None


class _ParameterInfoMapping(dict[str, ParameterInfo]):
    """Configured fields with readable terminal and notebook representations."""

    def __repr__(self) -> str:
        entries: list[str] = []
        for name, parameter in self.items():
            default = (
                "required" if parameter.required else f"default={parameter.default!r}"
            )
            heading = f"{name} ({formatannotation(parameter.annotation)}, {default})"
            description = parameter.description.replace("\n", "\n    ")
            entries.append(f"{heading}\n    {description}" if description else heading)
        return "\n\n".join(entries) if entries else "{}"

    def _repr_html_(self) -> str:
        """Render the declared fields as an escaped notebook table."""
        rows: list[str] = []
        for name, parameter in self.items():
            default = "required" if parameter.required else repr(parameter.default)
            values = (
                name,
                formatannotation(parameter.annotation),
                default,
                parameter.description,
            )
            cells = "".join(
                '<td style="text-align: left; white-space: normal;">'
                f"{escape(value)}</td>"
                for value in values
            )
            rows.append(f"<tr>{cells}</tr>")
        return (
            "<table><thead><tr>"
            '<th scope="col">Parameter</th><th scope="col">Type</th>'
            '<th scope="col">Default</th><th scope="col">Description</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )


@dataclass(frozen=True)
class _ParameterPolicy:
    search: ParameterSearch | None = None


def search_range(
    lower: int | float,
    upper: int | float,
    *,
    scale: SearchScale = "linear",
) -> RangeSearch:
    """Declare a linear or logarithmic numeric tuning range.

    Args:
        lower: Finite inclusive lower endpoint.
        upper: Finite inclusive upper endpoint, at least ``lower``.
        scale: How to space points between the endpoints: ``"linear"`` or
            ``"log"``. A logarithmic range requires a positive ``lower``.

    Returns:
        The declared range.

    Raises:
        ValueError: If ``scale`` is not ``"linear"`` or ``"log"``, an
            endpoint is not finite, ``lower`` exceeds ``upper``, or a
            logarithmic range's ``lower`` is not positive.

    Example:
        >>> search_range(1.0, 100.0, scale="log")
        RangeSearch(lower=1.0, upper=100.0, scale='log')
    """
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'.")
    return RangeSearch(lower=lower, upper=upper, scale=scale)


def parse_search_range(
    spec: Mapping[str, Any],
    *,
    name: str,
    default_scale: SearchScale = "linear",
) -> RangeSearch:
    """Parse an untyped ``{lower, upper[, scale]}`` mapping into a range.

    Args:
        spec: A mapping declaring ``lower`` and ``upper`` and optionally ``scale``.
        name: The field name, named in any raised error.
        default_scale: The scale to use when ``spec`` declares none.

    Returns:
        The declared range.

    Raises:
        ValueError: If ``spec`` declares an unexpected key, a non-numeric
            endpoint, an invalid scale, or an otherwise invalid range.

    Example:
        >>> parse_search_range({"lower": 1.0, "upper": 10.0}, name="rate")
        RangeSearch(lower=1.0, upper=10.0, scale='linear')
    """
    if set(spec) not in ({"lower", "upper"}, {"lower", "upper", "scale"}):
        raise ValueError(
            f"Numeric range for {name!r} must contain lower/upper and may "
            "additionally declare scale."
        )
    lower, upper = spec["lower"], spec["upper"]
    for endpoint in (lower, upper):
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            raise ValueError(
                f"Numeric range for {name!r} requires finite numeric endpoints."
            )
    scale = spec.get("scale", default_scale)
    if scale not in ("linear", "log"):
        raise ValueError(
            f"Numeric range for {name!r} has an invalid scale; expected "
            "'linear' or 'log'."
        )
    try:
        return search_range(lower, upper, scale=scale)
    except ValueError as error:
        raise ValueError(f"Numeric range for {name!r} is invalid: {error}") from None


def search_values(*values: Any) -> ValuesSearch:
    """Declare exact discrete tuning candidates.

    Args:
        *values: The exact candidates, in order; at least one is required.

    Returns:
        The declared candidates.

    Raises:
        ValueError: If no candidate is given.

    Example:
        >>> search_values(1, 2, 3)
        ValuesSearch(values=(1, 2, 3))
    """
    return ValuesSearch(values=tuple(values))


def _field_symbol(field: FieldInfo) -> str | None:
    """Return the short math symbol one field declares, or ``None``.

    Args:
        field: The Pydantic field to read, from ``model_fields`` or straight
            from :func:`parameter_field`.

    Returns:
        The declared symbol's text, or ``None`` when the field declares none.

    Example:
        >>> _field_symbol(
        ...     parameter_field(
        ...         default=1.0, description="Clipping norm.", symbol="C"
        ...     )
        ... )
        'C'
    """
    for metadata in field.metadata:
        if isinstance(metadata, FieldSymbol):
            return metadata.text
    return None


def parameter_field(
    *,
    default: Any,
    search: ParameterSearch | None = None,
    description: str,
    symbol: str | None = None,
    **field_options: Any,
) -> Any:
    """Declare one configured field's default, meaning, constraints, and tuning search.

    Wraps :func:`pydantic.Field`, so ``default``, ``description``, and any
    ``field_options`` (``gt``, ``le``, and the rest of Pydantic's constraints)
    apply exactly as they would there. ``search`` additionally records the
    field's recommended tuning range or discrete candidates, read back by
    :func:`inspect_parameters` and :meth:`BaseAlgorithm.default_tuning_grid`;
    it adds no runtime validation of its own.

    Args:
        default: The field's ordinary default value.
        search: The field's recommended tuning search, from
            :func:`search_range` or :func:`search_values`, or ``None`` to
            declare no tuning search.
        description: One-sentence statement of the field's meaning; its one
            home, replacing a class docstring's ``Hyperparameters:`` entry.
        symbol: The field's short math symbol for generated tables, as LaTeX
            carrying no delimiters, or ``None`` to declare none.
        **field_options: Further :func:`pydantic.Field` keyword arguments,
            such as ``gt`` or ``le``.

    Returns:
        A Pydantic ``FieldInfo`` usable as a class-body field default.

    Raises:
        TypeError: If ``search`` is not ``None`` and not a ``RangeSearch`` or
            ``ValuesSearch``.
        ValueError: If ``symbol`` is a blank string.

    Example:
        >>> from dpstats.algorithm import BaseAlgorithm
        >>> class MyAlgorithm(BaseAlgorithm):
        ...     clip_norm: float = parameter_field(
        ...         default=1.0,
        ...         gt=0.0,
        ...         search=search_range(0.1, 10.0, scale="log"),
        ...         description="Per-example gradient clipping norm.",
        ...     )
    """
    if (
        search is not None
        and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            search, (RangeSearch, ValuesSearch)
        )
    ):
        raise TypeError("search must come from search_range() or search_values().")

    field: FieldInfo = Field(default=default, description=description, **field_options)
    field.metadata.append(
        _ParameterPolicy(
            search=search,
        )
    )
    if symbol is not None:
        field.metadata.append(FieldSymbol(symbol))
    return field


def inspect_parameters(
    configured_class: type[PolymorphicModel],
    base_class: type[PolymorphicModel],
) -> dict[str, ParameterInfo]:
    """Inspect configured Pydantic fields while excluding base-class fields."""
    base_fields = set(base_class.model_fields)
    info = _ParameterInfoMapping()

    for name, configured_field in configured_class.model_fields.items():
        if name in base_fields:
            continue
        annotation = configured_field.annotation
        required = configured_field.is_required()
        default = (
            None
            if required
            else configured_field.get_default(call_default_factory=True)
        )
        policy = next(
            (
                metadata
                for metadata in configured_field.metadata
                if isinstance(metadata, _ParameterPolicy)
            ),
            _ParameterPolicy(),
        )
        parameter = ParameterInfo(
            annotation=annotation,
            default=default,
            required=required,
            search=policy.search,
            description=configured_field.description or "",
            symbol=_field_symbol(configured_field),
        )
        info[name] = parameter

    return info


class _ImportableRecord(BaseModel):
    """The persisted form of one keyword-parameterized importable callable."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kwargs: dict[str, Any]


def _load_importable(value: Any) -> Any:
    if isinstance(value, str):
        return load_object_from_path(value)
    if not isinstance(value, Mapping):
        return value
    record = _ImportableRecord.model_validate(dict(value))
    function = load_object_from_path(record.path)
    if not callable(function):
        raise ValueError(
            f"Importable callable path {record.path!r} does not name a callable."
        )
    return partial(function, **record.kwargs)


def _serialize_importable(value: Any) -> str | dict[str, Any]:
    if isinstance(value, partial):
        if value.args:
            raise ValueError(
                "Only keyword-parameterized callables can be persisted by import path."
            )
        return {
            "path": get_function_path(value.func),
            "kwargs": dict(value.keywords or {}),
        }
    if not callable(value):
        raise ValueError(
            f"Expected an importable callable, got {type(value).__name__}."
        )
    return get_function_path(value)


T = TypeVar("T")
Importable = Annotated[
    T,
    BeforeValidator(_load_importable),
    PlainSerializer(_serialize_importable),
]

__all__ = [
    "FieldSymbol",
    "Importable",
    "ParameterInfo",
    "RangeSearch",
    "ValuesSearch",
    "parameter_field",
    "parse_search_range",
    "search_range",
    "search_values",
]
