"""Private Pydantic configuration shared by public platform value classes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from functools import cached_property
from importlib import import_module
from typing import (
    cast,
    Annotated,
    Any,
    TypeVar,
)

from pydantic import (
    GetCoreSchemaHandler,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)

TPolymorphic = TypeVar("TPolymorphic", bound="PolymorphicModel")

LOAD_CONFIG_CONTEXT = "dpstats.load_config"
CONCRETE_CONFIG_ROOT = "dpstats.load_config.concrete_root"
DUMP_CONFIG_CONTEXT = "dpstats.dump_config"
IDENTITY_CONTEXT = "dpstats.identity"
STRUCTURAL_COMPARISON_CONTEXT = "dpstats.structural_comparison"


def load_object_from_path(path: str) -> Any:
    """Load one configured module-qualified object."""
    if ":" not in path:
        raise ValueError("Configured object paths use 'module:qualname'.")
    module_name, qualified_name = path.rsplit(":", 1)
    if not module_name or not qualified_name:
        raise ValueError(f"Invalid configured object path {path!r}.")
    resolved: Any = import_module(module_name)
    for name in qualified_name.split("."):
        resolved = getattr(resolved, name)
    return resolved


def get_function_path(function: Callable[..., Any]) -> str:
    """Return the portable path for one module-level configured callable."""
    module_name = getattr(function, "__module__", None)
    name = getattr(function, "__name__", None)
    qualified_name = getattr(function, "__qualname__", name)
    if not isinstance(module_name, str) or not isinstance(name, str):
        raise ValueError(f"Cannot serialize callable {function!r}: no importable name.")
    if module_name == "__main__":
        raise ValueError(
            f"Cannot serialize __main__:{qualified_name}: a __main__ definition "
            "does not import outside this process."
        )
    if qualified_name != name:
        raise ValueError(
            f"Cannot serialize {module_name}:{qualified_name}: only module-level "
            "functions round-trip through configuration."
        )
    return f"{module_name}:{name}"


class ConfigurationRecord(BaseModel):
    """The exact recursive envelope returned by configuration dumps."""

    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(min_length=1)]
    config: dict[str, Any]


def _load_config_class(path: str) -> type[PolymorphicModel]:
    """Resolve one qualified path to a PolymorphicModel class."""
    resolved = load_object_from_path(path)
    if not isinstance(resolved, type) or not issubclass(resolved, PolymorphicModel):
        raise TypeError(f"{path!r} does not resolve to a PolymorphicModel subclass.")
    return resolved


def _relax_tuple_nodes(schema: Any) -> None:
    """Accept JSON arrays at tuple fields while their items stay strict.

    A wrap model validator makes pydantic re-check parsed values under
    python-mode strictness, which refuses a list where a tuple is declared,
    so every tuple container node is relaxed once at class definition.
    """
    if isinstance(schema, dict):
        node = cast("dict[str, Any]", schema)
        if node.get("type") in ("tuple", "tuple-variable", "tuple-positional"):
            node["strict"] = False
        for key, item in node.items():
            if key != "cls":
                _relax_tuple_nodes(item)
    elif isinstance(schema, list):
        items = cast("list[Any]", schema)
        for item in items:
            _relax_tuple_nodes(item)


def evict_undeclared_copy_state(copied: BaseModel) -> None:
    """Drop attribute-dict entries a copy inherited but its class never declared.

    A cached derived value — a content identity above all — belongs to the
    value it was derived from; an updated copy that carried it would answer
    with the original's identity. Every frozen model that caches a derived
    property must route its copies through this eviction.
    """
    declared = type(copied).model_fields.keys() | type(copied).__class_vars__
    for name in tuple(copied.__dict__):
        if name not in declared:
            object.__delattr__(copied, name)


def identity_payload(value: BaseModel, *, exclude: Collection[str]) -> dict[str, Any]:
    """Dump one model as the exact payload its content identity covers.

    ``exclude`` names every field this identity does not cover, and is the
    only source of that decision: the names are dropped from the model's own
    fields here and, through ``IDENTITY_CONTEXT``, from every configured
    component nested inside, at any depth.
    """
    excluded = frozenset(exclude)
    return value.model_dump(
        mode="python",
        exclude=set(excluded),
        exclude_computed_fields=True,
        context={DUMP_CONFIG_CONTEXT: True, IDENTITY_CONTEXT: excluded},
    )


class PolymorphicModel(BaseModel):
    """Frozen, strict Pydantic model whose configuration dump names its concrete class, so it round-trips to that subclass."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        polymorphic_serialization=True,
        # The model serializer routes every dump through pydantic's JSON float
        # handling, which would otherwise null non-finite values silently.
        ser_json_inf_nan="constants",
        strict=True,
        validate_default=True,
    )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> Any:
        if set(cls.__pydantic_decorators__.model_serializers) != {
            "_envelope_configuration"
        }:
            raise TypeError(
                f"{cls.__qualname__} cannot declare a model serializer; "
                "configuration serialization owns that role."
            )
        schema = handler(source)
        _relax_tuple_nodes(schema)
        return schema

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy this model without carrying values derived from the original.

        An updated copy is a different configured value, so cached content
        identities must not travel with it; undeclared attribute-dict entries
        are dropped.
        """
        copied = super().model_copy(update=update, deep=deep)
        evict_undeclared_copy_state(copied)
        return copied

    @cached_property
    def _configuration_record(self) -> ConfigurationRecord:
        """Derive this frozen value's portable configuration once."""
        structural = self.model_dump(
            mode="python",
            exclude_computed_fields=True,
            context={DUMP_CONFIG_CONTEXT: True},
        )
        value = decode_json_document(canonical_json_bytes(structural))
        return ConfigurationRecord.model_validate(value)

    def dump_config(self) -> ConfigurationRecord:
        """Return an independent copy of this model's portable configuration."""
        return self._configuration_record.model_copy(deep=True)

    @model_serializer(mode="wrap")
    def _envelope_configuration(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        dumped = handler(self)
        context = info.context
        if not isinstance(context, dict):
            return dumped
        comparing = bool(context.get(STRUCTURAL_COMPARISON_CONTEXT))
        if not context.get(DUMP_CONFIG_CONTEXT) and not comparing:
            return dumped
        # The nested arm of ``identity_payload``: the caller's excluded names
        # leave every configured component inside, at any nesting depth.
        excluded = cast("Collection[str]", context.get(IDENTITY_CONTEXT, ()))
        if excluded:
            fields = cast("dict[str, Any]", dumped)
            for name in excluded:
                fields.pop(name, None)
        if comparing:
            return {"class": type(self), "config": dumped}
        if type(self).__module__ == "__main__":
            raise ValueError(
                f"Cannot serialize __main__:{type(self).__qualname__}: a "
                "__main__ definition does not import outside this process."
            )
        path = f"{type(self).__module__}:{type(self).__qualname__}"
        if _load_config_class(path) is not type(self):
            raise ValueError(
                f"Configured class path {path!r} resolves to another class."
            )
        return {"path": path, "config": dumped}

    @model_validator(mode="wrap")
    @classmethod
    def _restore_configuration(
        cls: type[TPolymorphic],
        value: Any,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> TPolymorphic:
        """Restore concrete nested classes only during configuration loading."""
        context: Any = info.context
        if not isinstance(context, dict) or not context.get(LOAD_CONFIG_CONTEXT):
            return handler(value)
        # The shared context's concrete-class flag is single-use: the
        # outermost model consumes it; nested models still resolve records.
        if context.pop(CONCRETE_CONFIG_ROOT, False):
            return handler(value)
        if not isinstance(value, Mapping) or set(value) != {"path", "config"}:
            return handler(value)
        record = ConfigurationRecord.model_validate(dict(value))
        path = record.path
        config = record.config
        concrete = _load_config_class(path)
        if not issubclass(concrete, cls):
            raise ValueError(
                f"Configured class {path!r} is outside the {cls.__name__} family."
            )
        return concrete.load_config(config)

    @classmethod
    def load_config(cls: type[TPolymorphic], config: dict[str, Any]) -> TPolymorphic:
        """Validate this class's own configuration fields into a value.

        The loading counterpart of ``model_validate``: validation runs with
        code loading enabled, so a nested configuration record imports and
        rebuilds the class it names. ``model_validate`` remains the pure form
        that loads nothing.
        """
        return cls.model_validate(
            config,
            context={LOAD_CONFIG_CONTEXT: True, CONCRETE_CONFIG_ROOT: True},
        )

    @classmethod
    def load_record(
        cls: type[TPolymorphic], record: ConfigurationRecord | dict[str, Any]
    ) -> TPolymorphic:
        """Load the configured value a record names within this family."""
        parsed = ConfigurationRecord.model_validate(record)
        concrete = _load_config_class(parsed.path)
        if not issubclass(concrete, cls):
            raise TypeError(
                f"Configured class {parsed.path!r} is outside the "
                f"{cls.__name__} family."
            )
        return concrete.load_config(parsed.config)
