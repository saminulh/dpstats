"""Describe the exact registered configuration behind a displayed result."""

from __future__ import annotations

import csv
import inspect
from importlib import import_module
from pathlib import Path

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    Transformation,
    load_algorithms,
)
from dpstats.parameters import inspect_parameters
from dpstats.transport.configuration import PolymorphicModel
from dpstats.workspace.layout import (
    algorithm_declarations_path,
    task_collection_directory,
)
from dpstats.workspace.layout import registry_directory_name
from dpstats.workspace.registries import read_selection

from api_reference import configured_base, format_search, object_route
from skipped_rows import reported_separately


# The file each results page reads its configuration provenance from, beside
# the page itself. A roster reaches a few hundred configurations, so it is
# fetched when a reader opens a Details popover rather than inlined.
PROVENANCE_NAME = "provenance.json"


def _family_base(cls: type[PolymorphicModel]) -> type[PolymorphicModel] | None:
    """The nearest ancestor of *cls* declared directly on PolymorphicModel.

    Generalizes ``configured_base`` to every configured family the platform
    ships (losses, search policies, ...), not only algorithms and transformations:
    that ancestor's own fields are structural to the family, never this
    class's own configuration.
    """
    for ancestor in cls.__mro__:
        if PolymorphicModel in ancestor.__bases__:
            return ancestor
    return None


def _component(value: PolymorphicModel, routes: dict[str, str]) -> dict:
    record = value.dump_config().model_dump(mode="json")
    cls = type(value)
    identifier = f"{cls.__module__}.{cls.__qualname__}"
    route = (
        object_route(identifier, routes)
        if any(
            identifier == name or identifier.startswith(name + ".") for name in routes
        )
        else None
    )
    if route is None:
        # Some API pages own a public re-export rather than the source module.
        for public_path, public_route in routes.items():
            module, _, name = public_path.rpartition(".")
            if name == cls.__name__ and getattr(import_module(module), name) is cls:
                route = public_route
                break
    children = []
    structural_fields: set[str] = set()
    for field in cls.model_fields:
        child = getattr(value, field)
        # Any configured object nested here -- not only an algorithm or
        # transformation -- gets its own named entry; it never serializes as a
        # mapping in the hyperparameter table.
        if isinstance(child, PolymorphicModel):
            children.append({"field": field, "component": _component(child, routes)})
            structural_fields.add(field)
        elif (
            isinstance(child, (tuple, list))
            and child
            and all(isinstance(item, PolymorphicModel) for item in child)
        ):
            children.extend(
                {"field": f"{field}[{index}]", "component": _component(item, routes)}
                for index, item in enumerate(child)
            )
            structural_fields.add(field)
    # Rows come from the same field introspection as the API reference's
    # hyperparameter table (never the raw configuration dict), so a
    # composition's own fields, being entirely structural, contribute none.
    base = configured_base(cls) or _family_base(cls)
    hyperparameters = []
    if base is not None:
        hyperparameters = [
            {
                "name": name,
                # A field left unset may be omitted from the stored
                # configuration; its value is then the instance's own.
                "value": repr(record["config"].get(name, getattr(value, name))),
                "default": (
                    "required" if parameter.required else repr(parameter.default)
                ),
                "search": format_search(parameter.search),
                "description": parameter.description,
            }
            for name, parameter in inspect_parameters(cls, base).items()
            if name not in structural_fields
        ]
    # The class's own opening paragraph, as the API reference summarises it; an
    # inherited docstring would describe the base class, not this component.
    own = inspect.cleandoc(cls.__dict__.get("__doc__") or "")
    return {
        "name": cls.__name__,
        "summary": " ".join(own.split("\n\n")[0].split()),
        "path": record["path"],
        "kind": (
            "Composition"
            if isinstance(value, CompositeAlgorithm)
            else (
                "Transformation"
                if isinstance(value, Transformation)
                else (
                    "Algorithm"
                    if isinstance(value, BaseAlgorithm)
                    else base.__name__ if base is not None else cls.__name__
                )
            )
        ),
        "docs": route.removesuffix(".md") + "/" if route else None,
        "hyperparameters": hyperparameters,
        "children": children,
    }


def _read_algorithm_declarations(path: Path) -> dict[str, dict[str, str]]:
    """Read this task's declarations as the release shipped them, unvalidated.

    The release build already validated the table; the site renders its
    bytes, the same way it renders ``release.json`` with ``json.loads``.

    Raises:
        ValueError: The header is missing a required column.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"uuid", "description", "provenance_ref"} - set(
            reader.fieldnames or ()
        )
        if missing:
            raise ValueError(
                f"Algorithm declarations are missing column(s) {sorted(missing)}: {path}."
            )
        return {
            row["uuid"]: {
                "description": row["description"],
                "provenance_ref": row["provenance_ref"],
            }
            for row in reader
        }


def provenance_payload(
    root: Path,
    task_id: str,
    routes: dict[str, str],
    *,
    algorithm_registries: list[str] | None = None,
    algorithm_ids: list[str] | None = None,
    strict: bool = True,
) -> dict:
    """Join by configuration identity, never by an algorithm's label."""
    task_root = task_collection_directory(root, task_id)
    if not (task_root / registry_directory_name("algorithm")).is_dir():
        return {}
    with reported_separately():
        registered = load_algorithms(
            read_selection(task_root, "algorithm", algorithm_registries).entries,
            task_id=task_id,
            strict=strict,
        )
    declarations_path = algorithm_declarations_path(task_root)
    declarations = (
        _read_algorithm_declarations(declarations_path)
        if declarations_path.is_file()
        else {}
    )
    payload = {
        uuid: {
            "component": _component(algorithm, routes),
            "configuration": algorithm.dump_config().model_dump(mode="json"),
        }
        for uuid, algorithm in registered.items()
        if algorithm_ids is None or uuid in algorithm_ids
    }
    for uuid, declaration in declarations.items():
        if uuid in payload:
            payload[uuid]["declaration"] = declaration
    return payload


def link_provenance(payload: dict, routes: dict[str, str]) -> None:
    """Attach current documentation links to already-prepared descriptions."""

    def link(component):
        path = component["path"].replace(":", ".")
        component["docs"] = (
            object_route(path, routes).removesuffix(".md") + "/"
            if any(path == name or path.startswith(name + ".") for name in routes)
            else None
        )
        for child in component["children"]:
            link(child["component"])

    for value in payload.values():
        link(value["component"])
