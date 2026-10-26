"""Read the installed library into the page records the renderers fill.

One direction only: this module imports the library and returns
:mod:`templates.model` records. Nothing here writes Markdown, and no renderer
imports a library class, so the filler registry can stand in for every value
on this side.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from html import escape
from importlib import import_module, util
from inspect import cleandoc, getattr_static, isfunction, signature
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from markdown import markdown
from griffe import Docstring, DocstringSectionKind, Parser
from pydantic import BaseModel

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
    PolymorphicModel,
    Transformation,
)
from dpstats.parameters import _field_symbol, inspect_parameters
from dpstats.task import Task
from dpstats import task as task_module
from dpstats import tasks as task_package
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metrics,
    list_built_in_metric_arguments,
)

from bibliography import CitedWork, cited_by, short_citation, source_text
from docstring_text import joined_block
from transformation_groups import load_transformation_groups
from profile_chart import (
    ALGORITHM_FAMILIES,
    DIAGNOSTIC_FAMILY,
    OTHER_FAMILY,
    algorithm_family,
)
from reader_terms import load_reader_terms
from site_copy import copy_html

from .blocks import code, display_math, math, math_name, reference_paragraphs
from .model import (
    ALGORITHM,
    TRANSFORMATION,
    Access,
    AlgorithmsIndex,
    ApiMethod,
    Component,
    Definition,
    Link,
    MetricSection,
    Metrics,
    ModelBlock,
    AlgorithmGroup,
    Parameter,
    TransformationGroup,
    TransformationRow,
    ReaderTerm,
    TransformationsIndex,
    Row,
    RosterLink,
    TaskOverview,
)

PACKAGE = "dpstats"

# The privacy interpretations in the order the library declares them, strongest
# to weakest. A table's chips and its class column read in this order, which is
# the order the privacy-semantics page explains them in.
CLASS_ORDER = tuple(member.value for member in PrivacyInterpretation)


def _by_strength(classes: Iterable[str]) -> tuple[str, ...]:
    """The classes present, in the library's own strongest-first order."""
    return tuple(name for name in CLASS_ORDER if name in set(classes))


FIELD_COLUMNS = ("Field", "Type", "Description")

# A Google-style docstring block opens at the docstring's own indentation with
# a capitalized name and a colon, and its body is indented under it.
_BLOCK = re.compile(r"^(?P<name>[A-Z][A-Za-z ]*):\s*$", re.M)
_ARGUMENT = re.compile(r"^(?P<name>\*{0,2}\w+)(?:\s*\([^)]*\))?:\s*(?P<text>.*)$")
# What a page sets around mathematics itself, so a declaration carries none.
_MATH_DELIMITERS = ("$", "\\(", "\\[")
# The array aliases a task declares its fields with, by the arguments each one
# resolves to. A resolved annotation is the ndarray shape the alias stands for,
# and a reader writes the alias, so a table shows the alias's own name.
_ARRAY_ALIASES = {
    name: get_args(getattr(task_module, name))
    for name in getattr(task_module, "__all__", ())
    if get_origin(getattr(task_module, name)) is Annotated
}


def task_symbol(task: Task) -> str:
    """The name a reader imports one built-in Task by.

    The keyed accessors name the Task value, not its id, so the Access block
    reads the exported symbol off the package rather than spelling a second
    naming rule.

    Args:
        task: The built-in Task.

    Returns:
        The exported symbol, or the task's id where nothing exports it.
    """
    for name in getattr(task_package, "__all__", ()):
        if getattr(task_package, name, None) is task:
            return name
    return task.id


def docstring_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Split one docstring into its opening prose and its named blocks.

    Args:
        text: The docstring, already dedented.

    Returns:
        The prose before the first named block, and each block's dedented body
        keyed by its name.

    Example:
        >>> docstring_blocks("Does a thing.\\n\\nReference:\\n    A paper.\\n")[1]
        {'Reference': 'A paper.'}
    """
    matches = list(_BLOCK.finditer(text))
    if not matches:
        return text.strip(), {}
    blocks: dict[str, str] = {}
    for position, match in enumerate(matches):
        end = (
            matches[position + 1].start() if position + 1 < len(matches) else len(text)
        )
        body = text[match.end() : end]
        lines = [line for line in body.splitlines() if line.strip()]
        if not lines:
            blocks[match["name"]] = ""
            continue
        indent = min(len(line) - len(line.lstrip()) for line in lines)
        blocks[match["name"]] = "\n".join(
            line[indent:] for line in body.splitlines()
        ).strip()
    return text[: matches[0].start()].strip(), blocks


def summary(text: str, limit: int) -> str:
    """The opening paragraph of one docstring, at most *limit* source lines."""
    opening = text.strip().split("\n\n")[0]
    return joined_block("\n".join(opening.splitlines()[:limit]))


def inline(text: str) -> str:
    """One declared description as the inner HTML of its own paragraph."""
    if not text:
        return ""
    rendered = markdown(
        " ".join(text.split()),
        extensions=["pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    return rendered.removeprefix("<p>").removesuffix("</p>")


def _alias_name(arguments: tuple[object, ...]) -> str:
    """The name of the array alias these resolved arguments open with, if any.

    A field may add its own metadata to an alias, so the alias is what the
    resolved arguments start with rather than all of them.
    """
    for name, alias in _ARRAY_ALIASES.items():
        if arguments[: len(alias)] == alias:
            return name
    return ""


def type_name(annotation: object) -> str:
    """One annotation as a reader writes it."""
    if isinstance(annotation, str):
        return annotation
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return _alias_name(arguments) or type_name(arguments[0])
    if origin in (Union, UnionType):
        return " | ".join(type_name(argument) for argument in arguments)
    if origin is not None:
        inner = ", ".join(type_name(argument) for argument in arguments)
        return f"{type_name(origin)}[{inner}]"
    return getattr(annotation, "__name__", str(annotation).removeprefix("typing."))


def type_html(annotation: object) -> str:
    """Render an annotation with API cross-references for library-owned types."""

    def reference(identifier: str, label: str) -> str:
        return f'<autoref identifier="{escape(identifier, quote=True)}">{escape(label)}</autoref>'

    def contents(value: object) -> str:
        origin = get_origin(value)
        arguments = get_args(value)
        if origin is Annotated:
            alias = _alias_name(arguments)
            return (
                reference(f"{PACKAGE}.task.{alias}", alias)
                if alias
                else contents(arguments[0])
            )
        if origin in (Union, UnionType):
            return " | ".join(contents(argument) for argument in arguments)
        if origin is Literal:
            return escape(type_name(value))
        if origin is not None:
            return (
                contents(origin)
                + "["
                + ", ".join(contents(argument) for argument in arguments)
                + "]"
            )
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", "")
        if module.startswith(PACKAGE + ".") and name:
            return reference(f"{module}.{name}", type_name(value))
        return escape(type_name(value))

    return "<code>" + contents(annotation) + "</code>"


def _declared_math(owner: str, kind: str, declared: str) -> str:
    """One declared expression, refusing a declaration that sets its own math.

    Raises:
        ValueError: The declaration carries a math delimiter.
    """
    for delimiter in _MATH_DELIMITERS:
        if delimiter in declared:
            raise ValueError(
                f"{owner}: {kind} carries the math delimiter {delimiter!r}; the "
                "page sets the mathematics around the declared expression."
            )
    return declared


def choices(annotation: object) -> str:
    """The declared choice set of a literal-typed field, as a reader reads it."""
    if get_origin(annotation) is not Literal:
        return ""
    return "one of: " + ", ".join(str(value) for value in get_args(annotation))


def constraints(symbol: str, metadata: Sequence[object]) -> str:
    """One field's declared domain as inline mathematics, set in *symbol*.

    The symbol is the field's declared one where it has one, and its own name
    otherwise.
    """
    lower = upper = ""
    extra: list[str] = []
    for item in metadata:
        for key, text in (("gt", "{value} < "), ("ge", "{value} \\le ")):
            value = getattr(item, key, None)
            if value is not None:
                lower = text.format(value=value)
        for key, text in (("lt", " < {value}"), ("le", " \\le {value}")):
            value = getattr(item, key, None)
            if value is not None:
                upper = text.format(value=value)
        multiple = getattr(item, "multiple_of", None)
        if multiple is not None:
            extra.append(f"{symbol} \\bmod {multiple} = 0")
        shortest = getattr(item, "min_length", None)
        longest = getattr(item, "max_length", None)
        if shortest is not None or longest is not None:
            low = f"{shortest} \\le " if shortest is not None else ""
            high = f" \\le {longest}" if longest is not None else ""
            extra.append(f"{low}\\lvert {symbol} \\rvert{high}")
    stated = [f"{lower}{symbol}{upper}"] if lower or upper else []
    return ", ".join(math(item) for item in stated + extra)


def _resolved_annotations(model: type[BaseModel]) -> dict[str, object]:
    """Every field's annotation as the interpreter resolves it.

    A model declaring postponed annotations carries the source's own text, and
    a reader of the page reads types rather than the text a declaration was
    written as, so the model's own module resolves them once.

    Raises:
        ValueError: An annotation does not resolve in the model's module.
    """
    try:
        return dict(get_type_hints(model, include_extras=True))
    except NameError as error:
        raise ValueError(
            f"{model.__module__}.{model.__name__}: an annotation does not "
            f"resolve in its own module ({error})."
        ) from error


def _field_rows(model: type[BaseModel]) -> tuple[Row, ...]:
    resolved = _resolved_annotations(model)
    rows: list[Row] = []
    for name, field in model.model_fields.items():
        annotation = resolved.get(name, field.annotation)
        optional = "" if field.is_required() else " (optional)"
        declared = _field_symbol(field)
        symbol = (
            math_name(name)
            if declared is None
            else _declared_math(f"{model.__name__}.{name}", "symbol", declared)
        )
        rows.append(
            Row(
                name=name,
                type=type_html(annotation) + optional,
                constraints=constraints(symbol, field.metadata) or choices(annotation),
                description=inline(field.description or ""),
            )
        )
    return tuple(rows)


def _model_block(
    title: str, task: Task, model: type[BaseModel], columns: tuple[str, ...]
) -> ModelBlock:
    # Readers import from the task's package, never its private defining module.
    module = f"{PACKAGE}.tasks.{task.id}"
    if getattr(import_module(module), model.__name__, None) is not model:
        raise ValueError(f"{module} does not export {model.__name__}.")
    return ModelBlock(
        title=title,
        access=Access(f"from {module} import {model.__name__}"),
        summary="",
        columns=columns,
        rows=_field_rows(model),
    )


def definition(task: Task, notes: str = "") -> Definition:
    """The task's declared Input, Metadata and Output."""
    return Definition(
        title=f"{task.name}: definition",
        models=(
            _model_block("Metadata", task, task.Metadata, FIELD_COLUMNS),
            _model_block("Input", task, task.Input, FIELD_COLUMNS),
            _model_block("Output", task, task.Output, FIELD_COLUMNS),
        ),
        notes=notes,
    )


def metric_summaries(task_id: str) -> dict[str, str]:
    """Each built-in Metric's one-sentence definition, its calculation
    docstring's summary line, as inner HTML; none for a task the library does
    not ship."""
    task = BUILT_IN_TASKS.get(task_id)
    if task is None:
        return {}
    return {
        key: inline(summary(cleandoc(metric.calculation.__doc__ or ""), 1))
        for key, metric in list_built_in_metrics(task).items()
    }


def _formula(key: str, declared: str | None) -> str:
    """One metric's declaration as the display mathematics the page sets."""
    if declared is None:
        return ""
    return display_math(_declared_math(key, "formula", declared))


def metrics(
    task: Task, baselines: Mapping[str, Sequence[Link]] | None = None
) -> Metrics:
    """The task's metric registry: the mapping, the summary, then each metric.

    *baselines* links each Metric's offered baselines into the page's own
    baselines table, where the build renders one.
    """
    sections = []
    registry = list_built_in_metric_arguments(task)
    metric_argument_descriptions: dict[str, str] = {
        name: inline(argument.description) for name, argument in registry.items()
    }
    for key, metric in list_built_in_metrics(task).items():
        docstring = cleandoc(metric.calculation.__doc__ or "")
        _, blocks = docstring_blocks(docstring)
        notes = joined_block(blocks.get("Notes", ""))
        notes = re.sub(r":math:`([^`]+)`", lambda match: "$" + match[1] + "$", notes)
        name = key.replace("_", " ").capitalize()
        needs = metric.required_names
        for section in Docstring(docstring).parse(Parser.google, warnings=False):
            if section.kind != DocstringSectionKind.parameters:
                continue
            for parameter in section.value:
                if (
                    parameter.name not in needs
                    or parameter.name in registry
                    or not parameter.description
                ):
                    continue
                description = inline(
                    re.sub(r":math:`([^`]+)`", r"$\1$", parameter.description)
                )
                previous = metric_argument_descriptions.get(parameter.name)
                if previous is not None and previous != description:
                    raise ValueError(
                        f"{task.id}: conflicting descriptions for {parameter.name}"
                    )
                metric_argument_descriptions[parameter.name] = description
        sections.append(
            MetricSection(
                name=key,
                key=key,
                anchor=name.lower().replace(" ", "-"),
                docstring=docstring,
                notes=notes,
                formula=_formula(key, metric.formula),
                needs=needs,
                baselines=tuple((baselines or {}).get(key, ())),
            )
        )
    namespace = "".join(word[0] for word in task.id.split("_"))
    return Metrics(
        title=f"{task.name}: metrics",
        access=Access(
            import_line=f"from {PACKAGE}.tasks import {task.id} as {namespace}",
            accessor="\n".join(
                f"{namespace}.metrics.{section.key}" for section in sections
            ),
        ),
        metrics=tuple(sections),
        metric_argument_descriptions=tuple(metric_argument_descriptions.items()),
        arguments_intro=copy_html("task", "metric_arguments_intro"),
    )


def metric_lines(task: Task) -> tuple[tuple[str, str, str], ...]:
    """Each metric's display name, its anchor, and its opening line."""
    return tuple(
        (section.name, section.anchor, summary(section.docstring, 1))
        for section in metrics(task).metrics
    )


def task_overview(
    task: Task,
    quick_links: Sequence[Link],
    problem: str,
    metric_links: Sequence[tuple[Link, str]],
) -> TaskOverview:
    """The page a reader lands on for a task."""
    return TaskOverview(
        name=task.name,
        description=estimand(task),
        access=Access(
            import_line=f"from {PACKAGE}.tasks import BUILT_IN_TASKS",
            accessor=f'BUILT_IN_TASKS["{task.id}"]',
        ),
        quick_links=tuple(quick_links),
        problem=problem,
        metrics=tuple(metric_links),
    )


def _cited(component: type) -> list[CitedWork]:
    """The works the class's own ``Reference:`` block cites, in order."""
    return cited_by(cleandoc(vars(component).get("__doc__") or ""))


def reference_block(component: type) -> str:
    """The class's own ``Reference:`` block, one hanging-indent paragraph per work."""
    return reference_paragraphs(_cited(component), locators=False)


def citation(component: type) -> str:
    """The in-text citation a table shows: the first work the class cites."""
    works = _cited(component)
    return short_citation(works[0].key) if works else ""


def class_help() -> ReaderTerm | None:
    """The maintainer's definition of a privacy interpretation, where written."""
    entry = load_reader_terms().get("privacy_interpretation")
    if entry is None:
        return None
    return ReaderTerm("privacy_interpretation", entry["text"], entry.get("link", ""))


def algorithms_index(
    task: Task,
    page_url: Callable[[str], str],
) -> AlgorithmsIndex:
    """Every Algorithm written against this task, grouped by mechanism family.

    The families are the ones the results charts colour by, nonprivate and
    diagnostic last. Within a family, rows run strongest class first, then by
    name. The catalogue key is the class name by construction, so the name is
    the key.
    """
    order = (*ALGORITHM_FAMILIES, OTHER_FAMILY, DIAGNOSTIC_FAMILY)
    grouped: dict[str, list[tuple[Link, str, str, str]]] = {f: [] for f in order}
    classes: list[str] = []
    for name in sorted(list_built_in_algorithms(task), key=str.lower):
        algorithm = list_built_in_algorithms(task)[name]
        declared = _privacy_interpretation(algorithm)
        if declared not in classes:
            classes.append(declared)
        grouped[algorithm_family(name)].append(
            (
                Link(name, page_url(name)),
                opening_line(algorithm),
                declared,
                citation(algorithm),
            )
        )
    groups = [
        AlgorithmGroup(
            title=family,
            rows=tuple(sorted(rows, key=lambda row: CLASS_ORDER.index(row[2]))),
        )
        for family, rows in grouped.items()
        if rows
    ]
    if not headed([group.title for group in groups], OTHER_FAMILY):
        rows = [row for group in groups for row in group.rows]
        groups = [
            AlgorithmGroup(
                title="",
                rows=tuple(sorted(rows, key=lambda row: CLASS_ORDER.index(row[2]))),
            )
        ]
    return AlgorithmsIndex(
        title=f"{task.name}: native algorithms",
        classes=_by_strength(classes),
        groups=tuple(groups),
        class_help=class_help(),
    )


def headed(titles: Sequence[str], other: str) -> bool:
    """Whether a page's groups earn headings.

    Only two or more named groups do, and then the catch-all is headed beside
    them; otherwise the page is one unheaded table rather than a lone or
    lone-"Other" heading.
    """
    return len([title for title in titles if title != other]) > 1


def opening_line(component: type) -> str:
    """The class docstring's summary line, as a table's description cell."""
    return inline(summary(cleandoc(vars(component).get("__doc__") or ""), 1))


# Which group each task-preserving transformation is listed under on its task's
# Transformations page. A transformation handing the problem to another task is a
# conversion by construction; one not named here lands in the last group
# rather than being guessed at. The groups' wording is the maintainer's file.
TASK_CONVERSION = "task_conversion"
OTHER_TRANSFORMATIONS = "other"
TRANSFORMATION_GROUP_BY_CLASS = {
    "ApproxBounds": "bounds",
    "ApproxBoundsAll": "bounds",
    "DataBounds": "bounds",
    "Standardize": "scaling_and_projection",
    "DPWhitening": "scaling_and_projection",
    "LargestOneHotFWLDemean": "scaling_and_projection",
    "DPKendall": "feature_selection",
    "ExactKendallSelection": "feature_selection",
    "SubLasso": "feature_selection",
    "ToZCDP": OTHER_TRANSFORMATIONS,
}


def transformation_group(
    task: Task, name: str, transformation: type[Transformation]
) -> str:
    """The group one exported transformation is listed under on its task's page."""
    if transformation.INNER_TASK.id != task.id:
        return TASK_CONVERSION
    return TRANSFORMATION_GROUP_BY_CLASS.get(name, OTHER_TRANSFORMATIONS)


def transformations_index(
    task: Task,
    exported: Mapping[str, type[Transformation]],
    page_url: Callable[[str], str],
    task_url: Callable[[str], str],
) -> TransformationsIndex:
    """Every Transformation this task exports, grouped by what it does.

    Within a group, rows run strongest class first, so a diagnostic transformation
    closes its group, then by name.
    """
    wording = load_transformation_groups()
    grouped: dict[str, list[TransformationRow]] = {group: [] for group in wording}
    classes: list[str] = []
    for name in sorted(exported, key=str.lower):
        transformation = exported[name]
        declared = _privacy_interpretation(transformation)
        if declared not in classes:
            classes.append(declared)
        grouped[transformation_group(task, name, transformation)].append(
            TransformationRow(
                name=Link(name, page_url(name)),
                description=opening_line(transformation),
                privacy_interpretation=declared,
                inner=Link(
                    transformation.INNER_TASK.name,
                    task_url(transformation.INNER_TASK.id),
                ),
                reference=citation(transformation),
            )
        )
    present = [group for group, rows in grouped.items() if rows]
    shown = headed(present, OTHER_TRANSFORMATIONS)
    if not shown and len(present) > 1:
        # One unheaded table: conversions keep their Hands off to column, so
        # the page stays one table only when every row shares a column set.
        conversions = grouped[TASK_CONVERSION]
        rest = [
            row
            for group in present
            if group != TASK_CONVERSION
            for row in grouped[group]
        ]
        grouped = {TASK_CONVERSION: conversions, OTHER_TRANSFORMATIONS: rest}
        present = [group for group, rows in grouped.items() if rows]
    groups = tuple(
        TransformationGroup(
            title=wording[group].title if shown else "",
            summary=wording[group].text if shown else "",
            rows=tuple(
                sorted(
                    grouped[group],
                    key=lambda row: CLASS_ORDER.index(row.privacy_interpretation),
                )
            ),
            converts=group == TASK_CONVERSION,
        )
        for group in present
    )
    return TransformationsIndex(
        title=f"{task.name}: transformations",
        groups=groups,
        classes=_by_strength(classes),
        class_help=class_help(),
    )


def _privacy_interpretation(component: type) -> str:
    """The class this component declares, read as the library computes it.

    A component that does not override ``privacy_interpretation`` answers
    from its declared ``PRIVACY_INTERPRETATION`` class attribute, with no
    instance required. A property reading the component's own fields — one
    that delegates to another component built from them — is read on the
    configuration the class declares by default, which is the same answer
    every page of that class shows. A class whose configuration has no
    default cannot be read this way and stops the build by name rather than
    being guessed at.

    Raises:
        TypeError: The class declares no ``privacy_interpretation`` property.
    """
    declared = getattr_static(component, "privacy_interpretation", None)
    if not isinstance(declared, property) or declared.fget is None:
        raise TypeError(
            f"{component.__name__} declares no privacy_interpretation property."
        )
    base_fgets = {
        getattr_static(BaseAlgorithm, "privacy_interpretation").fget,
        getattr_static(Transformation, "privacy_interpretation").fget,
    }
    if declared.fget in base_fgets:
        return component.PRIVACY_INTERPRETATION.value
    try:
        return declared.fget(None).value
    except AttributeError as error:
        if error.name not in component.model_fields:
            raise
    return component().privacy_interpretation.value


def _configuration_rows(component: type[PolymorphicModel]) -> tuple[Row, ...]:
    """Each configurable parameter's type and default, for the class signature."""
    from api_reference import configured_base

    base = configured_base(component)
    parameters = inspect_parameters(component, base) if base is not None else {}
    rows: list[Row] = []
    for name, parameter in parameters.items():
        field = component.model_fields[name]
        if parameter.required:
            default = "Required"
        elif field.default_factory is not None:
            default = "Factory: " + code(field.default_factory.__qualname__)
        else:
            default = code(repr(parameter.default))
        rows.append(
            Row(name=name, type=type_html(parameter.annotation), default=default)
        )
    return tuple(rows)


def _number(value: object) -> str:
    """One bound as a reader writes it: ``0`` rather than ``0.0``."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _series(items: Sequence[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def declared_type(annotation: object, metadata: Sequence[object]) -> str:
    """One parameter's type with its declared domain folded in, as HTML.

    The domain reads in words and comparison signs rather than as a separate
    constraint: ``float > 0``, ``float in (0, 1)``, ``int ≥ 1``, and
    ``positive int`` for an integer only bounded strictly above zero. A
    literal reads as its choices, ``'trim' or 'winsorize'``, and an optional
    type ends ``or None``.
    """
    arguments = get_args(annotation)
    optional = get_origin(annotation) in (Union, UnionType) and type(None) in arguments
    if optional:
        rest = tuple(argument for argument in arguments if argument is not type(None))
        annotation = rest[0] if len(rest) == 1 else Union[rest]  # noqa: UP007
    if get_origin(annotation) is Literal:
        shown = _series([escape(repr(value)) for value in get_args(annotation)])
        return shown + (" or None" if optional else "")
    base = type_html(annotation).removeprefix("<code>").removesuffix("</code>")
    lower: tuple[str, object] | None = None
    upper: tuple[str, object] | None = None
    extra: list[str] = []
    for item in metadata:
        for key, sign in (("gt", ">"), ("ge", "≥")):
            if getattr(item, key, None) is not None:
                lower = (sign, getattr(item, key))
        for key, sign in (("lt", "<"), ("le", "≤")):
            if getattr(item, key, None) is not None:
                upper = (sign, getattr(item, key))
        if getattr(item, "multiple_of", None) is not None:
            extra.append(f"multiple of {_number(item.multiple_of)}")
        shortest = getattr(item, "min_length", None)
        longest = getattr(item, "max_length", None)
        if shortest is not None and longest is not None:
            extra.append(f"length in [{shortest}, {longest}]")
        elif shortest is not None:
            extra.append(f"length ≥ {shortest}")
        elif longest is not None:
            extra.append(f"length ≤ {longest}")
    if lower and upper:
        opening = "(" if lower[0] == ">" else "["
        closing = ")" if upper[0] == "<" else "]"
        shown = f"{base} in {opening}{_number(lower[1])}, {_number(upper[1])}{closing}"
    elif lower and base == "int" and lower == (">", 0):
        shown = "positive int"
    elif lower:
        shown = f"{base} {escape(lower[0])} {_number(lower[1])}"
    elif upper:
        shown = f"{base} {escape(upper[0])} {_number(upper[1])}"
    else:
        shown = base
    shown = ", ".join([shown, *extra])
    return shown + (" or None" if optional else "")


# One ``Parametrization notes:`` entry: ``- ``name``: what it says``.
_PARAMETER_NOTE = re.compile(r"^- ``(?P<name>[A-Za-z_]\w*)``:\s*(?P<text>.+)$", re.S)


def parameter_notes(block: str, names: Iterable[str]) -> tuple[dict[str, str], str]:
    """Each parameter's own note, and whatever the block says outside them.

    Args:
        block: The docstring's ``Parametrization notes:`` block.
        names: The parameters the class declares.

    Returns:
        The note text keyed by parameter name, and the rest of the block as
        Markdown, empty where every entry names a declared parameter.
    """
    declared = set(names)
    notes: dict[str, str] = {}
    rest: list[str] = []
    for item in joined_block(block).split("\n"):
        match = _PARAMETER_NOTE.match(item.strip())
        if match is not None and match["name"] in declared:
            notes[match["name"]] = match["text"].strip()
        else:
            rest.append(item)
    # A blank line still separates the paragraphs and lists that remain.
    return notes, re.sub(r"\n{3,}", "\n\n", "\n".join(rest)).strip()


def _parameters(
    component: type[PolymorphicModel], notes: Mapping[str, str]
) -> tuple[Parameter, ...]:
    """Every configurable parameter, its declared type and its own note."""
    from api_reference import configured_base

    base = configured_base(component)
    declared = inspect_parameters(component, base) if base is not None else {}
    parameters: list[Parameter] = []
    for name, parameter in declared.items():
        field = component.model_fields[name]
        if parameter.required:
            default = ""
        elif field.default_factory is not None:
            default = escape(field.default_factory.__qualname__) + "()"
        else:
            default = escape(repr(parameter.default))
        parameters.append(
            Parameter(
                name=name,
                type=declared_type(parameter.annotation, field.metadata),
                default=default,
                description=inline(parameter.description or ""),
                note=inline(notes.get(name, "")),
            )
        )
    return tuple(parameters)


def _requires(block: str) -> tuple[str, ...]:
    """The ``Requires:`` block's bullets, each as inline HTML."""
    return tuple(
        inline(item.removeprefix("- "))
        for item in joined_block(block).split("\n")
        if item.strip()
    )


# The opening sentence that names the work a class implements or adapts, which
# the source line states instead: "An implementation of Algorithm 2 from Wang
# (2018)." It ends at the first full stop that a capital or the paragraph's
# end follows, so "et al. (2019)" does not end it.
_IMPLEMENTATION_SENTENCE = re.compile(
    r"^An (?:implementation|adaptation) of\b.*?\(\d{4}\).*?\.(?=\s+[A-Z]|\s*$)",
    re.S,
)


def opening_prose(prose: str, cited: bool) -> tuple[str, str]:
    """The summary line, and the prose after it, less the implementation sentence.

    The sentence goes only from the page, and only where the class cites the
    work it names: the source line then says the same.
    """
    paragraphs = prose.split("\n\n")
    summary_text = " ".join(paragraphs[0].split())
    rest = paragraphs[1:]
    if cited and rest:
        rest[0] = _IMPLEMENTATION_SENTENCE.sub("", rest[0].strip()).strip()
        rest = [paragraph for paragraph in rest if paragraph]
    return summary_text, "\n\n".join(rest)


def _api(
    component: type[PolymorphicModel],
    rows: Sequence[Row],
    methods: Sequence[str],
) -> tuple[str, tuple[ApiMethod, ...]]:
    """The class signature and the public methods its family declares."""
    if rows:
        arguments = "\n".join(
            f"    {row.name}: {_plain(row.type)} = {_plain(row.default)},"
            for row in rows
        )
        declaration = f"class {component.__name__}(\n{arguments}\n)"
    else:
        declaration = f"class {component.__name__}()"
    rendered: list[ApiMethod] = []
    for name in methods:
        member = getattr(component, name, None)
        if member is None or not isfunction(member):
            continue
        text = cleandoc(member.__doc__ or "")
        prose, blocks = docstring_blocks(text)
        annotations = get_type_hints(member, include_extras=True)
        rendered.append(
            ApiMethod(
                name=name,
                signature=f"{name}{_signature(member)}",
                summary=summary(prose, 3),
                parameters=_arguments(blocks.get("Args", ""), annotations),
            )
        )
    return declaration, tuple(rendered)


def _plain(cell: str) -> str:
    """One table cell's text, without the markup a table needs."""
    return re.sub(r"<[^>]+>", "", cell).replace("&amp;", "&").strip()


def _signature(member: Any) -> str:
    """One method's call signature, as a reader writes it.

    Deferred annotations reach here as strings, so each one is unquoted rather
    than rendered with the quotation marks the interpreter kept.
    """
    declared = signature(member)
    parameters: list[str] = []
    for name, parameter in declared.parameters.items():
        if name == "self":
            continue
        rendered = name
        if parameter.annotation is not declared.empty:
            rendered += f": {_annotation(parameter.annotation)}"
        if parameter.default is not declared.empty:
            rendered += f" = {parameter.default!r}"
        parameters.append(rendered)
    returns = (
        ""
        if declared.return_annotation is declared.empty
        else f" -> {_annotation(declared.return_annotation)}"
    )
    return "(" + ", ".join(parameters) + ")" + returns


def _annotation(annotation: object) -> str:
    """One annotation, whether the interpreter kept it deferred or evaluated."""
    if isinstance(annotation, str):
        return annotation.strip("'\"")
    return type_name(annotation)


def _arguments(block: str, annotations: Mapping[str, object]) -> tuple[Row, ...]:
    rows: list[Row] = []
    name = ""
    described: dict[str, str] = {}
    for line in block.splitlines():
        match = _ARGUMENT.match(line.strip()) if not line.startswith(" ") else None
        if match is not None:
            name = match["name"]
            described[name] = match["text"].strip()
        elif name:
            described[name] = (described[name] + " " + line.strip()).strip()
    for name, text in described.items():
        annotation = annotations.get(name)
        rows.append(
            Row(
                name=name,
                type=type_html(annotation) if annotation is not None else "",
                description=inline(text),
            )
        )
    return tuple(rows)


def component_page(
    task: Task,
    name: str,
    component: type[PolymorphicModel],
    *,
    task_link: Link,
    semantics: Link,
    inner_link: Link | None = None,
    roster: Sequence[RosterLink] = (),
) -> Component:
    """The one page record for one Algorithm or one Transformation.

    Both kinds are the same page. The kind decides the header line, the
    parameters the table reads, the usage snippet and the
    public methods the API section lists; every other section is built here
    once, from the same docstring blocks.

    Args:
        task: The task whose space publishes this page.
        name: The class name, which is also an Algorithm's built-in key.
        component: The Algorithm or Transformation class this page documents.
        task_link: An Algorithm's native task, or a Transformation's outer task.
        semantics: The link to the library privacy-semantics page.
        inner_link: A Transformation's inner task; an Algorithm passes none.
        roster: The task Rosters holding configured algorithms that use it,
            its own task's first.

    Returns:
        The record the component-page renderer fills its sections from.

    Raises:
        ValueError: A Transformation was passed without its inner-task link.
    """
    namespace = "".join(word[0] for word in task.id.split("_"))
    task_import = f"from {PACKAGE}.tasks import {task.id} as {namespace}"
    transformation = issubclass(component, Transformation)
    if transformation and inner_link is None:
        raise ValueError(
            f"{name} is a Transformation, so its page needs an inner task."
        )
    if transformation and inner_link is not None:
        kind = TRANSFORMATION
        task_line = (
            f"[{task_link.text}]({task_link.url}) → "
            f"[{inner_link.text}]({inner_link.url})"
        )
        usage = _usage(
            task,
            [
                f"import {PACKAGE} as dp",
                task_import,
            ],
            "algorithm = dp.CompositeAlgorithm(\n"
            f"    transformations=({_construction(component, f'{namespace}.transformations.{name}')},),\n"
            "    algorithm=inner,\n"
            ")",
        )
        methods: Sequence[str] = ("prepare", "restore")
    else:
        kind = ALGORITHM
        task_line = f"[{task_link.text}]({task_link.url})"
        usage = _usage(
            task,
            [task_import],
            f"algorithm = {_construction(component, f'{namespace}.algorithms.{name}')}",
        )
        methods = ("run", "applies_to")

    docstring = cleandoc(vars(component).get("__doc__") or "")
    prose, blocks = docstring_blocks(docstring)
    cited = _cited(component)
    opening, description = opening_prose(prose, bool(cited))
    for heading, body in blocks.items():
        if heading in {
            "Requires",
            "Parametrization notes",
            "Implementation notes",
            "Reference",
        }:
            continue
        if heading == "Attributes":
            body = "\n".join(
                f"- {code(row.name)}: {row.description}" for row in _arguments(body, {})
            )
        elif heading == "Examples":
            for section in Docstring(docstring).parse(Parser.google, warnings=False):
                if section.kind == DocstringSectionKind.examples:
                    body = "\n\n".join(
                        (
                            f"```pycon\n{text}\n```"
                            if kind == DocstringSectionKind.examples
                            else text
                        )
                        for kind, text in section.value
                    )
        description += f"\n\n### {heading}\n\n{body}"
    implementation = joined_block(blocks.get("Implementation notes", ""))
    configuration = _configuration_rows(component)
    declaration, api_methods = _api(component, configuration, methods)
    notes, other_notes = parameter_notes(
        blocks.get("Parametrization notes", ""), (row.name for row in configuration)
    )
    return Component(
        name=name,
        kind=kind,
        task=task_line,
        summary=opening,
        description=description.strip(),
        privacy_interpretation=_privacy_interpretation(component),
        semantics_link=semantics,
        usage=usage,
        api_signature=declaration,
        api_methods=api_methods,
        source=source_text(cited),
        reference=reference_block(component),
        requires=_requires(blocks.get("Requires", "")),
        parameters=_parameters(component, notes),
        roster=tuple(roster),
        parametrization_notes=other_notes,
        implementation_notes=implementation,
    )


def _construction(component: type[PolymorphicModel], name: str) -> str:
    """One constructor call, naming the parameters the class requires."""
    required = [
        row.name for row in _configuration_rows(component) if row.default == "Required"
    ]
    return f"{name}(" + ", ".join(f"{field}=..." for field in required) + ")"


def _usage(task: Task, imports: Sequence[str], construction: str) -> str:
    """Import, construct, run, over an instance the task's producer draws.

    A task with no simulated-instance producer has nothing to draw from, so
    the snippet names the metadata and input the caller already holds.
    """
    producers = simulated_producers(task)
    lines = list(imports)
    draw = ""
    if producers:
        producer = producers[0].name
        lines.insert(0, f"from {PACKAGE}.simulations.{task.id} import {producer}")
        draw = f"metadata, input, _ = {producer}(seed=0)\n"
    return (
        "\n".join(lines)
        + "\n\n"
        + draw
        + construction
        + "\noutput = algorithm.run(metadata, input, budget, seed=0)"
    )


def simulated_producers(task: Task) -> tuple[Row, ...]:
    """This task's public simulated-instance producers, as reference rows."""
    module_name = f"{PACKAGE}.simulations.{task.id}"
    if util.find_spec(module_name) is None:
        return ()
    module = import_module(module_name)
    return tuple(
        Row(
            name=name,
            type=", ".join(
                code(parameter)
                for parameter in signature(getattr(module, name)).parameters
            ),
            description=inline(
                summary(
                    docstring_blocks(cleandoc(getattr(module, name).__doc__ or ""))[0],
                    1,
                )
            ),
        )
        for name, value in vars(module).items()
        if not name.startswith("_")
        and isfunction(value)
        and value.__module__ == module.__name__
    )


def estimand(task: Task) -> str:
    """The task's declared estimation problem as a plain-text table cell.

    The description is rendered in full. Math delimiters belong in the
    authored problem statement, where the renderer can interpret them.

    Raises:
        ValueError: The declared description cannot be rendered as a cell.
    """
    text = " ".join(task.description.split())
    for delimiter in _MATH_DELIMITERS:
        if delimiter in text:
            raise ValueError(
                f"{task.id}: description carries the math delimiter {delimiter!r}; "
                "the task description is plain text and the mathematics belongs "
                "in the authored problem statement."
            )
    return escape(text, quote=False)
