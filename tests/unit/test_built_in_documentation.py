"""Documentation sources the generated pages read, over every built-in object.

Each built-in algorithm, transformation and tuner class carries the named
docstring blocks the reference pages read: a
``Requires:`` block, present only when the class refuses some public problem or
budget at run time, listing each such requirement as a ``- `` bullet and
nothing every input satisfies or the configuration settles at construction,
and, for a class that realizes a published source, a
``Reference:`` block of ``- key: locator.`` bullets, each a key of
``dpstats/references.bib`` and the algorithm, section or result of
that work the class follows, and its opening paragraph starts with the first
work's in-text citation, as "An implementation of ..." or "An adaptation of
...". An opening that states its privacy guarantee ("It is an approximate-DP
mechanism.") states it last, and no opening sentence restates a ``Requires:``
bounds, budget or intercept condition. Every bibliography entry is cited and
carries a DOI or an arXiv identifier. Every "Authors (year)" a class names is
the in-text citation of a work its ``Reference:`` block lists, and no cited
work is called "the source". A class with no source paper carries no such block and says
nothing in its place. A present ``Parametrization notes:`` block must name every
``parameter_field`` hyperparameter the class declares without restating its
default value, and a class that declares one carries the block. No
``Implementation notes:`` bullet merely lists the metadata fields read. Each built-in
task's Metadata, Input, Output and
loss-family classes describe every field they declare, and each built-in
Metric leads with a one-line description and carries its formula and its
dependencies in the body the metrics page renders whole, and a short label
the results pages print in its place.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import pkgutil
import inspect
import re
from typing import Any, ClassVar

import pytest
from griffe import Docstring, DocstringSectionKind, Parser
from pydantic import BaseModel, Field

from dpstats.algorithm import Transformation
from dpstats.metric import RESERVED_REQUIREMENTS, Metric
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metrics,
    list_built_in_metric_arguments,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss
from dpstats.transport.configuration import PolymorphicModel
from tests.docstring_blocks import (
    IMPLEMENTATION_NOTES_HEADER,
    PARAMETRIZATION_NOTES_HEADER,
    REFERENCE_HEADER,
    REQUIRES_HEADER,
    named_block,
    named_block_shape_violations,
)

# ``tests.docstring_blocks`` puts the site package on the path.
from bibliography import CitedWork, cited_works, load, short_citation  # noqa: E402

# The checks each documented built-in object must clear.
REFERENCE_SHAPE = "reference-shape"
OPENING_CITATION = "opening-citation"
GUARANTEE_LAST = "guarantee-last"
REQUIRES_RESTATED = "requires-restated"
REQUIRES = "requires"
DESCRIPTIONS = "descriptions"
METRIC_SUMMARY = "metric-summary"
METRIC_PARAMETERS = "metric-parameters"
METRIC_ARGUMENT_REGISTRATION = "metric-argument-registration"
METRIC_NOTES_SECTION = "metric-notes-section"
METRIC_FORMULA_DELIMITERS = "metric-formula-delimiters"
METRIC_LABEL = "metric-label"
MATH_ROLE_PLACEMENT = "math-role-placement"
BLOCK_SHAPE = "block-shape"
HYPERPARAMETER_FIELD = "hyperparameter-field"
PARAMETRIZATION_COVERAGE = "parametrization-coverage"
DEFAULT_VALUE_RESTATED = "default-value-restated"
IMPLEMENTATION_NOTES = "implementation-notes"
IN_TEXT_CITATION = "in-text-citation"
TASK_DESCRIPTION = "task-description"

# The built-in classes that realize no privacy interpretation from an empty
# shell: an abstract family base is not a released mechanism, and a tuner
# derives its class from its configured candidate and selection cost, so it
# has none until configured. Any other class resolving to none means the
# resolution broke, not that the class is exempt.
_NO_PRIVACY_INTERPRETATION: frozenset[str] = frozenset(
    {
        "dpstats.dp_tuning.base.DPHyperparameterTuned",
        "dpstats.dp_tuning.exhaustive.DPExhaustiveCompose",
        "dpstats.dp_tuning.liu_talwar.DPSelectionLiuTalwar",
        "dpstats.dp_tuning.papernot_steinke.DPSelectionPapernotSteinkePoisson",
        "dpstats.dp_tuning.papernot_steinke.DPSelectionPapernotSteinkeTNB",
    }
)

# Where in its source a class's method sits: a numbered or lettered algorithm,
# section, appendix, result or equation, or a named listing of a code work,
# named in the docstring.
_SOURCE_LOCATOR = re.compile(
    r"(?:\b(?:Algorithms?|Sections?|Appendix|Theorems?|Definitions?|Proposition"
    r"|Corollary|Lemma|Equations?|Listing)|\bEqs?\.)"
    r"\s+(?:\(?\d[\d.]*\)?|[A-Z](?:\.\d+)*\b)"
    r"|\bListing ``[^`\s]+``"
)

# Built-in classes whose source has no separate algorithm to locate; the
# prose names the baseline or classical result instead.
_UNLOCATED_SOURCES: dict[str, str] = {
    "dpstats.tasks.linear_regression.algorithms.ridge.Ridge": (
        "non-private baseline citing its solver's method paper"
    ),
    "dpstats.tasks.single_coefficient_estimation.transformations.fwl.LargestOneHotFWLDemean": (
        "the classical Frisch-Waugh-Lovell theorem"
    ),
}


def _undescribed_fields(cls: type[BaseModel]) -> list[str]:
    """Return the fields a definition table would render with an empty cell.

    Inherited fields are included: the table renders the whole model, so a
    field described nowhere in the hierarchy is still a blank cell.
    """
    return [name for name, info in cls.model_fields.items() if not info.description]


def _built_in_transformations(task_id: str) -> dict[str, type[Transformation]]:
    """Return the transformation classes one built-in Task's package declares.

    Mirrors ``list_built_in_algorithms``'s ``.algorithms`` walk, but over
    ``.transformations``; a task without a ``transformations`` package (there is no
    ``built_in_transformations`` in the library itself, only algorithms) simply
    contributes none.
    """
    module_name = f"dpstats.tasks.{task_id}.transformations"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return {}
        raise
    catalog: dict[str, type[Transformation]] = {}
    for exported_name in module.__all__:
        candidate = getattr(module, exported_name)
        if isinstance(candidate, type) and issubclass(candidate, Transformation):
            catalog[exported_name] = candidate
    return catalog


def _privacy_interpretation(cls: type[PolymorphicModel]) -> str | None:
    """Return one class's constant privacy interpretation, or ``None``.

    Every built-in class's ``privacy_interpretation`` returns a fixed value
    that does not depend on configured field values (some transformations read
    another class's own constant), so an unvalidated
    :meth:`~pydantic.BaseModel.model_construct` shell is enough when a
    required field has no default to build a fully validated instance with.
    An abstract family base cannot be instantiated either way and realizes no
    task-level guarantee of its own.
    """
    try:
        instance = cls()
    except Exception:
        try:
            instance = cls.model_construct()
        except Exception:
            return None
    privacy_interpretation = getattr(instance, "privacy_interpretation", None)
    return None if privacy_interpretation is None else str(privacy_interpretation.value)


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _curated_classes() -> list[tuple[str, type[PolymorphicModel]]]:
    """Return one pair per built-in algorithm, transformation and tuner class."""
    resolved: dict[str, type[PolymorphicModel]] = {}
    for task in BUILT_IN_TASKS.values():
        for cls in list_built_in_algorithms(task).values():
            resolved[_qualified(cls)] = cls
        for cls in _built_in_transformations(task.id).values():
            resolved[_qualified(cls)] = cls
    tuning = importlib.import_module("dpstats.dp_tuning")
    for exported_name in tuning.__all__:
        cls = getattr(tuning, exported_name)
        resolved[_qualified(cls)] = cls
    return sorted(resolved.items())


def _curated_models() -> list[tuple[str, type[BaseModel]]]:
    """Return one pair per built-in task definition model and loss family."""
    resolved: dict[str, type[BaseModel]] = {}
    for task in BUILT_IN_TASKS.values():
        for cls in (task.Metadata, task.Input, task.Output):
            resolved[_qualified(cls)] = cls
    # Each built-in loss lives in the transformation module of the task presenting it.
    package = importlib.import_module("dpstats.tasks")
    for found in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        if ".transformations." not in found.name:
            continue
        module = importlib.import_module(found.name)
        for exported_name in getattr(module, "__all__", ()):
            candidate = getattr(module, exported_name)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, MEstimationLoss)
                and candidate is not MEstimationLoss
            ):
                resolved[_qualified(candidate)] = candidate
    return sorted(resolved.items())


def _curated_metrics() -> list[tuple[str, Metric]]:
    """Return one pair per Metric a built-in Task's registry serves."""
    resolved: list[tuple[str, Metric]] = []
    for task in BUILT_IN_TASKS.values():
        for key, metric in list_built_in_metrics(task).items():
            resolved.append((f"{task.id}.{key}", metric))
    return sorted(resolved, key=lambda item: item[0])


def _curated_budgets() -> list[tuple[str, type[BaseModel]]]:
    """Return one pair per exported budget family."""
    budgets = importlib.import_module("dpstats.budgets")
    return sorted(
        (_qualified(candidate), candidate)
        for candidate in (getattr(budgets, name) for name in budgets.__all__)
        if isinstance(candidate, type) and issubclass(candidate, BaseModel)
    )


def _referenced_classes() -> list[tuple[str, type]]:
    """Return every library class whose own docstring carries a ``Reference:`` block."""
    package = importlib.import_module("dpstats")
    resolved: dict[str, type] = {}
    for found in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        if ".templates" in found.name:
            continue
        module = importlib.import_module(found.name)
        for candidate in vars(module).values():
            if (
                isinstance(candidate, type)
                and candidate.__module__ == module.__name__
                and named_block(
                    inspect.cleandoc(candidate.__dict__.get("__doc__") or ""),
                    REFERENCE_HEADER,
                )
                is not None
            ):
                resolved[_qualified(candidate)] = candidate
    return sorted(resolved.items())


_BIBLIOGRAPHY = load()
_CLASSES = _curated_classes()
_REFERENCED = _referenced_classes()
_CITING = sorted({**dict(_CLASSES), **dict(_REFERENCED)}.items())
_MODELS = _curated_models()
_METRICS = _curated_metrics()
_BUDGETS = _curated_budgets()
_TASKS = sorted(BUILT_IN_TASKS.items())


# The metadata ``parameter_field`` attaches and a plain ``Field`` lacks: a
# field built with no constraints and no symbol carries exactly that marker.
_PARAMETER_FIELD_MARKERS = tuple(
    type(item) for item in parameter_field(default=None, description="probe").metadata
)


def _cited(cls: type) -> list[CitedWork] | None:
    """The works a class's ``Reference:`` block cites, ``None`` without a block."""
    reference = named_block(inspect.getdoc(cls) or "", REFERENCE_HEADER)
    return None if reference is None else cited_works(reference)


def _reference_shape_failure(cls: type[PolymorphicModel]) -> str:
    """Report a ``Reference:`` block the bibliography cannot resolve.

    Each bullet is ``- key: locator.``: a key of ``references.bib`` and the
    algorithm, section or result of that work the class follows. A class in
    ``_UNLOCATED_SOURCES`` cites ``- key.`` instead.
    """
    try:
        works = _cited(cls)
    except ValueError as error:
        return str(error)
    if works is None:
        return ""
    if not works:
        return "has an empty Reference: block"
    unknown = [work.key for work in works if work.key not in _BIBLIOGRAPHY.entries]
    if unknown:
        return f"cites keys missing from references.bib: {unknown}"
    if _qualified(cls) in _UNLOCATED_SOURCES:
        located = [work.key for work in works if work.locator]
        return f"is declared unlocated but locates {located}" if located else ""
    unlocated = [work.key for work in works if not _SOURCE_LOCATOR.search(work.locator)]
    if unlocated:
        return (
            f"cites {unlocated} without naming an Algorithm, Section, Appendix, "
            "Theorem, Definition, Proposition, Corollary, Lemma, Equation or "
            "Listing of it"
        )
    return ""


# How a cited class's opening paragraph begins: the citation sentence.
_OPENING_CITATION_PREFIXES = ("An implementation of ", "An adaptation of ")

# Classes built only on a standard primitive: no locator sentence; the opening
# cites the primitive's work parenthetically where it names the primitive.
_PRIMITIVE_CITERS = frozenset(
    {
        "dpstats.tasks.linear_regression.algorithms.grid_exponential_mechanism.GridExponentialMechanism",
        "dpstats.tasks.mean_estimation.algorithms.dp_histogram.DPHistogram",
        "dpstats.tasks.mean_estimation.algorithms.histogram_gaussian.HistogramGaussian",
    }
)

# Classes that implement no cited work: their Reference: block lists only the
# works their notes name, so their opening has no locator sentence.
_NOTE_ONLY_CITERS: dict[str, str] = {
    "dpstats.dp_tuning.exhaustive.DPExhaustiveCompose": (
        "a grid tuner; cites the composition results it follows"
    ),
    "dpstats.tasks.linear_regression.transformations.m_estimation.ToMEstimation": (
        "a task conversion; cites the regression recipe one option follows"
    ),
}


def _opening_citation_failure(cls: type) -> str:
    """Report a cited class whose opening paragraph does not lead with its citation.

    The opening names the first cited work as the bibliography's in-text form
    does, so its author list and year cannot drift from ``references.bib``.
    """
    own = cls.__dict__.get("__doc__")
    doc = inspect.cleandoc(own) if own else ""
    if named_block(doc, REFERENCE_HEADER) is None:
        return ""
    blocks = re.split(r"\n[ \t]*\n", doc)
    opening = " ".join(blocks[1].split()) if len(blocks) > 1 else ""
    if f"{cls.__module__}.{cls.__qualname__}" in _PRIMITIVE_CITERS:
        return _primitive_citation_failure(cls, opening)
    if _qualified(cls) in _NOTE_ONLY_CITERS:
        if opening.startswith(_OPENING_CITATION_PREFIXES):
            return "implements no cited work but opens with a locator sentence"
        return ""
    if not opening.startswith(_OPENING_CITATION_PREFIXES):
        return (
            "has a Reference: block but its opening paragraph does not start with "
            "'An implementation of ' or 'An adaptation of '"
        )
    try:
        works = _cited(cls) or []
        cited = short_citation(works[0].key, _BIBLIOGRAPHY) if works else ""
    except (ValueError, KeyError):
        return ""  # The reference-shape check reports the block itself.
    sentence = opening.split(". ", 1)[0]
    if cited and cited not in opening:
        return f"opens with {sentence!r}, which does not cite {cited!r}"
    return ""


def _primitive_citation_failure(cls: type, opening: str) -> str:
    """Report a primitive-only class whose opening breaks the primitive form.

    The opening must cite the primitive's work parenthetically, as
    "(Authors, year)", and carry no locator sentence.
    """
    if opening.startswith(_OPENING_CITATION_PREFIXES):
        return "is built on a standard primitive but opens with a locator sentence"
    try:
        works = _cited(cls) or []
        cited = short_citation(works[0].key, _BIBLIOGRAPHY) if works else ""
    except (ValueError, KeyError):
        return ""  # The reference-shape check reports the block itself.
    match = re.fullmatch(r"(.+) \((\d{4}[a-z]?)\)", cited)
    parenthetical = f"({match.group(1)}, {match.group(2)})" if match else ""
    if parenthetical and parenthetical not in opening:
        return f"does not cite {parenthetical!r} where it names its primitive"
    return ""


# One surname of an in-text citation, with any lower-case particle ("van Handel").
_SURNAME = r"(?:(?:van|von|de|der|den|du|le|la) )?[A-Z][\w'’-]*"
# An in-text citation: "Wang (2018)", "Balle and Wang (2018)", "Liu, Kong,
# Kakade and Oh (2021)" or "Alabi et al. (2022)", the year optionally lettered.
# Only a four-digit year closes it, so "(2T log n)" or "(1.1)" never matches.
_IN_TEXT_CITATION = re.compile(
    rf"(?:{_SURNAME}(?:, {_SURNAME})*,? and {_SURNAME}|{_SURNAME} et al\.|{_SURNAME})"
    r" \((?:19|20)\d{2}[a-z]?\)"
)
# Text ending inside an author list, so a citation matched after it is a tail
# of a longer one: "Redberg, Koskela and " before "Wang (2023)".
_AUTHOR_LIST_CONTINUES = re.compile(r"[A-Z][\w'’-]*(?:,|,? and) $")
# "the source" naming a cited work; "the source code" is not one.
_THE_SOURCE = re.compile(r"\bthe source\b(?! code)", re.IGNORECASE)


def _own_prose(cls: type) -> str:
    """A class's own docstring and its own fields' descriptions, as one line."""
    own = cls.__dict__.get("__doc__")
    parts = [inspect.cleandoc(own) if own else ""]
    if issubclass(cls, PolymorphicModel):
        parts += [cls.model_fields[name].description or "" for name in _own_fields(cls)]
    return " ".join(" ".join(parts).split())


def _ends_citing(before: str, citation: str) -> bool:
    """Whether ``before`` ends with ``citation`` as a whole author list."""
    head = before[: -len(citation)]
    return (
        before.endswith(citation)
        and not head[-1:].isalnum()
        and not _AUTHOR_LIST_CONTINUES.search(head)
    )


def _in_text_citation_failure(cls: type) -> str:
    """Report a cited work named other than as the bibliography's in-text form.

    Each "Authors (year)" in a class's docstring or field descriptions must be
    ``short_citation(key)`` for a key of its own ``Reference:`` block, so a
    work named in passing is listed there too and its authors and year cannot
    drift from ``references.bib``. A cited work is never "the source": it is
    named, or "the paper" or "the authors' code" where the referent is clear.
    """
    own = cls.__dict__.get("__doc__")
    reference = named_block(inspect.cleandoc(own) if own else "", REFERENCE_HEADER)
    try:
        works = [] if reference is None else cited_works(reference)
        allowed = [short_citation(work.key, _BIBLIOGRAPHY) for work in works]
    except (ValueError, KeyError):
        return ""  # The reference-shape check reports the block itself.
    text = _own_prose(cls)
    stray = [
        match.group(0)
        for match in _IN_TEXT_CITATION.finditer(text)
        if not any(_ends_citing(text[: match.end()], citation) for citation in allowed)
    ]
    if stray:
        return (
            f"names {stray}, none of which is the in-text citation of a work in its "
            f"Reference: block ({allowed})"
        )
    if _THE_SOURCE.search(text):
        return "calls a cited work 'the source'; name it, or say 'the paper'"
    return ""


# An opening's privacy-guarantee sentence: "It is an approximate-DP mechanism."
_GUARANTEE_SENTENCE = re.compile(r"It is an? [^.]*\bmechanism\.")
# A sentence ends at a full stop before a capital, but not after "e.g." or "i.e.".
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)\s+(?=[A-Z])")


def _opening_sentences(cls: type) -> list[str]:
    """Return the sentences of a class's opening paragraph, after its summary line."""
    own = cls.__dict__.get("__doc__")
    blocks = re.split(r"\n[ \t]*\n", inspect.cleandoc(own) if own else "")
    if len(blocks) < 2 or blocks[1].splitlines()[0].rstrip().endswith(":"):
        return []
    return _SENTENCE_BREAK.split(" ".join(blocks[1].split()))


def _guarantee_last_failure(cls: type) -> str:
    """Report an opening whose privacy-guarantee sentence is not its last."""
    sentences = _opening_sentences(cls)
    early = [s for s in sentences[:-1] if _GUARANTEE_SENTENCE.fullmatch(s)]
    if early:
        return (
            f"states its privacy guarantee {early[0]!r} before the end of its "
            "opening paragraph; the guarantee sentence comes last"
        )
    return ""


# A sentence that states a requirement; one that waives it ("It needs no upper
# bound on the data.", "It does not need a bound") restates nothing.
_REQUIREMENT_VERB = re.compile(
    r"\b(?:needs?|needed|requires?|required|assumes?|assumed|must)\b", re.I
)
_WAIVER = re.compile(r"\b(?:no|not|neither)\b", re.I)
# The run-time conditions a Requires: bullet states that an opening restates:
# a sentence requiring one is a restatement when a bullet names the same kind.
# A bullet may name the kind inside a field (``l2_norm_bound``); the sentence
# names it as a word, so "unbounded" is not a bound.
_CONDITION_TERMS = ("bound", "box", "radius", "radii", "budget", "intercept")
# Naming an accepted budget family is a Requires: condition wherever it appears.
_BUDGET_FAMILY = re.compile(r"\bbudget famil|``\w*Budget``")


def _requires_restated_failure(cls: type) -> str:
    """Report an opening sentence restating a Requires: bounds, budget or intercept condition.

    A sentence that uses known bounds in the mechanism ("Rescales x into
    [0, 1] using its known bounds") describes the method, not a requirement.
    """
    own = cls.__dict__.get("__doc__")
    requires = named_block(inspect.cleandoc(own) if own else "", REQUIRES_HEADER)
    bullets = " ".join(_bullet_items(requires or "")).lower()
    restated = [
        sentence
        for sentence in _opening_sentences(cls)
        if _BUDGET_FAMILY.search(sentence)
        or (
            _REQUIREMENT_VERB.search(sentence)
            and not _WAIVER.search(sentence)
            and any(
                term in bullets and re.search(rf"\b{term}(?:s|es)?\b", sentence, re.I)
                for term in _CONDITION_TERMS
            )
        )
    ]
    if restated:
        return f"restates a Requires: condition in its opening: {restated}"
    return ""


def _non_bullet_lines(block: str) -> list[str]:
    """Return the lines of a block that neither open nor continue a ``- `` item."""
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return []
    base = min(len(line) - len(line.lstrip()) for line in lines)
    stray: list[str] = []
    in_item = False
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if indent == base:
            in_item = line.lstrip().startswith("- ")
            if not in_item:
                stray.append(line.strip())
        elif not in_item:
            stray.append(line.strip())
    return stray


# A Requires bullet names a run-time refusal: one every input satisfies, or a
# configuration fact settled at construction, refuses nothing.
_VACUOUS_REQUIRES_OPENINGS = ("Any budget family", "Any declared")
_CONSTRUCTION_REQUIRES_PHRASE = "at construction"
# A Requires bullet is the bare condition: the default ``InvalidRegime`` goes
# unnamed, and a reason or an account of the refused case is prose.
_PROSE_REQUIRES_PHRASES = (
    "; an unconstrained offer",
    " is refused",
    "raises ``InvalidRegime``",
)
_MAX_REQUIRES_BULLET_CHARS = 100


def _bullet_items(block: str) -> list[str]:
    """Return each ``- `` item of a block with its continuation lines joined."""
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:])
        elif stripped and items:
            items[-1] += " " + stripped
    return items


def _requires_failure(cls: type[PolymorphicModel]) -> str:
    """Report a Requires: block that is not a list of run-time refusals."""
    requires = named_block(inspect.getdoc(cls) or "", REQUIRES_HEADER)
    if requires is None:
        return ""
    if not requires.strip():
        return "has an empty Requires: block"
    stray = _non_bullet_lines(requires)
    if stray:
        return f"has Requires: lines outside a '- ' bulleted list: {stray}"
    vacuous = [
        item
        for item in _bullet_items(requires)
        if item.startswith(_VACUOUS_REQUIRES_OPENINGS)
        or _CONSTRUCTION_REQUIRES_PHRASE in item
    ]
    if vacuous:
        return (
            "has Requires: bullets that refuse nothing at run time (every "
            f"input satisfies them, or construction settles them): {vacuous}"
        )
    prose = [
        item
        for item in _bullet_items(requires)
        if any(phrase in item for phrase in _PROSE_REQUIRES_PHRASES)
        or len(" ".join(item.split())) > _MAX_REQUIRES_BULLET_CHARS
    ]
    if prose:
        return (
            "has Requires: bullets that are not a bare condition (explanatory "
            f"prose, a named default InvalidRegime, or over "
            f"{_MAX_REQUIRES_BULLET_CHARS} characters): {prose}"
        )
    return ""


def _block_shape_failure(cls: type) -> str:
    """Report sections the reference renderer would overwrite."""
    violations = named_block_shape_violations(inspect.getdoc(cls) or "")
    return "" if not violations else f"has malformed named blocks: {violations}"


def _parameter_field_names(cls: type[PolymorphicModel]) -> list[str]:
    """Return every field name, own or inherited, declared with ``parameter_field``."""
    return [
        name
        for name, info in cls.model_fields.items()
        if any(isinstance(item, _PARAMETER_FIELD_MARKERS) for item in info.metadata)
    ]


def _declaring_class(cls: type, name: str) -> type:
    """Return the class in ``cls``'s MRO whose own body declares field ``name``."""
    return next(
        (owner for owner in cls.__mro__ if name in inspect.get_annotations(owner)),
        cls,
    )


def _notes_name(cls: type, name: str) -> bool:
    """Whether ``cls``'s own Parametrization notes: block names ``name``."""
    block = named_block(inspect.getdoc(cls) or "", PARAMETRIZATION_NOTES_HEADER)
    return block is not None and bool(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", block)
    )


def _parametrization_coverage_failure(cls: type[PolymorphicModel]) -> str:
    """Report configured hyperparameters absent from their provenance notes.

    An inherited hyperparameter's note may live once, on the class that
    declares it, when that class is itself a checked, documented subject; a
    subclass restates it only to say something that class changes.
    """
    checked = {owner for _, owner in _CLASSES}
    missing = [
        name
        for name in _parameter_field_names(cls)
        if not _notes_name(cls, name)
        and not (
            (owner := _declaring_class(cls, name)) in checked
            and _notes_name(owner, name)
        )
    ]
    if named_block(inspect.getdoc(cls) or "", PARAMETRIZATION_NOTES_HEADER) is None:
        return (
            ""
            if not missing
            else f"declares hyperparameter(s) {missing} but has no Parametrization notes: block"
        )
    return (
        ""
        if not missing
        else f"has a Parametrization notes: block with no explanation of hyperparameter(s): {missing}"
    )


# A Parametrization note saying "The default, 0.1, ..." repeats the value the
# page already prints beside the field; the note says why and how instead.
_RESTATED_DEFAULT = re.compile(r"\bThe default, ")


def _default_value_restated_failure(cls: type[PolymorphicModel]) -> str:
    """Report a Parametrization note that restates a default's value."""
    block = named_block(inspect.getdoc(cls) or "", PARAMETRIZATION_NOTES_HEADER)
    return (
        "has a Parametrization note restating a default's value; say why and how "
        "it was chosen instead"
        if block is not None and _RESTATED_DEFAULT.search(block)
        else ""
    )


# Implementation notes record design decisions. A list of the metadata fields
# a class reads is mechanically recognizable as not being one. Repetition of a
# Requires or Parametrization bullet is not checked: word overlap both flags
# decisions that explain a requirement and misses paraphrases.
_METADATA_READ_NOTE = re.compile(r"^\s*- Reads ``metadata", re.MULTILINE)


def _implementation_notes_failure(cls: type[PolymorphicModel]) -> str:
    """Report an Implementation note that only lists the metadata fields read."""
    block = named_block(inspect.getdoc(cls) or "", IMPLEMENTATION_NOTES_HEADER)
    return (
        "has an Implementation note listing metadata reads, which is not a "
        "design decision"
        if block is not None and _METADATA_READ_NOTE.search(block)
        else ""
    )


def _own_fields(cls: type[PolymorphicModel]) -> list[str]:
    """Return the field names ``cls``'s own class body declares.

    A tuner base like ``DPHyperparameterTuned`` declares fields a concrete
    tuner subclass only inherits, so those are not this class's own; reading
    ``cls.__dict__["__annotations__"]`` instead of comparing against ancestor
    ``model_fields`` also catches a subclass that redeclares an ancestor's
    field name with its own new definition, which a name-absent-from-ancestors
    test would silently skip under the ancestor's original one.
    """
    own_annotations = vars(cls).get("__annotations__", {})
    return [name for name in own_annotations if name in cls.model_fields]


def _hyperparameter_field_failure(cls: type[PolymorphicModel]) -> str:
    """Report a field this class declares itself as a plain, defaulted ``Field``.

    A field with no declared default, or one filled by a ``default_factory``,
    cannot become a :func:`~dpstats.parameters.parameter_field`
    without inventing a default it never had, so it is not a hyperparameter in
    this sense and plain ``Field`` remains correct for it. Every other field
    this class declares itself is a hyperparameter and must carry the
    metadata that only :func:`~dpstats.parameters.parameter_field`
    attaches.
    """
    plain: list[str] = []
    for name in _own_fields(cls):
        info = cls.model_fields[name]
        if info.is_required() or info.default_factory is not None:
            continue
        if not any(
            isinstance(item, _PARAMETER_FIELD_MARKERS) for item in info.metadata
        ):
            plain.append(name)
    return (
        ""
        if not plain
        else f"declares defaulted hyperparameter field(s) with plain Field, not "
        f"parameter_field: {plain}"
    )


def _descriptions_failure(cls: type) -> str:
    """Report definition-table fields a page would render with an empty cell."""
    undescribed = _undescribed_fields(cls)
    return (
        "" if not undescribed else f"renders fields with no description: {undescribed}"
    )


def _metric_summary_failure(metric: Metric) -> str:
    """Report a metric whose calculation docstring has no one-line summary.

    The metrics page's summary row is the calculation docstring's first line.
    """
    doc = (inspect.getdoc(metric.calculation) or "").strip()
    first_line = doc.splitlines()[0].strip() if doc else ""
    return (
        ""
        if first_line.endswith(".")
        else f"has no one-line description on its calculation: {first_line!r}"
    )


_MATH_ROLE = re.compile(r":math:`")
# The delimiters the page sets around a declared formula itself.
_MATH_DELIMITERS = ("$", "\\(", "\\[")
_NOTES_HEADING = re.compile(r"^Notes:?$")


def _metric_doc(metric: Metric) -> str:
    return inspect.cleandoc(metric.calculation.__doc__ or "")


def _parameter_entries(doc: str) -> dict[str, str]:
    """Return the Google ``Args:`` entries the metrics page reads, by name."""
    entries: dict[str, str] = {}
    for section in Docstring(doc).parse(Parser.google, warnings=False):
        if section.kind == DocstringSectionKind.parameters:
            for parameter in section.value:
                entries[parameter.name] = parameter.description.strip()
    return entries


def _reference_names(metric: Metric) -> list[str]:
    """Return the ``requires`` names other than the Output, in declaration order."""
    return list(metric.required_names)


def _metric_parameters_failure(metric: Metric) -> str:
    """Report undocumented output or reference arguments in a calculation."""
    entries = _parameter_entries(_metric_doc(metric))
    needed = ["output", *_reference_names(metric)]
    missing = [name for name in needed if name not in entries]
    extra = [name for name in entries if name not in needed]
    blank = [name for name in needed if name in entries and not entries[name]]
    if not (missing or extra or blank):
        return ""
    return (
        "has a Google Args: section that is not one described entry per "
        f"calculation argument: missing {missing}, unexpected {extra}, "
        f"empty description {blank}"
    )


def _metric_argument_registration_failure(metric: Metric) -> str:
    """Report a stored dependency with no registered meaning."""
    registry = list_built_in_metric_arguments(metric.task)
    stored = [
        item
        for item in metric.requires
        if isinstance(item, str) and item not in RESERVED_REQUIREMENTS
    ]
    unregistered = [name for name in stored if name not in registry]
    return (
        f"requires unregistered stored metric arguments: {unregistered}"
        if unregistered
        else ""
    )


def _metric_notes_section_failure(metric: Metric) -> str:
    """Report a notes heading the Google parser cannot recognize."""
    lines = _metric_doc(metric).splitlines()
    return (
        "has a Notes section without the Google-style colon"
        if any(_NOTES_HEADING.match(line) and line != "Notes:" for line in lines)
        else ""
    )


def _metric_formula_delimiters_failure(metric: Metric) -> str:
    """Report a ``Metric.formula`` carrying a math delimiter the page adds itself."""
    found = [d for d in _MATH_DELIMITERS if d in (metric.formula or "")]
    return "" if not found else f"has a formula carrying math delimiter(s) {found}"


# The longest label a chart axis, picker chip or table heading prints whole.
_LABEL_LIMIT = 32


def _metric_label_failure(metric: Metric) -> str:
    """Report a built-in Metric whose ``label`` is not short reader-facing text.

    A page prints the label where a reader, not code, reads the Metric, so it
    is set, fits a chart axis, and is not the identifier repeated.
    """
    label = metric.label
    if not label or label != label.strip():
        return f"has no label, or one with surrounding whitespace: {label!r}"
    if len(label) > _LABEL_LIMIT:
        return f"has a label longer than {_LABEL_LIMIT} characters: {label!r}"
    if label == metric.name or "_" in label:
        return f"has a label that repeats its identifier: {label!r}"
    return ""


def _math_role_placement_failure(subject: Any) -> str:
    """Report math roles outside the sections the metric renderer converts."""
    if not isinstance(subject, Metric):
        return (
            "carries a :math: role unsupported by its reference renderer"
            if _MATH_ROLE.search(inspect.getdoc(subject) or "")
            else ""
        )
    doc = _metric_doc(subject)
    allowed = False
    for line in doc.splitlines():
        if line and not line[0].isspace():
            allowed = line in {"Args:", "Notes:"}
        if _MATH_ROLE.search(line) and not allowed:
            return "carries a :math: role outside the rendered Args and Notes sections"
    return ""


def _input_and_group_argument_descriptions_disagree() -> list[str]:
    """Return conflicting descriptions supplied by calculation docstrings."""
    seen: dict[tuple[str, str], set[str]] = {}
    for _, metric in _METRICS:
        entries = _parameter_entries(_metric_doc(metric))
        for name in metric.required_names:
            if name not in RESERVED_REQUIREMENTS:
                continue
            if name in entries:
                seen.setdefault((metric.task.id, name), set()).add(entries[name])
    return sorted(
        f"{task}.{ref}" for (task, ref), texts in seen.items() if len(texts) > 1
    )


def _summary_line(doc: str) -> str:
    lines = doc.strip().splitlines()
    return lines[0].strip() if lines else ""


def _task_description_failure(task: Task[Any, Any, Any]) -> str:
    """Report a Task with no description of its estimation problem."""
    description = task.description.strip()
    return "" if description else "has no description of the estimation problem"


_CHECKS: dict[str, Callable[[Any], str]] = {
    REFERENCE_SHAPE: _reference_shape_failure,
    OPENING_CITATION: _opening_citation_failure,
    IN_TEXT_CITATION: _in_text_citation_failure,
    GUARANTEE_LAST: _guarantee_last_failure,
    REQUIRES_RESTATED: _requires_restated_failure,
    REQUIRES: _requires_failure,
    DESCRIPTIONS: _descriptions_failure,
    METRIC_SUMMARY: _metric_summary_failure,
    METRIC_PARAMETERS: _metric_parameters_failure,
    METRIC_ARGUMENT_REGISTRATION: _metric_argument_registration_failure,
    METRIC_NOTES_SECTION: _metric_notes_section_failure,
    METRIC_FORMULA_DELIMITERS: _metric_formula_delimiters_failure,
    METRIC_LABEL: _metric_label_failure,
    MATH_ROLE_PLACEMENT: _math_role_placement_failure,
    BLOCK_SHAPE: _block_shape_failure,
    HYPERPARAMETER_FIELD: _hyperparameter_field_failure,
    PARAMETRIZATION_COVERAGE: _parametrization_coverage_failure,
    DEFAULT_VALUE_RESTATED: _default_value_restated_failure,
    IMPLEMENTATION_NOTES: _implementation_notes_failure,
    TASK_DESCRIPTION: _task_description_failure,
}

# Each check over the objects whose generated page reads what it checks.
_SUBJECTS: dict[str, list[tuple[str, Any]]] = {
    REFERENCE_SHAPE: _CITING,
    OPENING_CITATION: _CLASSES,
    IN_TEXT_CITATION: _CLASSES,
    GUARANTEE_LAST: _CLASSES,
    REQUIRES_RESTATED: _CLASSES,
    REQUIRES: _CLASSES,
    BLOCK_SHAPE: _CLASSES,
    HYPERPARAMETER_FIELD: _CLASSES,
    PARAMETRIZATION_COVERAGE: _CLASSES,
    DEFAULT_VALUE_RESTATED: _CLASSES,
    IMPLEMENTATION_NOTES: _CLASSES,
    DESCRIPTIONS: _MODELS,
    METRIC_SUMMARY: _METRICS,
    METRIC_PARAMETERS: _METRICS,
    METRIC_ARGUMENT_REGISTRATION: _METRICS,
    METRIC_NOTES_SECTION: _METRICS,
    METRIC_FORMULA_DELIMITERS: _METRICS,
    METRIC_LABEL: _METRICS,
    MATH_ROLE_PLACEMENT: [*_CLASSES, *_MODELS, *_METRICS],
    TASK_DESCRIPTION: _TASKS,
}
_CASES = [
    pytest.param(check, name, subject, id=f"{check}-{name}")
    for check, subjects in _SUBJECTS.items()
    for name, subject in subjects
]


@pytest.mark.parametrize("check,name,subject", _CASES)
def test_built_in_documentation_source_clears_its_check(
    check: str, name: str, subject: Any
) -> None:
    """Each generated page reads a source that meets the docstring standard."""
    failure = _CHECKS[check](subject)
    assert not failure, f"{name} {failure}"


def test_input_and_group_argument_descriptions_agree() -> None:
    """The rendered Input and group descriptions are unambiguous."""
    assert not _input_and_group_argument_descriptions_disagree()


def test_unlocated_sources_are_cited_built_ins_without_a_locator() -> None:
    """Every exemption still applies: a cited built-in that cites no locator."""
    stale = set()
    for name, cls in _CLASSES:
        if name in _UNLOCATED_SOURCES:
            works = _cited(cls)
            if not works or any(work.locator for work in works):
                stale.add(name)
    missing = set(_UNLOCATED_SOURCES) - {name for name, _ in _CLASSES}
    assert not stale | missing


# Works no DOI or arXiv identifier names, with the identifier they do have.
_UNIDENTIFIED_WORKS: dict[str, str] = {
    "kifer2012private": "COLT 2012 proceedings page (PMLR v23); no DOI or arXiv version",
    "brown2026differentially": "TPDP 2026 presentation; no DOI or arXiv version",
    "alabi2022code": "GitHub repository at a pinned commit; no DOI or arXiv version",
    "amin2022code": "GitHub repository at a pinned commit; no DOI or arXiv version",
    "dick2023code": "GitHub repository at a pinned commit; no DOI or arXiv version",
}


# Works only a Distribution Release provenance record cites, with that record.
_RELEASE_CITED_WORKS: dict[str, str] = {
    "yu2024gaussian": "benchmark-info/provenance/algorithms/yu-ren-zhou-huber-mean.md",
}


def test_every_bibliography_entry_is_cited_by_a_class() -> None:
    """``references.bib`` holds only works some class's ``Reference:`` cites,
    or that a declared release provenance record cites."""
    cited = {work.key for _, cls in _REFERENCED for work in (_cited(cls) or [])}
    assert set(_BIBLIOGRAPHY.entries) - cited == set(_RELEASE_CITED_WORKS)


def test_every_bibliography_entry_has_a_doi_or_an_arxiv_eprint() -> None:
    """A reader can resolve every cited work by a persistent identifier."""
    unidentified = {
        key
        for key, entry in _BIBLIOGRAPHY.entries.items()
        if "doi" not in entry.fields
        and not (
            "eprint" in entry.fields and entry.fields.get("archivePrefix") == "arXiv"
        )
    }
    assert unidentified == set(_UNIDENTIFIED_WORKS)


def test_every_reference_block_is_on_a_checked_class() -> None:
    """The gate reaches every class that cites a work, not a curated subset."""
    assert len(_REFERENCED) >= 49
    assert {name for name, _ in _REFERENCED} <= {name for name, _ in _CITING}


def test_only_declared_classes_realize_no_privacy_interpretation() -> None:
    """Every built-in class but the declared abstract base names its semantics."""
    unresolved = {
        name for name, cls in _CLASSES if _privacy_interpretation(cls) is None
    }
    assert unresolved == _NO_PRIVACY_INTERPRETATION


def test_curated_objects_cover_every_built_in_task() -> None:
    """A near-empty catalog would silently stop gating almost everything."""
    assert len(_CLASSES) > 50
    assert len(_MODELS) > 15
    assert len(_METRICS) > 10


# Fixtures distinguish missing scientific content from permitted formatting.


class _ReferenceBeforeAssumptionsFixture:
    """One line of prose.

    Reference:
        - wang2018revisiting: Algorithm 2.

    Requires:
        - Some public assumption.
    """


class _BadBlockDuplicateFixture:
    """One line of prose.

    Requires:
        - Some public assumption.

    Implementation notes:
        - First note.

    Implementation notes:
        - Second, duplicated block.
    """


class _AttributesSectionFixture:
    """One line of prose.

    Attributes:
        threshold: The configured threshold.
    """


class _ParametrizationOutOfOrderFixture:
    """One line of prose.

    Requires:
        - Some public assumption.

    Implementation notes:
        - A note.

    Parametrization notes:
        - ``threshold`` comes from the cited paper's cutoff.
    """


class _MissingHyperparameterParametrizationNotesFixture(Transformation):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.

    Parametrization notes:
        - ``scale`` comes from the cited paper's calibration.
    """

    OUTER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))
    INNER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))

    threshold: float = parameter_field(default=1.0, description="A hyperparameter.")

    def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
        raise NotImplementedError

    def restore(self, output: Any, restoration: Any) -> Any:
        raise NotImplementedError


class _AbsentParametrizationNotesFixture(Transformation):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.
    """

    OUTER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))
    INNER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))

    threshold: float = parameter_field(default=1.0, description="A hyperparameter.")

    def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
        raise NotImplementedError

    def restore(self, output: Any, restoration: Any) -> Any:
        raise NotImplementedError


class _ConformingParametrizationNotesFixture(Transformation):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.

    Parametrization notes:
        - ``threshold`` comes from the cited paper's cutoff.
    """

    OUTER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))
    INNER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))

    threshold: float = parameter_field(default=1.0, description="A hyperparameter.")

    def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
        raise NotImplementedError

    def restore(self, output: Any, restoration: Any) -> Any:
        raise NotImplementedError


class _NotesThatAreNotDecisionsFixture:
    """One line of prose.

    Requires:
        - A finite public bound on every row's L2 norm.

    Parametrization notes:
        - ``threshold``: The default, 0.5, comes from the paper.

    Implementation notes:
        - Reads ``metadata.n_samples``.
    """


def test_metadata_read_note_and_restated_default_fail() -> None:
    """A metadata-read note and a restated default fail; a provenance note passes."""
    assert _implementation_notes_failure(_NotesThatAreNotDecisionsFixture)
    assert _default_value_restated_failure(_NotesThatAreNotDecisionsFixture)
    assert not _default_value_restated_failure(_ConformingParametrizationNotesFixture)


class _PlainFieldHyperparameterFixture(Transformation):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.
    """

    OUTER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))
    INNER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))

    threshold: float = Field(
        default=1.0, gt=0, description="A defaulted field left on plain Field."
    )

    def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
        raise NotImplementedError

    def restore(self, output: Any, restoration: Any) -> Any:
        raise NotImplementedError


class _ProseRequiresFixture:
    """One line of prose.

    Requires:
        A finite radius on the domain, refused otherwise, which supplies the
        scale of the noise.
    """


class _BulletedRequiresFixture:
    """One line of prose.

    Requires:
        - A finite radius on the domain.
        - A budget family admitting a zCDP implication; an infinite one raises
          ``NotImplementedError``.
    """


class _AnyBudgetRequiresFixture:
    """One line of prose.

    Requires:
        - Any budget family.
        - A finite radius on the domain.
    """


class _AnyDeclaredProblemRequiresFixture:
    """One line of prose.

    Requires:
        - Any declared linear-regression problem.
    """


class _ConstructionRequiresFixture:
    """One line of prose.

    Requires:
        - A ``scale`` and ``num_bins`` whose magnitude grid is finite;
          otherwise ``ValueError`` at construction.
    """


class _RefusedOfferRequiresFixture:
    """One line of prose.

    Requires:
        - An ``ApproxDPBudget`` with finite epsilon; an unconstrained offer of
          any family is refused.
    """


class _NamedInvalidRegimeRequiresFixture:
    """One line of prose.

    Requires:
        - At least two declared samples; fewer raises ``InvalidRegime``.
    """


class _LongRequiresFixture:
    """One line of prose.

    Requires:
        - Finite ``design_domain.l2_norm_bound`` and
          ``response_domain.l2_norm_bound``, the public L2 bounds the paper's
          analysis assumes.
    """


class _BareRequiresFixture:
    """One line of prose.

    Requires:
        - A finite ``ApproxDPBudget``.
        - An ``ApproxDPBudget`` or an unconstrained budget.
        - A ``num_features`` within ``[1, dim - 1]``; otherwise ``NotApplicable``.
    """


class _NoRequiresFixture:
    """One line of prose.

    Implementation notes:
        - Refuses nothing, so it declares no requirement.
    """


class _RequiresThenReferenceFixture:
    """One line of prose.

    Requires:
        - Some public assumption.

    Reference:
        - wang2018revisiting: Algorithm 2.
    """


class _UnlocatedReferenceFixture:
    """Release a noisy mean.

    Adds Gaussian noise to the clipped sum.

    Requires:
        - Some public assumption.

    Reference:
        - wang2018revisiting.
    """


class _LocatorOnlyInNotesFixture:
    """Release a noisy mean.

    Adds Gaussian noise to the clipped sum.

    Requires:
        - Some public assumption.

    Implementation notes:
        - Follows Algorithm 2 of the cited paper.

    Reference:
        - wang2018revisiting.
    """


class _LocatedReferenceFixture:
    """Release a noisy mean.

    Adds Gaussian noise to the clipped sum.

    Requires:
        - Some public assumption.

    Reference:
        - wang2018revisiting: Algorithm 2.
        - balle2018improving: Theorem 8 and
          Algorithm 1.
    """


class _UnknownKeyReferenceFixture:
    """Release a noisy mean.

    Adds Gaussian noise to the clipped sum.

    Requires:
        - Some public assumption.

    Reference:
        - nobody2020missing: Algorithm 2.
    """


class _ProseReferenceFixture:
    """Release a noisy mean.

    Adds Gaussian noise to the clipped sum.

    Reference:
        Wang, Y.-X. (2018). "Revisiting Differentially Private Linear
        Regression." UAI.
        https://arxiv.org/abs/1803.02596
    """


class _LetteredAppendixFixture:
    """Release a noisy mean.

    Adapts the construction in an appendix of the cited paper.

    Requires:
        - Some public assumption.

    Reference:
        - wang2018revisiting: Appendix C.
    """


class _UnrecognizedHeadingFixture:
    """One line of prose.

    Requires:
        - Some public assumption.

    Notation:
        x is the input, y is the output.
    """


class _AncestorWithParameterFieldFixture(Transformation):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.
    """

    OUTER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))
    INNER_TASK: ClassVar[Task] = next(iter(BUILT_IN_TASKS.values()))

    threshold: float = parameter_field(
        default=1.0, description="A real hyperparameter."
    )

    def prepare(self, metadata: Any, input: Any, budget: Any, seed: int) -> Any:
        raise NotImplementedError

    def restore(self, output: Any, restoration: Any) -> Any:
        raise NotImplementedError


class _SubclassRedeclaresFieldWithPlainFieldFixture(_AncestorWithParameterFieldFixture):
    """One line of prose.

    Requires:
        - Nothing beyond the inner task's own contract.
    """

    threshold: float = Field(
        default=2.0, description="Redeclared with plain Field, evading the gate."
    )


def test_parametrization_coverage_names_class_and_missing_field() -> None:
    """The assembled failure states both what is undocumented and where.

    ``_parametrization_coverage_failure`` itself returns a bare fragment, like
    every sibling check in this file, because
    ``test_built_in_documentation_source_clears_its_check`` is what prepends
    the subject's name; this proves that assembled message, not the fragment
    alone, carries both facts.
    """
    fixture = _MissingHyperparameterParametrizationNotesFixture
    failure = _parametrization_coverage_failure(fixture)
    message = f"{_qualified(fixture)} {failure}"
    assert "_MissingHyperparameterParametrizationNotesFixture" in message
    assert "threshold" in message


def _fixture_metric(
    doc: str, *, requires: tuple[str, ...], formula: str | None
) -> Metric:
    def calculation(output: Any, **_: Any) -> float:
        return 0.0

    calculation.__doc__ = doc
    return Metric(
        task=next(iter(BUILT_IN_TASKS.values())),
        name="fixture",
        calculation=calculation,
        requires=requires,
        formula=formula,
    )


_CONFORMING_DOC = "Fixture metric.\n\nArgs:\n    output: The algorithm output to evaluate.\n    ground_truth (float):\n        The truth.\n\nNotes:\n    Uses :math:`x`."


@pytest.mark.parametrize(
    "check,doc,requires,formula",
    [
        (_metric_parameters_failure, "Fixture metric.\n\nBody.", ("a",), None),
        (
            _metric_parameters_failure,
            "Fixture.\n\nArgs:\n    output: The algorithm output to evaluate.\n    a (float):\n\nNotes:\n    x.",
            ("a",),
            None,
        ),
        (
            _metric_parameters_failure,
            "Fixture.\n\nArgs:\n    output: The algorithm output to evaluate.\n    a (float):\n        A.\n    b (float):\n        B.",
            ("a",),
            None,
        ),
        (
            _metric_argument_registration_failure,
            "Fixture.\n\nArgs:\n    not_registered (float):\n        Words.",
            ("not_registered",),
            None,
        ),
        (_metric_notes_section_failure, "Fixture.\n\nNotes\n-----\nWords.", (), None),
        (_metric_notes_section_failure, "Fixture.\n\nNotes\n\nWords.", (), None),
        (_metric_formula_delimiters_failure, "Fixture.", (), "$x$"),
        (_metric_label_failure, "Fixture.", (), None),
        (_math_role_placement_failure, "Fixture :math:`x`.\n\nBody.", (), None),
    ],
    ids=[
        "no-parameters",
        "empty-description",
        "unrequired-entry",
        "reference-not-registered",
        "numpy-notes",
        "notes-without-colon",
        "formula-delimiter",
        "no-label",
        "math-in-summary",
    ],
)
def test_metric_check_fails_on_a_violating_fixture(
    check: Callable[[Any], str],
    doc: str,
    requires: tuple[str, ...],
    formula: str | None,
) -> None:
    """Each Metric check refuses the shape it names."""
    assert check(_fixture_metric(doc, requires=requires, formula=formula))


def test_metric_checks_pass_a_conforming_fixture() -> None:
    """A Google-style metric docstring preserves its scientific content."""
    metric = _fixture_metric(
        _CONFORMING_DOC,
        requires=("ground_truth",),
        formula=r"\lVert x \rVert",
    )
    for check in (
        _metric_parameters_failure,
        _metric_argument_registration_failure,
        _metric_notes_section_failure,
        _metric_formula_delimiters_failure,
        _math_role_placement_failure,
    ):
        assert not check(metric)


def test_opening_citation_check_reads_the_opening_of_a_cited_class() -> None:
    """A cited class leads its opening with the citation; an uncited one need not."""

    class _CitedLast:
        """Summary.

        Does the thing. Follows Wang (2018).

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    class _CitedFirst:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018). Does the thing.

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    class _CitedWithTheWrongYear:
        """Summary.

        An implementation of Algorithm 2 from Wang (2017). Does the thing.

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    class _CitedWithTheWrongAuthors:
        """Summary.

        An implementation of Algorithm 1 from Dick et al. (2023). Does the thing.

        Reference:
            - dick2023better: Algorithm 1.
        """

    class _Uncited:
        """Summary.

        Does the thing.
        """

    assert _opening_citation_failure(_CitedLast)
    assert not _opening_citation_failure(_CitedFirst)
    assert _opening_citation_failure(_CitedWithTheWrongYear)
    assert _opening_citation_failure(_CitedWithTheWrongAuthors)
    assert not _opening_citation_failure(_Uncited)


def test_in_text_citation_check_wants_the_bibliography_form_of_a_listed_work() -> None:
    """Every "Authors (year)" is a listed work's in-text citation; no "the source"."""

    class _NamedAsListed:
        """Summary.

        An implementation of Algorithm 2 from Dick, Gillenwater and Joseph
        (2023), sized as in Wang (2018). The paper sets $k$ by Equation (1.1),
        with (2T log n) steps, and Section 4.2 (2023 version) keeps the source
        code's seed.

        Reference:
            - dick2023better: Algorithm 2.
            - wang2018revisiting: Algorithm 1.
        """

    class _NamedInPassing:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018). Calibrated as in
        Balle and Wang (2018).

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    class _NamedWithTheWrongAuthors:
        """Summary.

        An implementation of Algorithm 1 from Dick et al. (2023).

        Reference:
            - dick2023better: Algorithm 1.
        """

    class _NamedAsATailOfAnotherList:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018), as in Redberg,
        Koskela and Wang (2018).

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    class _CalledTheSource:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018). The source sets
        no value.

        Reference:
            - wang2018revisiting: Algorithm 2.
        """

    assert not _in_text_citation_failure(_NamedAsListed)
    assert _in_text_citation_failure(_NamedInPassing)
    assert _in_text_citation_failure(_NamedWithTheWrongAuthors)
    assert _in_text_citation_failure(_NamedAsATailOfAnotherList)
    assert _in_text_citation_failure(_CalledTheSource)


def test_note_only_citers_still_cite_a_work() -> None:
    """Every declared note-only citer is a class with a Reference: block."""
    assert set(_NOTE_ONLY_CITERS) <= {name for name, _ in _REFERENCED}


def test_guarantee_check_wants_the_guarantee_sentence_last() -> None:
    """An opening that states its guarantee states it after the description."""

    class _GuaranteeSecond:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018). It is a Gaussian-DP
        mechanism. Adds noise to the sufficient statistics.
        """

    class _GuaranteeLast:
        """Summary.

        An implementation of Algorithm 2 from Wang (2018). Adds noise to the
        sufficient statistics. It is a Gaussian-DP mechanism.
        """

    class _NoGuarantee:
        """Summary.

        Adds no noise, so it gives no privacy guarantee.
        """

    assert _guarantee_last_failure(_GuaranteeSecond)
    assert not _guarantee_last_failure(_GuaranteeLast)
    assert not _guarantee_last_failure(_NoGuarantee)


def test_requires_restated_check_reads_requirement_sentences() -> None:
    """An opening never restates a Requires: bounds or budget condition."""

    class _NeedsKnownBounds:
        """Summary.

        Draws one posterior sample. Needs known bounds on each record and the
        response.

        Requires:
            - A finite ``design_domain.l2_norm_bound``.
        """

    class _NamesBudgetFamily:
        """Summary.

        Accepts a budget family admitting a zCDP implication. Adds noise.
        """

    class _NamesBudgetClass:
        """Summary.

        Adds noise. It must be offered a finite ``ApproxDPBudget``.
        """

    class _RequiresAnIntercept:
        """Summary.

        Drops the intercept column. It requires a declared intercept.

        Requires:
            - A declared intercept.
        """

    class _UsesKnownBounds:
        """Summary.

        Rescales x into [0, 1] using its known bounds and adds noise.

        Requires:
            - A finite bounding box on ``metadata.domain``.
        """

    class _WaivesABound:
        """Summary.

        Stops at the first candidate above half the data. It needs no upper
        bound on the data.

        Requires:
            - A finite lower bound on ``metadata.domain``'s bounding box.
        """

    class _BoundsAreRequired:
        """Summary.

        Adds noise. Known bounds on x are required.

        Requires:
            - A finite bounding box on ``metadata.domain``.
        """

    class _DoesNotNeedABound:
        """Summary.

        Clips at a privately estimated radius. It does not need a bound on y.

        Requires:
            - A finite lower bound on ``metadata.domain``'s bounding box.
        """

    class _AssumesUnboundedData:
        """Summary.

        It assumes unbounded data and estimates a range first.

        Requires:
            - A finite lower bound on ``metadata.domain``'s bounding box.
        """

    class _NeedsAnUnlistedCondition:
        """Summary.

        Fits a Lasso per block. It needs at least two blocks.

        Requires:
            - At least ``dim`` declared samples.
        """

    assert _requires_restated_failure(_NeedsKnownBounds)
    assert _requires_restated_failure(_NamesBudgetFamily)
    assert _requires_restated_failure(_NamesBudgetClass)
    assert _requires_restated_failure(_RequiresAnIntercept)
    assert not _requires_restated_failure(_UsesKnownBounds)
    assert not _requires_restated_failure(_WaivesABound)
    assert not _requires_restated_failure(_NeedsAnUnlistedCondition)
    assert _requires_restated_failure(_BoundsAreRequired)
    assert not _requires_restated_failure(_DoesNotNeedABound)
    assert not _requires_restated_failure(_AssumesUnboundedData)


def test_math_role_in_a_class_docstring_is_refused() -> None:
    """Unsupported math roles are rejected before a reference page is built."""

    class _WithMath:
        r"""Mean :math:`\mu`."""

    assert _math_role_placement_failure(_WithMath)


def test_task_description_requires_content_without_a_line_limit() -> None:
    task = BUILT_IN_TASKS["mean_estimation"]
    original = task.description
    try:
        task.description = ""
        assert _task_description_failure(task)
        task.description = (
            "The population mean of vector-valued observations.\n\n"
            "Each observation has the same number of coordinates.\n"
            "The metadata describes the public domain of the observations.\n"
            "Algorithms use those public bounds when estimating the mean.\n"
            "The output contains one estimate per coordinate."
        )
        assert not _task_description_failure(task)
    finally:
        task.description = original


@pytest.mark.parametrize(
    "check,fixture",
    [
        (_block_shape_failure, _BadBlockDuplicateFixture),
        (_requires_failure, _ProseRequiresFixture),
        (_requires_failure, _AnyBudgetRequiresFixture),
        (_requires_failure, _AnyDeclaredProblemRequiresFixture),
        (_requires_failure, _ConstructionRequiresFixture),
        (_requires_failure, _RefusedOfferRequiresFixture),
        (_requires_failure, _NamedInvalidRegimeRequiresFixture),
        (_requires_failure, _LongRequiresFixture),
        (_hyperparameter_field_failure, _PlainFieldHyperparameterFixture),
        (_hyperparameter_field_failure, _SubclassRedeclaresFieldWithPlainFieldFixture),
        (
            _parametrization_coverage_failure,
            _MissingHyperparameterParametrizationNotesFixture,
        ),
        (_parametrization_coverage_failure, _AbsentParametrizationNotesFixture),
        (_reference_shape_failure, _UnlocatedReferenceFixture),
        (_reference_shape_failure, _LocatorOnlyInNotesFixture),
        (_reference_shape_failure, _UnknownKeyReferenceFixture),
        (_reference_shape_failure, _ProseReferenceFixture),
    ],
)
def test_scientific_content_check_rejects_missing_or_ambiguous_content(
    check: Callable[[Any], str], fixture: type
) -> None:
    assert check(fixture)


@pytest.mark.parametrize(
    "fixture",
    [
        _ReferenceBeforeAssumptionsFixture,
        _AttributesSectionFixture,
        _UnrecognizedHeadingFixture,
        _ParametrizationOutOfOrderFixture,
    ],
)
def test_documentation_sections_have_no_local_order_or_heading_restriction(
    fixture: type,
) -> None:
    assert not _block_shape_failure(fixture)


@pytest.mark.parametrize(
    "check,fixture",
    [
        (_hyperparameter_field_failure, _AncestorWithParameterFieldFixture),
        (_requires_failure, _BulletedRequiresFixture),
        (_requires_failure, _BareRequiresFixture),
        (_requires_failure, _NoRequiresFixture),
        (_parametrization_coverage_failure, _ConformingParametrizationNotesFixture),
        (_reference_shape_failure, _LocatedReferenceFixture),
        (_reference_shape_failure, _LetteredAppendixFixture),
    ],
)
def test_scientific_content_checks_accept_documented_facts(
    check: Callable[[Any], str], fixture: type
) -> None:
    assert not check(fixture)
