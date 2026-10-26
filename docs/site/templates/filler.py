"""A registry of invented objects, so the renderers can be read on their own.

Nothing here is real: the package, the task, the Algorithm, the Transformation, the
metrics and the producers are filler, and every number and citation is
invented. Feeding these records through the same renderers a build uses
produces the pages under ``/templates/``, which are checked against the golden
files so a renderer change shows up as a page change.
"""

from __future__ import annotations

from .blocks import (
    anchor,
    code,
    code_anchor,
    display_math,
    math,
    math_name,
    reference_paragraphs,
)
from .model import (
    TRANSFORMATION,
    Access,
    AlgorithmGroup,
    AlgorithmsIndex,
    ChainStep,
    Component,
    ConfiguredAlgorithm,
    CorpusIndex,
    Definition,
    InstanceDomain,
    InstancePage,
    Link,
    ApiMethod,
    Methodology,
    Metrics,
    MetricSection,
    ModelBlock,
    OutcomeRow,
    TransformationGroup,
    TransformationRow,
    ReaderTerm,
    TransformationsIndex,
    RecipeComponent,
    Row,
    BaselineEntry,
    Parameter,
    RosterLanding,
    RosterLink,
    RosterRow,
    RosterPage,
    Source,
    StandingRow,
    TaskOverview,
)
from bibliography import CitedWork, full_citation, load, short_citation, source_text
from site_copy import PLACEHOLDER, render_slot

from .render import render
from .spec import CORPUS_TABLE, TEMPLATES

PACKAGE = "dp_filler"
TASK_MODULE = f"{PACKAGE}.tasks.task_name"
SEMANTICS = Link(
    "What this means",
    "../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation/",
)

_ESTIMAND = "The quantity this task estimates, as the Task declares it."
_FIELD_DESCRIPTION = (
    "The field's full meaning: what it holds, in what order, and what a row is."
)
_PARAMETER_DESCRIPTION = "The description declared on the parameter field."
_PARAMETER_NOTE = (
    "Corresponds to a symbol in the cited work; why its default was chosen."
)

# Every filler page: the record it reads, and its file name under
# ``/templates/``. The mock these pages reproduce is the tuned hand-authored
# one, so the order here is the templates document's order. A template that
# renders more than one kind of page appears once per kind, and the record is
# keyed by the template and the kind: the component page is an Algorithm and a
# Transformation, and the area landing is the Tasks landing and the Corpus landing.
FILLER_PAGES: tuple[tuple[str, str], ...] = (
    ("T3", "t03-task-overview"),
    ("T6", "t06-definition"),
    ("T18", "t18-task-metrics"),
    ("T7", "t07-task-algorithms"),
    ("T8", "t08-task-transformations"),
    ("T9", "t09-algorithm"),
    ("T9-transformation", "t10-transformation"),
    ("T2-corpus", "t02-corpus-landing"),
    ("T20", "t20-source"),
    ("T24", "t24-instance"),
    ("T21", "t21-configured-algorithm"),
    ("T22", "t22-roster-landing"),
    ("T15", "t15-methodology"),
)

# The filler pages sit side by side under ``/templates/``, so a link between
# two of them names its neighbour rather than a task space.
SOURCE_PAGE = "../t20-source/"
INSTANCE_PAGE = "../t24-instance/"


def _task_link(text: str) -> Link:
    return Link(text, "../t03-task-overview/")


def _task_overview() -> TaskOverview:
    return TaskOverview(
        name="Task Name",
        description=_ESTIMAND,
        access=Access(
            import_line=f"from {PACKAGE}.tasks import BUILT_IN_TASKS",
            accessor='BUILT_IN_TASKS["task_name"]',
        ),
        quick_links=(
            Link("Leaderboard", ""),
            Link("Explore", ""),
            Link("Definition", "../t06-definition/"),
            Link("Metrics", "../t18-task-metrics/"),
            Link("Native algorithms", "../t07-task-algorithms/"),
            Link("Transformations", "../t08-task-transformations/"),
        ),
        problem=(
            "The estimand in words: what quantity a release is estimating, on what\n"
            "kind of sample, and what the analyst already knows about it.\n\n"
            "A second paragraph, where the estimand needs one: the sampling\n"
            "assumption, the bounded quantities, or what the analyst is allowed to\n"
            "know in public.\n\n"
            "$$\n"
            "\\theta^{\\star}(P) \\;=\\; \\arg\\min_{\\theta \\in \\Theta}\\;\n"
            "\\mathbb{E}_{Z \\sim P}\\big[\\, \\ell(\\theta; Z) \\,\\big]\n"
            "$$\n\n"
            "What a private release must output: the object, its shape, and\n"
            "anything the output carries alongside the estimate."
        ),
        metrics=(
            (
                Link("Metric one", "../t18-task-metrics/#metric-one"),
                "What metric one measures.",
            ),
            (
                Link("Metric two", "../t18-task-metrics/#metric-two"),
                "What metric two measures.",
            ),
            (
                Link("Metric three", "../t18-task-metrics/#metric-three"),
                "What metric three measures.",
            ),
        ),
    )


def _definition() -> Definition:
    columns = ("Field", "Type", "Description")
    return Definition(
        title="Task Name: definition",
        models=(
            ModelBlock(
                title="Input",
                access=Access(f"from {TASK_MODULE} import TaskNameInput"),
                summary=(
                    "What the analyst hands the method, in plain language, and what\n"
                    "one row of it stands for."
                ),
                columns=columns,
                rows=(
                    Row(
                        "sample", code("Array"), math("n \\times d"), _FIELD_DESCRIPTION
                    ),
                    Row(
                        "response",
                        code("Array"),
                        math("n \\times 1"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "weights",
                        code("Array") + " (optional)",
                        math("w_i \\ge 0"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "seed",
                        code("int"),
                        math("0 \\le s < 2^{32}"),
                        _FIELD_DESCRIPTION,
                    ),
                ),
            ),
            ModelBlock(
                title="Metadata",
                access=Access(f"from {TASK_MODULE} import TaskNameMetadata"),
                summary=(
                    "What is public about the instance before any method runs, and\n"
                    "why a method may rely on it."
                ),
                columns=columns,
                rows=(
                    Row(
                        "sample_size",
                        code("int"),
                        math(math_name("sample_size") + " \\ge 1"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "dimension",
                        code("int"),
                        math(math_name("dimension") + " \\ge 1"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "bounds",
                        code("tuple[float, float]"),
                        math("\\ell < u"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "share_public",
                        code("float"),
                        math("0 < " + math_name("share_public") + " < 1"),
                        _FIELD_DESCRIPTION,
                    ),
                ),
            ),
            ModelBlock(
                title="Output",
                access=Access(f"from {TASK_MODULE} import TaskNameOutput"),
                summary=(
                    "What a private release returns, and what a reader may do with\nit."
                ),
                columns=columns,
                rows=(
                    Row(
                        "estimate",
                        code("Array"),
                        math("d \\times 1"),
                        _FIELD_DESCRIPTION,
                    ),
                    Row(
                        "scale", code("float"), math("\\sigma > 0"), _FIELD_DESCRIPTION
                    ),
                    Row(
                        "interval",
                        code("tuple[float, float]"),
                        math("\\ell \\le u"),
                        _FIELD_DESCRIPTION,
                    ),
                ),
            ),
        ),
        notes=(
            "Material about the declared fields themselves, such as a structured\n"
            "design domain, how a domain is supplied, or how a configurable loss is\n"
            "declared."
        ),
    )


# The invented works the mock pages cite, rendered by the site's one style.
_BIBLIOGRAPHY = load(
    """
@misc{surname2019title,
  author = {Surname, Ann B.},
  title = {Title of the cited paper in sentence case},
  howpublished = {Venue Name},
  year = {2019},
  doi = {10.0000/000000.0000000}
}
@misc{surname2020title,
  author = {Surname, Ann B. and Othersurname, Cy},
  title = {Title of the second cited paper},
  howpublished = {Venue Name},
  year = {2020},
  eprint = {0000.00000},
  archivePrefix = {arXiv}
}
@misc{surname2021title,
  author = {Surname, Ann B. and Othersurname, Cy and Thirdsurname, Di
            and Fourthsurname, Ed and Fifthsurname, Flo},
  title = {Title of a paper with five authors},
  howpublished = {Venue Name},
  year = {2021},
  eprint = {0000.00001},
  archivePrefix = {arXiv}
}
"""
)

_ALGORITHM_REFERENCES = {
    "FillerAlgorithm": "surname2019title",
    "SecondFillerAlgorithm": "surname2021title",
    "ThirdFillerAlgorithm": "surname2020title",
    "FourthFillerAlgorithm": "",
    "FifthFillerAlgorithm": "",
}


def _citation(key: str) -> str:
    return short_citation(key, _BIBLIOGRAPHY) if key else ""


_ALGORITHM_CLASSES = {
    "FillerAlgorithm": "theoretical",
    "SecondFillerAlgorithm": "theoretical",
    "ThirdFillerAlgorithm": "informal",
    "FourthFillerAlgorithm": "diagnostic",
    "FifthFillerAlgorithm": "nonprivate",
}


# The filter label's help, as the maintainer's term file would supply it.
CLASS_HELP = ReaderTerm(
    "privacy_interpretation", "What a privacy interpretation means, in one sentence."
)


def _algorithm_row(name: str) -> tuple[Link, str, str, str]:
    return (
        Link(name, "../t09-algorithm/"),
        "What the Algorithm does, in one line.",
        _ALGORITHM_CLASSES[name],
        _citation(_ALGORITHM_REFERENCES[name]),
    )


def _algorithms_index() -> AlgorithmsIndex:
    return AlgorithmsIndex(
        title="Task Name: native algorithms",
        classes=("theoretical", "informal", "diagnostic", "nonprivate"),
        groups=(
            AlgorithmGroup(
                title="Mechanism Family",
                rows=tuple(
                    _algorithm_row(name)
                    for name in (
                        "FillerAlgorithm",
                        "SecondFillerAlgorithm",
                        "ThirdFillerAlgorithm",
                    )
                ),
            ),
            AlgorithmGroup(
                title="Nonprivate / diagnostic",
                rows=tuple(
                    _algorithm_row(name)
                    for name in ("FourthFillerAlgorithm", "FifthFillerAlgorithm")
                ),
            ),
        ),
        class_help=CLASS_HELP,
    )


_TRANSFORMATION_REFERENCES = {
    "FillerTransformation": _ALGORITHM_REFERENCES["FillerAlgorithm"],
    "SecondFillerTransformation": _ALGORITHM_REFERENCES["SecondFillerAlgorithm"],
    "ThirdFillerTransformation": "",
    "FourthFillerTransformation": _ALGORITHM_REFERENCES["ThirdFillerAlgorithm"],
}

_TRANSFORMATION_INNER = {
    "FillerTransformation": "Inner Task Name",
    "SecondFillerTransformation": "Inner Task Name",
    "ThirdFillerTransformation": "Another Task Name",
    "FourthFillerTransformation": "Another Task Name",
}


_TRANSFORMATION_CLASSES = {
    "FillerTransformation": "theoretical",
    "SecondFillerTransformation": "standard",
    "ThirdFillerTransformation": "standard",
    "FourthFillerTransformation": "diagnostic",
}


def _transformation_row(name: str) -> TransformationRow:
    return TransformationRow(
        name=Link(name, "../t10-transformation/"),
        description="What the Transformation does, in one line.",
        privacy_interpretation=_TRANSFORMATION_CLASSES[name],
        inner=_task_link(_TRANSFORMATION_INNER[name]),
        reference=_citation(_TRANSFORMATION_REFERENCES[name]),
    )


def _transformations_index() -> TransformationsIndex:
    return TransformationsIndex(
        title="Task Name: transformations",
        groups=(
            TransformationGroup(
                title="Task conversion",
                summary="What the transformations in this group do, in one line.",
                rows=(
                    _transformation_row("SecondFillerTransformation"),
                    _transformation_row("FillerTransformation"),
                ),
                converts=True,
            ),
            TransformationGroup(
                title="Group Name",
                summary="What the transformations in this group do, in one line.",
                rows=(
                    _transformation_row("ThirdFillerTransformation"),
                    _transformation_row("FourthFillerTransformation"),
                ),
            ),
        ),
        classes=("standard", "theoretical", "diagnostic"),
        class_help=CLASS_HELP,
    )


def algorithm_component() -> Component:
    """The invented Algorithm the component page renders as its first kind."""
    works = (
        CitedWork("surname2021title", "Algorithm 2"),
        CitedWork("surname2019title", "Theorem 3.1"),
    )
    return Component(
        name="FillerAlgorithm",
        task="[Task Name](../t03-task-overview/)",
        summary="What the Algorithm does, in one line.",
        description=(
            "What the Algorithm does, in the terms of this task's estimand. It says\n"
            "what the method is, not why a reader should choose it."
        ),
        privacy_interpretation="theoretical",
        semantics_link=SEMANTICS,
        source=source_text(list(works), _BIBLIOGRAPHY),
        reference=reference_paragraphs(works, _BIBLIOGRAPHY, locators=False),
        requires=(
            "A finite bound the class reads from the metadata.",
            "A budget family the class accepts.",
            "A dimension or sample-size condition.",
        ),
        parameters=(
            Parameter(
                "clip_norm",
                "float &gt; 0",
                "1.0",
                _PARAMETER_DESCRIPTION,
                _PARAMETER_NOTE,
            ),
            Parameter("step_count", "int ≥ 1", "64", _PARAMETER_DESCRIPTION),
            Parameter(
                "split_share",
                "float in (0, 1)",
                "0.5",
                _PARAMETER_DESCRIPTION,
                _PARAMETER_NOTE,
            ),
            Parameter(
                "shrinkage", "float in [0, 1] or None", "None", _PARAMETER_DESCRIPTION
            ),
            Parameter(
                "tail_rule",
                "&#x27;trim&#x27; or &#x27;winsorize&#x27;",
                "&#x27;trim&#x27;",
                _PARAMETER_DESCRIPTION,
            ),
        ),
        roster=(
            RosterLink("Task Name", 12, "../t22-roster-landing/?core=FillerAlgorithm"),
            RosterLink(
                "Second Task Name", 3, "../t22-roster-landing/?core=FillerAlgorithm"
            ),
        ),
        implementation_notes=(
            "Anything a reader of the code needs and would not guess: a numerical\n"
            "guard, an ordering that matters, a representation choice.\n\n"
            "Deviations from the cited paper:\n\n"
            "- One modification, stated as what the implementation does instead.\n"
            "- A second modification, one bullet each.\n"
        ),
        usage=(
            f"from {TASK_MODULE} import FillerAlgorithm\n\n"
            "algorithm = FillerAlgorithm(clip_norm=1.0, step_count=64)\n"
            "output = algorithm.run(instance.input, instance.metadata, budget)"
        ),
        api_signature=(
            "class FillerAlgorithm(\n"
            "    clip_norm: float = 1.0,\n"
            "    step_count: int = 64,\n"
            "    split_share: float = 0.5,\n"
            "    shrinkage: float | None = None,\n"
            '    tail_rule: str = "trim",\n'
            ")"
        ),
        api_methods=(
            ApiMethod(
                name="run",
                signature=(
                    "run(input: TaskNameInput, metadata: TaskNameMetadata, "
                    "budget: Budget) -> TaskNameOutput"
                ),
                summary="The method's docstring first paragraph.",
                parameters=(
                    Row(
                        "input",
                        code("TaskNameInput"),
                        description="The parameter description.",
                    ),
                    Row(
                        "metadata",
                        code("TaskNameMetadata"),
                        description="The parameter description.",
                    ),
                    Row(
                        "budget",
                        code("Budget"),
                        description="The parameter description.",
                    ),
                ),
            ),
            ApiMethod(
                name="applies_to",
                signature="applies_to(metadata: TaskNameMetadata, budget: Budget) -> None",
                summary=(
                    "The method's docstring first paragraph, including what it raises."
                ),
            ),
        ),
    )


def transformation_component() -> Component:
    """The invented Transformation the same component page renders as its second."""
    works = (CitedWork("surname2020title", "Section 4"),)
    return Component(
        name="FillerTransformation",
        task=(
            "[Task Name](../t03-task-overview/) → "
            "[Inner Task Name](../t03-task-overview/)"
        ),
        summary="What the Transformation does, in one line.",
        description=(
            "What the Transformation transports, and what the inner task then solves. It\n"
            "says what the reduction is, not why a reader should choose it."
        ),
        kind=TRANSFORMATION,
        privacy_interpretation="theoretical",
        semantics_link=SEMANTICS,
        source=source_text(list(works), _BIBLIOGRAPHY),
        reference=reference_paragraphs(works, _BIBLIOGRAPHY, locators=False),
        requires=(
            "A fact the outer task must declare for the transport to be defined; "
            "otherwise <code>NotApplicable</code>.",
            "A budget family the Transformation accepts.",
        ),
        parameters=(
            Parameter(
                "inner_share",
                "float in (0, 1)",
                "0.5",
                "The description declared on the field.",
                _PARAMETER_NOTE,
            ),
            Parameter(
                "centre", "bool", "True", "The description declared on the field."
            ),
            Parameter(
                "residual_rule",
                "str",
                "&#x27;pooled&#x27;",
                "The description declared on the field.",
            ),
        ),
        roster=(
            RosterLink(
                "Task Name",
                5,
                "../t22-roster-landing/?transformation=FillerTransformation",
            ),
        ),
        implementation_notes=(
            "What the transformation does to the estimand, what it assumes, and the\n"
            "case a reader is most likely to get wrong.\n\n"
            "The authored method note moved here from the task guide, where a\n"
            "Transformation's reduction was explained at length.\n\n"
            "Deviations from the cited paper:\n\n"
            "- One modification, one bullet.\n"
        ),
        usage=(
            f"from {TASK_MODULE} import FillerTransformation\n\n"
            "transformation = FillerTransformation(inner_share=0.5)\n"
            "output = transformation.run(instance.input, instance.metadata, budget, "
            "inner=inner_algorithm)"
        ),
        api_signature=(
            "class FillerTransformation(\n"
            "    inner_share: float = 0.5,\n"
            "    centre: bool = True,\n"
            '    residual_rule: str = "pooled",\n'
            ")"
        ),
        api_methods=(
            ApiMethod(
                name="run",
                signature=(
                    "run(input: TaskNameInput, metadata: TaskNameMetadata, "
                    "budget: Budget, inner: Algorithm) -> TaskNameOutput"
                ),
                summary="The method's docstring first paragraph.",
                parameters=(
                    Row(
                        "input",
                        code("TaskNameInput"),
                        description="The parameter description.",
                    ),
                    Row(
                        "metadata",
                        code("TaskNameMetadata"),
                        description="The parameter description.",
                    ),
                    Row(
                        "budget",
                        code("Budget"),
                        description="The parameter description.",
                    ),
                    Row(
                        "inner",
                        code("Algorithm"),
                        description="The parameter description.",
                    ),
                ),
            ),
            ApiMethod(
                name="transported_metadata",
                signature=(
                    "transported_metadata(metadata: TaskNameMetadata) -> "
                    "InnerTaskMetadata"
                ),
                summary="The method's docstring first paragraph.",
            ),
        ),
    )


def _metrics() -> Metrics:
    access = Access(
        import_line=f"from {PACKAGE} import list_built_in_metrics",
        accessor="list_built_in_metrics(TaskName)",
    )

    def section(
        name: str, key: str, needs: tuple[str, ...], formula: str
    ) -> MetricSection:
        return MetricSection(
            name=name,
            key=key,
            anchor=name.lower().replace(" ", "-"),
            docstring=(
                f"What {name.lower()} measures, over what, and how a trial that "
                "produces no estimate is treated."
            ),
            formula=formula,
            needs=needs,
        )

    return Metrics(
        title="Task Name: metrics",
        access=access,
        metrics=(
            section(
                "Metric one",
                "metric_one",
                (),
                display_math(
                    "M_1 \\;=\\; \\frac{1}{S}\\sum_{s=1}^{S}\n"
                    "\\lVert \\hat{\\theta}_s - \\theta^{\\star} \\rVert"
                ),
            ),
            section(
                "Metric two",
                "metric_two",
                ("sample", "response"),
                display_math(
                    "M_2 \\;=\\; \\frac{1}{n}\\sum_{i=1}^{n}\n"
                    "\\ell\\!\\left( \\hat{\\theta}; Z_i \\right)"
                ),
            ),
            section("Metric three", "metric_three", ("bounds",), ""),
            section(
                "Metric four",
                "metric_four",
                ("sample", "bounds"),
                display_math(
                    "M_4 \\;=\\; \\frac{1}{S}\\sum_{s=1}^{S}\n"
                    "\\min\\!\\left\\{\\, 1,\\;\n"
                    "\\frac{\\lVert \\hat{\\theta}_s - \\theta^{\\star} \\rVert}"
                    "{\\tau(\\text{bounds})}\n\\,\\right\\}"
                ),
            ),
        ),
        arguments_intro=render_slot("task", "metric_arguments_intro", PLACEHOLDER),
        baselines=(
            BaselineEntry(
                "filler_scale",
                "Benchmark protocol",
                "The description the protocol states for a value the source supplies.",
                '<span class="dp-sentence"><a href="#metric-one">metric_one</a></span>',
                "Headline baseline",
            ),
            BaselineEntry(
                "filler_reference",
                "Library",
                "The description the library's metric argument registry declares.",
                '<span class="dp-sentence"><a href="#metric-one">metric_one</a>, '
                '<a href="#metric-two">metric_two</a></span>',
                "Offered on Explore",
            ),
            BaselineEntry(
                "best_median",
                "Benchmark protocol",
                "The best median cost among the standard-class algorithms, "
                "per instance and budget.",
                '<span class="dp-sentence"><a href="#metric-one">metric_one</a>, '
                '<a href="#metric-two">metric_two</a></span>',
                "Offered on Explore",
            ),
        ),
    )


def _corpus_index() -> CorpusIndex:
    fields = ("demography", "genomics")
    return CorpusIndex(
        title="Corpus",
        banner="",
        summary=(
            "3 sources and 12 problem instances. Task Name: 7 instances. "
            "Second Task Name: 5 instances."
        ),
        tasks=(("task_name", "Task Name"), ("second_task_name", "Second Task Name")),
        fields=fields,
        columns=CORPUS_TABLE,
        rows=(
            (
                anchor("A Filler Study of Something", SOURCE_PAGE)
                + '<div class="text-secondary small mt-1">A study of a public '
                "outcome across two populations.</div>",
                "2019",
                "Task Name, Second Task Name",
                "7",
                "246–18,430",
            ),
            (
                anchor("A Second Filler Study", SOURCE_PAGE)
                + '<div class="text-secondary small mt-1">A replication using '
                "a second public sample.</div>",
                "2021",
                "Task Name",
                "3",
                "1,204",
            ),
            (
                anchor("A Third Filler Study", SOURCE_PAGE),
                "",
                "Second Task Name",
                "2",
                "640–2,000",
            ),
        ),
        filters=(
            {
                "data-tasks": "second_task_name task_name",
                "data-fields": " ".join(fields),
                "data-search": "a filler study of something ada filler bo filler "
                "journal of filler studies 2019 a study of a public outcome "
                "across two populations",
            },
            {
                "data-tasks": "task_name",
                "data-fields": "demography",
                "data-search": "a second filler study cy filler 2021 a replication "
                "using a second public sample",
            },
            {
                "data-tasks": "second_task_name",
                "data-fields": "genomics",
                "data-search": "a third filler study dee filler",
            },
        ),
    )


def _instance() -> InstancePage:
    return InstancePage(
        title="filler-study-01/full",
        facts=(
            ("Study", anchor("A Filler Study of Something", SOURCE_PAGE)),
            ("Field", "demography"),
            ("Data licence", "CC0 1.0"),
            ("Sample unit", "individual"),
            ("Samples (n)", "1,204"),
            ("Dimension", "6"),
            ("Samples per dimension", "200.7"),
            ("Design condition number κ(X)", "37.42"),
            ("Coefficient of interest", "Column 2"),
            ("Weight in the success rate", "0.427%"),
        ),
        domains=(
            InstanceDomain(
                "Covariates (X)",
                "intercept, 1 indicator group (3 columns), 2 range columns",
                (
                    ("1", "Intercept", "1"),
                    ("2", "Range · coefficient of interest", "[0, 1]"),
                    ("3–5", "Indicator group", "at most one is 1"),
                    ("6", "Range", "[20, 70]"),
                ),
            ),
            InstanceDomain("Response (y)", "[0, ∞)", (("1", "Range", "[0, ∞)"),)),
        ),
        variables=("filler_y", "filler_x", "filler_group"),
        coefficient="filler_x",
        claim=(
            "A filler sentence the paper states about its estimate.",
            "12",
            "Table 2",
        ),
    )


def _source() -> Source:
    return Source(
        title="A Filler Study of Something",
        supporting="Ada Filler, Bo Filler · Journal of Filler Studies · 2019",
        paper=full_citation("surname2019title", _BIBLIOGRAPHY),
        paper_doi="https://doi.org/10.0000/filler",
        data_doi="https://doi.org/10.0000/filler-data",
        facts=(
            ("Data licence", "CC0 1.0"),
            ("Sample unit", "individual"),
            ("Field", "demography; genomics"),
        ),
        columns=("Instance", "Task", "Samples (n)", "Dimension"),
        rows=(
            (
                code_anchor("filler-study-01", INSTANCE_PAGE),
                "Task Name",
                "1,204",
                "6",
            ),
            (
                code_anchor("filler-study-02", INSTANCE_PAGE),
                "Second Task Name",
                "18,430",
                "3",
            ),
        ),
    )


CONFIGURED_ALGORITHM_PAGE = "../t21-configured-algorithm/"


def _filler_row(
    uuid: str,
    chain: tuple[ChainStep, ...],
    privacy_interpretation: str,
    rank: int | None,
    tau: float | None,
) -> RosterRow:
    label = " + ".join(
        step.name + (f" ({step.hyperparameters})" if step.hyperparameters else "")
        for step in chain
    )
    return RosterRow(
        algorithm=ConfiguredAlgorithm(
            uuid=uuid,
            label=label,
            core=chain[-1].component,
            transformations=tuple(step.component for step in chain[:-1]),
            privacy_interpretation=privacy_interpretation,
            marker="circle",
            rank=rank,
            url=CONFIGURED_ALGORITHM_PAGE,
        ),
        chain=chain,
        headline=f"{tau:g}" if tau is not None else "",
        tau=tau,
    )


def roster_landing_page() -> RosterLanding:
    """The invented T22 Roster landing record."""
    algorithm = ChainStep("FillerAlgorithm", "FillerAlgorithm", "../t09-algorithm/")
    transformation = ChainStep(
        "FillerTransformation",
        "FillerTransformation",
        "../t10-transformation/",
        key="FillerTransformation",
    )
    tuned = ChainStep(
        "FillerAlgorithm",
        "FillerAlgorithm",
        "../t09-algorithm/",
        hyperparameters="clip_norm=2.0",
    )
    return RosterLanding(
        title="Task Name: roster",
        rows=(
            _filler_row("a", (algorithm,), "standard", 1, 1.42),
            _filler_row("b", (transformation, tuned), "standard", 2, 3.1),
            _filler_row("d", (transformation, algorithm), "qualified", None, None),
        ),
        classes=("standard", "qualified"),
        transformations=(("FillerTransformation", "FillerTransformation"),),
        metric="metric_one",
        budget="ε = 1.0",
        percent="90",
    )


def configured_algorithm_page() -> RosterPage:
    """The invented T21 configured-algorithm record."""
    return RosterPage(
        algorithm=ConfiguredAlgorithm(
            uuid="a",
            label="Filler Method A",
            core="FillerAlgorithm",
            transformations=(),
            privacy_interpretation="standard",
            marker="circle",
            rank=1,
        ),
        task_link=_task_link("Task Name"),
        description="What the release's own declaration says this configuration does.",
        # Anchors of the T4 preview, which renders the filler results' task
        # (MeanEstimation) and metrics (filler_results.METRICS).
        standings=(
            StandingRow(
                metric=Link(
                    "Metric one",
                    "../t04-leaderboard/"
                    "#profile-mean-estimation-l2-error-to-ground-truth",
                ),
                budget="ε = 1.0",
                rank="Rank 1",
                headline="1.42",
                coverage="90.0%",
            ),
            StandingRow(
                metric=Link(
                    "Metric two",
                    "../t04-leaderboard/"
                    "#profile-mean-estimation-l2-error-to-sample-mean",
                ),
                budget="ε = 1.0",
                rank="not ranked",
                headline="—",
                coverage="—",
            ),
        ),
        outcomes=(
            OutcomeRow(
                budget="ε = 1.0",
                rank="Rank 1",
                success="82 (91.1%)",
                no_release="0",
                invalid_regime="8 (8.9%)",
                algorithm_error="0",
                timeout="0",
                memory_limit_exceeded="0",
            ),
        ),
        recipe=(
            RecipeComponent(
                name=Link("FillerTransformation", "../t10-transformation/"),
                values=(("axis", "'row'", "Which axis the transform reduces."),),
                children=(),
            ),
            RecipeComponent(
                name=Link("FillerAlgorithm", "../t09-algorithm/"),
                values=(
                    ("probability", "0.5", "The parameter the search set opens at."),
                ),
                children=(
                    RecipeComponent(
                        name=Link("FillerLoss", ""),
                        values=(("k", "1.0", "The loss family's own parameter."),),
                        children=(),
                    ),
                ),
            ),
        ),
        percent="90",
        leaderboard_url="../t04-leaderboard/",
        explore_url="../t05-explore/",
        class_url="../t09-algorithm/",
        configuration_json='{\n  "path": "dp_filler.algorithms.FillerAlgorithm"\n}',
        provenance=(
            "### A Filler Recipe Source\n\n"
            "Recipe source: Ada Filler, [A Filler Paper](https://doi.org/10.0000/filler), "
            "Algorithm 1."
        ),
    )


# A release's protocol narrative, invented: headings, inline and display math,
# a list and a table, as a narrative may carry them.
FILLER_NARRATIVE = """# Filler protocol

The release's own account of its benchmark, rendered as written.

## 1. Filler section

Each instance runs at privacy level $\\varepsilon$, with $\\delta$ set from $n$:

$$
\\delta = \\min(n^{-1.1}, 0.01).
$$

## 2. Another filler section

- A first filler item.
- A second filler item, reading $\\tau$ against the baseline.
"""


def methodology_page() -> Methodology:
    """The invented T15 site-wide Methodology record."""
    return Methodology(
        title="Filler protocol",
        narrative=FILLER_NARRATIVE.partition("\n")[2].lstrip("\n"),
        identity="Distribution Release Filler",
    )


def filler_registry() -> dict[str, object]:
    """Return one invented page record per preview page.

    Returns:
        Each preview page mapped to the record its renderer reads, keyed by its
        template and, where one template renders more than one kind of page,
        the kind of page as well.

    Example:
        >>> sorted(filler_registry())[:3]
        ['T18', 'T2-corpus', 'T20']
    """
    return {
        "T3": _task_overview(),
        "T6": _definition(),
        "T7": _algorithms_index(),
        "T8": _transformations_index(),
        "T9": algorithm_component(),
        "T9-transformation": transformation_component(),
        "T18": _metrics(),
        "T2-corpus": _corpus_index(),
        "T20": _source(),
        "T24": _instance(),
        "T21": configured_algorithm_page(),
        "T22": roster_landing_page(),
        "T15": methodology_page(),
    }


def mock_pages() -> dict[str, str]:
    """Render every filler record through its template.

    Returns:
        Each page's file name under ``templates/`` mapped to its Markdown.

    Example:
        >>> mock_pages()["t08-task-transformations"].splitlines()[2]
        '# Task Name: transformations'
    """
    registry = filler_registry()
    return {
        filename: render(TEMPLATES[name.partition("-")[0]], registry[name])
        for name, filename in FILLER_PAGES
    }
