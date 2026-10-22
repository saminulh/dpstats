"""The declarative list of page templates: route, ordered sections, sources.

One template per page kind. A section names exactly one source and says
whether the page requires it: a required section whose source produces nothing
stops the build, and an optional one is omitted rather than filled with prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    PROVENANCE_DIRECTORY,
)


@dataclass(frozen=True)
class Section:
    """One section of one template.

    ``columns`` names the table this section renders, in order, where the
    template fixes it. ``repeated`` marks a section that expands into one
    heading per item from its source rather than one heading of its own.
    """

    key: str
    title: str
    source: str
    required: bool = True
    columns: tuple[str, ...] = ()
    repeated: bool = False


@dataclass(frozen=True)
class TemplateSpec:
    """One page template: where its pages live, and what they contain.

    ``routes`` holds every route the template produces; ``{name}`` matches one
    path segment and ``{task}`` the build's task slugs. One template produces
    more than one route where one page kind covers more than one object: the
    component page is the Algorithm page and the Transformation page, and the area
    landing is every area's. ``authored`` names the routes an author writes
    rather than a renderer fills: the template declares the page exists and
    what it is for, and the page's text comes from the authored tree.
    """

    name: str
    title: str
    routes: tuple[str, ...]
    sections: tuple[Section, ...]
    authored: tuple[str, ...] = ()


# The three table shapes more than one template renders, named once so a
# template declares the shape rather than respelling it.
FIELD_TABLE = ("Field", "Type", "Description")
# The Roster landing's columns; the two reading columns name their budget,
# and the last one holds each row's link to its own page.
ROSTER_LANDING_TABLE = (
    "Configured algorithm",
    "Privacy interpretation",
    "Rank",
    "Headline",
    "Page",
)
RECIPE_TABLE = ("Name", "Value", "Description")
# T24's two tables: a domain's columns, and the stored reference values.
INSTANCE_COLUMNS_TABLE = ("Columns", "Kind", "Values")
STANDING_TABLE = ("Budget", "Rank", "Metric", "Headline", "Coverage")
OUTCOME_TABLE = (
    "Budget",
    "Rank",
    "Released",
    "No release",
    "Invalid regime",
    "Algorithm error",
    "Timeout",
    "Memory limit",
)
PROTOCOL_BASELINE_TABLE = (
    "Baseline",
    "Declared by",
    "Definition",
    "Applies to",
    "Role",
)
PROTOCOL_GRID_TABLE = ("Minimum", "Maximum", "Scale", "Steps")
PROTOCOL_STANDINGS_TABLE = ("Rule", "Value")
CORPUS_TABLE = ("Source", "Year", "Tasks", "Instances", "Sample sizes")

FACT_TABLE = ("Fact", "Value")

# Where the Library's learning path is published, named once so a route and the
# assembler that writes those pages cannot drift apart.
_TUTORIALS = "library/tutorials"

# An unheaded section carries an empty title: the supporting line under a
# header, a code block, or a closing note is part of the page, not a heading.
TEMPLATES: dict[str, TemplateSpec] = {
    "T1": TemplateSpec(
        name="T1",
        title="Home",
        routes=("",),
        sections=(
            Section(
                "tabs",
                "",
                "the Benchmarked tasks in navigation order",
                repeated=True,
            ),
            Section("panel_heading", "", "the task's name and its results routes"),
            Section("source", "", "whether the snapshot published these standings"),
            Section("controls", "", "the prepared snapshot's comparison policy"),
            Section("absences", "", "the metrics the snapshot cannot report", False),
            Section("metric", "", "one prepared panel set per metric"),
            Section("budget", "", "the budgets that metric's panels carry"),
            Section("chart", "", "the panel the selected budget names"),
            Section("plotted", "", "the curves that panel's chart is drawing"),
            Section(
                "rankings",
                "",
                "the panel's opening standings, without the returned-estimate share",
            ),
            Section("build_source", "", "the snapshot's own identity"),
        ),
    ),
    "T2": TemplateSpec(
        name="T2",
        title="Area landing",
        # The generated areas by name, so a landing is never read as a task
        # overview and a task is never read as an area.
        routes=("corpus/", "submit/"),
        sections=(
            # A landing opens on one sentence saying what the area is for.
            # That sentence is approved copy and none is approved yet, so the
            # section has no source and every landing omits it rather than
            # carrying a sentence a renderer wrote.
            Section("purpose", "", "approved area-purpose copy", required=False),
            Section(
                "banner",
                "",
                "the status line when the build reads no release",
                required=False,
            ),
            Section(
                "summary",
                "",
                "the sources and instances this release distributes, per task",
                required=False,
            ),
            Section(
                "sources",
                "",
                "the release's provenance bundles",
                required=False,
                columns=CORPUS_TABLE,
            ),
            Section(
                "pages",
                "",
                "the area's own pages, as this build's navigation declares them",
                required=False,
                columns=("Page", "Description"),
            ),
        ),
    ),
    "T3": TemplateSpec(
        name="T3",
        title="Task overview",
        routes=("{task}/",),
        sections=(
            Section("description", "", "Task.description"),
            Section("access", "", "BUILT_IN_TASKS['task_id']"),
            Section("quick_links", "", "the task's own pages in this build"),
            Section(
                "problem",
                "Problem formulation",
                "sanctioned docs/content/tasks/<task>/problem-formulation.md",
            ),
            Section(
                "metrics",
                "Metrics at a glance",
                "list_built_in_metrics(Task) docstring first lines",
                required=False,
            ),
        ),
    ),
    "T4": TemplateSpec(
        name="T4",
        title="Task leaderboard",
        routes=("{task}/leaderboard/",),
        sections=(
            Section(
                "introduction",
                "",
                "what the standings rank and the withheld corpus they rank on",
            ),
            Section("controls", "", "the prepared snapshot's comparison policy"),
            Section("absences", "", "the metrics the snapshot cannot report", False),
            Section("metric", "", "one prepared panel set per metric"),
            Section("budget", "", "the budgets that metric's panels carry"),
            Section("chart", "", "the panel the selected budget names"),
            Section("plotted", "", "the curves that panel's chart is drawing"),
            Section(
                "rankings",
                "",
                "the panel's opening standings, with the returned-estimate share",
            ),
            Section("build_source", "", "the snapshot's own identity"),
        ),
    ),
    "T5": TemplateSpec(
        name="T5",
        title="Explore",
        routes=("{task}/public-results/",),
        sections=(
            Section("loading", "", "loading status with retry; a no-script note"),
            Section("controls", "", "the prepared snapshot's own index"),
            Section("absences", "", "the metrics the snapshot cannot report", False),
            Section("empty", "", "approved empty-selection copy"),
            Section("metric", "", "one prepared panel set per metric"),
            Section("budget", "", "the budgets that metric's panels carry"),
            Section("chart", "", "the panel the selected budget names"),
            Section("plotted", "", "the curves that panel's chart is drawing"),
            Section("rankings", "", "the panel's roster, with its class badges"),
            Section("comparisons", "", "the two pairwise analyses over the panel"),
        ),
    ),
    "T6": TemplateSpec(
        name="T6",
        title="Task definition",
        routes=("{task}/definition/",),
        sections=(
            Section(
                "metadata",
                "Metadata",
                "Task.Metadata model fields",
                columns=FIELD_TABLE,
            ),
            Section("input", "Input", "Task.Input model fields", columns=FIELD_TABLE),
            Section(
                "output", "Output", "Task.Output model fields", columns=FIELD_TABLE
            ),
            Section(
                "notes",
                "Notes on the definition",
                "authored docs/tasks/<task>.md, declared-field sections",
                required=False,
            ),
        ),
    ),
    "T7": TemplateSpec(
        name="T7",
        title="Task native algorithms",
        routes=("{task}/algorithms/",),
        sections=(
            Section(
                "table",
                "",
                "list_built_in_algorithms(Task)",
                columns=(
                    "Algorithm",
                    "What it does",
                    "Privacy interpretation",
                    "Reference",
                ),
            ),
        ),
    ),
    "T8": TemplateSpec(
        name="T8",
        title="Task transformations",
        routes=("{task}/transformations/",),
        sections=(
            Section(
                "table",
                "",
                "the task's exported transformations, one table per group; the Task conversion group adds Hands off to",
                columns=(
                    "Transformation",
                    "What it does",
                    "Privacy interpretation",
                    "Reference",
                ),
            ),
        ),
    ),
    "T9": TemplateSpec(
        name="T9",
        title="Component page",
        routes=(
            "{task}/algorithms/{algorithm}/",
            "{task}/transformations/{transformation}/",
        ),
        sections=(
            Section("summary", "", "class docstring summary line"),
            Section(
                "description",
                "",
                "class docstring opening prose, less the sentence source states",
                required=False,
            ),
            Section(
                "glance",
                "",
                "an Algorithm's native task or a Transformation's outer and inner "
                "tasks; the declared PrivacyInterpretation value; docstring "
                "'Requires:'; docstring 'Reference:' through the bibliography",
            ),
            Section(
                "parameters",
                "Parameters",
                "an Algorithm's parameter_field declarations, a Transformation's "
                "configuration fields, each with its docstring "
                "'Parametrization notes:' entry",
                required=False,
            ),
            Section(
                "implementation_notes",
                "",
                "the class docstring's 'Implementation notes:' block",
                required=False,
            ),
            Section(
                "reference", "References", "docstring 'Reference:'", required=False
            ),
            Section(
                "usage",
                "Example",
                "import, construct and run the class over an instance the task's "
                "producer draws",
            ),
            Section("api", "API", "the class signature and public methods"),
            Section(
                "roster",
                "",
                "the task Rosters this build's results carry, counted per task",
                required=False,
            ),
        ),
    ),
    "T13": TemplateSpec(
        name="T13",
        title="User guide",
        routes=(
            "library/",
            f"{_TUTORIALS}/{{tutorial}}/",
        ),
        sections=(
            Section("title", "", "the notebook's or the page's own title"),
            Section("body", "", "the notebook's cells, or the authored page"),
        ),
        authored=("library/",),
    ),
    "T15": TemplateSpec(
        name="T15",
        title="Methodology",
        routes=("methodology/",),
        sections=(
            Section(
                "banner", "", "the release identity this build reads", required=False
            ),
            Section(
                "narrative",
                "",
                f"the release's {BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_NARRATIVE_NAME}, "
                "verbatim under its own title",
                required=False,
            ),
            Section("build_source", "", "the release identity this build reads", False),
        ),
    ),
    "T16": TemplateSpec(
        name="T16",
        title="Submit page",
        routes=("submit/{page}/",),
        sections=(
            Section("steps", "How to enter", "authored docs/submission.md"),
            Section(
                "package", "Package a method request", "authored docs/submission.md"
            ),
            Section("eligibility", "Eligibility", "the protocol's own sections"),
            Section("next", "What happens next", "the protocol's own sections"),
            Section(
                "instead",
                "Extend the library instead",
                "the Library reference this work belongs to",
                required=False,
            ),
        ),
        authored=("submit/submission/",),
    ),
    "T17": TemplateSpec(
        name="T17",
        title="Reference page",
        routes=(
            "library/reference/",
            "library/reference/{group}/",
            "library/reference/{group}/{symbol}/",
        ),
        sections=(Section("body", "", "the source this reference is generated from"),),
    ),
    "T20": TemplateSpec(
        name="T20",
        title="Source",
        routes=("corpus/{source}/",),
        sections=(
            Section(
                "header",
                "",
                f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/<source>.json",
                required=False,
            ),
            Section(
                "paper",
                "Paper",
                f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/<source>.json "
                "bibtex and doi",
                required=False,
            ),
            Section(
                "data",
                "Data",
                f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/<source>.json "
                "bibtex, data_doi, data_license, discipline and instances",
                required=False,
            ),
            Section(
                "instances",
                "Instances",
                "the release's benchmark instance registries",
                columns=("Instance", "Task", "Samples (n)", "Dimension"),
            ),
        ),
    ),
    "T18": TemplateSpec(
        name="T18",
        title="Task metrics",
        routes=("{task}/metrics/",),
        sections=(
            Section("access", "", "list_built_in_metrics(Task)"),
            Section(
                "table",
                "",
                "list_built_in_metrics(Task)",
                columns=("Metric", "Required metric arguments", "Description"),
            ),
            Section(
                "sections",
                "",
                "each metric's calculation docstring and formula, plus the "
                "prepared results surfaces' baseline catalogues",
                repeated=True,
            ),
        ),
    ),
    "T21": TemplateSpec(
        name="T21",
        title="Configured algorithm",
        routes=("{task}/roster/{uuid}/",),
        sections=(
            Section("header", "", "the release registry, series style"),
            Section(
                "description",
                "",
                "algorithm-declarations.csv 'description'",
                required=False,
            ),
            Section(
                "standing",
                "Leaderboard ranks",
                "the released evaluation view's panels",
                required=False,
                columns=STANDING_TABLE,
            ),
            Section(
                "outcomes",
                "Results breakdown on the public corpus",
                "the release's distribution-lane trial outcomes over the public corpus",
                required=False,
                columns=OUTCOME_TABLE,
            ),
            Section(
                "recipe",
                "Recipe",
                "the measured provenance's configured component tree",
                columns=RECIPE_TABLE,
            ),
            Section(
                "provenance",
                "Source",
                "algorithm-declarations.csv 'provenance_ref'",
                required=False,
            ),
        ),
    ),
    "T24": TemplateSpec(
        name="T24",
        title="Instance",
        routes=("{task}/instances/{uuid}/",),
        sections=(
            Section(
                "facts",
                "",
                f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/<source>.json "
                "instances, the registry's Metadata, the release's instance weights and the "
                "design condition number Explore measures",
            ),
            Section(
                "bounds",
                "Data bounds",
                "the registry's Metadata domains",
                required=False,
                columns=INSTANCE_COLUMNS_TABLE,
            ),
            Section(
                "source",
                "",
                f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/<source>.json "
                "instances.<uuid>.source: its variables, coefficient and claims",
                required=False,
            ),
        ),
    ),
    "T23": TemplateSpec(
        name="T23",
        title="Task methodology",
        routes=("{task}/methodology/",),
        sections=(
            Section(
                "lead",
                "",
                "the released ranking's headline Metric, headline baseline and "
                "headline threshold",
            ),
            Section(
                "standings",
                "Standings",
                "the released ranking's weighting and eligibility",
                columns=PROTOCOL_STANDINGS_TABLE,
            ),
            Section(
                "baselines",
                "Baselines",
                "every baseline of the task: each Metric's statistical baseline "
                "from list_built_in_metrics(Task) described by "
                "list_built_in_metric_arguments(Task), applying to the Metrics whose "
                "baseline it is; the protocol's external baselines with their "
                "descriptions and Metrics; and the ranking's computed baselines",
                columns=PROTOCOL_BASELINE_TABLE,
            ),
            Section(
                "grid",
                "Threshold grid",
                "the released ranking's tau_grid",
                columns=PROTOCOL_GRID_TABLE,
            ),
            Section("build_source", "", "the release identity this build reads"),
        ),
    ),
    "T22": TemplateSpec(
        name="T22",
        title="Roster landing",
        routes=("{task}/roster/",),
        sections=(
            Section(
                "table",
                "",
                "the release's Roster for the task, each row's configured component "
                "chain, and its headline standing",
                columns=ROSTER_LANDING_TABLE,
            ),
        ),
    ),
}
