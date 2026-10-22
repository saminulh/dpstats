"""Generate the benchmark site and its root task spaces.

The home shows benchmark standings. Tasks own their results, definitions,
Algorithms, Transformations, and instances. The User guide teaches workflows;
API reference documents interfaces and privacy guarantees. Corpus and Submit
carry the remaining benchmark pages.

The library package and built-in Tasks are required build inputs: an import
failure stops the canonical build instead of publishing stale task navigation, and
so is the committed release lock. Result pages consume only the supplied release or
local collection roots; local collections publish nothing.
"""

from __future__ import annotations

import html
import json
import os
import posixpath
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from mkdocs_gen_files.editor import FilesEditor
from dpstats.tasks import (
    BUILT_IN_TASKS,
    list_built_in_algorithms,
    list_built_in_metrics,
)
from dpstats.workspace.layout import (
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
)
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.validators import is_immutable_revision
from dpstats.workspace.fetch import DISTRIBUTION_RELEASE_REPOSITORY

SITE_DIR = Path(__file__).resolve().parent  # docs/site/
DOCS_ROOT = SITE_DIR.parent  # docs/ — the authored library docs

# This build's own modules sit beside this script rather than in the library.
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))
from algorithm_provenance import PROVENANCE_NAME, link_provenance  # noqa: E402
from api_reference import reference_pages, rewrite_links  # noqa: E402
from area_index import area_pages, branch_for, page_summary  # noqa: E402
from claims import (  # noqa: E402
    INTERACTIVE_WORKFLOW,
    documented_task_ids,
    navigation_targets,
    nav_source,
    shared_reference_groups,
)
from cli_reference import cli_reference  # noqa: E402
from panel_data import DESIGN_CONDITION, INDEX_NAME, TaskData  # noqa: E402
from leaderboard import collection_standings, leaderboard_body  # noqa: E402
from playground import playground_page  # noqa: E402
from chart_style import chart_style, profile_series  # noqa: E402
from benchmark_lock import LOCK_NAME, read_benchmark_lock  # noqa: E402
from skipped_rows import banner, banner_html  # noqa: E402
from source_links import download_notebook, notebook_links, published_page  # noqa: E402
from prepare_results import (  # noqa: E402
    CACHE_DIR,
    prepared_snapshot,
    read_explored,
    read_instance_arguments,
    read_prepared,
    read_removed,
    read_skipped,
    write_build_summary,
)
from tutorials import (  # noqa: E402
    LOCAL_EXECUTED_DIRNAME,
    TUTORIALS_DIRNAME,
    UNRUN_NOTICE_CLASS,
    companion_scripts,
    executed_directory,
    executed_notebook,
    tutorial_sources,
    tutorials_index,
)
from notebook_headings import FIGURE_CLASS, FIGURE_ID_PREFIX  # noqa: E402
from profile_chart import (  # noqa: E402
    OPENING_TAU,
    privacy_interpretation_href,
    RenderedPage,
    absence_body,
    budget_short_label,
    default_metric,
    embed,
    runtime_scripts,
)
from recreate_notebook import (  # noqa: E402
    NOTEBOOK_NAMES,
    OUTCOMES_VIEW,
    ReleaseIdentity,
    Selection,
    notebook_bytes,
)
from results_roots import (  # noqa: E402
    CURRENT_PROTOCOL_ENV,
    LEADERBOARD_SURFACE,
    LOCAL_RELEASE_ENV,
    RESULTS_SURFACES,
    ReleaseInput,
    ReleasedComparison,
    ResultsRoute,
    read_comparison,
    release_input,
    results_routes,
)
from task_reference import (  # noqa: E402
    exported_transformations,
    outer_tasks,
    task_statuses,
)
from templates import corpus_source, filler_results, library_source  # noqa: E402
from templates.library_source import CLASS_ORDER, PACKAGE  # noqa: E402
from templates.roster import (  # noqa: E402
    component_links,
    eligible_rows,
    headline_readings,
    headline_panel,
    roster_entries,
    roster_rows,
)
from templates.model import (  # noqa: E402
    AreaIndex,
    Link,
    Protocol,
    RosterLanding,
    RosterLink,
)
from templates.blocks import link, markdown_table  # noqa: E402
from templates.render import MissingSource, TASK_PAGE_MARKER, render  # noqa: E402
from templates.conformance import NAV_FILE, refuse_untemplated  # noqa: E402
from templates.spec import TEMPLATES  # noqa: E402
from protocol_page import protocol_page  # noqa: E402
from methodology import (  # noqa: E402
    METHODOLOGY_ROUTE,
    methodology_page,
    narrative_title,
    read_narrative,
    release_relative_links,
    tuning_fragment,
)
from home_context import home_context  # noqa: E402
from site_copy import copy_attribute, copy_html, copy_span, copy_text  # noqa: E402
from instance_pages import (  # noqa: E402
    INSTANCE_CARDS_NAME,
    instance_card,
    instance_page,
)
from roster_pages import roster_page  # noqa: E402

from dpstats.task import Task  # noqa: E402

# The inline heading is refused once prose has its own sanctioned source.
PROBLEM_SECTION = "Problem Formulation"


def _resolve(env_var: str, default: Path) -> Path:
    """A build input — from an env override (CI) or the in-repo default (local)."""
    val = os.environ.get(env_var)
    return Path(val).resolve() if val else default


# Inputs default to this repo's one-folder layout (so local `mkdocs serve` works);
# CI overrides each — the library docs, renderer, and fetched release can all
# live elsewhere.
LIB_DOCS = _resolve("DOCS_LIB_DOCS_DIR", DOCS_ROOT)
# The tutorial notebooks sit at the repository's top level, beside docs/.
TUTORIALS_ROOT = LIB_DOCS.parent / TUTORIALS_DIRNAME

# Where the rendered filler pages are served, one per template, and the build
# flag that asks for them. They are a design preview of the page templates, not
# documentation, so a build carries them only when it says so.
TEMPLATES_DIRNAME = "templates"
SERVE_TEMPLATES_ENV = "DOCS_SERVE_TEMPLATES"

# Where the Corpus area's pages are published: its index, and one page per
# source the supplied Distribution Release distributes.
CORPUS_DIRNAME = "corpus"

# Each installed built-in Task with an authored concept or reference page becomes a
# clickable directory. The built-in Task facade owns IDs, titles, descriptions,
# and ordering (alphabetical by its current title); this file owns only assembly.
_DOCUMENTED = documented_task_ids(LIB_DOCS, BUILT_IN_TASKS)
TASKS = [
    (task.id, task.name, task.description)
    for task in sorted(BUILT_IN_TASKS.values(), key=lambda task: task.name)
    if task.id in _DOCUMENTED
]
TASK_SLUGS = [t[0] for t in TASKS]
REFERENCE_TASK = "single_coefficient_estimation"

# Where each task's own Algorithms also run, read once for navigation and
# every task space.
REACHABLE = outer_tasks(TASK_SLUGS)

# Authored pages selected by navigation publish under ``library/`` unless
# ``claims.RELOCATED_PAGES`` moves them. Library reference pages sit under
# ``library/reference/`` beside the generated API sections.

# The Library reference pages this build generates rather than copies: the hub
# over the API sections, and the command line as its own parsers print it.
REFERENCE_DIRNAME = "library/reference"
CLI_PAGE = f"{REFERENCE_DIRNAME}/cli.md"

# The areas whose landing is the list of that area's own pages. Tasks and
# Corpus list their subjects instead, from the registry and the release.
PAGE_LIST_AREAS = ("submit",)

# The site-wide Contact page: the addresses the benchmark's readers write to.
CONTACT_ROUTE = "contact/"
# The Corpus area's page on where its data comes from and what a reader may
# do with it, as the maintainer wrote it in ``docs/content/corpus-about.md``.
CORPUS_ABOUT = "about"
# The data page's own title, which links to it name it by.
DATA_PAGE_TITLE = "About the corpus"

# Each page this build wrote, and its opening sentence where it has one. An
# area landing names its pages with the sentence each page opens on, so the
# description on the landing is the page's own text rather than new copy.
SUMMARIES: dict[str, str] = {}


def _publish_links(text: str, source: str) -> str:
    """Repoint one authored page's relative links at their published homes.

    Authored pages link each other by their ``docs/`` paths. Task prose becomes
    its task space's guide, the SCE walkthrough becomes a Library tutorial, and
    the pages named in :data:`RELOCATED_PAGES` leave Library, so every relative
    link is resolved against ``docs/`` and re-expressed relative to the
    published page.
    """
    page = published_page(source)
    here = posixpath.dirname(page)
    directory = posixpath.dirname(source)

    def replace(match: re.Match[str]) -> str:
        link = match[1]
        if "://" in link or link.startswith(("/", "#")):
            return match[0]
        path, marker, fragment = link.partition("#")
        if not path.endswith((".md", ".ipynb")):
            return match[0]
        target = posixpath.normpath(posixpath.join(directory, path))
        moved = posixpath.relpath(published_page(target, fragment=fragment), here)
        return "](" + moved + (marker + fragment if marker else "") + ")"

    return re.sub(r"\]\(([^\s)]+)\)", replace, text)


def write_library() -> None:
    """Publish the authored pages selected by navigation, plus generated API."""
    if not LIB_DOCS.is_dir():
        return
    for target in sorted(NAV_TARGETS):
        source = nav_source(target)
        src = LIB_DOCS / source
        if not src.is_file() or src.suffix != ".md":
            continue
        if source.startswith(f"{TUTORIALS_DIRNAME}/"):
            continue  # → write_tutorials
        page = published_page(source)
        text = rewrite_links(
            _publish_links(src.read_text(), source),
            page,
            API_GROUPS,
            API_ROUTES,
        )
        _record_summary(page, text)
        with mkdocs_gen_files.open(page, "w") as f:
            f.write(text)
    for path in API_PAGES:
        _write_api_page(path)
    _write_page(CLI_PAGE, cli_reference())
    write_reference_index()


def _write_api_page(path: str) -> None:
    """Publish API directories and searchable object pages with their navigation."""
    text = API_PAGES[path]
    if path.startswith("library/reference/"):
        text = _publish_links(text, path.removeprefix("library/"))
    if path == "library/reference/algorithm/index.md":
        rows = [
            (link(task.name, f"../../../{slug}/transformations/index.md"),)
            for slug, task in BUILT_IN_TASKS.items()
            if exported_transformations(slug)
        ]
        text += "\n" + markdown_table(
            (copy_span("reference", "task_transformations_column"),), rows
        )
    _record_summary(path, text)
    with mkdocs_gen_files.open(path, "w") as page:
        page.write(text)


def write_templates() -> None:
    """Serve the tuned mock pages the renderers reproduce, under ``/templates/``.

    These pages are invented objects, not documentation: every name, number and
    citation on them is filler. A public build must not carry them, so they are
    written only when the build asks for them through
    :data:`SERVE_TEMPLATES_ENV`, which a design preview sets and a published
    build never does. When they are written they are kept out of the search
    index here and out of the sitemap by the theme override.

    The golden files are the checked-in copy of those pages; writing them here
    rather than rendering again keeps the served page and the file a test
    compares byte for byte identical.
    """
    if os.environ.get(SERVE_TEMPLATES_ENV, "") not in ("1", "true", "True"):
        return
    golden = SITE_DIR / "templates" / "golden"
    names = sorted(path.stem for path in golden.glob("*.md"))
    index = [
        "# Page templates",
        "",
        "Every page template, rendered from a registry of invented objects. No "
        "name, number or citation on these pages is real.",
        "",
    ]
    for name in names:
        _write_page(
            f"{TEMPLATES_DIRNAME}/{name}.md",
            (golden / f"{name}.md").read_text(encoding="utf-8"),
        )
    results = write_results_templates()
    entries = sorted([*((name, f"{name}.md") for name in names), *results])
    index += [f"- [{name}]({target})" for name, target in entries]
    _write_page(f"{TEMPLATES_DIRNAME}/index.md", "\n".join(index) + "\n")


# Where a results preview's baseline Definition links and its introduction's
# Roster link land: the filler pages beside them, since the preview publishes
# no task's own.
METHODOLOGY_PREVIEW = "../t23-task-methodology/"
ROSTER_PREVIEW = "../t22-roster-landing/"


def write_results_templates() -> list[tuple[str, str]]:
    """Render Home, Leaderboard and Explore over the filler prepared snapshot.

    The three results templates have no hand-authored mock: they are the live
    views, so their preview is what the live renderers draw from a snapshot of
    invented configured algorithms, budgets, seeds and instances. The snapshot is
    generated from that roster's seeds by its own script, never here: a build
    renders a prepared snapshot and measures nothing.

    Returns:
        Each rendered page's template name and its link target in the index.
    """
    snapshot = filler_results.snapshot()
    slug = filler_results.FILLER_TASK.id
    identity = filler_results.IDENTITY
    by_surface = {
        surface: ResultsRoute(
            surface=surface,
            route=surface,
            title=name,
            identity=identity,
            published=False,
            root=snapshot,
        )
        for surface, name in RESULTS_SURFACES
    }
    source = read_prepared(snapshot, slug)
    if source[0] is None or source[1] is None:
        raise MissingSource(
            f"The filler snapshot carries no measured panels and standings for "
            f"{slug!r}; generate it again."
        )
    written: dict[str, object] = {}
    for name, page, surface, data_route in filler_results.RESULTS_PAGES:
        home = name == "T1"
        route = replace(
            by_surface[surface], route=data_route, title=TEMPLATES[name].title
        )
        # The surface's own render ships the index, whose panel files are named
        # relative to that index wherever a page embeds it. A Home preview then
        # names the index and the provenance relative to itself, exactly as the
        # site root's Home names the task surface it embeds.
        # Both Leaderboard previews offer the Explore preview beside them.
        explore = (
            "../"
            + next(
                directory
                for _, directory, previewed, _ in filler_results.RESULTS_PAGES
                if previewed != LEADERBOARD_SURFACE
            )
            + "/"
        )
        rendered = _results_surface(
            route,
            source,
            slug,
            depth=2,
            explore=explore,
            roster=ROSTER_PREVIEW,
            protocol=METHODOLOGY_PREVIEW,
            # The Explore preview's introduction links the Leaderboard preview.
            leaderboard="../"
            + next(
                directory
                for template, directory, _, _ in filler_results.RESULTS_PAGES
                if template == "T4"
            )
            + "/",
        )
        body = rendered.body
        directory = f"{TEMPLATES_DIRNAME}/{page}"
        if home:
            preview = _results_surface(
                route,
                source,
                slug,
                depth=2,
                home=True,
                base=".",
                explore=explore,
                protocol=METHODOLOGY_PREVIEW,
            )
            body = home_leaderboards(
                [
                    (
                        slug,
                        filler_results.FILLER_TASK_NAME,
                        _filler_links(),
                        preview.body,
                    )
                ],
                depth=2,
            )
        sidecars = f"{directory}/{data_route}" if home else directory
        for sidecar, payload in rendered.sidecars.items():
            with mkdocs_gen_files.open(f"{sidecars}/{sidecar}", "w") as f:
                f.write(payload)
        data = f"{directory if home else TEMPLATES_DIRNAME}/{DATA_DIRNAME}"
        shared = dict(rendered.shared)
        _link_shared_provenance(shared)
        for shared_name, payload in shared.items():
            target = f"{data}/{shared_name}"
            if target in written:
                # Two surfaces of one snapshot read the same panels, so one copy
                # serves both. Different bytes under one name would mean one of
                # them is reading something else, which is not a file to drop.
                if written[target] != payload:
                    raise MissingSource(f"{target} is written twice, differently.")
                continue
            written[target] = payload
            mode = "wb" if isinstance(payload, bytes) else "w"
            with mkdocs_gen_files.open(target, mode) as f:
                f.write(payload)
        _declared_sections(name, body)
        _write_page(f"{directory}/index.md", _filler_page(name, body))
    return [
        (page, f"{page}/index.md") for _, page, _, _ in filler_results.RESULTS_PAGES
    ]


def _filler_links() -> str:
    """The routes a Home panel offers, pointing at the other filler previews.

    On the site root these name a task space; the preview has no task space, so
    the row is spelled here and points at the previews of the pages it would
    reach. It is the one part of this page composed for the preview.
    """
    pages = {name: page for name, page, _, _ in filler_results.RESULTS_PAGES}
    return " · ".join(
        f'<a href="../{target}/">{name}</a>'
        for name, target in (
            ("Task", "t03-task-overview"),
            ("Leaderboard", pages["T4"]),
            ("Explore", pages["T5"]),
        )
    )


def _filler_page(name: str, body: str) -> str:
    """One results template's preview page, said to be filler in as many words."""
    return (
        f"# {name} {TEMPLATES[name].title}\n\n"
        "The live view, rendered from a snapshot of invented configured algorithms, "
        "budgets, seeds and instances. No name or number on this page is "
        "real.\n\n"
        f"{body}"
    )


def _declared_sections(name: str, body: str) -> None:
    """Refuse a results page that does not match the template it declares.

    The results templates have no record a renderer fills: the live view is
    the page, so what it emits is read back and held against the template. A
    required section the renderers stopped emitting stops the build, and a
    section they emit in another order is the same failure, rather than a
    template document describing a page that has moved on.
    """
    template = TEMPLATES[name]
    declared = [section.key for section in template.sections]
    found = filler_results.rendered_sections(name, body)
    missing = [
        section.key
        for section in template.sections
        if section.required and section.key not in found
    ]
    if missing:
        raise MissingSource(f"{name} renders no {missing} section.")
    unknown = [key for key in found if key not in declared]
    if unknown or list(found) != [key for key in declared if key in found]:
        raise MissingSource(f"{name} renders {list(found)}, not {declared}.")


# The output Plotly's figure.show() leaves in a notebook, which nbconvert has
# no renderer for. A tutorial page draws it as the site draws its own charts:
# in the site's chart template, with the reader's theme painted over it.
PLOTLY_MIME_TYPE = "application/vnd.plotly.v1+json"
TUTORIAL_PAGE_DEPTH = 3  # library/tutorials/<notebook>/


def draw_plotly_outputs(document: dict[str, Any]) -> None:
    """Give each Plotly figure in *document* an HTML output the site draws.

    The first figure's output also loads the chart runtime, so a page with no
    figure loads nothing.
    """
    figures = [
        output["data"]
        for cell in document["cells"]
        for output in cell.get("outputs", [])
        if PLOTLY_MIME_TYPE in output.get("data", {})
    ]
    for number, data in enumerate(figures):
        figure = data[PLOTLY_MIME_TYPE]
        layout = figure.setdefault("layout", {})
        layout["template"] = chart_style()
        # The template hides Plotly's legend for pages that draw their own;
        # a notebook figure has only Plotly's.
        layout.setdefault("showlegend", True)
        data["text/html"] = (
            (runtime_scripts(TUTORIAL_PAGE_DEPTH) if number == 0 else "")
            + f'<div id="{FIGURE_ID_PREFIX}{number}" class="{FIGURE_CLASS}"></div>'
            + f'<script type="application/json">{embed(figure)}</script>'
        )


def write_tutorials() -> None:
    """Render the top-level ``tutorials/`` as pages under ``library/tutorials/``.

    A notebook's page is the executed copy the journeys harness wrote, rendered
    natively by mkdocs-jupyter, with the committed source notebook published
    beside it as the download, and any sibling script the notebook names
    beside that. The directory also holds the authored
    walkthroughs that are not notebooks; both are Library tutorials and are
    published beside each other, so the learning path has one home.

    Notebook titles and landing-page descriptions come from their first
    Markdown cells. Navigation supplies reading order, not filenames.
    """
    directory = TUTORIALS_ROOT
    sources = tutorial_sources(TUTORIALS_ROOT)
    notebooks = [
        sources[Path(target).name]
        for target in NAV_TARGETS
        if target.startswith(f"library/{TUTORIALS_DIRNAME}/")
        and target.endswith(".ipynb")
    ]
    if not notebooks:
        return
    # Only a local release may bring its own executed copies; a published or
    # absent one renders the committed copies an anonymous reader's run wrote.
    local_release = RELEASE is not None and not RELEASE.published
    guide = next(
        entry["User guide"]
        for entry in FilesEditor.current().config["nav"]
        if "User guide" in entry
    )
    groups = {
        title: [
            sources[Path(target).name]
            for target in navigation_targets(entries)
            if target.endswith(".ipynb")
        ]
        for entry in guide
        for title, entries in entry.items()
        if isinstance(entries, list)
    }
    _write_page("library/index.md", tutorials_index(groups))
    for path in notebooks:
        executed = executed_directory(directory, path.name, local_release=local_release)
        if executed.name == LOCAL_EXECUTED_DIRNAME:
            print(f"tutorials: {path.name} renders the copy in {executed}")
        document, unrun = executed_notebook(path, executed)
        draw_plotly_outputs(document)
        for cell in document["cells"]:
            if cell["cell_type"] == "markdown":
                cell["source"] = notebook_links("".join(cell["source"]), path.stem)
        opening = ""
        if unrun:
            # The harness could not meet this notebook's precondition here, so
            # the page carries its cells and says why they show no output. The
            # reason's first line is the page's share of it; the harness's
            # record and report keep the whole of it.
            notice = copy_html(
                "tutorials",
                "notebook_unrun",
                tag="div",
                classes=UNRUN_NOTICE_CLASS,
                fields={"precondition": unrun.splitlines()[0]},
            )
            opening += f"\n{notice}\n" if notice else ""
        if opening:
            document["cells"].insert(
                0,
                {
                    "id": "dp-title",
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": opening,
                },
            )
        with mkdocs_gen_files.open(
            f"library/{TUTORIALS_DIRNAME}/{path.stem}.ipynb", "w"
        ) as page:
            page.write(json.dumps(document))
        # The plugin names this file beside the page as the notebook's own
        # download; it holds the source a reader runs, not the executed copy.
        with mkdocs_gen_files.open(
            f"library/{TUTORIALS_DIRNAME}/{path.stem}/{path.name}", "wb"
        ) as download:
            download.write(
                download_notebook(path, FilesEditor.current().config["site_url"])
            )
        # A script the notebook wraps is offered beside it, at the relative
        # path the notebook's own cells name it by.
        for script in companion_scripts(path):
            with mkdocs_gen_files.open(
                f"library/{TUTORIALS_DIRNAME}/{path.stem}/{script.name}", "wb"
            ) as companion:
                companion.write(script.read_bytes())
    for path in sorted(directory.glob("*.md")):
        if f"library/{TUTORIALS_DIRNAME}/{path.name}" not in NAV_TARGETS:
            continue
        source = f"{TUTORIALS_DIRNAME}/{path.name}"
        _write_page(
            f"library/{source}",
            _publish_links(path.read_text(encoding="utf-8"), source),
        )


def _absent(
    route: ResultsRoute,
    slug: str,
    missing: Sequence[str],
    *,
    home: bool = False,
    notice: str = "",
) -> RenderedPage:
    """Keep full diagnostics off the opening page and out of its search text."""
    name = "unavailable.txt"
    url = f"{slug}/{route.route}/{name}" if home else name
    return RenderedPage(
        body=notice + absence_body(route.identity, slug, url),
        sidecars={name: "\n\n".join(missing) + "\n"},
    )


def _named(absent: Sequence[tuple[str, str]]) -> list[str]:
    """Each metric that cannot be reported, beside the reason it cannot."""
    return [f"{metric}: {reason}" for metric, reason in absent]


# Where one task's panel files are written. Every surface of that task reads
# the same matrices, so they are written once and named from each page.
DATA_DIRNAME = "data"


def _data_prefix(route: ResultsRoute) -> str:
    """How one surface's pages reach that location."""
    return "../" * len(route.route.split("/")) + f"{DATA_DIRNAME}/"


def _surface_index(data: TaskData, prefix: str) -> str:
    """The index one surface ships, naming the files its root wrote once.

    Both surfaces read one index shaped the same way, so both name the files
    their root wrote: the trial matrices a measured panel derives every
    reading from, or the curves a published panel draws verbatim. ``outcomes``
    is a build-time reading for the configured algorithm page; the browser derives its
    own from the trial matrix, so it never ships here.
    """
    index = {
        **{key: value for key, value in data.index.items() if key != "outcomes"},
        "panels": [
            {
                **entry,
                "runs": None if entry["runs"] is None else prefix + entry["runs"],
                **(
                    {"curves": prefix + entry["curves"]}
                    if entry.get("curves") is not None
                    else {}
                ),
            }
            for entry in data.index["panels"]
        ],
    }
    return embed(index)


def _preview_notice(index: Mapping[str, Any]) -> str:
    """Say that a page reads a local snapshot, and what it does not carry, as HTML."""
    coverage = index.get("preview_coverage")
    if coverage is None:
        return ""
    notice = copy_html("shared", "preview_notice")
    if not coverage["missing"]:
        return notice
    return notice + copy_html(
        "shared",
        "preview_missing_trials",
        fields={
            "missing": str(coverage["missing"]),
            "expected": str(coverage["expected"]),
        },
    )


def _opening_selection(data: TaskData, slug: str) -> Selection:
    """The reading an Explore page opens at, as its notebook's parameters.

    The page opens on its headline Metric's first panel, its standings at the
    protocol's quantile and its curves at ``OPENING_TAU``, over every algorithm
    the panel draws, with every instance property's range open.
    """
    index = data.index
    metric = default_metric(
        str(index["headline_metric"]), [panel.metric for panel in data.panels]
    )
    budget = next(panel.budget for panel in data.panels if panel.metric == metric)
    return Selection(
        task=slug,
        metric=metric,
        budget=budget,
        tau=OPENING_TAU,
        quantile=float(index["headline_quantile"]),
        baseline=str(index["baselines"][metric]["default"]),
        pool=str(index["pool"]),
        taus=list(index["taus"]),
        instance_filters={
            name: [None, None] for name in index["features"] if name != "epsilon"
        },
    )


def _release_identity() -> ReleaseIdentity | None:
    """How a reader's own notebook opens the release these pages render.

    A published build pins one revision, which a notebook fetches for itself.
    A preview reads a local cut: a hosted one names the Distribution Release
    revision that cut is, a local one the tree itself.
    """
    if RELEASE is None:
        return None
    if PIN is not None and RELEASE.published:
        return ReleaseIdentity(PIN.repo_id, PIN.revision, "", RELEASE.version)
    if RECREATE_REVISION:
        return ReleaseIdentity(
            DISTRIBUTION_RELEASE_REPOSITORY, RECREATE_REVISION, "", RELEASE.version
        )
    return ReleaseIdentity("", "", str(RELEASE.root), RELEASE.version)


def _results_surface(
    route: ResultsRoute,
    source: tuple,
    slug: str,
    *,
    depth: int,
    home: bool = False,
    base: str | None = None,
    notebook: ReleaseIdentity | None = None,
    explore: str | None = None,
    roster: str | None = None,
    protocol: str | None = None,
    leaderboard: str | None = None,
    skipped: Sequence[Mapping[str, str]] = (),
    removed: int = 0,
) -> RenderedPage:
    """Render one prepared snapshot without acquiring Results or Inputs.

    A release that published standings over Results it does not ship supplies
    them as prepared curves-only panels, which the Leaderboard reports in place
    of the panels measured from the Results the release does carry.

    A Home preview names the files it fetches from the directory it is embedded
    in, which is the task's own on the site root. ``base`` renames that
    directory for a preview embedded elsewhere; it is the task's own by default.

    ``notebook`` names where a reader's own copy of this reading is fetched
    from; a surface offers one only where the page reads a release a notebook
    can open for itself, and never from the Home preview, which holds no
    selection to recreate.

    ``explore`` and ``roster`` are the page-relative links to the task's
    Explore and Roster pages, which a Leaderboard's introduction links where
    the build publishes them.

    ``protocol`` is the page-relative link to the methodology page every baseline's
    Definition link points into, the task's own where ``None``.

    ``leaderboard`` is the page-relative link an Explore page's introduction
    gives the task's Leaderboard, the task's own where ``None``.

    ``skipped`` is what a local cut's task left out, and ``removed`` how many of
    those its standings also dropped; a page other than the Home preview says so.
    """
    carrier = slug if base is None else base
    methodology_href = f"{'../' * depth}{METHODOLOGY_ROUTE}"
    data, standings_data, missing, refused = source
    absent = list(refused)
    notice_html = _preview_notice(data.index if data is not None else {})
    notice_html += banner_html(skipped, removed, home=home)
    if missing and route.surface != LEADERBOARD_SURFACE:
        return _absent(route, carrier, missing, home=home, notice=notice_html)
    if data is None and standings_data is None:
        return _absent(route, carrier, _named(absent), home=home, notice=notice_html)
    prefix = f"{carrier}/{DATA_DIRNAME}/" if home else _data_prefix(route)
    offered = notebook if not home else None
    if route.surface == LEADERBOARD_SURFACE:
        reported = standings_data
        if reported is None or not reported.panels:
            return _absent(
                route, carrier, _named(absent), home=home, notice=notice_html
            )
        return RenderedPage(
            body=notice_html
            + leaderboard_body(
                slug,
                collection_standings(reported),
                route.identity,
                depth=depth,
                published=route.published,
                provenance=prefix + PROVENANCE_NAME,
                absent=absent,
                policy=reported.index,
                index=(f"{carrier}/{route.route}/{INDEX_NAME}" if home else INDEX_NAME),
                home=home,
                explore_href=explore,
                roster_href=roster,
                protocol_href=protocol,
                methodology_href=methodology_href,
            ),
            sidecars={INDEX_NAME: _surface_index(reported, prefix)},
            shared=reported.files,
        )
    if data is None or not data.panels:
        return _absent(route, carrier, _named(absent), home=home, notice=notice_html)
    page = playground_page(
        data,
        route.identity,
        depth=depth,
        provenance=prefix + PROVENANCE_NAME,
        absent=absent,
        notebooks=NOTEBOOK_NAMES if offered is not None else None,
        protocol_href=protocol,
        leaderboard_href=leaderboard,
    )
    sidecars = {INDEX_NAME: _surface_index(data, prefix)}
    if offered is not None:
        opening = _opening_selection(data, slug)
        # Each view's notebook opens on the page's own opening reading; the
        # page rewrites its parameters with the reader's selection on download.
        first = str(data.index["algorithms"][0]["uuid"])
        for view, name in NOTEBOOK_NAMES.items():
            chosen = replace(
                opening,
                view=view,
                algorithm=first if view == OUTCOMES_VIEW else None,
                algorithm_labels={
                    str(entry["uuid"]): str(entry["label"])
                    for entry in data.index["algorithms"]
                },
                series=profile_series(data.index["algorithms"]),
            )
            sidecars[name] = notebook_bytes(chosen, offered)
    return RenderedPage(
        body=notice_html + page.body,
        sidecars=sidecars,
        shared=data.files,
    )


# What the card renders for each step of a recipe.
_CARD_FIELDS = (
    "name",
    "kind",
    "docs",
    "summary",
    "hyperparameters",
)
# What the card shows of each hyperparameter: what it runs with, not what it
# means; the component's own page explains it.
_CARD_PARAMETER_FIELDS = ("name", "value", "default")


def _trim_component(component: dict[str, Any]) -> dict[str, Any]:
    """One component, kept to what the card's recipe reads."""
    return {
        **{key: component[key] for key in _CARD_FIELDS if key in component},
        "hyperparameters": [
            {key: row[key] for key in _CARD_PARAMETER_FIELDS}
            for row in component.get("hyperparameters", ())
        ],
        "children": [
            {"field": child["field"], "component": _trim_component(child["component"])}
            for child in component["children"]
        ],
    }


def _trim_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """The shipped provenance payload, trimmed to the card's own reading of it.

    The full, docs-linked tree still lives in the caller's own copy, which the
    server-rendered T21 Recipe section reads directly; only the copy this
    build ships for the runtime Details dialog to fetch is trimmed.
    """
    trimmed = {}
    for uuid, entry in provenance.items():
        one: dict[str, Any] = {"component": _trim_component(entry["component"])}
        declaration = entry.get("declaration")
        if declaration is not None:
            one["declaration"] = {"description": declaration.get("description", "")}
        trimmed[uuid] = one
    return trimmed


def _link_shared_provenance(shared: dict[str, Any]) -> dict[str, Any]:
    """Point one root's provenance at the pages it ships beside, then trim.

    The measured provenance names its components by reference; a page's
    Details dialog reads a route, so the references become the routes this
    build published them at, in place. Returns the full, docs-linked tree for
    the caller's own use; ``shared`` keeps only the trimmed copy this build
    ships for the runtime dialog to fetch.
    """
    if PROVENANCE_NAME not in shared:
        return {}
    provenance = json.loads(shared[PROVENANCE_NAME])
    link_provenance(provenance, COMPONENT_ROUTES)
    shared[PROVENANCE_NAME] = canonical_json_bytes(
        _trim_provenance(provenance)
    ).decode()
    return provenance


# The T21 and T22 routes sit at one fixed depth below the site root, whatever
# the task's slug: "<slug>/roster/index.md" and "<slug>/roster/<uuid>/index.md"
# always carry 2 and 3 path separators.
T22_DEPTH = 2
# The T23 methodology page sits at the same depth.
METHODOLOGY_DEPTH = 2
T21_DEPTH = 3
T22_CLIMB = "../" * T22_DEPTH
T21_CLIMB = "../" * T21_DEPTH


def _prefixed_docs(component: dict[str, Any], climb: str) -> dict[str, Any]:
    """A copy of one recipe component tree, its API links made page-relative."""
    docs = component.get("docs")
    return {
        **component,
        "docs": (climb + docs) if docs else None,
        "children": [
            {**child, "component": _prefixed_docs(child["component"], climb)}
            for child in component.get("children", ())
        ],
    }


def task_methodology(snapshot: Path, slug: str, title: str) -> Protocol:
    """One task's T23 record, from the snapshot's protocol and the library."""
    return protocol_page(
        BUILT_IN_TASKS[slug],
        f"{title}: methodology",
        read_comparison(snapshot / BENCHMARK_PROTOCOL_NAME, slug),
        RELEASE.identity if RELEASE is not None else "",
        methodology=Link(
            narrative_title(read_narrative(snapshot, BENCHMARK_NARRATIVE_NAME)),
            f"{'../' * METHODOLOGY_DEPTH}{METHODOLOGY_ROUTE}",
        ),
        depth=METHODOLOGY_DEPTH,
        explored=read_explored(snapshot, slug),
    )


def write_results(
    slug: str,
    title: str,
    routes: Sequence[ResultsRoute],
    snapshot: Path | None,
) -> tuple[list[str], str, dict[str, RosterLink], int | None]:
    """Write results and return navigation, Home, Roster links and registered count.

    The Roster links are each component class's link into this task's
    Roster, keyed by class path and measured from the site root.

    A surface is written at a depth below the site root, so a page loading the
    chart runtime that ships there can climb back out to it. Each surface ships
    its own index beside its page; the matrices and the provenance both surfaces
    read are written once, in the task's own data directory, so nothing a page
    fetches leaves this build's output and nothing is written twice.
    """
    entries: list[str] = []
    written: set[str] = set()
    home = ""
    roster: dict[str, RosterLink] = {}
    registered: int | None = None
    provenance_text = ""
    source_texts: dict[str, bytes] = {}
    if not routes or snapshot is None:
        return entries, home, roster, registered
    source = read_prepared(snapshot, slug)
    skipped = read_skipped(snapshot, slug)
    removed = read_removed(snapshot, slug)
    explore = next(
        (other.route for other in routes if other.surface != LEADERBOARD_SURFACE), None
    )
    for route in routes:
        page = f"{slug}/{route.route}/index.md"
        depth = page.count("/")
        rendered = _results_surface(
            route,
            source,
            slug,
            depth=depth,
            notebook=_release_identity(),
            explore=f"../{explore}/" if explore else None,
            roster="../roster/",
            skipped=skipped,
            removed=removed,
        )
        if route.surface == LEADERBOARD_SURFACE and not home:
            home = _results_surface(
                route,
                source,
                slug,
                depth=0,
                home=True,
                explore=f"{slug}/{explore}/" if explore else None,
            ).body
        for name, payload in rendered.sidecars.items():
            with mkdocs_gen_files.open(f"{slug}/{route.route}/{name}", "w") as f:
                f.write(payload)
        directory = f"{slug}/{DATA_DIRNAME}"
        shared = dict(rendered.shared)
        first_provenance = not provenance_text and PROVENANCE_NAME in shared
        if first_provenance:
            source_texts = {
                name.removeprefix("sources/"): payload
                for name, payload in shared.items()
                if name.startswith("sources/") and isinstance(payload, bytes)
            }
        linked = _link_shared_provenance(shared)
        if first_provenance:
            # Captured after linking, so a T21 Recipe component's link to its
            # own T9 page exists (docs is filled by _link_shared_provenance).
            provenance_text = canonical_json_bytes(linked).decode()
        for name, payload in shared.items():
            # The source notes ship inline on T21 (Provenance), read above
            # from this same payload; nothing links their own URL, so this
            # build reads them without also publishing them as files.
            if name.startswith("sources/"):
                continue
            if f"{directory}/{name}" in written:
                continue
            written.add(f"{directory}/{name}")
            mode = "wb" if isinstance(payload, bytes) else "w"
            with mkdocs_gen_files.open(f"{directory}/{name}", mode) as f:
                f.write(payload)
        body = rendered.body
        with mkdocs_gen_files.open(page, "w") as f:
            f.write(
                f"---\ntitle: {title} {route.title.lower()}\nresults_page: true\n---\n\n"
                f"# {title}: {route.title.lower()}\n\n"
                f"{TASK_PAGE_MARKER}\n\n"
                f"{body}"
            )
        entries.append(f"- [{route.title}]({route.route}/index.md)")
    for route in routes:
        if route.surface != LEADERBOARD_SURFACE:
            continue
        data, standings_data, missing, _ = source
        reported = standings_data if standings_data is not None else data
        if missing or reported is None:
            break
        registered = len(eligible_rows(reported.index))
        if not provenance_text:
            # The configuration provenance a Roster names each row from is
            # written by the measured surface. A task whose Results this
            # build cannot measure still publishes its standings, and they
            # rank rows no page here could describe, so the task lists none
            # rather than naming one it cannot reach.
            break
        provenance = json.loads(provenance_text)
        panels = collection_standings(reported)
        # The Roster is the distribution lane's full algorithm list, ranked or
        # not; the released view's own index carries only the ranked subset.
        roster_index = data.index if data is not None else reported.index
        entries_full = roster_entries(roster_index, panels, provenance)
        rows, facet = roster_rows(
            [replace(entry, url=f"{entry.uuid}/") for entry in entries_full],
            provenance,
            headline_readings(roster_index, panels),
            slug,
            {task_slug: task_title for task_slug, task_title, _ in TASKS},
            T22_CLIMB,
        )
        roster = component_links(rows, provenance, facet, title, f"{slug}/roster/")
        headline = headline_panel(roster_index, panels)
        landing = RosterLanding(
            title=f"{title}: roster",
            rows=tuple(rows),
            classes=tuple(
                name
                for name in CLASS_ORDER
                if any(row.algorithm.privacy_interpretation == name for row in rows)
            ),
            transformations=tuple(dict.fromkeys(facet.values())),
            metric=headline.metric if headline is not None else "",
            budget=(
                budget_short_label(headline.budget) if headline is not None else ""
            ),
            percent=f"{float(roster_index['headline_quantile']) * 100:g}",
            notice=banner(skipped, removed),
        )
        _write_page(f"{slug}/roster/index.md", render(TEMPLATES["T22"], landing))
        leaderboard_url = f"{T21_CLIMB}{slug}/{route.route}/"
        task_link = Link(title, f"{T21_CLIMB}{slug}/index.md")
        outcomes = data.index["outcomes"] if data is not None else {}
        # The breakdown ranks each budget on the public corpus at the headline
        # metric, among the algorithms Explore opens with.
        public = collection_standings(data) if data is not None else ()
        opening = headline_panel(data.index, public) if data is not None else None
        shown = (
            {str(row["uuid"]) for row in eligible_rows(data.index)}
            if data is not None
            else set()
        )
        public_panels = tuple(
            replace(panel, rows=tuple(row for row in panel.rows if row.uuid in shown))
            for panel in public
            if opening and panel.metric == opening.metric
        )
        budget_order = (
            list(dict.fromkeys(entry["budget"] for entry in data.index["panels"]))
            if data is not None
            else []
        )
        tuning_url = f"{T21_CLIMB}{METHODOLOGY_ROUTE}" + tuning_fragment(
            read_narrative(snapshot, BENCHMARK_NARRATIVE_NAME)
        )
        for entry in entries_full:
            described = provenance[entry.uuid]
            ref = (described.get("declaration") or {}).get("provenance_ref", "")
            if ref:
                key = f"{ref}.txt"
                if key not in source_texts:
                    raise RuntimeError(
                        f"{entry.label!r} ({entry.uuid}) declares provenance "
                        f"{ref!r}, which this release ships no document for."
                    )
                provenance_doc = source_texts[key].decode("utf-8")
            else:
                provenance_doc = ""
            page = roster_page(
                entry,
                task_link,
                {
                    **described,
                    "component": _prefixed_docs(described["component"], T21_CLIMB),
                },
                panels,
                leaderboard_url,
                slug,
                provenance_doc,
                outcomes,
                budget_order,
                float(roster_index["headline_quantile"]),
                f"{T21_CLIMB}{slug}/{explore}/" if explore else "",
                public_panels,
                privacy_interpretation_href(T21_DEPTH),
                methodology_url=tuning_url,
            )
            _write_page(
                f"{slug}/roster/{entry.uuid}/index.md",
                render(TEMPLATES["T21"], page)
                + runtime_scripts(T21_DEPTH, charts=False),
                searchable=False,
            )
        entries.append("- [Roster](roster/index.md)")
        break
    protocol_page_record = task_methodology(snapshot, slug, title)
    _write_page(
        f"{slug}/methodology/index.md", render(TEMPLATES["T23"], protocol_page_record)
    )
    entries.append("- [Methodology](methodology/index.md)")
    return entries, home, roster, registered


def write_tasks(
    results: Mapping[str, tuple[ResultsRoute, ...]],
    snapshot: Path | None,
    corpus: corpus_source.Corpus | None,
) -> dict[str, str]:
    """Build task navigation and every documented task's own root space."""
    if not LIB_DOCS.is_dir():
        return {}
    # A results surface hangs off its task's own pages, so a carried task the
    # library documents nowhere has no page to hang one from. That is a
    # mismatch between the supplied release and this checkout's documentation,
    # not a task to drop quietly.
    undocumented = sorted(set(results) - set(TASK_SLUGS))
    if undocumented:
        raise RuntimeError(
            "A supplied release carries results for task(s) this build has no "
            f"documentation page for: {undocumented}."
        )
    titles = {slug: title for slug, title, _ in TASKS}
    statuses = task_statuses(REACHABLE, set(results))
    # Results are read first: an Algorithm page names the configured algorithms that run
    # it, and those come from the rosters the prepared snapshot carries.
    surfaces: dict[str, list[str]] = {}
    bodies: dict[str, str] = {}
    roster: dict[str, dict[str, RosterLink]] = {}
    for slug in TASK_SLUGS:
        entries, body, links, count = write_results(
            slug, titles[slug], results.get(slug, ()), snapshot
        )
        surfaces[slug] = entries
        bodies[slug] = body
        for path, found in links.items():
            roster.setdefault(path, {})[slug] = found

    FilesEditor.current().config.extra["benchmarked_tasks"] = [
        slug for slug in TASK_SLUGS if statuses[slug] == "Benchmarked"
    ]

    # Section nav (literate-nav): DERIVED from the assembled tasks, so the nav
    # self-gates with the pages — a task absent from this build's library docs
    # gets no entry (and no strict-mode broken link) instead of a stale one.
    ordered = [slug for slug, _, _ in TASKS]
    ordered.sort(key=lambda slug: statuses[slug] != "Benchmarked")
    section_nav = [f"- [{titles[slug]}](../{slug}/)" for slug in ordered]
    _write_page("tasks/SUMMARY.md", "\n".join(section_nav) + "\n")

    for slug in TASK_SLUGS:
        write_task_space(
            BUILT_IN_TASKS[slug],
            LIB_DOCS / "tasks" / f"{slug}.md",
            results.get(slug, ()),
            surfaces[slug],
            roster,
            corpus,
            (
                (
                    read_comparison(snapshot / BENCHMARK_PROTOCOL_NAME, slug),
                    read_explored(snapshot, slug),
                )
                if slug in results and snapshot is not None
                else None
            ),
        )
    return bodies


def _metric_baselines(
    task: Task, comparison: ReleasedComparison, explored: Sequence[str]
) -> dict[str, tuple[Link, ...]]:
    """Each Explore Metric's offered baselines, linked into the task's catalogue.

    The pooled baseline and reading in a Metric's own units apply to every
    Metric alike, so a Metric lists its statistical baseline, the external
    baselines offered for it and a computed headline baseline. A Metric
    Explore does not list has none.
    """
    return {
        name: tuple(
            Link(key, f"../methodology/#baseline-{key}")
            for key in comparison.stored_for(name, metric.baseline_name)
            + tuple(
                key
                for key in comparison.computed_for(name)
                if key == comparison.headline_baseline
            )
        )
        for name, metric in list_built_in_metrics(task).items()
        if name in explored
    }


def _record_summary(path: str, text: str) -> None:
    """Keep one page's opening sentence, for the landing that lists the page."""
    summary = page_summary(text)
    if summary:
        SUMMARIES[path] = summary


def _write_page(
    path: str,
    text: str,
    *,
    searchable: bool = True,
) -> None:
    refuse_untemplated(path, TASK_SLUGS)
    _record_summary(path, text)
    text = rewrite_links(text, path, API_GROUPS, API_ROUTES)
    # A page may arrive with its own front matter; this adds to that one block,
    # since a second block would render as the page's opening text.
    declared_block = re.match(r"---\n(.*?)\n---\n\n?", text, flags=re.S)
    declared = declared_block[1].splitlines() if declared_block else []
    if declared_block:
        text = text[declared_block.end() :]
    keys = {line.split(":", 1)[0] for line in declared if not line.startswith(" ")}
    heading = re.search(r"^# (.+)$", text, flags=re.M)
    metadata = []
    if heading and "title" not in keys:
        metadata.append(f"title: {json.dumps(heading[1])}")
    if (
        not searchable
        or path.endswith(NAV_FILE)
        or path.startswith(f"{TEMPLATES_DIRNAME}/")
    ):
        metadata.extend(["search:", "  exclude: true"])
    clash = keys & {line.split(":", 1)[0] for line in metadata if line[0] != " "}
    if clash:
        raise ValueError(f"{path} already declares {sorted(clash)} in front matter.")
    if declared or metadata:
        text = "---\n" + "\n".join(declared + metadata) + "\n---\n\n" + text
    with mkdocs_gen_files.open(path, "w") as page:
        page.write(text)


def read_corpus(
    release: ReleaseInput | None, snapshot: Path | None
) -> corpus_source.Corpus | None:
    """Read the Corpus area's one input: the release this build resolved.

    Returns:
        The release's sources, or ``None`` where this build reads no release.
    """
    if release is None:
        return None
    corpus = corpus_source.read_corpus(
        release.root, identity=release.identity, published=release.published
    )
    assert snapshot is not None
    arguments = read_instance_arguments(snapshot)
    corpus = replace(
        corpus,
        sources=tuple(
            replace(
                source,
                instances=tuple(
                    replace(
                        instance,
                        references=tuple(
                            arguments[instance.task][instance.uuid].items()
                        ),
                    )
                    for instance in source.instances
                ),
            )
            for source in corpus.sources
        ),
    )
    _report_corpus(corpus)
    return corpus


def _report_corpus(corpus: corpus_source.Corpus) -> None:
    """State what this build read, what it left out, and why.

    A source or an instance the Corpus area does not list leaves its reason in
    the build's own output rather than in the silence of a shorter table, and
    a provenance key no ruled section renders is named for the template's
    owner instead of being shown on a page.
    """
    print(
        f"corpus: read {len(corpus.files_read)} declared files from "
        f"{corpus.identity}: {len(corpus.sources)} sources, "
        f"{sum(len(source.instances) for source in corpus.sources)} instances."
    )
    for reason in corpus.skipped:
        print(f"corpus: not listed: {reason}")
    if corpus.unrendered_keys:
        print(
            "corpus: the release carries provenance keys no ruled section "
            f"renders, and no page shows them: {', '.join(corpus.unrendered_keys)}."
        )


@dataclass(frozen=True)
class InstanceContext:
    """What one task's instance pages read beyond the corpus itself.

    ``conditions`` holds each instance's design condition number Explore
    measured; ``headline`` names the baseline the task's standings divide by.
    """

    conditions: Mapping[str, float]
    headline: str = ""


def _instance_context(snapshot: Path | None, slug: str) -> InstanceContext:
    """One task's condition numbers and headline baseline, from the snapshot.

    A build without a snapshot shows none.
    """
    if snapshot is None:
        return InstanceContext({})
    comparison = read_comparison(snapshot / BENCHMARK_PROTOCOL_NAME, slug)
    data = read_prepared(snapshot, slug)[0]
    return InstanceContext(
        conditions={
            str(entry["uuid"]): float(entry[DESIGN_CONDITION])
            for entry in (data.index.get("instances", ()) if data is not None else ())
            if entry.get(DESIGN_CONDITION) is not None
        },
        headline=comparison.headline_baseline,
    )


def write_instances(corpus: corpus_source.Corpus, snapshot: Path | None) -> None:
    """Write one page per instance the release distributes, out of search.

    Only the instances the release's own provenance bundles and registry carry get
    a page, so a withheld instance has no route and nothing points at one.
    """
    names = {slug: title for slug, title, _ in TASKS}
    context: dict[str, InstanceContext] = {}
    cards: dict[str, dict[str, dict[str, Any]]] = {}
    for source in corpus.sources:
        for instance in source.instances:
            if instance.task not in names:
                continue
            if instance.task not in context:
                context[instance.task] = _instance_context(snapshot, instance.task)
            known = context[instance.task]
            page = instance_page(
                corpus,
                source,
                instance,
                condition=known.conditions.get(instance.uuid),
            )
            _write_page(
                f"{corpus_source.instance_route(instance)}index.md",
                render(TEMPLATES["T24"], page),
                searchable=False,
            )
            cards.setdefault(instance.task, {})[instance.uuid] = instance_card(
                source,
                instance,
                condition=known.conditions.get(instance.uuid),
                headline=known.headline,
            )
    # The card every page opens on an instance reads its task's file once,
    # on first use, rather than every page carrying every task's instances.
    for task, held in cards.items():
        with mkdocs_gen_files.open(
            f"{task}/{DATA_DIRNAME}/{INSTANCE_CARDS_NAME}", "wb"
        ) as f:
            f.write(canonical_json_bytes(held))


def write_corpus(corpus: corpus_source.Corpus | None, snapshot: Path | None) -> None:
    """Write the Corpus area: its index, one page per source and per instance.

    A build supplied with no release writes the index alone, saying so, rather
    than a tab whose pages would claim a corpus this build cannot read.
    """
    names = {slug: title for slug, title, _ in TASKS}
    _write_page(
        f"{CORPUS_DIRNAME}/index.md",
        render(
            TEMPLATES["T2"],
            replace(
                corpus_source.corpus_index(corpus, names),
                data_link=copy_html(
                    "corpus",
                    "data_page_link",
                    pages={DATA_PAGE_TITLE: f"{CORPUS_ABOUT}/"},
                ),
            ),
        ),
    )
    _write_page(
        f"{CORPUS_DIRNAME}/{CORPUS_ABOUT}.md",
        (LIB_DOCS / "content" / "corpus-about.md").read_text(encoding="utf-8"),
    )
    entries = ["- [Corpus](index.md)", f"- [{DATA_PAGE_TITLE}]({CORPUS_ABOUT}.md)"]
    if corpus is not None:
        for source in corpus.sources:
            _write_page(
                f"{CORPUS_DIRNAME}/{source.slug}.md",
                render(TEMPLATES["T20"], corpus_source.source_page(corpus, source)),
            )
            entries.append(f"- [{_nav_label(source.title)}]({source.slug}.md)")
        write_instances(corpus, snapshot)
        # The licence texts the data licence facts link to, as the release ships them.
        for licence, text in corpus.licence_texts:
            with mkdocs_gen_files.open(f"licenses/{licence}.txt", "w") as file:
                file.write(text)
    _write_page(f"{CORPUS_DIRNAME}/SUMMARY.md", "\n".join(entries) + "\n")


def _summary(page: str) -> str:
    """One page's opening sentence, wherever this build can read the page.

    Most pages are written here; the few authored directly in the site's own
    document tree are read from it, so a landing describes those pages from
    their own text too.
    """
    if page not in SUMMARIES:
        source = SITE_DIR / "docs" / page
        if source.is_file():
            _record_summary(page, source.read_text(encoding="utf-8"))
    return SUMMARIES.get(page, "")


def _area_branch(nav: Sequence[Any], area: str) -> list[Any]:
    """The navigation branch whose landing is *area*'s own index."""
    branch = branch_for(nav, f"{area}/index.md")
    if not branch:
        raise RuntimeError(f"The navigation declares no {area}/ area to list.")
    return branch


def _landing_rows(branch: Sequence[Any], area: str) -> tuple[tuple[Link, str], ...]:
    """One row per page the area declares, each carrying its own sentence."""
    return tuple(
        (item, _summary(f"{area}/{item.url}")) for item in area_pages(branch, area)
    )


def write_reference_index() -> None:
    """List the API categories without duplicating their introductory prose."""
    nav = FilesEditor.current().config["nav"]
    branch = _area_branch(nav, REFERENCE_DIRNAME)
    rows = [
        (link(item.text, item.url),)
        for item, summary in _landing_rows(branch, REFERENCE_DIRNAME)
    ]
    _write_page(
        f"{REFERENCE_DIRNAME}/index.md",
        "# API reference\n\n"
        + markdown_table((copy_span("reference", "section_column"),), rows)
        + "\n\n## Run and analyze a local experiment\n\n```python\n"
        + (LIB_DOCS / INTERACTIVE_WORKFLOW).read_text()
        + "```\n",
    )


def write_methodology(release: ReleaseInput | None, snapshot: Path | None) -> None:
    """Write the site-wide Methodology page: the release's narrative.

    The narrative renders as the release wrote it. A build reading no release
    says so, as the Corpus area does, rather than publishing a page about no
    release.
    """
    if release is None:
        record = methodology_page(None, "", banner=copy_text("corpus", "no_release"))
    else:
        narrative = read_narrative(snapshot, BENCHMARK_NARRATIVE_NAME)
        if narrative is None:
            print(f"methodology: {release.identity} ships no protocol narrative.")
        for target in release_relative_links(narrative or ""):
            print(
                f"methodology: the narrative links {target!r}, a file inside the "
                "release the site does not publish; it shows as text."
            )
        record = methodology_page(narrative, release.identity)
    _write_page(f"{METHODOLOGY_ROUTE}index.md", render(TEMPLATES["T15"], record))


def write_contact() -> None:
    """Write the Contact page: who to write to, then the addresses."""
    _write_page(
        f"{CONTACT_ROUTE}index.md",
        f"# {copy_text('contact', 'title')}\n\n"
        + copy_html("contact", "introduction")
        + "\n\n"
        + copy_html("contact", "addresses", classes="dp-contact-addresses"),
    )


def write_area_landings() -> None:
    """Write the landing of every area whose subject is its own pages.

    Tasks and Corpus list what they are about; Submit
    list the pages they carry, which this build's navigation already declares.
    The landing is written last, so every page it names has been written and
    can supply the sentence it opens on.
    """
    nav = FilesEditor.current().config["nav"]
    for area in PAGE_LIST_AREAS:
        branch = _area_branch(nav, area)
        _write_page(
            f"{area}/index.md",
            render(
                TEMPLATES["T2"],
                AreaIndex(
                    title=next(iter(branch[0])) if branch else area.title(),
                    pages=_landing_rows(branch, area),
                ),
            ),
        )


def _nav_label(title: str) -> str:
    """One navigation entry's text, safe inside a Markdown link label."""
    return title.replace("[", "(").replace("]", ")")


def write_task_space(
    task: Task,
    concept_src: Path,
    routes: Sequence[ResultsRoute],
    surfaces: Sequence[str],
    roster: Mapping[str, Mapping[str, RosterLink]],
    corpus: corpus_source.Corpus | None,
    comparison: tuple[ReleasedComparison, Sequence[str]] | None = None,
) -> None:
    """Write one task's space: every page an instance of exactly one template.

    *comparison* is the task's protocol entry and the Metrics its Explore
    lists, where this build renders its results, so its Metrics page links
    each of those Metrics' baselines.
    """
    prefix = task.id
    head, sections = (
        _authored_sections(concept_src, task.id) if concept_src.exists() else ("", {})
    )
    problem = _authored_problem(concept_src, sections)

    semantics = Link(
        copy_text("component", "semantics_link"),
        "../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation.md",
    )
    exported = exported_transformations(task.id)
    metric_sections = library_source.metrics(
        task, None if comparison is None else _metric_baselines(task, *comparison)
    )
    pages: list[str] = []

    _write_page(
        f"{prefix}/definition.md",
        render(TEMPLATES["T6"], library_source.definition(task)),
    )
    pages.append("- [Definition](definition.md)")
    if task.id == "m_estimation":
        _write_page(
            f"{prefix}/definition/MEstimationLoss.md",
            "# MEstimationLoss\n\n"
            "::: dpstats.tasks.m_estimation.losses.MEstimationLoss\n"
            "    options:\n"
            "      show_root_heading: false\n"
            "      members: [__call__, gradient, hessian, properties]\n"
            "      heading_level: 2\n",
        )
        # The loss contract every task's own losses implement.
        pages.append("- [Loss contract](definition/MEstimationLoss.md)")
    if metric_sections.metrics:
        _write_page(f"{prefix}/metrics.md", render(TEMPLATES["T18"], metric_sections))
        pages.append("- [Metrics](metrics.md)")

    algorithms = list_built_in_algorithms(task)
    if algorithms:
        _write_page(
            f"{prefix}/algorithms/index.md",
            render(
                TEMPLATES["T7"],
                library_source.algorithms_index(task, lambda name: f"{name}/"),
            ),
        )
        entries = ["- [Native algorithms](index.md)"]
        for name, algorithm in algorithms.items():
            page = library_source.component_page(
                task,
                name,
                algorithm,
                task_link=Link(task.name, "../index.md"),
                semantics=semantics,
                roster=_component_roster(
                    algorithm, roster, task.id, f"{prefix}/algorithms/{name}.md"
                ),
            )
            _write_page(f"{prefix}/algorithms/{name}.md", render(TEMPLATES["T9"], page))
            entries.append(f"- [{name}]({name}.md)")
        _write_page(f"{prefix}/algorithms/SUMMARY.md", "\n".join(entries) + "\n")
        pages.append("- [Native algorithms](algorithms/)")

    if exported:
        _write_page(
            f"{prefix}/transformations/index.md",
            render(
                TEMPLATES["T8"],
                library_source.transformations_index(
                    task,
                    exported,
                    lambda name: f"{name}/",
                    lambda inner: f"../../{inner}/",
                ),
            ),
        )
        entries = ["- [Transformations](index.md)"]
        for name in sorted(exported):
            transformation = exported[name]
            page = library_source.component_page(
                task,
                name,
                transformation,
                task_link=Link(task.name, "../index.md"),
                inner_link=Link(
                    transformation.INNER_TASK.name,
                    f"../../{transformation.INNER_TASK.id}/index.md",
                ),
                semantics=semantics,
                roster=_component_roster(
                    transformation,
                    roster,
                    task.id,
                    f"{prefix}/transformations/{name}.md",
                ),
            )
            _write_page(
                f"{prefix}/transformations/{name}.md", render(TEMPLATES["T9"], page)
            )
            entries.append(f"- [{name}]({name}.md)")
        _write_page(f"{prefix}/transformations/SUMMARY.md", "\n".join(entries) + "\n")
        pages.append("- [Transformations](transformations/)")

    quick = [Link(route.title, f"{route.route}/index.md") for route in routes]
    if any(entry == "- [Roster](roster/index.md)" for entry in surfaces):
        quick.append(Link("Roster", "roster/index.md"))
    if any(entry == "- [Methodology](methodology/index.md)" for entry in surfaces):
        quick.append(Link("Methodology", "methodology/index.md"))
    quick.append(Link("Definition", "definition.md"))
    if metric_sections.metrics:
        quick.append(Link("Metrics", "metrics.md"))
    if algorithms:
        quick.append(Link("Native algorithms", "algorithms/index.md"))
    if exported:
        quick.append(Link("Transformations", "transformations/index.md"))
    metric_links = [
        (Link(name, f"metrics.md#{anchor}"), gloss)
        for name, anchor, gloss in library_source.metric_lines(task)
    ]
    _write_page(
        f"{prefix}/index.md",
        render(
            TEMPLATES["T3"],
            library_source.task_overview(task, quick, problem, metric_links),
        ),
    )

    if sections:
        # The guide is in no navigation and no template produces it: it is
        # kept only because it still holds per-Algorithm and per-Transformation
        # prose nothing else carries. Until that prose moves, search does not
        # send a reader to it either.
        guide = (
            head.strip()
            + "\n\n"
            + "\n\n".join(
                f"## {heading}\n{body}".rstrip() for heading, body in sections.items()
            )
        )
        _write_page(f"{prefix}/guide.md", guide.strip() + "\n", searchable=False)
    # Nested literate-nav sections: Benchmark only where the task has a
    # results surface, Reference always.
    summary = [f"- [{task.name}](index.md)"]
    if surfaces:
        summary.append("- Benchmark")
        summary.extend(f"    {entry}" for entry in surfaces)
    summary.append("- Reference")
    summary.extend(f"    {entry}" for entry in pages)
    _write_page(f"{prefix}/SUMMARY.md", "\n".join(summary) + "\n")


def _authored_sections(source: Path, task_id: str) -> tuple[str, dict[str, str]]:
    """One task's authored page, split into its opening and its ``##`` sections."""
    text = _publish_links(source.read_text(), f"tasks/{task_id}.md")
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    headings = parts[1::2]
    sections = dict(zip(headings, parts[2::2]))
    if len(sections) != len(headings):
        raise ValueError(f"Duplicate section headings in {source}")
    return parts[0], sections


def _authored_problem(source: Path, sections: dict[str, str]) -> str:
    """The task's problem statement, taken out of its authored page.

    The statement is the overview's own section, so it is removed from the
    sections the guide keeps: one source renders in exactly one place.
    """
    if PROBLEM_SECTION in sections:
        raise RuntimeError(f"{source} duplicates its sanctioned problem formulation")
    relative = f"content/tasks/{source.stem}/problem-formulation.md"
    content = LIB_DOCS / relative
    heading, _, body = content.read_text(encoding="utf-8").partition("\n")
    if heading != "## Problem formulation" or not body.strip():
        raise RuntimeError(f"{content} requires a problem formulation heading and body")
    return _publish_links(body.strip(), relative)


def _component_roster(
    component: type,
    roster: Mapping[str, Mapping[str, RosterLink]],
    task: str,
    page: str,
) -> list[RosterLink]:
    """The task Rosters holding configured algorithms that use one component class.

    The page's own task comes first, then the others by how many they hold.
    The links are measured from the site root, and this raw href is not one
    MkDocs rewrites, so each is re-measured from the directory the page itself
    is served at.

    Args:
        component: The Algorithm or Transformation class the page documents.
        roster: Each class path's link into each task's Roster.
        task: The slug of the task whose space publishes the page.
        page: The page's path under the document root, ending in ``.md``.
    """
    # ``a/b/c.md`` is served at ``a/b/c/``, so the page sits that many
    # directories below the root whatever depth the task space has.
    climb = "../" * len(page.split("/"))
    links = roster.get(f"{component.__module__}.{component.__qualname__}", {})
    ordered = sorted(
        links,
        key=lambda slug: (slug != task, -links[slug].count, TASK_SLUGS.index(slug)),
    )
    return [replace(links[slug], url=climb + links[slug].url) for slug in ordered]


def home_leaderboards(
    panels: Sequence[tuple[str, str, str, str]], *, depth: int
) -> str:
    """Compose the tabbed preview Home carries, wherever that page ships.

    Args:
        panels: One ``(slug, title, links, body)`` per Benchmarked task, in the
            order the tabs present them.
        depth: How many path segments separate the carrying page from the site
            root, so it reaches the scripts that ship there.

    Returns:
        The tab strip, one panel per task, and the scripts that drive them.
    """
    tabs = "".join(
        f'<button type="button" class="nav-link{" active" if index == 0 else ""}" role="tab" id="home-tab-{slug}" '
        f'data-bs-toggle="tab" data-bs-target="#home-panel-{slug}" '
        f'aria-controls="home-panel-{slug}" aria-selected="{str(index == 0).lower()}" '
        f'tabindex="{0 if index == 0 else -1}">{html.escape(title)}</button>'
        for index, (slug, title, _, _) in enumerate(panels)
    )
    # Bootstrap's tabs show the active pane and switch on a tab's click or arrow key.
    sections = [
        f'<section class="dp-home-panel tab-pane{" active" if index == 0 else ""}" '
        f'id="home-panel-{slug}" role="tabpanel" aria-labelledby="home-tab-{slug}" tabindex="0">'
        f'<header class="dp-home-task-heading"><h2>{html.escape(title)}</h2>'
        f"<div>{links}</div></header>{body}</section>"
        for index, (slug, title, links, body) in enumerate(panels)
    ]
    prefix = "../" * depth
    return "".join(
        [
            '<div class="dp-home" data-home-leaderboards'
            + copy_attribute("home")
            + ">",
            '<div class="dp-task-tabs nav nav-underline" role="tablist" '
            f'aria-label="{html.escape(copy_text("home", "tabs_label"), quote=True)}" '
            f"data-home-tabs>{tabs}</div>",
            '<div class="tab-content">',
            *sections,
            "</div>",
            f'</div><script src="{prefix}javascripts/home.js"></script>',
            runtime_scripts(depth, "javascripts/leaderboard.js"),
        ]
    )


def write_home(
    results: Mapping[str, tuple[ResultsRoute, ...]], bodies: Mapping[str, str]
) -> None:
    """Embed each Benchmarked task's opening Leaderboard, one task at a time.

    The release declares which tasks those are, so a build reading no release
    embeds none and says so. Each body was rendered from the same source as its
    full Leaderboard, with site-root-relative data URLs and the landing layout
    already in its HTML.
    """
    tasks = [(slug, title) for slug, title, _ in TASKS if slug in results]
    tasks.sort(key=lambda task: task[0] != REFERENCE_TASK)
    panels = []
    for slug, title in tasks:
        body = bodies.get(slug) or copy_html(
            "home", "task_no_standings", classes="dp-empty"
        )
        links = f'<a href="{slug}/">' + copy_span("home", "task_link") + "</a>"
        for route in results.get(slug, ()):
            links += (
                f' · <a href="{slug}/{route.route}/">{html.escape(route.title)}</a>'
            )
        if slug in bodies:
            body = home_context(slug) + body
        panels.append((slug, title, links, body))
    content = (
        home_leaderboards(panels, depth=0)
        if panels
        else copy_html("home", "no_release", classes="dp-empty")
    )
    source = (SITE_DIR / "docs" / "index.md").read_text()
    # The home page's opening text. A bare [Leaderboard] or [Explore] names that
    # page of the reference task, or plain text where the build has none.
    intro = (LIB_DOCS / "content" / "home.md").read_text(encoding="utf-8").strip()
    for route in results.get(REFERENCE_TASK, ()):
        intro = re.sub(
            rf"\[{re.escape(route.title)}\](?!\()",
            f"[{route.title}]({REFERENCE_TASK}/{route.route}/index.md)",
            intro,
        )
    for _, title in RESULTS_SURFACES:
        intro = re.sub(rf"\[{re.escape(title)}\](?!\()", title, intro)
    with mkdocs_gen_files.open("index.md", "w") as page:
        page.write(
            source.replace("<!-- HOME_INTRO -->", intro).replace(
                "<!-- TASK_LEADERBOARDS -->", content
            )
        )


NAV_TARGETS = navigation_targets(FilesEditor.current().config["nav"])
API_GROUPS = shared_reference_groups(FilesEditor.current().config["nav"], LIB_DOCS)
API_PAGES, API_ROUTES = reference_pages(API_GROUPS)

# Where a displayed result's component classes are documented now: the task's
# own Algorithm and Transformation pages. Built before any results surface renders,
# because the provenance popover links component names to these pages.
METHOD_ROUTES: dict[str, str] = {}
for _slug in TASK_SLUGS:
    _task = BUILT_IN_TASKS[_slug]
    for _name, _component in list_built_in_algorithms(_task).items():
        _route = f"{_slug}/algorithms/{_name}.md"
        METHOD_ROUTES[f"{PACKAGE}.tasks.{_slug}.algorithms.{_name}"] = _route
        METHOD_ROUTES[f"{_component.__module__}.{_component.__qualname__}"] = _route
    for _name, _component in exported_transformations(_slug).items():
        _route = f"{_slug}/transformations/{_name}.md"
        METHOD_ROUTES[f"{PACKAGE}.tasks.{_slug}.transformations.{_name}"] = _route
        METHOD_ROUTES[f"{_component.__module__}.{_component.__qualname__}"] = _route

# A component's own class page owns it; the API page owns everything else, so
# compositions and tuners without a profile still reach their documentation.
COMPONENT_ROUTES = {**API_ROUTES, **METHOD_ROUTES}
RECREATE_REVISION_ENV = "DOCS_RECREATE_RELEASE_REVISION"
# The one input, resolved and prepared before any page is written: every
# results route then reads that snapshot, so nothing is fetched or measured
# while a site is assembled.
PIN = read_benchmark_lock(DOCS_ROOT / LOCK_NAME)
# A hosted preview's readers cannot open the local tree it was built from, so
# its recreate notebooks fetch the Distribution Release revision that tree is.
RECREATE_REVISION = os.environ.get(RECREATE_REVISION_ENV, "")
if RECREATE_REVISION and not is_immutable_revision(RECREATE_REVISION):
    raise ValueError(
        f"{RECREATE_REVISION_ENV}={RECREATE_REVISION!r} is not one full commit hash."
    )
RELEASE = release_input(
    PIN,
    os.environ.get(LOCAL_RELEASE_ENV),
    cache_dir=CACHE_DIR / "releases",
    current_protocol=os.environ.get(CURRENT_PROTOCOL_ENV),
)
SNAPSHOT = None if RELEASE is None else prepared_snapshot(RELEASE)
write_library()
write_templates()
write_tutorials()
routes = results_routes(RELEASE)
CORPUS = read_corpus(RELEASE, SNAPSHOT)
home = write_tasks(routes, SNAPSHOT, CORPUS)
if SNAPSHOT is not None:
    print(f"build summary: {write_build_summary(SNAPSHOT)}")
write_corpus(CORPUS, SNAPSHOT)
write_methodology(RELEASE, SNAPSHOT)
write_contact()
write_area_landings()
write_home(routes, home)
