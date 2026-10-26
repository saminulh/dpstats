"""Whole-build sweeps: what a reader must never find on any published page.

These read a finished build rather than one renderer, because the defects they
guard against are ones a page can acquire from any of the routes that write it.
The build is supplied through ``DOCS_BUILT_SITE``; without one they skip.
"""

from __future__ import annotations

import html
import json
import keyword
import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest
from bs4 import BeautifulSoup, Tag
from bs4.element import PreformattedString

from dpstats.tasks import BUILT_IN_TASKS
from templates.conformance import UNTEMPLATED_TASK_PAGES
from tests.unit.docs.conftest import Element, parse_html
from tutorials import UNRUN_NOTICE_CLASS

# Markup that belongs to the source a page was generated from, never to the
# page: reStructuredText cross-reference roles and double-backtick literals.
SOURCE_MARKUP = re.compile(r":(?:class|func|meth|mod|attr|data|exc|obj):|``")
VERBATIM = re.compile(
    r"<pre\b.*?</pre>|<script\b.*?</script>|<style\b.*?</style>", re.S
)
TEMPLATES_PREFIX = "templates/"


def _visible(page: Path) -> str:
    """One page's reader-visible text, without its scripting."""
    stripped = VERBATIM.sub(" ", page.read_text(errors="replace"))
    return html.unescape(re.sub(r"<[^>]+>", " ", stripped))


def _pages(site: Path) -> list[Path]:
    return sorted(site.rglob("index.html"))


def test_shared_api_objects_are_searchable_in_their_current_categories(
    built_site: Path,
) -> None:
    for category, identifier in (
        ("collections", "dpstats.TaskCollection"),
        ("sweep", "dpstats.sweep.BudgetPolicy"),
        ("domains", "dpstats.domains.Domain"),
        ("metric", "dpstats.metric.Metric"),
        ("instance", "dpstats.instance.Instance"),
        ("algorithm", "dpstats.algorithm.Transformation"),
        ("parameters", "dpstats.parameters.parameter_field"),
        (
            "simulations",
            "dpstats.simulations.linear_regression.simulate_clipped_gaussian",
        ),
    ):
        page = built_site / f"library/reference/{category}/{identifier}/index.html"
        assert "data-pagefind-body" in page.read_text(encoding="utf-8")
    directory = (built_site / "library/reference/domains/index.html").read_text()
    sidebar = directory.split("<aside ", 1)[1].split("</aside>", 1)[0]
    assert "dpstats.domains.Domain/" in sidebar
    assert "<details open>" in sidebar


def test_no_page_shows_the_markup_of_the_source_it_came_from(built_site: Path) -> None:
    offenders = {
        str(page.relative_to(built_site)): sorted(
            set(SOURCE_MARKUP.findall(_visible(page)))
        )
        for page in _pages(built_site)
        if SOURCE_MARKUP.search(_visible(page))
    }
    assert offenders == {}


FRONT_MATTER_TEXT = re.compile(r"(?m)^\s*(?:title|hide|search):\s")


def test_no_page_shows_its_front_matter(built_site: Path) -> None:
    """A second front-matter block renders as the page's opening text."""
    offenders = []
    for page in _pages(built_site):
        text = page.read_text(errors="replace")
        start = text.find("<article")
        if start < 0:
            continue
        article = text[start : text.find("</article>", start)]
        visible = html.unescape(re.sub(r"<[^>]+>", "\n", VERBATIM.sub(" ", article)))
        if FRONT_MATTER_TEXT.search(visible):
            offenders.append(str(page.relative_to(built_site)))
    assert offenders == []


def test_every_configured_algorithm_link_reaches_a_page_in_this_build(
    built_site: Path,
) -> None:
    """A configured algorithm names its page, so its link has to arrive at one."""
    checked = 0
    dead: list[str] = []
    for page in _pages(built_site):
        if page.parent.parent.name not in ("algorithms", "transformations"):
            continue
        for href in re.findall(
            r'href="((?:\.\./)+[^"#]*)"', page.read_text(errors="replace")
        ):
            if "roster" not in href:
                continue
            checked += 1
            route = urlsplit(href).path
            if not (page.parent / route / "index.html").resolve().exists():
                dead.append(f"{page.relative_to(built_site)} -> {href}")
    assert checked, "this build publishes no configured algorithm links to check"
    assert dead == []


def test_a_build_that_did_not_ask_for_them_writes_no_page_template(
    public_site: Path,
) -> None:
    """Without the flag the filler pages are not built, so none can ship."""
    assert not (public_site / TEMPLATES_PREFIX).exists()
    sitemap = (public_site / "sitemap.xml").read_text(encoding="utf-8")
    assert f"/{TEMPLATES_PREFIX}" not in sitemap
    assert [
        page
        for page in _pages(public_site)
        if "dp_filler" in page.read_text(errors="replace")
    ] == []


def test_the_page_template_previews_are_not_offered_to_a_reader(
    built_site: Path,
) -> None:
    """Filler objects are a design preview: never crawled, never searchable."""
    sitemap = (built_site / "sitemap.xml").read_text(encoding="utf-8")
    assert f"/{TEMPLATES_PREFIX}" not in sitemap

    assert [
        page
        for page in (built_site / TEMPLATES_PREFIX).rglob("*.html")
        if "data-pagefind-body" in page.read_text(encoding="utf-8")
    ] == []


def test_the_landing_rankings_table_hides_no_column_it_rendered(
    built_site: Path,
) -> None:
    """Home shows fewer columns by rendering fewer, not by hiding cells.

    Home embeds one rankings table per Benchmarked task and privacy budget, all
    server-rendered (only the opening pair is unhidden), so more than one may
    carry a "Rank" header; every one of them still has to carry exactly the
    landing shape, never the full Leaderboard's Coverage column.
    """
    home = (built_site / "index.html").read_text(encoding="utf-8")
    # Each heading as the slot that writes it, else its text: the shape, not
    # the wording, is what this checks.
    tables = [
        [
            next(
                (
                    node.attrs["data-copy-slot"]
                    for node in header.find_all("data-copy-slot")
                ),
                header.text,
            )
            for header in table.find(tag="thead").find_all(tag="th")
        ]
        for table in parse_html(home).find_all(tag="table")
    ]
    rank_tables = [
        headers for headers in tables if "leaderboard.column_rank" in headers
    ]
    assert rank_tables, "Home renders no rankings table"
    assert all(
        headers
        == ["leaderboard.column_rank", "Algorithm", "leaderboard.column_headline"]
        for headers in rank_tables
    )
    assert "Released" not in home

    stylesheets = " ".join(
        path.read_text(encoding="utf-8")
        for path in (built_site / "stylesheets").glob("*.css")
    )
    assert "display: none" not in _home_table_rules(stylesheets)


def _home_table_rules(css: str) -> str:
    """Every rule whose selector reaches the landing rankings table."""
    return " ".join(
        rule
        for rule in re.findall(r"[^{}]*\{[^{}]*\}", css)
        if ".dp-home" in rule and ".dp-leaderboard" in rule
    )


TASK_SLUGS = tuple(BUILT_IN_TASKS)

# The Tabler shell's own section sidebar and the breadcrumb's way back to the
# task index, named once so the drawer test below and any other reader of this
# shell key on the same hooks the theme (docs/site/overrides/main.html) emits.
SIDEBAR_ID = "section-sidebar"
BREADCRUMB_LABEL = "Breadcrumb"
TASKS_AREA_TITLE = "Tasks"
# The fragment site.js reads to open the global task picker on Home.
TASK_MENU = "task-menu"


_NO_SIDEBAR = "<no #section-sidebar>"
_SITE_ROOT = "https://dpstats.invalid/"

# Site-wide sidebar links a task's drawer may legitimately carry outside its
# own task space (e.g. "Home", "Library"), named explicitly rather than
# excusing any link this check cannot resolve under the task. No page in
# this shell's task drawers emits one today.
SITE_WIDE_SIDEBAR_LABELS: frozenset[str] = frozenset()


def _resolved_route(page_dir: str, href: str) -> str:
    """*href*, found on a page at *page_dir*, resolved to a site-root path."""
    return urlsplit(urljoin(_SITE_ROOT + page_dir, href)).path.removeprefix("/")


def _opens_task_menu(page_dir: str, href: str) -> bool:
    """Whether *href* reaches Home with its task picker open, as Tasks must."""
    target = urlsplit(urljoin(_SITE_ROOT + page_dir, href))
    return target.path in ("/", "/index.html") and target.fragment == TASK_MENU


def _task_drawer_leak(
    page_dir: str, route: str, slug: str, document: Element
) -> tuple[bool, list[str] | None]:
    """Whether one task-space page's parsed document was checked, and its leak.

    `route in UNTEMPLATED_TASK_PAGES` (the authored guide and walkthrough)
    carries no navigation ancestors at all — not checked, never a leak. Every
    other templated task page must carry `#section-sidebar`; missing it
    entirely is itself the leak (`_NO_SIDEBAR`), not a page this check skips.

    Otherwise each sidebar link's `href` is resolved against *page_dir* (this
    page's own site-root directory) to the route it addresses, and must land
    under `f"{slug}/"` — Tabler's drawer labels are generic ("Overview",
    "Definition", "Algorithms") and identical across every task, so matching
    by label alone cannot tell one task's drawer from another's carrying the
    same labels. `SITE_WIDE_SIDEBAR_LABELS` names any link legitimately
    outside the task space. The leak is the foreign links' labels, empty when
    the only defect is a missing breadcrumb "Tasks" link, or `None` when
    clean.
    """
    sidebars = document.find_all(tag="aside", id=SIDEBAR_ID)
    if not sidebars:
        if route in UNTEMPLATED_TASK_PAGES:
            return False, None
        return True, [_NO_SIDEBAR]
    foreign: list[str] = []
    for link in sidebars[0].find_all(tag="a"):
        href = link.attrs.get("href")
        if href is None or link.text in SITE_WIDE_SIDEBAR_LABELS:
            continue
        if not _resolved_route(page_dir, href).startswith(f"{slug}/"):
            foreign.append(link.text)
    breadcrumbs = document.find_all(tag="ol", aria_label=BREADCRUMB_LABEL)
    back_to_tasks = route in UNTEMPLATED_TASK_PAGES or any(
        link.text == TASKS_AREA_TITLE
        and _opens_task_menu(page_dir, link.attrs.get("href") or "")
        for link in breadcrumbs[0].find_all(tag="a")
    )
    if foreign or not back_to_tasks:
        return True, foreign
    return True, None


def test_a_task_space_page_is_given_only_that_task_s_drawer(built_site: Path) -> None:
    """Inside a task, the sidebar is that task's pages, and the breadcrumb the way back.

    The Tabler shell has no nested or paired drawer: `overrides/main.html`
    renders one flat `<ul>` of the active task's own pages inside
    `#section-sidebar`, and the way back to every task lives in the
    breadcrumb's own "Tasks" link instead of an "All tasks" sidebar entry.
    `_task_drawer_leak` carries the per-page rule; see its own docstring.
    """
    checked = 0
    leaking: list[tuple[str, list[str]]] = []
    for page in _pages(built_site):
        slug = page.relative_to(built_site).parts[0]
        if slug not in TASK_SLUGS:
            continue
        text = page.read_text(errors="replace")
        route = "/".join(page.relative_to(built_site).parts[1:-1]) + "/"
        page_dir = "/".join(page.relative_to(built_site).parts[:-1]) + "/"
        document = parse_html(text)
        evaluated, leak = _task_drawer_leak(page_dir, route, slug, document)
        if not evaluated:
            continue
        checked += 1
        if leak is not None:
            leaking.append((str(page.relative_to(built_site)), leak))
    assert checked, "this build publishes no task-space pages to check"
    assert leaking == []


def test_a_templated_task_page_missing_its_sidebar_is_a_leak() -> None:
    """A page that lost `#section-sidebar` entirely must fail, not be skipped.

    Regression for a hole where `if not sidebars: continue` silently passed
    over any templated task-space page that lost its navigation outright.
    """
    document = parse_html("<html><body><p>no drawer here</p></body></html>")
    evaluated, leak = _task_drawer_leak(
        f"{TASK_SLUGS[0]}/leaderboard/", "leaderboard/", TASK_SLUGS[0], document
    )
    assert evaluated
    assert leak == [_NO_SIDEBAR]


def test_an_untemplated_page_missing_its_sidebar_is_exempt() -> None:
    """`UNTEMPLATED_TASK_PAGES` carries no drawer by design; not a leak."""
    route = next(iter(UNTEMPLATED_TASK_PAGES))
    document = parse_html("<html><body><p>no drawer here</p></body></html>")
    evaluated, leak = _task_drawer_leak(
        f"{TASK_SLUGS[0]}/{route}", route, TASK_SLUGS[0], document
    )
    assert not evaluated
    assert leak is None


def test_a_sidebar_link_into_another_task_is_a_leak_despite_a_generic_label() -> None:
    """A drawer link resolving outside the current task is foreign, by route.

    Tabler's drawer labels are the same generic text ("Algorithms") in every
    task, so a page whose sidebar actually links into another task's routes
    must still fail even though every label it carries also names a page of
    the *current* task.
    """
    this_task, other_task = TASK_SLUGS[0], TASK_SLUGS[1]
    document = parse_html(
        '<aside id="section-sidebar"><a href="../">Overview</a>'
        f'<a href="../../{other_task}/algorithms/">Algorithms</a></aside>'
        '<ol aria-label="Breadcrumb"><a>Tasks</a></ol>'
    )
    evaluated, leak = _task_drawer_leak(
        f"{this_task}/algorithms/", "algorithms/", this_task, document
    )
    assert evaluated
    assert leak == ["Algorithms"]


def test_the_tasks_breadcrumb_must_open_home_s_task_menu() -> None:
    """A "Tasks" link elsewhere is no way back to every task: the leak is empty."""
    slug = TASK_SLUGS[0]

    def leak(href: str) -> list[str] | None:
        document = parse_html(
            '<aside id="section-sidebar"><a href="../">Overview</a></aside>'
            f'<ol aria-label="Breadcrumb"><a href="{href}">Tasks</a></ol>'
        )
        page_dir = f"{slug}/algorithms/"
        return _task_drawer_leak(page_dir, "algorithms/", slug, document)[1]

    assert leak("../../index.html#task-menu") is None
    assert leak("../../#task-menu") is None
    assert leak("../../tasks/") == []
    assert leak("../../index.html") == []


from tutorials import published_tutorials

REPOSITORY = Path(__file__).resolve().parents[3]


def _identifiers(notebook: Path) -> set[str]:
    """The longer names a notebook's own code cells use."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    return {
        name
        for name in re.findall(r"\b[A-Za-z_]\w{5,}\b", code)
        if not keyword.iskeyword(name)
    }


def _tutorial_pages() -> dict[str, str]:
    """Each tutorial's route, and a name only its own notebook's code uses.

    A page is checked against its notebook rather than against a page count:
    a page carrying that name rendered its own cells, not a sibling's.
    """
    names = {
        notebook.stem: _identifiers(notebook)
        for notebook in published_tutorials(
            REPOSITORY / "docs", REPOSITORY / "tutorials"
        )
    }
    distinct: dict[str, str] = {}
    for stem, own in names.items():
        others = set().union(*(used for name, used in names.items() if name != stem))
        distinct[stem] = sorted(own - others)[0]
    return distinct


# The class nbconvert puts on a cell's output area, so a page carrying none is
# an unexecuted notebook rather than the executed copy the harness wrote.
EXECUTED_OUTPUT = 'class="jp-OutputArea'


def test_every_tutorial_page_is_its_notebook_with_the_download_beside_it(
    built_site: Path,
) -> None:
    """Each tutorial renders its own cells and offers the notebook to run."""
    for stem, cell in _tutorial_pages().items():
        directory = built_site / "library" / "tutorials" / stem
        page = directory / "index.html"
        assert page.is_file(), f"{stem} publishes no page"
        body = page.read_text(errors="replace")
        assert cell in body, f"{stem} renders none of its own cells"
        assert (
            f'href="{stem}.ipynb"' in body
        ), f"{stem} offers no download of its notebook"
        offered = directory / f"{stem}.ipynb"
        assert offered.is_file(), f"{stem} links a notebook this build did not write"
        source = json.loads(offered.read_text())
        assert all(
            written.get("outputs", []) == []
            for written in source["cells"]
            if written["cell_type"] == "code"
        ), f"{stem} offers the executed copy rather than the source notebook"

    companion = built_site / "library/tutorials/run_your_own_script"
    for script in (
        "my_regression_script.py",
        "my_regression.py",
        "my_regression_annotated.py",
    ):
        assert (
            companion / script
        ).is_file(), f"run_your_own_script offers no download of {script}"

    # compare_on_the_benchmark.ipynb names only my_regression.py; my_regression.py's own
    # import of my_regression_script reaches that further sibling too, so
    # its download must be offered here as well, not just where it is named.
    transitive = built_site / "library/tutorials/compare_on_the_benchmark"
    assert (transitive / "my_regression_script.py").is_file(), (
        "compare_on_the_benchmark offers no download of my_regression_script.py, "
        "reached only through my_regression.py's own import"
    )


def test_the_user_guide_links_every_chapter(built_site: Path) -> None:
    """The Library index links each tutorial chapter by its one route."""
    body = (built_site / "library/index.html").read_text()
    assert 'class="dp-guide-chapters"' not in body
    for route in (
        "tutorials/install_and_fetch/",
        "tutorials/extend_with_transformations/",
        "tutorials/compare_on_the_benchmark/",
    ):
        assert f'href="{route}"' in body
    assert 'href="tutorials/run_your_own_script/"' in body
    for stem in (
        "private_hyperparameter_tuning",
        "add_your_own_data",
        "analyze_results",
    ):
        assert f"tutorials/{stem}/" not in body
        assert not (built_site / "library/tutorials" / stem).exists()


def test_a_tutorial_page_carries_the_outputs_its_notebook_produced(
    built_site: Path,
) -> None:
    """A page shows its outputs, or says why this build has none for it.

    A notebook whose precondition this build could not meet renders its cells
    under the harness's stated reason; a page with neither is an unexecuted
    notebook published as a tutorial.
    """
    silent: list[str] = []
    for stem in _tutorial_pages():
        body = (built_site / "library" / "tutorials" / stem / "index.html").read_text(
            errors="replace"
        )
        notices = [
            element
            for element in parse_html(body).find_all("class")
            if UNRUN_NOTICE_CLASS in element.attrs["class"].split()
        ]
        if EXECUTED_OUTPUT not in body and not notices:
            silent.append(stem)
    assert silent == []


def test_a_tutorial_figure_is_drawn_by_the_site_chart_runtime(
    built_site: Path,
) -> None:
    """The Plotly figure compare_on_the_benchmark shows is drawn on its page.

    nbconvert has no renderer for Plotly's notebook output, so without the
    page's own drawing the figure would vanish without an error.
    """
    page = built_site / "library/tutorials/compare_on_the_benchmark/index.html"
    body = page.read_text(errors="replace")
    if UNRUN_NOTICE_CLASS in body:
        pytest.skip("this build has no executed copy of the notebook")
    bundles = re.findall(r'<script\b[^>]* src="([^"?]*plotly[^"?]*\.js)[?"]', body)
    assert len(bundles) == 1
    assert (page.parent / bundles[0]).resolve().is_file()
    assert body.count('class="dp-notebook-figure"') == 1


def test_algorithm_notebook_has_no_second_generated_description(
    built_site: Path,
) -> None:
    """The source notebook owns the title and introduction."""
    body = (built_site / "library/tutorials/run_your_own_script/index.html").read_text()
    assert "Requires: <code>dpstats[" not in body
    assert "We assume you already have a script" in body


def test_every_chart_heading_is_the_plain_question_above_a_formula_line(
    built_site: Path,
) -> None:
    """Home, Leaderboard and Explore share one heading; budget families link on."""
    question = "How often does each algorithm succeed?"
    charts = 0
    dead: list[str] = []
    for page in _pages(built_site):
        text = page.read_text(errors="replace")
        classed = [
            (element, element.attrs.get("class", "").split())
            for element in parse_html(text).walk()
        ]
        headings = [e for e, names in classed if "dp-profile-heading" in names]
        formulas = [e for e, names in classed if "dp-profile-formula" in names]
        for heading in headings:
            charts += 1
            assert heading.text == question
        assert len(formulas) == len(headings)
        for href in re.findall(r'dp-budget-family"><a href="([^"]+)"', text):
            if not (page.parent / href / "index.html").resolve().exists():
                dead.append(f"{page.relative_to(built_site)} -> {href}")
    assert charts, "this build publishes no chart heading to check"
    assert dead == []


def test_every_chart_page_ships_the_maintainer_s_reader_terms(
    built_site: Path,
) -> None:
    """Each chart page carries every term of the maintainer's file, verbatim.

    A term's link is written from the site root and ships relative to its page,
    so it is compared where it lands, which must be a built page.
    """
    from reader_terms import READER_TERMS, load_reader_terms

    defined = load_reader_terms()
    # A term the maintainer has not yet defined renders as its plain word.
    assert defined and set(defined) <= set(READER_TERMS)
    pages = 0
    for page in _pages(built_site):
        text = page.read_text(errors="replace")
        shipped = re.findall(r'data-reader-terms="([^"]*)"', text)
        if not shipped:
            continue
        pages += 1
        here = f"/{page.relative_to(built_site).as_posix()}"
        for one in shipped:
            terms = json.loads(html.unescape(one))
            assert terms.keys() == defined.keys()
            for term, entry in terms.items():
                expected = defined[term]
                assert entry["text"] == expected["text"]
                assert ("link" in entry) == ("link" in expected)
                if "link" in entry:
                    landed = urlsplit(urljoin(here, entry["link"])).path
                    assert landed == f"/{expected['link']}", (page, term)
                    assert (built_site / landed.lstrip("/") / "index.html").is_file()
    assert pages, "this build publishes no chart page to check"


def test_every_task_export_renders_in_exactly_one_place(built_site: Path) -> None:
    """A class or function a task module exports has one page: its Algorithm or
    Transformation page, its task's Definition, or the API reference; never none and
    never two."""
    import importlib
    import inspect
    import pkgutil
    import zlib

    import dpstats.tasks as tasks
    from dpstats.tasks import BUILT_IN_TASKS

    raw = (built_site / "objects.inv").read_bytes()
    inventory: dict[str, set[str]] = {}
    for line in zlib.decompress(raw.split(b"\n", 4)[4]).decode().splitlines():
        name, _, _, uri, *_ = line.split(" ")
        inventory.setdefault(name, set()).add(uri.partition("#")[0])
    exports: dict[int, tuple[object, set[str], str]] = {}
    for found in pkgutil.walk_packages(tasks.__path__, f"{tasks.__name__}."):
        module = importlib.import_module(found.name)
        slug = found.name.split(".")[2]
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if inspect.isclass(value) or inspect.isfunction(value):
                entry = exports.setdefault(id(value), (value, set(), slug))
                entry[1].add(f"{module.__name__}.{name}")
    unplaced, doubled = [], []
    for value, paths, slug in exports.values():
        name = value.__name__
        paths.add(f"{value.__module__}.{value.__qualname__}")
        homes = {uri for path in paths for uri in inventory.get(path, ())}
        for kind in ("algorithms", "transformations"):
            if (built_site / slug / kind / name / "index.html").is_file():
                homes.add(f"{slug}/{kind}/{name}/")
        task = BUILT_IN_TASKS.get(slug)
        if task is not None and value in (task.Input, task.Metadata, task.Output):
            homes.add(f"{slug}/definition/")
        if not homes:
            unplaced.append(sorted(paths)[0])
        elif len(homes) > 1:
            doubled.append((sorted(paths)[0], sorted(homes)))
    assert not unplaced, f"rendered nowhere: {sorted(unplaced)}"
    assert not doubled, f"rendered more than once: {sorted(doubled)}"


def test_every_downloadable_notebook_links_to_this_build(built_site: Path) -> None:
    """A downloaded tutorial reaches the build it came from, never another host.

    Its links into the site are absolute on this build's address, since a
    downloaded file has no page to be relative to; scripts it names sit beside it.
    """
    from site_links import PUBLIC_SITE_URL

    root = os.environ.get("DOCS_SITE_URL") or PUBLIC_SITE_URL
    downloads = sorted((built_site / "library" / "tutorials").glob("*/*.ipynb"))
    assert downloads, "this build publishes no notebook download"
    stray = []
    for download in downloads:
        notebook = json.loads(download.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for target in re.findall(r"\]\(([^\s)]+)\)", "".join(cell["source"])):
                beside = (download.parent / target.partition("#")[0]).is_file()
                if target.startswith(("#", root)) or beside:
                    continue
                if "://" in target and "saminulh.github.io" not in target:
                    continue
                stray.append(f"{download.relative_to(built_site)}: {target}")
    assert not stray, stray


def test_a_home_paragraph_states_the_quantile_its_standings_rank_at(
    built_site: Path,
) -> None:
    """A Home paragraph's "τ@..." is the headline quantile of its task's ranking."""
    home = (built_site / "index.html").read_text(encoding="utf-8")
    panels = re.findall(
        r'id="home-panel-([\w-]+)".*?<div class="dp-home-reading">(.*?)</div>',
        home,
        re.S,
    )
    stated = 0
    for slug, reading in panels:
        found = re.search(r"τ@([0-9.]+)", html.unescape(reading))
        if found is None:
            continue
        stated += 1
        index = json.loads(
            (built_site / slug / "leaderboard" / "index.json").read_text("utf-8")
        )
        assert float(found[1]) == float(index["headline_quantile"]) * 100, slug
    assert stated, "no Home paragraph states the quantile its ranking uses"


def test_leaderboard_and_explore_point_to_their_task_baselines(
    built_site: Path,
) -> None:
    """Each results page's Baseline definition link reaches its task's Metrics page."""
    pages = [
        page
        for surface in ("leaderboard", "public-results")
        for page in built_site.glob(f"*/{surface}/index.html")
    ]
    assert pages, "this build publishes no Leaderboard or Explore page"
    for page in pages:
        slug = page.relative_to(built_site).parts[0]
        found = re.search(
            r'data-baseline-summary-link(?: hidden)? href="([^"#]*)',
            page.read_text("utf-8"),
        )
        assert found, page
        here = f"/{page.relative_to(built_site).as_posix()}"
        assert urlsplit(urljoin(here, found[1])).path == f"/{slug}/metrics/", page


def test_a_benchmarked_tasks_baselines_table_is_on_its_metrics_page(
    built_site: Path,
) -> None:
    """The baselines table lives on the Metrics page; no task has its own methodology."""
    slugs = [
        page.parent.parent.name for page in built_site.glob("*/leaderboard/index.html")
    ]
    assert slugs, "this build publishes no Leaderboard"
    for slug in slugs:
        markup = (built_site / slug / "metrics" / "index.html").read_text("utf-8")
        assert 'id="baseline-' in markup, slug
        for key in re.findall(r'<a href="#baseline-([^"]+)"', markup):
            assert f'id="baseline-{key}"' in markup, (slug, key)
        assert not (built_site / slug / "methodology").exists(), slug


def _definition_tables(built_site: Path) -> list[tuple[Path, str]]:
    """Every methodology, metrics and definition page's tables, with its page."""
    pages = [
        page
        for page in _pages(built_site)
        if page.parent.name in ("methodology", "metrics", "definition")
    ]
    assert pages, "this build publishes no methodology, metrics or definition page"
    return [(page, page.read_text(errors="replace")) for page in pages]


def test_no_definitions_or_baselines_table_prints_a_dash_for_an_absent_cell(
    built_site: Path,
) -> None:
    """E12: an absent field is omitted, never printed as unavailable."""
    dashes = [
        str(page.relative_to(built_site))
        for page, markup in _definition_tables(built_site)
        if "<td>—</td>" in markup
    ]
    assert dashes == []


def test_every_definition_link_lands_on_text(built_site: Path) -> None:
    """E12: a Definition link reaches a page, anchor and row that say something."""
    checked = 0
    empty: list[str] = []
    for page in _pages(built_site):
        markup = page.read_text(errors="replace")
        hrefs = re.findall(
            r'<a [^>]*href="([^"]*)"[^>]*>\s*(?:Full d|D)efinition\b', markup
        )
        hrefs += re.findall(r'data-(?:baseline-)?definition-href="([^"]*)"', markup)
        for href in hrefs:
            checked += 1
            target = urlsplit(urljoin(f"/{page.relative_to(built_site)}", href))
            path = built_site / target.path.lstrip("/")
            path = path / "index.html" if path.is_dir() else path
            reason = _landing_problem(path, target.fragment)
            if reason:
                empty.append(f"{page.relative_to(built_site)} -> {href}: {reason}")
    assert checked, "this build publishes no Definition link to check"
    assert empty == []


def _landing_problem(path: Path, fragment: str) -> str:
    """Why a link target has no text for its reader, or an empty string."""
    if not path.is_file():
        return "no such page"
    document = parse_html(path.read_text(errors="replace"))
    if not fragment:
        return "" if document.text.strip() else "empty page"
    anchors = [item for item in document.walk() if item.attrs.get("id") == fragment]
    if not anchors:
        return "no such anchor"
    row = next(
        (
            item
            for item in document.find_all(tag="tr")
            if any(child is anchors[0] for child in item.walk())
        ),
        None,
    )
    scope = row if row is not None else anchors[0]
    cells = [cell.text.strip() for cell in scope.find_all(tag="td")]
    if row is not None and not all(cells):
        return "its table row has an empty cell"
    return "" if scope.text.strip() else "anchor has no text"


def test_every_ranks_table_names_the_threshold_it_reads(built_site: Path) -> None:
    """E11: a configured page's success rate names the tau it is read at."""
    bare = [
        str(page.relative_to(built_site))
        for page in _pages(built_site)
        if page.parent.parent.name == "roster"
        and ">Success rate" in page.read_text(errors="replace")
        and ">Success rate at τ = " not in page.read_text(errors="replace")
    ]
    assert bare == []


# Glossary terms a reader meets in running text, which is lowercase except at
# the start of a sentence.
CAPITALISED_TERM = re.compile(
    r"(?<=[a-z0-9,;)] )(?:Instances?|Sources?|Trials?|Baselines?|Competitors?"
    r"|Problem instances?|Performance ratios?|Configured algorithms?)\b"
)
# Elements whose text is not running prose: page metadata, code and drawings.
# Text the site reproduces verbatim from a source it does not author, such as a
# bibliography entry or a release's shipped narrative, carries ``data-verbatim``
# (or ``data-dp-citation``) and keeps its source's case; its source is checked.
NOT_PROSE = frozenset({"head", "script", "style", "pre", "code", "svg", "math"})
# Elements that start a new run of text, so a term opening one is not mid-sentence.
BLOCKS = frozenset(
    "p li td th h1 h2 h3 h4 h5 h6 div dd dt figcaption caption br summary "
    "label option button section article aside nav header footer blockquote "
    "table tr ul ol details form fieldset legend".split()
)
# Pages written from library docstrings, which the API reference renders and the
# site code does not author.
DOCSTRING_PAGES = "library/reference/"
# Phrases the site renders from text it does not author, each with the file
# that owns its wording; each leaves this list when its owner lowercases it.
OWNED_ELSEWHERE = {
    # Release provenance note benchmark-info/provenance/algorithms/sce-tuning-0924-v2.md.
    "211 Distribution Instances",
    # Library docstring, dpstats/tasks/single_coefficient_estimation/_metrics.py.
    "a corpus Instance's bound value",
}


# Marks where running text breaks: around a block and in place of exempt text.
RUN_BREAK = "\x00"


def _exempt(element: Tag, page_dir: str) -> bool:
    """Whether *element*'s text keeps its own capitals."""
    if element.name in NOT_PROSE or {"data-verbatim", "data-dp-citation"} & set(
        element.attrs
    ):
        return True
    if element.name != "a":
        return False
    # Link text naming a library class or a study by its published title, and
    # an on-page contents entry, which repeats a heading checked where it stands.
    href = element.get("href", "")
    target = _resolved_route(page_dir, href)
    return (
        href.startswith("#")
        or f"/{DOCSTRING_PAGES}" in f"/{target}"
        or (target.startswith("corpus/") and target.count("/") == 2)
    )


def _capitalised_terms(markup: str, page_dir: str) -> list[str]:
    """Each glossary term *markup* capitalises mid-sentence, with its context."""
    document = BeautifulSoup(markup, "html.parser")
    for element in document.find_all(True):
        if _exempt(element, page_dir):
            element.replace_with(RUN_BREAK)
        elif element.name in BLOCKS:
            element.insert_before(RUN_BREAK)
            element.insert_after(RUN_BREAK)
    # A corpus source page is titled with its study's published title.
    titles = [
        heading.get_text().replace(RUN_BREAK, "").strip()
        for heading in document.find_all("h1")
        if page_dir.startswith("corpus/") and page_dir != "corpus/"
    ]
    hits = []
    for run in document.get_text().split(RUN_BREAK):
        text = re.sub(r"\s+", " ", run)
        for title in titles:
            text = text.replace(title, " ")
        if any(phrase in text for phrase in OWNED_ELSEWHERE):
            continue
        hits.extend(
            text[max(0, match.start() - 40) : match.end()]
            for match in CAPITALISED_TERM.finditer(text)
        )
    return hits


def test_a_glossary_term_mid_sentence_is_flagged() -> None:
    assert _capitalised_terms("<p>Each Instance carries a weight.</p>", "x/") == [
        "Each Instance"
    ]
    assert _capitalised_terms("<p>Read 3 Trials.</p>", "x/") == ["Read 3 Trials"]


def test_a_sentence_start_a_class_link_or_a_published_title_is_not_flagged() -> None:
    markup = (
        "<h2>Instances</h2><p>Instances are drawn. Sources follow.</p>"
        '<p>An <a href="../library/reference/instance/X.Instance/">Instance</a> '
        "carries <code>Instance</code>.</p>"
        '<p data-dp-citation="">Evidence from a Randomized Trial.</p>'
        '<section data-verbatim=""><p>Each Instance is shipped as written.</p></section>'
    )
    assert _capitalised_terms(markup, "x/") == []
    title = "<h1>An Evaluation by Randomized Trial</h1>"
    assert _capitalised_terms(title, "corpus/study/") == []
    assert _capitalised_terms(title, "x/") == ["An Evaluation by Randomized Trial"]


@pytest.mark.timeout(60)  # Parses every page of the build.
def test_no_page_capitalises_a_glossary_term_mid_sentence(built_site: Path) -> None:
    """Glossary terms are lowercase in running text (maintainer, 2026-10-06)."""
    offenders = {}
    for page in _pages(built_site):
        page_dir = page.parent.relative_to(built_site).as_posix() + "/"
        if page_dir.startswith(DOCSTRING_PAGES):
            continue
        hits = _capitalised_terms(page.read_text(errors="replace"), page_dir)
        if hits:
            offenders[page_dir] = hits
    assert offenders == {}


# Internal words no public page says, in its text or in the attributes a reader
# sees as text, verbatim release text included (maintainer, 2026-10-07).
INTERNAL_WORDS = re.compile(r"\bmaintainers?\b", re.IGNORECASE)
NOT_SHOWN = frozenset({"head", "script", "style", "template"})
SHOWN_ATTRIBUTES = ("title", "aria-label", "alt", "placeholder")


def _internal_words(markup: str) -> list[str]:
    """Each internal word *markup* shows a reader, in its text or text-like attributes."""
    document = BeautifulSoup(markup, "html.parser")
    for hidden in document(list(NOT_SHOWN)):
        hidden.decompose()
    shown = []
    for node in document.descendants:
        if isinstance(node, Tag):
            shown.extend(node[name] for name in SHOWN_ATTRIBUTES if node.get(name))
        elif not isinstance(node, PreformattedString):
            shown.append(node)
    text = re.sub(r"\s+", " ", " ".join(shown))
    return [
        text[max(0, match.start() - 40) : match.end() + 20]
        for match in INTERNAL_WORDS.finditer(text)
    ]


def test_an_internal_word_is_flagged_wherever_a_reader_sees_it() -> None:
    markup = (
        '<p>As the maintainer chose.</p><span title="Maintainers only">x</span>'
        '<div data-verbatim=""><p>Our maintainer cut it.</p></div>'
        "<script>// maintainer</script>"
    )
    assert len(_internal_words(markup)) == 3


@pytest.mark.timeout(60)  # Parses every page of the build.
def test_no_page_says_maintainer(built_site: Path) -> None:
    """Public text never names the maintainer (maintainer, 2026-10-07)."""
    offenders = {}
    for page in _pages(built_site):
        hits = _internal_words(page.read_text(errors="replace"))
        if hits:
            offenders[page.relative_to(built_site).as_posix()] = hits
    assert offenders == {}


# A Markdown heading marker that reached a page as text: its source was never
# rendered, as when Markdown sits inside raw HTML without markdown="1".
RAW_MARKDOWN_HEADING = re.compile(r"^#{1,6} \S.*$", re.MULTILINE)


def _unrendered_headings(markup: str) -> list[str]:
    """Each line of *markup*'s visible text that is a raw Markdown heading."""
    soup = BeautifulSoup(markup, "html.parser")
    for hidden in soup(["head", "script", "style", "template", "pre", "code"]):
        hidden.decompose()
    for hidden in soup.select(".highlight, [class*=highlight-]"):
        hidden.decompose()
    text = soup.get_text("\n")
    return [match.group(0).strip() for match in RAW_MARKDOWN_HEADING.finditer(text)]


def test_an_unrendered_markdown_heading_is_flagged() -> None:
    assert _unrendered_headings("<div>## 4. The algorithms roster</div>") == [
        "## 4. The algorithms roster"
    ]
    assert _unrendered_headings("<h2>Roster</h2><pre>## a comment</pre>") == []


@pytest.mark.timeout(60)  # Parses every page of the build.
def test_no_page_shows_raw_markdown_headings(built_site: Path) -> None:
    """Every page's Markdown is rendered, none left as visible text."""
    offenders = {}
    for page in _pages(built_site):
        hits = _unrendered_headings(page.read_text(errors="replace"))
        if hits:
            offenders[page.relative_to(built_site).as_posix()] = hits[:3]
    assert offenders == {}


# Authored Markdown whose source reached a page as text, as when a block inside
# a list item is indented too little to stay in the item. Each rule names one
# shape; rendered math, code and scripting are not read.
UNRENDERED_MARKDOWN = (
    ("display math delimiter", re.compile(r"\$\$")),
    ("math delimiter", re.compile(r"\\[\[(]")),
    (
        "list item",
        re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(?:\*\*|\x00|#{1,6}[ \t])", re.M),
    ),
    ("bold marker", re.compile(r"\*\*[^\s*](?:[^*\n]*[^\s*])?\*\*")),
    ("attribute list", re.compile(r"\{:?[ \t]*[.#][A-Za-z][\w-]*[^{}\n]*\}")),
)
# Stands in front of rendered bold text, so a list marker left as text before
# it is still read as one; never shown and never matched as a literal marker.
RENDERED_BOLD = "\x00"
NOT_AUTHORED_TEXT = ("head", "script", "style", "template", "pre", "code", "svg")


def _unrendered_markdown(markup: str) -> list[str]:
    """Each rule *markup*'s visible text breaks, with the text that breaks it."""
    soup = BeautifulSoup(markup, "html.parser")
    for hidden in soup(list(NOT_AUTHORED_TEXT)):
        hidden.decompose()
    for hidden in soup.select(".arithmatex, .highlight, [class*=highlight-]"):
        hidden.decompose()
    for bold in soup(["strong", "b"]):
        bold.insert(0, RENDERED_BOLD)
    text = soup.get_text()
    hits: list[str] = []
    for rule, pattern in UNRENDERED_MARKDOWN:
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line = text[line_start : None if line_end < 0 else line_end]
            hits.append(f"{rule}: {line.replace(RENDERED_BOLD, '').strip()[:80]}")
    return hits


@pytest.mark.parametrize(
    ("markup", "rule"),
    [
        ("<p>$$\n\\delta = 1\n$$</p>", "display math delimiter"),
        ("<p>where \\(n\\) is the sample size</p>", "math delimiter"),
        ("<p>the bound \\[ x \\]</p>", "math delimiter"),
        ("<p>A first item.\n- <strong>Mean estimation</strong>: text</p>", "list item"),
        ("<p>Intro\n* <b>Term</b> text</p>", "list item"),
        ("<div>- **Mean estimation**: text</div>", "list item"),
        ("<p>1. **Tasks**</p>", "list item"),
        ("<div>- ## 4. The roster</div>", "list item"),
        ("<p>the **headline** metric</p>", "bold marker"),
        ("<p>{ .dp-note }</p>", "attribute list"),
        ("<p>Built from the release.\n{: .dp-note }</p>", "attribute list"),
        ("<p>See below { #anchor }</p>", "attribute list"),
    ],
)
def test_unrendered_markdown_is_flagged(markup: str, rule: str) -> None:
    assert any(hit.startswith(f"{rule}: ") for hit in _unrendered_markdown(markup))


@pytest.mark.parametrize(
    "markup",
    [
        "<p>It costs $5 per run, or $40 a month.</p>",
        "<p>Between $5 and $10.</p>",
        '<div class="arithmatex">\\[\n\\delta = 1\n\\]</div>',
        '<p>where <span class="arithmatex">\\(n\\)</span> is the size</p>',
        "<pre><code>$$ x $$\n- **a**\n**b**</code></pre>",
        "<p>Pass <code>**kwargs</code> or <code>\\(</code>.</p>",
        "<script>const m = '$$'; // **x**</script>",
        '<div class="highlight"><span>- **a**</span></div>',
        "<ul><li><strong>Mean estimation</strong>: text</li></ul>",
        "<p><strong>Note</strong> - the <b>bold</b> text</p>",
        "<p>x - y, a * b, and 2 ** 3</p>",
        "<p>A dash - <strong>then bold</strong> mid-line.</p>",
        "<p>The set {1, 2, 3} and {x: x > 0}.</p>",
        "<p>Placeholders like {share} and {tau} in templates.</p>",
    ],
)
def test_rendered_markdown_and_other_text_are_not_flagged(markup: str) -> None:
    assert _unrendered_markdown(markup) == []


@pytest.mark.timeout(60)  # Parses every page of the build.
def test_no_page_shows_unrendered_markdown(built_site: Path) -> None:
    """No page shows the math delimiters, list markers or bold markers of a
    Markdown source that was not rendered."""
    offenders = {}
    for page in _pages(built_site):
        hits = _unrendered_markdown(page.read_text(errors="replace"))
        if hits:
            offenders[page.relative_to(built_site).as_posix()] = hits[:3]
    assert offenders == {}


# A count written into an option's own name, such as "standard (44)".
NAMED_COUNT = re.compile(r"\(\d[\d,]*\)\s*$")


def _controls(site: Path):
    """Every page's own choice controls, as (page, kind, names) triples: an
    exclusive group of radios, a select, or a group of pressable chips."""
    for page in _pages(site):
        if page.relative_to(site).as_posix().startswith(TEMPLATES_PREFIX):
            continue
        soup = BeautifulSoup(page.read_text(errors="replace"), "html.parser")
        where = page.relative_to(site).as_posix()
        for group in soup.select('[role="radiogroup"]'):
            names = [
                label.get_text(" ", strip=True)
                for label in group.select("input[type=radio] + label")
            ]
            yield where, "radios", names, group
        for select in soup.select("article select, .md-content select"):
            names = [
                option.get_text(" ", strip=True) for option in select.select("option")
            ]
            yield where, "select", names, select
        for group in soup.select(".dp-chip-group"):
            chips = group.select("button[aria-pressed]")
            if chips:
                yield where, "chips", [
                    c.get_text(" ", strip=True) for c in chips
                ], group


def test_every_exclusive_button_row_keeps_within_the_segmented_bound(
    built_site: Path,
) -> None:
    """W72: buttons only for at most five short options; past that, a select."""
    from profile_chart import SEGMENTED_MAX_LABEL, SEGMENTED_MAX_OPTIONS

    found = sorted(
        {
            f"{where}: {names}"
            for where, kind, names, _ in _controls(built_site)
            if kind == "radios"
            and (
                len(names) > SEGMENTED_MAX_OPTIONS
                or any(len(name) > SEGMENTED_MAX_LABEL for name in names)
            )
        }
    )
    assert found == [], "\n".join(found)


def test_every_select_wears_the_shared_treatment(built_site: Path) -> None:
    """W23: one select treatment, never a page's own."""
    found = sorted(
        {
            f"{where}: {names[:3]}"
            for where, kind, names, element in _controls(built_site)
            if kind == "select" and "dp-select" not in (element.get("class") or [])
        }
    )
    assert found == [], "\n".join(found)


def test_no_option_carries_its_count_in_its_name(built_site: Path) -> None:
    """W93: a count is supporting text beside an option, never its name."""
    found = sorted(
        {
            f"{where}: {name}"
            for where, kind, names, _ in _controls(built_site)
            if kind in ("radios", "chips")
            for name in names
            if NAMED_COUNT.search(name)
        }
    )
    assert found == [], "\n".join(found)


def test_every_chip_row_keeps_within_the_chip_bound(built_site: Path) -> None:
    """A choice of several is chips only while short; past that, a checkbox list."""
    from site_controls import CHIP_MAX_LABEL, CHIP_MAX_OPTIONS

    found = sorted(
        {
            f"{where}: {names}"
            for where, kind, names, _ in _controls(built_site)
            if kind == "chips"
            and (
                len(names) > CHIP_MAX_OPTIONS
                or any(len(name) > CHIP_MAX_LABEL for name in names)
            )
        }
    )
    assert found == [], "\n".join(found)


def test_no_fixed_value_is_shown_as_a_disabled_control(built_site: Path) -> None:
    """W24: a value the benchmark fixes reads as text, never a disabled control."""
    found = sorted(
        {
            f"{where}: {names}"
            for where, kind, names, element in _controls(built_site)
            if (kind == "select" and element.has_attr("disabled"))
            or (
                kind == "radios"
                and all(
                    radio.has_attr("disabled")
                    for radio in element.select("input[type=radio]")
                )
            )
        }
    )
    assert found == [], "\n".join(found)


def test_every_setting_box_has_at_most_one_details_link(built_site: Path) -> None:
    """A setting's choice links its definition once, from its description."""
    found = []
    for page in _pages(built_site):
        soup = BeautifulSoup(page.read_text(errors="replace"), "html.parser")
        for box in soup.select(".dp-setting"):
            links = box.select("a")
            if len(links) > 1:
                found.append(f"{page.relative_to(built_site)}: {len(links)} links")
    assert found == [], "\n".join(found)


def test_the_contact_page_lists_each_contact_address(built_site: Path) -> None:
    """Contact lists the addresses its slot gives, one per line."""
    import yaml

    page = built_site / "contact" / "index.html"
    assert page.exists()
    content = Path(__file__).resolve().parents[3] / "docs" / "content" / "site-copy.yml"
    addresses = yaml.safe_load(content.read_text("utf-8"))["contact"][
        "addresses"
    ].split()
    shown = _visible(page)
    assert addresses and all(address in shown for address in addresses)


def test_every_licence_link_reaches_its_text(built_site: Path) -> None:
    """A data licence links its SPDX page or a licence text this build publishes."""
    found, local = [], 0
    for page in sorted(built_site.glob("corpus/*/index.html")):
        soup = BeautifulSoup(page.read_text(errors="replace"), "html.parser")
        for link in soup.select('a[href*="licenses/"]'):
            href = link["href"]
            if href.startswith("https://spdx.org/licenses/"):
                continue
            local += 1
            target = urlsplit(
                urljoin(f"/{page.relative_to(built_site).as_posix()}", href)
            ).path
            if not (built_site / target.lstrip("/")).is_file():
                found.append(f"{page.relative_to(built_site)}: {href}")
    assert found == [], "\n".join(found)
