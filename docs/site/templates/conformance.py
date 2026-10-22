"""Check a built site for what its renderers cannot already guarantee.

Every page is an instance of exactly one template, and a page a renderer fills
from that template carries the template's sections, in its order, with its
table columns and its Access block: the renderer builds the page out of them,
so checking them again would report a defect that cannot occur. What a renderer
cannot know is checked here: that the sitemap is the route set the registry and
the spec predict, that a rankings table keeps its columns through the client's
own re-render, that one class has one home, and that the text the library
supplies is complete, cites in the one ruled shape, shows no retired vocabulary, no unrendered math and no placeholder, and
discloses a preview once. :func:`refuse_untemplated` applies the route rule at
the other end, where the assembler writes a page.

It reads HTML and the template spec only, so the strict build can run it with
the documentation extra alone. Every check returns its findings as lines; one
finding never stops the others, and nothing this module cannot read is passed
over in silence.
"""

from __future__ import annotations

import html as html_module
import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .spec import TEMPLATES, TemplateSpec

# The reader-visible part of a built page: Material renders one article per
# page and every other element on it is navigation, header, or footer chrome.
ARTICLE = re.compile(r"<article[^>]*>(.*?)</article>", re.S)

# Markup whose text is not reader prose: scripting, styling, verbatim blocks,
# and typeset math. Math is masked because `javascripts/mathjax.js` typesets
# the `arithmatex` class, so delimiters inside it are the rendered form and
# delimiters outside it never typeset.
MASKED = re.compile(
    r"<pre\b.*?</pre>" r"|<script\b.*?</script>" r"|<style\b.*?</style>"
    # A notebook page ships each code cell a second time, hidden, for the
    # copy button. It is the same source the cell already shows in its `pre`.
    r"|<div[^>]*\bclass=\"clipboard-copy-txt\"[^>]*>.*?</div>"
    r"|<(span|div)[^>]*\bclass=\"[^\"]*arithmatex[^\"]*\"[^>]*>.*?</\1>",
    re.S,
)

TAG = re.compile(r"<[^>]+>")
TABLE = re.compile(r"<table([^>]*)>(.*?)</table>", re.S)
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
CELL = re.compile(r"<t([hd])[^>]*>(.*?)</t\1>", re.S)
HEAD = re.compile(r"<thead[^>]*>(.*?)</thead>", re.S)
BODY = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
H2 = re.compile(r"<h2([^>]*)>(.*?)</h2>", re.S)
ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.S)
HREF = re.compile(r'href="([^"]*)"')
IDENTIFIER = re.compile(r'id="([^"]*)"')
DATA_COLUMNS = re.compile(r'data-columns="([^"]*)"')

# The anchor the theme appends to every heading, which is chrome, not text.
PERMALINK = re.compile(r'<a[^>]*class="headerlink"[^>]*>.*?</a>', re.S)

# Vocabulary the site never shows a reader. `CONTEXT.md` owns why each is
# retired; a page says Algorithm, configured algorithm, baseline, or
# privacy interpretation.
BANNED_WORDS = (
    "catalogue",
    "catalog",
    "index of the literature",
    "yardstick",
    "entrant",
    "Library task",
    "Auxiliary core",
    "Performance profile",
    "privacy class",
    "allowance",
    "platform choice",
    "platform default",
    "house default",
    "house choice",
    "house choices",
)

# Retired names for one benchmark trial, as phrases: "run" and "seed" are also
# the verb and the sweep's seed grid, so a bare word would fire on correct prose.
RETIRED_UNIT_PHRASES = (
    "seeded run",
    "seeded runs",
    "per-seed",
    "seed group",
    "seed groups",
    "comparison cell",
    "comparison cells",
    "scientific cell",
    "scientific cells",
)

# Pages whose prose the maintainer owns and rewrites later: the tutorial
# notebooks and each task's methodology page.
UNIT_PHRASE_EXEMPT = re.compile(r"^(library/tutorials/|[^/]+/methodology/$)")

# Math delimiters and macros that mean the page shows a formula's source rather
# than the formula. Checked only outside the typeset class.
RAW_MATH = re.compile(r"\$|\\\(|\\\)|\\\[|\\\]|\\mathbb|\^\{")

# Invented objects from the design preview, and the placeholder shapes an
# unfinished generator leaves behind. "[Computer software]" is the medium label
# the bibliography gives a cited software release, not a placeholder.
FILLER = re.compile(
    r"\bdp_filler\b|\bFiller\b|\bTask Name\b|\blorem\b"
    r"|\[(?:TODO|TBD|PLACEHOLDER|(?!Computer software\])[A-Z][a-z]+(?: [A-Za-z]+)+)\]"
)

# A citation as a table cites it: the bibliography's in-text form.
_SURNAME = r"[A-Z][^\s(,]*(?: [A-Z][^\s(,]*)*"
TABLE_CITATION = re.compile(
    rf"^{_SURNAME}(?: et al\.|(?:, {_SURNAME})* and {_SURNAME})? \(\d{{4}}\)$"
)

# A reference line carries its year in parentheses; without one it is a
# fragment rather than a citation.
REFERENCE_YEAR = re.compile(r"\(\d{4}\)")

# A cell whose prose ends here was cut off mid-sentence.
DANGLING = (
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "from",
    "by",
    "as",
    "at",
    "and",
    "or",
    "its",
    "their",
)
TERMINAL = (".", "?", "!", ":", ";", "%", ")", "”", '"')

# Columns whose cells are authored or docstring prose, so a reader reads them
# as sentences and a cut one is visible.
PROSE_COLUMNS = ("Estimand", "Description", "Summary")

TASK_LINK = re.compile(r'<a\b[^>]*data-task-id="([^"]+)"[^>]*>(.*?)</a>', re.S)

# Where the classic reference renders a class by its import path, which is the
# one home a task-keyed class may not also have.
REFERENCE_AREA = "library/reference/"

# Where a build that asks for the design preview writes one page per template.
# That area is the preview's own subject, so its routes are not measured
# against the routes the templates produce on the site.
PREVIEW_AREA = "templates/"

# The fixed anchor a generated API block sits under, so a reader and this check
# find it in the same place on every object page.
API_ANCHOR = "api"

# The rankings table every results surface renders, and the attribute naming
# the columns its client-side re-render builds each row from.
RANKINGS = 'class="dp-leaderboard'

PREVIEW_SUFFIX = "(Preview)"

# The violations this build already carries, which a burn-down ticket shrinks.
BASELINE_NAME = "conformance-baseline.json"
BASELINE_PATH = Path(__file__).resolve().parent / BASELINE_NAME

# The pages a task space publishes that no template route produces: the
# authored guide and the walkthrough, which the guide split has not yet
# removed. The writer lets them through so the build finishes and this module
# can report them; they are violations either way, never an amnesty.
UNTEMPLATED_TASK_PAGES = ("guide/", "walkthrough/")
NAV_FILE = "SUMMARY.md"
NAV_SUFFIX = "SUMMARY/"


@dataclass(frozen=True)
class Page:
    """One built content page: where it lives and what it published."""

    route: str
    path: Path
    html: str
    article: str


@dataclass(frozen=True)
class Table:
    """One rendered table: its attributes, header, cell text and cell markup."""

    attributes: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    markup: tuple[tuple[str, ...], ...]


def load_baseline(path: Path) -> dict[str, list[str]]:
    """The violations this build is allowed to still carry, by check."""
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_findings(
    observed: Sequence[str], baseline: dict[str, list[str]]
) -> list[str]:
    """What this build must answer for, given the violations already recorded.

    The baseline is the exact set a burn-down shrinks, so a violation it does
    not list is new work and a violation it lists that does not occur is a
    line to delete. Reporting both is what keeps the file from rotting into a
    blanket exemption.

    Args:
        observed: Every violation :func:`check_site` reported, across every
            configuration this gate builds.
        baseline: The recorded violations, keyed by the check that reports them.

    Returns:
        One line per violation outside the baseline and per baseline entry that
        does not occur. An empty list means the build is at its baseline.
    """
    recorded = {line for lines in baseline.values() for line in lines}
    seen = set(observed)
    findings = [f"baseline: not recorded: {line}" for line in sorted(seen - recorded)]
    findings += [
        f"baseline: does not occur, delete it: {line}"
        for line in sorted(recorded - seen)
    ]
    return findings


def group_by_check(violations: Sequence[str]) -> dict[str, list[str]]:
    """Violations keyed by the check that reports them, ready to record."""
    grouped: dict[str, list[str]] = {}
    for line in violations:
        grouped.setdefault(line.split(":", 1)[0], []).append(line)
    return {check: sorted(set(lines)) for check, lines in sorted(grouped.items())}


def check_site(
    site: Path, registry: Sequence[str] = (), *, preview: bool = False
) -> list[str]:
    """Return every way this build departs from the page templates.

    Args:
        site: The finished build's directory, as ``mkdocs build`` wrote it.
        registry: The built-in task ids the library declares, where the caller
            has them. Supplying them cross-checks global task navigation the routes
            are derived from, so a task dropped from that navigation cannot take its
            whole space out of this check with it.
        preview: Whether this build is the design preview of the templates,
            whose pages render invented objects. Placeholder text is that
            build's content rather than a leak, so the filler check is the one
            check it is not held to.

    Returns:
        One line per violation, ordered by check and then by route. An empty
        list means the build conforms.
    """
    pages, unreadable = read_pages(site)
    tasks = task_ids(pages)
    templates = {page.route: template_of(page.route, tasks) for page in pages}
    parsed = {page.route: tables(page.article) for page in pages}
    findings = list(unreadable)
    findings += check_routes(pages, templates, tasks, registry, preview=preview)
    findings += check_sitemap(pages, preview=preview)
    findings += check_citations(pages, templates, parsed)
    findings += check_banned_words(pages)
    findings += check_rendered_math(pages)
    findings += check_complete_cells(pages, parsed)
    if not preview:
        findings += check_filler(pages)
    findings += check_preview_suffix(pages)
    findings += check_roster_parity(pages, parsed)
    findings += check_one_home(pages, templates)
    findings += check_rankings_columns(pages, parsed)
    findings += check_results_controls(pages, templates)
    return findings


# ── Reading the build ────────────────────────────────────────────────────────


def read_pages(site: Path) -> tuple[list[Page], list[str]]:
    """Every content page in *site*, and a line for every page unread.

    Anything this module cannot read is reported rather than skipped, because a
    page silently passed over is a page unchecked.
    """
    pages: list[Page] = []
    unreadable: list[str] = []
    for path in sorted(site.rglob("index.html")):
        text = path.read_text(errors="replace")
        route = path.relative_to(site).parent.as_posix()
        route = "" if route == "." else f"{route}/"
        article = ARTICLE.search(text)
        if article is None:
            unreadable.append(
                f"route: {route} publishes no article element, so nothing on it "
                "could be checked."
            )
            continue
        pages.append(Page(route, path, text, article[1]))
    return pages, unreadable


def task_ids(pages: list[Page]) -> tuple[str, ...]:
    """The task slugs the global navigation on the first content page declares."""
    return tuple(slug for slug, _ in TASK_LINK.findall(pages[0].html)) if pages else ()


def results_routes(pages: list[Page]) -> set[str]:
    """Results surfaces linked by each task's overview page."""
    tasks = task_ids(pages)
    return {
        posixpath.normpath(posixpath.join(page.route, href)) + "/"
        for page in pages
        if page.route.rstrip("/") in tasks
        for attributes, label in ANCHOR.findall(page.article)
        if text(label) in ("Leaderboard", "Explore")
        for href in HREF.findall(attributes)
    }


def tables(article: str) -> list[Table]:
    """Every table in one page's article, header and body separated."""
    found: list[Table] = []
    for attributes, body in TABLE.findall(article):
        head = HEAD.search(body)
        columns = tuple(text(cell) for _, cell in CELL.findall(head[1])) if head else ()
        rows_html = BODY.search(body)
        raw = ROW.findall(rows_html[1]) if rows_html else []
        rows = tuple(tuple(text(cell) for _, cell in CELL.findall(row)) for row in raw)
        markup = tuple(tuple(cell for _, cell in CELL.findall(row)) for row in raw)
        found.append(Table(attributes, columns, rows, markup))
    return found


def text(fragment: str) -> str:
    """One fragment's reader-visible text, without markup or typeset math."""
    stripped = MASKED.sub(" ", PERMALINK.sub("", fragment))
    return " ".join(html_module.unescape(TAG.sub(" ", stripped)).split())


def headings(article: str) -> list[tuple[str, str]]:
    """Each ``h2``'s anchor and text, in document order."""
    found: list[tuple[str, str]] = []
    for attributes, body in H2.findall(article):
        anchor = IDENTIFIER.search(attributes)
        found.append((anchor[1] if anchor else "", text(body)))
    return found


def route_pattern(route: str, tasks: Sequence[str]) -> re.Pattern[str]:
    """One declared route, with its task placeholder resolved."""
    parts: list[str] = []
    for piece in re.split(r"(\{[a-z_]+\})", route):
        if piece == "{task}":
            parts.append(f"(?:{'|'.join(re.escape(task) for task in tasks) or '(?!)'})")
        elif piece.startswith("{"):
            parts.append(r"[^/]+")
        else:
            parts.append(re.escape(piece))
    return re.compile(f"^{''.join(parts)}$")


def literal_routes(template: TemplateSpec) -> list[str]:
    """The routes this template names outright, placeholders aside."""
    return [route for route in template.routes if "{" not in route]


def produces(template: TemplateSpec, route: str, tasks: Sequence[str]) -> bool:
    """Whether *template* declares a route that produces *route*."""
    return any(
        route_pattern(declared, tasks).match(route) for declared in template.routes
    )


def template_of(route: str, tasks: Sequence[str]) -> TemplateSpec | None:
    """The template whose route produces *route*, or ``None`` for no template.

    A literal route wins over a placeholder route, so an area landing is never
    read as a task overview and a tutorial is never read as a reference page.
    """
    if route.endswith(NAV_SUFFIX):
        return None
    matched = [
        template for template in TEMPLATES.values() if produces(template, route, tasks)
    ]
    literal = [template for template in matched if route in literal_routes(template)]
    if literal:
        return literal[0]
    return matched[0] if matched else None


def page_route(path: str) -> str:
    """The published route of a page written at *path*."""
    trimmed = path.removesuffix(".md").removesuffix("/index")
    return "" if trimmed == "index" else f"{trimmed}/"


def untemplated_reason(route: str, tasks: Sequence[str]) -> str:
    """Why *route* is not a page any template produces, or an empty string.

    One rule, read at both ends: the assembler refuses a route this names, and
    the route check reports one this names on a finished build. Both know the
    pages the guide split has not yet removed, and neither treats them as
    permitted.
    """
    # This definition subpage uses the native mkdocstrings API renderer.
    if route == "m_estimation/definition/MEstimationLoss/":
        return ""
    slug = route.split("/")[0]
    if slug not in tasks:
        return ""
    if route.endswith(NAV_SUFFIX):
        return "it is a navigation file published as a page"
    for template in TEMPLATES.values():
        if produces(template, route, tasks):
            return ""
    if route[len(slug) + 1 :] in UNTEMPLATED_TASK_PAGES:
        return "no template produces it and the guide split has not removed it"
    return "no template route produces it"


def refuse_untemplated(path: str, tasks: Sequence[str]) -> None:
    """Refuse to write a task-space page that no template route produces.

    Every page inside a task space is an instance of exactly one template, so a
    new page written at a route no template declares would be a page no ruling
    describes. The pages the guide split has not yet removed are written, and
    the route check reports them; nothing may join them.

    Args:
        path: The page's path in the build, as the assembler writes it.
        tasks: The task slugs this build documents.

    Raises:
        RuntimeError: The path publishes a task-space route no template
            produces.
    """
    route = page_route(path)
    if path.endswith(NAV_FILE):
        return
    slug = route.split("/")[0]
    if route[len(slug) + 1 :] in UNTEMPLATED_TASK_PAGES:
        return
    if untemplated_reason(route, tasks):
        raise RuntimeError(
            f"{path} would publish {route!r} inside a task space, which no page "
            "template produces."
        )


# ── The checks ───────────────────────────────────────────────────────────────


def check_routes(
    pages: list[Page],
    templates: dict[str, TemplateSpec | None],
    tasks: Sequence[str],
    registry: Sequence[str] = (),
    *,
    preview: bool = False,
) -> list[str]:
    """The built routes are the ones the registry and the template spec give."""
    findings: list[str] = []
    published = {page.route for page in pages}
    if pages and not tasks:
        findings.append(
            "route: global task navigation links no task, so no route could be derived and no "
            "page was checked against a template."
        )
    surfaces = results_routes(pages)
    for page in pages:
        if page.route in surfaces:
            continue
        if preview and page.route.startswith(PREVIEW_AREA):
            continue
        reason = untemplated_reason(page.route, tasks)
        if reason:
            findings.append(f"route: {page.route} {reason}.")
        elif templates[page.route] is None:
            # Home, tutorials, guides, About, Submit and the reference
            # pages are template pages whose sections the spec does not declare
            # yet, so their route matches nothing here.
            findings.append(
                f"route: {page.route} sits outside every task space and the "
                "spec declares no template for it."
            )
    for task in tasks:
        if f"{task}/" not in published:
            findings.append(f"route: {task}/ is documented but this build has no page.")
    findings += _registry_findings(pages, tasks, registry)
    return findings


def declared_routes() -> dict[str, str]:
    """Every route the spec names outright, and the template that names it.

    A route with a placeholder in it is produced from a source this module does
    not read — the task registry, the release, the installed package — so it is
    checked from the built page back to the template. These are the routes the
    spec fixes, so they are checked in both directions.
    """
    return {
        route: template.name
        for template in TEMPLATES.values()
        for route in sorted({*literal_routes(template), *template.authored})
    }


def check_sitemap(pages: list[Page], *, preview: bool = False) -> list[str]:
    """The built sitemap is the route set the spec predicts, page for page.

    :func:`check_routes` answers the one direction: every page a build
    publishes is a page some template's route produces. This answers the other:
    every route a template names outright, and every page it declares an author
    writes, is a page this build published. A template that describes a page
    nobody builds is as much a departure as a page no template
    describes.
    """
    published = {page.route for page in pages}
    if not published:
        return []
    findings: list[str] = []
    for route, template in sorted(declared_routes().items()):
        if route in published:
            continue
        if preview and route.startswith(PREVIEW_AREA):
            continue
        findings.append(
            f"sitemap: {template} declares {route or '/'}, which this build "
            "does not publish."
        )
    return findings


def _registry_findings(
    pages: list[Page], tasks: Sequence[str], registry: Sequence[str]
) -> list[str]:
    """The global task navigation names the registry's tasks, and only those.

    Routes are read off navigation, so a menu that has lost a task would take
    that task's whole space out of every other check. The registry is the
    independent list that catches it.
    """
    if not registry:
        return []
    findings: list[str] = []
    for slug in tasks:
        if slug not in registry:
            findings.append(
                f"route: global task navigation links {slug}/, which is not a built-in task."
            )
    published = {page.route.split("/")[0] for page in pages}
    for slug in registry:
        if slug in tasks or slug not in published:
            continue
        findings.append(
            f"route: {slug}/ publishes a task space that global task navigation does "
            "not link, so its pages match no template."
        )
    return findings


def check_citations(
    pages: list[Page],
    templates: dict[str, TemplateSpec | None],
    parsed: dict[str, list[Table]],
) -> list[str]:
    """A table cites the in-text form; a reference line carries its year.

    Only the Algorithms and Transformations tables carry that citation-shaped
    "Reference" column; a Metric's "Reference" column names an input
    reference (`render.py`'s "Metric arguments" table), not a paper, so a
    table is checked only when it also carries an "Algorithm" or "Transformation"
    column.
    """
    findings: list[str] = []
    for page in pages:
        for table in parsed[page.route]:
            if "Reference" not in table.columns or not (
                "Algorithm" in table.columns or "Transformation" in table.columns
            ):
                continue
            column = table.columns.index("Reference")
            named = table.columns.index(
                "Algorithm" if "Algorithm" in table.columns else "Transformation"
            )
            for cells in table.rows:
                if column >= len(cells):
                    continue
                cited = cells[column].strip()
                if cited in ("", "—") or TABLE_CITATION.match(cited):
                    continue
                findings.append(
                    f"citations: {page.route} {cells[named].strip()!r} cites {cited!r}, "
                    "which is not the bibliography's in-text form, such as "
                    "'Surname (Year)' or 'Surname et al. (Year)'."
                )
        template = templates[page.route]
        if template is None or not any(
            section.key == "reference" for section in template.sections
        ):
            continue
        for heading, body in _sections(page.article):
            if heading != "References":
                continue
            shown = text(body)
            if shown and not REFERENCE_YEAR.search(shown):
                findings.append(
                    f"citations: {page.route} carries a References section with no "
                    "year, so it shows a citation fragment."
                )
    return findings


def _sections(article: str) -> list[tuple[str, str]]:
    """Each ``h2`` heading and the markup that follows it, in order."""
    found: list[tuple[str, str]] = []
    matches = list(H2.finditer(article))
    for position, match in enumerate(matches):
        following = matches[position + 1].start() if position + 1 < len(matches) else -1
        end = len(article) if following < 0 else following
        found.append((text(match[2]), article[match.end() : end]))
    return found


def check_banned_words(pages: list[Page]) -> list[str]:
    """Retired vocabulary never reaches a reader."""
    findings: list[str] = []
    for page in pages:
        shown = text(page.article)
        for word in BANNED_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", shown, re.I):
                findings.append(f"vocabulary: {page.route} shows {word!r}.")
        if UNIT_PHRASE_EXEMPT.match(page.route):
            continue
        for phrase in RETIRED_UNIT_PHRASES:
            if re.search(rf"\b{re.escape(phrase)}\b", shown, re.I):
                findings.append(f"vocabulary: {page.route} shows {phrase!r}.")
    return findings


def check_rendered_math(pages: list[Page]) -> list[str]:
    """Reject raw math outside the configured MathJax processing scopes."""
    findings: list[str] = []
    for page in pages:
        article = page.article
        if "jp-MarkdownOutput" in article:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(article, "html.parser")
            # Notebook Markdown is typeset in the browser by mathjax.js.
            # Remove balanced delimiters only; dangling ones remain findings.
            math = re.compile(
                r"(?<!\\)\$\$(?:(?!\$\$).)+?(?<!\\)\$\$"
                r"|(?<![\\$])\$(?!\$)[^\n$]+?(?<!\\)\$(?!\$)",
                re.S,
            )
            for cell in soup.select(".jp-MarkdownOutput"):
                for node in list(cell.find_all(string=True)):
                    if not node.find_parent(["pre", "code", "script"]):
                        node.replace_with(math.sub("", str(node)))
            article = str(soup)
        found = sorted(set(RAW_MATH.findall(text(article))))
        if found:
            findings.append(f"math: {page.route} shows unrendered math markup {found}.")
    return findings


def check_complete_cells(
    pages: list[Page], parsed: dict[str, list[Table]]
) -> list[str]:
    """A prose cell ends its sentence rather than being cut off."""
    findings: list[str] = []
    for page in pages:
        for table in parsed[page.route]:
            for column, name in enumerate(table.columns):
                if name not in PROSE_COLUMNS:
                    continue
                for row, cells in enumerate(table.rows):
                    if column >= len(cells):
                        continue
                    reason = _truncation(cells[column])
                    if reason:
                        findings.append(
                            f"cells: {page.route} {name} row {row + 1} {reason}: "
                            f"{cells[column]!r}"
                        )
    return findings


def _truncation(cell: str) -> str:
    """Why this cell reads as cut off, or an empty string when it reads whole."""
    shown = cell.strip()
    if not shown or shown == "—" or " " not in shown:
        return ""
    if shown.endswith("…") or shown.endswith("..."):
        return "ends in an ellipsis"
    if shown.split()[-1].lower().strip(",") in DANGLING:
        return "ends on a dangling word"
    if not shown.endswith(TERMINAL):
        return "ends without terminal punctuation"
    return ""


def check_filler(pages: list[Page]) -> list[str]:
    """No invented object or placeholder reaches a published page."""
    findings: list[str] = []
    for page in pages:
        found = sorted(set(FILLER.findall(text(page.article))))
        if found:
            findings.append(f"filler: {page.route} shows placeholder text {found}.")
    return findings


def check_preview_suffix(pages: list[Page]) -> list[str]:
    """Preview state is one banner per build, never a suffix on every link."""
    findings: list[str] = []
    for page in pages:
        for _, label in ANCHOR.findall(page.article):
            if PREVIEW_SUFFIX in text(label):
                findings.append(
                    f"preview: {page.route} suffixes link text "
                    f"{text(label)!r} with '(Preview)'."
                )
    return findings


# One Roster landing row's label and headline rank, as its filters read them.
ROSTER_ROW = re.compile(r"<tr\b([^>]*\bdata-dp-label=\"[^\"]*\"[^>]*)>", re.S)
_ROW_LABEL = re.compile(r'data-dp-label="([^"]*)"')
_ROW_RANK = re.compile(r'data-dp-rank="([^"]*)"')


def check_roster_parity(pages: list[Page], parsed: dict[str, list[Table]]) -> list[str]:
    """A configured algorithm a Roster marks ranked belongs to that pool.

    A row with no headline rank is a legitimate Roster entry no eligible pool
    names, so only a row the Roster landing claims to rank is checked against
    its task leaderboard's own eligible pool.
    """
    by_route = {page.route: page for page in pages}
    findings: list[str] = []
    if not pages:
        return findings
    for slug, _title in TASK_LINK.findall(pages[0].html):
        overview = by_route.get(f"{slug}/")
        roster = by_route.get(f"{slug}/roster/")
        if overview is None or roster is None:
            continue
        leaderboard = _leaderboard_route(overview, overview.article)
        if leaderboard is None or leaderboard not in by_route:
            continue
        ranked = _ranked_rows(parsed[leaderboard])
        for attributes in ROSTER_ROW.findall(roster.article):
            rank = _ROW_RANK.search(attributes)
            if rank is None or not rank[1]:
                continue
            label = " ".join(
                html_module.unescape(_ROW_LABEL.search(attributes)[1]).split()
            )
            if label in ranked:
                continue
            findings.append(
                f"parity: {roster.route} ranks {label!r}, which {leaderboard} "
                "does not rank."
            )
    return findings


def _leaderboard_route(index: Page, results_cell: str) -> str | None:
    """The task leaderboard linked from its overview."""
    for attributes, label in ANCHOR.findall(results_cell):
        if "Leaderboard" not in text(label):
            continue
        for href in HREF.findall(attributes):
            return posixpath.normpath(posixpath.join(index.route, href)) + "/"
    return None


def _ranked_rows(found: list[Table]) -> list[str]:
    """The algorithm labels the first rankings table on one page ranks."""
    for table in found:
        if RANKINGS not in table.attributes or "Algorithm" not in table.columns:
            continue
        column = table.columns.index("Algorithm")
        return [cells[column] for cells in table.rows if column < len(cells)]
    return []


# The templates whose pages draw a results chart, and so carry the series rule
# and name what that chart is drawing.
_CHART_TEMPLATES = ("T1", "T4", "T5")

_SERIES_RULE = re.compile(r'<script type="application/json" data-series-style>')
_COMBOBOX = re.compile(r'role="combobox"[^>]*aria-autocomplete="list"')
_PLOTTED_HEADING = re.compile(r'class="dp-plotted-heading"')
# The baseline control's own "Algorithm…" search box; it names the algorithm
# read against, not one a comparison chart draws, so it is never one of the
# pickers the comparison choices are counted against.
_BASELINE_COMBOBOX = "baseline-picker-options"
# Where a comparison chart's reader chooses its algorithms: a plotted list, or
# one slot of a fixed set.
_COMPARISON_CHOICE = re.compile(r"data-comparison-selection[\s>]|data-outcome-slot=")


def check_results_controls(
    pages: list[Page], templates: Mapping[str, TemplateSpec | None]
) -> list[str]:
    """Check the controls a results page is ruled to carry.

    A curve's colour is dealt from a categorical palette that ships with the
    page rather than being invented at draw time: a results page without it
    would fall back to Plotly's own colours. A
    page that lets a reader pick the compared algorithms carries the
    list-autocomplete combobox and the plotted set it commits into, because a
    combobox has no multiple selection of its own.
    """
    findings: list[str] = []
    for page in sorted(pages, key=lambda entry: entry.route):
        template = templates.get(page.route)
        if template is None or template.name not in _CHART_TEMPLATES:
            continue
        # A page of this template that draws no chart has no curve to colour.
        if 'class="dp-chart"' not in page.html:
            continue
        rules = len(_SERIES_RULE.findall(page.html))
        if rules != 1:
            findings.append(
                f"results controls: {page.route} ships {rules} series-colour "
                "rules, and a chart page ships exactly one, so every curve on "
                "it takes its colour from the same palette."
            )
        # Every chart names what it is drawing, and a chart whose drawn set the
        # reader chooses also carries the picker they choose it with.
        charts = page.html.count('class="dp-chart"') + page.html.count(
            'class="dp-comparison-plot"'
        )
        plotted = len(_PLOTTED_HEADING.findall(page.html))
        if plotted != charts:
            findings.append(
                f"results controls: {page.route} draws {charts} charts but names "
                f"the plotted set of {plotted}, and a reader cannot tell which "
                "curve is which on a chart that names none."
            )
        if template.name != "T5" or 'class="dp-comparison-plot"' not in page.html:
            continue
        pickers = 0
        for match in _COMBOBOX.finditer(page.html):
            tag_end = page.html.find(">", match.end())
            tag = page.html[match.start() : tag_end if tag_end >= 0 else match.end()]
            if _BASELINE_COMBOBOX not in tag:
                pickers += 1
        # Each comparison chart owns its selection: the Error ratio list
        # and each of the Outcomes slots is chosen with a picker of its own.
        choices = len(_COMPARISON_CHOICE.findall(page.html))
        if pickers != choices:
            findings.append(
                f"results controls: {page.route} pairs {pickers} algorithm "
                f"pickers with {choices} comparison choices, and each "
                "comparison choice is made with a picker of its own."
            )
        for match in _COMBOBOX.finditer(page.html):
            around = page.html[max(0, match.start() - 200) : match.end() + 200]
            if "aria-controls" not in around:
                findings.append(
                    f"results controls: {page.route} has a combobox that names "
                    "no popup, so nothing announces its suggestions."
                )
                break
    return findings


def check_one_home(
    pages: list[Page], templates: dict[str, TemplateSpec | None]
) -> list[str]:
    """A class is rendered in one place, and its API block sits at one anchor.

    An object reachable only through a domain key belongs in its task space
    with its API inline; an object reachable by import path alone belongs in
    the classic reference. A class rendered in both has two homes, and a reader
    following either arrives at a different half of the same object.
    """
    homes: dict[str, str] = {}
    findings: list[str] = []
    for page in pages:
        template = templates[page.route]
        if template is None or template.name != "T9":
            continue
        name = page.route.rstrip("/").split("/")[-1]
        homes.setdefault(name, page.route)
        anchors = {anchor for anchor, _ in headings(page.article)}
        if any(section.key == "api" for section in template.sections):
            if API_ANCHOR not in anchors:
                findings.append(
                    f"one home: {page.route} renders no API block under the "
                    f"'{API_ANCHOR}' anchor."
                )
    for page in pages:
        if not page.route.startswith(REFERENCE_AREA):
            continue
        name = page.route.rstrip("/").split("/")[-1].split(".")[-1]
        if name in homes:
            findings.append(
                f"one home: {page.route} renders {name}, which {homes[name]} "
                "already renders."
            )
    return findings


def check_rankings_columns(
    pages: list[Page], parsed: dict[str, list[Table]]
) -> list[str]:
    """A rankings table declares the columns its header and every row carries.

    The page's client rebuilds each row from ``data-columns`` alone, so a
    declaration that disagrees with the header produces a row that disagrees
    with the header the moment a control moves. Checking the three together on
    the static build is what makes the re-rendered table right as well.
    """
    findings: list[str] = []
    for page in pages:
        for table in parsed[page.route]:
            if RANKINGS not in table.attributes:
                continue
            declared = DATA_COLUMNS.search(table.attributes)
            if declared is None:
                findings.append(
                    f"rankings: {page.route} renders a rankings table that declares "
                    "no data-columns, so a re-render cannot keep its shape."
                )
                continue
            width = len(declared[1].split(","))
            if width != len(table.columns):
                findings.append(
                    f"rankings: {page.route} declares {width} data-columns and "
                    f"heads {len(table.columns)}."
                )
            for row, cells in enumerate(table.rows):
                if len(cells) != width:
                    findings.append(
                        f"rankings: {page.route} row {row + 1} carries "
                        f"{len(cells)} cells against {width} data-columns."
                    )
    return findings
