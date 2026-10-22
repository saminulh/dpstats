"""Template conformance over a built site, one check at a time.

Each check gets a build that satisfies it and a build that breaks it in exactly
one way, so a finding names the defect rather than the fixture. The conforming
build below is a whole small site: global task navigation, one task space, its
leaderboard and one Algorithm page.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


from templates.conformance import (
    BASELINE_PATH,
    TABLE_CITATION,
    Page,
    Table,
    baseline_findings,
    check_citations,
    check_results_controls,
    check_site,
    check_sitemap,
    declared_routes,
    group_by_check,
    load_baseline,
    read_pages,
    refuse_untemplated,
)
from templates.spec import TEMPLATES

TASK = "demo_task"
LEADERBOARD = f"{TASK}/leaderboard/set"
ALGORITHMS = f"{TASK}/algorithms"
ALGORITHM = f"{TASK}/algorithms/AlgOne"
ROSTER = f"{TASK}/roster"
TRANSFORMATION = f"{TASK}/transformations/ProcOne"
METRICS = f"{TASK}/metrics"
REFERENCE = "library/reference/analysis/dpstats.analysis.Summary"

TASK_OVERVIEW = f"""
<p>The mean of a bounded sample.</p>
<pre><code>from dpstats.tasks import BUILT_IN_TASKS
BUILT_IN_TASKS[&quot;{TASK}&quot;]</code></pre>
<p><a href="definition/">Definition</a><a href="leaderboard/set/">Leaderboard</a></p>
<h2 id="problem">Problem formulation</h2>
<p>Release the mean of a sample drawn from a bounded domain.</p>
"""

DEFINITION = """
<h2 id="input">Input</h2>
<table class="dp-table">
<thead><tr><th>Field</th><th>Type</th><th>Constraints</th><th>Description</th></tr>
</thead>
<tbody><tr><td><code>X</code></td><td><code>float array</code></td>
<td><span class="arithmatex">\\( n \\ge 1 \\)</span></td>
<td>The sample, one row per unit.</td></tr></tbody>
</table>
<h2 id="metadata">Metadata</h2>
<table class="dp-table">
<thead><tr><th>Field</th><th>Type</th><th>Constraints</th><th>Description</th></tr>
</thead>
<tbody><tr><td><code>n_samples</code></td><td><code>int</code></td>
<td><span class="arithmatex">\\( n \\ge 1 \\)</span></td>
<td>The number of units in the sample.</td></tr></tbody>
</table>
<h2 id="output">Output</h2>
<table class="dp-table">
<thead><tr><th>Field</th><th>Type</th><th>Constraints</th><th>Description</th></tr>
</thead>
<tbody><tr><td><code>mean_est</code></td><td><code>float array</code></td>
<td><span class="arithmatex">\\( d \\ge 1 \\)</span></td>
<td>The released mean.</td></tr></tbody>
</table>
"""

RANKINGS = """
<table class="dp-leaderboard" data-columns="rank,algorithm,share">
<thead><tr><th>Rank</th><th>Algorithm</th><th>Success rate</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Alg One</td><td>96.7%</td></tr>
<tr><td>2</td><td>Alg Two</td><td>90.1%</td></tr>
</tbody>
</table>
"""

ALGORITHMS_PAGE = """
<table class="dp-table">
<thead><tr><th>Algorithm</th><th>Privacy interpretation</th><th>Reference</th></tr></thead>
<tbody><tr><td>AlgOne</td><td>standard</td><td>Dwork and Roth (2014)</td></tr>
</tbody></table>
"""

ROSTER_PAGE = """
<table class="dp-table">
<thead><tr><th>Configured algorithm</th><th>Privacy interpretation</th><th>Rank</th></tr>
</thead>
<tbody>
<tr data-dp-label="Alg One" data-dp-rank="1"><td>Alg One</td><td>standard</td><td>Rank 1</td></tr>
<tr data-dp-label="Alg Ten" data-dp-rank=""><td>Alg Ten</td><td>diagnostic</td><td>not ranked</td></tr>
</tbody></table>
"""

ALGORITHM_PAGE = """
<p class="dp-summary">Estimates the mean by adding calibrated noise.</p>
<section class="dp-glance"><dl><dt>Privacy interpretation</dt><dd>standard</dd></dl></section>
<h2 id="references">References</h2>
<p>Dwork, C., and Roth, A. (2014). The Algorithmic Foundations. FnTTCS.</p>
<h2 id="example">Example</h2>
<pre><code>from dpstats.tasks import list_built_in_algorithms
list_built_in_algorithms(DemoTask)[&quot;AlgOne&quot;]</code></pre>
<h2 id="api">API</h2>
<pre><code>class AlgOne</code></pre>
<p class="dp-roster-link"><a href="../../roster/?core=AlgOne">1 Demo Task configured algorithm uses AlgOne →</a></p>
"""


TRANSFORMATION_PAGE = """
<p>Transports the outer task to the inner one.</p>
<h2 id="privacy-interpretation">Privacy interpretation</h2>
<p>Standard when every coordinate already declares public bounds.</p>
<h2 id="example">Example</h2>
<pre><code>from dpstats.tasks.demo import ProcOne</code></pre>
<h2 id="api">API</h2>
<pre><code>class ProcOne</code></pre>
"""

METRICS_PAGE = """
<pre><code>from dpstats.tasks import list_built_in_metrics
list_built_in_metrics(DemoTask)</code></pre>
<table class="dp-table">
<thead><tr><th>Metric</th><th>Key</th><th>Needs beyond Output</th>
<th>Description</th></tr></thead>
<tbody><tr><td>Absolute error</td><td><code>"absolute_error_to_ols"</code></td>
<td><code>reference</code></td><td>The absolute error of the release.</td></tr>
</tbody></table>
<h2 id="absolute-error-to-ols">Absolute error</h2>
<pre><code>from dpstats.tasks import list_built_in_metrics
list_built_in_metrics(DemoTask)[&quot;absolute_error_to_ols&quot;]</code></pre>
"""


def write(site: Path, route: str, article: str) -> Path:
    """Write one built page at *route*, in the shape the theme emits."""
    path = site / route / "index.html" if route else site / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<html><body><nav><a href="/{TASK}/" data-task-id="{TASK}">Demo Task</a></nav>'
        f'<article class="md-content__inner md-typeset">{article}</article>'
        "</body></html>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A small build that satisfies every check."""
    built = tmp_path / "site"
    write(built, TASK, TASK_OVERVIEW)
    write(built, f"{TASK}/definition", DEFINITION)
    write(built, LEADERBOARD, RANKINGS)
    write(built, ALGORITHMS, ALGORITHMS_PAGE)
    write(built, ALGORITHM, ALGORITHM_PAGE)
    write(built, ROSTER, ROSTER_PAGE)
    write(built, TRANSFORMATION, TRANSFORMATION_PAGE)
    write(built, METRICS, METRICS_PAGE)
    # The spec names some routes outright, and a build that publishes none of
    # them is a build with a hole in its sitemap. The pages this fixture models
    # are the task space; the rest are published as stubs so the sitemap check
    # reads a complete build.
    for route in declared_routes():
        if not (built / route / "index.html").exists():
            write(built, route.rstrip("/"), "<p>Stub.</p>")
    return built


def edit(site: Path, route: str, old: str, new: str) -> None:
    """Replace one fragment of one built page."""
    path = site / route / "index.html"
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def findings(site: Path, prefix: str) -> list[str]:
    """Only the findings one check reports."""
    return [line for line in check_site(site) if line.startswith(prefix)]


def test_a_conforming_build_reports_nothing(site: Path) -> None:
    assert check_site(site) == []


def test_a_task_space_page_outside_every_template_route_fails(site: Path) -> None:
    write(site, f"{TASK}/notes", "<p>Some notes.</p>")
    assert findings(site, "route:") == [
        f"route: {TASK}/notes/ no template route produces it."
    ]


def test_a_page_the_guide_split_has_not_removed_is_still_reported(site: Path) -> None:
    """The writer lets these through so the gate can name them, not excuse them."""
    write(site, f"{TASK}/guide", "<p>A detailed guide.</p>")
    assert findings(site, "route:") == [
        f"route: {TASK}/guide/ no template produces it and the guide split has "
        "not removed it."
    ]


def test_a_page_outside_every_task_space_is_reported_by_page(site: Path) -> None:
    """One line per page, so the baseline records it and a new one shows up."""
    write(site, "library/how-to", "<p>How to do the thing.</p>")
    write(site, "library/notes", "<p>Some notes.</p>")
    assert findings(site, "route:") == [
        "route: library/how-to/ sits outside every task space and the spec "
        "declares no template for it.",
        "route: library/notes/ sits outside every task space and the spec "
        "declares no template for it.",
    ]


def test_a_declared_route_the_build_does_not_publish_is_reported(site: Path) -> None:
    """The sitemap is checked both ways: a template with no page is a finding."""
    shutil.rmtree(site / "submit" / "submission")
    pages, _ = read_pages(site)
    assert check_sitemap(pages) == [
        "sitemap: T16 declares submit/submission/, which this build does not "
        "publish."
    ]


def test_a_task_space_the_index_does_not_link_is_reported(site: Path) -> None:
    """Routes come off global task navigation, so the registry catches a lost task."""
    write(site, "other_task/definition", DEFINITION)
    assert [
        line
        for line in check_site(site, registry=[TASK, "other_task"])
        if line.startswith("route:")
    ] == [
        "route: other_task/definition/ sits outside every task space and the "
        "spec declares no template for it.",
        "route: other_task/ publishes a task space that global task navigation does "
        "not link, so its pages match no template.",
    ]


def test_a_build_whose_navigation_links_no_task_fails(site: Path) -> None:
    pages, _ = read_pages(site)
    route = pages[0].route.rstrip("/")
    edit(site, route, f'data-task-id="{TASK}"', "")
    assert (
        "route: global task navigation links no task, so no route could be derived and no "
        "page was checked against a template." in findings(site, "route:")
    )


def test_a_documented_task_with_no_overview_fails(site: Path) -> None:
    (site / TASK / "index.html").unlink()
    assert findings(site, "route:") == [
        f"route: {LEADERBOARD}/ no template route produces it.",
        f"route: {TASK}/ is documented but this build has no page.",
    ]


def test_a_page_with_no_article_is_reported_rather_than_skipped(site: Path) -> None:
    path = site / TASK / "instances" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html><body>no article here</body></html>", encoding="utf-8")
    assert findings(site, "route:") == [
        f"route: {TASK}/instances/ publishes no article element, so nothing on "
        "it could be checked."
    ]


# One edit (or two) to the conforming build, the check that must name it, and
# exactly what that check reports. The finding text is what the committed
# baseline records, so it is asserted whole.
SINGLE_DEFECTS = [
    pytest.param(
        [(ALGORITHMS, "<td>Dwork and Roth (2014)</td>", "<td>see Theorem 3</td>")],
        "citations:",
        [
            f"citations: {ALGORITHMS}/ 'AlgOne' cites 'see Theorem 3', which is not "
            "the bibliography's in-text form, such as 'Surname (Year)' or "
            "'Surname et al. (Year)'."
        ],
        id="table citation outside the one shape",
    ),
    pytest.param(
        [(ALGORITHM, "(2014). The Algorithmic Foundations. FnTTCS.", "FnTTCS.")],
        "citations:",
        [
            f"citations: {ALGORITHM}/ carries a References section with no year, so "
            "it shows a citation fragment."
        ],
        id="reference section without its year",
    ),
    pytest.param(
        [(TASK, "bounded sample.</p>", "bounded sample. See the catalogue.</p>")],
        "vocabulary:",
        [f"vocabulary: {TASK}/ shows 'catalogue'."],
        id="retired vocabulary",
    ),
    pytest.param(
        [(TASK, "bounded sample.</p>", "bounded sample. Each seeded run counts.</p>")],
        "vocabulary:",
        [f"vocabulary: {TASK}/ shows 'seeded run'."],
        id="retired unit phrase",
    ),
    pytest.param(
        [(TASK, "<p>Release the mean", "<p>Release \\( \\mathbb{E} \\) of")],
        "math:",
        [
            f"math: {TASK}/ shows unrendered math markup "
            + str(["\\(", "\\)", "\\mathbb"])
            + "."
        ],
        id="unrendered math in visible text",
    ),
    pytest.param(
        [
            (
                TASK,
                "<p>Release the mean",
                '<div class="jp-MarkdownOutput"><p>$R/n$ and $$\\frac{R}{n}$$</p>'
                "</div><p>Release the mean",
            )
        ],
        "math:",
        [],
        id="notebook math inside the configured MathJax scope",
    ),
    pytest.param(
        [
            (
                TASK,
                "<p>Release the mean",
                '<div class="jp-MarkdownOutput"><p>$R/n</p></div><p>Release the mean',
            )
        ],
        "math:",
        [f"math: {TASK}/ shows unrendered math markup ['$']."],
        id="unclosed notebook math delimiter",
    ),
    pytest.param(
        [(f"{TASK}/definition", "<td>The released mean.</td>", "<td>The mean of</td>")],
        "cells:",
        [
            f"cells: {TASK}/definition/ Description row 1 ends on a dangling word: "
            "'The mean of'"
        ],
        id="truncated prose cell",
    ),
    pytest.param(
        [(TASK, "<p>The mean of a bounded sample.</p>", "<p>Filler B.</p>")],
        "filler:",
        [f"filler: {TASK}/ shows placeholder text ['Filler']."],
        id="placeholder text",
    ),
    pytest.param(
        [(TASK, ">Leaderboard</a>", ">Leaderboard (Preview)</a>")],
        "preview:",
        [
            f"preview: {TASK}/ suffixes link text 'Leaderboard (Preview)' with "
            "'(Preview)'."
        ],
        id="preview suffix on link text",
    ),
    pytest.param(
        [(ROSTER, 'data-dp-label="Alg One"', 'data-dp-label="Alg Nine"')],
        "parity:",
        [f"parity: {ROSTER}/ ranks 'Alg Nine', which {LEADERBOARD}/ does not " "rank."],
        id="configured algorithm the leaderboard does not rank",
    ),
    pytest.param(
        [
            (
                LEADERBOARD,
                "<tr><td>1</td><td>Alg One</td><td>96.7%</td></tr>",
                "<tr><td>1</td><td>Alg One</td></tr>",
            )
        ],
        "rankings:",
        [f"rankings: {LEADERBOARD}/ row 1 carries 2 cells against 3 data-columns."],
        id="rankings row outgrowing its declaration",
    ),
    pytest.param(
        [(LEADERBOARD, ' data-columns="rank,algorithm,share"', "")],
        "rankings:",
        [
            f"rankings: {LEADERBOARD}/ renders a rankings table that declares no "
            "data-columns, so a re-render cannot keep its shape."
        ],
        id="rankings table declaring no columns",
    ),
    pytest.param(
        [(ALGORITHM, '<h2 id="api">API</h2>', "<h2>API</h2>")],
        "one home:",
        [f"one home: {ALGORITHM}/ renders no API block under the 'api' anchor."],
        id="API block outside the fixed anchor",
    ),
]


@pytest.mark.parametrize(("edits", "check", "expected"), SINGLE_DEFECTS)
def test_one_defect_is_reported_by_its_own_check(
    site: Path, edits: list[tuple[str, str, str]], check: str, expected: list[str]
) -> None:
    for route, old, new in edits:
        edit(site, route, old, new)
    assert findings(site, check) == expected


@pytest.mark.parametrize(
    "route", ["library/tutorials/analyze_results/", f"{TASK}/methodology/"]
)
def test_maintainer_owned_pages_may_carry_a_retired_unit_phrase(
    site: Path, route: str
) -> None:
    write(site, route, "<p>The per-seed and scientific cell prose.</p>")
    assert findings(site, "vocabulary:") == []


def test_the_verb_run_is_not_a_retired_unit_word(site: Path) -> None:
    edit(site, TASK, "bounded sample.</p>", "bounded sample. It runs the seeds.</p>")
    assert findings(site, "vocabulary:") == []


def test_the_design_preview_may_carry_the_placeholder_text_it_renders(
    site: Path,
) -> None:
    """The preview's objects are invented, so its placeholders are its content."""
    edit(site, TASK, "<p>The mean of a bounded sample.</p>", "<p>Filler B.</p>")
    assert findings(site, "filler:")
    assert [
        line for line in check_site(site, preview=True) if line.startswith("filler:")
    ] == []


def test_the_design_preview_owns_its_own_area(site: Path) -> None:
    """One preview page per template is what that build is for."""
    write(site, "templates/t03-task-overview", "<p>A preview.</p>")
    assert findings(site, "route:")
    assert [
        line for line in check_site(site, preview=True) if line.startswith("route:")
    ] == []


def test_a_rankings_header_that_disagrees_with_its_declaration_fails(
    site: Path,
) -> None:
    edit(site, LEADERBOARD, '"rank,algorithm,share"', '"rank,algorithm"')
    assert f"rankings: {LEADERBOARD}/ declares 2 data-columns and heads 3." in findings(
        site, "rankings:"
    )


# ── The same route rule where the assembler writes a page ────────────────────

WRITTEN = (
    f"{TASK}/index.md",
    f"{TASK}/definition.md",
    f"{TASK}/metrics.md",
    f"{TASK}/algorithms/index.md",
    f"{TASK}/algorithms/AlgOne.md",
    f"{TASK}/transformations/index.md",
    f"{TASK}/transformations/ProcOne.md",
    f"{TASK}/algorithms/SUMMARY.md",
    f"{TASK}/guide.md",
    "library/tutorials/analyze_results.ipynb",
)


@pytest.mark.parametrize("path", WRITTEN)
def test_the_page_writer_accepts_every_page_a_template_route_produces(
    path: str,
) -> None:
    refuse_untemplated(path, [TASK])


def test_the_page_writer_refuses_a_stray_task_space_page() -> None:
    with pytest.raises(RuntimeError, match="no page template produces"):
        refuse_untemplated(f"{TASK}/notes.md", [TASK])


def test_a_class_rendered_in_two_places_fails(site: Path) -> None:
    """A task-keyed class has one home; the classic reference is the other."""
    write(site, "library/reference/algorithm/dpstats.AlgOne", "<p>x</p>")
    assert findings(site, "one home:") == [
        "one home: library/reference/algorithm/dpstats.AlgOne/ "
        f"renders AlgOne, which {ALGORITHM}/ already renders."
    ]


# ── The baseline this site's known violations are recorded in ────────────────


def test_a_build_at_its_baseline_reports_nothing() -> None:
    observed = ["route: a/ x.", "cells: b/ y."]
    assert baseline_findings(observed, group_by_check(observed)) == []


def test_a_violation_the_baseline_does_not_record_fails() -> None:
    """A stray page is new work, so it fails whatever the baseline carries."""
    assert baseline_findings(
        ["route: a/ x.", "route: stray/ no template route produces it."],
        {"route": ["route: a/ x."]},
    ) == ["baseline: not recorded: route: stray/ no template route produces it."]


def test_a_baseline_entry_that_no_longer_occurs_fails() -> None:
    """A repair deletes its line, so the file cannot rot into an exemption."""
    assert baseline_findings(
        ["route: a/ x."], {"route": ["route: a/ x.", "cells: b/ y."]}
    ) == ["baseline: does not occur, delete it: cells: b/ y."]


def test_the_committed_baseline_records_only_real_violation_lines() -> None:
    """Every recorded line is one a check reports, keyed by that check."""
    recorded = load_baseline(BASELINE_PATH)
    assert recorded
    for check, lines in recorded.items():
        assert lines == sorted(set(lines))
        assert all(line.startswith(f"{check}: ") for line in lines)


def test_a_citation_table_is_checked_but_a_metric_arguments_table_is_not() -> None:
    """The Reference column only marks a citation table beside Algorithm/Transformation.

    A Metrics page's own "Metric arguments" table also has a "Reference"
    column, but it names an input reference like ``ground_truth``, not a
    paper, so it is not held to the citation shape.
    """
    citation_table = Table(
        attributes="",
        columns=("Algorithm", "Reference"),
        rows=(("Alg One", "not a citation"),),
        markup=(),
    )
    metric_arguments_table = Table(
        attributes="",
        columns=("Reference", "Description", "Used by"),
        rows=(("ground_truth", "The released estimate.", "Absolute error"),),
        markup=(),
    )
    page = Page(route=ALGORITHM, path=Path("x"), html="", article="")
    found = check_citations(
        [page],
        {page.route: None},
        {page.route: [citation_table, metric_arguments_table]},
    )
    assert found == [
        f"citations: {page.route} 'Alg One' cites 'not a citation', which is not "
        "the bibliography's in-text form, such as 'Surname (Year)' or "
        "'Surname et al. (Year)'."
    ]


_COMPARISON_CHART_HTML = (
    '<div class="dp-chart"></div><h4 class="dp-plotted-heading">Plotted</h4>'
    '<section data-comparison-selection aria-labelledby="x">'
    '<h4 class="dp-plotted-heading">Algorithms</h4>'
    '<input role="combobox" aria-autocomplete="list" aria-controls="feature-picker-options">'
    "</section>"
    '<div data-outcome-slot="algorithm">'
    '<input role="combobox" aria-autocomplete="list" aria-controls="algorithm-picker-options">'
    "</div>"
    '<div data-outcome-slot="compare">'
    '<input role="combobox" aria-autocomplete="list" aria-controls="compare-picker-options">'
    "</div>"
    '<div class="dp-comparison-plot"></div>'
    '<script type="application/json" data-series-style></script>'
    '<input role="combobox" aria-autocomplete="list" aria-controls="baseline-picker-options">'
)


def test_a_comparison_chart_headed_algorithms_still_counts_as_named() -> None:
    """A comparison chart's "Algorithms" heading names its plotted set too.

    ``dp-plotted-heading`` is the one hook every chart's naming heading
    carries; its text differs by chart kind (the profile chart says
    "Plotted", a comparison chart says "Algorithms"), so the check matches
    the element, not one literal text.
    """
    page = Page(
        route=f"{TASK}/public-results/",
        path=Path("x"),
        html=_COMPARISON_CHART_HTML,
        article="",
    )
    assert check_results_controls([page], {page.route: TEMPLATES["T5"]}) == []


def test_the_baseline_picker_is_not_counted_as_a_comparison_picker() -> None:
    """Removing a real comparison picker still fails, once the baseline picker is excluded."""
    html = _COMPARISON_CHART_HTML.replace(
        '<input role="combobox" aria-autocomplete="list" aria-controls="compare-picker-options">',
        "",
    )
    page = Page(route=f"{TASK}/public-results/", path=Path("x"), html=html, article="")
    found = check_results_controls([page], {page.route: TEMPLATES["T5"]})
    assert found == [
        f"results controls: {page.route} pairs 2 algorithm pickers with 3 "
        "comparison choices, and each comparison choice is made with a picker "
        "of its own."
    ]


def test_a_deliberately_stray_page_is_not_covered_by_the_baseline(site: Path) -> None:
    """The stray page still fails the build with the baseline in place."""
    write(site, f"{TASK}/notes", "<p>Some notes.</p>")
    observed = check_site(site)
    baseline = load_baseline(BASELINE_PATH)
    assert (
        f"baseline: not recorded: route: {TASK}/notes/ no template route "
        "produces it." in baseline_findings(observed, baseline)
    )


@pytest.mark.parametrize(
    ("cited", "accepted"),
    [
        ("Wang (2018)", True),
        ("Balle and Wang (2018)", True),
        ("Dick, Gillenwater and Joseph (2023)", True),
        ("Aumüller, Lebeda, Nelson and Pagh (2024)", True),
        ("Harini S and Tandon (2026)", True),
        ("Alabi et al. (2022)", True),
        ("Wang, Y.-X. (2018)", False),
        ("see Theorem 3", False),
    ],
)
def test_a_table_citation_is_the_bibliographys_in_text_form(
    cited: str, accepted: bool
) -> None:
    assert bool(TABLE_CITATION.match(cited)) is accepted
