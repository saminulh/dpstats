"""Templates are the generator: the spec decides what a page may contain."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest


from templates.filler import (
    algorithm_component,
    filler_registry,
    transformation_component,
)
from templates.model import (
    Access,
    Component,
    Methodology,
    Metrics,
    AlgorithmsIndex,
    TransformationsIndex,
)
from site_copy import copy_span
from templates.render import (
    MissingSource,
    render,
)
from templates.spec import TEMPLATES
from tests.unit.docs.conftest import WriteCopy, parse_html

HEADING = re.compile(r"^## (.+)$", re.M)
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
LINKED_PAGE = re.compile(r" · \[[^\]]+\]\([^)]*\)$")
COLUMNS = re.compile(r'<th scope="col"[^>]*>(.*?)</th>')
TABLE_HEAD = re.compile(r"<thead><tr>(.*?)</tr></thead>", re.S)
SEMANTICS_URL = (
    "../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation/"
)

# Every preview page a renderer fills from a page record, and the page record
# kind each one reads. A template that renders more than one kind of page names
# its template and, after a dash, the kind.
TASK_SPACE = (
    "T3",
    "T6",
    "T7",
    "T8",
    "T9",
    "T9-transformation",
    "T18",
    "T21",
    "T22",
    "T23",
)

# The Corpus area's pages, filled from the release a build reads.
CORPUS = ("T2-corpus", "T20", "T24")

# The site-wide pages filled from the release a build reads.
SITE_WIDE = ("T15",)

# The templates whose pages are the live results views. They read a prepared
# snapshot rather than a page record, so their sections are read back off the
# page the live renderers emit.
RESULTS = ("T1", "T4", "T5")

# The templates whose pages a generator writes or an author writes, so no
# filler record fills them.
GENERATED_OR_AUTHORED = ("T13", "T16", "T17")


def test_every_template_is_declared_once() -> None:
    rendered = {name.partition("-")[0] for name in (*TASK_SPACE, *CORPUS, *SITE_WIDE)}
    assert set(TEMPLATES) == {*rendered, *RESULTS, *GENERATED_OR_AUTHORED}
    routes = [route for template in TEMPLATES.values() for route in template.routes]
    assert len(set(routes)) == len(routes)
    # A template produces more than one route only where one page kind covers
    # more than one object or one area.
    assert {
        template.name for template in TEMPLATES.values() if len(template.routes) > 1
    } == {"T2", "T9", "T13", "T17"}
    for template in TEMPLATES.values():
        keys = [section.key for section in template.sections]
        assert len(set(keys)) == len(keys), template.name


def test_a_rendered_page_carries_only_its_template_s_headings() -> None:
    registry = filler_registry()
    for name in (*TASK_SPACE, *CORPUS, *SITE_WIDE):
        template = TEMPLATES[name.partition("-")[0]]
        declared = [section.title for section in template.sections if section.title]
        # A heading may link the page it summarises, in its words or after them;
        # the declared words are what count.
        headings = [
            LINK.sub(r"\1", LINKED_PAGE.sub("", heading))
            for heading in HEADING.findall(render(template, registry[name]))
        ]
        if name == "T18":
            # The metric sections are one declared section carrying a heading
            # per metric, so the metric registry decides the headings.
            metrics = registry[name]
            assert isinstance(metrics, Metrics)
            assert headings == [
                f"{metric.key} {{#{metric.anchor}}}" for metric in metrics.metrics
            ] + [copy_span("task", "metric_arguments_heading") + " {#metric-arguments}"]
            continue
        if name == "T15":
            # The narrative carries its own headings, as the release wrote
            # them; the template's own heading follows them.
            narrative = registry[name]
            assert isinstance(narrative, Methodology)
            assert headings == [
                *HEADING.findall(narrative.narrative),
                *declared,
            ]
            continue
        if name in ("T7", "T8"):
            # The table section carries one heading per named group, so the
            # groups decide the headings.
            index = registry[name]
            assert isinstance(index, (AlgorithmsIndex, TransformationsIndex))
            assert headings == [group.title for group in index.groups if group.title]
            continue
        if name == "T24":
            # The paper's own section reads its heading from the site copy.
            assert headings == [*declared, copy_span("corpus", "source_heading")]
            continue
        present = [heading for heading in headings if heading in declared]
        assert present == headings, name
        assert present == [title for title in declared if title in present], name


def test_a_required_section_without_a_source_refuses_the_page() -> None:
    """A missing required source stops the build; it is never left blank."""
    empty = TransformationsIndex(title="Task Name: transformations", groups=())
    with pytest.raises(MissingSource, match="'table'"):
        render(TEMPLATES["T8"], empty)


def test_an_optional_section_without_a_source_is_omitted() -> None:
    page = filler_registry()["T9"]
    assert isinstance(page, Component)
    full = render(TEMPLATES["T9"], page)
    requires = f'<dt>{copy_span("component", "glance_requires")}</dt>'
    assert requires in full and "## Parameters" in full
    # The same page whose class declares no Requires block or parameters
    # carries no label or heading for them, rather than one over nothing.
    bare = render(TEMPLATES["T9"], replace(page, requires=(), parameters=()))
    assert requires not in bare and "## Parameters" not in bare


def test_a_component_page_shows_one_example_and_no_access_block() -> None:
    for page in (algorithm_component(), transformation_component()):
        text = render(TEMPLATES["T9"], page)
        assert "## Example" in text and "## Usage" not in text
        assert "list_built_in_algorithms" not in text
        assert 'data-copy-slot="task.access"' not in text


def test_a_record_of_the_wrong_kind_is_refused() -> None:
    with pytest.raises(TypeError):
        render(TEMPLATES["T9"], Access(import_line="from x import y"))


def _table_heads(page: str) -> list[tuple[str, ...]]:
    """Each table's column headers, as a reader sees them."""
    return [
        tuple(header.text for header in head.find_all(tag="th"))
        for head in parse_html(page).find_all(tag="thead")
    ]


def _parameter_names(page: str) -> set[str]:
    """The name every Parameters entry declares."""
    return {
        name.text
        for name in parse_html(page).find_all(**{"class": "dp-parameter-name"})
    }


def _glance_values(page: str) -> dict[str, str]:
    """Each at-a-glance label's slot and its value's source text, in order."""
    block = page[page.index('<section class="dp-glance"') : page.index("</section>")]
    return {
        re.search(r'data-copy-slot="component\.([a-z_]+)"', label)[1]: value
        for label, value in re.findall(
            r"<dt>(.*?)</dt>\n<dd[^>]*>(.*?)</dd>", block, re.S
        )
    }


def _structure(page: str) -> list[object]:
    """What a component page is, apart from the values it happens to carry."""
    return [
        HEADING.findall(page),
        _table_heads(page),
        list(_glance_values(page)),
        page.count("{ .dp-summary }"),
        page.count("```python"),
        page.count("<details"),
    ]


def test_the_two_component_kinds_differ_only_in_the_declared_deviations() -> None:
    """One template, one renderer: an Algorithm page and a Transformation page.

    The ruling lists the deviations by name, so this test removes exactly
    those and then requires the two pages to be the same page.
    """
    algorithm = render(TEMPLATES["T9"], algorithm_component())
    transformation = render(TEMPLATES["T9"], transformation_component())

    # Task: a native task against an outer → inner pair.
    native = f'<dt>{copy_span("component", "glance_native_task")}</dt>'
    outer = f'<dt>{copy_span("component", "glance_task")}</dt>'
    assert native in algorithm and outer in transformation
    assert native not in transformation
    assert "→" in _glance_values(transformation)["glance_task"]
    # Roster links: an Algorithm's core against a Transformation's chain.
    assert "⟨component.roster_algorithm_many⟩" in algorithm
    assert "⟨component.roster_transformation_many⟩" in transformation
    # Authored method notes are allowed in a Transformation's implementation notes.
    assert "authored method note" in transformation
    # Parameters source: an Algorithm's hyperparameters against a Transformation's
    # configuration fields, under one heading.
    assert "## Parameters" in algorithm and "## Parameters" in transformation
    assert "clip_norm" in _parameter_names(algorithm)
    assert "clip_norm" not in _parameter_names(transformation)
    assert "inner_share" in _parameter_names(transformation)
    assert "inner_share" not in _parameter_names(algorithm)

    settled = transformation.replace(outer, native)
    assert _structure(algorithm) == _structure(settled)


def test_a_component_declaring_no_class_stops_the_page() -> None:
    """A component with no source for its class is escalated, not published."""
    silent = replace(algorithm_component(), privacy_interpretation="")
    with pytest.raises(MissingSource, match="privacy interpretation"):
        render(TEMPLATES["T9"], silent)


def test_a_component_states_its_class_at_a_glance() -> None:
    """The class and a link to its explanation, with no heading or composition prose."""
    for record in (algorithm_component(), transformation_component()):
        page = render(TEMPLATES["T9"], record)
        shown = _glance_values(page)["glance_interpretation"]
        assert f">{record.privacy_interpretation}</span>" in shown
        assert f"[{record.semantics_link.text}]({record.semantics_link.url})" in shown
        assert "## Privacy interpretation" not in page
        assert "weakest" not in page


def test_the_glance_reads_task_class_requires_and_source_in_order() -> None:
    """Source states the cited work once and is absent for a class citing none."""
    page = render(TEMPLATES["T9"], algorithm_component())
    assert list(_glance_values(page)) == [
        "glance_native_task",
        "glance_interpretation",
        "glance_requires",
        "glance_source",
    ]
    assert "An implementation of" not in _glance_values(page)["glance_source"]
    cited = render(TEMPLATES["T9"], replace(algorithm_component(), source=""))
    assert "glance_source" not in _glance_values(cited)


def test_a_parameter_declares_its_type_and_default_then_its_note(
    write_copy: WriteCopy,
) -> None:
    """``name : type, default value``; the note follows the description in its entry.

    No search range is shown anywhere on the page.
    """
    write_copy(
        {
            "component": {
                "parameter_default": "default {default}",
                "parameter_required": "required",
            }
        }
    )
    record = algorithm_component()
    page = parse_html(render(TEMPLATES["T9"], record))
    entries = page.find(**{"class": "dp-parameters"})
    first = record.parameters[0]
    term = entries.find(tag="dt", id=f"param-{first.name}")
    assert term.text == f"{first.name} : float > 0, default 1.0"
    notes = [note.text for note in entries.find_all(**{"class": "dp-parameter-note"})]
    assert notes == [item.note for item in record.parameters if item.note]
    assert "Search" not in render(TEMPLATES["T9"], record)
    required = replace(first, default="")
    bare = parse_html(render(TEMPLATES["T9"], replace(record, parameters=(required,))))
    assert bare.find(tag="dt", id=f"param-{first.name}").text.endswith(", required")


def test_implementation_notes_fold_under_one_disclosure() -> None:
    """The notes sit in one collapsed block rather than under a heading."""
    page = render(TEMPLATES["T9"], algorithm_component())
    assert "## Implementation notes" not in page
    assert (
        '<details class="dp-fold" markdown="1">\n<summary>'
        + copy_span("component", "implementation_notes")
        in page
    )


def test_the_roster_links_name_their_tasks_and_counts(
    write_copy: WriteCopy,
) -> None:
    """The first task's link carries the sentence; the others follow by name."""
    write_copy(
        {
            "component": {
                "roster_algorithm_one": "{count} {task} configured algorithm uses {name}",
                "roster_algorithm_many": "{count} {task} configured algorithms use {name}",
            }
        }
    )
    page = parse_html(render(TEMPLATES["T9"], algorithm_component()))
    line = page.find(**{"class": "dp-roster-link"})
    links = line.find_all(tag="a")
    assert [link.text for link in links] == [
        "12 Task Name configured algorithms use FillerAlgorithm →",
        "Second Task Name (3)",
    ]
    single = replace(
        algorithm_component(),
        roster=(replace(algorithm_component().roster[0], count=1),),
    )
    lone = parse_html(render(TEMPLATES["T9"], single)).find(
        **{"class": "dp-roster-link"}
    )
    assert lone.text == "1 Task Name configured algorithm uses FillerAlgorithm →"
    assert "dp-roster-link" not in render(
        TEMPLATES["T9"], replace(algorithm_component(), roster=())
    )


def test_the_task_tables_carry_no_scope_sentence() -> None:
    """The class label's help carries the meaning; no sentence restates it."""
    registry = filler_registry()
    for name in ("T7", "T8"):
        page = render(TEMPLATES[name], registry[name])
        assert "weakest" not in page, name
        assert "dp-note" not in page, name


def test_the_class_filter_label_carries_the_term_s_definition() -> None:
    """Both task tables' filter label opens the maintainer's definition.

    A page without a definition shows the plain label and no help.
    """
    registry = filler_registry()
    for name in ("T7", "T8"):
        index = registry[name]
        page = parse_html(render(TEMPLATES[name], index))
        trigger = page.find("data-dp-term")
        assert trigger.text == "Privacy interpretation", name
        help_text = page.find(id=trigger.attrs["data-dp-term"])
        assert help_text.text == getattr(index, "class_help").text, name
        plain = render(TEMPLATES[name], replace(index, class_help=None))
        assert "data-dp-term" not in plain, name


def test_the_transformations_tables_carry_a_filterable_privacy_interpretation() -> None:
    """One table per group, the class column and the chips T7 already has.

    Only the conversion group names the task a row hands the problem to.
    """
    index = filler_registry()["T8"]
    assert isinstance(index, TransformationsIndex)
    page = parse_html(render(TEMPLATES["T8"], index))
    assert _table_heads(render(TEMPLATES["T8"], index)) == [
        (
            ("Transformation", "What it does", "⟨task.hands_off_column⟩")
            if group.converts
            else ("Transformation", "What it does")
        )
        + ("Privacy interpretation",)
        + (("Reference",) if any(row.reference for row in group.rows) else ())
        for group in index.groups
    ]
    chips = page.find("data-dp-class-filter")
    assert chips.find_all(data_dp_class="standard")
    # Every row carries a class the chips can match, never a blank cell.
    rows = [row for table in page.find_all(tag="table") for row in table.body_rows()]
    classes = [row.find(**{"class": "dp-class-badge"}).text for row in rows]
    assert len(classes) == sum(len(group.rows) for group in index.groups)
    assert set(classes) <= set(index.classes)


def test_both_task_tables_read_their_classes_strongest_first() -> None:
    """Chips and column sort follow the library's order, not the alphabet.

    The column's order is declared on its header, so the shared sort script
    orders the cells the way the chips above them are ordered.
    """
    registry = filler_registry()
    for name in ("T7", "T8"):
        index = registry[name]
        declared = getattr(index, "classes")
        page = render(TEMPLATES[name], index)
        document = parse_html(page)
        chips = [
            chip.attrs["data-dp-class"] for chip in document.find_all("data-dp-class")
        ]
        assert chips == ["All", *declared], name
        ordered = document.find("data-dp-order", tag="th")
        assert ordered.attrs["data-dp-order"].split(",") == list(declared), name


def test_a_transformations_table_cell_is_never_blank() -> None:
    """A Transformation with no declared class stops the table, not the reader."""
    index = filler_registry()["T8"]
    assert isinstance(index, TransformationsIndex)
    group = index.groups[0]
    row = replace(group.rows[0], privacy_interpretation="")
    silent = replace(index, groups=(replace(group, rows=(row,)),))
    with pytest.raises(MissingSource, match=row.name.text):
        render(TEMPLATES["T8"], silent)


def test_an_empty_note_renders_nothing() -> None:
    from templates.blocks import note

    assert note("") == ""
    assert note("  ") == ""
    assert note("Built from the release.") == (
        "Built from the release.\n{ .dp-note }\n"
    )
