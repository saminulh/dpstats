"""Fill one template's declared sections from one page record.

``render`` is the only way a task-space page is written. It walks the
template's sections in declared order, asks this module for each one's text,
omits an optional section whose source produced nothing, and refuses a page
whose required section is empty rather than publishing a gap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape

from privacy_interpretations import interpretation_badge
from site_controls import SELECT_CLASS, fits_chips
from site_copy import copy_attribute, copy_html, copy_span, copy_text

from .blocks import (
    anchor,
    code,
    link,
    markdown_table,
    note,
    python_block,
    table,
)
from .model import (
    TRANSFORMATION,
    AlgorithmGroup,
    AlgorithmsIndex,
    AreaIndex,
    Component,
    ConfiguredAlgorithm,
    CorpusIndex,
    Definition,
    InstancePage,
    Methodology,
    Metrics,
    ModelBlock,
    RosterRow,
    TransformationGroup,
    TransformationsIndex,
    BaselineEntry,
    ReaderTerm,
    RosterLanding,
    RosterPage,
    Source,
    TaskOverview,
    Access,
)
from .spec import (
    FACT_TABLE,
    INSTANCE_COLUMNS_TABLE,
    OUTCOME_TABLE,
    BASELINE_TABLE,
    RECIPE_TABLE,
    ROSTER_LANDING_TABLE,
    STANDING_TABLE,
    TemplateSpec,
)

# The marker a task-space page carries so the sidebar shows that task's pages
# and the way back to all tasks, rather than every task in the build.
TASK_PAGE_MARKER = '<div class="dp-task-page" hidden></div>'

# The column both task tables filter and sort on, named once.
CLASS_COLUMN = "Privacy interpretation"
DESCRIPTION_COLUMN = "What it does"
# The conversion group's extra column; its heading is a text slot.
HANDS_OFF_COLUMN = "Hands off to"


class MissingSource(ValueError):
    """A template's required section has no text in this build."""


def render(template: TemplateSpec, obj: object) -> str:
    """Return one page's Markdown, section by section, in the template's order.

    Args:
        template: The template the page is an instance of.
        obj: The page record this template reads, as :mod:`templates.model`
            declares it.

    Returns:
        The page's Markdown, opening with its title.

    Raises:
        MissingSource: A required section produced no text.
        TypeError: The record does not belong to this template.
    """
    if template.name == "T2" and isinstance(obj, AreaIndex):
        return _page(template, obj.title, _area_index(obj), marker=False)
    if template.name == "T2" and isinstance(obj, CorpusIndex):
        return _page(template, obj.title, _corpus_index(obj), marker=False)
    if template.name == "T3" and isinstance(obj, TaskOverview):
        return _page(template, obj.name, _task_overview(obj))
    if template.name == "T6" and isinstance(obj, Definition):
        return _page(template, obj.title, _definition(obj))
    if template.name == "T7" and isinstance(obj, AlgorithmsIndex):
        return _page(template, obj.title, _algorithms_index(obj))
    if template.name == "T8" and isinstance(obj, TransformationsIndex):
        return _page(template, obj.title, _transformations_index(obj))
    if template.name == "T9" and isinstance(obj, Component):
        return _page(template, obj.name, _component(obj))
    if template.name == "T15" and isinstance(obj, Methodology):
        return _page(template, obj.title, _methodology(obj), marker=False)
    if template.name == "T18" and isinstance(obj, Metrics):
        return _page(template, obj.title, _metrics(obj))
    if template.name == "T20" and isinstance(obj, Source):
        return _page(template, obj.title, _source(obj), marker=False)
    if template.name == "T21" and isinstance(obj, RosterPage):
        return _page(
            template,
            obj.algorithm.label,
            _roster_page(obj),
            headings=_roster_headings(obj),
        )
    if template.name == "T22" and isinstance(obj, RosterLanding):
        return _page(template, obj.title, _roster_landing(obj))
    if template.name == "T24" and isinstance(obj, InstancePage):
        return _page(
            template,
            obj.title,
            _instance_page(obj),
            headings={"source": copy_span("corpus", "source_heading")},
        )
    raise TypeError(f"{type(obj).__name__} is not a {template.name} page record.")


def _page(
    template: TemplateSpec,
    title: str,
    bodies: Mapping[str, str],
    *,
    marker: bool = True,
    headings: Mapping[str, str] | None = None,
) -> str:
    """Assemble one page from the sections its template declares.

    *headings* replaces a section's heading with a rendered one, such as the
    same words carrying a link.
    """
    parts = [TASK_PAGE_MARKER] if marker else []
    parts.append(f"# {title}")
    for section in template.sections:
        body = bodies.get(section.key, "").strip()
        if not body:
            if section.required:
                raise MissingSource(
                    f"{template.name} requires the {section.key!r} section, and "
                    f"its source ({section.source}) produced nothing for {title!r}."
                )
            continue
        heading = (headings or {}).get(section.key) or section.title
        if heading:
            parts.append(f"## {heading}")
        parts.append(body)
    if len(parts) == (2 if marker else 1):
        raise MissingSource(
            f"{template.name} rendered no section at all for {title!r}, so the "
            "page would carry its title and nothing else."
        )
    return "\n\n".join(parts) + "\n"


def _access(access: Access) -> str:
    """One Access block: the import line, then the keyed accessor it has."""
    lines = [access.import_line]
    if access.accessor:
        lines.append(access.accessor)
    return python_block(lines)


def _task_overview(page: TaskOverview) -> dict[str, str]:
    return {
        "description": page.description,
        "access": _access(page.access),
        "quick_links": " ·\n".join(
            link(item.text, item.url) for item in page.quick_links
        ),
        "problem": page.problem,
        "metrics": (
            markdown_table(
                (copy_span("task", "metric_column"), "Description"),
                [
                    (link(name.text, name.url), description)
                    for name, description in page.metrics
                ],
            )
            if page.metrics
            else ""
        ),
    }


def _model_block(block: ModelBlock) -> str:
    rows = [(code(row.name), row.type, row.description) for row in block.rows]
    return "\n\n".join(
        part.strip()
        for part in (_access(block.access), block.summary, table(block.columns, rows))
        if part.strip()
    )


def _definition(page: Definition) -> dict[str, str]:
    bodies = {block.title.lower(): _model_block(block) for block in page.models}
    bodies["notes"] = page.notes
    return bodies


def _term(word: str, term: ReaderTerm | None, depth: int) -> str:
    """*word*, carrying the maintainer's definition as help where one exists.

    The help sits hidden beside its word; the site script turns the pair into
    the same popover the chart readings use. *depth* reaches the site root
    for the definition's link. An undefined term is the plain word.
    """
    if term is None:
        return escape(word)
    more = (
        f' <a href="{"../" * depth}{escape(term.link, quote=True)}">'
        + copy_span("task", "term_more")
        + "</a>"
        if term.link
        else ""
    )
    help_id = f"dp-term-{term.key}"
    return (
        f'<button type="button" class="dp-term" data-dp-term="{help_id}">'
        f"{escape(word)}</button>"
        f'<span id="{help_id}" hidden>{escape(term.text)}{more}</span>'
    )


def _class_chips(classes: tuple[str, ...], term: ReaderTerm | None = None) -> str:
    """The privacy-interpretation filter both task tables carry.

    *classes* arrives in the order the library declares them, strongest first,
    so the chips and the column's sort order are the one reading order. The
    filter's label carries the term's definition; both task tables sit two
    levels below the site root.
    """
    if not classes:
        return ""
    # One exclusive choice: native radios as the site's segmented control.
    options = "".join(
        '<span class="dp-segmented-option">'
        f'<input class="btn-check" type="radio" name="class-filter" id="class-filter-{name}" '
        f'value="{name}" data-dp-class="{name}"{" checked" if name == "All" else ""}>'
        f'<label class="btn btn-sm" for="class-filter-{name}">'
        + (copy_span("task", "class_filter_all") if name == "All" else name)
        + "</label></span>"
        for name in ("All", *classes)
    )
    return (
        '<div class="dp-segmented" role="radiogroup" aria-labelledby="class-filter-legend" data-dp-class-filter>\n'
        f'<span class="dp-chip-label" id="class-filter-legend">{_term(CLASS_COLUMN, term, 2)}</span>\n'
        f'<div class="dp-segmented-row">{options}</div>\n'
        "</div>\n"
    )


def _class_cell(name: str, declared: str) -> str:
    """One privacy-interpretation cell, which is never blank.

    Raises:
        MissingSource: The component declares no class, so the cell would be
            an em dash in a column a reader filters on.
    """
    if not declared:
        raise MissingSource(
            f"{name} declares no privacy interpretation, so its Privacy "
            "interpretation cell would be blank."
        )
    return interpretation_badge(declared, "dp-class-badge")


def _group_block(title: str, intro: str, body: str) -> str:
    """One group of a task table page: its heading and intro, then its table.

    An unheaded group is the page's one table, in the same wrapper so the
    class chips and shared column widths treat every page alike.
    """
    heading = f"## {title}\n\n" if title else ""
    lead = f"{intro}\n\n" if intro else ""
    return (
        f'<div class="dp-table-group" markdown="1">\n\n{heading}{lead}'
        + body
        + "\n</div>\n"
    )


def _algorithm_group(group: AlgorithmGroup, classes: tuple[str, ...]) -> str:
    """One mechanism family's table; a family citing no paper has no Reference."""
    cited = any(reference for *_, reference in group.rows)
    columns = ("Algorithm", DESCRIPTION_COLUMN, CLASS_COLUMN)
    columns += ("Reference",) if cited else ()
    rows = [
        (
            anchor(name.text, name.url),
            description,
            _class_cell(name.text, declared),
            *((reference,) if cited else ()),
        )
        for name, description, declared, reference in group.rows
    ]
    return _group_block(
        group.title,
        "",
        table(columns, rows, ordered={CLASS_COLUMN: classes}, empty=""),
    )


def _algorithms_index(page: AlgorithmsIndex) -> dict[str, str]:
    return {
        "table": (
            _class_chips(page.classes, page.class_help)
            + "\n"
            + "\n".join(_algorithm_group(group, page.classes) for group in page.groups)
            if page.groups
            else ""
        ),
    }


def _transformation_group(group: TransformationGroup, classes: tuple[str, ...]) -> str:
    """One role's heading, its introducing line and its table.

    Only the conversion group names the task each row hands the problem to;
    every other transformation keeps the task it was given. A group none of whose
    transformations cites a paper carries no Reference column.
    """
    cited = any(row.reference for row in group.rows)
    columns = ("Transformation", DESCRIPTION_COLUMN)
    if group.converts:
        columns += (HANDS_OFF_COLUMN,)
    columns += (CLASS_COLUMN,)
    if cited:
        columns += ("Reference",)
    rows = [
        (
            anchor(row.name.text, row.name.url),
            row.description,
            *((anchor(row.inner.text, row.inner.url),) if group.converts else ()),
            _class_cell(row.name.text, row.privacy_interpretation),
            *((row.reference,) if cited else ()),
        )
        for row in group.rows
    ]
    return _group_block(
        group.title,
        group.summary,
        table(
            columns,
            rows,
            ordered={CLASS_COLUMN: classes},
            empty="",
            headings={HANDS_OFF_COLUMN: copy_span("task", "hands_off_column")},
        ),
    )


def _transformations_index(page: TransformationsIndex) -> dict[str, str]:
    return {
        "table": (
            _class_chips(page.classes, page.class_help)
            + "\n"
            + "\n".join(
                _transformation_group(group, page.classes) for group in page.groups
            )
            if page.groups
            else ""
        ),
    }


def _component(page: Component) -> dict[str, str]:
    return {
        "summary": f"{page.summary}\n{{ .dp-summary }}\n",
        "description": page.description,
        "glance": _glance(page),
        "parameters": _parameters(page),
        "implementation_notes": (
            '<details class="dp-fold" markdown="1">\n'
            "<summary>"
            + copy_span("component", "implementation_notes")
            + "</summary>\n\n"
            + f"{page.implementation_notes}\n\n</details>"
            if page.implementation_notes
            else ""
        ),
        "reference": page.reference,
        "usage": python_block(page.usage.splitlines()),
        "api": _api(page),
        "roster": _component_roster(page),
    }


def _glance(page: Component) -> str:
    """The task, the class, what the component requires, and the work it follows.

    The block is Markdown inside HTML, so the task links are the page's own
    Markdown links and the build resolves them like any other.

    Raises:
        MissingSource: The component declares no privacy interpretation.
    """
    if not page.privacy_interpretation:
        raise MissingSource(
            f"T9 states each component's privacy interpretation at a glance, and "
            f"{page.name!r} declares none."
        )
    task_label = "glance_task" if page.kind == TRANSFORMATION else "glance_native_task"
    shown = interpretation_badge(
        page.privacy_interpretation,
        f"dp-badge dp-badge--{page.privacy_interpretation.lower()} badge",
    )
    explained = link(page.semantics_link.text, page.semantics_link.url)
    rows = [
        (task_label, f'<dd markdown="span">{page.task}</dd>'),
        ("glance_interpretation", f'<dd markdown="span">{shown} · {explained}</dd>'),
    ]
    if page.requires:
        items = "\n".join(f"<li>{item}</li>" for item in page.requires)
        rows.append(("glance_requires", f"<dd><ul>\n{items}\n</ul></dd>"))
    if page.source:
        rows.append(("glance_source", f"<dd>{page.source}</dd>"))
    body = "\n".join(
        f'<dt>{copy_span("component", key)}</dt>\n{value}' for key, value in rows
    )
    spoken = escape(copy_text("component", "glance_label"), quote=True)
    return (
        f'<section class="dp-glance" aria-label="{spoken}" markdown="1">\n'
        f'<dl markdown="1">\n{body}\n</dl>\n</section>\n'
    )


def _parameters(page: Component) -> str:
    """One entry per parameter: its declaration, then what it is and why its default.

    A ``Parametrization notes:`` entry naming no parameter follows the list
    rather than being dropped.
    """
    entries = []
    for parameter in page.parameters:
        default = ", " + (
            copy_span(
                "component", "parameter_default", fields={"default": parameter.default}
            )
            if parameter.default
            else copy_span("component", "parameter_required")
        )
        body = "".join(
            f'<p class="{kind}">{text}</p>' if kind else f"<p>{text}</p>"
            for kind, text in (
                ("", parameter.description),
                ("dp-parameter-note", parameter.note),
            )
            if text
        )
        entries.append(
            f'<dt id="param-{escape(parameter.name, quote=True)}">'
            f'<span class="dp-parameter-name">{escape(parameter.name)}</span>'
            f' <span class="dp-parameter-type">: {parameter.type}{default}</span></dt>\n'
            f"<dd>{body}</dd>"
        )
    parts = []
    if entries:
        parts.append('<dl class="dp-parameters">\n' + "\n".join(entries) + "\n</dl>\n")
    if page.parametrization_notes:
        parts.append(page.parametrization_notes)
    return "\n".join(parts)


def _component_roster(page: Component) -> str:
    """The Roster links: the component's own task first, then every other task.

    Each opens that task's Roster filtered to the configured algorithms using
    the component: as their core for an Algorithm, anywhere in their chain
    for a Transformation.
    """
    if not page.roster:
        return ""
    first, *rest = page.roster
    kind = "transformation" if page.kind == TRANSFORMATION else "algorithm"
    number = "one" if first.count == 1 else "many"
    text = copy_span(
        "component",
        f"roster_{kind}_{number}",
        fields={"count": str(first.count), "task": first.task, "name": page.name},
    )
    shown = f'<a href="{escape(first.url, quote=True)}">{text} →</a>'
    if rest:
        shown += (
            "; "
            + copy_span("component", "roster_also_in")
            + " "
            + ", ".join(
                f'<a href="{escape(item.url, quote=True)}">'
                f"{escape(item.task)} ({item.count})</a>"
                for item in rest
            )
        )
    return f'<p class="dp-roster-link">{shown}</p>'


def _api(page: Component) -> str:
    """The class signature and its public methods, as the class declares them."""
    parts = [python_block(page.api_signature.splitlines())]
    for method in page.api_methods:
        parts.append(f"### `{method.name}`")
        parts.append(python_block([method.signature]))
        if method.summary:
            parts.append(method.summary)
        if method.parameters:
            parts.append(
                table(
                    ("Parameter", "Type", "Description"),
                    [
                        (code(row.name), row.type, row.description)
                        for row in method.parameters
                    ],
                    headings={
                        "Parameter": copy_span("component", "api_parameter_column"),
                        "Type": copy_span("component", "api_type_column"),
                    },
                )
            )
    return "\n\n".join(part.strip() for part in parts)


def _option(value: str, text: str) -> str:
    """One option in a filter's native select."""
    return f'<option value="{escape(value, quote=True)}">{escape(text)}</option>'


def _corpus_filters(page: CorpusIndex) -> str:
    """The search, task and field controls, and the state they leave behind.

    The table below them is server-rendered in full, so these narrow rows a
    reader already has. They stay hidden until the script that drives them
    runs, rather than offering a control that would do nothing.
    """
    tasks = "".join(_option(value, text) for value, text in page.tasks)
    fields = "".join(_option(name, name) for name in page.fields)
    return (
        '<div class="dp-corpus-filters" data-dp-corpus-filter hidden>\n'
        '<label class="dp-corpus-field"><span class="dp-chip-label">'
        + copy_span("corpus", "search_label")
        + "</span>"
        '<input type="search" class="dp-input form-control form-control-sm" data-dp-corpus-search></label>\n'
        '<label class="dp-corpus-field"><span class="dp-chip-label">'
        + copy_span("corpus", "task_filter")
        + "</span>"
        f'<select class="{SELECT_CLASS}" data-dp-corpus-task>'
        + _option("", copy_text("corpus", "all_tasks"))
        + f"{tasks}</select></label>\n"
        '<label class="dp-corpus-field"><span class="dp-chip-label">'
        + copy_span("site", "fact_field")
        + "</span>"
        f'<select class="{SELECT_CLASS}" data-dp-corpus-field>'
        + _option("", copy_text("corpus", "all_fields"))
        + f"{fields}</select></label>\n"
        '<button type="button" class="dp-action btn btn-sm" data-dp-corpus-clear>'
        + copy_span("corpus", "clear_filters")
        + "</button>\n"
        "</div>\n"
        + copy_html(
            "corpus",
            "empty_selection",
            classes="dp-empty",
            attributes=" data-dp-corpus-empty hidden",
        )
        + "\n"
    )


def _corpus_index(page: CorpusIndex) -> dict[str, str]:
    return {
        "banner": note(page.banner),
        "summary": page.summary + (f"\n\n{page.data_link}" if page.data_link else ""),
        "sources": (
            _corpus_filters(page)
            + table(
                page.columns,
                page.rows,
                numeric=(1, 3, 4),
                row_attributes=page.filters,
                empty="",
            )
            if page.rows
            else ""
        ),
    }


def _area_index(page: AreaIndex) -> dict[str, str]:
    """One area landing: the pages its navigation declares, and nothing else."""
    return {
        "pages": (
            markdown_table(
                ("Page", "Description"),
                [(link(item.text, item.url), summary) for item, summary in page.pages],
            )
            if page.pages
            else ""
        )
    }


def _source(page: Source) -> dict[str, str]:
    def cited(entry: str, doi: str) -> str:
        """One work's citation with its own copy button, or its bare DOI."""
        if entry:
            # The entry is already HTML from the bibliography renderer.
            return (
                f'<div data-dp-reference><p class="dp-citation" data-dp-citation>{entry}</p>\n'
                '<button type="button" class="dp-action btn btn-sm" data-dp-copy hidden>'
                + copy_span("corpus", "copy_citation")
                + "</button></div>\n"
            )
        return f"<p>{anchor(doi, doi)}</p>\n" if doi else ""

    note_html = (
        f'<p class="dp-note">{escape(page.citation_note)}</p>'
        if page.citation_note
        else ""
    )
    return {
        "header": note(page.supporting) if page.supporting else "",
        "paper": cited(page.paper, page.paper_doi) or note_html,
        "data": cited(page.data, page.data_doi)
        + (
            (
                '<dl class="row">'
                + "".join(
                    f'<dt class="col-sm-3">{name}</dt><dd class="col-sm-9">{value}</dd>'
                    for name, value in page.facts
                )
                + "</dl>"
            )
            if page.facts
            else ""
        ),
        "instances": "<div data-dp-sortable data-dp-instance-cards>"
        + table(
            page.columns,
            page.rows,
            numeric=(2, 3),
            empty="",
        )
        + "</div>",
    }


def _datagrid(items: Sequence[tuple[str, str]]) -> str:
    """Term and value pairs as Tabler's datagrid; each term and value is HTML."""
    if not items:
        return ""
    return (
        '<div class="datagrid">'
        + "".join(
            f'<div class="datagrid-item"><div class="datagrid-title">{name}</div>'
            f'<div class="datagrid-content">{value}</div></div>'
            for name, value in items
        )
        + "</div>"
    )


def _instance_page(page: InstancePage) -> dict[str, str]:
    # Each domain is one cell: its summary, and where it has more than one
    # column, the column table folded beneath it.
    bounds = [
        (
            domain.name,
            domain.summary
            + (
                "<details><summary>"
                + copy_span("corpus", "columns_summary")
                + "</summary>"
                + table(INSTANCE_COLUMNS_TABLE, domain.rows, numeric=(0,), wide=False)
                + "</details>"
                if len(domain.rows) > 1
                else ""
            ),
        )
        for domain in page.domains
    ]
    return {
        "facts": _datagrid(page.facts),
        "bounds": _datagrid(bounds),
        "source": _instance_source(page),
    }


def _instance_source(page: InstancePage) -> str:
    """The paper's own variables, coefficient and one claim, each shown verbatim."""
    names = [
        (
            copy_span("corpus", "source_variables"),
            " ".join(f"<code>{escape(name)}</code>" for name in page.variables),
        ),
        (
            copy_span("corpus", "source_coefficient"),
            f"<code>{escape(page.coefficient)}</code>" if page.coefficient else "",
        ),
    ]
    parts = [_datagrid([(term, value) for term, value in names if value])]
    if page.claim is not None:
        quote, page_number, table_name = page.claim
        parts.append("<h3>" + copy_span("corpus", "source_claim") + "</h3>")
        located = [
            (
                copy_span("corpus", "source_page", fields={"page": page_number})
                if page_number
                else ""
            ),
            f"<span data-verbatim>{escape(table_name)}</span>" if table_name else "",
        ]
        parts.append(
            '<figure class="dp-source-claim">'
            # The paper's quote keeps its own wording and capitals; the line
            # standing in for a missing quote is the site's own.
            + (
                f'<blockquote class="blockquote" data-verbatim>{escape(quote)}'
                if quote
                else '<blockquote class="blockquote">'
                + copy_span("corpus", "source_no_quote", fields={"table": table_name})
            )
            + '</blockquote><figcaption class="blockquote-footer">'
            + " · ".join(part for part in located if part)
            + "</figcaption></figure>"
        )
    return "".join(part for part in parts if part)


def _metrics(page: Metrics) -> dict[str, str]:
    rows = [
        (
            anchor(metric.key, f"#{metric.anchor}"),
            ", ".join(
                f'<a href="#metric-argument-{item}">{code(item)}</a>'
                for item in metric.needs
            ),
            escape(metric.docstring.split("\n\n")[0].replace("\n", " ")),
        )
        for metric in page.metrics
    ]
    sections = []
    for metric in page.metrics:
        body = [
            f"## {metric.key} {{#{metric.anchor}}}",
        ]
        if metric.notes:
            body.append(metric.notes)
        elif metric.formula:
            body.append(metric.formula)
        needs = (
            ", ".join(
                f'<a href="#metric-argument-{item}">{code(item)}</a>'
                for item in metric.needs
            )
            or "—"
        )
        body.append(copy_span("task", "required_arguments") + f" {needs}")
        if metric.baselines:
            body.append(
                copy_span("task", "baselines_label")
                + " "
                + ", ".join(
                    f'<a href="{escape(link.url, quote=True)}">{code(link.text)}</a>'
                    for link in metric.baselines
                )
            )
        sections.append("\n\n".join(body))
    arguments = sorted({item for metric in page.metrics for item in metric.needs})
    if page.metrics:
        sections.append(
            "## "
            + copy_span("task", "metric_arguments_heading")
            + " {#metric-arguments}\n\n"
            + (f"{page.arguments_intro}\n\n" if page.arguments_intro else "")
            + table(
                ("Metric argument", "Description", "Used by"),
                [
                    (
                        f'<span id="metric-argument-{item}">{code(item)}</span>',
                        dict(page.metric_argument_descriptions).get(item, ""),
                        ", ".join(
                            anchor(metric.key, f"#{metric.anchor}")
                            for metric in page.metrics
                            if item in metric.needs
                        ),
                    )
                    for item in arguments
                ],
                headings={
                    "Metric argument": copy_span("task", "metric_argument_column"),
                    "Used by": copy_span("task", "used_by_column"),
                },
            )
        )
    return {
        "access": _access(page.access),
        "table": (
            table(
                (
                    "Metric",
                    "Required metric arguments",
                    "Description",
                ),
                rows,
            )
            if rows
            else ""
        ),
        "sections": "\n\n".join(sections),
        "baselines": _baselines(page.baselines),
    }


def _baselines(baselines: Sequence[BaselineEntry]) -> str:
    """The task's baselines table, or nothing where the build has none."""
    if not baselines:
        return ""
    heads = "".join(f'<th scope="col">{escape(name)}</th>' for name in BASELINE_TABLE)
    # The baseline's name identifies its row and carries the anchor every
    # Definition link lands on.
    body = "\n".join(
        "<tr>"
        f'<th scope="row" id="baseline-{escape(row.name, quote=True)}">'
        f"{code(row.name)}</th>"
        f'<td><span class="dp-badge badge">{escape(row.declarer)}</span></td>'
        f"<td>{row.definition}</td>"
        f"<td>{row.applies}</td>"
        f"<td>{escape(row.role)}</td>"
        "</tr>"
        for row in baselines
    )
    return (
        '<table class="dp-table dp-table--wide table table-vcenter table-hover">\n'
        f"<thead><tr>{heads}</tr></thead>\n"
        f"<tbody>\n{body}\n</tbody>\n"
        "</table>\n"
    )


def _swatch_hook(marker: Mapping[str, str]) -> str:
    """A placeholder the client replaces with the series swatch it draws.

    Server-side rendering cannot read the theme or the series rule a browser
    resolves at run time, so the page ships the marker's identity and a small
    script that calls the same ``dpSeriesSwatch`` a chart's legend uses.
    """
    payload = escape(json.dumps(marker), quote=True)
    # The deferred chart runtime that supplies ``dpSeriesSwatch`` loads after
    # this inline script would otherwise run, so the paint waits for it. The
    # Markdown renderer lifts a raw ``<script>`` out of its wrapping paragraph,
    # so the element it paints is found by its own class, not by sibling order
    # -- one page carries exactly one Roster header swatch.
    return (
        f"<span class=\"dp-roster-swatch\" data-dp-roster-marker='{payload}'></span>\n"
        "<script>(function(){"
        "var el=document.querySelector('.dp-roster-swatch');"
        "function paint(){"
        "if(!el)return;"
        "if(!window.dpSeriesInk||!window.dpSeriesSwatch)return;"
        "var m=JSON.parse(el.dataset.dpRosterMarker);"
        "var ink=window.dpSeriesInk(m.core,m['class']);"
        "var emphasis=window.dpSeriesEmphasis();"
        "var swatch=window.dpSeriesSwatch({colour:ink.colour,width:emphasis.regularWidth,"
        "opacity:emphasis.regularOpacity,dash:window.dpSeriesDash(m['class']),marker:m.marker});"
        'swatch.classList.add("dp-roster-swatch-mark");el.replaceWith(swatch);'
        "}"
        "if(document.readyState==='loading'){"
        "document.addEventListener('DOMContentLoaded',paint);"
        "}else{paint();}"
        "})();</script>\n"
    )


def _class_badge(page: RosterPage) -> str:
    """The row's privacy-interpretation badge, linked to the class definition."""
    name = page.algorithm.privacy_interpretation
    shown = interpretation_badge(name, f"dp-badge dp-badge--{name.lower()} badge")
    if not page.class_url:
        return shown
    return f'<a href="{escape(page.class_url, quote=True)}">{shown}</a>'


def _marker(algorithm: ConfiguredAlgorithm) -> dict[str, str]:
    """The core, class and marker shape the swatch script reads."""
    return {
        "core": algorithm.core,
        "class": algorithm.privacy_interpretation,
        "marker": algorithm.marker or "circle",
    }


def _roster_page(page: RosterPage) -> dict[str, str]:
    header = [
        _class_badge(page),
        _swatch_hook(_marker(page.algorithm)),
        note(link(page.task_link.text, page.task_link.url)),
    ]
    standings = [
        (
            row.budget,
            row.rank,
            anchor(row.metric.text, row.metric.url),
            row.headline,
            row.coverage,
        )
        for row in page.standings
    ]
    if standings:
        standing_html = table(
            STANDING_TABLE,
            standings,
            wide=False,
            headings={
                STANDING_TABLE[3]: copy_span(
                    "leaderboard", "column_headline", fields={"percent": page.percent}
                )
            },
        )
    elif page.unranked_note:
        standing_html = f'{_class_badge(page)} {copy_span("roster", "not_ranked")}'
    else:
        standing_html = ""
    outcomes = [
        (
            row.budget,
            row.rank,
            row.success,
            row.no_release,
            row.invalid_regime,
            row.algorithm_error,
            row.timeout,
            row.memory_limit_exceeded,
        )
        for row in page.outcomes
    ]
    return {
        "header": "\n".join(header),
        "description": page.description,
        "standing": standing_html,
        "outcomes": table(OUTCOME_TABLE, outcomes) if outcomes else "",
        "recipe": _recipe(page),
        "provenance": page.provenance,
    }


def _roster_headings(page: RosterPage) -> dict[str, str]:
    """T21's two results headings, each linking the page it summarises."""
    headings = {"standing": f"{link('Leaderboard', page.leaderboard_url)} ranks"}
    if page.explore_url:
        headings["outcomes"] = (
            f"Results breakdown on the public corpus · {link('Explore', page.explore_url)}"
        )
    return headings


def _recipe_entry(component, level: int) -> list[str]:
    parts = ["#" * level + " " + link(component.name.text, component.name.url)]
    if component.values:
        parts.append(
            table(
                RECIPE_TABLE,
                [
                    (code(name), value, description)
                    for name, value, description in component.values
                ],
            )
        )
    for child in component.children:
        parts.extend(_recipe_entry(child, level + 1))
    return parts


def _recipe(page: RosterPage) -> str:
    parts: list[str] = []
    for component in page.recipe:
        parts.extend(_recipe_entry(component, 3))
    parts.append(
        "<details><summary>"
        + copy_span("roster", "exact_configuration")
        + "</summary>\n\n"
        "```json\n" + page.configuration_json + "\n```\n\n</details>"
    )
    return "\n\n".join(parts)


def _chain(row: RosterRow) -> str:
    """The configured algorithm's name above its chain of linked steps, the core last."""
    steps = []
    for position, step in enumerate(row.chain):
        core = position == len(row.chain) - 1
        attributes = (
            ' class="dp-chain-step is-core"' if core else ' class="dp-chain-step"'
        )
        if step.key:
            attributes += f' data-dp-key="{escape(step.key, quote=True)}"'
        if step.component != step.name:
            attributes += (
                f' title="{escape(step.component, quote=True)} '
                f'({escape(step.name, quote=True)})"'
            )
        suffix = (
            f'<span class="dp-hp"> ({escape(step.hyperparameters)})</span>'
            if step.hyperparameters
            else ""
        )
        name = escape(step.name) + suffix
        shown = (
            f'<a href="{escape(step.url, quote=True)}"{attributes}>{name}</a>'
            if step.url
            else f"<span{attributes}>{name}</span>"
        )
        steps.append(f"<li>{shown}</li>")
    label = escape(row.algorithm.label, quote=True)
    return (
        f'<span class="dp-roster-name">{escape(row.algorithm.label)}</span>'
        f'<ul class="dp-chain" aria-label="{label}">{"".join(steps)}</ul>'
    )


def _roster_row(row: RosterRow) -> str:
    """One table row, carrying what the filters and the sort read."""
    algorithm = row.algorithm
    core = row.chain[-1].component
    words = [algorithm.label, algorithm.privacy_interpretation]
    for step in row.chain:
        words += [step.name, step.component, step.hyperparameters]
    marks = {
        "data-dp-label": algorithm.label,
        "data-dp-class": algorithm.privacy_interpretation,
        "data-dp-core": core,
        "data-dp-transformations": " ".join(step.key for step in row.chain if step.key),
        "data-dp-rank": str(algorithm.rank or ""),
        # An infinite headline τ sorts last, as the browser reads "Infinity".
        "data-dp-headline": (
            "" if row.tau is None else f"{row.tau:.6g}".replace("inf", "Infinity")
        ),
        "data-dp-search": " ".join(word for word in words if word).lower(),
    }
    spoken = copy_text("roster", "details_aria", fields={"algorithm": algorithm.label})
    details = (
        f'<a href="{escape(algorithm.url, quote=True)}" '
        f'aria-label="{escape(spoken, quote=True)}">'
        + copy_span("roster", "details_link")
        + "</a>"
    )
    cells = (
        _chain(row),
        interpretation_badge(
            algorithm.privacy_interpretation,
            f"dp-badge dp-badge--{algorithm.privacy_interpretation.lower()} badge",
        ),
        escape(algorithm.standing),
        escape(row.headline) or "—",
        details,
    )
    return (
        "<tr"
        + "".join(
            f' {name}="{escape(value, quote=True)}"' for name, value in marks.items()
        )
        + ">"
        + "".join(
            (
                f'<td class="dp-value">{cell}</td>'
                if position in (2, 3)
                else f"<td>{cell}</td>"
            )
            for position, cell in enumerate(cells)
        )
        + "</tr>"
    )


def _chips(facet: str, values: Sequence[tuple[str, str]]) -> str:
    """One facet's options, toggle chips or a checkbox list by the shared
    rule; the script fills each one's count beside it, outside its name."""

    def attributes(value: str, label: str) -> str:
        return (
            f'data-dp-facet="{facet}" data-dp-value="{escape(value, quote=True)}" '
            f'data-dp-label="{escape(label, quote=True)}"'
        )

    count = '<span class="dp-option-count" data-dp-count></span>'
    if fits_chips([label for _, label in values]):
        return (
            '<div class="dp-chip-group">'
            + "".join(
                '<span class="dp-chip-option">'
                f'<button type="button" class="dp-chip btn btn-sm" aria-pressed="false" '
                f"{attributes(value, label)}>{escape(label)}</button>{count}</span>"
                for value, label in values
            )
            + "</div>"
        )
    return (
        '<div class="dp-check-list">'
        + "".join(
            '<span class="dp-check-option"><label>'
            f'<input type="checkbox" class="form-check-input" {attributes(value, label)}> '
            f"{escape(label)}</label>{count}</span>"
            for value, label in values
        )
        + "</div>"
    )


def _roster_controls(
    page: RosterLanding, rank: Mapping[str, str], headline: Mapping[str, str]
) -> str:
    """Search, sort and the three facets, hidden until the script drives them.

    The table below is complete without them; they only narrow and reorder
    rows a reader already has. Below 768px the facets fold behind one
    disclosure. *rank* and *headline* fill the two reading columns' headings.
    """
    cores = sorted({row.chain[-1].component for row in page.rows})
    core_options = "".join(
        f'<option value="{escape(core, quote=True)}">{escape(core)}</option>'
        for core in cores
    )
    sort_options = "".join(
        f'<option value="{value}">{escape(text)}</option>'
        for value, text in (
            ("rank", copy_text("roster", "column_at_budget", fields=rank)),
            ("headline", copy_text("roster", "column_at_budget", fields=headline)),
            ("name", ROSTER_LANDING_TABLE[0]),
        )
    )
    facets = [
        '<div class="dp-facet-row"><div><label for="dp-roster-core">'
        + copy_span("roster", "core_label")
        + "</label>"
        f'<select id="dp-roster-core" class="{SELECT_CLASS}" '
        'data-dp-roster-core><option value="">'
        + escape(copy_text("roster", "core_any"))
        + f"</option>{core_options}</select>"
        "</div></div>",
        f"<fieldset><legend>{escape(ROSTER_LANDING_TABLE[1])}</legend>"
        + _chips("class", [(name, name) for name in page.classes])
        + "</fieldset>",
    ]
    if page.transformations:
        facets.append(
            "<fieldset><legend>"
            + copy_span("roster", "transformation_label")
            + ' <span class="dp-muted">'
            + copy_span("roster", "transformation_hint")
            + "</span></legend>"
            + _chips("transformation", page.transformations)
            + "</fieldset>"
        )
    active = escape(copy_text("roster", "active_filters"), quote=True)
    return (
        '<div class="dp-roster-controls" data-dp-roster-controls hidden>\n'
        '<div class="dp-facet-row">'
        '<div><label for="dp-roster-q">'
        + copy_span("roster", "search_label")
        + "</label>"
        '<input id="dp-roster-q" type="search" class="dp-input form-control form-control-sm" '
        'autocomplete="off" data-dp-roster-search></div>'
        '<div><label for="dp-roster-sort">'
        + copy_span("roster", "sort_label")
        + "</label>"
        f'<select id="dp-roster-sort" class="{SELECT_CLASS}" '
        f"data-dp-roster-sort>{sort_options}</select></div>"
        "</div>\n"
        '<details class="dp-filters-fold" open data-dp-roster-fold><summary>'
        + copy_span("roster", "filters_summary")
        + "</summary>"
        f'<div class="dp-facets">{"".join(facets)}</div></details>\n'
        f'<div class="dp-active" role="group" aria-label="{active}" '
        "data-dp-roster-active></div>\n"
        '<p class="dp-count" aria-live="polite" data-dp-roster-count></p>\n'
        "</div>\n"
    )


def _roster_landing(page: RosterLanding) -> dict[str, str]:
    """One table of the task's configured algorithms, with its filters.

    Every row is in the markup, in rank order, so a reader without the
    script reads the whole Roster.
    """
    if not page.rows:
        return {"table": ""}
    rank = {"column": ROSTER_LANDING_TABLE[2], "budget": page.budget}
    headline = {
        "column": copy_text(
            "leaderboard", "column_headline", fields={"percent": page.percent}
        ),
        "budget": page.budget,
    }
    heads = (
        f'<th scope="col">{escape(ROSTER_LANDING_TABLE[0])}</th>'
        f'<th scope="col">{escape(ROSTER_LANDING_TABLE[1])}</th>'
        '<th scope="col">'
        + copy_span("roster", "column_at_budget", fields=rank)
        + '</th><th scope="col">'
        + copy_span("roster", "column_at_budget", fields=headline)
        + "</th>"
        f'<th scope="col"><span class="visually-hidden">{ROSTER_LANDING_TABLE[4]}</span></th>'
    )
    body = "\n".join(_roster_row(row) for row in page.rows)
    note = (
        copy_html(
            "roster",
            "ranking_note",
            classes="dp-table-notes",
            fields={
                "metric": page.metric,
                "budget": page.budget,
                "percent": page.percent,
            },
        )
        + "\n"
    )
    parts = [f'<p class="dp-note">{escape(page.notice)}</p>\n'] if page.notice else []
    parts.append(
        '<div class="dp-roster-landing" data-dp-roster-landing'
        + copy_attribute("roster")
        + ">\n"
        + _roster_controls(page, rank, headline)
        + '<div class="dp-table-scroll table-responsive">\n'
        '<table class="dp-table dp-table--wide table table-vcenter table-hover">\n'
        f'<caption class="visually-hidden">{escape(page.title)}</caption>\n'
        f"<thead><tr>{heads}</tr></thead>\n"
        f"<tbody>\n{body}\n</tbody>\n"
        "</table>\n"
        "</div>\n" + note + "</div>\n"
    )
    return {"table": "".join(parts)}


def _methodology(page: Methodology) -> dict[str, str]:
    """The narrative as the release wrote it."""
    return {
        "banner": (
            f'<p class="dp-build-source">{escape(page.banner)}</p>'
            if page.banner
            else ""
        ),
        # Shipped text keeps the release's own wording and case; markdown="1"
        # has MkDocs render it as Markdown inside the wrapper.
        "narrative": (
            f'<div data-verbatim="" markdown="1">\n\n{page.narrative}\n</div>'
            if page.narrative
            else ""
        ),
        "build_source": (
            copy_html(
                "shared",
                "built_from",
                classes="dp-build-source",
                fields={"identity": page.identity},
            )
            if page.identity
            else ""
        ),
    }
