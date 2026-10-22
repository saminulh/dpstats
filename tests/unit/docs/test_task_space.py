"""The task-space pages read the installed library's own declarations."""

from __future__ import annotations

import re
from dataclasses import replace
from inspect import signature
import typing
from typing import Annotated, Any, Literal

import annotated_types
import pytest
from pydantic import Field


from dpstats.algorithm import PrivacyInterpretation
from dpstats.tasks import (
    list_built_in_metric_arguments,
    BUILT_IN_TASKS,
    LinearRegression,
    LogisticRegression,
    MeanEstimation,
    SingleCoefficientEstimation,
    list_built_in_algorithms,
)
from dpstats.tasks import mean_estimation
from dpstats.tasks.linear_regression.algorithms import OLS, AdaSSP
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ApproxBounds,
)
from dpstats.metric import Metric
from dpstats.parameters import FieldSymbol

from dpstats.task import Task

from transformation_groups import load_transformation_groups
from task_reference import exported_transformations
from templates import library_source
from templates.model import Link
from templates.render import render
from docstring_text import page_text
from site_copy import copy_span
from templates.spec import TEMPLATES
from tests.unit.docs.conftest import TunedMean, parse_html

SEMANTICS = Link(
    "What this means",
    "../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation.md",
)
TASK_LINK = Link("Linear Regression", "../index.md")


class _RequiresProbe(AdaSSP):
    """Probe the Requires block a page renders.

    Requires:
        Finite ``design_domain.l2_norm_bound`` and ``response_domain
        .l2_norm_bound``; a non-finite bound is refused with
        ``InvalidRegime``.

        - Accepts an ``ApproxDPBudget`` or a ``ZCDPBudget``, each split into
          equal thirds by basic composition.
        - Refuses any other budget family.
    """


class _ParametrizationNotesProbe(AdaSSP):
    """Probe the Parametrization notes section a page renders.

    Parametrization notes:
        Sourced from the cited paper's calibration procedure.

        - ``failure_probability`` is the paper's own delta-style slack term,
          renamed for this library's convention.
        - Its default and search range come from platform calibration, not
          the paper, which states no recommended value.
    """


class _DelegatingProbe(ApproxBounds):
    """Probe a transformation whose class is read off its own configuration.

    Its ``privacy_interpretation`` reads a configured field, so the class alone
    cannot answer it; the configuration the class declares by default does.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        if self.bounds_share < 0.5:
            return PrivacyInterpretation.THEORETICAL
        return PrivacyInterpretation.STANDARD


class _SymbolProbeMetadata(MeanEstimation.Metadata):
    """Probe the type and the symbol a definition table shows.

    Every annotation here is postponed, exactly as each task's own models
    declare theirs, so the renderers see the source's text until the model's
    own module resolves it.
    """

    dimension: Annotated[int, FieldSymbol("d")] = Field(
        ge=1, description="Probe field."
    )
    tolerance: Annotated[float, FieldSymbol("t")] | None = Field(
        default=None, description="Probe field."
    )
    scale: typing.Annotated[float, FieldSymbol("s")] = Field(
        default=1.0, description="Probe field."
    )
    grid: dict[str, Annotated[int, FieldSymbol("g")]] = Field(
        default_factory=dict, description="Probe field."
    )
    mode: Literal["fast", "exact"] = Field(default="fast", description="Probe field.")


class _DelimitedSymbolMetadata(MeanEstimation.Metadata):
    """Probe a declared symbol that sets its own mathematics."""

    dimension: Annotated[int, FieldSymbol("$d$")] = Field(description="Probe field.")


def _section(page: str, title: str) -> str:
    """One rendered section's body, between its heading and the next one."""
    return (
        re.split(r"## " + re.escape(title) + r"(?: \{#[^}]+\})?\n\n", page)[1]
        .split("\n## ")[0]
        .strip()
    )


def _probe_metric(name: str, formula: str | None) -> Metric:
    """One Metric for the metrics page to read, declared as a task declares it."""

    def calculate(output: Any) -> float:
        """What the probe metric measures."""
        return 0.0

    return Metric(
        task=MeanEstimation, name=name, calculation=calculate, formula=formula
    )


def _algorithms(task: Task) -> dict[str, tuple[Link, str, str]]:
    index = library_source.algorithms_index(task, lambda name: f"{name}/")
    return {row[0].text: row for group in index.groups for row in group.rows}


def _tuned_mean_page() -> Any:
    return library_source.component_page(
        MeanEstimation,
        "TunedMean",
        TunedMean,
        task_link=Link("Mean Estimation", "../index.md"),
        semantics=SEMANTICS,
    )


def test_each_algorithm_row_is_named_by_its_built_in_key() -> None:
    """The name a row links is the key the task's catalogue uses."""
    for task in BUILT_IN_TASKS.values():
        assert set(_algorithms(task)) == set(list_built_in_algorithms(task)), task.id


def test_component_description_preserves_the_full_explanation() -> None:
    class DetailedOLS(OLS):
        """An ordinary least-squares estimator.

        The estimator fits the supplied design matrix and response.
        Each row represents one observation.
        Each column represents one design coordinate.
        The response contains one value per observation.
        The returned vector contains one coefficient per design coordinate.

        Attributes:
            task: The regression task implemented by this estimator.

        Examples:
            >>> algorithm = DetailedOLS()
            >>> algorithm.task == LinearRegression
            True

        Implementation notes:
            The solver uses the inherited implementation.
        """

    page = library_source.component_page(
        LinearRegression,
        "DetailedOLS",
        DetailedOLS,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    description = parse_html(render(TEMPLATES["T9"], page)).text
    assert "An ordinary least-squares estimator." in description
    assert "one coefficient per design coordinate" in description
    assert "The regression task implemented by this estimator." in description
    assert "algorithm = DetailedOLS()" in description
    assert "algorithm.task == LinearRegression" in description
    assert page.implementation_notes == "The solver uses the inherited implementation."


def test_every_registered_algorithm_declares_a_known_class() -> None:
    declared = {
        row[2]
        for task in BUILT_IN_TASKS.values()
        for group in library_source.algorithms_index(task, str).groups
        for row in group.rows
    }
    assert declared
    assert declared <= {member.value for member in PrivacyInterpretation}


def test_a_class_without_the_declaration_is_refused_rather_than_guessed() -> None:
    class Undeclared:
        """A class that declares no privacy semantics at all."""

    with pytest.raises(TypeError, match="privacy_interpretation"):
        library_source.component_page(
            LinearRegression,
            "Undeclared",
            Undeclared,  # type: ignore[arg-type]
            task_link=Link("Linear Regression", "../index.md"),
            semantics=SEMANTICS,
        )


def test_a_delegating_class_is_read_as_the_library_computes_it() -> None:
    """A transformation reading its own fields still has one answer, not none.

    The probe's class depends on a configured field, so the class-level read
    fails and the page reads the property on the default configuration, whose
    share of 0.1 makes it theoretical.
    """
    page = library_source.component_page(
        SingleCoefficientEstimation,
        "_DelegatingProbe",
        _DelegatingProbe,
        task_link=Link("Single Coefficient Estimation", "../index.md"),
        inner_link=Link("Single Coefficient Estimation", "../index.md"),
        semantics=SEMANTICS,
    )
    assert page.privacy_interpretation == "theoretical"
    glance = parse_html(render(TEMPLATES["T9"], page)).find(tag="section")
    assert "⟨component.glance_interpretation⟩ theoretical" in " ".join(
        glance.text.split()
    )


def test_every_exported_transformation_declares_a_class_the_table_can_show() -> None:
    """The Privacy interpretation column is never blank on any task."""
    for task in BUILT_IN_TASKS.values():
        exported = exported_transformations(task.id)
        index = library_source.transformations_index(task, exported, str, str)
        for group in index.groups:
            for row in group.rows:
                assert row.privacy_interpretation, f"{task.id}/{row.name.text}"


def test_a_table_cell_never_shows_a_citation_fragment() -> None:
    """A cell holds the whole short citation or nothing at all."""
    shape = re.compile(r"^[^(]+ \((?:19|20)\d{2}\)$")
    index = library_source.algorithms_index(LinearRegression, str)
    for row in (row for group in index.groups for row in group.rows):
        assert row[3] == "" or shape.match(row[3]), row[3]


def test_the_parameters_list_shows_only_configured_parameters() -> None:
    """The list reads the class's own configured fields, in declared order."""
    names = [parameter.name for parameter in _tuned_mean_page().parameters]
    assert names == ["clip_norm", "noise", "threshold"]


def test_a_declared_bound_folds_into_the_type_in_plain_words() -> None:
    """``float > 0``, never a keyword or a separate constraint; no default means required."""
    parameters = {item.name: item for item in _tuned_mean_page().parameters}
    assert parameters["clip_norm"].type == "float &gt; 0"
    assert parameters["clip_norm"].default == "1.0"
    assert parameters["threshold"].type == "float"
    assert parameters["threshold"].default == ""


@pytest.mark.parametrize(
    ("annotation", "metadata", "shown"),
    [
        (float, [annotated_types.Gt(0.0)], "float &gt; 0"),
        (float, [annotated_types.Ge(0)], "float ≥ 0"),
        (float, [annotated_types.Gt(0.0), annotated_types.Lt(1.0)], "float in (0, 1)"),
        (float, [annotated_types.Ge(0.0), annotated_types.Le(1.0)], "float in [0, 1]"),
        (int, [annotated_types.Gt(0)], "positive int"),
        (int, [annotated_types.Ge(1)], "int ≥ 1"),
        (float, [annotated_types.Le(2.5)], "float ≤ 2.5"),
        (float | None, [annotated_types.Gt(0.0)], "float &gt; 0 or None"),
        (int, [annotated_types.MultipleOf(2)], "int, multiple of 2"),
        (Literal["a", "b", "c"], [], "&#x27;a&#x27;, &#x27;b&#x27; or &#x27;c&#x27;"),
        (bool, [], "bool"),
    ],
)
def test_a_declared_type_reads_with_its_domain(
    annotation: object, metadata: list[object], shown: str
) -> None:
    """Each declared domain reads as the type a reader writes."""
    assert library_source.declared_type(annotation, metadata) == shown


def test_a_requires_block_reads_as_its_own_paragraphs_and_bullets() -> None:
    """A wrapped source line is the writer's wrapping, never a second item."""
    page = library_source.component_page(
        LinearRegression,
        "RequiresProbe",
        _RequiresProbe,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    assert page.requires == (
        "Finite <code>design_domain.l2_norm_bound</code> and "
        "<code>response_domain.l2_norm_bound</code>; a non-finite bound is refused "
        "with <code>InvalidRegime</code>.",
        "Accepts an <code>ApproxDPBudget</code> or a <code>ZCDPBudget</code>, each "
        "split into equal thirds by basic composition.",
        "Refuses any other budget family.",
    )


class _ParameterNoteProbe(AdaSSP):
    """Probe a parameter's own note, read into its Parameters entry.

    Parametrization notes:
        - ``failure_probability``: Corresponds to $ρ$ in the paper, and the
          authors' implementation chose it.
        - ``not_a_parameter``: Names no parameter, so it follows the list.
    """


def test_a_parameter_s_note_renders_in_its_own_entry() -> None:
    """A note naming a parameter sits under it; one naming none follows the list."""
    page = library_source.component_page(
        LinearRegression,
        "ParameterNoteProbe",
        _ParameterNoteProbe,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    (parameter,) = page.parameters
    assert parameter.note.startswith("Corresponds to")
    assert "authors' implementation chose it." in parameter.note
    assert page.parametrization_notes == (
        "- ``not_a_parameter``: Names no parameter, so it follows the list."
    )
    rendered = render(TEMPLATES["T9"], page)
    assert "## Parametrization notes" not in rendered
    parameters = _section(rendered, "Parameters")
    assert parameters.index("dp-parameter-note") < parameters.index("not_a_parameter")


def test_a_block_naming_no_parameter_still_renders_under_parameters() -> None:
    """Free prose in the block is kept whole, after the list, never dropped."""
    page = library_source.component_page(
        LinearRegression,
        "ParametrizationNotesProbe",
        _ParametrizationNotesProbe,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    assert page.parametrization_notes == (
        "Sourced from the cited paper's calibration procedure.\n\n"
        "- ``failure_probability`` is the paper's own delta-style slack "
        "term, renamed for this library's convention.\n"
        "- Its default and search range come from platform calibration, "
        "not the paper, which states no recommended value."
    )
    assert "Sourced from the cited paper" in _section(
        render(TEMPLATES["T9"], page), "Parameters"
    )


def test_the_opening_sentence_naming_the_work_moves_to_source() -> None:
    """The page states the cited work once, in Source; the docstring keeps it."""
    page = library_source.component_page(
        LinearRegression,
        "AdaSSP",
        AdaSSP,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    assert "An implementation of" not in page.description
    assert page.source == (
        "Algorithm 2 from Wang (2018), with Theorem 8 from "
        "Balle and Wang (2018), Theorem 4.1 from Tropp (2012) and Lemma 2.2 "
        "from Bandeira and van Handel (2016)"
    )


def test_a_class_without_a_reference_has_no_source() -> None:
    """No Reference block: no Source line and no References section."""
    page = library_source.component_page(
        LinearRegression,
        "OLS",
        OLS,
        task_link=TASK_LINK,
        semantics=SEMANTICS,
    )
    assert page.source == "" and page.reference == ""
    assert "<dt>Source</dt>" not in render(TEMPLATES["T9"], page)


def test_every_task_declares_an_estimand_a_table_cell_can_hold() -> None:
    """The task description is the declaration; no build may clip prose into it."""
    for task in BUILT_IN_TASKS.values():
        cell = library_source.estimand(task)
        assert "$" not in cell and "\\(" not in cell, task.id
        assert cell, task.id


@pytest.mark.parametrize(
    ("description", "refused"),
    [
        ("Estimate $\\beta$ for the sample.", "math delimiter"),
        ("Estimate \\( \\beta \\) for the sample.", "math delimiter"),
    ],
)
def test_an_unusable_description_stops_the_build(
    description: str, refused: str
) -> None:
    """Unsupported math markup fails instead of rendering as plain text."""
    # Constructing a Task binds the Metadata class to it, so the probe gets a
    # throwaway subclass rather than rebinding a built-in task's own Metadata.
    probe_metadata = type("ProbeMetadata", (MeanEstimation.Metadata,), {})
    declared = Task(
        id="probe",
        name="Probe",
        description=description,
        metadata_cls=probe_metadata,
        input_cls=MeanEstimation.Input,
        output_cls=MeanEstimation.Output,
    )
    with pytest.raises(ValueError, match=refused):
        library_source.estimand(declared)


def test_a_task_description_is_rendered_in_full() -> None:
    description = (
        "The population mean of vectors with a shared dimension. "
        "Each row contains the coordinates of one observation. "
        "The estimate contains one mean per coordinate."
    )
    probe_metadata = type("ProbeMetadata", (MeanEstimation.Metadata,), {})
    declared = Task(
        id="probe",
        name="Probe",
        description=description,
        metadata_cls=probe_metadata,
        input_cls=MeanEstimation.Input,
        output_cls=MeanEstimation.Output,
    )
    assert library_source.estimand(declared) == description


def test_a_literal_parameter_states_its_choices_as_its_type() -> None:
    """A choice set is the field's domain, so it reads as the type."""
    parameters = {item.name: item for item in _tuned_mean_page().parameters}
    assert parameters["noise"].type == "&#x27;gaussian&#x27; or &#x27;laplace&#x27;"


def test_the_usage_snippet_calls_run_as_the_library_declares_it() -> None:
    """The snippet draws an instance and passes run's declared arguments."""
    page = _tuned_mean_page()
    declared = list(signature(TunedMean.run).parameters)[1:]
    called = page.usage.rsplit("algorithm.run(", 1)[1].rstrip(")\n")
    assert [argument.split("=")[0] for argument in called.split(", ")] == declared
    # The snippet binds what it passes: the task's producer supplies the instance.
    assert "from dpstats.simulations.mean_estimation import" in page.usage
    assert "metadata, input, _ = " in page.usage


def test_a_transformation_snippet_composes_rather_than_calling_a_method_it_lacks() -> (
    None
):
    """A Transformation declares prepare and restore, so the snippet composes it."""
    page = library_source.component_page(
        SingleCoefficientEstimation,
        "_DelegatingProbe",
        _DelegatingProbe,
        task_link=Link("Single Coefficient Estimation", "../index.md"),
        inner_link=Link("Single Coefficient Estimation", "../index.md"),
        semantics=SEMANTICS,
    )
    assert "CompositeAlgorithm(" in page.usage
    assert "transformations=(sce.transformations._DelegatingProbe(),)" in page.usage
    assert ".run(" not in page.usage.split("CompositeAlgorithm(")[0]


def test_the_definition_shows_metadata_first_without_constraints_or_summaries() -> None:
    page = render(TEMPLATES["T6"], library_source.definition(MeanEstimation))
    assert "## Input" in page and "## Metadata" in page and "## Output" in page
    assert "Default" not in page
    # Metrics have their own page; the definition never carries their table.
    assert "Needs beyond Output" not in page
    assert page.index("## Metadata") < page.index("## Input") < page.index("## Output")
    assert '<th scope="col">Constraints</th>' not in page
    assert "Carry one mean-estimation" not in page
    assert "Declare the public shape" not in page
    assert '<th scope="col">Field</th>' in page


def test_a_declared_symbol_reads_in_the_constraints_and_not_in_the_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Type column holds the declared type; the symbol is the field's own."""
    # The page's import line names the task package, which must export the model.
    monkeypatch.setattr(
        mean_estimation, "_SymbolProbeMetadata", _SymbolProbeMetadata, raising=False
    )
    probe = Task(
        # The definition reads the task's package for loss families, so the
        # probe borrows a built-in task's id rather than naming a package that
        # no build ships.
        id=MeanEstimation.id,
        name="Probe",
        description="Estimate the probe quantity.",
        metadata_cls=_SymbolProbeMetadata,
        input_cls=MeanEstimation.Input,
        output_cls=MeanEstimation.Output,
    )
    blocks = {block.title: block for block in library_source.definition(probe).models}
    types = {row.name: parse_html(row.type).text for row in blocks["Metadata"].rows}
    constraints = {
        row.name: parse_html(row.constraints).text for row in blocks["Metadata"].rows
    }
    assert types["dimension"] == "int"
    assert constraints["dimension"] == "\\( 1 \\le d \\)"
    # A union, the qualified spelling and a nested generic all resolve.
    assert types["tolerance"] == "float | None (optional)"
    assert types["scale"] == "float (optional)"
    assert types["grid"] == "dict[str, int] (optional)"
    # A resolved annotation is what states a literal field's choice set.
    assert constraints["mode"] == "one of: fast, exact"
    # The aliases a task writes its arrays with stay the names a reader writes.
    assert {row.name: row.type for row in blocks["Input"].rows}["X"] == (
        '<code><autoref identifier="dpstats.task.Array2D">Array2D</autoref></code>'
    )


def test_library_types_link_inside_container_and_union_annotations() -> None:
    from dpstats.domains import Domain

    rendered = library_source.type_html(list[Domain | None])
    assert '<autoref identifier="dpstats.domains.Domain">Domain</autoref>' in rendered
    assert rendered.startswith("<code>list[")
    assert rendered.endswith(" | None]</code>")
    assert library_source.type_html(Literal["Domain"]) == "<code>Literal[Domain]</code>"


def test_a_symbol_setting_its_own_mathematics_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared symbol carries no delimiters, exactly as a formula does not."""
    monkeypatch.setattr(
        mean_estimation,
        "_DelimitedSymbolMetadata",
        _DelimitedSymbolMetadata,
        raising=False,
    )
    probe = Task(
        id=MeanEstimation.id,
        name="Probe",
        description="Estimate the probe quantity.",
        metadata_cls=_DelimitedSymbolMetadata,
        input_cls=MeanEstimation.Input,
        output_cls=MeanEstimation.Output,
    )
    with pytest.raises(ValueError, match="math delimiter"):
        library_source.definition(probe)


def test_the_definition_imports_each_model_from_its_task_package() -> None:
    """Readers import from the task's package, never its private module."""
    for task in BUILT_IN_TASKS.values():
        for block in library_source.definition(task).models:
            line = block.access.import_line
            package = f"from dpstats.tasks.{task.id} import "
            assert line.startswith(package), line


def test_the_metrics_page_keys_every_metric_and_gives_it_a_section() -> None:
    metrics = library_source.metrics(MeanEstimation)
    page = render(TEMPLATES["T18"], metrics)
    assert "from dpstats.tasks import mean_estimation as me" in page
    assert metrics.metrics
    for metric in metrics.metrics:
        assert f"me.metrics.{metric.key}" in page
        assert f"## {metric.key} {{#{metric.anchor}}}" in page
        assert f"## {metric.name}" in page
    assert page.count("⟨task.required_arguments⟩") == len(metrics.metrics)
    assert 'href="#metric-argument-ground_truth"' in page
    assert page.count('id="metric-argument-ground_truth"') == 1


def test_the_metrics_page_states_no_baseline_and_describes_references_from_the_registry() -> (
    None
):
    page = render(TEMPLATES["T18"], library_source.metrics(MeanEstimation))
    registry = list_built_in_metric_arguments(MeanEstimation)
    assert "### Baselines" not in page
    assert "Baseline" not in page
    described = registry["ground_truth"].description
    assert described in page
    references = page.split("{#metric-arguments}")[1]
    assert 'id="metric-argument-ground_truth"' in references
    assert "Population mean." in references


def test_mean_metric_summary_is_plain_and_notes_render_math() -> None:
    page = render(TEMPLATES["T18"], library_source.metrics(MeanEstimation))
    summary_table = page.split("## l2_error_to_ground_truth")[0]
    assert "L2 error between the estimate and population mean." in summary_table
    assert "Computes" not in summary_table
    detail = _section(page, "l2_error_to_ground_truth")
    assert r"Computes $\lVert \hat{\mu} - \mu \rVert_2$" in detail
    assert ":math:" not in detail
    assert "$$" not in detail
    references = page.split("{#metric-arguments}", 1)[1]
    assert "Population mean." in references
    assert "Sample mean." in references
    assert re.search(r"<th\b[^>]*>Description</th>", references)


def test_registry_description_changes_render_without_updating_metric_docstrings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = list_built_in_metric_arguments(MeanEstimation)
    description = "Declared sample-mean reference."
    registry["sample_mean"] = replace(registry["sample_mean"], description=description)
    monkeypatch.setattr(
        library_source, "list_built_in_metric_arguments", lambda task: registry
    )

    page = render(TEMPLATES["T18"], library_source.metrics(MeanEstimation))
    arguments = page.split("{#metric-arguments}", 1)[1]
    assert arguments.count(description) == 1
    assert "Sample mean." not in arguments
    assert "Sample mean vector." not in arguments


def test_every_metric_argument_of_every_built_in_task_is_described() -> None:
    """Every reference a metric requires has a description on its metrics page."""
    for task in BUILT_IN_TASKS.values():
        described = dict(library_source.metrics(task).metric_argument_descriptions)
        for key, metric in library_source.list_built_in_metrics(task).items():
            for item in metric.requires:
                name = getattr(item, "name", str(item))
                assert described.get(
                    name
                ), f"{task.id}.{key}: {name} has no description"


def test_no_release_does_not_claim_baseline_availability() -> None:
    page = render(TEMPLATES["T18"], library_source.metrics(MeanEstimation))
    assert "### Baselines" not in page
    assert "No baseline catalogue supplied" not in page
    assert "Best median" not in page


def test_a_declared_formula_reads_as_display_math_under_its_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metric that declares a formula shows it; one that declares none does not."""
    monkeypatch.setattr(
        library_source,
        "list_built_in_metrics",
        lambda task: {
            "with_formula": _probe_metric(
                "with_formula", r"\lVert \hat{\mu} - \mu \rVert_2"
            ),
            "without_formula": _probe_metric("without_formula", None),
        },
    )
    page = render(TEMPLATES["T18"], library_source.metrics(MeanEstimation))
    assert (
        "$$\n\\lVert \\hat{\\mu} - \\mu \\rVert_2\n$$\n\n"
        + copy_span("task", "required_arguments")
        + " —"
    ) in _section(page, "with_formula")
    assert "$$" not in _section(page, "without_formula")


@pytest.mark.parametrize("formula", ["$\\mu$", "\\( \\mu \\)", "\\[ \\mu \\]"])
def test_a_formula_carrying_its_own_delimiters_stops_the_build(
    monkeypatch: pytest.MonkeyPatch, formula: str
) -> None:
    """The page sets the display math, so the declaration carries none of it."""
    monkeypatch.setattr(
        library_source,
        "list_built_in_metrics",
        lambda task: {"delimited": _probe_metric("delimited", formula)},
    )
    with pytest.raises(ValueError, match="math delimiter"):
        library_source.metrics(MeanEstimation)


def test_a_task_exporting_no_transformations_has_no_transformation_page() -> None:
    assert exported_transformations(LogisticRegression.id) == {}
    assert exported_transformations(LinearRegression.id)


def test_the_task_s_simulated_producers_are_listed() -> None:
    producers = library_source.simulated_producers(MeanEstimation)
    assert {row.name for row in producers} == {
        "simulate_clipped_normal",
        "simulate_clipped_laplacian",
    }


def test_a_source_role_reads_as_code_not_as_its_markup() -> None:
    """A docstring's cross-reference role never reaches a reader verbatim."""
    assert page_text("<p>use :class:<code>ApproxDPBudget</code>.</p>") == (
        "<p>use <code>ApproxDPBudget</code>.</p>"
    )
    assert page_text("<p>see :func:<code>~a.b.measure_results</code></p>") == (
        "<p>see <code>measure_results</code></p>"
    )
    assert page_text("<p>on the instance's ``(X, y)``.</p>") == (
        "<p>on the instance's <code>(X, y)</code>.</p>"
    )


def test_a_verbatim_block_keeps_its_own_backticks() -> None:
    """A code sample is shown as written, whatever markup it contains."""
    block = "<pre><code>text = ``literal``</code></pre>"
    assert page_text(block) == block


def test_a_transformations_group_arrives_in_the_order_its_class_sort_returns() -> None:
    """Strongest class first, then by name, as the class column sorts.

    A diagnostic transformation closes its group, and a click can undo a click.
    """
    exported = exported_transformations(SingleCoefficientEstimation.id)
    index = library_source.transformations_index(
        SingleCoefficientEstimation, exported, str, str
    )
    for group in index.groups:
        shown = [(row.privacy_interpretation, row.name.text) for row in group.rows]
        assert shown == sorted(
            shown,
            key=lambda row: (library_source.CLASS_ORDER.index(row[0]), row[1].lower()),
        ), group.title


def test_every_transformation_lands_in_a_group() -> None:
    """Every exported transformation has an explicit group placement.

    A transformation handing off to another task is a conversion by construction.
    """
    for task in BUILT_IN_TASKS.values():
        exported = exported_transformations(task.id)
        index = library_source.transformations_index(task, exported, str, str)
        listed = [row.name.text for group in index.groups for row in group.rows]
        assert sorted(listed) == sorted(exported), task.id
        preserving = {
            name
            for name, transformation in exported.items()
            if transformation.INNER_TASK.id == task.id
        }
        assert (
            preserving <= library_source.TRANSFORMATION_GROUP_BY_CLASS.keys()
        ), task.id


def test_a_task_table_heads_its_groups_only_when_it_has_named_groups() -> None:
    """Linear Regression reads by family, its one-algorithm mechanisms under
    Other; a page with fewer than two named groups is one unheaded table."""
    families = [
        group.title
        for group in library_source.algorithms_index(LinearRegression, str).groups
    ]
    assert families[-2:] == ["Other", "Nonprivate / diagnostic"]
    for task in BUILT_IN_TASKS.values():
        for index in (
            library_source.algorithms_index(task, str),
            library_source.transformations_index(
                task, exported_transformations(task.id), str, str
            ),
        ):
            titles = [group.title for group in index.groups]
            named = [t for t in titles if t not in ("Other", "Other transformations")]
            if titles != [""] * len(titles):
                assert all(titles) and len(named) > 1, task.id
    mean = library_source.algorithms_index(MeanEstimation, str).groups
    assert [group.title for group in mean] == [""]
