"""The Corpus area's one input: what it reads, and what it may never print."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from bs4 import BeautifulSoup

from dpstats.domains import BoxDomain, EuclideanSpace
from dpstats.instance import (
    Instance,
    identify_instance,
    write_instance_registry,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    LANE_MAP_NAME,
    PROVENANCE_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    result_partition_directory,
    task_collection_directory,
)
from dpstats.workspace.registries import registry_document_path


from bibliography import full_citation, load
from instance_pages import instance_domain, instance_page, number
from site_copy import copy_span, copy_text
from templates import corpus_source
from templates.corpus_source import (
    CorpusInstance,
    licence_href,
    licence_html,
    registry_relative,
)
from templates.render import render
from templates.spec import TEMPLATES
from tests.unit.docs.conftest import (
    DIGEST,
    WriteCopy,
    parse_html,
    release_task,
)
from release_files import (
    sweep_relative,
    write_release_manifest,
)

TASK = "mean_estimation"

# The Instances the fixture release distributes, and the Source each declares.
DISTRIBUTED = {
    "alpha-one": ("benchmark-info/provenance/alpha.json", 4, 1, True),
    "alpha-two": ("benchmark-info/provenance/alpha.json", 40, 2, True),
    "beta-one": ("benchmark-info/provenance/beta.json", 7, 1, False),
}

# ``alpha`` is split across the lanes: it contributes the two Instances above
# and this third one, which the release withholds. Its identifiers exist only
# in bytes no corpus read may open, and its Source bundle is one the release
# does carry, so a naive count would expose it by subtraction.
WITHHELD_LABEL = "alpha-withheld"
WITHHELD_UUID = "i5-WITHHELDWITHHELDWITH"
WITHHELD_REF = "benchmark-info/provenance/alpha.json"

# A Source every one of whose Instances the release withholds. Its bundle ships
# and its declarations do not, so it is listed nowhere at all.
ABSENT_SOURCE_LABEL = "gamma-withheld"
ABSENT_SOURCE_REF = "benchmark-info/provenance/gamma-withheld-only.json"


def _instance(label: str, samples: int, dim: int, *, bounded: bool) -> Instance:
    """One Instance, identified without persisting any Input."""
    domain = (
        BoxDomain(bounds=np.tile(np.array([-10.0, 10.0]), (dim, 1)))
        if bounded
        else EuclideanSpace(dim=dim)
    )
    return identify_instance(
        MeanEstimationMetadata(n_samples=samples, domain=domain),
        MeanEstimationInput(X=np.zeros((samples, dim))),
        {"ground_truth": [0.0] * dim},
        label=label,
    )


def _name_instances(
    root: Path,
    name: str,
    instances: Sequence[Instance],
    sample_unit: str = "individual",
) -> None:
    """Add these Instances to one written study document's ``instances`` section."""
    path = root / BENCHMARK_INFO_DIRECTORY / PROVENANCE_DIRECTORY / f"{name}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["instances"].update(
        {
            instance.uuid: {
                "sample_unit": sample_unit,
                "source": {"variables": ["x"], "coefficient": "x"},
            }
            for instance in instances
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")


def _write_provenance(root: Path, name: str, record: Mapping[str, object]) -> None:
    directory = root / BENCHMARK_INFO_DIRECTORY / PROVENANCE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


PAPER_BIBTEX = """@article{alpha,
  title={A Study of Alpha},
  DOI={10.1000/alpha},
  journal={Journal of Alpha},
  author={Alpha, Ada and Beta, Bo},
  year={2019},
  month=Apr,
  pages={1--34}
}"""
DATA_BIBTEX = """@misc{alpha_data,
  doi = {10.5000/alpha-data},
  url = {https://doi.org/10.5000/alpha-data},
  author = {Alpha, Ada},
  title = {Replication Data for "A Study of Alpha"},
  publisher = {Harvard Dataverse},
  year = {2020}
}"""
PAPER_AND_DATA_BIBTEX = PAPER_BIBTEX + "\n\n" + DATA_BIBTEX

ALPHA = {
    "study_id": "alpha",
    "title": "A Study of Alpha",
    "authors": ["Ada Alpha", "Bo Beta"],
    "year": 2019,
    "venue": "Journal of Alpha",
    "doi": "10.1000/alpha",
    "data_doi": "10.5000/alpha-data",
    "bibtex": PAPER_AND_DATA_BIBTEX,
    "discipline": "political science",
    "data_license": "CC0 1.0",
    "instances": {},
}
BETA = {
    "study_id": "beta",
    "title": "The Beta Replication",
    "authors": ["Cy Gamma"],
    "year": None,
    "venue": "None",
    "doi": None,
    "data_doi": None,
    "bibtex": None,
    "discipline": "health",
    "data_license": "CC BY 4.0",
    "instances": {},
}


CORPUS_FILES = (
    f"{TASK}/instance_registries/benchmark.json",
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/alpha.json",
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/beta.json",
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/gamma-withheld-only.json",
)


def _corpus_task(task: str) -> dict[str, object]:
    return {**release_task(task), "results": {"distribution": {}, "evaluation": {}}}


def _seal_release(root: Path) -> None:
    """Write the manifest declaring every file this fixture release carries.

    The manifest is the release's own disclosure boundary, so the three corpus
    files are declared in it and the withheld-lane bytes below are not.
    """
    files = {sweep_relative(TASK): DIGEST, **{name: DIGEST for name in CORPUS_FILES}}
    write_release_manifest(
        root,
        {TASK: _corpus_task(TASK)},
        files,
        version="simulated-v1",
    )


@pytest.fixture
def corpus_release(tmp_path: Path) -> Path:
    """One release carrying a Source split across the distribution boundary.

    ``alpha`` contributes two Instances to this release and one more that the
    release withholds: its identifiers exist only in the bytes below that no
    corpus read is allowed to open.
    """
    root = tmp_path / "release"
    root.mkdir(parents=True)
    instances = [
        _instance(label, samples, dim, bounded=bounded)
        for label, (_, samples, dim, bounded) in DISTRIBUTED.items()
    ]
    write_instance_registry(
        registry_document_path(
            task_collection_directory(root, TASK),
            "instance",
            "benchmark",
        ),
        TASK,
        instances,
    )
    _write_provenance(root, "alpha", ALPHA)
    _write_provenance(root, "beta", BETA)
    for name in ("alpha", "beta"):
        _name_instances(
            root,
            name,
            [
                instance
                for instance in instances
                if DISTRIBUTED[str(instance.label)][0].endswith(f"/{name}.json")
            ],
        )
    # Stored rows of the withheld lane, in the two shapes a store carries them.
    # Nothing the Corpus area reads may open either of them.
    (root / LANE_MAP_NAME).write_text(
        f"uuid,provenance_ref,lane\n{WITHHELD_UUID},{WITHHELD_REF},evaluation\n",
        encoding="utf-8",
    )
    results = result_partition_directory(
        task_collection_directory(root, TASK), "a5-FIXTUREALGORITHM"
    )
    results.mkdir(parents=True)
    (results / "part-0.parquet").write_bytes(
        json.dumps(
            {"instance": WITHHELD_UUID, "label": WITHHELD_LABEL, "ref": WITHHELD_REF}
        ).encode()
    )
    _write_provenance(
        root,
        "gamma-withheld-only",
        {"study_id": ABSENT_SOURCE_LABEL, "data_license": "CC0 1.0", "instances": {}},
    )
    _seal_release(root)
    return root


def _read(root: Path) -> corpus_source.Corpus:
    return corpus_source.read_corpus(
        root, identity="Distribution Release simulated-v1", published=False
    )


def test_the_reader_lists_every_source_naming_a_distributed_instance(
    corpus_release: Path,
) -> None:
    """The reader lists every source whose document names a distributed instance."""
    corpus = _read(corpus_release)
    assert [source.slug for source in corpus.sources] == ["alpha", "beta"]
    assert corpus.version == "simulated-v1"
    assert corpus.tasks == (TASK,)


def test_the_reader_opens_only_the_corpus_files_the_release_declares(
    corpus_release: Path,
) -> None:
    """No Input, Result, view, lane map or recipe is on the corpus read path."""
    corpus = _read(corpus_release)
    assert sorted(corpus.files_read) == [
        "benchmark-info/provenance/alpha.json",
        "benchmark-info/provenance/beta.json",
        "benchmark-info/provenance/gamma-withheld-only.json",
        f"{TASK}/instance_registries/benchmark.json",
        "release.json",
    ]


def test_a_document_naming_no_distributed_instance_is_not_a_source(
    corpus_release: Path,
) -> None:
    """A Source with no distributed Instance would say the release withholds it."""
    corpus = _read(corpus_release)
    assert ABSENT_SOURCE_REF not in {
        f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/{s.slug}.json"
        for s in corpus.sources
    }
    assert any("gamma-withheld-only" in reason for reason in corpus.skipped)


def test_one_instance_named_by_two_documents_is_refused(corpus_release: Path) -> None:
    """Two documents naming one instance would let one's facts replace the other's."""
    alpha = json.loads(
        (corpus_release / "benchmark-info/provenance/alpha.json").read_text()
    )
    beta_path = corpus_release / "benchmark-info/provenance/beta.json"
    beta = json.loads(beta_path.read_text())
    beta["instances"].update(alpha["instances"])
    beta_path.write_text(json.dumps(beta))
    with pytest.raises(ValueError, match="named by both"):
        _read(corpus_release)


def test_a_source_split_across_lanes_counts_only_what_the_release_carries(
    corpus_release: Path,
) -> None:
    """The rule: every count counts released rows, and no complete count exists.

    ``alpha`` contributes three Instances and this release distributes two of
    them. The third is named only in the lane map and the Result table, which
    the corpus read path never opens, so the pages carry two and nothing a
    reader could subtract the two from.
    """
    corpus = _read(corpus_release)
    alpha = next(source for source in corpus.sources if source.slug == "alpha")
    assert len(alpha.instances) == 2
    assert [instance.n for instance in alpha.instances] == [4, 40]
    pages = _pages(corpus)
    assert pages["corpus/alpha/"].count(f"../../{TASK}/") == 2
    assert _index_rows(pages["corpus/"])["A Study of Alpha"]["Instances"] == "2"
    for page in pages.values():
        for word in ("withheld", "excluded", "total", "coverage", "%", " of 3"):
            assert word not in page.lower(), word


# ── The pages the release's records render ──────────────────────────────────

TASK_NAMES = {TASK: "Mean Estimation"}

# The Corpus labels written as they read before they became slots, so a
# test can read the counts and names that fill them.
CORPUS_LABELS = {
    "corpus": {
        "search_label": "Search",
        "task_filter": "Task",
        "all_tasks": "All tasks",
        "all_fields": "All fields",
        "clear_filters": "Clear filters",
        "summary_counts": "{sources} and {instances}.",
        "source_count_one": "{count} source",
        "source_count_many": "{count} sources",
        "instance_count_one": "{count} problem instance",
        "instance_count_many": "{count} problem instances",
        "summary_task": "{task}: {instances}.",
        "task_instance_count_one": "{count} instance",
        "task_instance_count_many": "{count} instances",
        "copy_citation": "Copy citation",
        "fact_study": "Study",
        "fact_data_licence": "Data licence",
        "fact_sample_unit": "Sample unit",
        "fact_samples": "Samples (n)",
        "fact_dimension": "Dimension",
        "fact_samples_per_dimension": "Samples per dimension",
        "fact_coefficient": "Coefficient of interest",
        "coefficient_column": "Column {column}",
        "fact_weight": "Weight in the success rate",
        "domain_covariates": "Covariates (X)",
        "domain_response": "Response (y)",
        "domain_data": "Data",
        "columns_summary": "Columns",
        "column_intercept": "Intercept",
        "column_range": "Range",
        "column_indicator_group": "Indicator group",
        "indicator_zero_or_one": "0 or 1",
        "indicator_at_most_one": "at most one is 1",
        "indicator_exactly_one": "exactly one is 1",
        "coefficient_mark": "{kind} · coefficient of interest",
        "summary_intercept": "intercept",
        "summary_groups_one": "{count} indicator group",
        "summary_groups_many": "{count} indicator groups",
        "summary_columns_one": "{count} column",
        "summary_columns_many": "{count} columns",
        "summary_ranges_one": "{count} range column",
        "summary_ranges_many": "{count} range columns",
    },
    "site": {
        "fact_field": "Field",
        "fact_condition_number": "Design condition number κ(X)",
    },
}


def _index_rows(page: str) -> dict[str, dict[str, str]]:
    """The corpus index table as a reader sees it: each Source's cells by column."""
    table = parse_html(page).find(tag="table")
    columns = [header.text for header in table.find(tag="thead").find_all(tag="th")]
    rows = table.body_rows()
    values = [
        dict(zip(columns, (cell.text for cell in row.find_all(tag="td"))))
        for row in rows
    ]
    return {row.find(tag="a").text: values[index] for index, row in enumerate(rows)}


def _pages(corpus: corpus_source.Corpus) -> dict[str, str]:
    """Every page this build would write from one corpus, keyed by its route."""
    pages = {
        "corpus/": render(
            TEMPLATES["T2"], corpus_source.corpus_index(corpus, TASK_NAMES)
        )
    }
    for source in corpus.sources:
        pages[f"corpus/{source.slug}/"] = render(
            TEMPLATES["T20"], corpus_source.source_page(corpus, source)
        )
        for instance in source.instances:
            record = instance_page(
                corpus,
                source,
                instance,
                condition=None,
            )
            pages[corpus_source.instance_route(instance)] = render(
                TEMPLATES["T24"], record
            )
    return pages


def test_the_corpus_index_lists_every_source_and_links_to_its_page(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """The corpus index lists every source and links to its page."""
    write_copy(CORPUS_LABELS)
    page = _pages(_read(corpus_release))["corpus/"]
    links = {
        link.text: link.attrs["href"] for link in parse_html(page).find_all(tag="a")
    }
    assert links["A Study of Alpha"] == "alpha/"
    assert links["The Beta Replication"] == "beta/"
    assert "2 sources and 3 problem instances" in page


def test_the_corpus_index_carries_its_filters_over_a_complete_table(
    corpus_release: Path,
) -> None:
    """Search, task and field narrow a table that is whole without them."""
    page = parse_html(_pages(_read(corpus_release))["corpus/"])
    assert len(page.find(tag="table").body_rows()) == 2
    assert page.find_all("data-dp-corpus-search")
    tasks = page.find("data-dp-corpus-task").find_all(tag="option")
    assert "mean_estimation" in {option.attrs["value"] for option in tasks}
    fields = page.find("data-dp-corpus-field").find_all(tag="option")
    assert {"health", "political science"} <= {
        option.attrs["value"] for option in fields
    }


def test_a_source_page_cites_its_paper_then_its_data(corpus_release: Path) -> None:
    """A source page cites its paper then its data."""
    html = _pages(_read(corpus_release))["corpus/alpha/"]
    paper = html.index(
        "Alpha, A., &amp; Beta, B. (2019). <span data-verbatim>A Study of Alpha</span>."
    )
    data = html.index("Replication Data for")
    assert paper < data
    assert "<em>Journal of Alpha</em>" in html
    assert 'href="https://doi.org/10.5000/alpha-data"' in html
    assert len(parse_html(html).find_all("data-dp-copy")) == 2


def test_a_source_page_with_only_a_paper_entry_cites_the_paper(
    corpus_release: Path,
) -> None:
    """A source page with only a paper entry cites the paper."""
    corpus = _read(corpus_release)
    alpha = next(source for source in corpus.sources if source.slug == "alpha")
    page = corpus_source.source_page(corpus, replace(alpha, bibtex=PAPER_BIBTEX))
    assert "<span data-verbatim>A Study of Alpha</span>." in page.paper
    assert page.data == ""


def test_a_source_page_omits_the_citation_the_release_does_not_carry(
    corpus_release: Path,
) -> None:
    """A source page states that its release carries no formatted citation."""
    pages = _pages(_read(corpus_release))
    assert not parse_html(pages["corpus/beta/"]).find_all("data-dp-copy")
    assert "data-dp-citation" not in pages["corpus/beta/"]
    assert copy_text("corpus", "missing_citation") in pages["corpus/beta/"]
    # A "None" string is absent: beta declares no venue, year, DOI or BibTeX.
    assert "None" not in pages["corpus/beta/"]


def test_the_paper_and_its_data_each_take_a_section_with_their_own_copy(
    corpus_release: Path,
) -> None:
    """Paper, then Data: each holds its own citation and the button copying it."""
    source = _pages(_read(corpus_release))["corpus/alpha/"]
    assert 0 < source.index("\n## Paper\n") < source.index("\n## Data\n")
    page = BeautifulSoup(source, "html.parser")
    references = page.select("[data-dp-reference]")
    assert len(references) == 2
    assert all(
        len(reference.select("[data-dp-citation]")) == 1
        and len(reference.select("[data-dp-copy]")) == 1
        for reference in references
    )
    assert "A Study of Alpha." in references[0].get_text()
    assert "Replication Data for" in references[1].get_text()


def test_a_source_without_a_paper_doi_shows_only_its_data_entry(
    corpus_release: Path,
) -> None:
    """A source without a paper doi shows only its data entry."""
    corpus = _read(corpus_release)
    alpha = next(source for source in corpus.sources if source.slug == "alpha")
    data_only = replace(
        alpha, doi="", bibtex=PAPER_BIBTEX.replace("{alpha,", "{alpha_data,")
    )
    page = corpus_source.source_page(corpus, data_only)
    assert page.paper == ""
    assert page.data == full_citation("alpha_data", load(data_only.bibtex))


def test_a_source_whose_bibtex_lacks_its_own_keys_fails_the_build(
    corpus_release: Path,
) -> None:
    """A source whose bibtex lacks its own keys fails the build."""
    corpus = _read(corpus_release)
    alpha = next(source for source in corpus.sources if source.slug == "alpha")
    wrong = replace(alpha, bibtex=PAPER_BIBTEX.replace("{alpha,", "{other,"))
    with pytest.raises(KeyError, match="alpha"):
        corpus_source.source_page(corpus, wrong)


def test_a_source_page_links_each_instance_to_its_own_page(
    corpus_release: Path,
) -> None:
    """A source page links each instance to its own page."""
    corpus = _read(corpus_release)
    pages = _pages(corpus)
    alpha = next(source for source in corpus.sources if source.slug == "alpha")
    for instance in alpha.instances:
        route = corpus_source.instance_route(instance)
        assert f'href="../../{route}"' in pages["corpus/alpha/"]
        assert route in pages
    assert "A Study of Alpha" in pages["corpus/alpha/"]
    assert "CC0 1.0" in pages["corpus/alpha/"]


def test_an_instance_page_shows_its_declared_facts_and_bounds_only(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """The page carries the declaration and the Metadata, and no stored argument."""
    write_copy(CORPUS_LABELS)
    corpus = _read(corpus_release)
    instance = next(
        item
        for source in corpus.sources
        for item in source.instances
        if item.label == "alpha-two"
    )
    page = BeautifulSoup(
        _pages(corpus)[corpus_source.instance_route(instance)], "html.parser"
    )
    facts = {
        item.select_one(".datagrid-title").get_text(): item.select_one(
            ".datagrid-content"
        )
        for item in page.select(".datagrid-item")
    }
    facts = {name: value.get_text() for name, value in facts.items()}
    assert facts["Study"] == "A Study of Alpha"
    assert facts["Data licence"] == "CC0 1.0"
    assert facts["Samples (n)"] == "40"
    assert facts["Dimension"] == "2"
    assert facts["Samples per dimension"] == "20"
    assert page.find("a", string="A Study of Alpha")["href"] == "../../../corpus/alpha/"
    assert facts["Data"].startswith("2 range columns")
    assert [cell.get_text() for cell in page.select("details td")] == [
        "1",
        "Range",
        "[−10, 10]",
        "2",
        "Range",
        "[−10, 10]",
    ]
    assert not any(
        table.find("th", string="Description") for table in page.find_all("table")
    )
    assert "ground_truth" not in page.get_text()


def test_an_unbounded_column_reads_as_the_whole_line(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """An unbounded domain is a range open at both ends, not a class name."""
    write_copy(CORPUS_LABELS)
    corpus = _read(corpus_release)
    instance = next(
        item
        for source in corpus.sources
        for item in source.instances
        if item.label == "beta-one"
    )
    page = _pages(corpus)[corpus_source.instance_route(instance)]
    assert (
        '<div class="datagrid-title">'
        '<span class="dp-authored-copy" data-copy-slot="corpus.domain_data">Data</span>'
        '</div><div class="datagrid-content">(−∞, ∞)</div>'
    ) in page
    assert "EuclideanSpace" not in page


def test_the_coefficient_column_reads_its_kind_as_text(write_copy: WriteCopy) -> None:
    """The marked row quotes its kind's words, never the kind's markup."""
    write_copy(CORPUS_LABELS)
    domain = instance_domain(
        "X", BoxDomain(bounds=((0.0, 1.0), (-1.0, 2.0))), coefficient=0
    )
    marked = BeautifulSoup(domain.rows[0][1], "html.parser")
    assert marked.get_text() == "Range · coefficient of interest"
    assert BeautifulSoup(domain.rows[1][1], "html.parser").get_text() == "Range"


def test_a_slot_refuses_rendered_slot_html_as_a_field(write_copy: WriteCopy) -> None:
    """A field is text; a rendered slot passed as one would show its tags."""
    write_copy(CORPUS_LABELS)
    with pytest.raises(ValueError, match="rendered slot HTML"):
        copy_span(
            "corpus",
            "coefficient_mark",
            fields={"kind": copy_span("corpus", "column_range")},
        )


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (0.0791528, "0.07915"),
        (12345.6, "12,350"),
        (20.0, "20"),
        (-3.5, "−3.5"),
        (1234567.0, "1.235 × 10⁶"),
        (-0.000012346, "−1.235 × 10⁻⁵"),
        (float("inf"), "∞"),
    ],
)
def test_a_reference_value_shows_four_significant_figures(
    value: float, shown: str
) -> None:
    """Four figures, plain between 10⁻⁴ and 10⁶, a power of ten outside them."""
    assert number(value) == shown


def test_no_withheld_identifier_reaches_any_page_this_build_writes(
    corpus_release: Path,
) -> None:
    """The disclosure boundary, read off the pages rather than the records."""
    pages = _pages(_read(corpus_release))
    for route, page in pages.items():
        assert WITHHELD_UUID not in page, route
        assert WITHHELD_LABEL not in page, route
        assert ABSENT_SOURCE_LABEL not in page, route
        assert ABSENT_SOURCE_REF not in page, route
    # The routes a build writes carry the same boundary as the bytes it writes.
    assert not any(
        WITHHELD_UUID in route or ABSENT_SOURCE_LABEL in route for route in pages
    )


def test_a_build_with_no_release_says_so_rather_than_listing_nothing() -> None:
    """A build with no release says so rather than listing nothing."""
    page = render(TEMPLATES["T2"], corpus_source.corpus_index(None, TASK_NAMES))
    assert "⟨corpus.no_release⟩" in page and "⟨corpus.no_sources⟩" in page
    assert not parse_html(page).find_all(tag="table")


def test_the_release_keys_no_ruled_section_renders_are_reported_not_shown(
    corpus_release: Path,
) -> None:
    """A provenance value with no ruled place is reported and never shown."""
    corpus = _read(corpus_release)
    assert corpus.unrendered_keys == ()
    _write_provenance(corpus_release, "alpha", {**ALPHA, "internal_note": "secret"})
    _name_instances(
        corpus_release,
        "alpha",
        [
            instance
            for source in corpus.sources
            for instance in source.instances
            if source.slug == "alpha"
        ],
    )
    reread = _read(corpus_release)
    assert reread.unrendered_keys == ("internal_note",)
    for page in _pages(reread).values():
        assert "secret" not in page


def test_a_registry_entry_no_document_names_leaves_a_reason_and_no_count(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """A row this build cannot place is reported, never quietly absorbed."""
    write_copy(CORPUS_LABELS)
    path = corpus_release / "benchmark-info/provenance/beta.json"
    beta = json.loads(path.read_text())
    (dropped,) = beta["instances"]
    beta["instances"] = {}
    path.write_text(json.dumps(beta))
    corpus = _read(corpus_release)
    assert any(dropped in reason for reason in corpus.skipped)
    summary = _pages(corpus)["corpus/"]
    assert "2 problem instances" in summary
    assert dropped not in summary


def test_an_instance_carries_its_verbatim_source_block(corpus_release: Path) -> None:
    """The document's source block reaches the instance row unchanged."""
    instance = next(
        item for source in _read(corpus_release).sources for item in source.instances
    )
    assert dict(instance.source_facts) == {"variables": ["x"], "coefficient": "x"}
    assert instance.sample_unit == "individual"


SOURCE_LABELS = {
    **CORPUS_LABELS,
    "corpus": {
        **CORPUS_LABELS["corpus"],
        "source_heading": "Source in the paper",
        "source_variables": "Variables",
        "source_coefficient": "Coefficient",
        "source_claim": "Quote",
        "source_page": "Page {page}",
        "source_no_quote": "N/A, printed in {table}",
    },
}


def _source_page(corpus_release: Path, source: Mapping[str, object]) -> BeautifulSoup:
    """alpha-two's instance page, its study document carrying *source*."""
    instance = next(
        item
        for study in _read(corpus_release).sources
        for item in study.instances
        if item.label == "alpha-two"
    )
    path = (
        corpus_release / BENCHMARK_INFO_DIRECTORY / PROVENANCE_DIRECTORY / "alpha.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["instances"][instance.uuid]["source"] = source
    path.write_text(json.dumps(record), encoding="utf-8")
    corpus = _read(corpus_release)
    return BeautifulSoup(
        _pages(corpus)[corpus_source.instance_route(instance)], "html.parser"
    )


def _claims(page: BeautifulSoup) -> list[tuple[str, str]]:
    return [
        (
            claim.select_one("blockquote").get_text(),
            claim.select_one("figcaption").get_text(),
        )
        for claim in page.select(".dp-source-claim")
    ]


def test_an_instance_page_quotes_its_source_paper_verbatim(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """Variables, coefficient and the most prominent quoted claim show as the
    paper wrote them; a claim without a quote is passed over for one with."""
    write_copy(SOURCE_LABELS)
    page = _source_page(
        corpus_release,
        {
            "variables": ["lnwage", "educ"],
            "coefficient": "educ",
            "claims": [
                {
                    "quote": None,
                    "page": "33",
                    "table": "Table A.6",
                    "prominence": "headline",
                    "quote_absence": {"reason": "curation note"},
                },
                {
                    "quote": "A secondary remark.",
                    "page": "40",
                    "table": "Table 5",
                    "prominence": "secondary",
                },
                {
                    "quote": "Returns <b>rise</b> with schooling.",
                    "page": "18",
                    "table": "Table 3",
                    "prominence": "primary",
                },
                {
                    "quote": "A later primary remark.",
                    "page": "19",
                    "table": "Table 4",
                    "prominence": "primary",
                },
            ],
        },
    )
    assert "Source in the paper" in page.get_text()
    assert [code.get_text() for code in page.select(".datagrid-content code")] == [
        "lnwage",
        "educ",
        "educ",
    ]
    assert _claims(page) == [
        ("Returns <b>rise</b> with schooling.", "Page 18 · Table 3")
    ]
    # The paper's own words keep their capitals past the glossary check.
    assert page.select_one(".dp-source-claim blockquote").has_attr("data-verbatim")
    assert "curation note" not in page.get_text()


def test_an_instance_whose_claims_have_no_quote_names_the_table(
    corpus_release: Path, write_copy: WriteCopy
) -> None:
    """With no quoted claim, the first claim shows its table, never its reason."""
    write_copy(SOURCE_LABELS)
    page = _source_page(
        corpus_release,
        {
            "claims": [
                {
                    "quote": None,
                    "page": "33",
                    "table": "Table A.6",
                    "prominence": "primary",
                    "quote_absence": {"reason": "curation note"},
                },
                {
                    "quote": None,
                    "page": "34",
                    "table": "Table A.7",
                    "prominence": "headline",
                },
            ],
        },
    )
    assert _claims(page) == [("N/A, printed in Table A.6", "Page 33 · Table A.6")]
    assert not page.select(".datagrid-content code")
    assert "curation note" not in page.get_text()


def test_a_task_whose_corpus_files_the_release_omits_leaves_a_reason(
    corpus_release: Path,
) -> None:
    """A release may publish a task's results and declare no Instances for it."""
    files = {sweep_relative("median_estimation"): DIGEST}
    files.update({name: DIGEST for name in CORPUS_FILES})
    files[sweep_relative(TASK)] = DIGEST
    (corpus_release / RELEASE_MANIFEST_NAME).unlink()
    write_release_manifest(
        corpus_release,
        {
            TASK: _corpus_task(TASK),
            "median_estimation": _corpus_task("median_estimation"),
        },
        files,
        version="simulated-v1",
    )
    corpus = _read(corpus_release)
    assert any(
        reason.startswith("median_estimation: the release declares no")
        for reason in corpus.skipped
    )
    assert [source.slug for source in corpus.sources] == ["alpha", "beta"]


def test_no_page_carries_a_release_banner_or_a_store_path(
    corpus_release: Path,
) -> None:
    """No Corpus page names the release it reads, or the directory it read.

    The directory this build happened to read the release from is not a fact
    about the release, and a preview carries no "Preview · " remnant.
    """
    corpus = _read(corpus_release)
    pages = _pages(corpus)
    assert len(pages) > len(corpus.sources)
    for route, page in pages.items():
        assert "preview_badge" not in page and " · ⟨" not in page, route
        assert str(corpus_release) not in page, route
        assert "/private/" not in page and "/Users/" not in page, route


def test_an_absent_value_is_an_empty_cell_not_a_substitute_glyph(
    corpus_release: Path,
) -> None:
    """`beta` declares no year, so its Year cell carries nothing at all."""
    pages = _pages(_read(corpus_release))
    for route, page in pages.items():
        assert "—" not in page, route
    assert _index_rows(pages["corpus/"])["The Beta Replication"]["Year"] == ""


def _corrupt_one_instance(root: Path) -> str:
    """Break one filed Instance record so this library cannot rebuild it."""
    path = registry_document_path(
        task_collection_directory(root, TASK), "instance", "benchmark"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    broken = sorted(document["entries"])[0]
    document["entries"][broken] = {"input_digest": "not-a-digest"}
    path.write_text(json.dumps(document), encoding="utf-8")
    return broken


def test_a_local_cut_lists_an_unrebuildable_instance_and_keeps_its_siblings(
    corpus_release: Path,
) -> None:
    """A local cut lists an unrebuildable instance and keeps its siblings."""
    broken = _corrupt_one_instance(corpus_release)

    corpus = _read(corpus_release)

    assert any(
        broken in reason and "cannot be rebuilt by this library" in reason
        for reason in corpus.skipped
    )
    listed = {row.uuid for source in corpus.sources for row in source.instances}
    assert broken not in listed
    assert len(listed) == len(DISTRIBUTED) - 1


def test_a_published_release_refuses_an_unrebuildable_instance(
    corpus_release: Path,
) -> None:
    """A published release refuses an unrebuildable instance."""
    broken = _corrupt_one_instance(corpus_release)

    with pytest.raises(ValueError, match=broken):
        corpus_source.read_corpus(
            corpus_release, identity="Distribution Release simulated-v1", published=True
        )


def test_the_task_filter_data_is_the_chosen_tasks_own_counts_and_ranges(
    corpus_release: Path,
) -> None:
    """A Source spanning two tasks carries separate counts and ranges per task."""
    regression = "linear_regression"
    instances = [
        identify_instance(
            LinearRegressionMetadata(
                n_samples=samples,
                design_domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
                response_domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
            ),
            LinearRegressionInput(X=np.zeros((samples, 1)), y=np.zeros(samples)),
            {},
            label=f"lr-{samples}",
        )
        for samples in (100, 1800, 2300)
    ]
    write_instance_registry(
        registry_document_path(
            task_collection_directory(corpus_release, regression),
            "instance",
            "benchmark",
        ),
        regression,
        instances,
    )
    _name_instances(corpus_release, "alpha", instances, sample_unit="x")
    files = {sweep_relative(TASK): DIGEST, sweep_relative(regression): DIGEST}
    files.update({name: DIGEST for name in CORPUS_FILES})
    files.update({registry_relative(regression): DIGEST})
    (corpus_release / RELEASE_MANIFEST_NAME).unlink()
    write_release_manifest(
        corpus_release,
        {TASK: _corpus_task(TASK), regression: _corpus_task(regression)},
        files,
        version="simulated-v1",
    )
    page = parse_html(
        render(
            TEMPLATES["T2"],
            corpus_source.corpus_index(
                _read(corpus_release),
                {TASK: "Mean Estimation", regression: "Linear Regression"},
            ),
        )
    )
    alpha = next(
        row
        for row in page.find(tag="table").body_rows()
        if "Study of Alpha" in row.text
    )
    assert alpha.attrs["data-instances-mean_estimation"] == "2"
    assert alpha.attrs["data-samples-mean_estimation"] == "4–40"
    assert alpha.attrs["data-instances-linear_regression"] == "3"
    assert alpha.attrs["data-samples-linear_regression"] == "100–2,300"
    beta = next(row for row in page.find(tag="table").body_rows() if "Beta" in row.text)
    assert not any(name.endswith("linear_regression") for name in beta.attrs)


def test_a_licence_links_its_shipped_text_else_its_spdx_page() -> None:
    shipped = frozenset({"MIT"})
    assert licence_href("MIT", shipped) == "licenses/MIT.txt"
    assert licence_href("CC-BY-4.0", shipped) == (
        "https://spdx.org/licenses/CC-BY-4.0.html"
    )
    # An id the SPDX list does not name has no page to link.
    assert licence_href("LicenseRef-GTEx-open-access-terms", shipped) == ""
    assert licence_href("", shipped) == ""


def test_a_licence_fact_links_its_name_and_carries_its_copyright_line() -> None:
    instance = CorpusInstance(
        uuid="a",
        task="t",
        label="a",
        n=1,
        dimension="1",
        bounded=False,
        declared=(),
        data_license="MIT",
        sample_unit="",
        field="",
        source="s",
        data_license_href="licenses/MIT.txt",
        data_copyright="Copyright (c) 2023 A & B",
    )
    page = parse_html(licence_html(instance, "../../"))
    (link,) = page.find_all(tag="a")
    assert link.attrs["href"] == "../../licenses/MIT.txt" and link.text == "MIT"
    assert page.find(tag="span").text == "Copyright (c) 2023 A & B"
    plain = replace(instance, data_license_href="", data_copyright="")
    assert licence_html(plain, "../../") == "MIT"
