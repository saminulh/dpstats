"""Read one Distribution Release's public corpus facts, and nothing else.

The Corpus area has one input: the Distribution Release tree the build is
given. This module opens exactly three kinds of file inside it — the per-study
``benchmark-info/provenance/<study>.json`` document, whose ``instances``
section carries each distributed instance's facts (read through the library's
one provenance reader), each task's ``benchmark`` instance registry and each
task's ``instance-weights.csv`` — and only where the release's own manifest
declares them. No Input artifact, Result table, lane map, view or recipe is on
this path, so a build cannot read what the release withholds and cannot render
what it did not seal.

**Distributed-only counting.** Every number the Corpus area prints counts rows
this release carries: the instances its own provenance and its own
``benchmark`` registry both name, and the sources those instances name. The
reader opens no lane map and no working store, so it never holds a complete
count to subtract one of these from, and the pages carry no denominator,
share, percentage, coverage or withheld figure, so no two published numbers
can be differenced to expose an instance the release withholds. A source no
instance in this release names is not listed at all, and a source some of
whose instances the release withholds appears with exactly its distributed
ones and nothing that implies the rest.
"""

from __future__ import annotations

import csv
import json
import math
from html import escape
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpstats.domains import Domain
from dpstats.instance import (
    Instance,
    load_instances_partitioned,
    read_instance_registry,
)
from dpstats.transport.registry import read_registry
from dpstats.task import TaskMetadata
from bibliography import full_citation, load
from site_copy import copy_span, copy_text
from . import model
from .blocks import anchor, code_anchor
from .spec import CORPUS_TABLE
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    INSTANCE_WEIGHTS_NAME,
    PROVENANCE_DIRECTORY,
    RELEASE_MANIFEST_NAME,
)
from dpstats.workspace.layout import registry_directory_name
from dpstats.workspace.provenance import (
    InstanceFacts,
    Provenance,
    read_provenance,
)

# The instance registry a Distribution Release publishes its roster in.
BENCHMARK_REGISTRY = "benchmark"

# The study-document keys the ruled pages render: ``instances``,
# ``data_license`` and ``discipline`` (shown as the instance rows' field) and
# the source page's sections. ``study_id`` names the page's own route rather
# than a section, so it joins them. Every other key a release carries is
# reported through ``Corpus.unrendered_keys`` and shown nowhere: a key with no
# ruled section is a decision for the template's owner, not one a renderer may
# take.
RENDERED_KEYS = (
    "instances",
    "data_license",
    "discipline",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "data_doi",
    "bibtex",
    "study_id",
    "data_license_id",
    "data_copyright",
)

# Where a release ships the licence texts its sources' notices refer to, one
# per licence id; the site publishes each beside its pages, at ``licenses/``.
LICENCE_TEXTS = f"{BENCHMARK_INFO_DIRECTORY}/licenses/"
# The canonical page of a licence the SPDX licence list names.
SPDX_LICENCE = "https://spdx.org/licenses/{id}.html"


def licence_href(licence_id: str, shipped: frozenset[str]) -> str:
    """Where a licence's text is read: the release's own copy where it ships
    one (a site-root path), else its SPDX page; nothing for an id the SPDX
    list does not name (``LicenseRef-``)."""
    if not licence_id:
        return ""
    if licence_id in shipped:
        return f"licenses/{licence_id}.txt"
    if licence_id.startswith("LicenseRef-"):
        return ""
    return SPDX_LICENCE.format(id=licence_id)


def licence_html(instance: "CorpusInstance", climb: str) -> str:
    """An instance's data licence as a fact value: its name, linked to its
    text, and the source's copyright line under it where it declares one."""
    name = escape(instance.data_license)
    href = instance.data_license_href
    if href:
        target = href if "://" in href else climb + href
        name = (
            f'<a href="{escape(target, quote=True)}" target="_blank" rel="noopener">'
            f"{name}</a>"
        )
    if instance.data_copyright:
        name += f'<br><span class="dp-licence-copyright">{escape(instance.data_copyright)}</span>'
    return name


def registry_relative(task_id: str) -> str:
    """Where one task's benchmark instance registry sits inside a release."""
    directory = registry_directory_name("instance")
    return f"{task_id}/{directory}/{BENCHMARK_REGISTRY}.json"


def weights_relative(task_id: str) -> str:
    """Where one task's released instance weights sit inside a release."""
    return f"{task_id}/{INSTANCE_WEIGHTS_NAME}"


@dataclass(frozen=True)
class CorpusInstance:
    """One instance a release distributes, as the Corpus area shows it."""

    uuid: str
    task: str
    label: str
    n: int
    dimension: str
    bounded: bool
    declared: tuple[str, ...]
    data_license: str
    sample_unit: str
    field: str
    source: str
    # Where the licence's text is read (see ``licence_href``), and the source's
    # copyright line where its licence requires one.
    data_license_href: str = ""
    data_copyright: str = ""
    # The instance's public Metadata and stored metric arguments, as its
    # registry entry carries them, and its released weight where the release
    # declares weights for its task.
    metadata: TaskMetadata | None = None
    references: tuple[tuple[str, Any], ...] = ()
    weight: float | None = None
    # The study document's verbatim source block for this instance: its Stata
    # ``variables``, its ``coefficient`` of interest and its routed ``claims``,
    # each key present only where the producer recorded a value.
    source_facts: Mapping[str, Any] = dataclasses.field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class CorpusSource:
    """One source, from its provenance bundle and the instances naming it."""

    slug: str
    title: str
    authors: tuple[str, ...]
    year: str
    venue: str
    doi: str
    data_doi: str
    bibtex: str
    instances: tuple[CorpusInstance, ...]


@dataclass(frozen=True)
class Corpus:
    """One Distribution Release's sources, and how this build read them."""

    version: str
    identity: str
    published: bool
    tasks: tuple[str, ...]
    sources: tuple[CorpusSource, ...]
    files_read: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    # Each licence text the release ships, by licence id, which the site
    # publishes at ``licenses/<id>.txt``.
    licence_texts: tuple[tuple[str, str], ...] = ()
    # Provenance keys the release carries that no ruled section renders. The
    # build reports them; no page shows them.
    unrendered_keys: tuple[str, ...] = ()


class _ReleaseFiles:
    """The read path into a release's registries and weights: declared files only.

    The provenance files (study records and the tables linking instances to
    them) are read through the library's one reader and checked against the
    manifest afterwards.
    """

    def __init__(
        self,
        root: Path,
        declared: frozenset[str],
        allowed: frozenset[str],
    ) -> None:
        self.root = root
        self.declared = declared
        self.allowed = allowed
        self.read: list[str] = []

    def read_text(self, relative: str) -> str:
        """Return one declared corpus file's text, refusing anything else.

        Args:
            relative: The file's path inside the release, POSIX style.

        Returns:
            The file's decoded text.

        Raises:
            ValueError: The path is not a corpus file, or the release's own
                manifest does not declare it.
        """
        if relative not in self.allowed:
            raise ValueError(
                f"The Corpus area reads instance registries and instance "
                f"weights here; {relative!r} is neither."
            )
        if relative not in self.declared:
            raise ValueError(
                f"The release manifest does not declare {relative!r}, so this "
                "build does not read it."
            )
        self.read.append(relative)
        return (self.root / relative).read_text(encoding="utf-8")


def _present(value: Any) -> str:
    """One provenance value as text, where the release carries one.

    A missing value, an empty string and the string ``"None"`` are all absent,
    and an absent value renders nothing rather than a stand-in.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def _authors(value: Any) -> tuple[str, ...]:
    """The author list a provenance bundle declares, in its own order."""
    if not isinstance(value, list):
        return ()
    return tuple(name for name in map(_present, value) if name)


def _dimension(metadata: TaskMetadata) -> str:
    """The instance's declared dimension, for a Metadata that declares one."""
    value = getattr(metadata, "dim", None)
    return f"{value:,}" if isinstance(value, int) else ""


def _bounded(metadata: TaskMetadata) -> bool:
    """Whether every domain this Metadata declares is publicly bounded."""
    domains = [
        value
        for value in (getattr(metadata, name) for name in type(metadata).model_fields)
        if isinstance(value, Domain)
    ]
    return bool(domains) and all(
        math.isfinite(domain.l2_diameter()) for domain in domains
    )


def _declared(metadata: TaskMetadata) -> tuple[str, ...]:
    """The Metadata fields this instance carries a value for."""
    return tuple(
        name
        for name in type(metadata).model_fields
        if getattr(metadata, name) is not None
    )


def _read_weights(files: _ReleaseFiles, task_id: str) -> dict[str, float]:
    """One task's released instance weights, or none where it declares none."""
    relative = weights_relative(task_id)
    if relative not in files.declared:
        return {}
    reader = csv.DictReader(files.read_text(relative).splitlines())
    return {row["instance"]: float(row["weight"]) for row in reader}


def _instance_rows(
    files: _ReleaseFiles,
    task_id: str,
    provenance: Provenance,
    skipped: list[str],
    *,
    published: bool,
    shipped: frozenset[str],
) -> tuple[list[CorpusInstance], set[str]]:
    """Every instance one task distributes, joined to its recorded facts.

    A local cut lists an entry this library cannot rebuild in ``skipped``;
    those UUIDs are returned beside the rows so the caller does not report
    them a second time.
    """
    weights = _read_weights(files, task_id)
    text = files.read_text(registry_relative(task_id)).encode("utf-8")
    rejected: dict[str, str] = {}
    if published:
        registry = read_instance_registry(text, task_id)
    else:
        registry, rejected = load_instances_partitioned(
            read_registry(text).entries, task_id=task_id
        )
    skipped.extend(
        f"{registry_relative(task_id)}: {uuid} cannot be rebuilt by this library: "
        f"{(reason.splitlines() or [''])[0]}"
        for uuid, reason in rejected.items()
    )
    rows: list[CorpusInstance] = []
    for uuid, instance in sorted(registry.items()):
        facts = provenance.instances.get(uuid)
        if facts is None:
            skipped.append(
                f"{registry_relative(task_id)}: {uuid} is named by no study "
                "document, so no source page can carry it."
            )
            continue
        rows.append(
            _declared_instance(
                uuid,
                task_id,
                facts,
                provenance.studies[facts.study].record,
                instance,
                weights.get(uuid),
                shipped,
            )
        )
    rows.sort(key=lambda row: (row.label, row.uuid))
    return rows, set(rejected)


def _declared_instance(
    uuid: str,
    task_id: str,
    facts: InstanceFacts,
    study: Mapping[str, Any],
    instance: Instance,
    weight: float | None,
    shipped: frozenset[str],
) -> CorpusInstance:
    """One instance's row: its public Metadata beside its recorded facts.

    The licence and field are the study's; the sample unit and the source
    block are the instance's own entry in the study document.
    """
    metadata = instance.metadata
    return CorpusInstance(
        uuid=uuid,
        task=task_id,
        label=instance.label or uuid,
        n=metadata.n_samples,
        dimension=_dimension(metadata),
        bounded=_bounded(metadata),
        declared=_declared(metadata),
        data_license=_present(study.get("data_license")),
        data_license_href=licence_href(_present(study.get("data_license_id")), shipped),
        data_copyright=_present(study.get("data_copyright")),
        sample_unit=facts.sample_unit,
        field=_present(study.get("discipline")),
        source=facts.study,
        metadata=metadata,
        references=tuple(instance.metric_arguments.items()),
        weight=weight,
        source_facts=facts.source,
    )


def _source(
    relative: str,
    record: Mapping[str, Any],
    instances: list[CorpusInstance],
    unrendered: set[str],
) -> CorpusSource:
    """One source: its study record, and the instances that came from it."""
    unrendered |= {key for key in record if key not in RENDERED_KEYS}
    slug = relative.removeprefix(
        f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/"
    ).removesuffix(".json")
    return CorpusSource(
        slug=slug,
        title=_present(record.get("title")) or slug,
        authors=_authors(record.get("authors")),
        year=_present(record.get("year")),
        venue=_present(record.get("venue")),
        doi=_present(record.get("doi")),
        data_doi=_present(record.get("data_doi")),
        bibtex=_present(record.get("bibtex")),
        instances=tuple(instances),
    )


def _cited(source: CorpusSource) -> tuple[str, str]:
    """The paper's citation and its data's, each as HTML or empty.

    Both come from the record's BibTeX through the site's one bibliography
    renderer; a side with no entry is empty. A record without BibTeX shows no
    citation.

    Raises:
        KeyError: The BibTeX holds neither of the source's own keys.
    """
    if not source.bibtex:
        return "", ""
    bibliography = load(source.bibtex)
    keys = (source.slug, f"{source.slug}_data")
    if not set(keys) & set(bibliography.entries):
        raise KeyError(f"{source.slug!r}: its BibTeX holds neither of its own keys")
    paper, data = (
        full_citation(key, bibliography) if key in bibliography.entries else ""
        for key in keys
    )
    return paper, data


def read_corpus(root: Path, *, identity: str, published: bool) -> Corpus:
    """Read one Distribution Release tree into the sources it distributes.

    Args:
        root: The release tree's root directory.
        identity: The release as the reader is told about it.
        published: Whether these pages report a published release rather than
            a local cut.

    Returns:
        The release's sources, ordered by title, each carrying exactly the
        instances this release links to it.

    Raises:
        ValueError: The release's provenance is not the ruled shape, or names
            a file the release manifest does not declare.
    """
    manifest: dict[str, Any] = json.loads(
        (root / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    tasks = tuple(sorted(manifest["tasks"]))
    declared = frozenset(manifest["files"])
    provenance = read_provenance(root, declared=declared)
    allowed = {registry_relative(task) for task in tasks}
    allowed |= {weights_relative(task) for task in tasks}
    texts = sorted(
        path
        for path in declared
        if path.startswith(LICENCE_TEXTS) and path.endswith(".txt")
    )
    allowed |= set(texts)
    files = _ReleaseFiles(root=root, declared=declared, allowed=frozenset(allowed))
    licence_texts = tuple(
        (path[len(LICENCE_TEXTS) : -len(".txt")], files.read_text(path))
        for path in texts
    )
    shipped = frozenset(licence for licence, _ in licence_texts)

    skipped: list[str] = []
    by_source: dict[str, list[CorpusInstance]] = {}
    placed: set[str] = set()
    for task_id in tasks:
        # A release states which instances it distributes for a task by
        # declaring its benchmark registry. A task it declares none for
        # distributes none, which is an absence with a reason rather than a
        # release this build refuses to read.
        if registry_relative(task_id) not in declared:
            skipped.append(
                f"{task_id}: the release declares no {registry_relative(task_id)}, "
                "so this build lists no instances for it."
            )
            continue
        rows, rejected = _instance_rows(
            files, task_id, provenance, skipped, published=published, shipped=shipped
        )
        placed |= rejected
        for row in rows:
            by_source.setdefault(row.source, []).append(row)
            placed.add(row.uuid)
    skipped += [
        f"{facts.study}: {uuid} is named but is not in a {BENCHMARK_REGISTRY} "
        "instance registry this release declares."
        for uuid, facts in sorted(provenance.instances.items())
        if uuid not in placed
    ]
    unrendered: set[str] = set()
    sources = [
        _source(relative, provenance.studies[relative].record, rows, unrendered)
        for relative, rows in sorted(by_source.items())
        if rows
    ]
    skipped += [
        f"{ref}: no instance this release distributes is named by it."
        for ref in sorted(provenance.studies)
        if ref not in by_source
    ]
    sources.sort(key=lambda source: (source.title.casefold(), source.slug))
    return Corpus(
        version=manifest["version"],
        identity=identity,
        published=published,
        tasks=tasks,
        sources=tuple(sources),
        files_read=(RELEASE_MANIFEST_NAME, *provenance.files_read, *files.read),
        skipped=tuple(skipped),
        unrendered_keys=tuple(sorted(unrendered)),
        licence_texts=licence_texts,
    )


# ── The records the Corpus renderers read ───────────────────────────────────

DOI_RESOLVER = "https://doi.org/"


def label_without_task(instance: CorpusInstance) -> str:
    """One instance's label as a reader reads it, without its task's name."""
    return instance.label.replace(f":{instance.task}/", "/")


def instance_route(instance: CorpusInstance) -> str:
    """One instance's own page, from the site root."""
    return f"{instance.task}/instances/{instance.uuid}/"


def _locator(value: str) -> str:
    """One declared locator as the URL a reader can open."""
    return value if value.startswith("http") else f"{DOI_RESOLVER}{value}"


def _count(value: int, key: str) -> str:
    """One count of rows this release carries, never of rows it withholds.

    *key* names the slot pair ``key_one`` and ``key_many`` that words it.
    """
    number = "one" if value == 1 else "many"
    return copy_text("corpus", f"{key}_{number}", fields={"count": f"{value:,}"})


def _sample_sizes(instances: Sequence[CorpusInstance]) -> str:
    """The span of declared sample counts across one source's instances."""
    sizes = sorted(instance.n for instance in instances)
    if not sizes:
        return ""
    if sizes[0] == sizes[-1]:
        return f"{sizes[0]:,}"
    return f"{sizes[0]:,}–{sizes[-1]:,}"


def corpus_index(
    corpus: Corpus | None, task_names: Mapping[str, str]
) -> model.CorpusIndex:
    """The Corpus index: every source this release distributes, once.

    Args:
        corpus: The release this build reads, or ``None`` where it reads none.
        task_names: Each carried task's reader-facing name.

    Returns:
        The area-landing record, complete before any filter runs.
    """
    if corpus is None:
        return model.CorpusIndex(
            title="Corpus",
            banner=copy_text("corpus", "no_release"),
            summary=copy_text("corpus", "no_sources"),
        )
    per_task: dict[str, int] = {task: 0 for task in corpus.tasks}
    rows: list[tuple[str, ...]] = []
    filters: list[dict[str, str]] = []
    fields: set[str] = set()
    for source in corpus.sources:
        tasks = sorted({instance.task for instance in source.instances})
        declared = sorted(
            {instance.field for instance in source.instances if instance.field}
        )
        fields |= set(declared)
        for instance in source.instances:
            per_task[instance.task] = per_task.get(instance.task, 0) + 1
        rows.append(
            (
                anchor(source.title, f"{source.slug}/"),
                source.year,
                ", ".join(task_names.get(task, task) for task in tasks),
                f"{len(source.instances):,}",
                _sample_sizes(source.instances),
            )
        )
        by_task = {
            task: [item for item in source.instances if item.task == task]
            for task in tasks
        }
        filters.append(
            {
                **{
                    f"data-instances-{task}": f"{len(held):,}"
                    for task, held in by_task.items()
                },
                **{
                    f"data-samples-{task}": _sample_sizes(held)
                    for task, held in by_task.items()
                },
                "data-tasks": " ".join(tasks),
                "data-fields": " ".join(declared),
                "data-search": " ".join(
                    part
                    for part in (
                        source.title,
                        *source.authors,
                        source.venue,
                        source.year,
                    )
                    if part
                ).lower(),
            }
        )
    total = sum(per_task.values())
    summary = " ".join(
        [
            copy_span(
                "corpus",
                "summary_counts",
                fields={
                    "sources": _count(len(corpus.sources), "source_count"),
                    "instances": _count(total, "instance_count"),
                },
            ),
            *(
                copy_span(
                    "corpus",
                    "summary_task",
                    fields={
                        "task": task_names.get(task, task),
                        "instances": _count(count, "task_instance_count"),
                    },
                )
                for task, count in per_task.items()
            ),
        ]
    )
    return model.CorpusIndex(
        title="Corpus",
        banner="",
        summary=summary,
        tasks=tuple((task, task_names.get(task, task)) for task in corpus.tasks),
        fields=tuple(sorted(fields)),
        columns=CORPUS_TABLE,
        rows=tuple(rows),
        filters=tuple(filters),
    )


def source_page(corpus: Corpus, source: CorpusSource) -> model.Source:
    """One source page: its bibliography, its declared facts, its instances.

    Args:
        corpus: The release this build reads, which the page names.
        source: The source to render.

    Returns:
        The T20 page record. A section the release carries nothing for is
        empty here and omitted by the renderer.
    """
    supporting = " · ".join(
        part for part in (", ".join(source.authors), source.venue, source.year) if part
    )
    licences = sorted(
        {licence_html(item, "../../") for item in source.instances if item.data_license}
    )
    facts = [(copy_span("corpus", "fact_data_licence"), "; ".join(licences))] + [
        (
            copy_span(*slot),
            escape(
                "; ".join(
                    sorted(
                        {
                            getattr(item, key)
                            for item in source.instances
                            if getattr(item, key)
                        }
                    )
                )
            ),
        )
        for slot, key in (
            (("corpus", "fact_sample_unit"), "sample_unit"),
            (("site", "fact_field"), "field"),
        )
    ]
    rows = [
        (
            code_anchor(
                label_without_task(instance), f"../../{instance_route(instance)}"
            ),
            instance.task.replace("_", " ").title(),
            f"{instance.n:,}",
            instance.dimension,
        )
        for instance in sorted(
            source.instances, key=lambda item: (item.task, item.label)
        )
    ]
    paper, data = _cited(source)
    return model.Source(
        title=source.title,
        supporting=supporting,
        paper=paper,
        paper_doi=_locator(source.doi) if source.doi else "",
        data=data,
        data_doi=_locator(source.data_doi) if source.data_doi else "",
        citation_note=(
            "" if paper or data else copy_text("corpus", "missing_citation")
        ),
        facts=tuple((name, value) for name, value in facts if value),
        columns=("Instance", "Task", "Samples (n)", "Dimension"),
        rows=tuple(rows),
    )
