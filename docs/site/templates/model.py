"""The plain page records the renderers read, and nothing that reads a library.

A renderer never touches a class, a registry, or a results root: it fills a
template's sections from these records. The installed library supplies them on
a real build (:mod:`templates.library_source`) and the filler registry supplies
them for the mock pages (:mod:`templates.filler`), so both go through exactly
one set of renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from site_copy import copy_text


@dataclass(frozen=True)
class Link:
    """One link: its visible text and its target, relative to the page."""

    text: str
    url: str


@dataclass(frozen=True)
class Access:
    """One object's Access block: the import line, then its keyed accessor."""

    import_line: str
    accessor: str = ""


@dataclass(frozen=True)
class Row:
    """One field-like table row. Empty cells render as an em dash."""

    name: str
    type: str
    constraints: str = ""
    description: str = ""
    default: str = ""


@dataclass(frozen=True)
class ModelBlock:
    """One declared model on the definition page: import, prose, field table."""

    title: str
    access: Access
    summary: str
    columns: tuple[str, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class MetricSection:
    """One metric's own section on the task's metrics page."""

    name: str
    key: str
    anchor: str
    docstring: str
    formula: str
    needs: tuple[str, ...]
    notes: str = ""
    # Each baseline the Metric is offered, linked to its entry in the page's
    # baselines table; empty where the build has no release.
    baselines: tuple[Link, ...] = ()


def rank_label(rank: int | None) -> str:
    """A headline standing as every table shows it: its rank, or not ranked."""
    if rank:
        return copy_text("roster", "rank", fields={"rank": str(rank)})
    return copy_text("roster", "not_ranked")


@dataclass(frozen=True)
class ConfiguredAlgorithm:
    """One configured algorithm on a task's Roster, as every page views it.

    ``rank`` is its position at the task's headline comparison, or ``None``
    where that panel does not rank it. ``task`` and ``url`` are set where a
    table links the algorithm's own page.
    """

    uuid: str
    label: str
    core: str
    transformations: tuple[str, ...]
    privacy_interpretation: str
    marker: str
    rank: int | None
    # Set only once a table links the page: the Roster join has neither.
    task: str = ""
    url: str = ""

    @property
    def standing(self) -> str:
        """The headline standing every table shows."""
        return rank_label(self.rank)


@dataclass(frozen=True)
class AreaIndex:
    """T2 for an area whose landing lists its pages.

    ``pages`` is one row per page the area's navigation declares, each with
    the page's own opening sentence where it has one.
    """

    title: str
    pages: tuple[tuple[Link, str], ...] = ()


@dataclass(frozen=True)
class TaskOverview:
    """T3: the page a reader lands on for a task."""

    name: str
    description: str
    access: Access
    quick_links: tuple[Link, ...]
    problem: str
    metrics: tuple[tuple[Link, str], ...] = ()


@dataclass(frozen=True)
class Definition:
    """T6: the task's declared Input, Metadata and Output."""

    title: str
    models: tuple[ModelBlock, ...]
    notes: str = ""


@dataclass(frozen=True)
class AlgorithmGroup:
    """One mechanism family on the Native algorithms page.

    Each row is the Algorithm, its docstring's opening line, its class and its
    short citation. An empty ``title`` is the page's one unheaded table.
    """

    title: str
    rows: tuple[tuple[Link, str, str, str], ...]


@dataclass(frozen=True)
class AlgorithmsIndex:
    """T7: every Algorithm written against this task, grouped by family."""

    title: str
    classes: tuple[str, ...]
    groups: tuple[AlgorithmGroup, ...]
    class_help: ReaderTerm | None = None


@dataclass(frozen=True)
class ReaderTerm:
    """One maintainer-defined term, shown as help on the word that names it.

    ``link`` is site-root relative, as the maintainer's file writes it.
    """

    key: str
    text: str
    link: str = ""


@dataclass(frozen=True)
class TransformationRow:
    """One Transformation on its task's table, and the class it carries there."""

    name: Link
    description: str
    privacy_interpretation: str
    inner: Link
    reference: str


@dataclass(frozen=True)
class TransformationGroup:
    """One role the Transformations page groups its rows under.

    ``converts`` marks the group whose rows hand the problem to another task,
    the only group whose table names that task. An empty ``title`` is the
    page's one unheaded table.
    """

    title: str
    summary: str
    rows: tuple[TransformationRow, ...]
    converts: bool = False


@dataclass(frozen=True)
class TransformationsIndex:
    """T8: every Transformation this task exports, grouped by what it does."""

    title: str
    groups: tuple[TransformationGroup, ...]
    classes: tuple[str, ...] = ()
    class_help: ReaderTerm | None = None


@dataclass(frozen=True)
class CorpusIndex:
    """T2 for Corpus: the sources one Distribution Release distributes.

    ``filters`` carries one attribute map per row, so the search, task and
    field controls narrow a table that is already complete without them.
    ``data-instances-<task>`` and ``data-samples-<task>`` hold each source's
    per-task instance count and sample sizes for the task filter to show.
    ``banner`` is the status line a build reading no release shows.
    """

    title: str
    banner: str
    summary: str
    tasks: tuple[tuple[str, str], ...] = ()
    fields: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    filters: tuple[dict[str, str], ...] = ()
    # The link to the About area's data and licensing page, as HTML, under
    # the summary; empty where the build publishes no such page.
    data_link: str = ""


@dataclass(frozen=True)
class Source:
    """T20: one source, and the instances this release draws from it.

    ``paper`` and ``data`` are the paper's and its data's bibliography
    entries as HTML, each empty where the release carries none; ``paper_doi``
    and ``data_doi`` are their DOI addresses, shown where no entry is.
    ``citation_note`` states why neither entry is shown. ``facts``
    are term and value pairs, each term and value HTML.
    """

    title: str
    supporting: str = ""
    paper: str = ""
    paper_doi: str = ""
    data: str = ""
    data_doi: str = ""
    citation_note: str = ""
    facts: tuple[tuple[str, str], ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class InstanceDomain:
    """One declared domain of one instance: its summary and its columns.

    Each row is a column or a run of columns: their positions, their kind and
    the values they take. The name, the summary and every cell are HTML.
    """

    name: str
    summary: str
    rows: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class InstancePage:
    """T24: the one page for one instance a release distributes.

    ``facts`` are term and value pairs, each term and value already HTML.
    ``variables``, the
    ``coefficient`` and the one ``claim`` shown, as ``(quote, page, table)``,
    are the paper's own words as plain text, verbatim; a claim with no quote
    has ``None``.
    """

    title: str
    facts: tuple[tuple[str, str], ...] = ()
    domains: tuple[InstanceDomain, ...] = ()
    variables: tuple[str, ...] = ()
    coefficient: str = ""
    claim: tuple[str | None, str, str] | None = None


@dataclass(frozen=True)
class BaselineEntry:
    """T18: one baseline of the task's catalogue.

    ``declarer`` names the side declaring it, the library or the benchmark
    protocol. ``definition``, ``applies`` and ``role`` are already rendered.
    """

    name: str
    declarer: str
    definition: str
    applies: str
    role: str


@dataclass(frozen=True)
class Metrics:
    """T18: the task's metric registry, summarized and then in full.

    ``baselines`` catalogues every baseline the task offers; empty where the
    build has no release.
    """

    title: str
    access: Access
    metrics: tuple[MetricSection, ...]
    metric_argument_descriptions: tuple[tuple[str, str], ...] = ()
    arguments_intro: str = ""
    baselines: tuple[BaselineEntry, ...] = ()


# The two kinds of component the one component page renders.
ALGORITHM = "algorithm"
TRANSFORMATION = "transformation"


@dataclass(frozen=True)
class ApiMethod:
    """One public method on the API section: signature, summary, parameters."""

    name: str
    signature: str
    summary: str = ""
    parameters: tuple[Row, ...] = ()


@dataclass(frozen=True)
class Parameter:
    """T9: one configurable parameter, as its Parameters entry reads.

    ``type`` is the declared type with its declared domain folded in, such as
    ``float > 0``; ``description`` and ``note`` are HTML. ``note`` is the
    parameter's own ``Parametrization notes:`` entry, empty where it has none.
    """

    name: str
    type: str
    default: str
    description: str = ""
    note: str = ""


@dataclass(frozen=True)
class RosterLink:
    """T9: one task's Roster, pre-filtered to the configured algorithms using a component.

    ``url`` is measured from the site root; the page re-measures it.
    """

    task: str
    count: int
    url: str


@dataclass(frozen=True)
class Component:
    """T9: the one page for an Algorithm or a Transformation.

    ``kind`` is the switch the one renderer reads: ``"algorithm"`` or
    ``"transformation"``. ``task`` is the native task for an Algorithm, the
    outer-to-inner task pair for a Transformation, as Markdown. ``summary`` is
    the docstring's opening line and ``description`` the prose after it, less
    the sentence ``source`` replaces. ``source`` is the HTML the class's
    ``Reference:`` block reads as in a sentence. ``api_signature`` and
    ``api_methods`` are the API section, rendered from the class declaration
    rather than from prose. ``roster`` holds the component's own task's Roster
    link first.
    """

    name: str
    task: str
    summary: str
    description: str
    privacy_interpretation: str
    semantics_link: Link
    usage: str
    api_signature: str
    kind: str = "algorithm"
    api_methods: tuple[ApiMethod, ...] = ()
    source: str = ""
    reference: str = ""
    requires: tuple[str, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    roster: tuple[RosterLink, ...] = ()
    parametrization_notes: str = ""
    implementation_notes: str = ""


@dataclass(frozen=True)
class ChainStep:
    """T22: one component of a configured algorithm's chain, in application order.

    ``name`` is the step's name in the configured algorithm's label and
    ``hyperparameters`` the suffix the label gives it; ``component`` is its
    class name. ``key`` is the transformation filter's value for a
    transformation step and empty for the core.
    """

    name: str
    component: str
    url: str
    hyperparameters: str = ""
    key: str = ""


@dataclass(frozen=True)
class RosterRow:
    """T22: one configured algorithm's row: its chain and its headline reading.

    ``headline`` is its error ratio at the headline quantile, empty where the
    headline comparison does not rank the row; ``tau`` is that value.
    """

    algorithm: ConfiguredAlgorithm
    chain: tuple[ChainStep, ...]
    headline: str = ""
    tau: float | None = None


@dataclass(frozen=True)
class RosterLanding:
    """T22: every configured algorithm the release publishes for the task.

    ``transformations`` is the transformation filter's ``(key, label)`` pairs.
    ``budget`` and ``metric`` name the headline reading every Rank and
    headline cell is read at, and ``percent`` the quantile in percent. ``notice`` is a banner
    the page opens with, empty when there is none.
    """

    title: str
    rows: tuple[RosterRow, ...]
    classes: tuple[str, ...] = ()
    transformations: tuple[tuple[str, str], ...] = ()
    metric: str = ""
    budget: str = ""
    percent: str = ""
    notice: str = ""


@dataclass(frozen=True)
class StandingRow:
    """T21: one panel's rank, headline error ratio and coverage for one algorithm."""

    metric: Link
    budget: str
    rank: str
    headline: str
    coverage: str = ""


@dataclass(frozen=True)
class OutcomeRow:
    """T21: one budget's declared distribution-lane trial outcomes.

    ``rank`` is the algorithm's rank among every algorithm on the public
    corpus at the headline metric and threshold, as Explore opens; every other
    field but ``budget`` is a count and its share of recorded trials, already
    rendered.
    """

    budget: str
    rank: str
    success: str
    no_release: str
    invalid_regime: str
    algorithm_error: str
    timeout: str
    memory_limit_exceeded: str


@dataclass(frozen=True)
class RecipeComponent:
    """T21: one component in a configured algorithm's recipe chain, with its own values.

    ``values`` is one ``(name, value, description)`` triple per configured
    field, already rendered.
    """

    name: Link
    values: tuple[tuple[str, str, str], ...]
    children: tuple["RecipeComponent", ...] = ()


@dataclass(frozen=True)
class RosterPage:
    """T21: the one page for one configured algorithm.

    ``unranked_note`` is set instead of populating ``standings`` where the
    algorithm appears in no released panel: the section then shows its
    class and "not ranked" in one line, not one all-dashes row per panel.
    ``leaderboard_url`` and ``explore_url`` are the task's results pages the
    two results headings link to; an empty ``explore_url`` leaves its plain.
    """

    algorithm: ConfiguredAlgorithm
    task_link: Link
    description: str
    standings: tuple[StandingRow, ...]
    outcomes: tuple[OutcomeRow, ...]
    recipe: tuple[RecipeComponent, ...]
    configuration_json: str
    percent: str
    leaderboard_url: str
    explore_url: str
    class_url: str
    provenance: str = ""
    unranked_note: bool = False


@dataclass(frozen=True)
class Methodology:
    """T15: the release's protocol narrative.

    ``narrative`` is the release's own Markdown under its title, rendered as
    written. ``banner`` is the status line a build reading no release shows.
    """

    title: str
    narrative: str
    identity: str
    banner: str = ""
