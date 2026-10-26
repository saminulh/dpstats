"""Resolve the one Distribution Release a documentation build renders.

A build reads exactly one input. ``docs/benchmark.lock`` pins one immutable
published release, which the library installs as one tree, or
``DOCS_LOCAL_RELEASE_ROOT`` names one local Distribution Release tree; that
local cut is the preview, and there is no other preview mode. A fetched tree
is the tree a local cut of the same release holds, so one reader serves both.
Nothing infers what a directory is by looking inside it, and the public
workflow sets no variable, so a local tree cannot reach a public deploy by
construction.

The release declares what it released. ``release.json`` names the Benchmarked
tasks and the recipe each ran under, so the site opens a Leaderboard and an
Explore for exactly those tasks and no others; each task ships its standings
view and its instance weights at their fixed layout paths.

``benchmark-protocol.json`` states each task in two sections. Its ``ranking``
section decides the official standings: the headline Metric, its baseline and
what that baseline resolves to, the threshold, the grid, the weighting, the
eligibility class and the computed baselines. Its ``external_baselines``
section declares stored per-instance values the source supplies, each with a
description and the Metrics it is offered for. Only the fields a page shows
are read, and a field this build does not read is never refused.

The ranking always comes from the release the build renders. The external
baselines may lead it: ``DOCS_CURRENT_PROTOCOL`` names the current protocol in
its released JSON encoding, and its external baselines replace the shipped
ones, provided its ranking section is the shipped one exactly. A differing
ranking is refused, because it would be standings this release did not
publish. With the variable unset the shipped document is read alone.

Nothing here computes, filters, weighs, orders, or ranks a value.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import huggingface_hub

from dpstats.transport.validators import is_immutable_revision
from dpstats.workspace.fetch import _fetch_release_tree
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_PROTOCOL_NAME,
    INSTANCE_WEIGHTS_NAME,
    RELEASE_MANIFEST_NAME,
    WITHHELD_PROFILES_NAME,
    task_collection_directory,
)

from benchmark_lock import BenchmarkPin

# The environment variable naming one local Distribution Release tree. The
# public build leaves it unset.
LOCAL_RELEASE_ENV = "DOCS_LOCAL_RELEASE_ROOT"

# The environment variable naming the current protocol, in its released JSON
# encoding, whose external baselines a build shows beside the release's own
# ranking. Unset, the release's own document is read alone.
CURRENT_PROTOCOL_ENV = "DOCS_CURRENT_PROTOCOL"

# The surface rendering a release's standings.
LEADERBOARD_SURFACE = "leaderboard"

# The results surfaces one release opens for each task it released, as
# (route segment, page and navigation name), in the order pages present them.
# A route segment is a published URL and a name is what a reader reads, so the
# two are separate: renaming a surface never moves the pages a link reaches.
RESULTS_SURFACES = (
    (LEADERBOARD_SURFACE, "Leaderboard"),
    ("public-results", "Explore"),
)

# The manifest roles of the two views each Benchmarked task ships: the
# evaluation lane's aggregated curves, which the Leaderboard renders as
# published, and the distribution lane's instance weights, which Explore weighs
# its own measurements by.
WITHHELD_PROFILES_ROLE = "withheld-profiles"
INSTANCE_WEIGHTS_ROLE = "instance-weights"

# The kind a headline baseline resolves to when it is a value each instance
# stores, and the kinds the ranking section defines by name: the best median
# over a pool, the best over a named roster, one algorithm's own cost, or no
# division at all.
REFERENCE_BASELINE_KIND = "reference"
COMPUTED_BASELINE_KINDS = ("best_median", "best_of", "algorithm", "none")

# The name the pooled baseline carries in every task, and in the panels' own
# vocabulary.
BEST_MEDIAN = "best_median"

# The name reading a Metric in its own units carries: division by one.
NO_BASELINE = "none"


@dataclass(frozen=True)
class ReleasedTask:
    """One Benchmarked task, as its release declares it.

    Attributes:
        sweep: The recipe the release ran this task under.
        views: The sanctioned view this release ships under each role.
    """

    sweep: Path
    views: dict[str, Path]
    results: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ReleaseInput:
    """The one Distribution Release tree this build renders.

    Attributes:
        root: The tree itself.
        version: The version the release names itself by.
        manifest_id: The release manifest's own content identity.
        identity: The release as the reader is told about it.
        revision: The repository commit a published release was installed
            from, or empty for a local cut.
        published: Whether these pages report a published release rather than
            a local cut.
        tasks: Each Benchmarked task the release declares, in manifest order.
        protocol: The protocol document this build reads: the shipped one, or
            the shipped ranking beside the current external baselines.
    """

    root: Path
    version: str
    manifest_id: str
    identity: str
    revision: str
    published: bool
    tasks: dict[str, ReleasedTask]
    protocol: bytes


@dataclass(frozen=True)
class ResultsRoute:
    """One task's one results surface under the release this build reads.

    Attributes:
        surface: Which results surface this route renders.
        route: Where the surface renders, relative to the task's own pages.
        title: The surface's navigation entry and page heading.
        identity: The release the page renders, named for the reader.
        published: Whether the page reports a published release.
        root: The release tree the page's data was prepared from.
    """

    surface: str
    route: str
    title: str
    identity: str
    published: bool
    root: Path


@dataclass(frozen=True)
class ReleasedBaseline:
    """One baseline the ranking defines or resolves, by its name and kind.

    Attributes:
        name: The baseline's name; for a stored value, its key.
        kind: ``reference`` or one of ``COMPUTED_BASELINE_KINDS``.
        detail: The declaration's remaining fields as stated, in order: a pool
            and aggregation, a roster, or one algorithm.
    """

    name: str
    kind: str
    detail: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ExternalBaseline:
    """One stored per-instance value the protocol offers as a baseline.

    Attributes:
        name: The baseline's name, which is the key each instance stores it
            under.
        description: Where the value comes from, as the protocol states it.
        metrics: The Metrics the protocol offers it for, in the order stated.
    """

    name: str
    description: str
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class ReleasedGrid:
    """The threshold grid a released task's curves are evaluated on."""

    min: float
    max: float
    scale: str
    steps: int


@dataclass(frozen=True)
class ReleasedSlice:
    """One official slice of a task's Instances, as the protocol states it.

    Attributes:
        key: The slice's name in the release.
        feature: The Instance property it splits on (``d`` or ``n_over_d``).
        bound: The value it splits at.
        above: Whether the slice holds Instances above ``bound`` (else at most).
    """

    key: str
    feature: str
    bound: float
    above: bool


# The Instance properties a slice may split on.
SLICE_FEATURES = ("d", "n_over_d")


@dataclass(frozen=True)
class ReleasedComparison:
    """The fields of one task's protocol entry a page shows.

    Attributes:
        headline_metric: The Metric the official standings rank.
        headline_baseline: The baseline those standings divide it by.
        headline_definition: What that baseline resolves to: a stored value
            under its key, or a computed baseline.
        headline_quantile: The share of weighted trials the standings rank
            by: each algorithm's error ratio at that quantile, lower first.
        taus: The thresholds the released grid names, ascending.
        grid: The grid those thresholds were stated as.
        definitions: Every computed baseline the ranking defines, by name, in
            the order it states them; ``best_median`` is always among them.
        external: Every external baseline, by name, in the order stated.
        weighting: How the standings weight instances, as the protocol names
            it.
        eligibility: The privacy interpretation the standings rank.
        slices: The official slices the standings are also ranked within, in
            order; the whole set ("all") is implicit and comes first.
    """

    headline_metric: str
    headline_baseline: str
    headline_definition: ReleasedBaseline
    headline_quantile: float
    taus: tuple[float, ...]
    grid: ReleasedGrid
    definitions: dict[str, ReleasedBaseline]
    external: dict[str, ExternalBaseline]
    weighting: str
    eligibility: str
    slices: tuple[ReleasedSlice, ...] = ()

    @property
    def pool(self) -> str:
        """The privacy interpretation the pooled baseline ranks."""
        return dict(self.definitions[BEST_MEDIAN].detail)["pool"]

    def stored_for(self, metric: str, statistical: str | None) -> tuple[str, ...]:
        """The stored values one Metric may be divided by, in offered order.

        *statistical* is the Metric's own statistical baseline, if it has
        one; the external baselines the protocol offers for this Metric
        follow, then the headline baseline on the headline Metric where it is
        a stored value neither names.
        """
        named = [
            *([statistical] if statistical else []),
            *(name for name, entry in self.external.items() if metric in entry.metrics),
        ]
        if (
            metric == self.headline_metric
            and self.headline_definition.kind == REFERENCE_BASELINE_KIND
        ):
            named.append(self.headline_baseline)
        return tuple(dict.fromkeys(named))

    def computed_for(self, metric: str) -> tuple[str, ...]:
        """The computed baselines one Metric is offered: the headline's, pooled."""
        named = [BEST_MEDIAN]
        if (
            metric == self.headline_metric
            and self.headline_definition.kind != REFERENCE_BASELINE_KIND
        ):
            named.insert(0, self.headline_baseline)
        return tuple(dict.fromkeys(named))

    def opening_for(
        self, metric: str, statistical: str | None, offered: Collection[str]
    ) -> str:
        """The baseline one Metric opens on, among those *offered* for it.

        The headline Metric opens on its headline baseline. Any other Metric
        opens on *statistical*, its own statistical baseline, where that is
        offered, and otherwise on ``none``. The external and pooled baselines
        are offered but never a Metric's opening unless one is the headline
        baseline.
        """
        if metric == self.headline_metric:
            return self.headline_baseline
        if statistical is not None and statistical in offered:
            return statistical
        return NO_BASELINE


def release_input(
    pin: BenchmarkPin | None,
    local_root: str | None,
    *,
    cache_dir: Path,
    current_protocol: str | None = None,
) -> ReleaseInput | None:
    """Return the one release this build reads, or ``None`` where it reads none.

    Args:
        pin: The release the committed lock names, or ``None`` where it pins
            none.
        local_root: The raw ``DOCS_LOCAL_RELEASE_ROOT`` value, or ``None`` when
            it is unset.
        cache_dir: The directory a pinned release installs into.
        current_protocol: The raw ``DOCS_CURRENT_PROTOCOL`` value, or ``None``
            when it is unset. A build reading no release has nothing for it to
            supplement.

    Raises:
        ValueError: If both inputs are supplied, since a build reads one, if
            the release this build reads does not ship what it declares, or if
            the current protocol cannot stand beside it.
    """
    if local_root and pin is not None:
        raise ValueError(
            f"{LOCAL_RELEASE_ENV} names a local release while the committed "
            "lock pins one; a build renders exactly one Distribution Release."
        )
    current = Path(current_protocol) if current_protocol else None
    if local_root:
        return _local_release(Path(local_root).resolve(), current)
    if pin is not None:
        return _pinned_release(pin, cache_dir, current)
    return None


def results_routes(release: ReleaseInput | None) -> dict[str, tuple[ResultsRoute, ...]]:
    """Return every results route this build renders, by task.

    Args:
        release: The one release this build reads, or ``None``.

    Returns:
        Each Benchmarked task's routes, in the order the release declares the
        tasks and the surfaces are presented.
    """
    if release is None:
        return {}
    return {
        task_id: tuple(
            ResultsRoute(
                surface=surface,
                route=surface,
                title=name,
                identity=release.identity,
                published=release.published,
                root=release.root,
            )
            for surface, name in RESULTS_SURFACES
        )
        for task_id in release.tasks
    }


def _document(source: Path | bytes, name: str) -> dict[str, Any]:
    """Read one protocol document as JSON, refusing anything else."""
    try:
        text = source if isinstance(source, bytes) else source.read_bytes()
        document = json.loads(text)
    except (OSError, ValueError) as error:
        raise ValueError(f"Benchmark protocol is not readable JSON: {name}.") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("comparison"), dict
    ):
        raise ValueError(f"{name} states no comparison table.")
    return document


def composed_protocol(shipped: Path, current: Path | None) -> bytes:
    """The protocol document a build reads, as bytes.

    With no *current* document this is the shipped document unchanged. With
    one, each task keeps the shipped ``ranking`` section and takes the current
    ``external_baselines`` section.

    Raises:
        ValueError: If *current* is not a released JSON encoding, rules other
            tasks than the release, or states any task's ranking differently;
            the differing tasks and fields are named.
    """
    try:
        shipped_bytes = shipped.read_bytes()
    except OSError as error:
        raise ValueError(f"Benchmark protocol is not readable: {shipped}.") from error
    if current is None:
        return shipped_bytes
    if current.suffix != ".json":
        raise ValueError(
            f"{CURRENT_PROTOCOL_ENV} names {current}, which is not a released "
            "JSON encoding; write one with `maintenance protocol --encode`."
        )
    before = _document(shipped_bytes, str(shipped))
    after = _document(current, str(current))
    tasks, current_tasks = before["comparison"], after["comparison"]
    if set(tasks) != set(current_tasks):
        raise ValueError(
            f"The current protocol {current} rules tasks {sorted(current_tasks)}; "
            f"the release rules {sorted(tasks)}. A re-cut is needed."
        )
    differing: dict[str, list[str]] = {}
    for task_id, entry in tasks.items():
        ranking = _section(entry, "ranking", str(shipped), task_id)
        changed = _section(current_tasks[task_id], "ranking", str(current), task_id)
        fields = sorted(
            field
            for field in set(ranking) | set(changed)
            if ranking.get(field) != changed.get(field)
        )
        if fields:
            differing[task_id] = fields
    if differing:
        raise ValueError(
            f"The current protocol's ranking differs from the release's: "
            f"{differing}. Only external baselines may lead a release; a "
            "re-cut is needed."
        )
    composed = {
        **before,
        "comparison": {
            task_id: {
                **entry,
                "external_baselines": current_tasks[task_id].get(
                    "external_baselines", {}
                ),
            }
            for task_id, entry in tasks.items()
        },
    }
    return json.dumps(composed, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _section(entry: Any, name: str, path: str, task_id: str) -> dict[str, Any]:
    """One task's named section, refusing a document from before the split."""
    if not isinstance(entry, dict):
        raise ValueError(f"{path} states task {task_id!r} as something but a table.")
    section = entry.get(name)
    if name == "ranking" and "ranking" not in entry:
        raise ValueError(
            f"{path} states task {task_id!r} without a ranking section. This "
            "build reads a protocol whose every task states a ranking section "
            "and an external_baselines section; the release needs a re-cut."
        )
    if not isinstance(section, dict):
        raise ValueError(
            f"{path} states task {task_id!r}'s {name} as something but a table."
        )
    return section


def read_comparison(source: Path | bytes, task_id: str) -> ReleasedComparison:
    """Read the fields of one task's protocol entry a page shows.

    Every ranking field is stated in the released document, so nothing is
    defaulted here and no other field is read.

    Args:
        source: The protocol document, as a file or its bytes.
        task_id: The released task to read.

    Raises:
        ValueError: If the document is unreadable, rules no comparison for the
            task, or states a field a page shows in a form it cannot show.
    """
    name = str(source) if isinstance(source, Path) else BENCHMARK_PROTOCOL_NAME
    ruled = _document(source, name)["comparison"]
    if task_id not in ruled:
        raise ValueError(
            f"{name} rules no comparison for task {task_id!r}, which this "
            "release declares."
        )
    entry = ruled[task_id]
    ranking = _section(entry, "ranking", name, task_id)
    external = _external_baselines(name, task_id, entry.get("external_baselines", {}))
    headline = _name(name, task_id, ranking, "headline_metric")
    taus = _grid_values(name, task_id, ranking.get("tau_grid"))
    grid = ranking["tau_grid"]
    quantile = ranking.get("headline_quantile")
    if (
        not isinstance(quantile, (int, float))
        or isinstance(quantile, bool)
        or not 0 < quantile < 1
    ):
        raise ValueError(
            f"{name} states no headline quantile in (0, 1) for {task_id!r}."
        )
    definitions = _computed_baselines(name, task_id, ranking.get("baselines"))
    shared = sorted(set(definitions) & set(external))
    if shared:
        raise ValueError(
            f"{name} names external baseline(s) {shared} of task {task_id!r} "
            "like computed ones; one name is one baseline."
        )
    headline_baseline = _name(name, task_id, ranking, "headline_baseline")
    return ReleasedComparison(
        headline_metric=headline,
        headline_baseline=headline_baseline,
        headline_definition=_headline_definition(
            name, task_id, headline_baseline, ranking.get("headline_definition")
        ),
        headline_quantile=float(quantile),
        taus=taus,
        grid=ReleasedGrid(
            min=float(grid["min"]),
            max=float(grid["max"]),
            scale=str(grid["scale"]),
            steps=int(grid["steps"]),
        ),
        definitions=definitions,
        external=external,
        weighting=_name(name, task_id, ranking, "weighting"),
        eligibility=_name(name, task_id, ranking, "eligibility"),
        slices=_slices(name, task_id, ranking.get("slices", [])),
    )


def _slices(path: str, task_id: str, stated: Any) -> tuple[ReleasedSlice, ...]:
    """The slices one ranking section states, in order."""
    if not isinstance(stated, list):
        raise ValueError(f"{path} states slices for {task_id!r} that are not a list.")
    slices: list[ReleasedSlice] = []
    for entry in stated:
        bounds = [side for side in ("at_most", "above") if side in entry]
        key = entry.get("key") if isinstance(entry, dict) else None
        if (
            not isinstance(key, str)
            or not key
            or key == "all"
            or entry.get("feature") not in SLICE_FEATURES
            or len(bounds) != 1
            or isinstance(entry[bounds[0]], bool)
            or not isinstance(entry[bounds[0]], (int, float))
        ):
            raise ValueError(
                f"{path} states a slice for {task_id!r} that is not a key, a "
                f"feature among {SLICE_FEATURES} and one at_most or above bound: "
                f"{entry!r}."
            )
        slices.append(
            ReleasedSlice(
                key, entry["feature"], float(entry[bounds[0]]), bounds[0] == "above"
            )
        )
    if len({entry.key for entry in slices}) != len(slices):
        raise ValueError(f"{path} names one slice twice for {task_id!r}.")
    return tuple(slices)


def _name(path: str, task_id: str, entry: Mapping[str, Any], field: str) -> str:
    """One named field the entry states."""
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} states no {field} name for task {task_id!r}.")
    return value


# The declaration fields beyond its kind that a page states, per kind.
_DETAIL_FIELDS = {
    "best_median": ("pool", "aggregation"),
    "best_of": ("algorithms", "aggregation"),
    "algorithm": ("uuid", "aggregation"),
}


def _definition(
    path: str, task_id: str, name: str, record: Any, kinds: Sequence[str]
) -> ReleasedBaseline:
    """One baseline declaration, by its kind and the fields that state it.

    A kind this build does not show is refused by name rather than mapped to
    one it does.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"{path} states baseline {name!r} of task {task_id!r} as something "
            "but a table."
        )
    kind = record.get("kind")
    if kind not in kinds:
        raise ValueError(
            f"{path} declares baseline {name!r} of task {task_id!r} with kind "
            f"{kind!r}, which is none of {list(kinds)}."
        )
    detail = []
    for field in _DETAIL_FIELDS.get(str(kind), ()):
        value = record.get(field)
        if value is None:
            raise ValueError(
                f"{path} declares baseline {name!r} of task {task_id!r} without "
                f"its {field!r}."
            )
        text = ", ".join(map(str, value)) if isinstance(value, list) else value
        detail.append((field, str(text)))
    return ReleasedBaseline(name, str(kind), tuple(detail))


def _computed_baselines(
    path: str, task_id: str, declared: Any
) -> dict[str, ReleasedBaseline]:
    """Every computed baseline the ranking defines, ``best_median`` included."""
    if not isinstance(declared, dict) or BEST_MEDIAN not in declared:
        raise ValueError(
            f"{path} states task {task_id!r}'s computed baselines without "
            f"{BEST_MEDIAN!r}."
        )
    found = {
        name: _definition(path, task_id, name, record, COMPUTED_BASELINE_KINDS)
        for name, record in declared.items()
    }
    if found[BEST_MEDIAN].kind != "best_median":
        raise ValueError(
            f"{path} declares {BEST_MEDIAN!r} of task {task_id!r} as another "
            "kind; it is the pooled baseline and only that."
        )
    return found


def _headline_definition(
    path: str, task_id: str, name: str, record: Any
) -> ReleasedBaseline:
    """What the headline baseline resolves to, as the ranking states it.

    A stored value is named by its key alone, so a headline reading a stored
    key other than its own name is refused.
    """
    definition = _definition(
        path,
        task_id,
        name,
        record,
        (REFERENCE_BASELINE_KIND, *COMPUTED_BASELINE_KINDS),
    )
    if definition.kind == REFERENCE_BASELINE_KIND and record.get("key") != name:
        raise ValueError(
            f"{path} reads task {task_id!r}'s headline against stored key "
            f"{record.get('key')!r} under the name {name!r}; a stored value is "
            "named by its key alone."
        )
    return definition


def _external_baselines(
    path: str, task_id: str, declared: Any
) -> dict[str, ExternalBaseline]:
    """Every external baseline, by name: its description and its Metrics.

    A stored value is named by its key alone, which is also the name a page
    and a reader's notebook read it by. The protocol's ``metrics`` list is
    what the site offers it for; the library gates nothing.
    """
    if not isinstance(declared, dict):
        raise ValueError(
            f"{path} states task {task_id!r}'s external baselines as something "
            "but a table."
        )
    found: dict[str, ExternalBaseline] = {}
    for name, record in declared.items():
        if not isinstance(record, dict) or record.get("key") != name:
            raise ValueError(
                f"{path} declares external baseline {name!r} of task {task_id!r} "
                "other than under its own stored key; a stored value is named "
                "by its key alone."
            )
        description = record.get("description")
        metrics = record.get("metrics")
        if not isinstance(description, str) or not description:
            raise ValueError(
                f"{path} declares external baseline {name!r} of task "
                f"{task_id!r} without a description."
            )
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(metric, str) and metric for metric in metrics)
        ):
            raise ValueError(
                f"{path} declares external baseline {name!r} of task "
                f"{task_id!r} for no list of Metrics."
            )
        found[name] = ExternalBaseline(name, description, tuple(metrics))
    return found


def _grid_values(path: str, task_id: str, grid: Any) -> tuple[float, ...]:
    """Expand the threshold grid one released task states.

    ``steps`` thresholds run from ``min`` through ``max`` inclusive, spaced
    evenly on the stated scale.
    """
    if not isinstance(grid, dict) or {"min", "max", "scale", "steps"} - set(grid):
        raise ValueError(
            f"{path} states no min, max, scale and steps grid for task {task_id!r}."
        )
    values: dict[str, Any] = grid
    low = float(values["min"])
    high = float(values["max"])
    steps = int(values["steps"])
    scale = str(values["scale"])
    if steps < 2 or not 0.0 <= low < high or not math.isfinite(high):
        raise ValueError(
            f"{path} states a grid for task {task_id!r} that does not rise "
            "from a nonnegative minimum to a finite maximum in at least two steps."
        )
    if scale == "linear":
        span = (high - low) / (steps - 1)
        return tuple(low + index * span for index in range(steps))
    if scale == "log":
        if low <= 0.0:
            raise ValueError(
                f"{path} states a log grid for task {task_id!r} that does not "
                "rise from a positive minimum."
            )
        start = math.log10(low)
        end = math.log10(high)
        return tuple(
            10.0 ** (start + index * (end - start) / (steps - 1))
            for index in range(steps)
        )
    if scale == "log1p":
        # Evenly spaced in log(1 + τ): near zero almost linear, far out
        # logarithmic, with τ = 0 on the grid.
        start = math.log1p(low)
        end = math.log1p(high)
        # The ends are the stated values exactly, as the release writes them;
        # expm1(log1p(high)) can fall a rounding short.
        inner = tuple(
            math.expm1(start + index * (end - start) / (steps - 1))
            for index in range(1, steps - 1)
        )
        return (low, *inner, high)
    raise ValueError(f"{path} states the unknown grid scale {scale!r} for {task_id!r}.")


def _released_tasks(root: Path) -> tuple[dict[str, Any], dict[str, ReleasedTask]]:
    """Read one Distribution Release tree, gating on the views each task ships.

    A fetched tree is the tree a local cut of the same release holds, so both
    inputs arrive here and nothing downstream can tell which it was reading.
    The manifest is read as plain JSON: only its task list and each task's
    sweep path are used, and every Benchmarked
    task's standings view and instance weights live at their fixed layout
    paths, never at a manifest-declared one.
    """
    try:
        manifest: dict[str, Any] = json.loads(
            (root / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read Distribution Release manifest: {root}."
        ) from error
    tasks: dict[str, ReleasedTask] = {}
    for task_id, record in manifest["tasks"].items():
        if "results" not in record:
            raise ValueError(
                f"Release at {root} predates Results availability; a re-cut is needed."
            )
        task_root = task_collection_directory(root, task_id)
        views = {
            WITHHELD_PROFILES_ROLE: task_root
            / BENCHMARK_INFO_DIRECTORY
            / WITHHELD_PROFILES_NAME,
            INSTANCE_WEIGHTS_ROLE: task_root / INSTANCE_WEIGHTS_NAME,
        }
        absent = sorted(str(path) for path in views.values() if not path.is_file())
        if absent:
            raise ValueError(
                f"Release at {root} declares view(s) it does not carry: {absent}."
            )
        availability = record["results"]
        if set(availability) != {"distribution", "evaluation"}:
            raise ValueError(
                f"Release task {task_id!r} must state both lanes' Results availability."
            )
        for lane, algorithms in availability.items():
            for uuid, result in algorithms.items():
                if result["state"] not in {"ready", "pending"}:
                    raise ValueError(
                        f"Unknown Results state for {task_id} {lane} {uuid}."
                    )
                if result["state"] == "ready" and (
                    result["recorded"] != result["expected"]
                    or result["expected"] <= 0
                    or not result.get("evidence_id")
                ):
                    raise ValueError(
                        f"Incomplete admitted Results for {task_id} {lane} {uuid}."
                    )
                if result["state"] == "pending" and not result.get("reason"):
                    raise ValueError(
                        f"Pending Results have no reason for {task_id} {lane} {uuid}."
                    )
        tasks[task_id] = ReleasedTask(
            sweep=root / record["sweep"],
            views=views,
            results=availability,
        )
    return manifest, tasks


def _protocol(root: Path, current: Path | None) -> bytes:
    """The protocol document a build of the tree at *root* reads."""
    return composed_protocol(
        root / BENCHMARK_INFO_DIRECTORY / BENCHMARK_PROTOCOL_NAME, current
    )


def _local_release(root: Path, current: Path | None) -> ReleaseInput:
    """Read one local Distribution Release tree as this build's preview.

    A local tree is rendered as a preview whether or not its bytes were also
    published, because nothing in the tree says which, and a page never claims
    a release it cannot identify.
    """
    manifest, tasks = _released_tasks(root)
    return ReleaseInput(
        root=root,
        version=manifest["version"],
        manifest_id=manifest["manifest_id"],
        identity=f"Benchmark {manifest['version']}, Results {manifest['manifest_id'][:12]}",
        revision="",
        published=False,
        tasks=tasks,
        protocol=_protocol(root, current),
    )


def _pinned_release(
    pin: BenchmarkPin, cache_dir: Path, current: Path | None
) -> ReleaseInput:
    """Install the latest revision of the version the lock pins, and read it.

    The repository's latest commit is resolved once and installed by that
    commit, which binds every byte; the pages name it, so a built page stays
    reproducible. A release naming another benchmark version is refused: a new
    version is a deliberate edit of the lock, a Results update is not.
    """
    revision = _latest_revision(pin.repo_id)
    root = _installed(pin.repo_id, revision, cache_dir)
    manifest, tasks = _released_tasks(root)
    if manifest["version"] != pin.version:
        raise ValueError(
            f"{pin.repo_id} at {revision} is benchmark version "
            f"{manifest['version']!r}, but the committed lock pins version "
            f"{pin.version!r}; name the new version in the lock to render it."
        )
    return ReleaseInput(
        root=root,
        version=manifest["version"],
        manifest_id=manifest["manifest_id"],
        identity=(
            f"Benchmark {manifest['version']}, Results {manifest['manifest_id'][:12]}, "
            f"{pin.repo_id} at {revision}"
        ),
        revision=revision,
        published=True,
        tasks=tasks,
        protocol=_protocol(root, current),
    )


def _latest_revision(repo_id: str) -> str:
    """The commit the dataset repository's default branch holds now."""
    info = huggingface_hub.HfApi(token=False).repo_info(
        repo_id=repo_id, repo_type="dataset"
    )
    revision = info.sha
    if not is_immutable_revision(revision):
        raise ValueError(f"{repo_id} reported no commit for its latest revision.")
    return revision


def _installed(repo_id: str, revision: str, cache_dir: Path) -> Path:
    """The installed tree of one revision, fetching it once.

    The revision is an immutable commit, so it names the bytes: an install
    already under that name is that release and is reused rather than fetched
    again. A fetch lands in a sibling directory and is renamed into place, so
    an interrupted one is never read as a release.
    """
    root = cache_dir / revision
    if (root / RELEASE_MANIFEST_NAME).is_file():
        return root
    staging = root.with_name(f"{root.name}.staging-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        _fetch_release_tree(repo_id, revision=revision, destination=staging)
        if (root / RELEASE_MANIFEST_NAME).is_file():
            shutil.rmtree(staging, ignore_errors=True)
        else:
            staging.rename(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root
