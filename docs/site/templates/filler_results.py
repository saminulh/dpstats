"""An invented results store, prepared and rendered exactly as a real one is.

Nothing here is real: the competitors, the budgets, the seeds, the instances
and the thresholds are filler, and every number the pages show is computed from
them. The roster runs through the library's own sweep, the store runs through
``prepare_results``, and the snapshot runs through the live Home, Leaderboard
and Explore renderers, so the pages under ``/templates/`` are those views with
fake data rather than a second implementation of them.

The snapshot is derived, not authored: it is generated from the seeds below and
is ignored on disk, so only this module is checked in.

Run ``python docs/site/templates/filler_results.py`` to write it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

SITE_DIR = Path(__file__).resolve().parents[1]
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))

from dpstats.algorithm import (  # noqa: E402
    BaseAlgorithm,
    PrivacyInterpretation,
    register_algorithm,
)
from dpstats.budgets import DPBudget  # noqa: E402
from dpstats.domains import BoxDomain  # noqa: E402
from dpstats.instance import (  # noqa: E402
    persist_instance,
    register_instance,
)
from dpstats.tasks import MeanEstimation  # noqa: E402
from dpstats.tasks.mean_estimation import (  # noqa: E402
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.analysis import (  # noqa: E402
    compute_profile_curves,
    measure_results,
    compute_error_ratio,
)
from dpstats.workspace import init_workspace, open_collection  # noqa: E402
from dpstats.tasks import list_built_in_metrics  # noqa: E402
from dpstats.transport.canonical import (  # noqa: E402
    canonical_json_bytes,
)
from dpstats.workspace.layout import (  # noqa: E402
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
    INPUT_ARTIFACTS_DIRECTORY,
    PROVENANCE_DIRECTORY,
    RELEASE_MANIFEST_NAME,
    SWEEPS_DIRECTORY,
    task_collection_directory,
    workspace_local_root,
)

from prepare_results import prepared_snapshot  # noqa: E402
from results_roots import (  # noqa: E402
    INSTANCE_WEIGHTS_ROLE,
    WITHHELD_PROFILES_ROLE,
    release_input,
)
import release_files  # noqa: E402
from release_files import (  # noqa: E402
    instance_weights_relative,
    protocol_entry,
    seal_released_results,
    withheld_profiles_relative,
    write_narrative,
    write_protocol,
    write_release_manifest,
    write_sweep,
    write_profile_view,
    write_weights_view,
)

# The task the filler roster is written against, and the invented name the
# preview shows for it. The library measures results through a declared Task,
# so the carrier is a real one; the roster it carries, its instances, its name
# and every number on the mock pages are invented.
FILLER_TASK = MeanEstimation
FILLER_TASK_NAME = "Filler Task"

# Where the generated snapshot lives, and what the build and the tests read.
# The stamp records the seeds it was generated from, so an edited roster is
# regenerated rather than previewed from the roster it replaced.
SNAPSHOT_DIRNAME = "filler-results"
SNAPSHOT_DIR = Path(__file__).resolve().parent / SNAPSHOT_DIRNAME
STAMP_NAME = "seeds.sha256"
GENERATOR_COMMAND = "python docs/site/templates/filler_results.py"

# The registry group the invented roster and instances are registered in, and
# the release as the preview's build-provenance line names it.
LABEL = "filler"
IDENTITY = "an invented Distribution Release: nothing in it is real"

# The invented external baseline: a stored value every filler instance
# carries, which the protocol declares for both Metrics and reads the
# headline against. The views were taken against it.
VIEW_BASELINE = "filler_scale"


class MissingSnapshot(FileNotFoundError):
    """No generated filler snapshot matches the seeds this module declares."""


# The invented roster: each competitor's label and the cost it pays before the
# budget, the instance and the seed scale it. The costs rise geometrically
# across one order, so at the opening threshold the standings have a leading
# group, a middle and a tail, ties that move as the threshold moves, and more
# than ten rows, which is what puts both the rankings table's ten visible rows
# and its route to the rest of the pool on the page. The second number is the
# power of the budget a competitor turns into accuracy: they differ, so the
# standings are not the same list under every budget.
COMPETITORS = {
    "Filler Alpha": (0.04, 1.0),
    "Filler Beta": (0.048, 0.5),
    "Filler Gamma": (0.057, 1.0),
    "Filler Delta": (0.068, 0.5),
    "Filler Epsilon": (0.081, 1.0),
    "Filler Zeta": (0.097, 0.5),
    "Filler Eta": (0.116, 1.0),
    "Filler Theta": (0.138, 0.5),
    "Filler Iota": (0.165, 1.0),
    "Filler Kappa": (0.197, 0.5),
    "Filler Lambda": (0.235, 1.0),
    "Filler Mu": (0.28, 0.5),
}

# The competitor whose privacy claim is read at a weaker interpretation, and the
# ranked one it otherwise matches. The two draw one curve and differ only in the
# class their claim is read at, so the Leaderboard can be seen to exclude it and
# Explore to reveal it beside its twin.
QUALIFIED_COMPETITOR = "Filler Nu"
QUALIFIED_TWIN = "Filler Beta"

# The nonprivate stand-in every cost is divided by. It is an analysis baseline,
# never a ranked entry.
REFERENCE_COMPETITOR = "Filler Reference"
REFERENCE_COST = 0.5

# The budgets and algorithm seeds the filler sweep declares. Four seeds
# put a share on the quarters, so a curve is a staircase rather than a step.
BUDGETS = (1.0, 2.0)
SEEDS = (0, 1, 2, 3)

# The grid the threshold slider spans, where the standings open on it, and the
# Metrics the invented release carries, headline first.
TAU_GRID = {"min": 0.0, "max": 10.0, "scale": "linear", "steps": 21}
HEADLINE_QUANTILE = 0.9
METRICS = ("l2_error_to_ground_truth", "l2_error_to_sample_mean")

# Each filler instance: its sample count, its dimension and the provenance
# bundle it declares. The three cut the set three different ways, so an
# instance filter on Explore visibly changes a curve and the weighting the
# protocol gives each provenance group is visible in the standings.
PROVENANCE_REFS = (
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/filler-bundle-one.json",
    f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/filler-bundle-two.json",
)
INSTANCES = {
    "filler-narrow-small": (2, 1, PROVENANCE_REFS[0]),
    "filler-wide-small": (2, 4, PROVENANCE_REFS[1]),
    "filler-narrow-large": (20, 1, PROVENANCE_REFS[1]),
    "filler-wide-large": (20, 4, PROVENANCE_REFS[0]),
}

_BASE_SAMPLES = 2
_SWEEP_NAME = "filler.json"


class FillerReferenceMean(BaseAlgorithm):
    """Nonprivate stand-in spending one cost on every instance."""

    TASK = MeanEstimation

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        estimate = np.zeros(metadata.dim)
        estimate[0] = REFERENCE_COST
        return MeanEstimationOutput(mean_est=estimate)


class FillerMean(BaseAlgorithm):
    """Ranked stand-in whose cost steps on the instance, seed and budget.

    ``decay`` is the power of the budget this stand-in converts into
    accuracy, so two competitors at the same cost separate as it rises.
    """

    TASK = MeanEstimation
    cost: float
    decay: float

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        epsilon = getattr(budget, "eps", 1.0)
        scale = (1.0 + seed) * (metadata.n_samples / _BASE_SAMPLES) * metadata.dim
        estimate = np.zeros(metadata.dim)
        estimate[0] = self.cost * scale / epsilon**self.decay
        return MeanEstimationOutput(mean_est=estimate)


class QualifiedFillerMean(FillerMean):
    """The same stand-in, read at a qualified interpretation of its claim."""

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.QUALIFIED


def _write_store(root: Path) -> Path:
    """Register the filler roster and its instances, then run its sweep.

    Returns:
        The installed recipe.
    """
    workspace = init_workspace(root / "workspace")
    collections = workspace_local_root(workspace)
    artifacts = collections / INPUT_ARTIFACTS_DIRECTORY
    task_root = task_collection_directory(collections, FILLER_TASK.id)
    task_root.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    register_algorithm(
        collections,
        FillerReferenceMean(label=REFERENCE_COMPETITOR),
        registry=LABEL,
    )
    for label, (cost, decay) in COMPETITORS.items():
        register_algorithm(
            collections,
            FillerMean(label=label, cost=cost, decay=decay),
            registry=LABEL,
        )
    register_algorithm(
        collections,
        QualifiedFillerMean(
            label=QUALIFIED_COMPETITOR,
            cost=COMPETITORS[QUALIFIED_TWIN][0],
            decay=COMPETITORS[QUALIFIED_TWIN][1],
        ),
        registry=LABEL,
    )

    named: dict[str, dict[str, Any]] = {ref: {} for ref in PROVENANCE_REFS}
    for label, (samples, dim, ref) in INSTANCES.items():
        metadata = MeanEstimationMetadata(
            n_samples=samples,
            domain=BoxDomain(bounds=np.tile(np.array([-100.0, 100.0]), (dim, 1))),
        )
        instance = persist_instance(
            metadata,
            MeanEstimationInput(X=np.zeros((samples, dim))),
            {
                "ground_truth": [0.0] * dim,
                "sample_mean": [0.0] * dim,
                VIEW_BASELINE: REFERENCE_COST,
            },
            artifacts,
            label=label,
        )
        register_instance(collections, instance, registry=LABEL)
        named[ref][instance.uuid] = {"sample_unit": "row", "source": {}}
    for ref in PROVENANCE_REFS:
        bundle = root / ref
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(
            json.dumps(
                {
                    "source": "Filler provenance bundle; nothing here is real.",
                    "data_license": "CC0-1.0",
                    "discipline": "filler",
                    "instances": named[ref],
                }
            ),
            encoding="utf-8",
        )

    sweep = write_sweep(
        task_root / SWEEPS_DIRECTORY / _SWEEP_NAME, FILLER_TASK.id, BUDGETS, SEEDS
    )
    open_collection(workspace, FILLER_TASK).run_sweep(sweep)
    shutil.copytree(task_root, root / FILLER_TASK.id)
    shutil.copytree(artifacts, root / INPUT_ARTIFACTS_DIRECTORY)
    shutil.rmtree(workspace)
    released_sweep = root / sweep.relative_to(collections)
    seal_released_results(root, FILLER_TASK.id, released_sweep)
    return released_sweep


def _taus() -> tuple[float, ...]:
    """The thresholds the invented grid names, ascending."""
    span = (TAU_GRID["max"] - TAU_GRID["min"]) / (TAU_GRID["steps"] - 1)
    return tuple(TAU_GRID["min"] + index * span for index in range(TAU_GRID["steps"]))


def _measured(root: Path, sweep: Path):
    """Measure the invented store's every run, as the library measures one."""
    collection = open_collection(root, FILLER_TASK)
    settings = collection.load_sweep(sweep)
    defined = list_built_in_metrics(FILLER_TASK)
    frame = measure_results(
        collection.load_results(settings),
        [defined[name] for name in METRICS],
        collection=collection,
    )
    # Both views name the recipe's budget group, never a run's resolved budget:
    # a run under no declared budget belongs to no panel.
    frame = frame[frame["budget_group"].notna()].copy()
    frame["group"] = frame["budget_group"].map(repr)
    return frame


def _write_views(root: Path, frame) -> dict[str, str]:
    """Publish the two sanctioned views the invented release ships.

    The denominator is the invented scale every instance stores, the
    nonprivate stand-in's own cost, and the instances weigh by the provenance
    bundle they declare. The curves are this store's own profile under both, taken through
    the library's analysis, so the preview's two surfaces report one number.
    """
    instances = list(dict.fromkeys(frame["instance"]))
    by_ref: dict[str, list[str]] = {}
    refs = dict(zip(INSTANCES, (ref for _, _, ref in INSTANCES.values())))
    for instance in instances:
        by_ref.setdefault(refs[str(instance.label)], []).append(instance.uuid)
    weights = {
        uuid: 1.0 / (len(by_ref) * len(members))
        for members in by_ref.values()
        for uuid in members
    }
    write_weights_view(
        root,
        FILLER_TASK.id,
        {
            instance.uuid: weights[instance.uuid]
            for instance in sorted(instances, key=lambda value: value.uuid)
        },
    )

    shares = pd.DataFrame(
        [
            {"instance": instance, "weight": weights[instance.uuid]}
            for instance in instances
        ],
        columns=["instance", "weight"],
    )
    reference = (
        frame[["task", "instance"]].drop_duplicates().assign(baseline=REFERENCE_COST)
    )
    counted = str(len(SEEDS) * len(INSTANCES))
    curves: list[tuple[str, ...]] = []
    for metric in METRICS:
        groups: list[str] = list(dict.fromkeys(frame["group"]))
        for budget in groups:
            traced = compute_profile_curves(
                compute_error_ratio(
                    frame[frame["group"] == budget], reference, metric=metric
                ),
                _taus(),
                metric=metric,
                weights=shares,
            )
            for algorithm, tau, fraction in zip(
                traced["algorithm"], traced["tau"], traced["fraction"]
            ):
                curves.append(
                    (
                        FILLER_TASK.id,
                        str(algorithm),
                        algorithm.uuid,
                        str(budget),
                        metric,
                        VIEW_BASELINE,
                        repr(float(tau)),
                        repr(float(fraction)),
                        counted,
                        counted,
                    )
                )
    profiles = withheld_profiles_relative(FILLER_TASK.id)
    write_profile_view(root / profiles, curves)
    return {
        INSTANCE_WEIGHTS_ROLE: instance_weights_relative(FILLER_TASK.id),
        WITHHELD_PROFILES_ROLE: profiles,
    }


def _write_release(root: Path, sweep: Path) -> None:
    """Write the protocol, the narrative, the views and the release manifest."""
    # A placeholder manifest first: open_collection recognizes a bare bundle
    # only by release.json's presence, and _measured below opens one to
    # compute the views the real manifest's own file digests name. The final
    # write at the end of this function replaces it with the real content.
    (root / RELEASE_MANIFEST_NAME).write_bytes(
        canonical_json_bytes(
            {
                "version": LABEL,
                "manifest_id": "f" * 64,
                "tasks": {FILLER_TASK.id: {}},
                "files": {},
            }
        )
    )
    write_protocol(
        root,
        {
            FILLER_TASK.id: protocol_entry(
                METRICS[0],
                headline_baseline=VIEW_BASELINE,
                headline_quantile=HEADLINE_QUANTILE,
                tau_grid=TAU_GRID,
                external={
                    VIEW_BASELINE: {
                        "key": VIEW_BASELINE,
                        "description": "Filler. An invented scale every filler "
                        "instance stores; nothing here is real.",
                        "metrics": list(METRICS),
                    }
                },
            )
        },
    )
    write_narrative(
        root,
        "# Benchmark protocol\n\nFiller. This invented release compares one "
        "invented roster on one task, at the thresholds and against the "
        "denominators its own views carry. No choice stated here is real.\n",
    )
    views = _write_views(root, _measured(root, sweep))
    write_release_manifest(
        root,
        {FILLER_TASK.id: {"sweep": sweep.relative_to(root).as_posix()}},
        {
            f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_PROTOCOL_NAME}": "f" * 64,
            f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_NARRATIVE_NAME}": "f" * 64,
            sweep.relative_to(root).as_posix(): "f" * 64,
            **{path: "f" * 64 for path in views.values()},
        },
        version=LABEL,
    )


def build_snapshot(destination: Path) -> Path:
    """Generate the filler store from the seeds above and prepare it.

    Args:
        destination: Where the prepared snapshot is written. An existing
            directory is replaced, because the snapshot is derived.

    Returns:
        The emitted selection file the build and the tests read.
    """
    destination = destination.resolve()
    if destination.exists():
        if any(destination.iterdir()) and not (destination / STAMP_NAME).is_file():
            raise ValueError(
                f"{destination} holds something this generator did not write; "
                "it is not replaced. Name an empty directory, or remove that "
                "one by hand."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    store = destination / "store"
    store.mkdir()
    _write_release(store, _write_store(store))
    release = release_input(None, str(store), cache_dir=destination / "releases")
    assert release is not None
    emitted = prepared_snapshot(release, cache=destination / "prepared")
    (destination / STAMP_NAME).write_text(
        json.dumps({"seeds": _seed_digest(), "snapshot": emitted.name}),
        encoding="utf-8",
    )
    return emitted


def _seed_digest() -> str:
    """Digest the seeds a snapshot was generated from.

    Those are this module and ``release_files``, which writes the release's
    on-disk shape, so a change to either marks an earlier snapshot stale.
    """
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(Path(release_files.__file__).read_bytes())
    return digest.hexdigest()


def snapshot(directory: Path | None = None) -> Path:
    """Return the filler snapshot's selection file, refusing a stale one.

    Preparing and rendering stay separate, so this reads what the generator
    wrote and measures nothing. A snapshot left from another roster carries
    competitors the current roster does not declare, so the stamp is checked rather
    than the file's presence, and a build wanting the previews runs the
    generator first.

    Args:
        directory: Where the snapshot lives, or ``None`` for :data:`SNAPSHOT_DIR`.

    Returns:
        The prepared filler snapshot the preview renders.

    Raises:
        MissingSnapshot: No snapshot is there, or it was generated from seeds
            this module does not declare.
    """
    root = SNAPSHOT_DIR if directory is None else directory
    stamp = root / STAMP_NAME
    if not stamp.is_file():
        raise MissingSnapshot(
            f"No filler snapshot at {root}. Generate it with {GENERATOR_COMMAND}."
        )
    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    emitted = root / "prepared" / recorded["snapshot"]
    if recorded["seeds"] != _seed_digest() or not emitted.is_dir():
        raise MissingSnapshot(
            f"The filler snapshot at {root} was generated from seeds this "
            f"module does not declare. Generate it again with "
            f"{GENERATOR_COMMAND}."
        )
    return emitted


# Each results page under ``/templates/``: its template, its own directory, the
# surface it renders, and the route its data is named from. A Home preview names
# the files it fetches from beside itself, exactly as the site root's Home names
# them from beside the root, so it keeps the surface's own route name inside its
# directory; the two full pages name theirs from their own.
RESULTS_PAGES = (
    ("T1", "t01-home", "leaderboard", "leaderboard"),
    ("T4", "t04-leaderboard", "leaderboard", "t04-leaderboard"),
    ("T5", "t05-explore", "public-results", "t05-explore"),
)

# How one rendered results page is read back into the sections its template
# declares: the marker each section emits, in the order the page carries them.
# The spec says what a page contains; this table is the only way a rendered
# page states it back, so a renderer that stops emitting a section fails
# against the spec rather than leaving the spec describing a page that is gone.
SECTION_MARKERS: dict[str, tuple[tuple[str, str], ...]] = {
    "T1": (
        ("tabs", "data-home-tabs"),
        ("panel_heading", 'class="dp-home-task-heading"'),
        ("source", 'class="dp-home-source"'),
        ("controls", 'class="dp-filters dp-comparison-controls'),
        ("absences", 'class="dp-note dp-absences"'),
        ("metric", 'class="dp-metric"'),
        ("budget", "data-budget-select"),
        ("chart", 'class="dp-profile-heading dp-authored-copy'),
        ("plotted", 'class="dp-plotted-heading"'),
        ("rankings", 'class="dp-rankings-heading"'),
        ("build_source", 'data-copy-slot="shared.built_from"'),
    ),
    "T4": (
        ("introduction", 'class="dp-protocol-pointer'),
        ("controls", 'class="dp-filters dp-comparison-controls'),
        ("absences", 'class="dp-note dp-absences"'),
        ("metric", 'class="dp-metric"'),
        ("budget", "data-budget-select"),
        ("chart", 'class="dp-profile-heading dp-authored-copy'),
        ("plotted", 'class="dp-plotted-heading"'),
        ("rankings", 'class="dp-rankings-heading"'),
        ("build_source", 'data-copy-slot="shared.built_from"'),
    ),
    "T5": (
        ("loading", "data-explore-loading"),
        ("controls", 'class="dp-filters dp-comparison-controls'),
        ("absences", 'class="dp-note dp-absences"'),
        ("empty", 'class="dp-empty" role="status"'),
        ("metric", 'class="dp-metric"'),
        ("budget", "data-budget-select"),
        ("chart", 'class="dp-profile-heading dp-authored-copy'),
        ("plotted", 'class="dp-plotted-heading"'),
        ("rankings", 'class="dp-rankings-heading"'),
        ("comparisons", 'class="dp-comparison-plot"'),
    ),
}


def rendered_sections(name: str, page: str) -> tuple[str, ...]:
    """Read one rendered results page back as the sections it carries.

    Args:
        name: The template the page is an instance of.
        page: The page's rendered HTML.

    Returns:
        Each section the page emits, in the order it first emits it. A section
        the page does not carry is left out rather than reported empty.
    """
    found = [
        (page.find(marker), key) for key, marker in SECTION_MARKERS[name] if marker
    ]
    return tuple(key for position, key in sorted(found) if position >= 0)


if __name__ == "__main__":
    # Running this file defines its Algorithms in ``__main__``, which no other
    # process can import and the registry therefore refuses. Generate through
    # the imported module instead, so each registered class names its real home.
    from templates import filler_results

    print(filler_results.build_snapshot(filler_results.SNAPSHOT_DIR))
