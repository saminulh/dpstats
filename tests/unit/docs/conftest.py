"""Shared fixtures for the documentation-assembly tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest

SITE_DIR = Path(__file__).resolve().parents[3] / "docs" / "site"
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))

from panel_data import TaskData, released_weights, task_data  # noqa: E402
from results_roots import (  # noqa: E402
    BEST_MEDIAN,
    ExternalBaseline,
    ReleasedBaseline,
    ReleasedComparison,
    ReleasedGrid,
)
from release_files import (  # noqa: E402
    instance_weights_relative,
    protocol_entry,
    seal_released_results,
    sweep_relative,
    withheld_profiles_relative,
    write_narrative,
    write_protocol,
    write_release_manifest,
    write_sweep,
    write_profile_view,
    write_weights_view,
)

from dpstats.algorithm import register_algorithm
from dpstats.instance import Instance, persist_instance, register_instance
from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import BoxDomain
from dpstats.parameters import (
    parameter_field,
    search_range,
    search_values,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    BENCHMARK_NARRATIVE_NAME,
    BENCHMARK_PROTOCOL_NAME,
    INPUT_ARTIFACTS_DIRECTORY,
    PROVENANCE_DIRECTORY,
    task_collection_directory,
    workspace_local_root,
)
from dpstats.workspace import init_workspace, open_collection

DIGEST = "d" * 64

# The narrative every fixture release ships.
FIXTURE_NARRATIVE = "# Benchmark protocol\n\nFixture.\n"

# The Metrics the measurable fixture release carries, headline first.
MEASURED_METRIC = "l2_error_to_ground_truth"
SECOND_METRIC = "l2_error_to_sample_mean"
MAHALANOBIS_METRIC = "mahalanobis_error_to_sample_mean"
MAHALANOBIS_REFERENCE = "sample_mean_mahalanobis_se"
RELEASED_METRICS = (MEASURED_METRIC, SECOND_METRIC, MAHALANOBIS_METRIC)

# What the fixture views call the baseline they were taken against. The site
# shows this name; it never checks it, so it is the view's own word.
VIEW_BASELINE = "best_median"

# A library reference of the second Metric that the fixture Instances store,
# so a page offers reading that Metric's costs against it.
STORED_REFERENCE = "sample_mean_l2_se"

# A value the fixture Instances store that the fixture protocol declares an
# external baseline for the headline Metric.
EXTERNAL_REFERENCE = "fixture_scale"
EXTERNAL_DESCRIPTION = "The fixture's invented scale, as its source supplies it."

# A scalar the fixture Instances store that the protocol declares no baseline
# for. A stored value is not a scale to divide by merely because it is a
# number, so this one is never offered.
UNDECLARED_REFERENCE = "reference_estimate"

WriteReleaseRoot = Callable[..., Path]

# The thresholds every fixture curve publishes, as the producer writes them.
# All three are on the released grid, and they bracket the ruled threshold so a
# page's default position is always one of them.
FIXTURE_TAUS = ("1", "10", "100")

# One task's fixture curves: budget, algorithm, its published identifier, the
# fraction at each threshold above, and the runs behind the curve. Two budgets
# so a page can never pool them, three algorithms so a ranking has a middle,
# and, in each budget, one exact tie that moves as the threshold moves.
FIXTURE_CURVES = (
    ("eps=1.0", "Alg Alpha", "aaaa-1111", ("0.25", "0.75", "1"), 8, 8),
    ("eps=1.0", "Alg Beta", "bbbb-2222", ("0.5", "0.75", "0.875"), 8, 6),
    ("eps=1.0", "Alg Gamma", "cccc-3333", ("0.125", "0.5", "1"), 8, 8),
    ("eps=4.0", "Alg Alpha", "aaaa-1111", ("0.5", "0.875", "1"), 8, 8),
    ("eps=4.0", "Alg Beta", "bbbb-2222", ("0.25", "0.625", "1"), 8, 8),
    ("eps=4.0", "Alg Gamma", "cccc-3333", ("0.75", "0.75", "1"), 8, 8),
)

# The Instances the fixture weights view carries where the release carries
# no measurable store of its own.
FIXTURE_WEIGHTED_INSTANCES = ("i5-fixture-one", "i5-fixture-two")


def comparison_entry(
    headline: str,
    *,
    headline_quantile: float = 0.9,
    steps: int = 1001,
    maximum: float = 10.0,
    baseline: str = VIEW_BASELINE,
) -> dict[str, Any]:
    """One released task's protocol entry, as the document states it.

    It declares the fixture's external baseline for the headline Metric.
    """
    return protocol_entry(
        headline,
        headline_baseline=baseline,
        headline_quantile=headline_quantile,
        tau_grid={"min": 0.0, "max": maximum, "scale": "linear", "steps": steps},
        external={
            EXTERNAL_REFERENCE: {
                "key": EXTERNAL_REFERENCE,
                "description": EXTERNAL_DESCRIPTION,
                "metrics": [headline],
            }
        },
    )


def write_view(
    root: Path,
    relative: str,
    metrics: Sequence[str],
    task_id: str,
    baseline: str = VIEW_BASELINE,
) -> Path:
    """Write one profile view publishing ``FIXTURE_CURVES`` for each of *metrics*.

    Metrics keep the order given, repeats included, so a caller can exercise
    how a reader of the view treats them. Every row names the baseline the
    release took it against, which the page shows as the view states it.
    """
    return write_profile_view(
        root / relative,
        (
            (
                task_id,
                algorithm,
                uuid,
                budget,
                metric,
                baseline,
                tau,
                fraction,
                str(runs),
                str(measured),
            )
            for metric in metrics
            for budget, algorithm, uuid, fractions, runs, measured in FIXTURE_CURVES
            for tau, fraction in zip(FIXTURE_TAUS, fractions)
        ),
    )


def write_fixture_weights(
    root: Path,
    task_id: str,
    instances: Sequence[str] = FIXTURE_WEIGHTED_INSTANCES,
) -> Path:
    """Write one task's Instance weights view, every Instance weighing alike."""
    return write_weights_view(
        root, task_id, {instance: 1.0 / len(instances) for instance in instances}
    )


def release_task(task_id: str) -> dict[str, Any]:
    """One task's manifest record: its recipe."""
    return {"sweep": sweep_relative(task_id)}


@pytest.fixture(scope="session")
def placeholder_copy_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A site-copy file with every slot unwritten, whatever the maintainer's says."""
    import json

    from site_copy import PLACEHOLDER, SITE_COPY_KEYS

    path = tmp_path_factory.mktemp("site-copy") / "site-copy.yml"
    path.write_text(
        json.dumps(
            {
                group: {key: PLACEHOLDER for key in sorted(keys)}
                for group, keys in SITE_COPY_KEYS.items()
            }
        )
    )
    return path


@pytest.fixture(autouse=True)
def placeholder_site_copy(
    placeholder_copy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Render every page from the all-placeholder copy, so filled text breaks nothing."""
    import site_copy

    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", placeholder_copy_file)
    return placeholder_copy_file


WriteCopy = Callable[[Mapping[str, Mapping[str, str]]], None]


@pytest.fixture
def write_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placeholder_copy_file: Path
) -> WriteCopy:
    """Write some slots, the rest unwritten, so a test can read what fills them."""
    import json

    import site_copy

    def write(written: Mapping[str, Mapping[str, str]]) -> None:
        groups = json.loads(placeholder_copy_file.read_text())
        for group, entries in written.items():
            groups[group].update(entries)
        path = tmp_path / "written-site-copy.yml"
        path.write_text(json.dumps(groups))
        monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)

    return write


@pytest.fixture
def write_release_root() -> WriteReleaseRoot:
    """Return a factory writing one Distribution-Release-shaped tree.

    The factory writes a readable release manifest and the view bytes the
    manifest declares. Its keyword arguments are the states a task can be in:
    shipping both sanctioned views, declaring them without carrying their
    bytes, and declaring only one of the two roles.
    """

    def write(
        root: Path,
        *,
        published: Mapping[str, Sequence[str]] | None = None,
        unwritten: Sequence[str] = (),
        one_role: Sequence[str] = (),
        version: str = "simulated-v1",
        headline: Mapping[str, str] | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        tasks: dict[str, dict[str, Any]] = {}
        files: dict[str, str] = {
            f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_PROTOCOL_NAME}": DIGEST
        }
        comparison: dict[str, dict[str, Any]] = {}
        ruled = dict(headline or {})
        for task_id, metrics in (published or {}).items():
            profiles = withheld_profiles_relative(task_id)
            weights = instance_weights_relative(task_id)
            write_view(root, profiles, metrics, task_id)
            write_fixture_weights(root, task_id)
            tasks[task_id] = release_task(task_id)
            files.update({profiles: DIGEST, weights: DIGEST})
            comparison[task_id] = comparison_entry(ruled.get(task_id, metrics[0]))
        for task_id in unwritten:
            profiles = withheld_profiles_relative(task_id)
            weights = instance_weights_relative(task_id)
            tasks[task_id] = release_task(task_id)
            files.update({profiles: DIGEST, weights: DIGEST})
            comparison[task_id] = comparison_entry(MEASURED_METRIC)
        for task_id in one_role:
            profiles = withheld_profiles_relative(task_id)
            write_view(root, profiles, [MEASURED_METRIC], task_id)
            tasks[task_id] = release_task(task_id)
            files[profiles] = DIGEST
            comparison[task_id] = comparison_entry(MEASURED_METRIC)
        for task_id in tasks:
            write_fixture_sweep(root / sweep_relative(task_id), task_id)
            files[sweep_relative(task_id)] = DIGEST
        write_protocol(root, comparison)
        write_narrative(root, FIXTURE_NARRATIVE)
        files[f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_NARRATIVE_NAME}"] = DIGEST
        write_release_manifest(root, tasks, files, version=version)
        return root

    return write


# ── A release whose Results a playground can actually measure ────────────────

# The deterministic stand-in roster. A ranked algorithm's released cost is its
# own ``error`` stepped by one decade per algorithm seed, per unit of Instance
# size, and against the budget it was offered, so the share a filtered
# subset keeps within a threshold can be written down by hand. The released
# denominator is one everywhere, so an error ratio is the released one.
REFERENCE_ERROR = 0.5
# Two of these share an error, and so a curve, but differ in the class
# their privacy claim is read at, so a page can be seen to slice on it.
FIXTURE_ALGORITHMS = {"Alg Alpha": 0.05, "Alg Beta": 0.5, "Alg Gamma": 5.0}
QUALIFIED_ALGORITHM = "Alg Delta"
FIXTURE_EPSILONS = (1.0, 10.0)
FIXTURE_SEEDS = (0, 1)
_BASE_SAMPLES = 2

REF_ALPHA = f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/alpha-bundle.json"
REF_BETA = f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/beta-bundle.json"

# Each fixture Instance: its sample count, its dimension, and the provenance
# bundle it declares. The refs, the sample-count decades, and the dimension
# decades cut the four Instances three different ways, so a filter that reads
# the wrong one draws a visibly different curve.
FIXTURE_INSTANCES = {
    "narrow-small": (2, 1, REF_ALPHA),
    "wide-small": (2, 10, REF_BETA),
    "narrow-large": (20, 1, REF_BETA),
    "wide-large": (20, 10, REF_ALPHA),
}

# What the measurable fixture release adds beyond the ranked roster: one
# informal comparison curve and one algorithm that refuses its smallest
# Instances.
INFORMAL_ALGORITHM = "Alg Informal"
FAILING_ALGORITHM = "Alg Failing"


class ReferenceMean(BaseAlgorithm):
    """Nonprivate reference spending one cost on every Instance."""

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
        estimate[0] = REFERENCE_ERROR
        return MeanEstimationOutput(mean_est=estimate)


class StepMean(BaseAlgorithm):
    """Ranked stand-in whose cost steps by a decade on each sliced axis.

    Reading the budget's epsilon with a default is what keeps the step
    total across budget families; this roster declares pure-DP budgets
    only.
    """

    TASK = MeanEstimation
    error: float

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
        scale = 10.0**seed * (metadata.n_samples / _BASE_SAMPLES) * metadata.dim
        estimate = np.zeros(metadata.dim)
        estimate[0] = self.error * scale / epsilon
        return MeanEstimationOutput(mean_est=estimate)


class QualifiedStepMean(StepMean):
    """The same stand-in, read at a weaker interpretation of its claim."""

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.QUALIFIED


class InformalStepMean(StepMean):
    """A stand-in whose privacy claim is read informally.

    An informal claim is a comparison rather than a standing: its curve is
    drawn dashed and it is ranked nowhere, exactly as a registered nonprivate
    reference would be if it ran under a private budget.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.INFORMAL


class FailingMean(BaseAlgorithm):
    """A stand-in outside its public regime on the smallest Instances.

    An invalid-regime run is an ordinary row at infinite cost until a reader
    explicitly asks to compare algorithms within this stand-in's regime.
    """

    TASK = MeanEstimation

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
        if metadata.n_samples == _BASE_SAMPLES:
            raise InvalidRegime("FailingMean requires n_samples > 2.")
        estimate = np.zeros(metadata.dim)
        estimate[0] = REFERENCE_ERROR
        return MeanEstimationOutput(mean_est=estimate)


class TunedMean(BaseAlgorithm):
    """Probe algorithm whose configuration exercises every table column.

    It declares a bounded range search, a choice set and a required field, and
    cites two works of the library's bibliography.

    Reference:
        - balle2018improving: Theorem 8 and
          Algorithm 1.
        - alabi2022differentially: Algorithm 3.
    """

    TASK = MeanEstimation

    clip_norm: float = parameter_field(
        default=1.0,
        search=search_range(0.1, 10.0, scale="log"),
        gt=0.0,
        description="Per-example clipping norm.",
    )
    noise: Literal["gaussian", "laplace"] = parameter_field(
        default="gaussian",
        search=search_values("gaussian", "laplace"),
        description="Which noise distribution to add.",
    )
    threshold: float = parameter_field(default=..., description="Required cutoff.")

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
        return MeanEstimationOutput(mean_est=input.X.mean(axis=0))


def budget_labels() -> tuple[str, ...]:
    """The released labels of the budgets the fixture sweep declares."""
    return tuple(repr(PureDPBudget(eps=eps)) for eps in FIXTURE_EPSILONS)


def write_fixture_sweep(path: Path, task_id: str) -> Path:
    """Write the fixture recipe: every fixture budget and algorithm seed."""
    return write_sweep(path, task_id, FIXTURE_EPSILONS, FIXTURE_SEEDS)


def _write_measurable_store(root: Path) -> list[Instance]:
    """Register the fixture roster and run its sweep into the store."""
    workspace = init_workspace(root / "workspace")
    task_collections_root = workspace_local_root(workspace)
    input_artifacts_root = task_collections_root / INPUT_ARTIFACTS_DIRECTORY
    task_root = task_collection_directory(task_collections_root, MeanEstimation.id)
    task_root.mkdir(parents=True)
    input_artifacts_root.mkdir(parents=True)

    register_algorithm(task_collections_root, ReferenceMean(), registry="benchmark")
    for label, error in FIXTURE_ALGORITHMS.items():
        register_algorithm(
            task_collections_root,
            StepMean(label=label, error=error),
            registry="benchmark",
        )
    register_algorithm(
        task_collections_root,
        QualifiedStepMean(
            label=QUALIFIED_ALGORITHM, error=FIXTURE_ALGORITHMS["Alg Beta"]
        ),
        registry="benchmark",
    )
    register_algorithm(
        task_collections_root,
        InformalStepMean(
            label=INFORMAL_ALGORITHM, error=FIXTURE_ALGORITHMS["Alg Alpha"]
        ),
        registry="benchmark",
    )
    register_algorithm(
        task_collections_root,
        FailingMean(label=FAILING_ALGORITHM),
        registry="benchmark",
    )

    bundles: dict[str, dict[str, Any]] = {REF_ALPHA: {}, REF_BETA: {}}
    instances: list[Instance] = []
    for label, (samples, dim, ref) in FIXTURE_INSTANCES.items():
        metadata = MeanEstimationMetadata(
            n_samples=samples,
            domain=BoxDomain(bounds=np.tile(np.array([-100.0, 100.0]), (dim, 1))),
        )
        instance = persist_instance(
            metadata,
            # A Vandermonde design over evenly spaced nodes: full column rank
            # wherever there are enough rows, and worse conditioned the wider
            # it gets, so the fixture Instances carry distinct condition
            # numbers. Sample-mean Metrics derive their references from this
            # Input, despite the deliberately stale stored producer values.
            MeanEstimationInput(
                X=np.vander(np.linspace(0.2, 1.0, samples), dim, increasing=True)
            ),
            {
                "ground_truth": [0.0] * dim,
                "sample_mean": [0.0] * dim,
                # Distinct on every fixture Instance, so a denominator landing
                # in the wrong slot is visible rather than coincidentally equal.
                STORED_REFERENCE: float(10 * samples + dim),
                EXTERNAL_REFERENCE: float(20 * samples + dim),
                UNDECLARED_REFERENCE: float(samples),
            },
            input_artifacts_root,
            label=label,
        )
        instances.append(instance)
        bundles[ref][instance.uuid] = {"sample_unit": "row", "source": {}}
    for instance in instances:
        register_instance(task_collections_root, instance, registry="benchmark")
    for ref, named in bundles.items():
        bundle = root / ref
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(
            json.dumps(
                {"data_license": "CC0-1.0", "discipline": "fixture", "instances": named}
            ),
            encoding="utf-8",
        )

    sweep = write_fixture_sweep(
        task_collections_root / sweep_relative(MeanEstimation.id), MeanEstimation.id
    )
    open_collection(workspace, MeanEstimation).run_sweep(sweep)
    shutil.copytree(task_root, root / MeanEstimation.id)
    shutil.copytree(input_artifacts_root, root / INPUT_ARTIFACTS_DIRECTORY)
    shutil.rmtree(workspace)
    seal_released_results(
        root, MeanEstimation.id, root / sweep.relative_to(task_collections_root)
    )
    return instances


@pytest.fixture(scope="session")
def measurable_release_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Distribution Release whose own Results a page can measure end to end.

    Built once: it registers the deterministic roster, runs its sweep, writes
    the Instance declarations a reader groups by, and ships both sanctioned
    views — the curves the Leaderboard renders as published, and the
    denominators and weights Explore divides its own measurements by.
    """
    root = tmp_path_factory.mktemp("measurable-release")
    instances = _write_measurable_store(root)
    profiles = withheld_profiles_relative(MeanEstimation.id)
    weights = instance_weights_relative(MeanEstimation.id)
    write_view(root, profiles, [MEASURED_METRIC], MeanEstimation.id)
    write_fixture_weights(
        root, MeanEstimation.id, [instance.uuid for instance in instances]
    )
    write_protocol(root, {MeanEstimation.id: comparison_entry(MEASURED_METRIC)})
    write_narrative(root, FIXTURE_NARRATIVE)
    write_release_manifest(
        root,
        {MeanEstimation.id: release_task(MeanEstimation.id)},
        {
            f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_PROTOCOL_NAME}": DIGEST,
            f"{BENCHMARK_INFO_DIRECTORY}/{BENCHMARK_NARRATIVE_NAME}": DIGEST,
            sweep_relative(MeanEstimation.id): DIGEST,
            profiles: DIGEST,
            weights: DIGEST,
        },
        version="measurable-v1",
    )
    return root


# The fields the measurable release's protocol states: the headline Metric,
# its baseline, the threshold its standings open at, its grid, its pooled
# baseline and the external baseline it declares for the headline Metric.
POOLED = ReleasedBaseline(
    BEST_MEDIAN, "best_median", (("pool", "standard"), ("aggregation", "median"))
)
COMPARISON = ReleasedComparison(
    headline_metric=MEASURED_METRIC,
    headline_baseline=VIEW_BASELINE,
    headline_definition=POOLED,
    headline_quantile=0.9,
    taus=tuple(index / 100.0 for index in range(1001)),
    grid=ReleasedGrid(0.0, 10.0, "linear", 1001),
    definitions={BEST_MEDIAN: POOLED},
    external={
        EXTERNAL_REFERENCE: ExternalBaseline(
            EXTERNAL_REFERENCE, EXTERNAL_DESCRIPTION, (MEASURED_METRIC,)
        )
    },
    weighting="provenance_ref",
    eligibility="standard",
)


def measured(
    root: Path,
    *,
    comparison: ReleasedComparison = COMPARISON,
    published: bool = False,
    identity: str = "the fixture release",
) -> TaskData:
    """Measure a measurable release's distribution lane, as a build does."""
    return task_data(
        root,
        MeanEstimation.id,
        comparison=comparison,
        weights=released_weights(root, MeanEstimation.id),
        identity=identity,
        published=published,
        sweep=root / sweep_relative(MeanEstimation.id),
    )


@pytest.fixture(scope="session")
def measured_release(measurable_release_root: Path) -> TaskData:
    """The measurable release's panels, measured once and only read after."""
    return measured(measurable_release_root)


# A built site these checks may read, when a reviewer has one to hand. The
# repository carries no build output, so every whole-build sweep is opt-in.
BUILT_SITE_ENV = "DOCS_BUILT_SITE"


@pytest.fixture
def built_site() -> Path:
    """A built site directory, or a skip when no reviewer supplied one."""
    directory = os.environ.get(BUILT_SITE_ENV)
    if not directory:
        pytest.skip(f"{BUILT_SITE_ENV} does not name a built site.")
    return Path(directory)


# A build made without ``DOCS_SERVE_TEMPLATES``, which is what a published
# build is. Supplied separately because it is a second build of the same head.
PUBLIC_SITE_ENV = "DOCS_PUBLIC_SITE"


@pytest.fixture
def public_site() -> Path:
    """A build made without the page-template flag, or a skip."""
    directory = os.environ.get(PUBLIC_SITE_ENV)
    if not directory:
        pytest.skip(f"{PUBLIC_SITE_ENV} does not name a build.")
    return Path(directory)


# ── Reading rendered markup the way a script or a reader does ────────────────

# Elements HTML never closes, so the reader below never waits for them to end.
_VOID_ELEMENTS = frozenset(
    ("area base br col embed hr img input link meta param source track wbr").split()
)

# Elements whose content a reader never sees as prose.
_UNSEEN_ELEMENTS = frozenset(("script", "style", "template"))


@dataclass(eq=False)
class Element:
    """One parsed element: its tag, its attributes, and what it contains.

    Site tests read markup through its contract with scripts and readers: the
    ``data-*`` hooks the page's JavaScript binds to, accessible roles, and the
    text a reader sees. Class names and attribute order are presentation.
    """

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Element | str] = field(default_factory=list)

    def walk(self) -> Iterator[Element]:
        """This element and every element inside it, in document order."""
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.walk()

    def find_all(
        self,
        hook: str | None = None,
        *,
        role: str | None = None,
        tag: str | None = None,
        **attributes: str,
    ) -> list[Element]:
        """Every element carrying *hook*, *role*, *tag* and each attribute value.

        *hook* is a full attribute name such as ``data-metric``; a keyword
        attribute is matched by value, with underscores read as hyphens.
        """
        wanted = {name.replace("_", "-"): value for name, value in attributes.items()}
        return [
            element
            for element in self.walk()
            if (hook is None or hook in element.attrs)
            and (role is None or element.attrs.get("role") == role)
            and (tag is None or element.tag == tag)
            and all(element.attrs.get(name) == value for name, value in wanted.items())
        ]

    def find(
        self,
        hook: str | None = None,
        *,
        role: str | None = None,
        tag: str | None = None,
        **attributes: str,
    ) -> Element:
        """The first element :meth:`find_all` matches; there must be one."""
        found = self.find_all(hook, role=role, tag=tag, **attributes)
        assert found, f"no element matches {hook=} {role=} {tag=} {attributes}"
        return found[0]

    @property
    def raw_text(self) -> str:
        """Every character inside the element, scripts included, unaltered."""
        return "".join(
            child if isinstance(child, str) else child.raw_text
            for child in self.children
        )

    @property
    def text(self) -> str:
        """The text a reader sees, with whitespace collapsed."""
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag not in _UNSEEN_ELEMENTS:
                parts.append(f" {child.text} ")
        return " ".join("".join(parts).split())

    def body_rows(self) -> list[Element]:
        """The rows of this table's bodies, header rows excluded."""
        return [
            row
            for body in self.find_all(tag="tbody")
            for row in body.find_all(tag="tr")
        ]


class _ElementReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element("#document")
        self._open = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = Element(tag, {name: value or "" for name, value in attrs})
        self._open[-1].children.append(element)
        if tag not in _VOID_ELEMENTS:
            self._open.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._open[-1].children.append(
            Element(tag, {name: value or "" for name, value in attrs})
        )

    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self._open) - 1, 0, -1):
            if self._open[depth].tag == tag:
                del self._open[depth:]
                return

    def handle_data(self, data: str) -> None:
        self._open[-1].children.append(data)


def parse_html(markup: str) -> Element:
    """Parse rendered markup into a document :class:`Element`."""
    reader = _ElementReader()
    reader.feed(markup)
    reader.close()
    return reader.root


# ── Serving one release tree to a fetch, with no network and no token ────────

# The immutable revision the fixture repository resolves its pin to.
FIXTURE_REVISION = "e" * 40

ServeRelease = Callable[..., list[str]]


@pytest.fixture
def serve_release(monkeypatch: pytest.MonkeyPatch) -> ServeRelease:
    """Return a factory serving one release tree through the library's fetch.

    The factory stands in for the hub, so a fetch reads the tree from disk: no
    network, no token, no credentials. It returns the list the stand-in appends
    every downloaded path to, which is what tells a second resolve apart from a
    second fetch.
    """

    def serve(root: Path, *, repo_id: str = "owner/dataset") -> list[str]:
        downloads: list[str] = []

        class Api:
            def __init__(self, **kwargs: Any) -> None:
                assert kwargs == {"token": False}

            def repo_info(self, **kwargs: Any) -> SimpleNamespace:
                assert kwargs["repo_id"] == repo_id
                assert kwargs["repo_type"] == "dataset"
                return SimpleNamespace(sha=FIXTURE_REVISION, private=False)

            def list_repo_tree(self, **kwargs: Any) -> list[SimpleNamespace]:
                assert kwargs["revision"] == FIXTURE_REVISION
                # Every file stored outside Xet, so each takes the plain download.
                return [
                    SimpleNamespace(
                        path=path.relative_to(root).as_posix(), size=0, xet_hash=None
                    )
                    for path in root.rglob("*")
                    if path.is_file()
                ]

        def download(**kwargs: Any) -> str:
            assert kwargs["repo_id"] == repo_id
            assert kwargs["token"] is False
            filename = str(kwargs["filename"])
            downloads.append(filename)
            # A download into a directory: the file in place, bookkeeping under .cache/.
            target = Path(kwargs["local_dir"]) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / filename, target)
            metadata = (
                Path(kwargs["local_dir"])
                / ".cache"
                / "huggingface"
                / f"{filename}.metadata"
            )
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text("commit\n")
            return str(target)

        transport = import_module("dpstats.workspace.fetch")
        monkeypatch.setattr(
            transport,
            "huggingface_hub",
            SimpleNamespace(HfApi=Api, hf_hub_download=download),
        )
        return downloads

    return serve
