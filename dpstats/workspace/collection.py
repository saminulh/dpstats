"""One Task collection: installed declarations, observed Results, and execution.

``TaskCollection`` holds the storage roots and Task. Reading Results needs no
Sweep; execution needs explicit finite settings. Both accept registry selections.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast, overload
import warnings

import numpy as np
import pandas as pd
from pydantic import ValidationError
from typing_extensions import TypeVar as TypeVarWithDefault

from dpstats.algorithm import (
    BaseAlgorithm,
    load_algorithms,
    register_algorithm,
)
from dpstats.budgets import DPBudget
from dpstats.instance import (
    Instance,
    load_instance_input,
    load_instances,
    persist_instance,
    register_instance,
)
from dpstats.results import (
    results_frame,
)
from dpstats.results._frame import RESULT_COLUMNS
from dpstats.sweep import (
    BudgetPolicy,
    CellScheduler,
    read_sweep_task,
    schedule_cells_serially,
)
from dpstats.sweep._sweep import (
    Sweep,
    read_sweep,
    write_sweep,
)
from dpstats.sweep._execution import (
    execute_sweep_groups,
    require_limit_enforcement,
    validate_mode,
)
from dpstats.sweep._claims import reconcile_orphaned_work
from dpstats.sweep.coordinates import (
    ResolvedSelection,
    registry_entries,
    require_bound_task,
    require_task,
    resolve_selection,
)
from dpstats.results._resolution import recorded_statuses
from dpstats.task import (
    OutputT,
    Task,
    TaskInput,
    TaskMetadata,
)
from dpstats.workspace.layout import (
    INPUT_ARTIFACTS_DIRECTORY,
    ReleaseIdentity,
    input_artifact_path,
    instance_weights_path,
    read_release_identity,
    sweeps_directory,
    task_collection_directory,
    workspace_roots,
)
from dpstats.workspace.registries import installed_registry_names
from dpstats.transport.validators import is_portable_name


class _Omitted:
    """Distinguish an omitted override from an explicit dynamic selection."""


_OMITTED = _Omitted()

_SelectedObject = TypeVar("_SelectedObject", BaseAlgorithm, Instance)

# Collections consume as well as return these types, so they are invariant.
# An omitted parameter represents a Task selected dynamically by its id.
_CollectionMetadataT = TypeVarWithDefault(
    "_CollectionMetadataT", bound=TaskMetadata, default=Any
)
_CollectionInputT = TypeVarWithDefault(
    "_CollectionInputT", bound=TaskInput, default=Any
)


@dataclass(frozen=True)
class _SelectionRequest:
    """One validated Sweep plus unpersisted live-object substitutions."""

    sweep: Sweep
    algorithms: Mapping[str, BaseAlgorithm]
    instances: Mapping[str, Instance]
    budget_filter: Callable[[DPBudget], bool] | None


def _normalize_live_selection(
    value: object,
    object_type: type[_SelectedObject],
    task: Task,
    name: str,
) -> tuple[tuple[str, ...] | None, dict[str, _SelectedObject]]:
    """Separate live objects from the UUID selection persisted in a Sweep."""
    if value is None:
        return None, {}
    if isinstance(value, str):
        raise TypeError(f"{name} requires an iterable, not a bare string.")
    if not isinstance(value, Iterable):
        raise TypeError(f"{name} requires an iterable of UUIDs or configured objects.")
    ids: list[str] = []
    live: dict[str, _SelectedObject] = {}
    for entry in cast("Iterable[object]", value):
        if isinstance(entry, str):
            ids.append(entry)
            continue
        if not isinstance(entry, object_type):
            raise TypeError(f"{name} requires UUIDs or {object_type.__name__} objects.")
        require_bound_task((entry,), task, f"Selected {name} object")
        previous = live.get(entry.uuid)
        if previous is not None and previous != entry:
            raise ValueError(f"Conflicting live records for UUID {entry.uuid!r}.")
        live.setdefault(entry.uuid, entry)
        ids.append(entry.uuid)
    return tuple(dict.fromkeys(ids)), live


def _accounting(
    requested: Sequence[str] | None,
    resolved: Mapping[str, Any],
    unresolved: Sequence[str],
) -> dict[str, Any]:
    """Count one kind's requested, resolved, and unresolved selections."""
    return {
        "requested": None if requested is None else len(resolved) + len(unresolved),
        "resolved": len(resolved),
        "unresolved": list(unresolved),
    }


@dataclass(frozen=True)
class TaskCollection(Generic[_CollectionMetadataT, _CollectionInputT, OutputT]):
    """A TaskCollection manages the objects in a workspace associated with one task.

    Use the collection to load algorithm and instance registries, run
    experiments, and inspect the results.

    A workspace holds ``local/`` (this handle's own store; every write lands
    here) and optionally a fetched ``official-release/`` (read-only, byte for
    byte what was fetched). A bare bundle — a directory holding
    ``release.json`` directly, with no ``local/`` — opens read-only, the same
    way a workspace with no ``local/`` write access would: its own Task
    directories are read as the release tree.

    An Algorithm or Instance with the same UUID in both trees may differ only
    in label without a conflict; the local label wins.
    """

    _workspace: Path
    task: Task[_CollectionMetadataT, _CollectionInputT, OutputT]
    _local_root: Path | None
    _release_root: Path | None

    # The one Input ``instance_features`` is holding while a single Instance's
    # feature callables run. Its key is entered, with no value, when that
    # Instance's row opens, which is how ``load_input`` knows to keep what it
    # reads; the entry is dropped when the row closes, so no Input outlives it.
    _held_input: dict[str, TaskInput | None] = field(
        init=False,
        default_factory=dict[str, "TaskInput | None"],
        repr=False,
        compare=False,
    )

    @classmethod
    def from_sweep(
        cls,
        sweep_path: str | Path,
        workspace: str | Path,
        *,
        task: Task | None = None,
    ) -> TaskCollection:
        """Open the Task collection named by a saved Sweep.

        A saved Sweep carries a Task id, not its runtime typed contract. An
        explicit custom Task is therefore required when the id is not built
        in. Opening reads only that id, never the Sweep's budget classes.

        Args:
            sweep_path: The authored Sweep.
            workspace: The workspace or bare bundle to open it in.
            task: The custom Task the Sweep declares, if it is not built in.

        Returns:
            A collection handle, without a default Sweep. Pass the path again
            to operations that should use its settings.

        Raises:
            OSError: If the saved Sweep cannot be read.
            ValueError: If its Task declaration is invalid, or ``workspace``
                is neither a workspace nor a bare bundle.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.tasks import MeanEstimation
            >>> with TemporaryDirectory() as root:
            ...     original = open_collection(init_workspace(root), MeanEstimation)
            ...     path = original.save_sweep("demo", {"seeds": [0]})
            ...     reopened = TaskCollection.from_sweep(path, root)
            ...     reopened.task == original.task
            True
        """
        declared = read_sweep_task(Path(sweep_path))
        if task is not None:
            require_task(declared, task)
            return open_collection(workspace, task)
        return open_collection(workspace, task_id=declared)

    @property
    def workspace(self) -> Path:
        """The workspace or bare bundle this collection was opened in."""
        return self._workspace

    @property
    def local_root(self) -> Path | None:
        """This handle's own store (``local/``), or ``None`` on a bare bundle."""
        return self._local_root

    @property
    def release_root(self) -> Path | None:
        """The fetched release tree, or ``None`` when none is installed."""
        return self._release_root

    @property
    def _input_artifacts_root(self) -> Path | None:
        """The root ``load_input``/``TaskCollection.run_sweep`` stage exact Inputs into for reading.

        ``local/input-artifacts/`` when this handle can write, else the
        release's own ``input-artifacts/`` on a read-only bundle; ``None``
        when neither tree is present.
        """
        if self.local_root is not None:
            return self.local_root / INPUT_ARTIFACTS_DIRECTORY
        if self.release_root is not None:
            return self.release_root / INPUT_ARTIFACTS_DIRECTORY
        return None

    @property
    def release(self) -> ReleaseIdentity | None:
        """This collection's fetched Distribution Release, or ``None`` without one."""
        if self.release_root is None:
            return None
        return read_release_identity(self.release_root)

    def load_weights(self) -> pd.DataFrame:
        """Load the release's per-Instance weights for this Task.

        Returns:
            One row per weighted Instance: ``instance_uuid`` and its float
            ``weight``, the ``weights`` frame ``compute_profile_curves``
            accepts.

        Raises:
            ValueError: If this collection carries no release, or the view's
                columns are not ``instance`` and ``weight``, names an Instance
                more than once, or carries a weight that is not a finite
                nonnegative number.
            FileNotFoundError: If the release ships no weights view for this
                Task.
        """
        if self._release_directory is None:
            raise ValueError(
                f"Task {self.task.id!r} has no installed release to read weights "
                "from. Fetch one with fetch_benchmark."
            )
        path = instance_weights_path(self._release_directory)
        if not path.is_file():
            raise FileNotFoundError(
                f"Release {self.release_root} ships no weights view for task "
                f"{self.task.id!r}: {path} is absent."
            )
        view = pd.read_csv(
            path,
            dtype={"instance": str, "weight": str},
            keep_default_na=False,
        )
        if list(view.columns) != ["instance", "weight"]:
            raise ValueError(
                f"{path} must carry the columns instance,weight; it carries "
                f"{','.join(map(str, view.columns))}."
            )
        repeated = sorted(set(view.loc[view["instance"].duplicated(), "instance"]))
        if repeated:
            raise ValueError(f"{path} names Instance(s) more than once: {repeated}.")
        # Python's float() reads each written decimal to the exact double.
        weights: list[float] = []
        refused: list[str] = []
        for instance, text in zip(view["instance"], view["weight"]):
            try:
                weight = float(text)
            except ValueError:
                weight = math.nan
            if not (math.isfinite(weight) and weight >= 0.0):
                refused.append(instance)
            weights.append(weight)
        if refused:
            raise ValueError(
                f"{path} carries no finite nonnegative weight for Instance(s) "
                f"{refused}."
            )
        return pd.DataFrame(
            {
                "instance_uuid": view["instance"],
                "weight": pd.Series(weights, dtype=float),
            }
        )

    @property
    def _local_directory(self) -> Path | None:
        return (
            None
            if self.local_root is None
            else task_collection_directory(self.local_root, self.task.id)
        )

    @property
    def _release_directory(self) -> Path | None:
        return (
            None
            if self.release_root is None
            else task_collection_directory(self.release_root, self.task.id)
        )

    @property
    def _directory(self) -> Path:
        """The one tree operations that write or read Attempts/Results use.

        The local tree when this handle can write; the release tree of a
        read-only bare bundle otherwise.
        """
        directory = self._local_directory
        return directory if directory is not None else self._release_directory  # type: ignore[return-value]

    @property
    def _task_roots(self) -> tuple[Path, ...]:
        """Every installed tree's Task directory, local first, for dual reads."""
        return tuple(
            directory
            for directory in (self._local_directory, self._release_directory)
            if directory is not None
        )

    def _refuse_write(self, operation: str) -> None:
        """Refuse an operation on a handle with no writable ``local/`` tree.

        Raises:
            ValueError: naming the operation, when this handle opens a bare,
                read-only bundle.
        """
        if self.local_root is None:
            raise ValueError(
                f"{operation} would write into a bare Distribution Release "
                "bundle, which is read-only. Open a workspace with "
                "init_workspace/fetch_benchmark and record your Attempts "
                "there instead."
            )

    def _stage_input_artifact(self, digest: str) -> None:
        """Ensure one Input artifact named by digest is present in the writable root.

        A digest present only in the release's Input-artifacts store is
        linked into ``local/input-artifacts/`` so execution and reads always
        resolve against one root; ``official-release/`` is never written.
        """
        if self.local_root is None or self.release_root is None:
            return
        local_root = self.local_root / INPUT_ARTIFACTS_DIRECTORY
        release_root = self.release_root / INPUT_ARTIFACTS_DIRECTORY
        target = input_artifact_path(local_root, digest)
        if target.exists() or target.is_symlink():
            return
        source = input_artifact_path(release_root, digest)
        if not source.is_file():
            return
        local_root.mkdir(parents=True, exist_ok=True)
        # Resolved to an absolute path: a relative source is relative to the
        # workspace root the caller passed in, not to the symlink's own
        # directory, so writing it unresolved would create a target that
        # only happens to look right from the process's current directory.
        target.symlink_to(source.resolve())

    def _readable_input_root(self, instance: Instance) -> Path:
        """Return the Input-artifacts root holding this Instance's artifact.

        The local store is preferred, then the release's; nothing is written.
        A dangling link left by an earlier staging counts as absent.

        Raises:
            ValueError: If neither store holds the artifact.
        """
        for root in (self.local_root, self.release_root):
            if root is None:
                continue
            store = root / INPUT_ARTIFACTS_DIRECTORY
            if input_artifact_path(store, instance.input_digest).is_file():
                return store
        raise ValueError(
            f"The Input for Instance {instance.uuid} is not installed in this "
            "collection's local or release Input-artifacts store."
        )

    def _writable_input_root(self, operation: str) -> Path:
        """Return the Input-artifacts root a write-capable run stages into.

        Raises:
            ValueError: If this handle opens a bare, read-only bundle, or was
                opened without an Input-artifacts store.
        """
        self._refuse_write(operation)
        if self._input_artifacts_root is None:
            raise ValueError("Running requires an Input-artifacts root.")
        return self._input_artifacts_root

    def list_sweeps(
        self, *, source: Literal["local", "release", "both"] = "both"
    ) -> tuple[Path, ...]:
        """List saved Sweeps across the selected tree(s), without reading their contents.

        A release's own Sweeps are readable in place; a name installed in
        both trees is read from ``local/``, the same local-then-release
        precedence every other dual-tree read uses.

        Args:
            source: Which installed tree(s) this reads: ``"local"``,
                ``"release"``, or ``"both"`` (the default) reads every
                installed tree, local first when a name is in both.

        Returns:
            JSON file paths in name order, or an empty tuple if none exist.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     open_collection(init_workspace(root), task_id="mean_estimation").list_sweeps()
            ()
        """
        by_name: dict[str, Path] = {}
        for directory in reversed(self._roots_for_source(source).values()):
            for path in sweeps_directory(directory).glob("*.json"):
                by_name[path.name] = path
        return tuple(sorted(by_name.values(), key=lambda path: path.name))

    def list_algorithm_registries(
        self, *, source: Literal["local", "release", "both"] = "both"
    ) -> tuple[str, ...]:
        """List installed Algorithm registry names without loading algorithms.

        Args:
            source: Which installed tree(s) this reads: ``"local"``,
                ``"release"``, or ``"both"`` (the default) reads every
                installed tree, local first when a name is in both.

        Returns:
            Names without the JSON suffix, in name order; an empty tuple when
            no registry is installed.

        Raises:
            ValueError: If the registry directory has an invalid member.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     open_collection(init_workspace(root), task_id="mean_estimation").list_algorithm_registries()
            ()
        """
        return self._installed_registries("algorithm", source)

    def list_instance_registries(
        self, *, source: Literal["local", "release", "both"] = "both"
    ) -> tuple[str, ...]:
        """List installed Instance registry names without reading Inputs.

        Args:
            source: Which installed tree(s) this reads: ``"local"``,
                ``"release"``, or ``"both"`` (the default) reads every
                installed tree, local first when a name is in both.

        Returns:
            Names without the JSON suffix, in name order; an empty tuple when
            no registry is installed.

        Raises:
            ValueError: If the registry directory has an invalid member.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     open_collection(init_workspace(root), task_id="mean_estimation").list_instance_registries()
            ()
        """
        return self._installed_registries("instance", source)

    def _roots_for_source(
        self, source: Literal["local", "release", "both"]
    ) -> dict[str, Path]:
        """Resolve which installed tree(s) *source* selects, as ``tag: root``, local first.

        The one resolution every ``source``-accepting method shares:
        ``"both"`` is every installed tree, ``"local"``/``"release"`` is that
        one tree alone (empty when it is not installed), and any other value
        refuses.
        """
        if source not in ("local", "release", "both"):
            raise ValueError("source must be 'local', 'release', or 'both'.")
        directories = {
            "local": self._local_directory,
            "release": self._release_directory,
        }
        return {
            tag: directory
            for tag, directory in directories.items()
            if directory is not None and source in (tag, "both")
        }

    def _declaration_roots(
        self, results_source: Literal["local", "release", "both"]
    ) -> tuple[Path, ...]:
        """The trees whose registries resolve a read of ``results_source``'s Results.

        A release is self-contained: its Results reference only its own
        declarations, so a release-only read resolves against the release's
        registries alone and a stale local registry cannot refuse it. A local
        Result may reference a released declaration (an official coordinate
        run locally), so a local or dual read resolves across every installed
        tree, local first.
        """
        if results_source == "release":
            return tuple(self._roots_for_source(results_source).values())
        return self._task_roots

    def _installed_registries(
        self, kind: str, source: Literal["local", "release", "both"] = "both"
    ) -> tuple[str, ...]:
        return installed_registry_names(
            tuple(self._roots_for_source(source).values()), kind
        )

    def load_instances(
        self,
        *,
        source: Literal["local", "release", "both"] = "both",
        registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> dict[str, Instance[_CollectionMetadataT]]:
        """Load the installed direct Instances, keyed by Instance UUID.

        Args:
            source: Which installed tree(s) this reads: ``"local"``,
                ``"release"``, or ``"both"`` (the default) reads every
                installed tree, local first when a name is in both.
            registries: One registry name, a sequence of names, or ``None`` to
                read every installed Instance registry. A named group filed
                under both trees is merged; conflicting content under the
                same name refuses.
            strict: Refuse the group on an entry this library cannot rebuild
                (the default); ``False`` skips it with one warning naming it.

        Returns:
            Live Instances keyed by UUID, deduplicated across the selected
            registries. This does not load their Input arrays.

        Raises:
            ValueError: If the selection is empty or missing, or records have
                conflicting declarations, invalid identities, or another Task.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> import numpy as np
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.domains import BoxDomain
            >>> from dpstats.tasks.mean_estimation import MeanEstimationInput, MeanEstimationMetadata
            >>> metadata = MeanEstimationMetadata(n_samples=2, domain=BoxDomain(bounds=np.array([[0., 1.]])))
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     instance = collection.add_instance(metadata, MeanEstimationInput(X=np.array([[0.], [1.]])), {}, registry="mine")
            ...     instance.uuid in collection.load_instances(registries="mine")
            True
        """
        roots = tuple(self._roots_for_source(source).values())
        entries, sources = registry_entries(roots, "instance", registries)
        instances = load_instances(
            entries, task_id=self.task.id, strict=strict, sources=sources
        )
        require_bound_task(instances.values(), self.task, "Installed entry")
        return cast("dict[str, Instance[_CollectionMetadataT]]", instances)

    def load_input(self, instance: Instance[_CollectionMetadataT]) -> _CollectionInputT:
        """Load one Instance's exact Input from this collection's store.

        While ``instance_features`` computes one Instance's features, the Input
        a callable loads here serves that Instance's remaining callables
        and is released with its row, so one row reads one artifact. Every other
        call reads the store.

        Args:
            instance: The Instance whose persisted Input is read.

        Returns:
            The decoded, Task-validated Input this Instance refers to.

        Raises:
            ValueError: If neither the local nor the release Input-artifacts
                store holds this Instance's artifact. A read never writes.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> import numpy as np
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.domains import BoxDomain
            >>> from dpstats.tasks.mean_estimation import MeanEstimationInput, MeanEstimationMetadata
            >>> metadata = MeanEstimationMetadata(n_samples=2, domain=BoxDomain(bounds=np.array([[0., 1.]])))
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     instance = collection.add_instance(metadata, MeanEstimationInput(X=np.array([[0.], [1.]])), {})
            ...     collection.load_input(instance).X.shape
            (2, 1)
        """
        if instance.task is not self.task:
            raise ValueError("The Instance belongs to another Task.")
        held = self._held_input.get(instance.uuid)
        if held is not None:
            return cast("_CollectionInputT", held)
        loaded = load_instance_input(instance, self._readable_input_root(instance))
        if instance.uuid in self._held_input:
            self._held_input[instance.uuid] = loaded
        return cast("_CollectionInputT", loaded)

    def instance_features(
        self,
        features: Mapping[str, Callable[[Instance[_CollectionMetadataT]], float]],
        *,
        registries: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Compute named per-Instance quantities over the installed Instances.

        Each callable receives one live Instance and returns one number for it.
        A callable reading confidential records loads them through
        ``load_input``; this collection holds one Instance's Input at a time and
        releases it before the next, so a callable reading only public metadata
        opens no artifact at all. Nothing is defined here: the caller's
        callables are the formulas, and their exceptions propagate rather than
        leaving an Instance silently unfeatured.

        Args:
            features: One callable per column, keyed by the column name it
                fills. An empty mapping computes nothing.
            registries: One registry name, a sequence of names, or ``None`` to
                read every installed Instance registry.

        Returns:
            One float column per callable, indexed by ``instance_uuid``.

        Raises:
            ValueError: If the Instance selection is empty or missing, or an
                Input a callable reads is not installed.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> import numpy as np
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.domains import BoxDomain
            >>> from dpstats.tasks.mean_estimation import MeanEstimationInput, MeanEstimationMetadata
            >>> metadata = MeanEstimationMetadata(n_samples=2, domain=BoxDomain(bounds=np.array([[0., 1.]])))
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     _ = collection.add_instance(metadata, MeanEstimationInput(X=np.array([[0.], [1.]])), {}, registry="mine")
            ...     frame = collection.instance_features(
            ...         {"n": lambda instance: instance.metadata.n_samples}, registries="mine"
            ...     )
            ...     list(frame.columns), frame.index.name, list(frame["n"])
            (['n'], 'instance_uuid', [2.0])
        """
        selected = self.load_instances(registries=registries)
        columns: dict[str, list[float]] = {name: [] for name in features}
        for uuid, instance in selected.items():
            self._held_input[uuid] = None
            try:
                for name, calculation in features.items():
                    columns[name].append(float(calculation(instance)))
            finally:
                del self._held_input[uuid]
        return pd.DataFrame(
            {name: np.asarray(values, dtype=float) for name, values in columns.items()},
            index=pd.Index(list(selected), name="instance_uuid"),
        )

    def load_algorithms(
        self,
        *,
        source: Literal["local", "release", "both"] = "both",
        registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> dict[str, BaseAlgorithm]:
        """Load the installed configured Algorithms, keyed by algorithm UUID.

        Args:
            source: Which installed tree(s) this reads: ``"local"``,
                ``"release"``, or ``"both"`` (the default) reads every
                installed tree, local first when a name is in both.
            registries: One registry name, a sequence of names, or ``None`` to
                read every installed Algorithm registry. A named group filed
                under both trees is merged; conflicting content under the
                same name refuses.
            strict: Refuse the group on an entry this library cannot rebuild
                (the default); ``False`` skips it with one warning naming it.

        Returns:
            Configured Algorithm objects keyed by UUID, deduplicated across
            the selected registries; configuration classes must be importable.

        Raises:
            ValueError: If the selection is empty or missing, or a declaration
                conflicts, cannot be reconstructed, or has another identity/Task.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.tasks.mean_estimation.algorithms import SampleMean
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     uuid = collection.add_algorithm(SampleMean(), registry="mine")
            ...     collection.load_algorithms(registries="mine")[uuid] == SampleMean()
            True
        """
        roots = tuple(self._roots_for_source(source).values())
        entries, sources = registry_entries(roots, "algorithm", registries)
        algorithms = load_algorithms(
            entries, task_id=self.task.id, strict=strict, sources=sources
        )
        require_bound_task(algorithms.values(), self.task, "Installed entry")
        return algorithms

    def add_instance(
        self,
        metadata: _CollectionMetadataT,
        input: _CollectionInputT,
        metric_arguments: Mapping[str, Any],
        *,
        label: str | None = None,
        registry: str = "default",
    ) -> Instance[_CollectionMetadataT]:
        """Persist one exact Input, file its Instance, and return the Instance.

        ``metadata`` states public facts about the problem — bounds, domains,
        dimensions, sample size. Author them; never derive one from ``input``,
        whose contents no cell's accounting covers reading.

        Args:
            metadata: This Instance's public problem declaration. Its declared
                Task must be this collection's.
            input: The exact typed arrays to persist.
            metric_arguments: The evaluation-side metric argument values Metrics
                read; they are never part of Instance identity.
            label: An optional label, which never enters identity.
            registry: The named Instance group it joins.

        Returns:
            The registered Instance, referring to its persisted Input artifact.

        Raises:
            ValueError: If this handle opens a Release, if the Metadata declares
                another Task, which no cell of this collection could select, or
                if this handle was opened without the Input-artifacts root the
                exact arrays are persisted in.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> import numpy as np
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.domains import BoxDomain
            >>> from dpstats.tasks.mean_estimation import MeanEstimationInput, MeanEstimationMetadata
            >>> metadata = MeanEstimationMetadata(n_samples=2, domain=BoxDomain(bounds=np.array([[0., 1.]])))
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     instance = collection.add_instance(metadata, MeanEstimationInput(X=np.array([[0.], [1.]])), {})
            ...     instance.uuid in collection.load_instances()
            True
        """
        self._refuse_write("Adding an Instance")
        if metadata.task is not self.task:
            raise ValueError(
                f"This collection holds Task {self.task.id!r}, but the metadata "
                f"declares Task {metadata.task.id!r}."
            )
        instance = persist_instance(
            metadata,
            input,
            metric_arguments,
            self._input_artifacts_root,  # type: ignore[arg-type]
            label=label,
        )
        register_instance(self.local_root, instance, registry=registry)  # type: ignore[arg-type]
        return cast("Instance[_CollectionMetadataT]", instance)

    def add_algorithm(
        self, algorithm: BaseAlgorithm, *, registry: str = "default"
    ) -> str:
        """File one configured Algorithm in a named group and return its UUID.

        Args:
            algorithm: The configured Algorithm to install. Its declared Task
                must be this collection's.
            registry: The named Algorithm group it joins.

        Returns:
            The configured algorithm's UUID. Adding it executes no cells.

        Raises:
            ValueError: If this handle opens a Release, or if the Algorithm
                declares another Task, which no cell of this collection could
                select.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.tasks.mean_estimation.algorithms import SampleMean
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     collection.add_algorithm(SampleMean()) == SampleMean().uuid
            True
        """
        self._refuse_write("Adding an Algorithm")
        if algorithm.task is not self.task:
            raise ValueError(
                f"This collection holds Task {self.task.id!r}, but the algorithm "
                f"declares Task {algorithm.task.id!r}."
            )
        return register_algorithm(self.local_root, algorithm, registry=registry)  # type: ignore[arg-type]

    def save_sweep(
        self, name: str, sweep: Mapping[str, Any], overwrite: bool = False
    ) -> Path:
        """Save a Sweep, recording object selections as UUIDs rather than registries.

        Saving does not register supplied objects or execute work. Their
        declarations must be available when the saved UUIDs are resolved.

        Args:
            name: Portable filename stem, without ``.json``.
            sweep: Settings for this Task: optional ``algorithms``, ``instances``,
                typed ``budgets``/policies, ``seeds``, ``time_limit`` in seconds,
                and ``memory_limit_gb`` in GiB. ``None`` leaves a selection
                unconstrained or a resource uncapped.
            overwrite: Replace an existing saved Sweep, not its Results.

        Returns:
            The saved JSON path under this collection's sweeps directory.

        Raises:
            FileExistsError: If the name exists and overwrite is false.
            TypeError: If a setting has the wrong type.
            ValueError: If this handle opens a Release, or if the name, Task,
                setting keys, or values are invalid.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from dpstats.budgets import NO_PRIVACY
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     path = collection.save_sweep("demo", {"budgets": [NO_PRIVACY], "seeds": [0, 1]})
            ...     path.name
            'demo.json'
        """
        self._refuse_write("Saving a Sweep")
        if not is_portable_name(name) or name.endswith(".json"):
            raise ValueError("Sweep name must be a portable name without .json.")
        request = self._request(sweep)
        path = sweeps_directory(self._directory) / f"{name}.json"
        write_sweep(path, request.sweep, overwrite=overwrite)
        return path

    def _resolve_sweep_path(self, name: str | Path) -> Path:
        """Resolve a saved Sweep's exact file, by portable name or literal path.

        A literal path (containing a separator or ending ``.json``, as
        ``save_sweep`` returns) is used as-is. A bare name is looked up under
        ``local/`` first, then a release Sweep of the same name, read in
        place rather than copied.
        """
        text = str(name)
        if "/" in text or "\\" in text or text.endswith(".json"):
            return Path(name)
        for directory in self._task_roots:
            candidate = sweeps_directory(directory) / f"{text}.json"
            if candidate.is_file():
                return candidate
        searched = ", ".join(
            str(sweeps_directory(directory)) for directory in self._task_roots
        )
        raise ValueError(
            f"No saved Sweep named {text!r}; looked in {searched or 'no installed tree'}."
        )

    def load_sweep(self, name: str | Path) -> dict[str, Any]:
        """Load a saved Sweep as ordinary editable settings.

        Args:
            name: A saved Sweep's portable name (resolved under ``local/``
                then, in place, the release), or a literal JSON path such as
                ``save_sweep`` returns.

        Returns:
            A dictionary with ``task``, UUID selections ``algorithms`` and
            ``instances``, reconstructed ``budgets``/policies, ``seeds``, and
            ``time_limit``/``memory_limit_gb``. Omitted selections and unlimited
            resources become ``None``. Algorithm/Instance objects are not loaded.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the saved declaration is invalid or names another
                Task, or a bare name is installed in no tree.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     path = collection.save_sweep("demo", {"seeds": [0, 1]})
            ...     settings = collection.load_sweep(path)
            ...     settings["seeds"] == [0, 1]
            True
        """
        return self._read_sweep(name).settings()

    def _read_sweep(self, name: str | Path) -> Sweep:
        sweep = read_sweep(self._resolve_sweep_path(name))
        require_task(sweep.task, self.task)
        return sweep

    def _request(
        self,
        sweep: str | Path | Mapping[str, Any] | None,
        *,
        budgets: (
            Iterable[DPBudget | BudgetPolicy]
            | Callable[[DPBudget], bool]
            | None
            | _Omitted
        ) = _OMITTED,
        **overrides: Any,
    ) -> _SelectionRequest:
        saved = isinstance(sweep, (str, Path))
        merged: dict[str, Any] = (
            dict(self._read_sweep(sweep)) if saved else dict(sweep or {})
        )
        merged.setdefault("task", self.task.id)
        # A predicate narrows the declared budgets rather than declaring its own.
        budget_filter: Callable[[DPBudget], bool] | None = None
        if callable(budgets):
            budget_filter = budgets
        else:
            overrides["budgets"] = budgets
        merged.update(
            {name: value for name, value in overrides.items() if value is not _OMITTED}
        )
        algorithm_ids, live_algorithms = _normalize_live_selection(
            merged.get("algorithms"), BaseAlgorithm, self.task, "algorithm"
        )
        instance_ids, live_instances = _normalize_live_selection(
            merged.get("instances"), Instance, self.task, "instance"
        )
        merged.update(algorithms=algorithm_ids, instances=instance_ids)
        try:
            declared = Sweep(**merged)
        except ValidationError as error:
            unknown = [
                str(item["loc"][0])
                for item in error.errors()
                if item["type"] == "extra_forbidden"
            ]
            if unknown:
                raise ValueError(f"Unknown Sweep fields: {sorted(unknown)}.") from error
            raise
        if not saved:
            require_task(declared.task, self.task)
        return _SelectionRequest(
            sweep=declared,
            algorithms=live_algorithms,
            instances=live_instances,
            budget_filter=budget_filter,
        )

    def load_results(
        self,
        sweep: str | Path | Mapping[str, Any] | None = None,
        *,
        algorithms: Iterable[BaseAlgorithm | str] | None | _Omitted = _OMITTED,
        instances: Iterable[Instance | str] | None | _Omitted = _OMITTED,
        budgets: (
            Iterable[DPBudget | BudgetPolicy]
            | Callable[[DPBudget], bool]
            | None
            | _Omitted
        ) = _OMITTED,
        seeds: Iterable[int] | None | _Omitted = _OMITTED,
        results_source: Literal["local", "release", "both"] = "both",
        algorithm_registries: str | Sequence[str] | None = None,
        instance_registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Load observed Results, warning about gaps in a finite selection.

        Omitted overrides inherit the Sweep; explicit ``None`` clears that
        restriction. No Sweep is inferred, no cells run, and missing cells are
        absent rather than synthetic Results. Nonprivate algorithms use their
        single no-privacy budget when a budget selection is supplied.
        ``results_source`` decides which tree(s)' recorded Results are read.
        A ``"release"`` read resolves Algorithm, Instance, and registry
        selection against the release's own registries alone, so a UUID the
        release does not declare is unresolved there; a ``"local"`` or
        ``"both"`` read resolves across every installed tree, since a local
        Result may reference a released declaration.

        Args:
            sweep: Saved path, settings dictionary, or ``None`` for no defaults.
            algorithms: Configured objects and/or UUIDs; ``None`` selects all
                installed algorithms in registry scope.
            instances: Instances and/or UUIDs; ``None`` selects all in scope.
            budgets: Typed budgets or policies; ``None`` reads every observed
                budget. Policies resolve against each Instance's metadata. A
                predicate on each row's budget keeps the rows it accepts, and
                narrows the Sweep's declared budgets when it has any; a row
                under an unconstrained budget always passes.
            seeds: Mechanism seeds, or ``None`` for every observed seed.
            results_source: Which installed tree(s)' Results this reads:
                ``"local"``, ``"release"``, or ``"both"`` (the default) reads
                every installed tree, local first when a coordinate is
                answered in both. ``"release"`` also confines registry
                resolution to the release's own registries.
            algorithm_registries: One name, several names, or all installed
                groups; used to resolve UUIDs and unrestricted selections.
            instance_registries: Corresponding Instance registry scope.
            strict: Refuse an entry this library cannot rebuild (the
                default), naming the entry and its registry file; ``False``
                skips it with a warning naming them.

        Returns:
            One row per selected current Result, across the ``results_source``-selected tree(s).
            Each row carries ``task``, ``instance``, ``instance_uuid``, ``algorithm``, ``alg_uuid``, ``budget``, ``alg_seed``, ``status``, ``payload``, ``runtime_s``, and ``peak_rss_mb``.
            ``algorithm_label`` and ``instance_label`` hold ``str()`` of each object, its label or derived name, as plain text.
            ``source`` is ``"local"`` or ``"release"``.
            ``release_version`` is the release's own version on a ``"release"`` row, and ``None`` on a ``"local"`` row.
            ``budget_group`` is the ``budgets`` entry declaring that row's budget, deduplicated to the first entry reaching it, or ``None`` when no budget selection was declared or the row's budget lies outside it.
            A local run of an official coordinate is a second row, never a merge with the release's own row.
            Configured objects stay live; measurements and descriptions are not added.

        Raises:
            TypeError: If an override has the wrong type.
            ValueError: If settings or declarations are invalid or UUIDs
                cannot be resolved. Finite missing coverage emits a warning,
                counted against the ``results_source``-selected tree(s).

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     collection.load_results(algorithms=[], instances=[]).empty
            True
        """
        request = self._request(
            sweep,
            algorithms=algorithms,
            instances=instances,
            budgets=budgets,
            seeds=seeds,
        )
        selection = resolve_selection(
            request.sweep,
            self.task,
            self._declaration_roots(results_source),
            live_algorithms=request.algorithms,
            live_instances=request.instances,
            budgets=request.budget_filter,
            algorithm_registries=algorithm_registries,
            instance_registries=instance_registries,
            strict=strict,
        )
        frame = self._selected_results(selection, source=results_source)
        if request.budget_filter is not None and selection.groups is None:
            # With no declared grid to narrow, the predicate selects rows.
            accepted = {
                budget: budget.is_unconstrained or request.budget_filter(budget)
                for budget in dict.fromkeys(frame["budget"])
            }
            frame = frame.loc[
                [accepted[budget] for budget in frame["budget"]]
            ].reset_index(drop=True)
        return frame

    def _selected_results(
        self,
        selection: ResolvedSelection,
        *,
        source: Literal["local", "release", "both"],
    ) -> pd.DataFrame:
        """Read the ``source``-selected tree(s) for one already-resolved selection.

        The sole reader behind both ``load_results`` and ``TaskCollection.run_sweep``'s
        return, so a coordinate a run leaves answered only in a fetched
        release is reported the same way whichever entry point asks for it.
        """
        parts: list[pd.DataFrame] = []
        for tag, directory in self._roots_for_source(source).items():
            identity = self.release if tag == "release" else None
            parts.append(
                results_frame(
                    directory,
                    selection.instances,
                    selection.algorithms,
                    seeds=selection.seeds,
                    groups=selection.groups,
                ).assign(
                    source=tag,
                    release_version=identity.version if identity else None,
                )
            )
        frame = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=[*RESULT_COLUMNS, "source", "release_version"])
        )
        # The resolved selection records the sole per-Instance policy evaluation
        # and its first-declaration-wins labels beside the execution grid; ``None``
        # means no budget selection was declared or a row lies outside it.
        frame["budget_group"] = pd.Series(
            [
                selection.budget_groups.get((instance.uuid, budget))
                for instance, budget in zip(frame["instance"], frame["budget"])
            ],
            dtype=object,
        )
        frame["algorithm_label"] = frame["algorithm"].map(str).astype(object)
        frame["instance_label"] = frame["instance"].map(str).astype(object)
        if selection.groups is not None and selection.seeds is not None:
            expected = sum(len(budgets) for _, _, budgets in selection.groups) * len(
                set(selection.seeds)
            )
            # A coordinate answered under more than one source-selected tree is
            # two rows in ``frame`` but one recorded cell.
            recorded = len(
                frame[
                    ["instance_uuid", "alg_uuid", "budget", "alg_seed"]
                ].drop_duplicates()
            )
            if expected > recorded:
                warnings.warn(
                    f"{expected - recorded} missing of {expected} "
                    "expected trials. Recorded Results are retained.",
                    stacklevel=2,
                )
        return frame

    def run_sweep(
        self,
        sweep: str | Path | Mapping[str, Any] | None = None,
        *,
        algorithms: Iterable[BaseAlgorithm | str] | None | _Omitted = _OMITTED,
        instances: Iterable[Instance | str] | None | _Omitted = _OMITTED,
        budgets: (
            Iterable[DPBudget | BudgetPolicy]
            | Callable[[DPBudget], bool]
            | None
            | _Omitted
        ) = _OMITTED,
        seeds: Iterable[int] | None | _Omitted = _OMITTED,
        time_limit: float | None | _Omitted = _OMITTED,
        memory_limit_gb: float | None | _Omitted = _OMITTED,
        mode: Literal["missing", "retry_resources", "all"] = "missing",
        results_source: Literal["local", "release", "both"] = "both",
        reuse_results: pd.DataFrame | None = None,
        schedule_cells: CellScheduler = schedule_cells_serially,
        verbose: bool = False,
        algorithm_registries: str | Sequence[str] | None = None,
        instance_registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Run selected cells and return their current Results, including reused ones.

        Omitted overrides inherit the Sweep; explicit ``None`` clears them.
        Execution requires finite budget and seed selections even without a
        saved file. New Attempts supersede old Attempt records without deleting them.
        Supplied Algorithm/Instance objects are not automatically registered.
        Algorithm, Instance, and registry selection always resolves across
        every installed tree; ``results_source`` decides only which tree(s)'
        recorded Results are read.

        Args:
            sweep: Saved path, settings dictionary, or ``None`` for inline choices.
            algorithms: Configured objects and/or UUIDs; ``None`` selects all
                installed algorithms in registry scope.
            instances: Instances and/or UUIDs; ``None`` selects all in scope.
            budgets: Finite iterable of typed budgets/policies, required after
                overrides. Nonprivate algorithms use one no-privacy budget. A
                predicate on each concrete budget narrows the Sweep's declared
                budgets; an unconstrained budget always passes.
            seeds: Finite iterable of algorithm seeds, required after overrides; one
                trial per seed and budget.
            time_limit: Per-cell seconds, or ``None`` to remove the saved limit.
            memory_limit_gb: Per-cell GiB, or ``None`` to remove the saved limit.
            mode: ``missing`` fills unanswered cells; ``retry_resources``
                also retries timeout/memory failures; ``all`` records a new
                Attempt for every cell.
                Existing claims are reconciled under the sole-writer contract.
            reuse_results: Successful typed Results for selected cells, using
                the columns returned by ``load_results``. Each becomes a new
                Attempt without running its Algorithm or Transformations.
                Coordinates must match the selection exactly; duplicates and
                invalid Outputs are refused before writes. Runtime and peak
                memory are unknown for these Attempts. ``mode`` governs the
                remaining cells. ``None`` (the default) supplies no Results.
            results_source: Which installed tree(s)' Results this reads:
                ``"local"``, ``"release"``, or ``"both"`` (the default) reads
                every installed tree, local first when a coordinate is
                answered in both. Decides which recorded Results already
                answer a cell for ``mode="missing"``/``"retry_resources"``;
                new Attempts are still written to ``local/`` only.
                ``results_source="local"`` reproduces an official coordinate
                locally, since its Algorithm/Instance selection is unaffected.
            schedule_cells: Scheduler owning concurrency, resource
                enforcement, and cancellation across the whole invocation. It
                receives a lazy stream of ready jobs, the library's executor,
                and a parent-side ``complete`` callback that durably records
                each outcome and returns the jobs it made ready. Finite caps
                run only under a scheduler wrapped in ``declare_limits_enforced``;
                the default serial scheduler and ``build_executor_scheduler``
                enforce none and are refused with them.
            verbose: Emit execution progress.
            algorithm_registries: Registry scope for UUID/unrestricted selection;
                one name, several names, or ``None`` for all installed groups.
            instance_registries: Corresponding Instance registry scope.
            strict: Refuse an entry this library cannot rebuild (the
                default); ``False`` skips it with a warning naming it.

        Returns:
            The same selected Results DataFrame as ``load_results`` for this selection and ``results_source``; it contains current Results only and no metric columns.

        Raises:
            TypeError: If an override has the wrong type.
            ValueError: If this handle opens a Release, if budgets/seeds or the
                Input root are absent, finite caps lack an enforcing scheduler,
                mode/settings are invalid, or declarations cannot be resolved.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from dpstats.budgets import NO_PRIVACY
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     collection.run_sweep(algorithms=[], instances=[], budgets=[NO_PRIVACY], seeds=[0]).empty
            True
        """
        self._refuse_write("Running a Sweep")
        request = self._request(
            sweep,
            algorithms=algorithms,
            instances=instances,
            budgets=budgets,
            seeds=seeds,
            time_limit=time_limit,
            memory_limit_gb=memory_limit_gb,
        )
        require_limit_enforcement(
            schedule_cells,
            time_limit=request.sweep.time_limit,
            memory_limit_gb=request.sweep.memory_limit_gb,
        )
        if request.sweep.budgets is None or request.sweep.seeds is None:
            raise ValueError("Running requires explicit finite budgets and seeds.")
        input_artifacts_root = self._writable_input_root("Running a Sweep")
        validate_mode(mode)
        selection = resolve_selection(
            request.sweep,
            self.task,
            self._task_roots,
            live_algorithms=request.algorithms,
            live_instances=request.instances,
            budgets=request.budget_filter,
            algorithm_registries=algorithm_registries,
            instance_registries=instance_registries,
            strict=strict,
        )
        for instance in selection.instances.values():
            self._stage_input_artifact(instance.input_digest)
        execute_sweep_groups(
            self._directory,
            selection.groups or (),
            selection.seeds or (),
            input_artifacts_root,
            task=self.task,
            time_limit=selection.time_limit,
            memory_limit_gb=selection.memory_limit_gb,
            mode=mode,
            answered_roots=tuple(self._roots_for_source(results_source).values()),
            reuse_results=reuse_results,
            schedule_cells=schedule_cells,
            verbose=verbose,
        )
        return self._selected_results(selection, source=results_source)

    def get_sweep_status(
        self,
        sweep: str | Path | Mapping[str, Any] | None = None,
        *,
        algorithms: Iterable[BaseAlgorithm | str] | None | _Omitted = _OMITTED,
        instances: Iterable[Instance | str] | None | _Omitted = _OMITTED,
        budgets: (
            Iterable[DPBudget | BudgetPolicy]
            | Callable[[DPBudget], bool]
            | None
            | _Omitted
        ) = _OMITTED,
        seeds: Iterable[int] | None | _Omitted = _OMITTED,
        results_source: Literal["local", "release", "both"] = "both",
        algorithm_registries: str | Sequence[str] | None = None,
        instance_registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Summarize selected coverage without loading payloads or recording Attempts.

        Omitted overrides inherit the Sweep; explicit ``None`` clears them.
        Counts concern recorded Results, not whether a worker currently holds
        a claim. Inspection does not reconcile or dispatch work.
        ``results_source`` decides which tree(s)' recorded Results are
        counted; a ``"release"`` read resolves its selection against the
        release's own registries alone, as ``load_results`` does.

        Args:
            sweep: Saved path, settings dictionary, or ``None`` for no defaults.
            algorithms: Configured objects and/or UUIDs; ``None`` selects all
                installed algorithms in registry scope.
            instances: Instances and/or UUIDs; ``None`` selects all in scope.
            budgets: Typed budgets/policies; ``None`` leaves expectations unknown.
                A predicate on each concrete budget narrows the Sweep's
                declared budgets; an unconstrained budget always passes.
            seeds: Mechanism seeds; ``None`` leaves expectations unknown.
            results_source: Which installed tree(s)' Results this reads:
                ``"local"``, ``"release"``, or ``"both"`` (the default) reads
                every installed tree, local first when a coordinate is
                answered in both.
            algorithm_registries: Registry scope for UUID/unrestricted selection;
                one name, several names, or ``None`` for all installed groups.
            instance_registries: Corresponding Instance registry scope.
            strict: Refuse an entry this library cannot rebuild (the
                default); ``False`` skips it with a warning naming it.

        Returns:
            A dictionary: ``algorithms`` and ``instances`` contain requested and
            resolved counts plus unresolved UUIDs; ``cells`` contains expected,
            recorded, resource-failure, and never-attempted counts and the
            recorded percentage. Every recorded Result is terminal, resource
            failures included. Unknown expectations use ``None`` with an
            ``expected_unavailable_reason``; zero expected cells have no
            percentage.
            ``recorded`` counts distinct cells: a coordinate answered under more than one ``results_source``-selected tree (an official coordinate also run locally) is one recorded cell here, local first, even though ``load_results`` returns it as two rows.

        Raises:
            TypeError: If an override has the wrong type.
            ValueError: If settings or installed declarations are invalid.
                Missing requested UUIDs are reported, not raised.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from dpstats.budgets import NO_PRIVACY
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     status = collection.get_sweep_status(budgets=[NO_PRIVACY], seeds=[0])
            ...     status["cells"].get("expected")
            0
        """
        request = self._request(
            sweep,
            algorithms=algorithms,
            instances=instances,
            budgets=budgets,
            seeds=seeds,
        )
        selection = resolve_selection(
            request.sweep,
            self.task,
            self._declaration_roots(results_source),
            live_algorithms=request.algorithms,
            live_instances=request.instances,
            budgets=request.budget_filter,
            algorithm_registries=algorithm_registries,
            instance_registries=instance_registries,
            allow_unresolved=True,
            strict=strict,
        )
        reason = None
        if request.sweep.budgets is None or request.sweep.seeds is None:
            reason = (
                "Finite budgets and seeds are required to determine expected cells."
            )
        elif selection.unresolved_algorithms or selection.unresolved_instances:
            reason = "Unresolved requested objects prevent resolving the complete cell population."
        expected = None
        if (
            reason is None
            and selection.groups is not None
            and selection.seeds is not None
        ):
            expected = sum(len(budgets) for _, _, budgets in selection.groups) * len(
                set(selection.seeds)
            )
        statuses = recorded_statuses(
            self._roots_for_source(results_source).values(),
            self.task,
            tuple(selection.algorithms),
            tuple(selection.instances),
            groups=selection.groups,
            seeds=selection.seeds,
        ).values()
        recorded = len(statuses)
        summary: dict[str, Any] = {
            "algorithms": _accounting(
                request.sweep.algorithms,
                selection.algorithms,
                selection.unresolved_algorithms,
            ),
            "instances": _accounting(
                request.sweep.instances,
                selection.instances,
                selection.unresolved_instances,
            ),
        }
        summary["cells"] = {
            "expected": expected,
            "recorded": recorded,
            "resource_failures": sum(status.retryable for status in statuses),
            "never_attempted": None if expected is None else expected - recorded,
            "recorded_percent": 100 * recorded / expected if expected else None,
        }
        summary["expected_unavailable_reason"] = reason
        return summary

    def get_sweep_coordinates(
        self,
        sweep: str | Path | Mapping[str, Any] | None = None,
        *,
        algorithms: Iterable[BaseAlgorithm | str] | None | _Omitted = _OMITTED,
        instances: Iterable[Instance | str] | None | _Omitted = _OMITTED,
        budgets: (
            Iterable[DPBudget | BudgetPolicy]
            | Callable[[DPBudget], bool]
            | None
            | _Omitted
        ) = _OMITTED,
        seeds: Iterable[int] | None | _Omitted = _OMITTED,
        algorithm_registries: str | Sequence[str] | None = None,
        instance_registries: str | Sequence[str] | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Export selected coordinates and declared limits for external dispatch.

        All selected coordinates are exported, including already recorded ones.
        This neither chooses pending work nor executes or claims a cell.

        Args:
            sweep: Saved path, settings dictionary, or ``None`` for inline choices.
            algorithms: Configured objects and/or UUIDs; ``None`` selects all
                installed algorithms in registry scope.
            instances: Instances and/or UUIDs; ``None`` selects all in scope.
            budgets: Finite typed budget/policy selection, inherited when omitted,
                or a predicate on each concrete budget narrowing the Sweep's
                declared budgets; an unconstrained budget always passes.
            seeds: Finite mechanism seed selection, inherited when omitted.
            algorithm_registries: Registry scope for UUID/unrestricted selection;
                one name, several names, or ``None`` for all installed groups.
            instance_registries: Corresponding Instance registry scope.
            strict: Refuse an entry this library cannot rebuild (the
                default); ``False`` skips it with a warning naming it.

        Returns:
            A DataFrame with ``coordinate_uuid``, ``task``, ``alg_uuid``,
            ``instance_uuid``, live ``budget``, ``alg_seed``, ``time_limit`` in
            seconds, and ``memory_limit_gb`` in GiB. Uncapped limits are ``None``.

        Raises:
            TypeError: If an override has the wrong type.
            ValueError: If budgets/seeds are absent, settings are invalid, or
                selected UUIDs cannot be resolved.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from dpstats.budgets import NO_PRIVACY
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     collection.get_sweep_coordinates(algorithms=[], instances=[], budgets=[NO_PRIVACY], seeds=[0]).empty
            True
        """
        request = self._request(
            sweep,
            algorithms=algorithms,
            instances=instances,
            budgets=budgets,
            seeds=seeds,
        )
        if request.sweep.budgets is None or request.sweep.seeds is None:
            raise ValueError("Coordinate export requires finite budgets and seeds.")
        selection = resolve_selection(
            request.sweep,
            self.task,
            self._task_roots,
            live_algorithms=request.algorithms,
            live_instances=request.instances,
            budgets=request.budget_filter,
            algorithm_registries=algorithm_registries,
            instance_registries=instance_registries,
            strict=strict,
        )
        coordinates = selection.coordinates()
        return pd.DataFrame(
            [
                {
                    "coordinate_uuid": cell.uuid,
                    "task": self.task.id,
                    "alg_uuid": cell.algorithm.uuid,
                    "instance_uuid": cell.instance.uuid,
                    "budget": cell.budget,
                    "alg_seed": cell.alg_seed,
                    "time_limit": selection.time_limit,
                    "memory_limit_gb": selection.memory_limit_gb,
                }
                for cell in coordinates
            ],
            columns=[
                "coordinate_uuid",
                "task",
                "alg_uuid",
                "instance_uuid",
                "budget",
                "alg_seed",
                "time_limit",
                "memory_limit_gb",
            ],
        )

    def reconcile(self) -> list[str]:
        """Settle abandoned journals and claims without selecting a Sweep.

        Run only after every collection worker has stopped. Durable journals
        are recovered before the claims a vanished runner left are cleared.
        This writes to the collection and never executes an Algorithm.
        The collection already binds its Task, so reconciliation never loads
        Algorithm registries. Only ``local/`` claims are settled.

        Returns:
            The coordinates whose claims were cleared, in claim order. Each of
            those cells is unanswered unless this call recovered its runner's
            journal for it.

        Raises:
            ValueError: If this handle opens a Release, or journals/claims cannot
                be settled. Successfully settled siblings remain recorded when
                another item fails.
            RuntimeError: If invoked by a spawned worker during import.

        Example:
            >>> from dpstats.workspace.layout import init_workspace
            >>> from tempfile import TemporaryDirectory
            >>> from dpstats.tasks.mean_estimation.algorithms import SampleMean
            >>> with TemporaryDirectory() as root:
            ...     collection = open_collection(init_workspace(root), task_id="mean_estimation")
            ...     _ = collection.add_algorithm(SampleMean())
            ...     collection.reconcile()
            []
        """
        self._refuse_write("Reconciling")
        return reconcile_orphaned_work(self._directory, self.task)


@overload
def open_collection(
    workspace: str | Path,
    task: Task[_CollectionMetadataT, _CollectionInputT, OutputT],
    *,
    task_id: None = None,
) -> TaskCollection[_CollectionMetadataT, _CollectionInputT, OutputT]: ...


@overload
def open_collection(
    workspace: str | Path,
    task: Task | None = None,
    *,
    task_id: str | None = None,
) -> TaskCollection: ...


def open_collection(
    workspace: str | Path,
    task: Task | None = None,
    *,
    task_id: str | None = None,
) -> TaskCollection:
    """Open one Task's view of a workspace, for authoring and the lifecycle over it.

    ``workspace`` is either a workspace (a directory holding ``local/``, every
    write's destination) or a bare Distribution Release bundle (a directory
    holding ``release.json`` directly, with no ``local/``), opened read-only.
    Anything else is refused. Registries, Sweeps, and Results are read
    across both a workspace's trees; ``add_instance``, ``add_algorithm``,
    ``save_sweep``, ``TaskCollection.run_sweep``, and ``reconcile`` write ``local/`` only and
    refuse on a bare bundle.

    Args:
        workspace: The workspace or bare bundle to open.
        task: The live Task whose collection this handle opens.
        task_id: A built-in Task id. This is resolved at opening; custom Tasks
            must be passed as ``task``.

    Returns:
        A handle over the resolved tree(s), with no Attempts read or written
        and no default Sweep selected by opening it.

    Raises:
        ValueError: If ``workspace`` is neither a workspace nor a bare bundle.

    Example:
        >>> from dpstats.workspace.layout import init_workspace
        >>> from tempfile import TemporaryDirectory
        >>> from dpstats.tasks import MeanEstimation
        >>> with TemporaryDirectory() as root:
        ...     workspace = init_workspace(root)
        ...     collection = open_collection(workspace, MeanEstimation)
        ...     collection.task, collection.release
        (Task(id='mean_estimation', name='Mean Estimation'), None)
    """
    if (task is None) == (task_id is None):
        raise ValueError("Provide exactly one of task or task_id.")
    if task_id is not None:
        from dpstats.tasks import BUILT_IN_TASKS

        try:
            task = BUILT_IN_TASKS[task_id]
        except KeyError as error:
            raise ValueError(
                f"Unknown built-in Task {task_id!r}; pass a live custom Task object."
            ) from error
    if not isinstance(task, Task):
        raise TypeError("task must be a Task object.")
    root = Path(workspace)
    local, release = workspace_roots(root)
    if local is not None or release is not None:
        return TaskCollection(
            _workspace=root, task=task, _local_root=local, _release_root=release
        )
    raise ValueError(
        f"{root} is neither a workspace (a directory with a local/ tree) nor "
        "a bare Distribution Release bundle (a directory with release.json "
        "directly inside it)."
    )
