"""Execution, cleanup, and operational preview of benchmark Sweeps."""

from __future__ import annotations

import logging
import json
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias, cast

import pandas as pd
import pyarrow as pa
from tqdm.auto import tqdm

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    Transformation,
)
from dpstats.budgets import DPBudget
from dpstats.exceptions import NoRelease
from dpstats.instance import Instance, identify_instance
from dpstats.task import Task, TaskInput, TaskOutput
from dpstats.workspace.layout import task_collection_directory
from dpstats.results import (
    ResultAttempt,
    ResultStatus,
    results_frame,
)
from dpstats.sweep._cell import (
    CellContext,
    result_attempt,
    run_coordinate,
    validate_output,
)
from dpstats.sweep.preparation import PreparedCell
from dpstats.sweep._claims import (
    claim_attempt,
    reconcile_orphaned_work,
    release_attempt,
    standing_claims,
)
from dpstats.results._frame import RESULT_COLUMNS
from dpstats.results._journal import AttemptWriter
from dpstats.results._resolution import (
    ResultLookup,
    current_outcomes,
    recorded_statuses,
    selected_attempts,
)
from dpstats.results._selection import SELECTION_ROW
from dpstats.coordinates import CellCoordinate, coordinate_record_identity
from dpstats.sweep.coordinates import (
    expand_sweep_groups,
    resolve_sweep_selection,
    sweep_selection_table,
)
from dpstats.sweep._sweep import read_sweep

logger = logging.getLogger("benchmark.sweep")


CellScheduler: TypeAlias = Callable[
    [
        Iterable[CellContext],
        Callable[[CellContext], PreparedCell | ResultAttempt],
        Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ],
    None,
]

_Restorations: TypeAlias = tuple[tuple[Transformation, Any], ...]
_Key: TypeAlias = tuple[str, int]
_PreparationKey: TypeAlias = tuple[tuple[Transformation, ...], DPBudget, int]
_ScheduledOutcome: TypeAlias = PreparedCell | ResultAttempt

_PREVIEW_COLUMNS = [
    "coordinate_uuid",
    "task",
    "algorithm",
    "alg_uuid",
    "instance_uuid",
    "budget",
    "alg_seed",
    "action",
    "reason",
    "claim",
    "time_limit",
    "memory_limit_gb",
]


def schedule_cells_serially(
    initial: Iterable[CellContext],
    execute: Callable[[CellContext], PreparedCell | ResultAttempt],
    complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
) -> None:
    """Execute a Sweep's offered work serially, in process, enforcing no limits.

    Work that ``complete`` makes ready runs before the next initial job, so
    a composite's leaves finish before the next Instance's Input is loaded.

    Args:
        initial: Lazy stream of the first ready jobs.
        execute: Runs one offered preparation or leaf.
        complete: Acknowledges one outcome and returns newly ready jobs.

    Example:
        >>> schedule_cells_serially([], print, lambda outcome: [])
    """
    pending: deque[CellContext] = deque()
    source = iter(initial)
    while True:
        if not pending:
            following = next(source, None)
            if following is None:
                return
            pending.append(following)
        pending.extend(complete(execute(pending.popleft())))


def build_executor_scheduler(executor: Executor, workers: int) -> CellScheduler:
    """Schedule a Sweep's jobs on a ``concurrent.futures`` executor.

    At most ``workers`` jobs are in flight; set it to the executor's worker
    count. Outcomes are acknowledged in the calling thread. It enforces no
    resource limit, so ``TaskCollection.run_sweep`` refuses a Sweep that
    declares one under this scheduler; wrap an adapter that does enforce
    them in ``declare_limits_enforced``.

    A job whose future raises, rather than returning an outcome, is a
    machinery incident: its cell records no Attempt and is left pending, no
    further job is submitted, every job still in flight is acknowledged when
    it finishes, and the first such exception is then re-raised.

    Args:
        executor: A thread or process pool the caller owns and shuts down.
        workers: Jobs submitted at once.

    Returns:
        A scheduler for ``TaskCollection.run_sweep(schedule_cells=...)``.

    Raises:
        ValueError: If ``workers`` is less than one.

    Example:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with ThreadPoolExecutor(max_workers=4) as pool:
        ...     schedule = build_executor_scheduler(pool, 4)
    """
    if workers < 1:
        raise ValueError("workers must be at least one.")

    def schedule(
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        source = iter(initial)
        ready: deque[CellContext] = deque()
        running: set[Future[PreparedCell | ResultAttempt]] = set()
        exhausted = False
        failure: Exception | None = None
        try:
            while True:
                while failure is None and len(running) < workers:
                    if not ready and not exhausted:
                        following = next(source, None)
                        exhausted = following is None
                        if following is not None:
                            ready.append(following)
                    if not ready:
                        break
                    running.add(executor.submit(execute, ready.popleft()))
                if not running:
                    if failure is not None:
                        raise failure
                    return
                done, running = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        outcome = future.result()
                    except Exception as error:
                        if failure is None:
                            failure = error
                        continue
                    ready.extend(complete(outcome))
        finally:
            for future in running:
                future.cancel()
            wait(running)

    return schedule


class _LimitEnforcing:
    """A scheduler its author declared to enforce each cell's limits."""

    def __init__(self, schedule_cells: CellScheduler) -> None:
        self._schedule_cells = schedule_cells

    def __call__(
        self,
        initial: Iterable[CellContext],
        execute: Callable[[CellContext], PreparedCell | ResultAttempt],
        complete: Callable[[PreparedCell | ResultAttempt], Iterable[CellContext]],
    ) -> None:
        self._schedule_cells(initial, execute, complete)


def declare_limits_enforced(schedule_cells: CellScheduler) -> CellScheduler:
    """Declare that a scheduler enforces each cell's time and memory limits.

    A Sweep declaring ``time_limit`` or ``memory_limit_gb`` runs only under a
    scheduler passed through this, so a declared limit is never silently
    unenforced: the library's own schedulers enforce nothing and are refused
    with it. The declaration is the adapter author's; the library checks
    nothing beyond it, and the adapter reports a cell it stops as a
    ``timeout`` or ``memory_limit_exceeded`` Attempt.

    Args:
        schedule_cells: An adapter that stops a cell past its context's
            ``time_limit_s`` or ``memory_limit_bytes``.

    Returns:
        The same scheduler, accepted by ``TaskCollection.run_sweep`` with
        finite limits.

    Example:
        >>> def bounded_schedule(initial, execute, complete):
        ...     ...  # runs each job under its context's limits
        >>> schedule = declare_limits_enforced(bounded_schedule)
    """
    return _LimitEnforcing(schedule_cells)


def require_limit_enforcement(
    schedule_cells: CellScheduler,
    *,
    time_limit: float | None,
    memory_limit_gb: float | None,
) -> None:
    """Refuse finite limits under a scheduler not declared to enforce them."""
    if time_limit is None and memory_limit_gb is None:
        return
    if isinstance(schedule_cells, _LimitEnforcing):
        return
    raise ValueError(
        "Finite resource limits require an enforcing schedule_cells callable: "
        "wrap a scheduler that enforces them in declare_limits_enforced, or run with "
        "time_limit=None and memory_limit_gb=None."
    )


def validate_mode(mode: str) -> None:
    """Refuse a run mode outside the three the executor understands."""
    if mode not in {"missing", "retry_resources", "all"}:
        raise ValueError("mode must be 'missing', 'retry_resources', or 'all'.")


def answers_coordinate(status: ResultStatus, *, retry_resource_failures: bool) -> bool:
    """Whether one current Result leaves its coordinate nothing to execute.

    A ``timeout`` or ``memory_limit_exceeded`` Result answers the cell under
    the limits it ran with, and only an explicit request reopens it. Every
    other terminal outcome is final.
    """
    return not (retry_resource_failures and status.retryable)


def preview_sweep(
    sweep_path: str | Path,
    task_collections_root: str | Path,
    task: Task,
    *,
    algorithm_registries: str | Sequence[str] | None = None,
    instance_registries: str | Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
    budgets: Sequence[DPBudget] | None = None,
    seeds: Sequence[int] | None = None,
    recompute: bool = False,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Preview a sweep's selected coordinates and current Result gaps.

    Resolve every coordinate and report its current execution action. Preview
    is a reader: it executes no cell and writes nothing, so it is safe
    against a task collection a run is still writing. A coordinate some
    invocation has taken and not yet accounted for is reported in the ``claim``
    column rather than decided; ``TaskCollection.reconcile`` is what decides it.

    It reports what is pending from the same Attempts ``TaskCollection.run_sweep`` executes
    from — the newest Attempt each selected coordinate has — so a store several
    runs have filled is reported as the work it actually still has.

    Each row also carries the key that addresses its cell and the per-cell
    resource envelope the Sweep declares for it.

    ``columns`` projects the returned frame in the requested order. In addition
    to the default columns, ``status`` exposes the current Result's status or
    ``None`` for a coordinate with no recorded Attempt. Summary readers can select only ``action``
    and avoid constructing cell identities, budget dictionaries, and display
    explanations. Projection does not weaken current-Attempt conflict checks.

    Args:
        sweep_path: The authored one-task Sweep.
        task_collections_root: Exact root whose direct children are task
            collections.
        task: The live Task the saved Sweep must declare.
        algorithm_registries: Named installed Algorithm groups to select, or
            every installed group when omitted.
        instance_registries: Named installed Instance groups to select, or
            every installed group when omitted.
        algorithm_ids: Installed Algorithm UUIDs to retain after registry
            selection, or no further restriction when omitted.
        instance_ids: Installed Instance UUIDs to retain after registry
            selection, or no further restriction when omitted.
        budgets: Concrete typed budgets to retain after live policy resolution,
            or no further restriction when omitted.
        seeds: Declared algorithm seeds to retain, or no further
            restriction when omitted.
        recompute: Marks every answered selected coordinate as ``recompute``
            while leaving missing and already-pending coordinates on their
            ordinary actions. It remains a read-only preview.
        columns: The returned columns, in this order, from the default set
            plus ``status``; every column is returned when omitted.

    Returns:
        One row per selected coordinate. The default columns are
        ``coordinate_uuid``, ``task``, ``algorithm``, ``alg_uuid``,
        ``instance_uuid``, ``budget``, ``alg_seed``, ``action``, ``reason``,
        ``claim``, ``time_limit``, and ``memory_limit_gb``.

    Raises:
        ValueError: If ``columns`` names a column outside the default set and
            ``status``.

    Example:
        >>> from dpstats.tasks import MeanEstimation  # doctest: +SKIP
        >>> preview = preview_sweep(  # doctest: +SKIP
        ...     "<workspace>/local/mean_estimation/sweeps/local.json",
        ...     "<workspace>/local",
        ...     MeanEstimation,
        ... )
        >>> preview["action"].value_counts()  # doctest: +SKIP
    """
    sweep = read_sweep(sweep_path)
    requested = list(_PREVIEW_COLUMNS if columns is None else columns)
    unknown = set(requested) - {*_PREVIEW_COLUMNS, "status"}
    if unknown:
        raise ValueError(f"Unknown preview columns: {sorted(unknown)}.")
    selected = resolve_sweep_selection(
        sweep,
        task_collections_root,
        task,
        algorithm_registries=algorithm_registries,
        instance_registries=instance_registries,
        algorithm_ids=algorithm_ids,
        instance_ids=instance_ids,
        budgets=budgets,
        seeds=seeds,
    )
    resolved = selected.groups or ()
    if not resolved:
        return pd.DataFrame(columns=requested).astype(object)
    selection = sweep_selection_table(resolved, selected.seeds or ())
    selection = selection.append_column(
        SELECTION_ROW, pa.array(range(selection.num_rows))
    )
    task_collection = task_collection_directory(task_collections_root, sweep.task)
    current = selected_attempts(task_collection, task, selection, columns=["status"])
    actions: dict[str | None, str] = {None: "missing"}
    reasons = {None: "the task collection holds no current Result for this coordinate"}
    for value in current["status"].unique().to_pylist():
        status = ResultStatus(value)
        actions[value] = (
            "recompute"
            if recompute
            else "retryable" if status.retryable else "reusable"
        )
        reasons[value] = f"the current Result is {status.value}"
    snapshot = selection.join(current, keys=SELECTION_ROW, join_type="left outer")
    frame = (
        snapshot.sort_by(SELECTION_ROW)
        .to_pandas()
        .rename(
            columns={
                "instance_id": "instance_uuid",
                "alg_id": "alg_uuid",
                "seed": "alg_seed",
            }
        )
    )
    frame["status"] = (
        frame["status"].astype(object).where(frame["status"].notna(), None)
    )
    frame["action"] = frame["status"].map(actions)
    if "reason" in requested:
        frame["reason"] = frame["status"].map(reasons)
    needs_identity = "coordinate_uuid" in requested or "claim" in requested
    if "budget" in requested or needs_identity:
        frame["budget"] = pd.Series(
            [
                {"path": path, "config": json.loads(config)}
                for path, config in zip(frame["budget_path"], frame["budget_config"])
            ],
            dtype=object,
        )
    if needs_identity:
        frame["coordinate_uuid"] = [
            coordinate_record_identity(
                {
                    "instance_uuid": instance,
                    "alg_uuid": algorithm,
                    "budget": budget,
                    "alg_seed": seed,
                }
            )
            for instance, algorithm, budget, seed in zip(
                frame["instance_uuid"],
                frame["alg_uuid"],
                frame["budget"],
                frame["alg_seed"],
            )
        ]
    if "claim" in requested:
        claimed = {
            claim["coordinate_uuid"]: (
                f"attempt {claim['attempt_number']} held by pid {claim['pid']}"
            )
            for claim in standing_claims(task_collection)
        }
        frame["claim"] = frame["coordinate_uuid"].map(claimed).fillna("")
    for name, value in (
        ("task", sweep.task),
        ("time_limit", sweep.time_limit),
        ("memory_limit_gb", sweep.memory_limit_gb),
    ):
        if name in requested:
            frame[name] = value
    return frame[requested].astype(object)


def _project(consumer: _Consumer, leaf: ResultAttempt) -> ResultAttempt:
    """Restore one exact leaf outcome under one official outer coordinate.

    A leaf that refused a budget an earlier stage had reduced answers its
    outer coordinate as ``no_release`` carrying the leaf's message, the
    composition's accounted bottom; the leaf's own Result is kept.
    """
    context, started_at, restorations = consumer
    output = leaf.output
    if output is None:
        if (
            leaf.status is ResultStatus.INVALID_REGIME
            and leaf.coordinate.budget != context.coordinate.budget
        ):
            return result_attempt(
                context,
                NoRelease(
                    f"{type(leaf.coordinate.algorithm).__name__} refused its "
                    f"regime after an earlier stage spent budget: {leaf.payload}"
                ),
                started_at=started_at,
            )
        return result_attempt(context, leaf, started_at=started_at)
    outer = context.coordinate.algorithm
    try:
        if isinstance(outer, CompositeAlgorithm):
            output = outer.restore(output, restorations)
    except Exception as error:
        return result_attempt(context, error, started_at=started_at)
    return result_attempt(context, output, started_at=started_at)


def _prepare_coordinate(context: CellContext) -> PreparedCell | ResultAttempt:
    """Prepare one composite anchor and measure only transformation execution.

    The caller chooses the scheduler and owns any resource classification it
    returns. Native transformation refusals and errors retain their ordinary
    ``ResultAttempt`` classification under the representative outer cell.
    """
    started_at = time.perf_counter()
    coordinate = context.coordinate
    algorithm = coordinate.algorithm
    if not isinstance(algorithm, CompositeAlgorithm):
        raise TypeError("Composite preparation requires a CompositeAlgorithm anchor.")
    original = context.resolve_input()
    try:
        metadata, input, budget, seed, restorations = algorithm.process(
            coordinate.instance.metadata,
            original,
            coordinate.budget,
            coordinate.alg_seed,
        )
    except Exception as error:
        return result_attempt(context, error, started_at=started_at)
    return PreparedCell(
        coordinate=coordinate,
        attempt_number=context.attempt_number,
        metadata=metadata,
        input=input,
        budget=budget,
        seed=seed,
        restorations=restorations,
        runtime_s=time.perf_counter() - started_at,
    )


class _Execute:
    """The lifecycle's execute step: load a cell's Input where it runs, then run it.

    A cell of an original Instance is offered with the artifact root as its
    ``input``; this loads it in the process that executes the cell, so an
    adapter's child loads under the deadline and cap it enforces. The Input
    last loaded is kept, so the serial route, which runs one Instance's cells
    before the next Instance is offered, loads each original Input once. A
    process pool that pickles the step per job starts each job with nothing
    loaded and so loads per cell.

    A load the adapter's declared cap refuses is that cell's
    ``memory_limit_exceeded``. Any other load failure is raised: an unreadable
    artifact is a machinery incident, not an outcome, so the cell records no
    Attempt and its claim is released.
    """

    def __init__(self) -> None:
        self._loaded_from: tuple[str, Path] | None = None
        self._loaded: TaskInput | None = None

    def __reduce__(self) -> tuple[type[_Execute], tuple[()]]:
        """Pickle as a fresh step: a loaded Input never crosses to another process."""
        return (_Execute, ())

    def __call__(self, context: CellContext) -> _ScheduledOutcome:
        started_at = time.perf_counter()
        if isinstance(context.input, Path):
            try:
                context = replace(context, input=self._load(context, context.input))
            except MemoryError as error:
                if context.memory_limit_bytes is None:
                    raise
                return result_attempt(context, error, started_at=started_at)
        if isinstance(context.coordinate.algorithm, CompositeAlgorithm):
            return _prepare_coordinate(context)
        return run_coordinate(context)

    def _load(self, context: CellContext, root: Path) -> TaskInput:
        source = (context.coordinate.instance_uuid, root)
        if self._loaded is None or self._loaded_from != source:
            self._loaded = context.resolve_input()
            self._loaded_from = source
        return self._loaded


def _key(job: CellContext | _ScheduledOutcome) -> _Key:
    return (job.coordinate.uuid, job.attempt_number)


class _Consumer(NamedTuple):
    """One claimed outer cell, with how its leaf's outcome is restored for it."""

    context: CellContext
    started_at: float
    restorations: _Restorations = ()


class _Prepared(NamedTuple):
    """One shared prepared Input and the pending leaves still reading it."""

    input: TaskInput
    readers: set[str]


class _Run:
    """The state one invocation's lifecycle steps share."""

    def __init__(
        self,
        contexts: Sequence[CellContext],
        write_attempt: Callable[[ResultAttempt], None],
        *,
        reuse_durable: bool,
        retry_resource_failures: bool,
        supplied_outputs: Mapping[str, TaskOutput],
        verbose: bool,
    ) -> None:
        self.contexts = contexts
        self.write_attempt = write_attempt
        self.reuse_durable = reuse_durable
        self.retry_resource_failures = retry_resource_failures
        self.supplied_outputs = supplied_outputs
        self.progress = tqdm(total=len(contexts), unit="attempt") if verbose else None
        self.claimed: dict[str, CellContext] = {}
        self.in_flight: dict[_Key, CellContext] = {}
        self.consumers: dict[_Key, list[_Consumer]] = {}
        self.projections: dict[str, list[_Consumer]] = {}
        self.prepared: dict[str, _Prepared] = {}
        self.leaf_cache: dict[str, ResultAttempt] = {}
        self.lookups: dict[Task, ResultLookup] = {}
        self.recorded = self.dispatched = self.reused = 0

    def offer(self) -> Iterator[CellContext]:
        """Claim and plan each original Instance lazily, as the scheduler pulls."""
        by_instance: dict[str, list[CellContext]] = defaultdict(list)
        for context in self.contexts:
            by_instance[context.coordinate.instance_uuid].append(context)
        for group in by_instance.values():
            claims: list[_Consumer] = []
            for context in group:
                claim_attempt(
                    context.task_collection,
                    context.coordinate,
                    context.attempt_number,
                )
                self.claimed[context.coordinate.uuid] = context
                claims.append(_Consumer(context, time.perf_counter()))
            yield from self.plan_instance(claims)

    def plan_instance(self, claims: Sequence[_Consumer]) -> Iterator[CellContext]:
        """Offer one original Instance's composite anchors, then its direct leaves.

        Each offered job carries the artifact root as its ``input``; the
        execute step loads the Input where the job runs. Composite cells
        sharing a transformation chain, budget and seed share one
        preparation, run by the first of them as the anchor.
        """
        anchors: dict[_PreparationKey, CellContext] = {}
        direct: list[CellContext] = []
        for claim in claims:
            coordinate = claim.context.coordinate
            if coordinate.uuid in self.supplied_outputs:
                self.answer(
                    ResultAttempt(
                        coordinate=coordinate,
                        attempt_number=claim.context.attempt_number,
                        status=ResultStatus.SUCCESS,
                        payload=self.supplied_outputs[coordinate.uuid],
                    )
                )
                self.reused += 1
                continue
            algorithm = coordinate.algorithm
            if not isinstance(algorithm, CompositeAlgorithm):
                leaf = self.add_leaf(claim, coordinate, claim.context.input, None)
                if leaf is not None:
                    direct.append(leaf)
                continue
            shared = (algorithm.transformations, coordinate.budget, coordinate.alg_seed)
            if shared not in anchors:
                anchors[shared] = claim.context
                self.consumers[_key(claim.context)] = []
            self.consumers[_key(anchors[shared])].append(claim)
        for anchor in anchors.values():
            self.in_flight[_key(anchor)] = anchor
            yield anchor
        yield from self.admit(direct)

    def add_leaf(
        self,
        consumer: _Consumer,
        coordinate: CellCoordinate,
        input: TaskInput | Path,
        preparation_runtime_s: float | None,
    ) -> CellContext | None:
        """Attach one outer consumer to its leaf call, returning a newly added leaf."""
        context = consumer.context
        added = None
        if coordinate.uuid not in self.projections:
            self.projections[coordinate.uuid] = []
            time_limit_s = context.time_limit_s
            if preparation_runtime_s is not None and time_limit_s is not None:
                time_limit_s = max(0.0, time_limit_s - preparation_runtime_s)
            added = replace(
                context,
                coordinate=coordinate,
                task_collection=task_collection_directory(
                    context.task_collection.parent, coordinate.task.id
                ),
                input=input,
                time_limit_s=time_limit_s,
            )
        self.projections[coordinate.uuid].append(consumer)
        return added

    def admit(self, added: Sequence[CellContext]) -> list[CellContext]:
        """Answer leaves already known, and offer the rest for execution."""
        if self.reuse_durable:
            self.reuse_leaves(added)
        offered: list[CellContext] = []
        for leaf in added:
            cached = self.leaf_cache.get(leaf.coordinate.uuid)
            if cached is not None:
                self.reused += 1
                self.finish_leaf(cached)
                continue
            self.dispatched += 1
            self.in_flight[_key(leaf)] = leaf
            offered.append(leaf)
        return offered

    def reuse_leaves(self, leaves: Sequence[CellContext]) -> None:
        """Add eligible durable Results for these exact leaves to the run cache."""
        unresolved: dict[Task, dict[str, CellContext]] = defaultdict(dict)
        for leaf in leaves:
            if leaf.coordinate.uuid not in self.leaf_cache:
                unresolved[leaf.coordinate.task][leaf.coordinate.uuid] = leaf
        for task, by_uuid in unresolved.items():
            if task not in self.lookups:
                first = next(iter(by_uuid.values()))
                self.lookups[task] = ResultLookup(first.task_collection, task)
            coordinates = [leaf.coordinate for leaf in by_uuid.values()]
            for record in self.lookups[task].records(coordinates):
                coordinate = by_uuid[record["coordinate_uuid"]].coordinate
                result = ResultAttempt.from_resolved_record(
                    record, coordinate, task=task
                )
                if answers_coordinate(
                    result.status,
                    retry_resource_failures=self.retry_resource_failures,
                ):
                    self.leaf_cache[coordinate.uuid] = result

    def acknowledge(self, outcome: object) -> CellContext:
        """Match one completed outcome to the job it answers, refusing any other."""
        if not isinstance(outcome, (PreparedCell, ResultAttempt)):
            raise TypeError(
                "A Sweep scheduler must complete ResultAttempt values for leaves "
                "or PreparedCell values for preparations."
            )
        key = _key(outcome)
        context = self.in_flight.pop(key, None)
        if context is None:
            raise ValueError(
                f"A Sweep scheduler completed an unplanned coordinate or Attempt: {key}."
            )
        composite = isinstance(context.coordinate.algorithm, CompositeAlgorithm)
        if isinstance(outcome, PreparedCell) and not composite:
            raise TypeError(
                "A Sweep scheduler must complete ResultAttempt values for leaves."
            )
        if (
            isinstance(outcome, ResultAttempt)
            and composite
            and outcome.status is ResultStatus.SUCCESS
        ):
            raise ValueError(
                "A Sweep scheduler may complete a composite preparation only "
                "with a failed or refused ResultAttempt, never a success."
            )
        return context

    def complete(self, outcome: _ScheduledOutcome) -> list[CellContext]:
        """Acknowledge one outcome and return the jobs it made ready."""
        context = self.acknowledge(outcome)
        if isinstance(outcome, ResultAttempt):
            if isinstance(context.coordinate.algorithm, CompositeAlgorithm):
                for consumer in self.consumers.pop(_key(context)):
                    self.answer(
                        result_attempt(
                            consumer.context, outcome, started_at=consumer.started_at
                        )
                    )
            else:
                self.finish_leaf(outcome)
            return []
        instance = identify_instance(outcome.metadata, outcome.input, {})
        prepared = self.prepared.get(instance.uuid, _Prepared(outcome.input, set()))
        added: list[CellContext] = []
        for consumer in self.consumers.pop(_key(context)):
            # Every consumer of an anchor is a composite sharing its chain.
            outer = cast(CompositeAlgorithm, consumer.context.coordinate.algorithm)
            leaf = self.add_leaf(
                consumer._replace(restorations=outcome.restorations),
                CellCoordinate(
                    instance=instance,
                    algorithm=outer.algorithm,
                    budget=outcome.budget,
                    alg_seed=outcome.seed,
                ),
                prepared.input,
                outcome.runtime_s,
            )
            if leaf is not None:
                prepared.readers.add(leaf.coordinate.uuid)
                added.append(leaf)
        if prepared.readers:
            self.prepared[instance.uuid] = prepared
        return self.admit(added)

    def finish_leaf(self, leaf: ResultAttempt) -> None:
        """Answer every outer cell waiting on one leaf, then retire the leaf call."""
        leaf_id = leaf.coordinate.uuid
        self.leaf_cache[leaf_id] = leaf
        for consumer in self.projections.pop(leaf_id):
            self.answer(_project(consumer, leaf))
        prepared = self.prepared.get(leaf.coordinate.instance_uuid)
        if prepared is not None:
            prepared.readers.discard(leaf_id)
            if not prepared.readers:
                del self.prepared[leaf.coordinate.instance_uuid]

    def answer(self, attempt: ResultAttempt) -> None:
        """Record one outer cell's Attempt, then drop its claim."""
        self.write_attempt(attempt)
        context = self.claimed.pop(attempt.coordinate.uuid)
        release_attempt(
            context.task_collection, context.coordinate, context.attempt_number
        )
        self.recorded += 1
        if self.progress is not None:
            self.progress.update()

    def abandon(self) -> None:
        """Release every claim still standing, leaving those cells pending."""
        while self.claimed:
            _, context = self.claimed.popitem()
            release_attempt(
                context.task_collection, context.coordinate, context.attempt_number
            )


def _schedule_attempts(
    contexts: Sequence[CellContext],
    write_attempt: Callable[[ResultAttempt], None],
    schedule_cells: CellScheduler,
    *,
    verbose: bool,
    reuse_durable: bool,
    retry_resource_failures: bool,
    supplied_outputs: Mapping[str, TaskOutput],
) -> None:
    """Schedule pending cells while the parent owns every durable transition.

    Original Instances are offered lazily, in order, as the scheduler pulls
    the initial stream. A preparation's leaves are returned by ``complete``
    as soon as it is acknowledged, so the scheduler may run them beside other
    Instances' work. Every exact leaf executes at most once per invocation.
    Every claim this takes and does not answer is released on every exit, so
    an unrecorded cell is pending again.

    ``reuse_durable`` is whether a durable local Result may answer a leaf: a
    leaf's identity coincides with its outer coordinate for a non-composite
    algorithm, so an excluded local Result must not silently answer a cell
    the outer selection decided must run.
    """
    if not contexts:
        return
    run = _Run(
        contexts,
        write_attempt,
        reuse_durable=reuse_durable,
        retry_resource_failures=retry_resource_failures,
        supplied_outputs=supplied_outputs,
        verbose=verbose,
    )
    logger.info(
        "Sweep schedules %d pending outer cell(s) across %d original Instance(s).",
        len(contexts),
        len({context.coordinate.instance_uuid for context in contexts}),
    )
    try:
        schedule_cells(run.offer(), _Execute(), run.complete)
        if run.in_flight or run.recorded != len(contexts):
            examples = ", ".join(map(str, sorted(run.in_flight)[:3]))
            raise ValueError(
                "A Sweep scheduler completed no outcome for "
                f"{len(contexts) - run.recorded} outer cell(s): {examples}."
            )
        logger.info(
            "Sweep recorded %d outer cell(s); %d unique leaf/leaves dispatched and "
            "%d answered by result reuse.",
            run.recorded,
            run.dispatched,
            run.reused,
        )
    finally:
        if run.progress is not None:
            run.progress.close()
        run.abandon()


def _supplied_outputs(
    frame: pd.DataFrame | None,
    task: Task,
    coordinates: Sequence[CellCoordinate],
) -> dict[str, TaskOutput]:
    """Validate the whole supplied frame before any durable sweep transition."""
    if frame is None:
        return {}
    if frame.empty:
        return {}
    columns = [
        "task",
        "instance",
        "instance_uuid",
        "algorithm",
        "alg_uuid",
        "budget",
        "alg_seed",
        "status",
        "payload",
    ]
    if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
        raise ValueError(
            "reuse_results requires unique typed Results columns: "
            + ", ".join(columns)
            + "."
        )
    selected = {coordinate.uuid for coordinate in coordinates}
    supplied: dict[str, TaskOutput] = {}
    for (
        task_id,
        instance,
        instance_uuid,
        algorithm,
        alg_uuid,
        budget,
        seed,
        status,
        payload,
    ) in frame[columns].itertuples(index=False, name=None):
        try:
            coordinate = CellCoordinate(
                instance=instance, algorithm=algorithm, budget=budget, alg_seed=seed
            )
        except (TypeError, ValueError):
            raise ValueError(
                "reuse_results has an invalid coordinate; supply typed "
                "Instance, Algorithm, budget and integer seed values."
            ) from None
        if (
            not all(
                isinstance(value, str) for value in (task_id, instance_uuid, alg_uuid)
            )
            or task_id != task.id
            or coordinate.task is not task
            or instance_uuid != coordinate.instance_uuid
            or alg_uuid != coordinate.alg_uuid
        ):
            raise ValueError(
                "reuse_results Task and UUID columns must match its typed coordinate."
            )
        if coordinate.uuid not in selected:
            raise ValueError(
                "reuse_results contains a coordinate outside the selection."
            )
        if coordinate.uuid in supplied:
            raise ValueError(
                "reuse_results contains duplicate coordinates; supply one per cell."
            )
        if status is not ResultStatus.SUCCESS or not isinstance(payload, task.Output):
            raise ValueError(
                "reuse_results accepts only successful Results with the target TaskOutput."
            )
        try:
            validate_output(payload)
        except Exception:
            raise ValueError(
                "reuse_results requires serializable Outputs without NaN values."
            ) from None
        supplied[coordinate.uuid] = payload
    return supplied


def execute_sweep_groups(
    task_collection: Path,
    groups: Sequence[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]],
    seeds: Sequence[int],
    input_artifacts_root: str | Path,
    *,
    task: Task,
    time_limit: float | None,
    memory_limit_gb: float | None,
    mode: Literal["missing", "retry_resources", "all"],
    answered_roots: Sequence[Path],
    reuse_results: pd.DataFrame | None = None,
    schedule_cells: CellScheduler = schedule_cells_serially,
    verbose: bool = False,
) -> pd.DataFrame:
    """Execute resolved factors with immutable Attempts and scheduler-owned limits.

    ``answered_roots`` names the tree(s) whose recorded Results already
    answer a cell for ``mode="missing"``/``"retry_resources"``: the local
    Result answers when ``task_collection`` is one of them, and otherwise the
    first other root that records one. New Attempts are always written to
    ``task_collection``, whether or not it is one of ``answered_roots``, and
    Attempt numbering continues from its own history whichever roots answer.

    ``task`` is the collection's live typed contract. It is supplied even when
    no coordinate is selected, because abandoned journals still need their
    output schema recovered.
    """
    coordinates = expand_sweep_groups(groups, seeds)
    supplied_outputs = _supplied_outputs(reuse_results, task, coordinates)
    reconcile_orphaned_work(task_collection, task)
    if not coordinates:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    memory_limit_bytes = (
        None if memory_limit_gb is None else int(memory_limit_gb * 1024**3)
    )
    recompute = mode == "all"
    current = current_outcomes(task_collection, task, coordinates)
    # Selection row ``i`` is ``coordinates[i]``: both walk the groups in order,
    # then each group's budgets, then the distinct seeds.
    other_status: dict[int, ResultStatus] = {}
    if not recompute:
        other_status = recorded_statuses(
            [root for root in answered_roots if root != task_collection],
            task,
            list(dict.fromkeys(algorithm.uuid for _, algorithm, _ in groups)),
            list(dict.fromkeys(instance.uuid for instance, _, _ in groups)),
            groups=groups,
            seeds=seeds,
        )
    pending: list[CellContext] = []
    for row, coordinate in enumerate(coordinates):
        outcome = current.get(coordinate.uuid)
        status: ResultStatus | None = None
        if task_collection in answered_roots and outcome is not None:
            status = outcome[1]
        elif row in other_status:
            status = other_status[row]
        if (
            status is not None
            and not recompute
            and coordinate.uuid not in supplied_outputs
            and answers_coordinate(
                status, retry_resource_failures=mode == "retry_resources"
            )
        ):
            continue
        pending.append(
            CellContext(
                coordinate=coordinate,
                task_collection=task_collection,
                input=Path(input_artifacts_root),
                attempt_number=(0 if outcome is None else outcome[0]),
                time_limit_s=time_limit,
                memory_limit_bytes=memory_limit_bytes,
            )
        )

    writer = AttemptWriter(task_collection, task)
    try:
        _schedule_attempts(
            pending,
            writer.add,
            schedule_cells,
            verbose=verbose,
            reuse_durable=mode != "all" and task_collection in answered_roots,
            retry_resource_failures=mode == "retry_resources",
            supplied_outputs=supplied_outputs,
        )
    finally:
        writer.close()

    return results_frame(task_collection, groups=groups, seeds=seeds)
