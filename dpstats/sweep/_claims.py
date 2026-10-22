"""Durable claims on in-flight ResultAttempts and their reconciliation.

A claim is the only thing a runner leaves behind when it is killed rather than
stopped, whatever removed its process — a supervisor, a kernel out-of-memory
kill, or a signal it cannot intercept. An interruption is not a terminal
outcome, so settling such a claim records nothing and leaves its cell
unanswered for the maintainer to rerun. Deciding a claim is a writer's act, so
this module offers it two ways: reconciliation, which settles the previous
invocation, and a read of the claims that stand, which reports work in limbo
without touching it.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
    parse_json_object,
)
from dpstats.transport.documents import (
    STAGING_PREFIX,
    publish_bytes_exclusive,
)
from dpstats.workspace.layout import claims_directory
from dpstats.results._journal import recover_orphaned_attempt_journals
from dpstats.coordinates import CellCoordinate, coordinate_record_identity
from dpstats.task import Task

logger = logging.getLogger("benchmark")

_CLAIM_SUFFIX = ".json"


def _claim_path(
    task_root: Path, coordinate: CellCoordinate, attempt_number: int
) -> Path:
    """Name one claim by the numbered observation identity it covers."""
    name = f"{coordinate.uuid}-{attempt_number}{_CLAIM_SUFFIX}"
    return claims_directory(task_root) / name


def claim_attempt(
    task_root: Path, coordinate: CellCoordinate, attempt_number: int
) -> None:
    """Declare that this process is starting one numbered ResultAttempt.

    The claim is written before the attempt begins and dropped only once its
    outcome is durable history, so the interval a claim covers is exactly the
    interval in which an external kill can make work vanish. It is one small
    write-then-link per cell, published under the durability contract
    ``transport.documents`` states; losing the link itself leaves the cell
    exactly as unanswered as one that never started. The link refuses an
    existing name, so two writers racing one numbered attempt produce one
    claim and one loud refusal rather than two contradictory histories.

    Raises:
        ValueError: when a claim already stands on this numbered attempt.
    """
    payload = canonical_json_bytes(
        {
            "coordinate_uuid": coordinate.uuid,
            "coordinate": coordinate.to_record(),
            "attempt_number": attempt_number,
            "pid": os.getpid(),
            "started_at": time.time(),
        }
    )
    try:
        publish_bytes_exclusive(
            _claim_path(task_root, coordinate, attempt_number), payload
        )
    except FileExistsError:
        raise ValueError(
            f"A claim already stands on attempt {attempt_number} of coordinate "
            f"{coordinate.uuid}: another invocation took this cell and has not "
            "accounted for it. If its process is alive, the attempt is already "
            "executing; if it is gone, reconcile-sweep records what became of it."
        ) from None


def release_attempt(
    task_root: Path, coordinate: CellCoordinate, attempt_number: int
) -> None:
    """Drop the claim on one attempt whose outcome is durably recorded."""
    _claim_path(task_root, coordinate, attempt_number).unlink(missing_ok=True)


def _read_claim(path: Path) -> dict[str, Any]:
    """Read the facts one absent runner recorded for itself.

    Preview acts on a claim's stored coordinate uuid before any reconciliation
    builds a record from it, so this is the boundary that establishes the
    token names the coordinate beside it. The attempt number and the
    coordinate's own shape remain ``AttemptRecord``'s, on the record built
    from this claim.
    """
    claim = decode_json_document(path)
    if set(claim) != {
        "coordinate_uuid",
        "coordinate",
        "attempt_number",
        "pid",
        "started_at",
    }:
        raise ValueError(f"Attempt claim {path.name} has an unexpected shape.")
    coordinate_uuid = claim["coordinate_uuid"]
    carried = coordinate_record_identity(parse_json_object(claim["coordinate"]))
    if coordinate_uuid != carried:
        raise ValueError(
            f"Attempt claim {path.name} names coordinate {coordinate_uuid!r} but "
            f"carries {carried!r}."
        )
    pid = claim["pid"]
    if type(pid) is not int or pid <= 0:
        raise ValueError(f"Attempt claim {path.name} pid must be a positive integer.")
    started_at = claim["started_at"]
    if (
        isinstance(started_at, bool)
        or not isinstance(started_at, (int, float))
        or not 0 <= started_at < math.inf
    ):
        raise ValueError(
            f"Attempt claim {path.name} start time must be a finite epoch second."
        )
    return claim


def standing_claims(task_root: str | Path) -> list[dict[str, Any]]:
    """Read the claims that stand in one task's Results directory, changing nothing.

    A standing claim is a cell some invocation took and has not accounted for:
    either a runner is executing it right now, or its runner is gone and the
    next reconciliation will clear the claim, leaving the cell unanswered.
    Telling those apart is a writer's decision, so a reader reports the claim
    and decides nothing.

    A staging name is the temporary file a claim write renames from, so it
    covers an attempt that had not started and stands for nothing. Any other
    entry that does not read as a claim is reported here and refused by
    reconciliation, which is the operation that may act on it.
    """
    directory = claims_directory(Path(task_root))
    if not directory.is_dir() or directory.is_symlink():
        return []
    claims: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(STAGING_PREFIX):
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Attempt claim {path.name} is not a regular file.")
            claims.append(_read_claim(path))
        except (OSError, ValueError) as error:
            logger.warning("Ignoring an unreadable attempt claim: %s", error)
    return claims


def refuse_spawned_reentry() -> None:
    """Abort a Sweep operation re-entered while a spawned worker imports its script.

    CPython sets ``_inheriting`` while a spawned child imports its parent's
    ``__main__``, which is the one way the library can observe that another
    runner owns the store and is still alive. Such a child would read its live
    parent's claims as abandoned and would take cells its parent is already
    executing; aborting before it touches the store also caps unguarded-script
    recursion at one child.

    Raises:
        RuntimeError: naming the script guard that prevents the re-entry.
    """
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise RuntimeError(
            "A Sweep operation was re-entered inside a spawned worker while it "
            "was importing your script. Guard the script entry point with "
            '`if __name__ == "__main__": main()` so workers do not run or '
            "reconcile a Sweep while importing it."
        )


def reconcile_orphaned_work(task_root: str | Path, task: Task) -> list[str]:
    """Settle everything a previous invocation left unfinished in one store.

    This is a writer's operation and assumes the one-writer-per-task-collection
    contract: every claim it finds belongs to an invocation that is gone,
    because a live one would be the writer. What that invocation durably
    journaled becomes a committed Attempt record first, and only then are its claims cleared, so
    an outcome that was merely still journaled is never lost.

    Returns:
        The coordinates whose claims were cleared, in claim order. A claim
        covers work its runner never reported, so each of these cells is
        unanswered unless this call recovered that runner's journal for it.

    Raises:
        RuntimeError: when this process is a spawned worker importing its
            parent's script.
        ValueError: naming every journal or claim that could not be settled.
            Everything that could be settled is already durably recorded when
            this raises.
    """
    refuse_spawned_reentry()
    root = Path(task_root)
    recover_orphaned_attempt_journals(root, task)
    return _reconcile_orphaned_claims(root)


def _reconcile_orphaned_claims(task_root: Path) -> list[str]:
    """Clear every claim a previous invocation left behind.

    One active runner owns a task collection at a time, so a claim present when
    the root is opened belongs to an invocation that is already gone. An
    interruption is not a terminal outcome, so the claim is dropped with its
    facts reported and its cell is simply unanswered, whether or not the vanished
    runner had already journaled an outcome for it: a journaled outcome became
    a committed Attempt record a moment earlier and answers the cell on its own.

    A partially written claim covers an attempt that had not started, because
    the claim becomes visible before the attempt does; it is removed with that
    reason.
    """
    directory = claims_directory(task_root)
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Attempt claim store must be a non-symlink directory.")

    failures: dict[Path, str] = {}
    cleared: list[str] = []
    for path in sorted(directory.iterdir()):
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Attempt claim {path.name} is not a regular file.")
            if path.name.startswith(STAGING_PREFIX):
                path.unlink()
                logger.warning(
                    "Removed partially written attempt claim %s: its attempt never "
                    "started.",
                    path.name,
                )
                continue
            claim = _read_claim(path)
        except (OSError, ValueError) as error:
            failures[path] = str(error)
            continue
        held = datetime.fromtimestamp(claim["started_at"], tz=timezone.utc).isoformat()
        logger.warning(
            "Cleared orphaned attempt claim %s: runner pid %s held it from %s and "
            "was gone; the cell is unanswered.",
            path.name,
            claim["pid"],
            held,
        )
        path.unlink(missing_ok=True)
        cleared.append(claim["coordinate_uuid"])

    if failures:
        details = "\n".join(
            f"{path.name}: {failures[path]}" for path in sorted(failures)
        )
        raise ValueError(f"One or more attempt claims were not reconciled:\n{details}")
    return cleared
