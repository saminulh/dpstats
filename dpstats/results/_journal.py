"""The write-ahead journal standing behind one run's Attempt records.

Open, append and fsync, seal through the partition writer, list what is
unsealed, and recover what a dead writer left.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpstats.results._outcomes import AttemptRecord, ResultAttempt
from dpstats.results._partitions import (
    attempt_store,
    compact_partition,
    fragmented_partitions,
    mint_id,
    publish_records,
    recover_staged_partition_files,
    require_directory,
)
from dpstats.task import Task
from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_canonical_jsonl,
)
from dpstats.transport.documents import STAGING_PREFIX
from dpstats.workspace.layout import attempts_directory

logger = logging.getLogger("benchmark")

_JOURNAL_SUFFIX = ".jsonl"
_JOURNAL_SIZE_BYTES = 32 * 1024**2
_JOURNAL_MAX_AGE_SECONDS = 5 * 60.0


@dataclass(frozen=True)
class _Journal:
    """One open append-only journal and the moment its batch opened."""

    path: Path
    handle: io.BufferedWriter
    opened_at: float
    records: list[dict[str, Any]]


class AttemptWriter:
    """Journal completed ResultAttempts durably and seal them into partitions.

    ``add`` appends one canonical record line to ``attempts/.incomplete-<id>.jsonl``
    and reaches the device (flush and fsync) before returning, so losing the
    process loses no attempt the caller observed. A seal groups the journaled
    records by algorithm, publishes one file per partition under the staged
    publication ``transport.documents`` states, and unlinks the journal only
    once every partition file stands; it happens at ``close``, when the
    journal reaches its size cap, and when an added attempt finds the journal
    older than its age limit. Opening compacts every partition another writer
    left fragmented, and ``close`` compacts every partition this run wrote. A
    journal a dead writer left is ``recover_orphaned_attempt_journals``'s to
    seal.
    """

    def __init__(self, task_root: Path, task: Task):
        self._task_root = task_root
        self._task = task
        self._run_id = mint_id()
        self._seals = 0
        self._journal: _Journal | None = None
        self._written: set[str] = set()
        for alg_uuid in fragmented_partitions(attempt_store(task_root)):
            compact_partition(task_root, task, alg_uuid)

    def add(self, attempt: ResultAttempt) -> None:
        """Durably journal one completed execution observation."""
        journal = self._journal
        if journal is None:
            journal = self._open_journal()
        record = attempt.to_record()
        journal.handle.write(canonical_json_bytes(record) + b"\n")
        journal.handle.flush()
        os.fsync(journal.handle.fileno())
        journal.records.append(record)
        if (
            journal.handle.tell() >= _JOURNAL_SIZE_BYTES
            or time.monotonic() - journal.opened_at >= _JOURNAL_MAX_AGE_SECONDS
        ):
            self._seal()

    def close(self) -> None:
        """Seal what is journaled, then compact every partition this run wrote."""
        self._seal()
        for alg_uuid in sorted(self._written):
            compact_partition(self._task_root, self._task, alg_uuid)
        self._written.clear()

    def _open_journal(self) -> _Journal:
        descriptor, name = tempfile.mkstemp(
            dir=attempt_store(self._task_root),
            prefix=STAGING_PREFIX,
            suffix=_JOURNAL_SUFFIX,
        )
        journal = _Journal(
            Path(name), os.fdopen(descriptor, "wb"), time.monotonic(), []
        )
        self._journal = journal
        return journal

    def _seal(self) -> None:
        """Publish the journaled batch to its partitions, then drop the journal.

        A batch this cannot publish stays journaled for recovery; the sequence
        still advances, so a retried seal never collides with a file it
        published.
        """
        journal = self._journal
        if journal is None:
            return
        self._journal = None
        journal.handle.close()
        sequence = self._seals
        self._seals += 1
        self._written.update(
            publish_records(
                self._task_root,
                self._task,
                journal.records,
                f"{self._run_id}.{sequence}",
            )
        )
        journal.path.unlink()


def _complete_lines(path: Path) -> bytes:
    """Return a journal's complete record lines, dropping a torn final one."""
    data = path.read_bytes()
    end = data.rfind(b"\n")
    if len(data) > end + 1:
        logger.warning(
            "Dropping %d bytes of an unterminated final observation in "
            "ResultAttempt journal %s.",
            len(data) - end - 1,
            path,
        )
    return data[: end + 1]


def recover_orphaned_attempt_journals(task_root: Path, task: Task) -> None:
    """Seal every journal a previous writer left, and drop partition debris.

    A journal holds attempts its writer durably recorded but never committed; an
    unterminated final line is the fragment its writer died appending and is
    dropped with a report. Each journal seals under ``recovered-<id>``: one
    that seals is unlinked, one holding no complete record is removed with
    that reason, and one that cannot be committed is kept and named. The
    partition store then drops the staging files a killed cell writer left:
    a staged file confirms nothing was durably recorded, so its cell is
    simply unanswered and runs again.
    """
    directory = attempts_directory(task_root)
    if not directory.exists():
        return
    require_directory(directory)
    failures: dict[Path, str] = {}
    for path in sorted(directory.glob(f"{STAGING_PREFIX}*{_JOURNAL_SUFFIX}")):
        if path.is_symlink() or not path.is_file():
            failures[path] = "is not a regular journal file."
            continue
        try:
            payload = _complete_lines(path)
            if not payload:
                path.unlink()
                logger.warning(
                    "Removed ResultAttempt journal %s: no complete observation.",
                    path,
                )
                continue
            records = decode_canonical_jsonl(payload, f"ResultAttempt journal {path}")
            for record in records:
                AttemptRecord.model_validate(record)
            journal_id = path.name[len(STAGING_PREFIX) : -len(_JOURNAL_SUFFIX)]
            written = publish_records(
                task_root, task, records, f"recovered-{journal_id}"
            )
            path.unlink()
        except (OSError, ValueError) as error:
            failures[path] = str(error)
            continue
        logger.warning(
            "Recovered orphaned ResultAttempt journal %s into %d partition(s).",
            path,
            len(written),
        )
    recover_staged_partition_files(directory)
    if failures:
        details = "\n".join(
            f"{path.name}: {failures[path]}" for path in sorted(failures)
        )
        raise ValueError(
            f"One or more ResultAttempt journals were not recovered:\n{details}"
        )
