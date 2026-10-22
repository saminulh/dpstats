"""Durable file I/O for records: staged publication.

Every durable write follows one discipline: bytes are staged under
``STAGING_PREFIX`` and reach the storage device before the name that
publishes them does, so a reader never sees a torn document. That ordering
buys survival of the writing process's death and nothing wider — no
containing directory is ever synced, so a publish that has already returned
can still be absent after a power failure — and it is the whole durability
contract every store in this library rests on.
"""

from __future__ import annotations

import json
import os
import tempfile

from contextlib import contextmanager
from typing import Any, BinaryIO, Generator

from pathlib import Path


from dpstats.transport.canonical import (
    encode_transport,
)

STAGING_PREFIX = ".incomplete-"


@contextmanager
def stage_durable_file(
    directory: Path, *, suffix: str
) -> Generator[tuple[Path, BinaryIO], None, None]:
    """Yield one writable staging file, durably flushed on successful exit.

    The name wears ``STAGING_PREFIX``, so every reader of a store knows the
    file belongs to a writer rather than to the store, and the given suffix
    lets a recovery pass classify debris as what it would have become.
    Bytes become durable only after the context exits successfully. The caller
    owns the staging file after that exit and removes it after publication.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=STAGING_PREFIX, suffix=suffix
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            yield temporary, destination
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stage_durable_bytes(directory: Path, payload: bytes, *, suffix: str) -> Path:
    """Write one staging file whose bytes are on the storage device."""
    with stage_durable_file(directory, suffix=suffix) as (temporary, destination):
        destination.write(payload)
    return temporary


def publish_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably publish bytes under a name, replacing whatever stood there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = stage_durable_bytes(path.parent, payload, suffix="".join(path.suffixes))
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Durably publish bytes under a name that must not already exist.

    The link refuses an existing name, so writers racing one name produce one
    document and one refusal rather than a silent replacement.

    Raises:
        FileExistsError: when the name already exists. The staging file is
            removed before this propagates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = stage_durable_bytes(path.parent, payload, suffix="".join(path.suffixes))
    try:
        os.link(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write ``obj`` to ``path`` as indented standard JSON atomically."""
    encoded = encode_transport(obj)
    payload = json.dumps(encoded, indent=2, allow_nan=False).encode("utf-8")
    publish_bytes_atomic(path, payload)
