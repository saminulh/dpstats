"""Durable logical named arrays: canonical digest, reader and writer, publication.

An artifact stores named arrays as a deflated npz whose name is the logical
digest of its contents: dtype, shape, and little-endian bytes under one
domain-separated hash, so archive layout, compression, and native byte order
never enter the identity. The reader accepts stored and deflated archives
alike. The writer pins deflate level 6 and the zip's fixed member timestamp,
so its file bytes are identical for the same arrays on the same zlib build;
the logical digest is identical everywhere. Which arrays an artifact holds, and where one lives
in an Input-artifacts root, belong to the callers that own those concerns.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.documents import stage_durable_file

Array = np.ndarray[tuple[int, ...], np.dtype[np.generic]]

# Fixed hashing domain: renaming it would change every Input digest.
_ARRAY_DIGEST_DOMAIN = b"dp-testing-platform.logical-named-arrays.v1"
_DEFLATE_LEVEL = 6


def named_arrays_digest(arrays: Mapping[str, Array]) -> str:
    """Hash named arrays independently of layout and native byte order."""
    digest = hashlib.sha256()

    def update_frame(value: bytes | memoryview) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    update_frame(_ARRAY_DIGEST_DOMAIN)
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"Stored input array {name!r} cannot use object dtype.")
        if array.dtype.fields is not None:
            raise ValueError(
                f"Stored input array {name!r} cannot use structured dtype."
            )
        dtype = array.dtype.newbyteorder("<")
        normalized = np.asarray(array, dtype=dtype, order="C")
        descriptor = canonical_json_bytes(
            {"name": name, "dtype": dtype.str, "shape": list(array.shape)}
        )
        update_frame(descriptor)
        update_frame(normalized.data.cast("B"))
    return digest.hexdigest()


def read_named_arrays(path: Path) -> dict[str, Array]:
    """Read one artifact's named arrays exactly as stored."""
    with path.open("rb") as source:
        with np.load(source, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}


def verify_named_arrays(path: Path, expected_digest: str) -> dict[str, Array]:
    """Read stored arrays, refusing contents the expected digest does not name."""
    arrays = read_named_arrays(path)
    found = named_arrays_digest(arrays)
    if found != expected_digest:
        raise ValueError(
            f"Input artifact {path} has logical digest {found}, "
            f"not {expected_digest}."
        )
    return arrays


def write_named_arrays(
    arrays: Mapping[str, Array], target: Path, *, digest: str
) -> None:
    """Idempotently publish one artifact under its already-derived digest name.

    The archive is streamed to a staging file whose bytes are on the storage
    device before the name appears, and the link refuses an existing name, so
    a collision is read back and verified against the digest rather than
    replaced.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_named_arrays(target, digest)
        return

    with stage_durable_file(target.parent, suffix=".npz") as (temporary, destination):
        with zipfile.ZipFile(
            destination,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=_DEFLATE_LEVEL,
        ) as archive:
            for name, array in arrays.items():
                with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                    np.save(member, array, allow_pickle=False)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            verify_named_arrays(target, digest)
    finally:
        temporary.unlink(missing_ok=True)
