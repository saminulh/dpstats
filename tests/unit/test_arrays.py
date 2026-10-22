"""Durable publication of logical named-array artifacts."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

import dpstats.transport.arrays as arrays_transport
from dpstats.transport.documents import STAGING_PREFIX


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_named_array_writer_removes_staging_file_when_serialization_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    """An interrupted serializer cannot leave a partial input artifact behind."""
    values = {"X": np.array([1.0, 2.0])}
    target = tmp_path / "input.npz"

    def interrupt_serialization(*args: object, **kwargs: object) -> None:
        raise interruption()

    monkeypatch.setattr(arrays_transport.np, "save", interrupt_serialization)

    with pytest.raises(interruption):
        arrays_transport.write_named_arrays(
            values,
            target,
            digest=arrays_transport.named_arrays_digest(values),
        )

    assert not target.exists()
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))


def _write_stored(arrays: dict[str, np.ndarray], target: Path) -> None:
    """Write the stored (uncompressed) archive earlier library versions wrote."""
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.save(member, array, allow_pickle=False)


def _publish(arrays: dict[str, np.ndarray], root: Path) -> tuple[Path, str]:
    digest = arrays_transport.named_arrays_digest(arrays)
    target = root / f"{digest}.npz"
    arrays_transport.write_named_arrays(arrays, target, digest=digest)
    return target, digest


def test_named_array_writer_round_trips_deflated_archive(tmp_path: Path) -> None:
    """Written arrays read back exactly, from members the writer deflated."""
    values = {
        "X": np.arange(12, dtype=np.float64).reshape(3, 4),
        "y": np.array([1, -2, 3], dtype=np.int32),
    }
    target, digest = _publish(values, tmp_path)

    with zipfile.ZipFile(target) as archive:
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_DEFLATED
        }
    read = arrays_transport.verify_named_arrays(target, digest)
    assert read.keys() == values.keys()
    for name, array in values.items():
        assert read[name].dtype == array.dtype
        np.testing.assert_array_equal(read[name], array)


def test_deflated_and_stored_archives_verify_against_one_logical_digest(
    tmp_path: Path,
) -> None:
    """Compression changes file bytes only, never the digest an artifact verifies."""
    values = {"X": np.linspace(0.0, 1.0, 50).reshape(10, 5)}
    deflated, digest = _publish(values, tmp_path / "deflated")
    stored = tmp_path / "stored" / deflated.name
    stored.parent.mkdir()
    _write_stored(values, stored)

    assert deflated.read_bytes() != stored.read_bytes()
    for path in (deflated, stored):
        np.testing.assert_array_equal(
            arrays_transport.verify_named_arrays(path, digest)["X"], values["X"]
        )


def test_root_mixing_stored_and_deflated_artifacts_reads(tmp_path: Path) -> None:
    """A root holding earlier stored artifacts beside new deflated ones reads whole."""
    old = {"X": np.eye(4)}
    old_digest = arrays_transport.named_arrays_digest(old)
    _write_stored(old, tmp_path / f"{old_digest}.npz")
    new_path, new_digest = _publish({"X": np.ones((2, 3))}, tmp_path)

    arrays_transport.write_named_arrays(
        old, tmp_path / f"{old_digest}.npz", digest=old_digest
    )
    assert new_path.exists()
    for digest in (old_digest, new_digest):
        arrays_transport.verify_named_arrays(tmp_path / f"{digest}.npz", digest)


def test_mostly_zero_design_shrinks_substantially(tmp_path: Path) -> None:
    """A one-hot-style design, almost all zeros, deflates far below its raw size."""
    design = np.zeros((2_000, 500), dtype=np.float64)
    design[np.arange(2_000), np.arange(2_000) % 500] = 1.0
    target, _ = _publish({"X": design}, tmp_path)

    assert target.stat().st_size < design.nbytes / 20


def test_writer_bytes_are_reproducible_for_the_same_arrays(tmp_path: Path) -> None:
    """Pinned level and fixed member timestamps make rewrites byte-identical."""
    values = {"X": np.arange(1_000, dtype=np.float64), "y": np.ones(7)}
    first, _ = _publish(values, tmp_path / "first")
    second, _ = _publish(values, tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert {info.date_time for info in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }
