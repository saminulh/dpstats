"""Durable publication contracts: a create-only publish never replaces a name."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpstats.transport.documents import (
    STAGING_PREFIX,
    publish_bytes_exclusive,
)


def test_exclusive_publication_never_replaces_an_existing_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    publish_bytes_exclusive(path, b"first\n")

    with pytest.raises(FileExistsError):
        publish_bytes_exclusive(path, b"second\n")

    assert path.read_bytes() == b"first\n"
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))
