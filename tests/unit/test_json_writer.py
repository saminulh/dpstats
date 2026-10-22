"""Durability contracts for the shared authored-JSON writer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from dpstats.transport.documents import STAGING_PREFIX, write_json_atomic


def test_a_failed_write_keeps_the_previous_document(tmp_path: Path) -> None:
    path = tmp_path / "document.json"
    write_json_atomic(path, {"kept": 1})

    with pytest.raises(TypeError):
        write_json_atomic(path, {"unserializable": {1, 2}})

    assert json.loads(path.read_bytes()) == {"kept": 1}
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))


def test_nonfinite_floats_use_the_strict_json_transport(tmp_path: Path) -> None:
    path = tmp_path / "document.json"

    write_json_atomic(path, {"positive": math.inf, "negative": -math.inf})

    assert json.loads(path.read_bytes()) == {
        "positive": "+inf_json",
        "negative": "-inf_json",
    }


def test_content_is_synced_before_the_rename_publishes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(descriptor: int) -> None:
        order.append("fsync")
        real_fsync(descriptor)

    def recording_replace(source: Path, destination: Path) -> None:
        order.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    write_json_atomic(tmp_path / "document.json", {"a": 1})

    assert order == ["fsync", "replace"]
