"""Attempt claims and the reconciliation that clears what a kill left."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from dpstats.instance import Instance
from dpstats.budgets import PureDPBudget
from dpstats.sweep._claims import (
    claim_attempt,
    reconcile_orphaned_work,
    release_attempt,
    standing_claims,
)
from dpstats.coordinates import CellCoordinate
from dpstats.workspace.layout import (
    claims_directory,
)
from tests.attempts import read_attempt_rows, write_attempt_partition
from tests.registry_components import SIMPLE_TASK, SimpleAlgorithm


def _coordinate(*, alg_seed: int = 0) -> CellCoordinate:
    instance = Instance(
        metadata=SIMPLE_TASK.Metadata(n_samples=1, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
    )
    return CellCoordinate(
        instance=instance,
        algorithm=SimpleAlgorithm(),
        budget=PureDPBudget(eps=1.0),
        alg_seed=alg_seed,
    )


def test_an_orphaned_claim_is_cleared_and_leaves_its_cell_unanswered(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An interruption is not an outcome, so nothing is recorded for it.

    The cell a vanished runner held is exactly as unanswered as one never
    attempted, so the next run takes it again under the same attempt number.
    """
    coordinate = _coordinate()
    claim_attempt(tmp_path, coordinate, 3)

    with caplog.at_level(logging.WARNING, logger="benchmark"):
        cleared = reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert cleared == [coordinate.uuid]
    assert read_attempt_rows(tmp_path) == []
    assert not list(claims_directory(tmp_path).iterdir())
    assert str(os.getpid()) in caplog.text
    assert "unanswered" in caplog.text


def test_a_second_claim_on_one_numbered_attempt_is_refused(tmp_path: Path) -> None:
    """Two writers racing one attempt produce one claim and one loud refusal.

    Silently replacing the standing claim would let both racers execute the
    cell and journal two contradictory histories for one attempt number.
    """
    coordinate = _coordinate()
    claim_attempt(tmp_path, coordinate, 0)

    with pytest.raises(ValueError, match="already stands"):
        claim_attempt(tmp_path, coordinate, 0)

    (standing,) = standing_claims(tmp_path)
    assert standing["pid"] == os.getpid()
    (kept,) = claims_directory(tmp_path).iterdir()
    assert kept.name == f"{coordinate.uuid}-0.json"


def test_a_released_claim_leaves_nothing_to_reconcile(tmp_path: Path) -> None:
    coordinate = _coordinate()
    claim_attempt(tmp_path, coordinate, 0)
    release_attempt(tmp_path, coordinate, 0)

    reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert read_attempt_rows(tmp_path) == []
    assert not list(claims_directory(tmp_path).iterdir())


def test_a_claim_whose_outcome_is_history_leaves_that_outcome_standing(
    tmp_path: Path,
) -> None:
    """A runner that died between journaling and releasing loses nothing.

    Its outcome is already history, so clearing the stale claim must not
    disturb the record that already answers that numbered attempt.
    """
    coordinate = _coordinate()
    recorded = {
        "coordinate_uuid": coordinate.uuid,
        "coordinate": coordinate.to_record(),
        "attempt_number": 0,
        "status": "timeout",
        "payload": None,
        "runtime_s": 1.0,
        "peak_rss_mb": None,
    }
    write_attempt_partition(tmp_path, [recorded])
    claim_attempt(tmp_path, coordinate, 0)

    reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert [(row["attempt"], row["status"]) for row in read_attempt_rows(tmp_path)] == [
        (0, "timeout")
    ]
    assert not list(claims_directory(tmp_path).iterdir())


def test_a_root_that_never_claimed_reconciles_to_nothing(tmp_path: Path) -> None:
    reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert not claims_directory(tmp_path).exists()
    assert read_attempt_rows(tmp_path) == []


def test_one_unreadable_claim_is_reported_while_its_siblings_reconcile(
    tmp_path: Path,
) -> None:
    kept = _coordinate(alg_seed=0)
    broken = _coordinate(alg_seed=1)
    claim_attempt(tmp_path, broken, 0)
    claims = claims_directory(tmp_path)
    (broken_path,) = claims.iterdir()
    broken_path.write_text("{ not json")
    claim_attempt(tmp_path, kept, 0)

    with pytest.raises(ValueError, match=broken_path.name):
        reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert sorted(path.name for path in claims.iterdir()) == [broken_path.name]


def test_a_claim_whose_uuid_does_not_name_its_coordinate_is_refused_at_read(
    tmp_path: Path,
) -> None:
    """A claim is acted on before any record is built from it.

    Preview decides from the stored token alone, so a claim carrying a token
    that is not its coordinate's identity has to be refused where it is
    decoded rather than where a reconciliation would notice.
    """
    coordinate = _coordinate()
    claim_attempt(tmp_path, coordinate, 0)
    (path,) = claims_directory(tmp_path).iterdir()
    claim = json.loads(path.read_text())
    foreign = _coordinate(alg_seed=9).uuid
    claim["coordinate_uuid"] = foreign
    path.write_text(json.dumps(claim))

    with pytest.raises(ValueError) as refusal:
        reconcile_orphaned_work(tmp_path, SIMPLE_TASK)

    assert foreign in str(refusal.value)
    assert coordinate.uuid in str(refusal.value)
    assert read_attempt_rows(tmp_path) == []


def test_a_reader_sees_the_same_claims_the_writer_would_decide(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Debris is not a standing claim, and an entry that hides one is reported.

    A claim write renames from a staging name, so a staging file covers an
    attempt that had not started. Reading one as a standing claim would report
    a cell as spoken for that reconciliation is about to delete.
    """
    coordinate = _coordinate()
    claim_attempt(tmp_path, coordinate, 0)
    claims = claims_directory(tmp_path)
    (real,) = claims.iterdir()
    (claims / ".incomplete-abc.json").write_bytes(real.read_bytes())
    (claims / ("f" * 64 + ".json")).write_text("{ not json")

    with caplog.at_level(logging.WARNING, logger="benchmark"):
        standing = standing_claims(tmp_path)

    assert [claim["coordinate"] for claim in standing] == [coordinate.to_record()]
    assert "unreadable attempt claim" in caplog.text
