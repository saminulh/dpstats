"""Public behavior of the outcome table over Results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpstats.analysis import summarize_outcomes
from dpstats.budgets import PureDPBudget
from dpstats.instance import Instance, persist_instance
from dpstats.coordinates import CellCoordinate
from dpstats.results import ResultAttempt, ResultStatus
from tests.registry_components import (
    SimpleAlgorithm,
    simple_producer,
    SimpleOutput,
)


@pytest.fixture
def results(tmp_path: Path) -> list[ResultAttempt]:
    """Three observed outcomes of one algorithm on one Instance."""
    metadata, input_data, references = simple_producer(n=2, d=1)(0)
    instance = persist_instance(metadata, input_data, references, tmp_path, label="toy")
    algorithm = SimpleAlgorithm(label="A")

    def coordinate(alg_seed: int) -> CellCoordinate:
        return CellCoordinate(
            instance=instance,
            algorithm=algorithm,
            budget=PureDPBudget(eps=1.0),
            alg_seed=alg_seed,
        )

    return [
        ResultAttempt(
            coordinate=coordinate(0),
            attempt_number=0,
            status=ResultStatus.SUCCESS,
            payload=SimpleOutput(result=np.zeros(1)),
            runtime_s=0.5,
            peak_rss_mb=2.0,
        ),
        ResultAttempt(
            coordinate=coordinate(1),
            attempt_number=1,
            status=ResultStatus.ALGORITHM_ERROR,
            payload="ValueError",
        ),
        ResultAttempt(
            coordinate=coordinate(2),
            attempt_number=0,
            status=ResultStatus.TIMEOUT,
        ),
    ]


def _results_frame(results: list[ResultAttempt]) -> pd.DataFrame:
    """Build the Results-frame fields the outcome table reads."""
    return pd.DataFrame(
        [
            {
                "task": result.coordinate.task.id,
                "instance": result.coordinate.instance,
                "algorithm": result.coordinate.algorithm,
                "algorithm_label": str(result.coordinate.algorithm),
                "budget": result.coordinate.budget,
                "alg_seed": result.coordinate.alg_seed,
                "status": result.status,
                "runtime_s": result.runtime_s,
                "peak_rss_mb": result.peak_rss_mb,
            }
            for result in results
        ]
    )


def test_outcomes_count_each_recorded_result_under_its_coordinate_group(
    results: list[ResultAttempt],
) -> None:
    frame = _results_frame(results)

    outcomes = summarize_outcomes(frame, by=["algorithm_label"])

    counted = dict(zip(outcomes["status"], outcomes["count"]))
    assert counted == {
        "success": 1,
        "no_release": 0,
        "invalid_regime": 0,
        "algorithm_error": 1,
        "timeout": 1,
        "memory_limit_exceeded": 0,
    }
    assert outcomes["algorithm_label"].unique().tolist() == ["A"]
    assert outcomes["denominator"].unique().tolist() == [3]
    # The caller's frame is read, never written to.
    assert list(frame.columns) == list(_results_frame(results).columns)


def test_outcomes_report_every_recorded_outcome_over_its_denominator() -> None:
    statuses = pd.DataFrame(
        {
            "algorithm": ["A", "A", "A", "B"],
            "status": ["success", "timeout", "success", "algorithm_error"],
        }
    )

    shares = summarize_outcomes(statuses, by=["algorithm"])

    assert list(shares.columns) == [
        "algorithm",
        "status",
        "count",
        "denominator",
        "share",
    ]
    assert set(shares["status"]) == {status.value for status in ResultStatus}
    first = shares[shares["algorithm"] == "A"]
    assert first["denominator"].unique().tolist() == [3]
    counted = dict(zip(first["status"], first["count"]))
    assert counted["success"] == 2
    assert counted["timeout"] == 1
    assert counted["no_release"] == 0
    shared = dict(zip(first["status"], first["share"]))
    assert shared["success"] == pytest.approx(2 / 3)
    assert shared["no_release"] == pytest.approx(0.0)
    second = shares[shares["algorithm"] == "B"]
    assert second["denominator"].unique().tolist() == [1]
    assert dict(zip(second["status"], second["share"]))["algorithm_error"] == 1.0


def test_outcomes_nest_by_each_column_in_first_seen_order() -> None:
    statuses = pd.DataFrame(
        {
            # Within eps=1, A appears before B; across the frame, B comes first.
            "eps": [10.0, 1.0, 10.0, 1.0, 10.0],
            "algorithm": ["B", "A", "A", "B", "B"],
            "status": [
                "invalid_regime",
                "success",
                "no_release",
                "success",
                "success",
            ],
        }
    )

    shares = summarize_outcomes(statuses, by=["eps", "algorithm"])

    vocabulary = [status.value for status in ResultStatus]
    groups: list[tuple[float, str]] = list(
        zip(shares["eps"].tolist(), shares["algorithm"].tolist())
    )
    expected = [(10.0, "B"), (10.0, "A"), (1.0, "B"), (1.0, "A")]
    assert groups == [group for group in expected for _ in vocabulary]
    assert shares["status"].tolist() == vocabulary * len(expected)


def test_outcome_groups_need_no_ordering_and_keep_a_missing_key() -> None:
    class Budget:
        """A grouping value that answers equality but not order."""

    first, second = Budget(), Budget()
    statuses = pd.DataFrame(
        {
            "budget": [second, first, second],
            "panel": [None, "x", None],
            "status": ["success", "timeout", "no_release"],
        },
        index=[10, 3, 7],
    )

    shares = summarize_outcomes(statuses, by=["budget", "panel"])

    per_group = len(ResultStatus)
    assert shares["budget"].tolist() == [second] * per_group + [first] * per_group
    assert shares["panel"].iloc[:per_group].isna().all()
    assert shares["denominator"].tolist() == [2] * per_group + [1] * per_group


def test_outcomes_state_one_denominator_when_no_group_is_named() -> None:
    statuses = pd.DataFrame({"status": ["success", "timeout", "success", "success"]})

    shares = summarize_outcomes(statuses, by=[])

    assert shares["denominator"].unique().tolist() == [4]
    taken = dict(zip(shares["status"], shares["share"]))
    assert taken["success"] == pytest.approx(0.75)
    assert taken["timeout"] == pytest.approx(0.25)


def test_outcomes_refuse_what_they_cannot_account_for() -> None:
    with pytest.raises(ValueError, match="Unrecorded Result status"):
        summarize_outcomes(pd.DataFrame({"status": ["invented"]}), by=[])
    with pytest.raises(ValueError, match="at least one Result"):
        summarize_outcomes(pd.DataFrame({"status": []}), by=[])
    with pytest.raises(ValueError, match="more than once"):
        summarize_outcomes(pd.DataFrame({"status": ["success"]}), by=["eps", "eps"])
    with pytest.raises(ValueError, match="missing column\\(s\\): budget"):
        summarize_outcomes(pd.DataFrame({"status": ["success"]}), by=["budget"])
