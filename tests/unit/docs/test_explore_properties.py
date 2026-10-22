"""Every property Explore lists for a task computes on that task's Instances."""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dpstats.domains import BoxDomain
from dpstats.tasks import BUILT_IN_TASKS, MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.workspace import init_workspace, open_collection
from explore_properties import PROPERTIES, instance_properties, vif
from tests.unit.tasks.test_all_tasks import PRODUCERS_BY_TASK_ID


def test_every_built_in_task_lists_its_properties() -> None:
    assert set(PROPERTIES) == set(BUILT_IN_TASKS)


@pytest.mark.parametrize("task_id", sorted(PROPERTIES))
def test_every_listed_property_computes_on_the_tasks_simulated_instance(
    task_id: str, tmp_path: Path
) -> None:
    """A listed property that does not work for its task fails here."""
    metadata, input, metric_arguments = PRODUCERS_BY_TASK_ID[task_id][0](seed=42)
    collection = open_collection(init_workspace(tmp_path), BUILT_IN_TASKS[task_id])
    instance = collection.add_instance(metadata, input, metric_arguments)

    row = instance_properties(collection).loc[instance.uuid]

    assert list(row.index) == [prop.__name__ for prop in PROPERTIES[task_id]]
    assert row["n"] == metadata.n_samples
    assert all(math.isfinite(value) for value in row)


DESIGN = ["n", "d", "n_over_d", "design_condition_number"]


def test_the_released_tasks_offer_size_ratio_and_condition_number() -> None:
    """Single-coefficient estimation adds the VIF at its coefficient."""
    assert {
        task_id: [prop.__name__ for prop in PROPERTIES[task_id]]
        for task_id in (
            "linear_regression",
            "mean_estimation",
            "single_coefficient_estimation",
        )
    } == {
        "linear_regression": DESIGN,
        "mean_estimation": DESIGN,
        "single_coefficient_estimation": [*DESIGN, "vif"],
    }


def _vif(X: list[list[float]], index: int, has_intercept: bool) -> float:
    metadata = SimpleNamespace(
        coefficient_of_interest_index=index, has_intercept=has_intercept
    )
    collection = SimpleNamespace(load_input=lambda _: SimpleNamespace(X=np.array(X)))
    return vif(SimpleNamespace(metadata=metadata), collection)


def test_the_vif_is_the_centred_variance_over_the_residual_variance() -> None:
    """x = 1, 2, 3, 4 has centred sum of squares 5; regressed on an intercept
    and z = 0, 0, 1, 1 it leaves residuals -0.5, 0.5, -0.5, 0.5, a residual sum
    of squares of 1, so VIF = 5."""
    X = [[1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 1.0], [1.0, 4.0, 1.0]]
    assert _vif(X, 1, has_intercept=True) == pytest.approx(5.0)


def test_the_vif_is_undefined_for_a_coefficient_with_no_variance() -> None:
    X = [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]
    assert math.isnan(_vif(X, 1, has_intercept=True))


def test_each_property_is_read_from_the_instance(tmp_path: Path) -> None:
    """X has singular values 2 and 1, so κ(X) = 2; n = 3 and d = 2."""
    collection = open_collection(init_workspace(tmp_path), MeanEstimation)
    metadata = MeanEstimationMetadata(
        n_samples=3, domain=BoxDomain(bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]))
    )
    instance = collection.add_instance(
        metadata,
        MeanEstimationInput(X=np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])),
        {},
    )

    row = instance_properties(collection).loc[instance.uuid]

    assert row.to_dict() == pytest.approx(
        {"n": 3.0, "d": 2.0, "n_over_d": 1.5, "design_condition_number": 2.0}
    )


def _notebook_kept(filters: dict[str, list[float | None]]) -> list[str]:
    """The instances the notebook's filter cell keeps over fixed properties."""
    from recreate_notebook import _filter_source

    frame = pd.DataFrame(
        {"n": [10.0, 100.0, 1000.0], "vif": [1.5, math.nan, 40.0]},
        index=["a", "b", "c"],
    )
    namespace: dict[str, Any] = {
        "collection": SimpleNamespace(instance_features=lambda _: frame),
        "INSTANCE_FILTERS": filters,
        "ALGORITHMS": None,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_filter_source("single_coefficient_estimation"), namespace)  # noqa: S102
    return list(namespace["kept"].index)


def test_an_open_range_keeps_an_undefined_value_and_a_set_one_excludes_it() -> None:
    assert _notebook_kept({"vif": [None, None]}) == ["a", "b", "c"]
    assert _notebook_kept({"vif": [None, 100.0]}) == ["a", "c"]
    assert _notebook_kept({"vif": [1.0, 2.0], "n": [None, 500.0]}) == ["a"]
    assert _notebook_kept({"n": [50.0, None]}) == ["b", "c"]
