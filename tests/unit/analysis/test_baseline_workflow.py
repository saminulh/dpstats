"""Profiles reuse measured values and a Metric's own baseline."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from dpstats import analysis
from dpstats.domains import BoxDomain
from dpstats.instance import Instance
from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import linear_regression as lr
from dpstats.tasks import mean_estimation as me
from dpstats.workspace.collection import TaskCollection


def _refuse_computation(*args, **kwargs):
    raise AssertionError(
        "Inspection or profiling attempted measurement or Input access"
    )


def _metric():
    return Metric(
        task=me.MeanEstimation,
        name="error",
        calculation=_refuse_computation,
        baseline=MetricArgument(name="scale", description="A stored scale."),
    )


def _instance(number, arguments):
    metadata = me.MeanEstimationMetadata(
        n_samples=2,
        domain=BoxDomain(bounds=np.array([[0.0, 1.0]])),
    )
    instance = Instance(
        metadata=metadata,
        input_digest=f"{number:064x}",
        metric_arguments={},
    )
    # Exercise inspection of in-memory values, including invalid/nonfinite
    # references, without requiring the persistence format to accept them.
    return instance.model_copy(update={"metric_arguments": arguments})


@pytest.fixture
def no_measurement(monkeypatch):
    monkeypatch.setattr(TaskCollection, "load_input", _refuse_computation)
    monkeypatch.setattr(Metric, "calculate", _refuse_computation)
    monkeypatch.setattr(np.linalg, "lstsq", _refuse_computation)
    monkeypatch.setattr(np.linalg, "inv", _refuse_computation)


def _measured():
    first = _instance(1, {"scale": 2.0})
    second = _instance(2, {"scale": 1.0})
    return pd.DataFrame(
        {
            "task": [me.MeanEstimation.id] * 4,
            "instance": [first, first, second, second],
            "instance_uuid": [first.uuid, first.uuid, second.uuid, second.uuid],
            "algorithm": ["A"] * 4,
            "error": [1.0, 2.0, 2.0, math.inf],
        },
        index=[10, 20, 30, 40],
    )


@pytest.mark.parametrize("kind", ["stored", "table"])
def test_profile_baseline_matches_explicit_path_and_independent_oracle(
    no_measurement, kind
):
    frame = _measured()
    original = frame.copy(deep=True)
    metric = _metric()
    baseline = (
        "scale"
        if kind == "stored"
        else pd.DataFrame(
            {
                "instance_uuid": frame["instance_uuid"].drop_duplicates().tolist(),
                "baseline": [2.0, 1.0],
            }
        )
    )
    thresholds = [0.0, 0.5, 1.0, 2.0, math.inf]
    actual = analysis.compute_profile_curves(
        frame, thresholds, metric=metric, baseline=baseline
    )
    # Half the trials of the first instance pass at 0.5; all pass at 1.
    # Half the second instance's trials pass at 2; its failed trial never does.
    np.testing.assert_allclose(actual["fraction"], [0.0, 0.25, 0.5, 0.75, 0.75])
    normalized = analysis.compute_error_ratio(frame, baseline, metric=metric)
    explicit = analysis.compute_profile_curves(normalized, thresholds, metric=metric)
    pd.testing.assert_frame_equal(actual, explicit)
    pd.testing.assert_frame_equal(frame, original)
    weights = pd.DataFrame(
        {
            "instance_uuid": frame["instance_uuid"].drop_duplicates().tolist(),
            "weight": [3.0, 1.0],
        }
    )
    weighted = analysis.compute_profile_curves(
        frame, thresholds, metric=metric, baseline=baseline, weights=weights
    )
    np.testing.assert_allclose(weighted["fraction"], [0, 0.375, 0.75, 0.875, 0.875])


def test_profile_keeps_existing_zero_infinite_and_failure_conventions(no_measurement):
    instances = [_instance(i, {}) for i in (1, 2, 3)]
    frame = pd.DataFrame(
        {
            "task": [me.MeanEstimation.id] * 5,
            "instance": [instances[i] for i in [0, 0, 1, 2, 2]],
            "instance_uuid": [instances[i].uuid for i in [0, 0, 1, 2, 2]],
            "algorithm": ["A"] * 5,
            "error": [0.0, 2.0, math.inf, 2.0, math.inf],
        }
    )
    baseline = pd.DataFrame(
        {
            "instance_uuid": [instance.uuid for instance in instances],
            "baseline": [0.0, 2.0, math.inf],
        }
    )
    normalized = analysis.compute_error_ratio(frame, baseline, metric=_metric())
    np.testing.assert_array_equal(
        normalized["error"].to_numpy(), [1.0, math.inf, math.inf, 0.0, math.inf]
    )
    curves = analysis.compute_profile_curves(
        frame, [0.0, 1.0, math.inf], metric=_metric(), baseline=baseline
    )
    np.testing.assert_allclose(curves["fraction"], [1 / 6, 1 / 3, 1 / 3])


def test_profile_rejects_missing_and_repeated_normalization(no_measurement):
    frame = _measured()
    with pytest.raises(ValueError, match="no value named"):
        analysis.compute_profile_curves(
            frame, [1.0], metric=_metric(), baseline="absent"
        )
    normalized = analysis.compute_error_ratio(frame, "scale", metric=_metric())
    with pytest.raises(ValueError, match="already carries a baseline"):
        analysis.compute_profile_curves(
            normalized, [1.0], metric=_metric(), baseline="scale"
        )
    missing_table = pd.DataFrame(
        {"instance_uuid": [frame["instance_uuid"].iloc[0]], "baseline": [2.0]}
    )
    with pytest.raises(ValueError, match="have no baseline"):
        analysis.compute_profile_curves(
            frame, [1.0], metric=_metric(), baseline=missing_table
        )


def test_the_metrics_own_baseline_is_the_default_divisor(no_measurement):
    frame = _measured()
    by_default = analysis.compute_error_ratio(frame, metric=_metric())
    by_name = analysis.compute_error_ratio(frame, "scale", metric=_metric())
    pd.testing.assert_frame_equal(by_default, by_name)
    with pytest.raises(ValueError, match="no statistical baseline"):
        analysis.compute_error_ratio(frame, metric=replace(_metric(), baseline=None))
    with pytest.raises(ValueError, match="needs the Metric"):
        analysis.compute_error_ratio(frame, metric="error")
    # Any stored value may be named, whether or not the Metric declares it.
    other = frame.assign(
        instance=[
            instance.model_copy(update={"metric_arguments": {"other": 4.0}})
            for instance in frame["instance"]
        ]
    )
    assert (
        analysis.compute_error_ratio(other, "other", metric=_metric())[
            "baseline"
        ].tolist()
        == [4.0] * 4
    )


def test_metric_display_escapes_text_and_keeps_its_column_name(no_measurement):
    metric = replace(
        _metric(),
        label="<script>alert('x')</script>",
        formula=r"e < s & t",
        requires=("input",),
    )
    html = metric._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;" in html and "&amp;" in html
    assert "input" in html and "scale" in html
    assert str(metric) == "error"
