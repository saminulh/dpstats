"""Instance-side Metric availability without Input access or computation."""

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

import dpstats.workspace.collection as _collection_module
from dpstats.analysis import get_metric_availability
from dpstats.instance import Instance, identify_instance
from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import MeanEstimation
from tests.registry_components import SIMPLE_TASK, simple_instance


def _must_not_calculate(*args: Any, **kwargs: Any) -> float:
    raise AssertionError("Availability must not execute a calculation.")


@pytest.fixture
def instances() -> list[Instance]:
    metadata, input, references = simple_instance(0, n=2, d=1)
    return [
        identify_instance(metadata, input, references),
        identify_instance(*simple_instance(1, n=2, d=1)[:2], {}),
    ]


def test_matrix_checks_direct_and_group_dependencies_without_loading_inputs(
    instances: list[Instance], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_collection_module, "load_instance_input", _must_not_calculate)
    output_only = Metric(SIMPLE_TASK, "output_only", _must_not_calculate)
    stored = Metric(SIMPLE_TASK, "stored", _must_not_calculate, ("reference",))
    raw = Metric(SIMPLE_TASK, "raw", _must_not_calculate, ("input",))
    grouped = Metric(
        SIMPLE_TASK, "grouped", _must_not_calculate, ("outputs", "reference")
    )
    combined = Metric(
        SIMPLE_TASK, "combined", _must_not_calculate, ("input", "outputs", "reference")
    )
    foreign = Metric(MeanEstimation, "foreign", _must_not_calculate)
    metrics = {
        metric.name: metric
        for metric in (output_only, stored, raw, grouped, combined, foreign)
    }

    actual = get_metric_availability(
        {instance.uuid: instance for instance in instances}.values(), metrics.values()
    )

    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "output_only": ["available", "available"],
                "stored": ["available", "unavailable"],
                "raw": ["available_with_input", "available_with_input"],
                "grouped": ["available", "unavailable"],
                "combined": ["available_with_input", "unavailable"],
                "foreign": ["unavailable", "unavailable"],
            },
            index=pd.Index(
                [instance.uuid for instance in instances], name="instance_uuid"
            ),
        ),
    )


def test_single_metric_and_iterables_preserve_order_and_unique_instances(
    instances: list[Instance],
) -> None:
    metric = Metric(SIMPLE_TASK, "reference_error", _must_not_calculate, ("reference",))
    selected: Iterable[Instance] = iter([instances[1], instances[0], instances[1]])
    matrix = get_metric_availability(selected, metric)
    assert matrix.index.tolist() == [instances[1].uuid, instances[0].uuid]
    assert matrix[metric.name].tolist() == ["unavailable", "available"]
    pd.testing.assert_frame_equal(
        get_metric_availability(instances, metric),
        get_metric_availability(instances, iter([metric])),
    )


def test_empty_axes_and_duplicate_metric_names(instances: list[Instance]) -> None:
    metric = Metric(SIMPLE_TASK, "cost", _must_not_calculate)
    assert get_metric_availability([], metric).columns.tolist() == ["cost"]
    assert get_metric_availability(instances, []).index.tolist() == [
        instance.uuid for instance in instances
    ]
    assert get_metric_availability([], []).index.name == "instance_uuid"
    with pytest.raises(ValueError, match="distinct.*names"):
        get_metric_availability(instances, [metric, metric])


def test_a_selected_metric_cannot_redeclare_a_library_argument() -> None:
    """A library-produced name keeps its producer whatever else is selected."""
    shadow = MetricArgument(name="sample_mean", description="Stored instead.")
    metric = Metric(MeanEstimation, "shadowed", _must_not_calculate, (shadow,))

    with pytest.raises(ValueError, match="differently"):
        get_metric_availability([], metric)


def test_conflicting_references_for_one_uuid_are_not_silently_overwritten(
    instances: list[Instance],
) -> None:
    instance = instances[0]
    conflicting = Instance(
        metadata=instance.metadata,
        input_digest=instance.input_digest,
        metric_arguments={},
    )
    assert conflicting.uuid == instance.uuid
    metric = Metric(SIMPLE_TASK, "cost", _must_not_calculate, ("reference",))
    with pytest.raises(ValueError, match="conflicting Metric arguments"):
        get_metric_availability([instance, conflicting], metric)
