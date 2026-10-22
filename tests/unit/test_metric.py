"""Public value behavior for explicit post-run Metrics and their arguments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.metric import (
    Metric,
    MetricArgument,
    declared_arguments,
    produce_arguments,
)
from tests.registry_components import SIMPLE_TASK, SimpleOutput


def _first(output: SimpleOutput) -> float:
    return float(output.result[0])


def test_metric_calculates_with_exactly_its_declared_dependencies() -> None:
    """A Metric injects only the values its declaration names."""

    def signed_error(output: SimpleOutput, *, reference: float) -> float:
        return float(output.result[0] - reference)

    metric = Metric(
        task=SIMPLE_TASK,
        name="signed_error",
        calculation=signed_error,
        requires=("reference",),
    )
    output = SimpleOutput(result=np.array([5.0]))

    assert metric.calculate(output, reference=2.0) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="missing.*reference"):
        metric.calculate(output)
    with pytest.raises(ValueError, match="undeclared"):
        metric.calculate(output, reference=2.0, ignored=1.0)


def test_declared_arguments_and_bare_names_are_both_requirements() -> None:
    """A declared argument is required by name; an undeclared key by its string."""
    scale = MetricArgument(name="scale", description="A stored scale.")
    metric = Metric(
        task=SIMPLE_TASK,
        name="scaled",
        calculation=_first,
        requires=(scale, "published_scale", "input", "outputs"),
    )

    assert metric.required_names == ("scale", "published_scale", "input", "outputs")
    assert metric.arguments == (scale,)
    assert repr(scale) == "MetricArgument('scale')"


def test_metric_declarations_reject_mutable_or_invalid_requirements() -> None:
    """Metric declarations are immutable, named dependency values."""
    list_requires: Any = ["reference"]
    with pytest.raises(ValidationError, match="tuple"):
        Metric(
            task=SIMPLE_TASK,
            name="invalid_list",
            calculation=_first,
            requires=list_requires,
        )
    invalid_requires: Any = (1,)
    with pytest.raises(ValidationError, match="MetricArgument"):
        Metric(
            task=SIMPLE_TASK,
            name="invalid_element",
            calculation=_first,
            requires=invalid_requires,
        )
    with pytest.raises(ValueError, match="identifier"):
        Metric(task=SIMPLE_TASK, name="m", calculation=_first, requires=("not a name",))
    with pytest.raises(ValueError, match="repeat"):
        Metric(
            task=SIMPLE_TASK,
            name="m",
            calculation=_first,
            requires=(MetricArgument(name="a", description="A."), "a"),
        )


def test_names_have_deliberate_boundaries() -> None:
    """Metric names are columns; argument names are injected keywords."""
    invalid_name: Any = 1
    with pytest.raises(ValidationError) as refused:
        Metric(task=SIMPLE_TASK, name=invalid_name, calculation=_first)
    assert refused.value.errors()[0]["loc"] == ("name",)
    with pytest.raises(ValidationError, match="at least 1"):
        Metric(task=SIMPLE_TASK, name="", calculation=_first)
    with pytest.raises(ValueError, match="identifier"):
        MetricArgument(name="not a name", description="Bad.")
    for reserved in ("input", "outputs"):
        with pytest.raises(ValueError, match="reserved"):
            MetricArgument(name=reserved, description="Reserved.")
    with pytest.raises(ValueError, match="non-blank"):
        MetricArgument(name="blank", description="  ")


def test_an_unknown_declaration_field_is_refused() -> None:
    """A misspelled field never constructs a Metric with a silent default."""
    with pytest.raises(ValidationError) as refused:
        Metric(task=SIMPLE_TASK, name="m", calculation=_first, lable="Label")  # type: ignore[call-arg]
    assert refused.value.errors()[0]["loc"] == ("lable",)


def test_a_blank_formula_is_refused_at_the_declaration() -> None:
    """An empty formula would render as an empty display-math block."""
    with pytest.raises(ValueError, match="formula"):
        Metric(task=SIMPLE_TASK, name="m", calculation=_first, formula="   ")


def test_a_metric_names_at_most_one_statistical_baseline() -> None:
    """A Metric's baseline is a default divisor, declared or stored by name."""
    scale = MetricArgument(name="scale", description="Scale.")
    metric = Metric(
        task=SIMPLE_TASK,
        name="error",
        calculation=_first,
        requires=("target",),
        baseline=scale,
    )
    assert metric.baseline_name == "scale"
    assert metric.arguments == (scale,)
    stored = Metric(task=SIMPLE_TASK, name="stored", calculation=_first, baseline="se")
    assert stored.baseline_name == "se"
    assert stored.arguments == ()
    bare = Metric(task=SIMPLE_TASK, name="bare", calculation=_first)
    assert bare.baseline is None and bare.baseline_name is None
    assert "None declared" in bare._repr_html_()

    with pytest.raises(ValueError, match="identifier"):
        Metric(
            task=SIMPLE_TASK, name="error", calculation=_first, baseline="not a name"
        )
    several: Any = (scale,)
    with pytest.raises(ValidationError, match="baseline"):
        Metric(task=SIMPLE_TASK, name="error", calculation=_first, baseline=several)


def test_a_metric_may_be_divided_by_what_it_requires() -> None:
    """Nothing restricts which value a Metric is read against."""
    target = MetricArgument(name="target", description="Target.")
    metric = Metric(
        task=SIMPLE_TASK,
        name="error",
        calculation=_first,
        requires=(target,),
        baseline=target,
    )
    assert metric.arguments == (target,)


def test_declared_arguments_collect_once_and_refuse_conflicts() -> None:
    """One name has one declaration across a Task's Metrics."""
    scale = MetricArgument(name="scale", description="Scale.")
    other_scale = MetricArgument(name="scale", description="Another scale.")
    first = Metric(task=SIMPLE_TASK, name="a", calculation=_first, requires=(scale,))
    second = Metric(task=SIMPLE_TASK, name="b", calculation=_first, baseline=scale)

    assert declared_arguments([first, second]) == {"scale": scale}
    conflicting = Metric(
        task=SIMPLE_TASK, name="c", calculation=_first, requires=(other_scale,)
    )
    with pytest.raises(ValueError, match="differently"):
        declared_arguments([first, conflicting])


def test_shared_producers_run_once_and_must_yield_what_they_declare() -> None:
    """Arguments naming one producer share its single computation."""
    calls: list[str] = []

    def fit(metadata: Any, input: Any) -> dict[str, float]:
        calls.append("fit")
        return {"slope": 2.0, "scale": 3.0}

    def other(metadata: Any, input: Any) -> dict[str, float]:
        return {}

    slope = MetricArgument(name="slope", description="Slope.", producer=fit)
    scale = MetricArgument(name="scale", description="Scale.", producer=fit)
    stored = MetricArgument(name="stored", description="Stored.")
    missing = MetricArgument(name="missing", description="Missing.", producer=other)

    assert produce_arguments([slope, scale, stored], None, None) == {
        "slope": 2.0,
        "scale": 3.0,
    }
    assert calls == ["fit"]
    with pytest.raises(ValueError, match="'missing'"):
        produce_arguments([missing], None, None)
