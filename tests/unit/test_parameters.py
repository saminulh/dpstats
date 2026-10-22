"""Unit tests for dpstats.parameters: description is one home."""

from __future__ import annotations

from typing import Annotated, Literal
from xml.etree import ElementTree

import pytest
from IPython.core.formatters import DisplayFormatter
from pydantic import Field

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.parameters import (
    FieldSymbol,
    ParameterInfo,
    inspect_parameters,
    parameter_field,
)
from dpstats.transport.configuration import PolymorphicModel
from dpstats.tasks.mean_estimation.algorithms.coinpress import CoinPress
from dpstats.tasks.mean_estimation.algorithms.sample_mean import SampleMean


class TestParameterFieldDescription:
    """``description`` is a required, named argument forwarded to ``Field``."""

    def test_description_is_required(self):
        with pytest.raises(TypeError, match="description"):
            parameter_field(default=1.0)  # type: ignore[call-arg]


class TestInspectParametersDescription:
    """``inspect_parameters`` reads ``description`` from the declared field."""

    def test_parameter_info_carries_the_declared_description(self, simple_task):
        task, _Metadata, _Input, Output = simple_task

        class DescribedAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            rate: float = parameter_field(
                default=0.1, description="Test learning rate."
            )

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        info = inspect_parameters(DescribedAlgorithm, BaseAlgorithm)

        assert DescribedAlgorithm.parameter_info() == info
        assert info["rate"] == ParameterInfo(
            annotation=float,
            default=0.1,
            required=False,
            search=None,
            description="Test learning rate.",
            symbol=None,
        )

    def test_a_required_field_still_carries_its_description(self, simple_task):
        task, _Metadata, _Input, Output = simple_task

        class RequiredFieldAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            threshold: float = parameter_field(
                default=..., description="Test required threshold."
            )

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        info = inspect_parameters(RequiredFieldAlgorithm, BaseAlgorithm)

        assert info["threshold"].required is True
        assert info["threshold"].description == "Test required threshold."


class TestFieldSymbol:
    """A field's short math symbol has one home and one reader."""

    def test_inspect_parameters_reads_inline_and_annotated_symbols(self):
        class Sized(PolymorphicModel):
            n_samples: Annotated[int, FieldSymbol("n")] = Field(
                description="Number of sample units."
            )
            clip_norm: float = parameter_field(
                default=1.0, description="Clipping norm.", symbol="C"
            )
            rate: float = parameter_field(default=1.0, description="Rate.")

        info = inspect_parameters(Sized, PolymorphicModel)

        assert info["n_samples"].symbol == "n"
        assert info["clip_norm"].symbol == "C"
        assert info["rate"].symbol is None

    def test_an_empty_symbol_is_refused_at_its_declaration(self):
        with pytest.raises(ValueError, match="symbol"):
            FieldSymbol("  ")


def test_parameter_mapping_preserves_records_order_and_tuning_consumers():
    info = CoinPress.parameter_info()
    before = dict(info)
    assert isinstance(info, dict)
    assert list(info) == ["num_iters", "final_budget_fraction", "failure_probability"]
    assert list(info.items()) == list(before.items())
    assert info["num_iters"].default == 2
    assert info.get("missing") is None
    with pytest.raises(KeyError):
        _ = info["missing"]

    assert CoinPress.default_tuning_grid()["num_iters"] == [1, 2, 3, 4, 10]
    assert str(CoinPress(num_iters=4)) == "CoinPress[num_iters=4]"
    repr(info)
    DisplayFormatter().format(info)
    assert info == before
    assert info["num_iters"] is before["num_iters"]


def test_parameter_mapping_displays_readable_text_and_notebook_table():
    info = CoinPress.parameter_info()
    assert repr(info).startswith(
        "num_iters (int, default=2)\n" "    Number of CoinPress refinement iterations."
    )
    assert "final_budget_fraction (float, default=0.75)" in str(info)
    assert "ParameterInfo(" not in repr(info)

    formats, _ = DisplayFormatter().format(info)
    assert formats["text/plain"] == repr(info)
    table = ElementTree.fromstring(formats["text/html"])
    assert [cell.text for cell in table.findall("thead/tr/th")] == [
        "Parameter",
        "Type",
        "Default",
        "Description",
    ]
    rows = [[cell.text for cell in row] for row in table.findall("tbody/tr")]
    assert rows[0] == [
        "num_iters",
        "int",
        "2",
        "Number of CoinPress refinement iterations.",
    ]
    assert [row[0] for row in rows] == list(info)


def test_parameter_display_distinguishes_required_and_none_and_escapes_html(
    simple_task,
):
    task, _Metadata, _Input, Output = simple_task

    class DisplayExample(BaseAlgorithm):
        TASK = task
        threshold: float = parameter_field(
            default=..., description="Required threshold."
        )
        optional: float | None = parameter_field(
            default=None, description="Optional value."
        )
        mode: Literal["<script>", "safe"] = parameter_field(
            default="<script>", description='Keep <values> & "quotes" as text.'
        )
        weights: list[float] = parameter_field(
            default=[1.0, 2.0], description="Weights.\nOne per component."
        )

        def run(self, metadata, input, budget, seed):
            return Output(result=input.X.mean(axis=0))

    info = DisplayExample.parameter_info()
    assert "threshold (float, required)" in repr(info)
    assert "optional (float | None, default=None)" in repr(info)
    assert "weights (list[float], default=[1.0, 2.0])" in repr(info)
    formats, _ = DisplayFormatter().format(info)
    html = formats["text/html"]
    assert "<script>" not in html
    table = ElementTree.fromstring(html)
    rows = [[cell.text for cell in row] for row in table.findall("tbody/tr")]
    assert rows[0][2] == "required"
    assert rows[1][2] == "None"
    assert rows[2][2:] == ["'<script>'", 'Keep <values> & "quotes" as text.']
    assert "<script>" in rows[2][1]


def test_empty_parameter_mapping_stays_empty_in_both_displays():
    info = SampleMean.parameter_info()
    assert not info
    assert repr(info) == "{}"
    formats, _ = DisplayFormatter().format(info)
    assert ElementTree.fromstring(formats["text/html"]).findall("tbody/tr") == []
