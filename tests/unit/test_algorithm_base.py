"""
Unit tests for dpstats.algorithm module.
"""

from typing import Literal

import pytest
from pydantic import ValidationError

from dpstats.algorithm import PrivacyInterpretation

from dpstats.algorithm import BaseAlgorithm
from dpstats.parameters import (
    RangeSearch,
    ValuesSearch,
    parameter_field,
    parse_search_range,
    search_range,
    search_values,
)
from tests.registry_components import DeclaredAlgorithm


class TestBaseAlgorithm:
    """Tests for BaseAlgorithm class."""

    def test_configuration_surface(self, simple_algorithm):
        """One declaration drives construction, introspection, and serialization."""
        AlgClass = simple_algorithm
        alg = AlgClass(scale=2.5)

        assert alg.scale == 2.5
        assert list(AlgClass.parameter_info()) == ["scale"]
        assert alg.model_dump(mode="python") == {"scale": 2.5, "label": None}

    def test_equality_is_structural_for_nonportable_authored_classes(self, simple_task):
        """Live local classes compare without requiring a configuration import path."""
        task, _Metadata, _Input, Output = simple_task

        class LocalAlgorithm(BaseAlgorithm):
            TASK = task

            scale: float

            def run(self, metadata, input, budget, seed):
                del metadata, budget, seed
                return Output(result=input.X.mean(axis=0) * self.scale)

        class OuterAlgorithm(BaseAlgorithm):
            TASK = task

            child: BaseAlgorithm

            def run(self, metadata, input, budget, seed):
                return self.child.run(metadata, input, budget, seed)

        assert LocalAlgorithm(scale=1.0, label="first") == LocalAlgorithm(
            scale=1.0, label="second"
        )
        assert LocalAlgorithm(scale=1.0) != LocalAlgorithm(scale=2.0)
        assert OuterAlgorithm(
            child=LocalAlgorithm(scale=1.0, label="first")
        ) == OuterAlgorithm(child=LocalAlgorithm(scale=1.0, label="second"))

        def make_child() -> type[BaseAlgorithm]:
            class FactoryChild(BaseAlgorithm):
                TASK = task

                def run(self, metadata, input, budget, seed):
                    del metadata, budget, seed
                    return Output(result=input.X.mean(axis=0))

            return FactoryChild

        first_child = make_child()
        second_child = make_child()
        assert first_child.__qualname__ == second_child.__qualname__
        assert OuterAlgorithm(child=first_child()) != OuterAlgorithm(
            child=second_child()
        )


class TestAlgorithmWithoutHyperparams:
    """Tests for algorithms without hyperparameters."""

    def test_empty_parameter_surface(self, simple_task):
        """A recipe with no declared parameters needs no extra plumbing."""
        from dpstats.algorithm import BaseAlgorithm

        task, Metadata, Input, Output = simple_task

        class NoHyperparamAlg(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        alg = NoHyperparamAlg()
        assert alg.parameter_info() == {}
        assert alg.model_dump(mode="python") == {"label": None}


class TestPlainConfiguredFields:
    """Ordinary model fields are the default authoring path."""

    def test_required_and_defaulted_fields_need_no_wrapper(self, simple_task):
        task, _Metadata, _Input, Output = simple_task

        class PlainAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            method: str
            iterations: int = 10

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        info = PlainAlgorithm.parameter_info()

        assert info["method"].required
        assert not info["iterations"].required
        assert info["iterations"].default == 10
        assert PlainAlgorithm(method="direct").model_dump(mode="python") == {
            "method": "direct",
            "iterations": 10,
            "label": None,
        }


class TestDeclaredAlgorithmParameters:
    """The configured-model facade derives plumbing from typed fields."""

    def test_one_declaration_drives_defaults_search_and_identity(self):
        implicit = DeclaredAlgorithm()
        explicit = DeclaredAlgorithm(rate=0.1, mode="careful", fixed=3)

        assert implicit.model_dump(mode="python") == explicit.model_dump(mode="python")
        assert {
            name: parameter.default
            for name, parameter in DeclaredAlgorithm.parameter_info().items()
        } == {"rate": 0.1, "mode": "careful", "fixed": 3}
        grid = DeclaredAlgorithm.default_tuning_grid(n_points=3)
        assert grid["rate"] == pytest.approx([0.01, 0.1, 1.0])
        assert grid["mode"] == ["fast", "careful"]
        assert implicit.model_dump(mode="python") == {
            "rate": 0.1,
            "mode": "careful",
            "fixed": 3,
            "label": None,
        }

    def test_declared_grids_do_not_duplicate_field_validation(self, simple_task):
        task, _Metadata, _Input, Output = simple_task

        class InvalidGridAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            iterations: int = parameter_field(
                default=2,
                search=search_values(2, 2.5),
                description="Test tuning knob.",
            )

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        assert InvalidGridAlgorithm.default_tuning_grid() == {"iterations": [2, 2.5]}
        with pytest.raises(ValidationError, match="iterations"):
            InvalidGridAlgorithm(iterations=2.5)

    def test_ordinary_pydantic_annotations_need_no_platform_grammar(self, simple_task):
        task, _Metadata, _Input, Output = simple_task

        class UnsupportedAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            coordinates: tuple[int, int] = (0, 0)

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        algorithm = UnsupportedAlgorithm()
        assert algorithm.coordinates == (0, 0)
        assert UnsupportedAlgorithm.parameter_info()["coordinates"].default == (0, 0)

    def test_presentation_state_is_not_algorithm_configuration(self):
        labelled = DeclaredAlgorithm(label="initial")
        assert labelled.label == "initial"
        assert "label" not in DeclaredAlgorithm.parameter_info()

    def test_search_grid_requires_a_positive_number_of_points(self):
        with pytest.raises(ValueError, match="n_points"):
            DeclaredAlgorithm.default_tuning_grid(n_points=0)


class TestPrintedName:
    """An unlabelled configuration prints as the fields it departs from."""

    @pytest.mark.parametrize(
        "configuration, suffix",
        [
            ({}, ""),
            ({"rate": 0.5}, "[rate=0.5]"),
            ({"mode": "fast", "rate": 0.5}, "[rate=0.5, mode='fast']"),
        ],
        ids=["defaults", "one-departure", "declaration-order"],
    )
    def test_departing_fields_appear_in_declaration_order(self, configuration, suffix):
        assert str(DeclaredAlgorithm(**configuration)) == f"DeclaredAlgorithm{suffix}"

    def test_an_authored_name_wins_over_the_fields(self):
        variant = DeclaredAlgorithm(rate=0.5, label="Aggressive")

        assert str(variant) == f"{variant}" == "Aggressive"

    def test_a_nested_class_prints_by_its_own_name(self):
        """The derived name uses the class name, not its qualified path."""

        class Outer:
            class Nested(DeclaredAlgorithm):
                pass

        assert str(Outer.Nested()) == "Nested"

    def test_repr_keeps_the_class_and_every_field(self):
        """Only ``str`` changes; ``repr`` stays the full configuration."""
        assert repr(DeclaredAlgorithm(label="Aggressive")).startswith(
            "DeclaredAlgorithm(label='Aggressive', "
        )


class TestDefaultSearchGrid:
    """Tests for BaseAlgorithm.tuning_grid (the DP-selector candidate grid)."""

    @pytest.fixture
    def grid_algorithm(self, simple_task):
        """An algorithm exercising every tuning_grid code path."""
        from dpstats.algorithm import BaseAlgorithm
        from dpstats.parameters import (
            parameter_field,
            search_range,
            search_values,
        )

        task, _Metadata, _Input, Output = simple_task

        class GridAlg(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            C_clip: float = parameter_field(
                default=1.0,
                search=search_range(0.01, 1000.0, scale="log"),
                description="Test log-scale tuning knob.",
            )
            alpha: float = parameter_field(
                default=0.5,
                search=search_range(0.0, 1.0),
                description="Test linear-scale tuning knob.",
            )
            num_steps: int = parameter_field(
                default=3,
                search=search_range(2, 5),
                description="Test integer tuning knob.",
            )
            explicit: float = parameter_field(
                default=0.1,
                search=search_values(0.05, 0.1, 0.25),
                description="Test discrete tuning knob.",
            )
            mode: Literal["fast", "exact"] = parameter_field(
                default="fast",
                search=search_values("fast", "exact"),
                description="Test categorical tuning knob.",
            )
            unsound: float = parameter_field(
                default=0.0,
                description="Test knob declaring no tuning search.",
            )
            fixed: float = 0.1

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        return GridAlg

    def test_log_param_geometrically_spaced_and_spans_range(self, grid_algorithm):
        """A scale='log' param yields a geometric grid spanning its endpoints."""
        grid = grid_algorithm.default_tuning_grid(n_points=7)
        c = grid["C_clip"]
        assert len(c) == 7
        assert c[0] <= 0.01
        assert c[-1] >= 1000.0
        ratios = [c[i + 1] / c[i] for i in range(len(c) - 1)]
        # Geometric => constant successive ratios.
        assert all(abs(r - ratios[0]) < 1e-9 for r in ratios)

    def test_linear_param_evenly_spaced(self, grid_algorithm):
        """A param without scale='log' yields an evenly spaced grid."""
        grid = grid_algorithm.default_tuning_grid(n_points=5)
        a = grid["alpha"]
        diffs = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        assert all(abs(d - diffs[0]) < 1e-9 for d in diffs)

    def test_int_param_integers_and_deduped(self, grid_algorithm):
        """Int params round to ints with rounding duplicates removed."""
        grid = grid_algorithm.default_tuning_grid(n_points=7)
        steps = grid["num_steps"]
        assert all(isinstance(s, int) for s in steps)
        assert len(steps) == len(set(steps))  # no dups
        assert min(steps) == 2 and max(steps) == 5

    def test_explicit_search_values_used_verbatim(self, grid_algorithm):
        """Declared discrete candidates are copied as-is."""
        grid = grid_algorithm.default_tuning_grid()
        assert grid["explicit"] == [0.05, 0.1, 0.25]

    def test_categorical_search_values_enumerated_verbatim(self, grid_algorithm):
        """Categorical candidates are enumerated, never interpolated."""
        grid = grid_algorithm.default_tuning_grid()
        assert grid["mode"] == ["fast", "exact"]

    def test_parameters_without_search_policy_are_excluded(self, grid_algorithm):
        """Configured parameters without search policy are omitted."""
        grid = grid_algorithm.default_tuning_grid()
        assert "unsound" not in grid
        assert "fixed" not in grid

    def test_generated_tuning_values_are_search_policy_not_constructor_rules(
        self, simple_task
    ):
        task, _Metadata, _Input, Output = simple_task

        class NonconvexDomainAlgorithm(BaseAlgorithm):
            TASK = task

            @property
            def privacy_interpretation(self) -> PrivacyInterpretation:
                return PrivacyInterpretation.STANDARD

            value: float = parameter_field(
                default=0.0,
                search=search_range(0.0, 1.0),
                description="Test tuning knob.",
            )

            def run(self, metadata, input, budget, seed):
                return Output(result=input.X.mean(axis=0))

        assert NonconvexDomainAlgorithm.default_tuning_grid(n_points=3) == {
            "value": [0.0, 0.5, 1.0]
        }


class TestParseSearchRange:
    """parse_search_range: the one validator of the untyped range-mapping form."""

    def test_valid_mapping_parses_to_typed_range(self):
        parsed = parse_search_range(
            {"lower": 1, "upper": 100, "scale": "log"}, name="lr"
        )
        assert parsed == RangeSearch(lower=1, upper=100, scale="log")

    def test_omitted_scale_uses_the_supplied_default(self):
        parsed = parse_search_range(
            {"lower": 0.5, "upper": 2.0}, name="lr", default_scale="log"
        )
        assert parsed.scale == "log"
        assert parse_search_range({"lower": 0.5, "upper": 2.0}, name="lr").scale == (
            "linear"
        )

    def test_unexpected_keys_are_rejected(self):
        with pytest.raises(ValueError, match="lower/upper"):
            parse_search_range({"lower": 0.0, "upper": 1.0, "step": 0.1}, name="a")
        with pytest.raises(ValueError, match="lower/upper"):
            parse_search_range({"lower": 0.0}, name="a")

    def test_non_numeric_and_bool_endpoints_are_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            parse_search_range({"lower": "0", "upper": 1.0}, name="a")
        with pytest.raises(ValueError, match="numeric"):
            parse_search_range({"lower": False, "upper": True}, name="a")

    def test_nonfinite_endpoints_are_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            parse_search_range({"lower": 0.0, "upper": float("inf")}, name="a")

    def test_reversed_endpoints_are_rejected(self):
        with pytest.raises(ValueError, match="exceed"):
            parse_search_range({"lower": 2.0, "upper": 1.0}, name="a")

    def test_invalid_scale_vocabulary_is_rejected(self):
        with pytest.raises(ValueError, match="invalid scale"):
            parse_search_range(
                {"lower": 0.0, "upper": 1.0, "scale": "geometric"}, name="a"
            )

    def test_log_scale_requires_positive_endpoints(self):
        with pytest.raises(ValueError, match="positive"):
            search_range(0.0, 10.0, scale="log")
        with pytest.raises(ValueError, match="positive"):
            parse_search_range({"lower": 0.0, "upper": 1.0, "scale": "log"}, name="a")


class TestRangeSearchConstructorValidation:
    """RangeSearch validates its own invariants, not only through search_range."""

    def test_reversed_endpoints_are_rejected_on_direct_construction(self):
        with pytest.raises(ValidationError, match="exceed"):
            RangeSearch(lower=5, upper=1, scale="log")

    def test_log_scale_requires_positive_lower_on_direct_construction(self):
        with pytest.raises(ValidationError, match="positive"):
            RangeSearch(lower=0.0, upper=10.0, scale="log")

    def test_nonfinite_endpoints_are_rejected_on_direct_construction(self):
        with pytest.raises(ValidationError, match="finite"):
            RangeSearch(lower=0.0, upper=float("inf"), scale="linear")

    def test_valid_range_still_constructs(self):
        assert RangeSearch(lower=1.0, upper=10.0, scale="log") == search_range(
            1.0, 10.0, scale="log"
        )

    def test_bool_endpoints_are_rejected_on_direct_construction(self):
        with pytest.raises(ValidationError):
            RangeSearch(lower=True, upper=2, scale="linear")

    def test_bool_endpoints_are_rejected_through_search_range(self):
        with pytest.raises(ValidationError):
            search_range(True, 2)


class TestValuesSearchConstructorValidation:
    """ValuesSearch validates its own invariants, not only through search_values."""

    def test_empty_values_are_rejected_on_direct_construction(self):
        with pytest.raises(ValidationError, match="at least one"):
            ValuesSearch(values=())

    def test_nonempty_values_still_construct(self):
        assert ValuesSearch(values=(1, 2, 3)) == search_values(1, 2, 3)
