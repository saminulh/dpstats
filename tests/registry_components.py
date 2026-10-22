"""Importable components used by tests that persist a registry and reload it."""

import math
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, ClassVar, Literal, final

from pydantic import JsonValue

from typing_extensions import Self

import numpy as np

from dpstats.algorithm import BaseAlgorithm, Transformation
from dpstats.exceptions import NotApplicable
from dpstats.task import (
    Array1D,
    Array2D,
    Task,
    TaskInput,
    TaskMetadata,
    TaskOutput,
)
from dpstats.algorithm import PrivacyInterpretation
from dpstats.parameters import (
    parameter_field,
    search_range,
    search_values,
)
from dpstats.budgets import ApproxDPBudget, DPBudget, PureDPBudget
from dpstats.sweep import BudgetPolicy


class SimpleMetadata(TaskMetadata):
    n_samples: int
    dim: int

    def validate_input(self, input: TaskInput) -> None:
        input = SimpleInput.model_validate(input)
        if input.X.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"SimpleTask X has shape {input.X.shape}, but its metadata "
                f"declares n_samples={self.n_samples} and dim={self.dim}."
            )


class SimpleInput(TaskInput):
    X: Array2D


class SimpleOutput(TaskOutput):
    result: Array1D


SIMPLE_TASK = Task(
    id="simple_test_task",
    name="Simple Test Task",
    description="A simple task for unit testing",
    metadata_cls=SimpleMetadata,
    input_cls=SimpleInput,
    output_cls=SimpleOutput,
)


class SimpleAlgorithm(BaseAlgorithm):
    TASK = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    scale: float = parameter_field(
        default=1.0, description="Multiplier applied to the mean result."
    )

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        return SimpleOutput(result=input.X.mean(axis=0) * self.scale)


class DeclaredAlgorithm(BaseAlgorithm):
    """Declare a log-range, a discrete and an untuned field on one algorithm."""

    TASK = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    rate: float = parameter_field(
        default=0.1,
        gt=0.0,
        le=1.0,
        search=search_range(0.01, 1.0, scale="log"),
        description="Test tuning knob.",
    )
    mode: Literal["fast", "careful"] = parameter_field(
        default="careful",
        search=search_values("fast", "careful"),
        description="Test discrete tuning knob.",
    )
    fixed: int = 3

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        return SimpleOutput(result=input.X.mean(axis=0))


def recording_transformation(
    outer: Task,
    inner: Task,
    *,
    trace: list[str] | None = None,
    name: str = "P",
    budget_factor: float | None = None,
    child_budget: DPBudget | None = None,
    not_applicable: str | None = None,
) -> type[Transformation]:
    """Build a transformation between two tasks that can record and reshape a call.

    It passes the problem through, advances the seed by one and restores the
    Output unchanged; ``trace`` records each prepare and restore by ``tag``,
    ``budget_factor`` scales a pure-DP budget, ``child_budget`` replaces
    it, and ``not_applicable`` declines the stage.
    """

    class RecordingTransformation(Transformation):
        OUTER_TASK: ClassVar[Task] = outer
        INNER_TASK: ClassVar[Task] = inner

        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.STANDARD

        tag: str = name

        def prepare(
            self, metadata: Any, input: Any, budget: DPBudget, seed: int
        ) -> Any:
            if not_applicable is not None:
                raise NotApplicable(not_applicable)
            if trace is not None:
                trace.append(f"prepare:{self.tag}")
            child = budget
            if budget_factor is not None and isinstance(budget, PureDPBudget):
                child = PureDPBudget(eps=budget.eps * budget_factor)
            if child_budget is not None:
                child = child_budget
            return metadata, input, child, seed + 1, f"restoration:{self.tag}"

        def restore(self, output: Any, restoration: Any) -> Any:
            if trace is not None:
                trace.append(f"restore:{restoration}")
            return output

    return RecordingTransformation


def spy_algorithm(
    task: Task, output_cls: type[Any], record: dict[str, Any] | None = None
) -> type[BaseAlgorithm]:
    """Build an algorithm that records the budget and seed it was offered."""

    class SpyAlgorithm(BaseAlgorithm):
        TASK = task

        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.STANDARD

        def run(self, metadata: Any, input: Any, budget: DPBudget, seed: int) -> Any:
            if record is not None:
                record["budget"] = budget
                record["seed"] = seed
            return output_cls(result=np.zeros(3))

    return SpyAlgorithm


class PacedAlgorithm(SimpleAlgorithm):
    """Hold its cell open long enough for an external event to reach it."""

    delay_s: float

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        time.sleep(self.delay_s)
        return super().run(metadata, input, budget, seed)


class NonprivateAlgorithm(BaseAlgorithm):
    TASK = SIMPLE_TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def run(
        self,
        metadata: SimpleMetadata,
        input: SimpleInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleOutput:
        return SimpleOutput(result=input.X.mean(axis=0))


def simple_instance(
    seed: int, *, n: int = 100, d: int = 5
) -> tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]:
    """Produce one ordinary test Instance payload."""
    X = np.random.default_rng(seed).standard_normal((n, d))
    return (
        SimpleMetadata(n_samples=n, dim=d),
        SimpleInput(X=X),
        {"reference": np.zeros(d).tolist()},
    )


def constant_instance(
    seed: int, *, reference: float
) -> tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]:
    """Produce fixed Input bytes with a caller-selected reference."""
    del seed
    return (
        SimpleMetadata(n_samples=2, dim=1),
        SimpleInput(X=np.zeros((2, 1))),
        {"reference": [reference]},
    )


def sized_instance(
    seed: int, *, n_samples: int = 1024, d: int = 2
) -> tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]:
    """Produce an Input whose supplied sample size is observable in Metadata."""
    X = np.random.default_rng(seed).standard_normal((n_samples, d))
    return (
        SimpleMetadata(n_samples=n_samples, dim=d),
        SimpleInput(X=X),
        {"reference": np.zeros(d).tolist()},
    )


def simple_producer(
    *, n: int = 100, d: int = 5
) -> Callable[[int], tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]]:
    """Bind ordinary test-instance parameters before choosing a seed."""
    return partial(simple_instance, n=n, d=d)


def constant_producer(
    *, reference: float
) -> Callable[[int], tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]]:
    """Bind a test reference before choosing a seed."""
    return partial(constant_instance, reference=reference)


def sized_producer(
    *, n_samples: int = 1024, d: int = 2
) -> Callable[[int], tuple[SimpleMetadata, SimpleInput, dict[str, JsonValue]]]:
    """Bind a test Instance size before choosing a seed."""
    return partial(sized_instance, n_samples=n_samples, d=d)


class InverseSquareDelta(BudgetPolicy):
    """Policy offering a delta of one over the squared sample size."""

    eps: float

    def resolve(self, metadata: TaskMetadata) -> ApproxDPBudget:
        return ApproxDPBudget(eps=self.eps, delta=1.0 / metadata.n_samples**2)


class OversizedDelta(BudgetPolicy):
    """Policy whose rule cannot produce a valid approximate-DP budget."""

    eps: float

    def resolve(self, metadata: TaskMetadata) -> ApproxDPBudget:
        del metadata
        return ApproxDPBudget(eps=self.eps, delta=5.0)


@final
class ExternalBudget(DPBudget):
    """A third-party budget family with additive credits."""

    credits: float

    @property
    def is_unconstrained(self) -> bool:
        return self.credits == math.inf

    def split(self, weights: Sequence[float]) -> tuple[Self, ...]:
        if not weights or any(weight <= 0 for weight in weights):
            raise ValueError("weights must be positive.")
        total = sum(weights)
        return tuple(
            self.model_copy(update={"credits": self.credits * weight / total})
            for weight in weights
        )
