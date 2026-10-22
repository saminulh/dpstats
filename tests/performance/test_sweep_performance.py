"""Opt-in wall-clock smells for public Sweep execution paths."""

from __future__ import annotations

import json
import statistics
import time
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from dpstats.workspace import init_workspace, open_collection
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
    Transformation,
)
from dpstats.budgets import DPBudget, NoPrivacy
from dpstats.domains import (
    BoxDomain,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Task
from dpstats.tasks import LinearRegression, SingleCoefficientEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    LargestOneHotFWLDemean,
    ToLinearRegression,
)
from tests.sweep_helpers import write_sweep

pytestmark = pytest.mark.performance


class CheapLinearRegression(BaseAlgorithm):
    """Keep the timing dominated by FWL rather than the child algorithm."""

    TASK: ClassVar[Task] = LinearRegression
    variant: int

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.NONPRIVATE

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        del input, budget, seed
        return LinearRegressionOutput(beta=np.full(metadata.dim, self.variant))


class TaggedFWLDemean(Transformation):
    """Run equivalent FWL work under distinct configured identities."""

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    tag: int

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return LargestOneHotFWLDemean().privacy_interpretation

    def prepare(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        SingleCoefficientEstimationMetadata,
        SingleCoefficientEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        return LargestOneHotFWLDemean().prepare(metadata, input, budget, seed)

    def restore(
        self,
        output: SingleCoefficientEstimationOutput,
        restoration: None,
    ) -> SingleCoefficientEstimationOutput:
        return output


def fwl_instance(
    seed: int,
) -> tuple[
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationInput,
    dict[str, Any],
]:
    """Construct enough grouped rows for repeated FWL work to be observable."""
    rng = np.random.default_rng(seed)
    n_samples = 100_000
    groups = 50
    group = rng.integers(groups, size=n_samples)
    one_hot = np.eye(groups)[group]
    numeric = rng.normal(size=(n_samples, 1))
    input = SingleCoefficientEstimationInput(
        X=np.column_stack((np.ones(n_samples), numeric, one_hot)),
        y=rng.normal(size=n_samples),
    )
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=n_samples,
        coefficient_of_interest_index=1,
        design_domain=ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.array([[-8.0, 8.0]])),
                OneHotDomain(dim=groups),
            )
        ),
        response_domain=BoxDomain(bounds=np.array([[-8.0, 8.0]])),
    )
    return metadata, input, {"reference": 0.0}


def timed_sweep(
    root: Path,
    algorithms: list[CompositeAlgorithm],
) -> float:
    """Time only execution after constructing an equivalent stored Sweep."""
    collection = open_collection(init_workspace(root), SingleCoefficientEstimation)
    sweep = write_sweep(collection, algorithms, [fwl_instance], [NoPrivacy()])
    started_at = time.perf_counter()
    results = collection.run_sweep(sweep)
    elapsed = time.perf_counter() - started_at
    if len(results) != len(algorithms) or set(results["status"]) != {"success"}:
        raise RuntimeError("The timing workload did not complete successfully.")
    return elapsed


def test_shared_fwl_planning_reports_its_speedup(tmp_path: Path) -> None:
    """Report when shared preprocessing stops beating equivalent repeated work."""
    shared = [
        CompositeAlgorithm(
            transformations=(LargestOneHotFWLDemean(), ToLinearRegression()),
            algorithm=CheapLinearRegression(variant=variant),
        )
        for variant in range(12)
    ]
    repeated = [
        CompositeAlgorithm(
            transformations=(TaggedFWLDemean(tag=variant), ToLinearRegression()),
            algorithm=CheapLinearRegression(variant=variant),
        )
        for variant in range(12)
    ]

    shared_seconds = [
        timed_sweep(tmp_path / f"shared-{repeat}", shared) for repeat in range(3)
    ]
    repeated_seconds = [
        timed_sweep(tmp_path / f"repeated-{repeat}", repeated) for repeat in range(3)
    ]
    shared_median = statistics.median(shared_seconds)
    repeated_median = statistics.median(repeated_seconds)
    speedup = repeated_median / shared_median
    report = {
        "benchmark": "shared_fwl_planning",
        "repeated_median_s": repeated_median,
        "shared_median_s": shared_median,
        "speedup": speedup,
    }
    print(json.dumps(report, sort_keys=True))
    if speedup < 2.0:
        warnings.warn(
            f"Performance smell: shared FWL planning is only {speedup:.2f}x faster "
            "than equivalent repeated preprocessing.",
            stacklevel=1,
        )
