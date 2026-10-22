"""Tests for the ResponseMean linear-regression-to-mean-estimation transformation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import NO_PRIVACY, ApproxDPBudget, DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations import ResponseMean
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
    SampleMean,
)
from tests.unit.tasks.task_builders import box, lr_metadata, mean_metadata


def _instance(
    has_intercept: bool = True,
) -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    n = 5
    x = np.array([3.0, -1.0, 2.0, 0.5, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    if has_intercept:
        design_domain = ProductDomain(parts=(InterceptDomain(), box((-5.0, 5.0))))
        X = np.column_stack((np.ones(n), x))
    else:
        design_domain = EuclideanSpace(dim=2)
        X = np.column_stack((x, x))
    metadata = lr_metadata(n, design=design_domain, response=box((0.0, 12.0)))
    return metadata, LinearRegressionInput(X=X, y=y)


class _RecordingMeanChild(BaseAlgorithm):
    """Native MeanEstimation stand-in capturing the prepared problem it receives."""

    TASK = MeanEstimation
    record: dict[str, Any]

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        self.record.update(input=input, budget=budget, seed=seed)
        self.record["calls"] = self.record.get("calls", 0) + 1
        return MeanEstimationOutput(mean_est=np.array([7.0]))


def test_sample_mean_composition_releases_mean_at_intercept_and_is_nonprivate():
    metadata, input = _instance()

    composition = CompositeAlgorithm(
        transformations=(ResponseMean(),), algorithm=SampleMean()
    )
    output = composition.run(metadata, input, NO_PRIVACY, seed=0)

    # Hand-computed: mean([1.0, 2.0, 3.0, 4.0, 10.0]) = 20.0 / 5 = 4.0.
    assert output.beta == pytest.approx([4.0, 0.0])
    assert composition.privacy_interpretation == PrivacyInterpretation.NONPRIVATE


def test_gaussian_composition_matches_direct_child_run_and_is_standard():
    metadata, input = _instance()
    budget = ApproxDPBudget(eps=1.0, delta=1e-6)
    seed = 5

    composition = CompositeAlgorithm(
        transformations=(ResponseMean(),), algorithm=GaussianMechanism()
    )
    output = composition.run(metadata, input, budget, seed)

    direct_metadata = mean_metadata(metadata.n_samples, metadata.response_domain)
    direct_input = MeanEstimationInput(X=input.y[:, None])
    direct = GaussianMechanism().run(direct_metadata, direct_input, budget, seed)

    assert output.beta == pytest.approx([direct.mean_est[0], 0.0])
    assert composition.privacy_interpretation == PrivacyInterpretation.STANDARD


def test_prepared_input_shape_matches_y_with_budget_and_seed_unchanged():
    metadata, input = _instance()
    budget = ApproxDPBudget(eps=1.0, delta=1e-6)
    child = _RecordingMeanChild(record={})

    CompositeAlgorithm(transformations=(ResponseMean(),), algorithm=child).run(
        metadata, input, budget, seed=11
    )

    assert child.record["input"].X.shape == (metadata.n_samples, 1)
    np.testing.assert_array_equal(child.record["input"].X[:, 0], input.y)
    assert child.record["budget"] is budget
    assert child.record["seed"] == 11


def test_missing_declared_intercept_is_refused_before_the_child_runs():
    metadata, input = _instance(has_intercept=False)
    child = _RecordingMeanChild(record={})

    with pytest.raises(InvalidRegime, match="intercept"):
        CompositeAlgorithm(transformations=(ResponseMean(),), algorithm=child).run(
            metadata, input, NO_PRIVACY, seed=0
        )

    assert "calls" not in child.record
