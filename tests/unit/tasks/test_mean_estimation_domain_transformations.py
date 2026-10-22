"""Unit tests for mean-estimation domain transformations."""

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import ApproxDPBudget, DPBudget, NO_PRIVACY
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    OneHotDomain,
    ProductDomain,
)
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import SampleMean
from dpstats.tasks.mean_estimation.transformations import (
    DataBounds,
    Standardize,
)
from tests.unit.tasks.task_builders import box, mean_metadata


class _SpyAlgorithm(BaseAlgorithm):
    """Record the prepared mean problem and release a marker vector."""

    TASK = MeanEstimation
    record: dict[str, Any] | None = None

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if self.record is not None:
            self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        return MeanEstimationOutput(mean_est=np.arange(metadata.dim, dtype=float))


def _record(composition: CompositeAlgorithm) -> dict[str, Any]:
    assert isinstance(composition.algorithm, _SpyAlgorithm)
    assert composition.algorithm.record is not None
    return composition.algorithm.record


def test_standardize_maps_public_blocks_and_restores_the_mean() -> None:
    category = OneHotDomain(dim=2, allow_zero=True)
    metadata = mean_metadata(
        3, ProductDomain(parts=(box((-2.0, 4.0), (10.0, 14.0)), category))
    )
    task_input = MeanEstimationInput(
        X=np.array(
            [
                [-2.0, 10.0, 1.0, 0.0],
                [1.0, 12.0, 0.0, 0.0],
                [4.0, 14.0, 0.0, 1.0],
            ]
        )
    )
    composition = CompositeAlgorithm(
        transformations=(Standardize(),), algorithm=_SpyAlgorithm(record={})
    )

    output = composition.run(
        metadata,
        task_input,
        ApproxDPBudget(eps=1.0, delta=1e-6),
        seed=7,
    )

    record = _record(composition)
    prepared = record["metadata"]
    numeric, prepared_category = prepared.domain.parts
    assert isinstance(numeric, BoxDomain)
    np.testing.assert_allclose(numeric.bounds, [[-1.0, 1.0], [-1.0, 1.0]])
    assert prepared_category is category
    np.testing.assert_allclose(
        record["input"].X,
        [[-1.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(output.mean_est, [1.0, 14.0, 2.0, 3.0])
    assert Standardize().privacy_interpretation is PrivacyInterpretation.STANDARD


def test_standardized_sample_mean_is_exactly_invariant() -> None:
    metadata = mean_metadata(4, box((-2.0, 4.0), (10.0, 14.0)))
    task_input = MeanEstimationInput(
        X=np.array([[-2.0, 10.0], [1.0, 11.0], [3.0, 13.0], [4.0, 14.0]])
    )

    direct = SampleMean().run(metadata, task_input, NO_PRIVACY, seed=0)
    standardized = CompositeAlgorithm(
        transformations=(Standardize(),), algorithm=SampleMean()
    ).run(metadata, task_input, NO_PRIVACY, seed=0)

    np.testing.assert_allclose(standardized.mean_est, direct.mean_est, atol=1e-12)


def test_data_bounds_preserves_structure_and_bounds_numeric_values() -> None:
    category = OneHotDomain(dim=2, allow_zero=True)
    metadata = mean_metadata(3, ProductDomain(parts=(category, EuclideanSpace(dim=2))))
    task_input = MeanEstimationInput(
        X=np.array([[1.0, 0.0, -2.0, 4.0], [0.0, 0.0, 3.0, 4.0], [0.0, 1.0, 0.0, 4.0]])
    )
    composition = CompositeAlgorithm(
        transformations=(DataBounds(),), algorithm=_SpyAlgorithm(record={})
    )

    composition.run(
        metadata,
        task_input,
        ApproxDPBudget(eps=1.0, delta=1e-6),
        seed=2,
    )

    prepared = _record(composition)["metadata"]
    prepared_category, numeric = prepared.domain.parts
    assert prepared_category is category
    assert isinstance(numeric, BoxDomain)
    np.testing.assert_array_equal(numeric.bounds, [[-2.0, 3.0], [4.0, 4.0]])
    assert composition.privacy_interpretation is PrivacyInterpretation.DIAGNOSTIC


def test_standardize_refuses_unbounded_domain_before_reading_input() -> None:
    metadata = mean_metadata(2, 2)
    task_input = MeanEstimationInput(X=np.full((2, 2), np.nan))
    record: dict[str, Any] = {}

    with pytest.raises(InvalidRegime, match="bounded"):
        CompositeAlgorithm(
            transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
        ).run(
            metadata,
            task_input,
            ApproxDPBudget(eps=1.0, delta=1e-6),
            seed=0,
        )

    assert record == {}
