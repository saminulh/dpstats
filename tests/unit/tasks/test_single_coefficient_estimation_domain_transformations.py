"""Unit tests for SCE-native regression domain transformations."""

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.budgets import NO_PRIVACY, ApproxDPBudget, DPBudget
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations import (
    DataBounds as LRDataBounds,
    Standardize as LRStandardize,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.linear_regression.algorithms import OLS
from dpstats.tasks.single_coefficient_estimation.transformations import (
    DataBounds,
    DPWhitening,
    Standardize,
    ToLinearRegression,
)
from tests.unit.tasks.task_builders import box, sce_metadata


class _SpyAlgorithm(BaseAlgorithm):
    """Record the prepared SCE instance and release a marker coefficient."""

    TASK = SingleCoefficientEstimation
    record: dict[str, Any] | None = None

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def run(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> SingleCoefficientEstimationOutput:
        if self.record is not None:
            self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        return SingleCoefficientEstimationOutput(coefficient=3.0)


def _record(composition: CompositeAlgorithm) -> dict[str, Any]:
    """Return the concrete spy child's run record."""
    assert isinstance(composition.algorithm, _SpyAlgorithm)
    assert composition.algorithm.record is not None
    return composition.algorithm.record


def _instance(coefficient_of_interest_index: int = 1):
    metadata = sce_metadata(
        3,
        coefficient_of_interest_index,
        ProductDomain(parts=(InterceptDomain(), box((-2.0, 4.0), (-6.0, 2.0)))),
        box((-5.0, 15.0)),
    )
    task_input = SingleCoefficientEstimationInput(
        X=np.array([[1.0, -2.0, -6.0], [1.0, 1.0, -2.0], [1.0, 4.0, 2.0]]),
        y=np.array([-5.0, 5.0, 15.0]),
    )
    return metadata, task_input


def test_sce_metadata_and_input_extend_linear_regression_contracts() -> None:
    metadata, task_input = _instance()

    assert isinstance(metadata, LinearRegressionMetadata)
    assert isinstance(task_input, LinearRegressionInput)


def test_sce_metadata_retains_inherited_scalar_response_validation() -> None:
    with pytest.raises(ValueError, match="response_domain"):
        sce_metadata(3, 1, 2, 2)


def test_standardize_delegates_and_restores_nonintercept_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, task_input = _instance(coefficient_of_interest_index=2)
    record: dict[str, Any] = {}
    delegated: list[tuple[LinearRegressionMetadata, LinearRegressionInput, bool]] = []
    lr_prepare = LRStandardize.prepare_with_centering

    def recording_prepare(
        self: LRStandardize,
        received_metadata: LinearRegressionMetadata,
        received_input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
        *,
        center: bool,
    ):
        delegated.append((received_metadata, received_input, center))
        return lr_prepare(
            self, received_metadata, received_input, budget, seed, center=center
        )

    monkeypatch.setattr(LRStandardize, "prepare_with_centering", recording_prepare)

    composition = CompositeAlgorithm(
        transformations=(Standardize(),), algorithm=_SpyAlgorithm(record=record)
    )
    output = composition.run(
        metadata, task_input, ApproxDPBudget(eps=1.0, delta=1e-6), seed=8
    )

    record = _record(composition)
    prepared = record["metadata"]
    assert prepared.coefficient_of_interest_index == 2
    assert delegated == [(metadata, task_input, True)]
    np.testing.assert_allclose(record["input"].X[:, 1:], [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(record["input"].y, [-1, 0, 1])
    assert output.coefficient == pytest.approx(7.5)
    assert Standardize().privacy_interpretation is PrivacyInterpretation.STANDARD


def test_standardize_only_scales_for_an_intercept_target() -> None:
    metadata, task_input = _instance(coefficient_of_interest_index=0)
    composition = CompositeAlgorithm(
        transformations=(Standardize(),), algorithm=_SpyAlgorithm(record={})
    )

    output = composition.run(
        metadata, task_input, ApproxDPBudget(eps=1.0, delta=1e-6), seed=0
    )

    record = _record(composition)
    prepared = record["metadata"]
    assert prepared.coefficient_of_interest_index == 0
    assert prepared.has_intercept
    # Half-widths 3 and 4 for the columns and 10 for the response; no shift.
    np.testing.assert_allclose(
        record["input"].X,
        [[1.0, -2 / 3, -1.5], [1.0, 1 / 3, -0.5], [1.0, 4 / 3, 0.5]],
    )
    np.testing.assert_allclose(record["input"].y, [-0.5, 0.5, 1.5])
    # The intercept column is unscaled, so only the response scale returns.
    assert output.coefficient == pytest.approx(30.0)


def test_whitened_chain_recovers_an_intercept_target_without_noise() -> None:
    rng = np.random.default_rng(0)
    nuisances = rng.uniform((-2.0, -6.0), (4.0, 2.0), size=(200, 2))
    X = np.column_stack([np.ones(200), nuisances])
    y = X @ np.array([1.5, -0.4, 0.2])
    metadata = sce_metadata(
        200,
        0,
        ProductDomain(parts=(InterceptDomain(), box((-2.0, 4.0), (-6.0, 2.0)))),
        box((-10.0, 10.0)),
    )
    chain = CompositeAlgorithm(
        transformations=(Standardize(), DPWhitening(), ToLinearRegression()),
        algorithm=OLS(),
    )

    # Whitening maps the nuisances invertibly and keeps the intercept column,
    # so least squares returns the same intercept as on the raw data.
    output = chain.run(
        metadata,
        SingleCoefficientEstimationInput(X=X, y=y),
        NO_PRIVACY,
        seed=3,
    )

    assert output.coefficient == pytest.approx(1.5, abs=1e-6)


def test_data_bounds_delegates_and_preserves_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = sce_metadata(
        3, 1, ProductDomain(parts=(InterceptDomain(), EuclideanSpace(dim=2)))
    )
    task_input = SingleCoefficientEstimationInput(
        X=np.array([[1.0, -2.0, 4.0], [1.0, 3.0, 4.0], [1.0, 0.0, 4.0]]),
        y=np.array([-1.0, 5.0, 2.0]),
    )
    record: dict[str, Any] = {}
    delegated: list[tuple[LinearRegressionMetadata, LinearRegressionInput]] = []
    lr_prepare = LRDataBounds.prepare

    def recording_prepare(
        self: LRDataBounds,
        received_metadata: LinearRegressionMetadata,
        received_input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ):
        delegated.append((received_metadata, received_input))
        return lr_prepare(self, received_metadata, received_input, budget, seed)

    monkeypatch.setattr(LRDataBounds, "prepare", recording_prepare)
    composition = CompositeAlgorithm(
        transformations=(DataBounds(),), algorithm=_SpyAlgorithm(record=record)
    )

    output = composition.run(
        metadata, task_input, ApproxDPBudget(eps=1.0, delta=1e-6), seed=2
    )

    record = _record(composition)
    prepared = record["metadata"]
    assert prepared.coefficient_of_interest_index == 1
    assert delegated == [(metadata, task_input)]
    np.testing.assert_array_equal(
        prepared.design_domain.parts[1].bounds, [[-2.0, 3.0], [4.0, 4.0]]
    )
    np.testing.assert_array_equal(prepared.response_domain.bounds, [[-1.0, 5.0]])
    assert output.coefficient == 3.0
    assert composition.privacy_interpretation is PrivacyInterpretation.DIAGNOSTIC
