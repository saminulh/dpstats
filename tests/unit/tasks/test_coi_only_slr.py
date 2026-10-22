"""COIOnlySLR through the public CompositeAlgorithm surface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import DPBudget, NO_PRIVACY, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.tasks import SimpleLinearRegression
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)
from dpstats.tasks.simple_linear_regression.algorithms.dpgd import (
    DPGD,
)
from dpstats.tasks.simple_linear_regression.algorithms.noisy_stats import (
    NoisyStats,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    COIOnlySLR,
)


class _RecordingChild(BaseAlgorithm):
    """Native SLR stand-in capturing the prepared problem it receives."""

    TASK = SimpleLinearRegression
    record: dict[str, Any]

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    def run(
        self,
        metadata: SimpleLinearRegressionMetadata,
        input: SimpleLinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> SimpleLinearRegressionOutput:
        self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        self.record["calls"] = self.record.get("calls", 0) + 1
        return SimpleLinearRegressionOutput(slope=41.5, intercept=-9.0)


def _instance(
    coi: int,
    design_domain: ProductDomain,
    X: np.ndarray,
    y: np.ndarray,
    response_domain: BoxDomain | None = None,
) -> tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]:
    metadata = SingleCoefficientEstimationMetadata(
        n_samples=X.shape[0],
        coefficient_of_interest_index=coi,
        design_domain=design_domain,
        response_domain=response_domain
        or BoxDomain(bounds=np.array([[float(y.min()) - 1.0, float(y.max()) + 1.0]])),
    )
    return metadata, SingleCoefficientEstimationInput(X=X, y=y)


def _rct_style_instance() -> (
    tuple[SingleCoefficientEstimationMetadata, SingleCoefficientEstimationInput]
):
    """Intercept, one nuisance covariate, and a COI column, each its own part."""
    rng = np.random.default_rng(0)
    n = 60
    z = rng.uniform(0.0, 1.0, size=n)
    w = rng.normal(size=n)
    y = 1.0 + 4.0 * z + 0.7 * w + rng.normal(scale=0.05, size=n)
    X = np.column_stack([np.ones(n), z, w])
    design_domain = ProductDomain(
        parts=(
            InterceptDomain(),
            BoxDomain(bounds=np.array([[0.0, 1.0]])),
            BoxDomain(bounds=np.array([[w.min() - 1.0, w.max() + 1.0]])),
        )
    )
    return _instance(coi=1, design_domain=design_domain, X=X, y=y)


class TestSameSeedEquality:
    """The composition equals directly running the child on the selected pair."""

    @pytest.mark.parametrize(
        "child, budget",
        [
            (DPGD(tau=1.0, num_steps=20), PureDPBudget(eps=3.0)),
            (NoisyStats(), PureDPBudget(eps=3.0)),
        ],
    )
    def test_composed_output_matches_direct_child_run(
        self, child: BaseAlgorithm, budget: PureDPBudget
    ) -> None:
        metadata, task_input = _rct_style_instance()
        seed = 27

        composed = CompositeAlgorithm(
            transformations=(COIOnlySLR(),), algorithm=child
        ).run(metadata, task_input, budget=budget, seed=seed)

        j = metadata.coefficient_of_interest_index
        direct_metadata = SimpleLinearRegressionMetadata(
            n_samples=metadata.n_samples,
            x_domain=metadata.design_domain.select_coordinates(
                np.asarray([j], dtype=np.int64)
            ),
            y_domain=metadata.response_domain,
        )
        direct_input = SimpleLinearRegressionInput(x=task_input.X[:, j], y=task_input.y)
        direct = child.run(direct_metadata, direct_input, budget=budget, seed=seed)

        assert isinstance(composed, SingleCoefficientEstimationOutput)
        assert composed.coefficient == pytest.approx(direct.slope)


class TestBudgetAndOutputWiring:
    """Only the released slope reaches the outer task, budget and seed intact."""

    def test_only_slope_returned_and_budget_reaches_child_once(self) -> None:
        metadata, task_input = _rct_style_instance()
        budget = PureDPBudget(eps=2.5)
        child = _RecordingChild(record={})

        composition = CompositeAlgorithm(
            transformations=(COIOnlySLR(),), algorithm=child
        )
        output = composition.run(metadata, task_input, budget=budget, seed=13)

        assert output == SingleCoefficientEstimationOutput(coefficient=41.5)
        assert child.record["calls"] == 1
        assert child.record["budget"] is budget
        assert child.record["seed"] == 13


class TestDomainSelection:
    """Exact COI-domain and column selection for supported domain products."""

    def test_single_column_part_is_selected_unchanged(self) -> None:
        coi_bounds = np.array([[-3.0, 3.0]])
        design_domain = ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.array([[-2.0, 2.0]])),
                BoxDomain(bounds=coi_bounds),
            )
        )
        X = np.column_stack(
            [
                np.ones(4),
                np.array([0.1, -0.2, 0.3, -0.4]),
                np.array([1.0, -1.0, 2.0, -2.0]),
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metadata, task_input = _instance(coi=2, design_domain=design_domain, X=X, y=y)

        prepared_metadata, prepared_input, _, _, _ = COIOnlySLR().prepare(
            metadata, task_input, NO_PRIVACY, 0
        )

        assert isinstance(prepared_metadata.x_domain, ProductDomain)
        assert len(prepared_metadata.x_domain.parts) == 1
        selected = prepared_metadata.x_domain.parts[0]
        assert isinstance(selected, BoxDomain)
        np.testing.assert_allclose(selected.bounds, coi_bounds)
        np.testing.assert_array_equal(prepared_input.x, X[:, 2])
        assert prepared_metadata.y_domain is metadata.response_domain

    def test_one_coordinate_of_a_wider_one_hot_block_widens_to_allow_zero(self) -> None:
        # Selecting one coordinate out of a width-3 one-hot block yields the
        # canonical width-1 allow_zero image, per Domain.select_coordinates.
        design_domain = ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.array([[-2.0, 2.0]])),
                OneHotDomain(dim=3),
            )
        )
        X = np.column_stack(
            [
                np.ones(4),
                np.array([1.0, -1.0, 0.5, -0.5]),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metadata, task_input = _instance(coi=2, design_domain=design_domain, X=X, y=y)

        prepared_metadata, prepared_input, _, _, _ = COIOnlySLR().prepare(
            metadata, task_input, NO_PRIVACY, 0
        )

        assert prepared_metadata.x_domain == ProductDomain(
            parts=(OneHotDomain(dim=1, allow_zero=True),)
        )
        np.testing.assert_array_equal(prepared_input.x, [1.0, 0.0, 0.0, 1.0])


class TestPublicRefusals:
    """Both regime limits are terminal public refusals from metadata alone."""

    def test_no_declared_intercept_is_refused(self) -> None:
        design_domain = ProductDomain(
            parts=(
                BoxDomain(bounds=np.array([[-2.0, 2.0]])),
                BoxDomain(bounds=np.array([[-2.0, 2.0]])),
            )
        )
        metadata, task_input = _instance(
            coi=1,
            design_domain=design_domain,
            X=np.array([[1.0, 1.0], [2.0, -1.0], [3.0, 0.5]]),
            y=np.array([1.0, 2.0, 3.0]),
        )
        with pytest.raises(InvalidRegime, match="declares an intercept"):
            COIOnlySLR().prepare(metadata, task_input, NO_PRIVACY, 0)

    def test_coi_on_the_declared_intercept_is_refused(self) -> None:
        design_domain = ProductDomain(
            parts=(InterceptDomain(), BoxDomain(bounds=np.array([[-2.0, 2.0]])))
        )
        metadata, task_input = _instance(
            coi=0,
            design_domain=design_domain,
            X=np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 0.5]]),
            y=np.array([1.0, 2.0, 3.0]),
        )
        with pytest.raises(InvalidRegime, match="other than the declared intercept"):
            COIOnlySLR().prepare(metadata, task_input, NO_PRIVACY, 0)


class TestOmittedVariableFixtures:
    """Nonprivate algebra fixtures locate the treatment-only mismatch exactly."""

    def _fit_full_and_treatment_only(
        self, z: np.ndarray, w: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        n = z.shape[0]
        X = np.column_stack([np.ones(n), z, w])
        full_coi = float(np.linalg.lstsq(X, y, rcond=None)[0][1])

        design_domain = ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(
                    bounds=np.array([[float(z.min()) - 1.0, float(z.max()) + 1.0]])
                ),
                BoxDomain(
                    bounds=np.array([[float(w.min()) - 1.0, float(w.max()) + 1.0]])
                ),
            )
        )
        metadata, task_input = _instance(coi=1, design_domain=design_domain, X=X, y=y)
        composed = CompositeAlgorithm(
            transformations=(COIOnlySLR(),), algorithm=NoisyStats()
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)
        return full_coi, composed.coefficient

    def test_orthogonal_nuisance_leaves_the_coefficient_unchanged(self) -> None:
        rng = np.random.default_rng(1)
        n = 100
        z = rng.uniform(-1.0, 1.0, size=n)
        z_centered = z - z.mean()
        w_raw = rng.normal(size=n)
        # Exact Gram-Schmidt orthogonalization against centered z: the
        # nuisance column carries no linear association with the COI column.
        w = (
            w_raw
            - (np.dot(w_raw, z_centered) / np.dot(z_centered, z_centered)) * z_centered
        )
        y = 1.0 + 3.0 * z + 2.0 * w + rng.normal(scale=1e-3, size=n)

        full_coi, treatment_only = self._fit_full_and_treatment_only(z, w, y)

        assert treatment_only == pytest.approx(full_coi, abs=1e-6)

    def test_correlated_nuisance_moves_the_coefficient(self) -> None:
        rng = np.random.default_rng(2)
        n = 100
        z = rng.uniform(-1.0, 1.0, size=n)
        w = 0.9 * z + rng.normal(scale=0.3, size=n)
        y = 1.0 + 3.0 * z + 2.0 * w + rng.normal(scale=1e-3, size=n)

        full_coi, treatment_only = self._fit_full_and_treatment_only(z, w, y)

        assert abs(treatment_only - full_coi) > 0.5
