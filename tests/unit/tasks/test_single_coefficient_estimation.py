"""Tests for the single_coefficient_estimation task, its projection and metrics."""

from __future__ import annotations


import numpy as np
import pytest

from dpstats.domains import LpBallDomain
from dpstats.budgets import DPBudget, NO_PRIVACY, ApproxDPBudget
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks import list_built_in_metrics
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.algorithm import CompositeAlgorithm
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ToLinearRegression,
)
from dpstats.simulations.single_coefficient_estimation import (
    simulate_clipped_gaussian,
)
from dpstats.tasks import (
    SingleCoefficientEstimation,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationOutput,
)
from tests.unit.tasks.task_builders import sce_metadata

single_coefficient_metrics = list_built_in_metrics(SingleCoefficientEstimation)
sign_disagreement_with_ols = single_coefficient_metrics["sign_disagreement_with_ols"]


def _make_noiseless_instance(coefficient_of_interest_index: int = 1):
    """A noiseless SCE instance whose OLS fit recovers a known beta exactly."""
    rng = np.random.default_rng(0)
    beta_true = np.array([2.0, -3.0, 0.5])
    X = rng.normal(size=(200, 3))
    y = X @ beta_true
    metadata = sce_metadata(200, coefficient_of_interest_index, 3)
    input_data = SingleCoefficientEstimationInput(X=X, y=y)
    return metadata, input_data, beta_true


def test_coefficient_index_must_be_in_design_domain():
    with pytest.raises(ValueError, match="coordinate"):
        sce_metadata(10, 2, 2)


class TestGaussianSimulatedInstanceDomains:
    """The SCE producer truncates into the domains composed DP algorithms see."""

    def test_declares_matching_l2_balls_and_the_ols_reference(self):
        metadata, input_data, metric_arguments = simulate_clipped_gaussian(
            seed=0,
            dim=3,
            n_samples=200,
            covariate_radius=0.75,
            response_radius=0.5,
        )

        assert isinstance(metadata.design_domain, LpBallDomain)
        assert isinstance(metadata.response_domain, LpBallDomain)
        assert metadata.design_domain.radius == 0.75
        assert metadata.response_domain.radius == 0.5
        assert np.all(np.linalg.norm(input_data.X, axis=1) <= 0.75 + 1e-12)
        assert np.all(np.abs(input_data.y) <= 0.5 + 1e-12)
        expected_ols = np.linalg.lstsq(input_data.X, input_data.y, rcond=None)[0]
        assert metric_arguments["ols_coef"] == pytest.approx(
            expected_ols[metadata.coefficient_of_interest_index]
        )


class TestCoefficientProjection:
    """The transformation projects its configured algorithm's full-beta output."""

    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_ols_projects_to_coefficient_of_interest(self, j: int):
        meta, inp, beta_true = _make_noiseless_instance(coefficient_of_interest_index=j)
        out = CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=OLS()
        ).run(metadata=meta, input=inp, budget=NO_PRIVACY, seed=0)
        assert isinstance(out, SingleCoefficientEstimationOutput)
        assert out.coefficient == pytest.approx(beta_true[j], abs=1e-8)

    def test_forwards_parent_instance_and_projects_released_coordinate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        design_domain = LpBallDomain.at_origin(p=2, radius=4.0, dim=3)
        response_domain = LpBallDomain.at_origin(p=2, radius=5.0, dim=1)
        metadata = sce_metadata(2, 1, design_domain, response_domain)
        input_data = SingleCoefficientEstimationInput(
            X=np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            y=np.array([4.0, -2.0]),
        )
        budget = ApproxDPBudget(eps=2.0, delta=1e-4)
        received: list[
            tuple[LinearRegressionMetadata, LinearRegressionInput, DPBudget, int]
        ] = []

        def fake_run(
            self: OLS,
            parent_metadata: LinearRegressionMetadata,
            parent_input: LinearRegressionInput,
            budget: DPBudget,
            seed: int,
        ) -> LinearRegressionOutput:
            received.append((parent_metadata, parent_input, budget, seed))
            return LinearRegressionOutput(beta=np.array([8.0, -3.0, 5.0]))

        monkeypatch.setattr(OLS, "run", fake_run)

        output = CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=OLS()
        ).run(metadata, input_data, budget, seed=19)

        assert output == SingleCoefficientEstimationOutput(coefficient=-3.0)
        assert len(received) == 1
        parent_metadata, parent_input, offered_budget, seed = received[0]
        assert parent_metadata.n_samples == 2
        assert parent_metadata.design_domain is design_domain
        assert parent_metadata.response_domain is response_domain
        assert parent_input.X is input_data.X
        assert parent_input.y is input_data.y
        assert offered_budget is budget
        assert seed == 19


class TestSignDisagreement:
    """Sign disagreement uses the instance's stored OLS coefficient."""

    def test_sign_disagreement_is_a_lower_is_better_cost(self):
        agree = SingleCoefficientEstimationOutput(coefficient=-2.5)
        disagree = SingleCoefficientEstimationOutput(coefficient=4.0)
        assert sign_disagreement_with_ols.calculate(agree, ols_coef=-3.0) == 0.0
        assert sign_disagreement_with_ols.calculate(disagree, ols_coef=-3.0) == 1.0
