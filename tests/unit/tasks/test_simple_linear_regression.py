"""Unit tests for the simple_linear_regression (scalar-x, d=1) task.

Each ``linear_regression`` algorithm is presented on this task as a native
composed algorithm (``ToLinearRegression -> OLS``, ``ToLinearRegression -> SSP``, …):
the scalar covariate ``x`` is promoted to a 2-column
design matrix ``[1, x]`` so the parent's fit estimates an intercept.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpstats.algorithm import CompositeAlgorithm
from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.budgets import ApproxDPBudget, DPBudget, NO_PRIVACY
from dpstats.tasks import list_built_in_metrics
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks import (
    SimpleLinearRegression,
)
from dpstats.tasks.simple_linear_regression.transformations import (
    ToLinearRegression,
)
from dpstats.simulations.simple_linear_regression import (
    simulate_clipped_gaussian,
)
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionInput,
    SimpleLinearRegressionMetadata,
    SimpleLinearRegressionOutput,
)
from tests.unit.tasks.task_builders import box, slr_metadata

l2_error_to_ols = list_built_in_metrics(SimpleLinearRegression)["l2_error_to_ols"]


def _make_instance(
    n: int = 300, slope: float = 2.5, intercept: float = -1.3, seed: int = 0
) -> tuple[SimpleLinearRegressionMetadata, SimpleLinearRegressionInput, np.ndarray]:
    """Build a deterministic scalar (x, y) instance and its intercept-OLS fit."""
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(0, 1.0, n), -10.0, 10.0)
    y = slope * x + intercept + rng.normal(0, 0.5, n)
    y = np.clip(y, -20.0, 20.0)
    meta = slr_metadata(n, box((-10.0, 10.0)), box((-20.0, 20.0)))
    inp = SimpleLinearRegressionInput(x=x, y=y)
    slope_ols, intercept_ols = np.polyfit(x, y, 1)
    return meta, inp, np.array([slope_ols, intercept_ols])


class TestMetadataContract:
    def test_metadata_domains_must_be_scalar(self):
        with pytest.raises(ValueError, match="scalar"):
            slr_metadata(10, x=2)


class TestToLinearRegressionTransformation:
    """The public transformation owns the complete child-to-parent wiring contract."""

    @pytest.mark.parametrize(
        "x_domain",
        [
            box((-2.0, 3.0)),
            EuclideanSpace(dim=1),
        ],
        ids=["bounded", "unbounded"],
    )
    def test_forwards_canonical_parent_instance_and_projects_output(
        self, monkeypatch: pytest.MonkeyPatch, x_domain: Domain
    ) -> None:
        y_domain = EuclideanSpace(dim=1)
        metadata = slr_metadata(3, x_domain, y_domain)
        input_data = SimpleLinearRegressionInput(
            x=np.array([-2.0, 0.5, 3.0]),
            y=np.array([4.0, 1.0, -5.0]),
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
            return LinearRegressionOutput(beta=np.array([-4.0, 7.0]))

        monkeypatch.setattr(OLS, "run", fake_run)

        output = CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=OLS()
        ).run(metadata, input_data, budget, seed=19)

        assert output == SimpleLinearRegressionOutput(slope=7.0, intercept=-4.0)
        assert len(received) == 1
        parent_metadata, parent_input, offered_budget, seed = received[0]
        assert parent_metadata.n_samples == metadata.n_samples
        assert parent_metadata.response_domain is y_domain
        assert isinstance(parent_metadata.design_domain, ProductDomain)
        intercept_domain, parent_x_domain = parent_metadata.design_domain.parts
        assert isinstance(intercept_domain, InterceptDomain)
        assert parent_x_domain is x_domain
        np.testing.assert_array_equal(
            parent_input.X,
            np.array([[1.0, -2.0], [1.0, 0.5], [1.0, 3.0]]),
        )
        assert parent_input.y is input_data.y
        assert offered_budget is budget
        assert seed == 19


class TestGaussianSimulatedInstances:
    def test_generate_publishes_its_radii_and_the_intercept_ols_reference(self):
        meta, inp, metric_arguments = simulate_clipped_gaussian(
            seed=42, n_samples=150, x_radius=0.75, response_radius=1.5
        )
        assert isinstance(meta.x_domain, BoxDomain)
        assert isinstance(meta.y_domain, BoxDomain)
        np.testing.assert_array_equal(meta.x_domain.bounds, [[-0.75, 0.75]])
        np.testing.assert_array_equal(meta.y_domain.bounds, [[-1.5, 1.5]])
        assert np.all(np.abs(inp.x) <= 0.75)
        assert np.all(np.abs(inp.y) <= 1.5)
        # The reference is the intercept-OLS fit: a unit slope error scores 1.
        slope_ols, intercept_ols = np.polyfit(inp.x, inp.y, 1)
        good = SimpleLinearRegressionOutput(slope=slope_ols, intercept=intercept_ols)
        bad = SimpleLinearRegressionOutput(
            slope=slope_ols + 1.0, intercept=intercept_ols
        )
        references = metric_arguments["ols_coef"]
        assert l2_error_to_ols.calculate(good, ols_coef=references) == pytest.approx(
            0.0, abs=1e-9
        )
        assert l2_error_to_ols.calculate(bad, ols_coef=references) == pytest.approx(1.0)

    def test_ols_metric_argument_uses_canonical_lift_and_fit(self, monkeypatch):
        import dpstats.simulations.simple_linear_regression as producer_module

        expected = SimpleLinearRegressionOutput(slope=8.0, intercept=-2.0)
        received = []

        def fake_run(self, metadata, input, budget, seed):
            received.append((metadata, input, budget, seed))
            return LinearRegressionOutput(
                beta=np.array([expected.intercept, expected.slope])
            )

        monkeypatch.setattr(producer_module.OLS, "run", fake_run)
        metadata, input_data, metric_arguments = (
            producer_module.simulate_clipped_gaussian(seed=0, n_samples=20)
        )

        # The reference fit runs at the linear-regression boundary, so the spy
        # sees the promoted intercept-first design rather than the scalar one.
        assert len(received) == 1
        promoted_metadata, promoted_input, offered, seed = received[0]
        assert isinstance(promoted_metadata, LinearRegressionMetadata)
        np.testing.assert_array_equal(
            promoted_input.X[:, 0], np.ones(metadata.n_samples)
        )
        np.testing.assert_array_equal(promoted_input.X[:, 1], input_data.x)
        assert (offered, seed) == (NO_PRIVACY, 0)
        assert l2_error_to_ols.calculate(
            SimpleLinearRegressionOutput(slope=8.0, intercept=-2.0),
            ols_coef=metric_arguments["ols_coef"],
        ) == pytest.approx(0.0)


class TestComposedTreatments:
    """A composed LR algorithm runs on an SLR instance and is scored there."""

    def test_ols_slr_recovers_intercept_ols(self):
        meta, inp, ols = _make_instance()
        out = CompositeAlgorithm(
            transformations=(ToLinearRegression(),), algorithm=OLS()
        ).run(metadata=meta, input=inp, budget=NO_PRIVACY, seed=0)
        assert isinstance(out, SimpleLinearRegressionOutput)
        assert out.slope == pytest.approx(ols[0], rel=1e-6)
        assert out.intercept == pytest.approx(ols[1], rel=1e-6)
        # The with-intercept OLS reference is ~0 for the exact OLS fit.
        assert l2_error_to_ols.calculate(out, ols_coef=ols) == pytest.approx(
            0.0, abs=1e-8
        )
