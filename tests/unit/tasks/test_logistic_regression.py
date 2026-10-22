"""Tests for the logistic-regression Task, its algorithms, producer and metric."""

import numpy as np
import pytest

from dpstats.tasks import list_built_in_algorithms, list_built_in_metrics
from dpstats.simulations.logistic_regression import (
    simulate_gaussian,
)
from dpstats.budgets import NO_PRIVACY
from dpstats.tasks import LogisticRegression
from dpstats.tasks.logistic_regression import LogisticRegressionOutput

ALGORITHMS = sorted(list_built_in_algorithms(LogisticRegression).items())


@pytest.fixture
def logistic_regression_task():
    """Get the built-in logistic regression task."""
    return LogisticRegression


class TestLogisticRegressionAlgorithms:
    """Exercise every built-in algorithm against a behavioral oracle."""

    def test_separable_data_recovers_sign(self, logistic_regression_task):
        """On a strongly-signaled 1-D problem the fitted coefficient is positive."""
        task = logistic_regression_task
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 1))
        # Strong but not perfectly separable (5% label flips) so y truly tracks
        # the sign of x without driving the optimizer's logits to overflow.
        y = (X[:, 0] > 0).astype(np.float64)
        flip = rng.random(200) < 0.05
        y[flip] = 1.0 - y[flip]
        metadata = task.Metadata(dim=1, n_samples=200)
        input_data = task.Input(X=X, y=y)
        for _, alg_cls in ALGORITHMS:
            out = alg_cls().run(metadata, input_data, budget=NO_PRIVACY, seed=0)
            assert out.theta.shape == (1,)
            assert np.all(np.isfinite(out.theta))
            assert out.theta[0] > 0

    def test_vanilla_gd_survives_extreme_logits(self, logistic_regression_task):
        """Huge covariates drive logits far past the exp overflow threshold."""
        from dpstats.tasks.logistic_regression.algorithms.gd import (
            GD,
        )

        task = logistic_regression_task
        # One always-zero label on a covariate of magnitude 1000: the first
        # update sends theta to -50, so subsequent logits reach -50000.
        metadata = task.Metadata(dim=1, n_samples=1)
        input_data = task.Input(X=np.array([[1000.0]]), y=np.array([0.0]))
        with np.errstate(over="raise"):
            out = GD(lr=0.1, max_iter=5).run(
                metadata, input_data, budget=NO_PRIVACY, seed=0
            )
        assert np.all(np.isfinite(out.theta))
        assert out.theta[0] < 0


class TestLogisticRegressionSimulatedInstances:
    """Tests for built-in simulated-instance producers."""

    def test_generate_binary_labels_and_both_references(self):
        """Labels are 0/1; references carry the simulation truth and the MLE."""
        _, input_data, metric_arguments = simulate_gaussian(
            seed=42, n_samples=100, dim=4
        )

        assert set(np.unique(input_data.y)) <= {0.0, 1.0}
        assert set(metric_arguments) == {"ground_truth", "mle_coef"}
        assert len(metric_arguments["ground_truth"]) == 4


class TestLogisticRegressionMetrics:
    """Tests for the task's metric functions."""

    def test_l2_error_to_ground_truth(self, logistic_regression_task):
        l2_error_to_ground_truth = list_built_in_metrics(LogisticRegression)[
            "l2_error_to_ground_truth"
        ]

        reference = np.array([1.0, -2.0, 3.0])
        output = LogisticRegressionOutput(theta=reference.copy())
        assert l2_error_to_ground_truth.calculate(
            output, ground_truth=reference
        ) == pytest.approx(0.0)
        off = LogisticRegressionOutput(theta=reference + 1.0)
        assert l2_error_to_ground_truth.calculate(
            off, ground_truth=reference
        ) == pytest.approx(np.sqrt(3.0))
