"""Behavior of bounded correlated regression simulations."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from dpstats.instance import Instance, load_instance_input, persist_instance
from dpstats.simulations.linear_regression import simulate_hypercube
from dpstats.tasks import LinearRegression, list_built_in_metrics
from dpstats.tasks.linear_regression import LinearRegressionInput


def test_noiseless_hypercube_preserves_truth_and_round_trips(tmp_path: Path) -> None:
    metadata, data, references = simulate_hypercube(
        seed=42, dim=4, n_samples=100, response_noise=0.0
    )
    truth = np.asarray(references["ground_truth"])
    np.testing.assert_allclose(data.y, data.X @ truth)
    np.testing.assert_allclose(np.asarray(references["ols_coef"]), truth, atol=1e-12)
    assert np.max(np.abs(data.X)) <= 1.0
    instance = persist_instance(metadata, data, references, tmp_path)
    restored = load_instance_input(instance, tmp_path)
    assert isinstance(restored, LinearRegressionInput)
    np.testing.assert_array_equal(restored.X, data.X)
    np.testing.assert_array_equal(restored.y, data.y)


def test_residual_baselines_stand_for_the_fit_and_the_response() -> None:
    """The stored baselines are the OLS fit's own cost and sd(y)."""
    residual_rmse = list_built_in_metrics(LinearRegression)["residual_rmse"]
    _, data, references = simulate_hypercube(seed=7, dim=3, n_samples=200)
    ols_fit = LinearRegression.Output(beta=np.asarray(references["ols_coef"]))
    ols_residual_rmse = references["ols_residual_rmse"]
    response_std = references["response_std"]

    # A baseline a comparison divides by is stored as a plain JSON number.
    assert isinstance(ols_residual_rmse, float)
    assert isinstance(response_std, float)
    assert ols_residual_rmse == pytest.approx(
        residual_rmse.calculate(ols_fit, input=data)
    )
    assert response_std == pytest.approx(float(np.std(data.y)))


def test_a_noiseless_response_leaves_the_ols_fit_no_residual() -> None:
    """A response the design reproduces exactly costs the OLS fit nothing."""
    _, _, references = simulate_hypercube(
        seed=42, dim=4, n_samples=100, response_noise=0.0
    )
    response_std = references["response_std"]

    assert references["ols_residual_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert isinstance(response_std, float)
    assert response_std > 0.0


def test_ols_baselines_match_a_tiny_design_computed_by_hand() -> None:
    """The two new references match a closed-form 2x2 inverse, not the producer's."""
    _, data, references = simulate_hypercube(
        seed=3, dim=2, n_samples=6, response_noise=0.5
    )
    n, d = data.X.shape
    ols_residual_rmse = references["ols_residual_rmse"]
    assert isinstance(ols_residual_rmse, float)
    # ols_residual_rmse is sqrt(RSS/n); the two new references scale by the unbiased
    # RSS/(n - d) instead, n - d = 4 residual degrees of freedom here.
    residual_variance = ols_residual_rmse**2 * n / (n - d)

    # Closed-form 2x2 inverse of X'X, independent of the producer's np.linalg.inv.
    sxx = float(np.sum(data.X[:, 0] ** 2))
    sxy = float(np.sum(data.X[:, 0] * data.X[:, 1]))
    syy = float(np.sum(data.X[:, 1] ** 2))
    determinant = sxx * syy - sxy * sxy
    se_squared_0 = residual_variance * syy / determinant
    se_squared_1 = residual_variance * sxx / determinant

    assert references["ols_prediction_se"] == pytest.approx(
        (residual_variance * d / n) ** 0.5
    )
    assert references["ols_l2_se"] == pytest.approx(
        (se_squared_0 + se_squared_1) ** 0.5
    )


def test_insufficient_rows_refuse_before_the_rank_check() -> None:
    """More free coefficients than rows leaves no residual degrees of freedom."""
    with pytest.raises(ValueError, match="no residual degrees of freedom"):
        simulate_hypercube(seed=0, dim=5, n_samples=2)


def test_a_full_rank_square_design_still_has_no_variance_estimate() -> None:
    """n == d fits every row exactly, leaving no residual to estimate scale."""
    with pytest.raises(ValueError, match="no residual degrees of freedom"):
        simulate_hypercube(seed=0, dim=2, n_samples=2)


@pytest.mark.parametrize("condition_num", [1.0, 10.0, 100.0])
def test_uniform_marginals_and_population_conditioning(condition_num: float) -> None:
    metadata, data, _ = simulate_hypercube(
        seed=12,
        dim=4,
        n_samples=100_000,
        condition_num=condition_num,
        covariate_radius=2.0,
    )
    metadata.validate_input(data)
    np.testing.assert_allclose(data.X.mean(axis=0), 0.0, atol=0.015)
    np.testing.assert_allclose(data.X.var(axis=0), 4.0 / 3.0, rtol=0.015)
    np.testing.assert_allclose(
        np.quantile(data.X, [0.25, 0.5, 0.75], axis=0),
        np.tile([[-1.0], [0.0], [1.0]], (1, 4)),
        atol=0.025,
    )
    assert np.linalg.cond(np.cov(data.X, rowvar=False)) == pytest.approx(
        condition_num, rel=0.05
    )


@pytest.mark.parametrize("direction", ["strong", "weak", "generic"])
def test_signal_and_noise_controls_are_separate(
    direction: Literal["strong", "weak", "generic"],
) -> None:
    metadata, data, references = simulate_hypercube(
        seed=9,
        dim=4,
        n_samples=30_000,
        condition_num=10.0,
        response_noise=0.6,
        signal_direction=direction,
    )
    truth = np.asarray(references["ground_truth"])
    assert np.linalg.norm(truth) == pytest.approx(1.0)
    if direction == "strong":
        np.testing.assert_allclose(truth, [0.5, 0.5, 0.5, 0.5])
    elif direction == "weak":
        assert truth.sum() == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(truth[2:], 0.0)
    residual = data.y - data.X @ truth
    assert residual.std() == pytest.approx(0.6, rel=0.02)
    assert abs(residual.mean()) < 0.015
    metadata.validate_input(data)
    repeated = simulate_hypercube(
        seed=9,
        dim=4,
        n_samples=30_000,
        condition_num=10.0,
        response_noise=0.6,
        signal_direction=direction,
    )
    np.testing.assert_allclose(repeated[1].X, data.X)
    np.testing.assert_allclose(repeated[1].y, data.y)
    changed = simulate_hypercube(
        seed=9,
        dim=4,
        n_samples=100,
        condition_num=100.0,
        response_noise=3.0,
        signal_direction=direction,
    )
    np.testing.assert_allclose(np.asarray(changed[2]["ground_truth"]), truth)


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"dim": 0}, "dim"),
        ({"dim": True}, "dim"),
        ({"n_samples": 0}, "n_samples"),
        ({"condition_num": 0.5}, "condition_num"),
        ({"condition_num": float("nan")}, "condition_num"),
        ({"condition_num": float("inf")}, "condition_num"),
        ({"dim": 1, "condition_num": 2.0}, "condition_num"),
        ({"covariate_radius": 0.0}, "covariate_radius"),
        ({"response_noise": -1.0}, "response_noise"),
        ({"signal_direction": "other"}, "signal_direction"),
        ({"dim": 1, "signal_direction": "weak"}, "signal_direction"),
    ],
)
def test_invalid_simulation_parameters(
    parameters: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_hypercube(seed=0, **parameters)
