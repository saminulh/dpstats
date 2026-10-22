"""Public contracts and hand-calculated covariance-estimation references."""

from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from dpstats.budgets import NO_PRIVACY
from dpstats.domains import BoxDomain, EuclideanSpace
from dpstats.instance import persist_instance
from dpstats.simulations.covariance_estimation import simulate_gaussian
from dpstats.tasks import CovarianceEstimation, list_built_in_metrics
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
    CovarianceEstimationOutput,
)
from dpstats.tasks.covariance_estimation.algorithms import SampleCovariance


def test_sample_covariance_centers_and_divides_by_n() -> None:
    metadata = CovarianceEstimationMetadata(n_samples=2, domain=EuclideanSpace(dim=2))
    input_data = CovarianceEstimationInput(X=np.array([[1.0, 2.0], [3.0, 4.0]]))

    output = SampleCovariance().run(metadata, input_data, NO_PRIVACY, seed=0)

    # Deviations are (-1,-1) and (1,1); ddof=1 would double this matrix.
    np.testing.assert_allclose(output.covariance_est, [[1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize(
    "reference_name, suffix",
    [("ground_truth", "ground_truth"), ("sample_covariance", "sample_covariance")],
)
def test_covariance_metrics_use_frobenius_and_operator_norms(
    reference_name: str, suffix: str
) -> None:
    metrics = list_built_in_metrics(CovarianceEstimation)
    output = CovarianceEstimationOutput(covariance_est=np.ones((2, 2)))
    reference = {reference_name: [[2.0, 0.0], [0.0, 1.0]]}

    # Difference [[-1,1],[1,0]] has squared entries summing to 3 and
    # eigenvalues (-1 +/- sqrt(5))/2.
    assert metrics[f"frobenius_error_to_{suffix}"].calculate(
        output, **reference
    ) == pytest.approx(sqrt(3))
    assert metrics[f"spectral_error_to_{suffix}"].calculate(
        output, **reference
    ) == pytest.approx((1 + sqrt(5)) / 2)


def test_single_scalar_record_has_zero_sample_covariance() -> None:
    metadata = CovarianceEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    output = SampleCovariance().run(
        metadata, CovarianceEstimationInput(X=[[17.0]]), NO_PRIVACY, seed=0
    )
    np.testing.assert_allclose(output.covariance_est, [[0.0]])


def test_indefinite_estimate_is_measured_without_projection() -> None:
    output = CovarianceEstimationOutput(covariance_est=[[2.0, 0.0], [0.0, -1.0]])
    restored = CovarianceEstimationOutput.model_validate_json(output.model_dump_json())
    np.testing.assert_allclose(restored.covariance_est, [[2.0, 0.0], [0.0, -1.0]])
    for metric in list_built_in_metrics(CovarianceEstimation).values():
        reference = {metric.required_names[0]: [[0.0, 0.0], [0.0, 0.0]]}
        expected = sqrt(5) if metric.name.startswith("frobenius") else 2.0
        assert metric.calculate(restored, **reference) == pytest.approx(expected)


def test_symmetry_accepts_rounding_error_without_rewriting_the_matrix() -> None:
    covariance = np.array([[1.0, 0.1 + 0.2], [0.3, 1.0]])
    output = CovarianceEstimationOutput(covariance_est=covariance)
    _, _, references = simulate_gaussian(seed=0, covariance=covariance)

    np.testing.assert_array_equal(output.covariance_est, covariance)
    np.testing.assert_array_equal(references["ground_truth"], covariance)


@pytest.mark.parametrize(
    "estimate, message",
    [
        (np.empty((0, 0)), "square"),
        (np.ones((2, 3)), "square"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 0.3 + 1e-9], [0.3, 1.0]]), "symmetric"),
    ],
)
def test_output_enforces_the_matrix_contract(
    estimate: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CovarianceEstimationOutput(covariance_est=estimate)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_output_leaves_non_finite_entries_to_the_engine(value: float) -> None:
    """An infinite estimate is a release; execution alone refuses NaN."""
    estimate = np.array([[1.0, value], [value, 1.0]])

    output = CovarianceEstimationOutput(covariance_est=estimate)

    np.testing.assert_array_equal(output.covariance_est, estimate)


@pytest.mark.parametrize(
    "rows, message",
    [([[0.0]], "shape"), ([[0.0], [2.0]], "domain")],
)
def test_instance_boundary_enforces_shape_and_domain(
    tmp_path: Path, rows: list[list[float]], message: str
) -> None:
    metadata = CovarianceEstimationMetadata(
        n_samples=2, domain=BoxDomain(bounds=[[-1.0, 1.0]])
    )
    with pytest.raises(ValueError, match=message):
        persist_instance(metadata, CovarianceEstimationInput(X=rows), {}, tmp_path)


def test_gaussian_producer_preserves_authored_shift_and_singular_covariance() -> None:
    _, centered, _ = simulate_gaussian(
        seed=19, mean=[0.0, 0.0], covariance=[[1.0, 1.0], [1.0, 1.0]]
    )
    metadata, shifted, refs = simulate_gaussian(
        seed=19, mean=[100.0, -100.0], covariance=[[1.0, 1.0], [1.0, 1.0]]
    )
    np.testing.assert_allclose(
        shifted.X - centered.X, np.tile([100.0, -100.0], (100, 1))
    )
    np.testing.assert_allclose(centered.X[:, 0], centered.X[:, 1])
    np.testing.assert_allclose(refs["ground_truth"], [[1.0, 1.0], [1.0, 1.0]])
    metadata.validate_input(shifted)


@pytest.mark.parametrize(
    "covariance",
    [
        [[1.0, 2.0], [0.0, 1.0]],
        [[1.0, 0.3 + 1e-9], [0.3, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
)
def test_gaussian_producer_refuses_invalid_population_covariance(covariance) -> None:
    with pytest.raises(ValueError):
        simulate_gaussian(seed=0, covariance=covariance)
