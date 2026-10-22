"""Static regression coverage for built-in object return types."""

from typing import assert_type

from dpstats.simulations.linear_regression import simulate_clipped_gaussian
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)


def check_linear_regression_simulated_instance_types() -> None:
    """Public producer calls expose the Task's concrete value types."""
    metadata, input_data, _ = simulate_clipped_gaussian(seed=0, dim=5, n_samples=2_000)

    assert_type(metadata, LinearRegressionMetadata)
    assert_type(input_data, LinearRegressionInput)
    _ = metadata.dim
    _ = input_data.X
