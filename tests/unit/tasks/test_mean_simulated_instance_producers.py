import numpy as np
import pytest

from dpstats.domains import LpBallDomain
from dpstats.simulations.mean_estimation import (
    simulate_clipped_laplacian,
    simulate_clipped_normal,
)
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order


@pytest.mark.parametrize(
    "producer", [simulate_clipped_normal, simulate_clipped_laplacian]
)
def test_clipped_noise_is_projected_into_the_public_ball(
    fake_randomness: FakeRandomness,
    producer,
) -> None:
    # The unit mean direction is drawn first from ``normal``; each producer
    # then draws its own noise matrix, whose third row sits outside the
    # radius-1 noise ball and must come back projected onto it.
    mean_direction = np.array([1.0, 0.0])
    noise = np.array([[0.0, 0.0], [0.3, 0.4], [3.0, 4.0]])
    fake_randomness.script("normal", draws_in_order([mean_direction, noise]))
    fake_randomness.script("laplace", draws_in_order([noise]))
    metadata, input_data, _ = producer(
        seed=0,
        mean_radius=2.0,
        noise_radius_upper_bound=1.0,
        dimension=2,
        n_samples=3,
    )

    np.testing.assert_allclose(
        input_data.X,
        np.array([[2.0, 0.0], [2.3, 0.4], [2.6, 0.8]]),
    )
    assert np.all(np.isfinite(input_data.X))
    assert isinstance(metadata.domain, LpBallDomain)
    assert np.all(
        np.linalg.norm(input_data.X, axis=1) <= metadata.domain.radius + 1e-12
    )
