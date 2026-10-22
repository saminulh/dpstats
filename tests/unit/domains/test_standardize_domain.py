import math

import numpy as np
import pytest

from dpstats.domains import LpBallDomain
from dpstats.domains.transforms import standardize_domain


@pytest.mark.parametrize(
    "p, radius", [(1.0, 4.0), (2.0, 2.0), (math.inf, 1.0)], ids=["l1", "l2", "linf"]
)
def test_a_ball_becomes_the_ball_through_the_all_ones_vector(p, radius):
    ball = LpBallDomain(p=p, radius=6.0, center=np.array([1.0, -2.0, 3.0, 0.5]))

    standardized = standardize_domain(ball, center=True).domain

    assert isinstance(standardized, LpBallDomain)
    assert standardized.radius == pytest.approx(radius)
    np.testing.assert_allclose(standardized.center, np.zeros(4))
