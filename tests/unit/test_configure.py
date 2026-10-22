"""Unit tests for dpstats.configure module."""

import pytest

from dpstats.configure import grid
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from tests.registry_components import DeclaredAlgorithm


class TestGrid:
    """One configuration per point of a field grid, registered nowhere."""

    def test_every_point_is_a_distinct_configuration(self):
        arms = grid(DeclaredAlgorithm, rate=[0.5, 0.9], mode=["fast", "careful"])
        name = DeclaredAlgorithm.__qualname__

        assert [str(arm) for arm in arms] == [
            f"{name}[rate=0.5, mode='fast']",
            f"{name}[rate=0.5]",
            f"{name}[rate=0.9, mode='fast']",
            f"{name}[rate=0.9]",
        ]

    def test_an_empty_candidate_list_is_rejected(self):
        with pytest.raises(ValueError, match="no candidate values for 'rate'"):
            grid(DeclaredAlgorithm, rate=[])


def test_grid_configures_one_distinct_identity_and_label_per_point() -> None:
    """A grid arm is an ordinary configured value the registry can file."""
    arms = grid(Ridge, regularization_multiplier=[0.1, 1.0])

    assert [str(arm) for arm in arms] == [
        "Ridge[regularization_multiplier=0.1]",
        "Ridge[regularization_multiplier=1.0]",
    ]
    assert len({arm.uuid for arm in arms}) == 2
