"""Public contracts of the shared deterministic utilities."""

from dpstats.utils import split_seed


def test_split_seed_returns_distinct_valid_algorithm_seeds() -> None:
    parent_seed = 2**64 - 1

    children = split_seed(parent_seed, 1_000)

    assert children == split_seed(parent_seed, 1_000)
    assert len(children) == 1_000
    assert len(set(children)) == len(children)
    assert all(0 <= child < 2**64 for child in children)
    assert any(child >= 2**63 for child in children)
