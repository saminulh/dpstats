"""Public-contract tests for durable external-content references."""

from __future__ import annotations

from typing import Any

import pytest

from dpstats.transport.validators import (
    is_immutable_revision,
    is_normalized_relative_path,
    is_portable_name,
)


@pytest.mark.parametrize(
    "value", ["study-1", "a", "public.npz", "mean_estimation", "a b"]
)
def test_portable_name_admits_one_direct_entry(value: str) -> None:
    """A stored direct-entry reference works on every supported platform."""
    assert is_portable_name(value)


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", " leading", "trailing ", "a/b", "a\\b", "/a", "a/", 42],
)
def test_portable_name_refuses_nonportable_or_nonentry_values(value: Any) -> None:
    assert not is_portable_name(value)


@pytest.mark.parametrize("value", [".", "a", "a/b", "datasets/public.npz"])
def test_normalized_relative_path_preserves_one_spelling(value: str) -> None:
    assert is_normalized_relative_path(value)


@pytest.mark.parametrize(
    "value",
    ["", "a//b", "./a", "a/./b", "a/", "/a", "..", "../a", "a/../b", "a\\b"],
)
def test_normalized_relative_path_refuses_ambiguous_or_escaping_values(
    value: Any,
) -> None:
    assert not is_normalized_relative_path(value)


def test_immutable_revision_accepts_only_a_full_lowercase_git_commit() -> None:
    assert is_immutable_revision("3f2b1c9d8e7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c")
    assert not is_immutable_revision("main")
