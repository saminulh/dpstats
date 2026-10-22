"""Shared NumPy aliases own array normalization and basic validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dpstats.transport.configuration import PolymorphicModel
from dpstats.task import Array1D, Array2D, IntArray1D


class ArrayValue(PolymorphicModel):
    vector: Array1D
    matrix: Array2D
    indices: IntArray1D


def test_array_aliases_normalize_numeric_lists_once():
    value = ArrayValue(
        vector=[1, 2],
        matrix=[[1.0, 2.0], [3.0, 4.0]],
        indices=[0, 2],
    )

    assert isinstance(value.vector, np.ndarray)
    assert value.vector.dtype == np.dtype(float)
    assert value.matrix.shape == (2, 2)
    assert np.issubdtype(value.indices.dtype, np.integer)


def test_floating_alias_normalizes_integer_arrays_and_preserves_float_dtype():
    integer_vector = np.array([1, 2], dtype=np.int64)
    float32_vector = np.array([1.0, 2.0], dtype=np.float32)

    integer_value = ArrayValue(
        vector=integer_vector,
        matrix=np.array([[1.0]]),
        indices=np.array([0], dtype=np.int32),
    )
    float32_value = ArrayValue(
        vector=float32_vector,
        matrix=np.array([[1.0]]),
        indices=np.array([0], dtype=np.int32),
    )

    assert integer_value.vector.dtype == np.dtype(float)
    assert float32_value.vector is float32_vector
    assert float32_value.vector.dtype == np.dtype(np.float32)
    assert np.array_equal(integer_value.vector, np.array([1.0, 2.0]))
    assert np.array_equal(float32_value.vector, np.array([1.0, 2.0]))


def test_integer_alias_preserves_integer_dtype():
    indices = np.array([0, 2], dtype=np.int32)
    value = ArrayValue(
        vector=[1.0],
        matrix=[[1.0]],
        indices=indices,
    )

    assert value.indices is indices
    assert value.indices.dtype == np.dtype(np.int32)


def test_integer_alias_rejects_values_outside_int64_range():
    """A uint64 value the store cannot persist as int64 is refused."""
    with pytest.raises(ValidationError, match="int64") as excinfo:
        ArrayValue(
            vector=np.array([1.0]),
            matrix=np.array([[1.0]]),
            indices=np.array([2**63], dtype=np.uint64),
        )

    assert str(2**63) in str(excinfo.value)


def test_integer_alias_accepts_uint64_values_within_int64_range():
    """The largest int64 value, and an empty array, still validate."""
    largest = np.array([2**63 - 1], dtype=np.uint64)
    empty = np.array([], dtype=np.uint64)

    largest_value = ArrayValue(
        vector=np.array([1.0]), matrix=np.array([[1.0]]), indices=largest
    )
    empty_value = ArrayValue(
        vector=np.array([1.0]), matrix=np.array([[1.0]]), indices=empty
    )

    assert np.array_equal(largest_value.indices, largest)
    assert empty_value.indices.size == 0


def test_array_aliases_reject_wrong_dimension_or_integer_family():
    with pytest.raises(ValidationError, match="expected a 1D array"):
        ArrayValue(
            vector=[[1.0]],
            matrix=[[1.0]],
            indices=[0],
        )

    with pytest.raises(ValidationError, match="expected an integer array"):
        ArrayValue(
            vector=[1.0],
            matrix=[[1.0]],
            indices=[0.5],
        )


def test_array_aliases_dump_as_plain_json_lists():
    value = ArrayValue(
        vector=[1, 2],
        matrix=[[1.0, 2.0]],
        indices=[0, 1],
    )

    assert value.model_dump(mode="json") == {
        "vector": [1.0, 2.0],
        "matrix": [[1.0, 2.0]],
        "indices": [0, 1],
    }


@pytest.mark.parametrize("token", ["-inf", "nan", "inf"])
def test_floating_aliases_do_not_decode_nonfinite_text_tokens(token: str):
    with pytest.raises(ValidationError, match="array of real numbers"):
        ArrayValue(vector=[token], matrix=[[0.0]], indices=[0])
