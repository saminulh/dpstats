"""Which rows of a table are one row, judged column by column without Python rows.

Two rows are one row when every column holds the same canonical value: floats
compare by bits once every NaN is one NaN, so ``-0.0`` and ``0.0`` stay apart;
a null, an empty list, and a list of nulls are three values. That is the
judgement canonical JSON makes of a row's values, computed on whole columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

_CANONICAL_NAN = np.uint64(0x7FF8000000000000)
_HASH_KEYS = ("dp-row-identity1", "dp-row-identity2")
_NULL_HASHES = (np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707344))
_MIX_SEEDS = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F))
_WORD_BYTES = 8

_Segments = tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]


def _is_fixed(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
    )


def _is_text(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
    )


def _is_list(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _validity(array: pa.Array[Any]) -> np.ndarray[Any, Any]:
    return np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)


def _fixed_words(array: pa.Array[Any]) -> np.ndarray[Any, Any]:
    """One canonical 64-bit word per value of a boolean, integer or float array.

    A float becomes its bits with every NaN one NaN; a null becomes zero, which
    the caller tells apart through validity.
    """
    data_type = array.type
    filled: pa.Array[Any]
    if pa.types.is_floating(data_type):
        floats = array.cast(pa.float64()).fill_null(0.0)
        values = np.asarray(floats.to_numpy(zero_copy_only=False), dtype=np.float64)
        words = values.view(np.uint64).copy()
        words[np.isnan(values)] = _CANONICAL_NAN
        return words
    if pa.types.is_boolean(data_type):
        filled = array.fill_null(False).cast(pa.uint64())
    elif pa.types.is_signed_integer(data_type):
        filled = array.cast(pa.int64()).fill_null(0)
    else:
        filled = array.cast(pa.uint64()).fill_null(0)
    return np.asarray(filled.to_numpy(zero_copy_only=False)).view(np.uint64)


def _copy_segments(
    out: np.ndarray[Any, Any],
    destinations: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    sources: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
) -> None:
    """Copy each byte run ``data[source:source+length]`` to its destination."""
    total = int(lengths.sum())
    if total == 0:
        return
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    out[np.repeat(destinations, lengths) + within] = data[
        np.repeat(sources, lengths) + within
    ]


def _interleave(
    parts: list[
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ],
    count: int,
) -> _Segments:
    """Join element *i* of every part, in part order, into one byte run each.

    Each part is ``(data, starts, lengths)`` naming one byte run per element.
    """
    lengths = np.zeros(count, dtype=np.int64)
    for _, _, part_lengths in parts:
        lengths += part_lengths
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    out = np.empty(int(offsets[-1]), dtype=np.uint8)
    cursor = offsets[:-1].copy()
    for data, starts, part_lengths in parts:
        _copy_segments(out, cursor, data, starts, part_lengths)
        cursor += part_lengths
    return out, offsets


def _framed(
    valid: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    offsets: np.ndarray[Any, Any],
) -> _Segments:
    """Prefix each element's payload with its validity and length.

    A null element keeps neither payload nor length, so every encoding reads
    back unambiguously when encodings are concatenated.
    """
    count = len(valid)
    lengths = np.where(valid, np.diff(offsets), 0).astype(np.int64)
    header = np.zeros((count, 1 + _WORD_BYTES), dtype=np.uint8)
    header[:, 0] = valid
    header[:, 1:] = lengths.astype("<u8").view(np.uint8).reshape(count, _WORD_BYTES)
    width = 1 + _WORD_BYTES
    header_starts = np.arange(count, dtype=np.int64) * width
    return _interleave(
        [
            (header.ravel(), header_starts, np.full(count, width, dtype=np.int64)),
            (data, offsets[:-1].astype(np.int64), lengths),
        ],
        count,
    )


def _text_payload(array: pa.Array[Any]) -> _Segments:
    """The raw bytes and per-element offsets of a string or binary array."""
    large = pa.types.is_large_string(array.type) or pa.types.is_large_binary(array.type)
    buffers = array.buffers()
    offset_buffer = buffers[1]
    if offset_buffer is None:
        raise ValueError("A string or binary array carries no offsets buffer.")
    offset_type = np.int64 if large else np.int32
    offsets: np.ndarray[Any, Any] = np.frombuffer(offset_buffer, dtype=offset_type)[
        array.offset : array.offset + len(array) + 1
    ].astype(np.int64)
    data = (
        np.empty(0, dtype=np.uint8)
        if buffers[2] is None
        else np.frombuffer(buffers[2], dtype=np.uint8)
    )
    return data, offsets


def _list_payload(array: pa.Array[Any], column: str) -> _Segments:
    """Each list's elements' encodings, as one byte run per list."""
    if pa.types.is_fixed_size_list(array.type):
        array = array.cast(pa.list_(array.type.value_type))
    lists = cast("pa.ListArray[Any] | pa.LargeListArray[Any]", array)
    child_data, child_offsets = _element_bytes(lists.values, column)
    positions: np.ndarray[Any, Any] = np.asarray(
        lists.offsets.to_numpy(zero_copy_only=False), np.int64
    )
    return child_data, child_offsets[positions]


def _element_bytes(array: pa.Array[Any], column: str) -> _Segments:
    """Encode each element as self-delimiting bytes carrying its validity."""
    data_type = array.type
    valid = _validity(array)
    count = len(array)
    if _is_fixed(data_type):
        width = 1 + _WORD_BYTES
        words = np.zeros((count, width), dtype=np.uint8)
        words[:, 0] = valid
        words[:, 1:] = (
            _fixed_words(array).astype("<u8").view(np.uint8).reshape(count, _WORD_BYTES)
        )
        return words.ravel(), np.arange(count + 1, dtype=np.int64) * width
    if _is_text(data_type):
        return _framed(valid, *_text_payload(array))
    if _is_list(data_type):
        return _framed(valid, *_list_payload(array, column))
    if pa.types.is_struct(data_type):
        structs = cast("pa.StructArray", array)
        children = cast("list[pa.Array[Any]]", structs.flatten())
        fields = [_element_bytes(child, column) for child in children]
        joined, joined_offsets = _interleave(
            [(data, offsets[:-1], np.diff(offsets)) for data, offsets in fields],
            count,
        )
        return _framed(valid, joined, joined_offsets)
    raise TypeError(
        f"Column {column!r} holds {data_type}, whose values row identity does not "
        "judge; only booleans, numbers, text, lists and structs are judged."
    )


def _exact_key(array: pa.Array[Any], column: str) -> pa.Array[Any]:
    """An array whose dictionary encoding is exactly canonical value equality."""
    data_type = array.type
    if pa.types.is_floating(data_type):
        words = _fixed_words(array)
        return pa.array(words, pa.uint64(), mask=~_validity(array))
    if _is_fixed(data_type) or _is_text(data_type):
        return array
    if _is_list(data_type):
        # Arrow lets offsets start past zero, so the lists' payload runs index
        # the elements' encoding in place.
        data, offsets = _list_payload(array, column)
        return pa.LargeBinaryArray.from_buffers(
            pa.large_binary(),
            len(array),
            [
                array.is_valid().buffers()[1] if array.null_count else None,
                pa.py_buffer(np.ascontiguousarray(offsets)),
                pa.py_buffer(data),
            ],
            array.null_count,
        )
    data, offsets = _element_bytes(array, column)
    return pa.LargeBinaryArray.from_buffers(
        pa.large_binary(), len(array), [None, pa.py_buffer(offsets), pa.py_buffer(data)]
    )


@dataclass(frozen=True)
class _Column:
    """One column's exact per-row codes and the distinct values they index."""

    codes: np.ndarray[Any, Any]
    dictionary: pa.Array[Any]


def _encode(column: pa.ChunkedArray[Any], name: str) -> _Column:
    chunks = [chunk for chunk in column.chunks if len(chunk)]
    if not chunks:
        return _Column(np.zeros(0, dtype=np.int64), pa.array([], pa.null()))
    keyed = pa.chunked_array([_exact_key(chunk, name) for chunk in chunks])
    encoded = cast(
        "list[pa.DictionaryArray[Any, Any]]",
        keyed.dictionary_encode(null_encoding="encode").chunks,
    )
    codes = np.concatenate(
        [
            np.asarray(chunk.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
            for chunk in encoded
        ]
    )
    return _Column(codes, encoded[0].dictionary)


@dataclass(frozen=True)
class _RowSet:
    """A table's columns encoded, and the first index of each distinct row."""

    columns: tuple[_Column, ...]
    first: np.ndarray[Any, Any]


def _row_set(rows: pa.Table) -> _RowSet:
    columns = tuple(
        _encode(rows.column(index), name)
        for index, name in enumerate(rows.column_names)
    )
    if rows.num_rows == 0 or not columns:
        return _RowSet(columns, np.arange(min(rows.num_rows, 1), dtype=np.int64))
    stacked = np.ascontiguousarray(
        np.stack([column.codes for column in columns], axis=1)
    )
    width = 8 * len(columns)
    keys = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width), rows.num_rows, [None, pa.py_buffer(stacked)]
    )
    row_codes = np.asarray(
        keys.dictionary_encode().indices.to_numpy(zero_copy_only=False), np.int64
    )
    # Codes are minted in order of first appearance, so the first index of
    # each code, read in code order, is already ascending.
    _, first = np.unique(row_codes, return_index=True)
    return _RowSet(columns, first.astype(np.int64))


def distinct_rows(rows: pa.Table) -> pa.Table:
    """Return each of the table's rows once, judged on every column it holds.

    The first copy of each row is kept, in order; a table with no repeats is
    returned as it is.

    Raises:
        TypeError: naming a column whose type row identity does not judge.
    """
    first = _row_set(rows).first
    if len(first) == rows.num_rows:
        return rows
    return rows.take(pa.array(first, pa.int64()))


def _mix(words: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The SplitMix64 finalizer, elementwise and wrapping."""
    with np.errstate(over="ignore"):
        words = (words ^ (words >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        words = (words ^ (words >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return words ^ (words >> np.uint64(31))


def _dictionary_hashes(dictionary: pa.Array[Any], lane: int) -> np.ndarray[Any, Any]:
    """Hash each distinct value by its content, never by its code."""
    valid = _validity(dictionary)
    if _is_text(dictionary.type):
        values = np.asarray(dictionary.to_numpy(zero_copy_only=False), dtype=object)
        values[~valid] = b""
        hashed = np.asarray(
            pd.util.hash_array(values, hash_key=_HASH_KEYS[lane], categorize=False),
            dtype=np.uint64,
        )
    elif pa.types.is_null(dictionary.type):
        hashed = np.zeros(len(dictionary), dtype=np.uint64)
    else:
        with np.errstate(over="ignore"):
            hashed = _mix(_fixed_words(dictionary) + _MIX_SEEDS[lane])
    hashed[~valid] = _NULL_HASHES[lane]
    return hashed


def distinct_row_census(rows: pa.Table) -> tuple[int, str]:
    """Count the table's distinct rows and name their set by its content.

    Two tables holding the same distinct rows under the same columns share a
    digest whatever their order or repeats; a changed value, a lost row, or an
    added one changes it.

    Returns:
        The number of distinct rows and a hex digest of their set.

    Raises:
        TypeError: naming a column whose type row identity does not judge.
    """
    row_set = _row_set(rows)
    lanes: list[np.ndarray[Any, Any]] = []
    for lane in range(2):
        hashed = np.full(len(row_set.first), _MIX_SEEDS[lane], dtype=np.uint64)
        for column in row_set.columns:
            values = _dictionary_hashes(column.dictionary, lane)[
                column.codes[row_set.first]
            ]
            with np.errstate(over="ignore"):
                hashed = _mix(hashed * _MIX_SEEDS[lane] + values)
        lanes.append(hashed)
    order = np.lexsort((lanes[1], lanes[0]))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [rows.column_names, [str(t) for t in rows.schema.types], len(order)]
        ).encode()
    )
    digest.update(np.stack([lanes[0][order], lanes[1][order]], axis=1).tobytes())
    return len(order), digest.hexdigest()
