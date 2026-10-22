"""Columnar row distinctness agrees with the canonical-JSON judgement it replaced."""

from __future__ import annotations

import math
import struct
from typing import Any

import pyarrow as pa
import pytest
from hypothesis import given, settings, strategies as st

from dpstats.results._row_identity import (
    distinct_rows,
    distinct_row_census,
)
from dpstats.transport.canonical import canonical_json_bytes


def row_set_digest(rows: pa.Table) -> str:
    return distinct_row_census(rows)[1]


def _referencedistinct_rows(rows: pa.Table) -> pa.Table:
    """The per-row canonical-JSON judgement compaction used before it went columnar."""
    seen: set[bytes] = set()
    keep: list[int] = []
    for index, row in enumerate(rows.to_pylist()):
        encoded = canonical_json_bytes(row)
        if encoded not in seen:
            seen.add(encoded)
            keep.append(index)
    return rows.take(pa.array(keep, pa.int64()))


def _nan(payload: int) -> float:
    """A quiet NaN carrying *payload* in its mantissa."""
    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("seed", pa.uint64(), nullable=False),
        pa.field("attempt", pa.int32(), nullable=False),
        pa.field("runtime_s", pa.float64()),
        pa.field("reason", pa.string()),
        pa.field("flag", pa.bool_()),
        pa.field("beta", pa.list_(pa.float64())),
        pa.field("grid", pa.list_(pa.list_(pa.int64()))),
        pa.field("pair", pa.struct([("a", pa.float64()), ("b", pa.string())])),
    ]
)

_FLOATS = st.sampled_from(
    [0.0, -0.0, 1.0, 1.5, -2.25, math.inf, -math.inf, _nan(0), _nan(7), 1e-300, None]
)
_TEXT = st.one_of(
    st.none(),
    st.sampled_from(["", "a", "b", "ab", '{"k": 1}', '{"k":1}', "é"]),
    st.text(st.characters(blacklist_categories=("Cs",)), max_size=4).filter(
        lambda text: not text.endswith("_json")
    ),
)
_ROW = st.fixed_dictionaries(
    {
        "instance_id": st.sampled_from(["i-a", "i-b"]),
        "seed": st.sampled_from([0, 1, 2**64 - 1]),
        "attempt": st.sampled_from([0, 1, -1]),
        "runtime_s": _FLOATS,
        "reason": _TEXT,
        "flag": st.sampled_from([True, False, None]),
        "beta": st.one_of(st.none(), st.lists(_FLOATS, max_size=3)),
        "grid": st.one_of(
            st.none(),
            st.lists(
                st.one_of(
                    st.none(), st.lists(st.sampled_from([0, 1, None]), max_size=2)
                ),
                max_size=2,
            ),
        ),
        "pair": st.one_of(
            st.none(),
            st.fixed_dictionaries({"a": _FLOATS, "b": _TEXT}),
        ),
    }
)


def _assert_same_rows(actual: pa.Table, expected: pa.Table) -> None:
    """Compare canonically, so NaN rows compare equal to themselves."""
    assert [canonical_json_bytes(row) for row in actual.to_pylist()] == [
        canonical_json_bytes(row) for row in expected.to_pylist()
    ]


# Values that a looser judgement would confuse with one another.
_CONFUSABLE: dict[str, list[Any]] = {
    "runtime_s": [0.0, -0.0, _nan(0), _nan(7), None, math.inf],
    "reason": [None, "", "a", '{"k": 1}', '{"k":1}'],
    "flag": [True, False, None],
    "beta": [None, [], [None], [0.0], [-0.0], [_nan(1)], [_nan(2)], [0.0, None]],
    "grid": [None, [], [None], [[]], [[None]], [[0]], [[0], []], [[], [0]]],
    "pair": [None, {"a": None, "b": None}, {"a": 0.0, "b": ""}, {"a": -0.0, "b": ""}],
}


@settings(max_examples=300, deadline=None)
@given(
    base=_ROW,
    edits=st.lists(
        st.lists(
            st.tuples(st.sampled_from(sorted(_CONFUSABLE)), st.integers(0, 7)),
            max_size=2,
        ),
        min_size=1,
        max_size=30,
    ),
    chunk=st.integers(min_value=1, max_value=7),
)
def test_columnar_distinct_rows_match_the_canonical_json_judgement(
    base: dict[str, Any], edits: list[list[tuple[str, int]]], chunk: int
) -> None:
    rows: list[dict[str, Any]] = []
    for row_edits in edits:
        row = dict(base)
        for field, choice in row_edits:
            options = _CONFUSABLE[field]
            row[field] = options[choice % len(options)]
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    # Several files concatenated arrive as a chunked table.
    chunked = pa.concat_tables(
        [table.slice(start, chunk) for start in range(0, len(rows), chunk)]
    )
    actual = distinct_rows(chunked)
    _assert_same_rows(actual, _referencedistinct_rows(table))


def test_nan_payloads_are_one_value_and_signed_zeros_are_two() -> None:
    table = pa.table(
        {
            "x": pa.array([_nan(1), _nan(2), 0.0, -0.0, None, math.nan]),
            "xs": pa.array([[_nan(3)], [_nan(4)], [0.0], [-0.0], None, []]),
        }
    )
    assert distinct_rows(table).num_rows == 5
    assert _referencedistinct_rows(table).num_rows == 5


def test_null_empty_and_absent_elements_are_distinct() -> None:
    table = pa.table(
        {
            "xs": pa.array(
                [None, [], [None], [None, None], [], None], pa.list_(pa.int64())
            )
        }
    )
    distinct = distinct_rows(table)
    assert distinct.column("xs").to_pylist() == [None, [], [None], [None, None]]


def test_the_first_copy_of_each_row_is_kept_in_order() -> None:
    table = pa.table({"k": ["b", "a", "b", "c", "a"], "n": [1, 2, 1, 3, 2]})
    assert distinct_rows(table).to_pylist() == [
        {"k": "b", "n": 1},
        {"k": "a", "n": 2},
        {"k": "c", "n": 3},
    ]


def test_a_table_with_no_repeats_is_returned_whole() -> None:
    table = pa.table({"k": ["a", "b"]})
    assert distinct_rows(table) is table


def test_an_empty_table_has_no_rows_to_repeat() -> None:
    table = _SCHEMA.empty_table()
    assert distinct_rows(table).num_rows == 0
    assert row_set_digest(table) == row_set_digest(_SCHEMA.empty_table())


def test_an_unsupported_column_type_is_refused_by_name() -> None:
    table = pa.table({"when": pa.array([0], pa.timestamp("s"))})
    with pytest.raises(TypeError, match="when"):
        distinct_rows(table)


@settings(max_examples=150, deadline=None)
@given(
    pool=st.lists(_ROW, min_size=1, max_size=6),
    picks=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=20),
    shuffled=st.randoms(use_true_random=False),
)
def test_the_row_set_digest_names_the_distinct_set_whatever_its_order(
    pool: list[dict[str, Any]], picks: list[int], shuffled: Any
) -> None:
    rows = [pool[pick % len(pool)] for pick in picks]
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    distinct = distinct_rows(table)
    reordered = distinct.take(
        pa.array(shuffled.sample(range(distinct.num_rows), distinct.num_rows))
    )
    assert row_set_digest(table) == row_set_digest(reordered)
    assert distinct_row_census(table)[0] == distinct.num_rows
    if distinct.num_rows > 1:
        assert row_set_digest(table) != row_set_digest(distinct.slice(1))


def test_the_row_set_digest_tells_a_changed_value_apart() -> None:
    base = pa.table({"x": pa.array([1.0, 2.0]), "k": ["a", "b"]})
    assert row_set_digest(base) != row_set_digest(
        pa.table({"x": pa.array([1.0, -0.0]), "k": ["a", "b"]})
    )
    assert row_set_digest(base) != row_set_digest(
        pa.table({"x": pa.array([1.0, 2.0]), "k": ["a", "c"]})
    )
    # Values swapped between columns are a different row set.
    assert row_set_digest(pa.table({"a": ["x"], "b": ["y"]})) != row_set_digest(
        pa.table({"a": ["y"], "b": ["x"]})
    )
