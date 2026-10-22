"""Property coverage for the canonical-JSON transport contracts.

Laws checked over generated strict-JSON values: encoding is byte-stable across
an encode-decode-encode cycle, and exactly the three reserved transport
spellings are refused as authored strings.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, strategies as st

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from tests import hypothesis_profile  # pyright: ignore[reportUnusedImport]

_RESERVED_TOKENS = ("+inf_json", "-inf_json", "nan_json")

_json_leaves = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True, allow_infinity=True)
    # Reserved transport spellings are documented-refused authored strings,
    # so the round-trip domain excludes them; their refusal has its own law.
    | st.text().filter(lambda text: text not in _RESERVED_TOKENS)
)
_json_values = st.recursive(
    _json_leaves,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=20,
)


@given(value=st.dictionaries(st.text(), _json_values, max_size=4))
def test_encoding_is_byte_stable_across_a_decode_cycle(value: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(value)

    assert canonical_json_bytes(decode_json_document(encoded)) == encoded


@given(authored=st.text())
def test_only_the_exact_reserved_spellings_are_refused(authored: str) -> None:
    if authored in _RESERVED_TOKENS:
        with pytest.raises(ValueError, match="reserved"):
            canonical_json_bytes({"key": authored})
        return

    restored = decode_json_document(canonical_json_bytes({"key": authored}))
    assert restored == {"key": authored}
