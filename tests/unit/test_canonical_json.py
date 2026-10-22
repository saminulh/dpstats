"""Public canonical-record transport contracts."""

from __future__ import annotations

from typing import Any

import json
import math
import re
from pathlib import Path

import numpy as np
import pydantic_core
import pytest

from dpstats.transport.canonical import (
    algorithm_identity,
    canonical_json_bytes,
    compute_identity,
    coordinate_identity,
    decode_json_document,
    instance_identity,
    parse_json_object,
    sha256_hex,
)
from dpstats.transport.validators import is_sha256_hex

MESSAGE = "Repair the document before re-running"


def test_sha256_helpers_share_one_exact_lowercase_digest_contract() -> None:
    digest = sha256_hex(b"records")

    assert digest == "a94e7bcfcbed3c846d491d4f47c948e53a23908d480248b3ffe9e126e83ea865"
    assert is_sha256_hex(digest)
    assert not is_sha256_hex(digest.upper())


def test_nonfinite_float_tokens_are_exact_and_recursive() -> None:
    payload = {
        "finite": 1.25,
        "nested": [math.inf, {"negative": -math.inf, "nan": math.nan}],
    }

    assert canonical_json_bytes(payload) == (
        b'{"finite":1.25,"nested":["+inf_json",'
        b'{"nan":"nan_json","negative":"-inf_json"}]}'
    )


def test_reserved_float_tokens_are_not_authored_strings() -> None:
    for reserved in ("+inf_json", "-inf_json", "nan_json"):
        with pytest.raises(ValueError, match="reserved"):
            canonical_json_bytes({"value": [reserved]})

    assert canonical_json_bytes({"value": "+inf_jsonish"}) == (
        b'{"value":"+inf_jsonish"}'
    )


def test_a_lone_surrogate_is_refused_naming_its_field() -> None:
    """Text that is not valid UTF-8 is refused where it is written."""
    with pytest.raises(ValueError, match=re.escape("$.payload")):
        canonical_json_bytes({"payload": "bad \ud800 text"})


def test_a_nested_lone_surrogate_is_refused_naming_its_path() -> None:
    """The refusal names the path through nested containers."""
    with pytest.raises(ValueError, match=re.escape("$.nested[0].message")):
        canonical_json_bytes({"nested": [{"message": "bad \udfff text"}]})


def test_a_lone_surrogate_in_an_object_name_is_refused() -> None:
    """An object name is text too, so it is held to the same rule."""
    with pytest.raises(ValueError, match="lone surrogate"):
        canonical_json_bytes({"bad \ud800 name": 1})


def test_non_ascii_text_round_trips_through_a_strict_reader() -> None:
    """A non-BMP character is valid text, so a strict UTF-8 reader accepts it."""
    payload = {"label": "caf\u00e9 \U0001f600"}

    encoded = canonical_json_bytes(payload)

    assert decode_json_document(encoded) == payload
    assert pydantic_core.from_json(encoded) == payload


class TestIdentity:
    PAYLOAD = {
        "schema": 1,
        "task": "mean_estimation",
        "algorithm": {"path": "m:C", "hyperparams": {"a": 1}},
    }
    CANONICAL = (
        b'{"algorithm":{"hyperparams":{"a":1},"path":"m:C"},'
        b'"schema":1,"task":"mean_estimation"}'
    )
    IDENTIFIER = "x1-2J65FQDXKYBFTVGS"

    def test_exact_canonical_bytes_and_identifier(self):
        assert canonical_json_bytes(self.PAYLOAD) == self.CANONICAL
        assert compute_identity(self.PAYLOAD, prefix="x1-") == self.IDENTIFIER

    def test_identity_ignores_key_order_and_distinguishes_payloads(self):
        reordered = {
            "task": "mean_estimation",
            "algorithm": {"hyperparams": {"a": 1}, "path": "m:C"},
            "schema": 1,
        }
        assert compute_identity(reordered, prefix="x1-") == self.IDENTIFIER
        assert (
            compute_identity({**self.PAYLOAD, "task": "median"}, prefix="x1-")
            != self.IDENTIFIER
        )

    def test_nonfinite_values_receive_distinct_stable_identities(self):
        finite = compute_identity({"schema": 1, "x": 0.0}, prefix="x1-")
        positive_infinity = compute_identity({"schema": 1, "x": math.inf}, prefix="x1-")
        negative_infinity = compute_identity(
            {"schema": 1, "x": -math.inf}, prefix="x1-"
        )
        nan = compute_identity({"schema": 1, "x": math.nan}, prefix="x1-")

        assert len({finite, positive_infinity, negative_infinity, nan}) == 4
        assert compute_identity({"schema": 1, "x": math.nan}, prefix="x1-") == nan

    def test_non_json_native_values_are_refused(self):
        with pytest.raises(TypeError):
            compute_identity({"schema": 1, "x": np.int64(1)}, prefix="x1-")
        with pytest.raises(TypeError):
            compute_identity({"schema": 1, "x": np.arange(3)}, prefix="x1-")
        with pytest.raises(TypeError):
            compute_identity({"schema": 1, "x": {1, 2}}, prefix="x1-")

    def test_tuples_use_their_standard_json_array_shape(self):
        assert canonical_json_bytes({"schema": 1, "x": (1, 2)}) == (
            b'{"schema":1,"x":[1,2]}'
        )
        assert compute_identity(
            {"schema": 1, "x": (1, 2)}, prefix="x1-"
        ) == compute_identity({"schema": 1, "x": [1, 2]}, prefix="x1-")

    @pytest.mark.parametrize("name", [1, None, True, 2.5])
    def test_non_string_object_names_are_refused(self, name: Any):
        """Standard JSON would coerce these to strings and collide identities."""
        with pytest.raises(ValueError, match="string names"):
            compute_identity({name: "a"}, prefix="x1-")

    def test_nested_non_string_object_names_are_refused(self):
        with pytest.raises(ValueError, match="string names"):
            canonical_json_bytes({"schema": 1, "nested": [{2: "b"}]})


class TestFamilyIdentity:
    PAYLOAD = {"task": "mean_estimation", "value": 1}

    def test_one_payload_earns_a_distinct_token_per_family(self):
        tokens = {
            "a3-": algorithm_identity(self.PAYLOAD),
            "i5-": instance_identity(self.PAYLOAD),
            "c3-": coordinate_identity(self.PAYLOAD),
        }
        for prefix, token in tokens.items():
            assert token.startswith(prefix)
        digests = {token.partition("-")[2] for token in tokens.values()}
        assert len(digests) == 3

    @pytest.mark.parametrize(
        "derive", [algorithm_identity, instance_identity, coordinate_identity]
    )
    def test_a_payload_carrying_its_own_schema_name_is_refused(
        self, derive: Any
    ) -> None:
        """The marker versions the derivation; a payload key must not displace it."""
        with pytest.raises(ValueError, match="family marker"):
            derive({"schema": 9, **self.PAYLOAD})


class TestNarrowing:
    def test_object_values_pass_through_unchanged(self):
        value = {"a": 1, "b": [2]}
        assert parse_json_object(value) is value

    @pytest.mark.parametrize("value", [[], "text", 1, None])
    def test_non_objects_are_refused(self, value: Any):
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_json_object(value)


class TestDecoding:
    def test_bytes_and_paths_decode_to_the_same_value(self, tmp_path: Path):
        path = tmp_path / "document.json"
        path.write_bytes(b'{"a": [1, 2]}')

        assert decode_json_document(path) == {"a": [1, 2]}
        assert decode_json_document(b'{"a": [1, 2]}') == {"a": [1, 2]}

    def test_native_read_and_parse_errors_propagate(self, tmp_path: Path):
        with pytest.raises(OSError):
            decode_json_document(tmp_path / "absent.json")
        with pytest.raises(json.JSONDecodeError):
            decode_json_document(b"{not json")
        with pytest.raises(json.JSONDecodeError):
            decode_json_document(b"\xff\xfe")

    def test_a_document_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            decode_json_document(b"[1, 2]")
        with pytest.raises(ValueError, match="must be a mapping"):
            decode_json_document(b"3")

    def test_a_file_that_will_not_decode_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "document.json"
        path.write_bytes(b"{not json")

        with pytest.raises(ValueError, match=str(path)):
            decode_json_document(path)

    def test_a_file_holding_something_other_than_an_object_is_named(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "document.json"
        path.write_bytes(b"[1, 2]")

        with pytest.raises(ValueError, match=str(path)) as refusal:
            decode_json_document(path)

        assert "must be a mapping" in str(refusal.value.__cause__)

    @pytest.mark.parametrize(
        "payload",
        [b"NaN", b"Infinity", b"-Infinity", b'{"x":NaN}', b"[Infinity]"],
    )
    def test_nonstandard_constants_are_refused(self, payload: bytes):
        with pytest.raises(ValueError, match="nonstandard"):
            decode_json_document(payload)

    @pytest.mark.parametrize(
        "payload",
        [b'{"x":1,"x":2}', b'{"outer":{"x":1,"x":2}}'],
    )
    def test_duplicate_object_names_are_refused(self, payload: bytes):
        with pytest.raises(ValueError, match="duplicate"):
            decode_json_document(payload)

    def test_float_transport_tokens_are_restored_recursively(self):
        decoded = decode_json_document(
            b'{"positive":"+inf_json","negative":"-inf_json",'
            b'"nested":["nan_json","nan_jsonish"]}'
        )

        assert isinstance(decoded, dict)
        assert decoded["positive"] == math.inf
        assert decoded["negative"] == -math.inf
        assert math.isnan(decoded["nested"][0])
        assert decoded["nested"][1] == "nan_jsonish"
