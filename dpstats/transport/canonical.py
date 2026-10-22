"""Canonical JSON mechanics: exact bytes, transport floats, content identity.

This module is the library's one home for byte determinism. It encodes
JSON-compatible Python values as exact canonical bytes, derives the content
identities every family mints, and decodes persisted JSON back; durable file
writing belongs to ``transport.documents``.

The exact string values ``"+inf_json"``, ``"-inf_json"``, and ``"nan_json"``
are reserved transport tokens for positive infinity, negative infinity, and
NaN. They are not available as authored string values, so a canonical read can
restore their corresponding float without ambiguity.

Filesystem errors propagate naturally and already name their path. A file
whose bytes will not decode into a JSON object is named here, the one failure
that cannot say where it came from.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

_IDENTITY_DIGEST_CHARS = 16
_FLOAT_TRANSPORT_TOKENS = {
    math.inf: "+inf_json",
    -math.inf: "-inf_json",
}
_NAN_TRANSPORT_TOKEN = "nan_json"
_RESERVED_FLOAT_TRANSPORT_TOKENS = frozenset(
    (*_FLOAT_TRANSPORT_TOKENS.values(), _NAN_TRANSPORT_TOKEN)
)


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of already-loaded bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file's exact bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_transport(value: Any) -> Any:
    """Prepare one value recursively for strict-JSON transport.

    Non-finite floats become their reserved token values. An object name that
    standard JSON would coerce to a string is refused so two payloads cannot
    share one encoding, and a string or object name holding a lone surrogate is
    refused, naming its path, because it cannot round-trip as valid UTF-8.
    """
    return _encode_transport(value, "$")


def _require_utf8(text: str, path: str) -> None:
    """Refuse text that is not valid UTF-8, naming where it sits."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{path} contains a lone surrogate and is not valid UTF-8 text."
        ) from error


def _encode_transport(value: Any, path: str) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return _NAN_TRANSPORT_TOKEN
        return _FLOAT_TRANSPORT_TOKENS.get(value, value)
    if isinstance(value, str):
        if value in _RESERVED_FLOAT_TRANSPORT_TOKENS:
            raise ValueError(f"{value!r} is reserved for canonical float transport.")
        _require_utf8(value, path)
        return value
    if isinstance(value, (list, tuple)):
        items: list[Any] = list(value)
        return [
            _encode_transport(item, f"{path}[{index}]")
            for index, item in enumerate(items)
        ]
    if isinstance(value, dict):
        source = cast("dict[Any, Any]", value)
        encoded: dict[str, Any] = {}
        for name, item in source.items():
            if not isinstance(name, str):
                raise ValueError(
                    "Canonical JSON objects need string names; "
                    f"{name!r} would be encoded as a string and collide."
                )
            _require_utf8(name, f"{path} (object name)")
            encoded[name] = _encode_transport(item, f"{path}.{name}")
        return encoded
    return value


def decode_transport(value: Any) -> Any:
    """Restore non-finite floats recursively from strict-JSON token values."""
    if isinstance(value, str):
        if value == _NAN_TRANSPORT_TOKEN:
            return math.nan
        for float_value, token in _FLOAT_TRANSPORT_TOKENS.items():
            if value == token:
                return float_value
        return value
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return [decode_transport(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("dict[str, Any]", value)
        return {name: decode_transport(item) for name, item in mapping.items()}
    return value


def _decode_json(data: bytes) -> Any:
    """Decode standard JSON without silently accepting ambiguous extensions."""

    def reject_constant(name: str) -> Any:
        raise ValueError(f"nonstandard JSON constant {name!r}")

    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate object name {name!r}")
            value[name] = item
        return value

    decoded = json.loads(
        data, parse_constant=reject_constant, object_pairs_hook=build_object
    )
    return decode_transport(decoded)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one JSON-compatible Python value as exact canonical JSON bytes.

    Object names are sorted, separators are compact, and the text is ASCII, so
    one value has exactly one byte representation to name it by.

    Numbers stay type-faithful: the integer ``18`` and the float ``18.0`` are
    distinct values with distinct bytes, so they earn distinct identities. The
    encoder never rewrites a value; a surface whose payloads must not fork on
    authored numeric type normalizes them at its own typed-field validation.
    """
    text = json.dumps(
        encode_transport(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 identity of one canonical JSON value."""
    return sha256_hex(canonical_json_bytes(value))


def compute_identity(payload: Any, *, prefix: str) -> str:
    """Derive the deterministic content-addressed identifier of a payload."""
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=")
    return prefix + encoded[:_IDENTITY_DIGEST_CHARS]


def _family_identity(payload: Mapping[str, Any], *, marker: str) -> str:
    """Derive one identity family's token from its marker and payload.

    The one marker is folded into the hashed payload and mints the visible
    token prefix, so the hashed version and the readable prefix cannot
    disagree: changing a family's derivation bumps both in one place, and a
    stale token becomes unreadable rather than silently wrong.
    """
    if "schema" in payload:
        raise ValueError(
            "An identity payload cannot carry its own 'schema' name: that name "
            "holds the family marker versioning the derivation."
        )
    return compute_identity({"schema": marker, **payload}, prefix=f"{marker}-")


def algorithm_identity(payload: Mapping[str, Any]) -> str:
    """Derive one configured algorithm's content identity."""
    return _family_identity(payload, marker="a3")


def instance_identity(payload: Mapping[str, Any]) -> str:
    """Derive one direct Instance's content identity."""
    return _family_identity(payload, marker="i5")


def coordinate_identity(payload: Mapping[str, Any]) -> str:
    """Derive one Sweep coordinate's content identity."""
    return _family_identity(payload, marker="c3")


def parse_json_object(value: Any) -> dict[str, Any]:
    """Narrow one decoded JSON value to an object; its names are always strings."""
    if not isinstance(value, dict):
        raise ValueError("A JSON object must be a mapping of string names to values.")
    return cast(dict[str, Any], value)


def decode_json_document(source: bytes | Path) -> dict[str, Any]:
    """Decode one standard JSON document holding an object."""
    data = source.read_bytes() if isinstance(source, Path) else source
    try:
        return parse_json_object(_decode_json(data))
    except ValueError as error:
        if isinstance(source, Path):
            raise ValueError(f"{source} does not hold a JSON object.") from error
        raise


def decode_canonical_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    """Decode one JSON-lines payload whose every row must be canonical bytes.

    This establishes that each line is exactly one canonically encoded JSON
    object; which model a row satisfies is settled where the rows are
    consumed.
    """
    records: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = decode_json_document(line)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"{label} has invalid JSON at line {number}.") from error
        if line != canonical_json_bytes(value) + b"\n":
            raise ValueError(f"{label} has a noncanonical row at line {number}.")
        records.append(value)
    return records
