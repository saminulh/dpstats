"""The grammar of persisted values: names, paths, revisions, and digests.

Each family's spelling has one authority here, consumed in the two shapes its
consumers need: a predicate for boolean composition, judging spelling and
never refusing, and an ``Annotated`` field type for models, where pydantic
owns the refusal and names the offending field. Each artifact owner remains
responsible for safely observing what a well-spelled reference points at.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, TypeGuard

from pydantic import AfterValidator


_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")


def is_sha256_hex(value: Any) -> TypeGuard[str]:
    """Return whether ``value`` is one lowercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_portable_name(value: Any) -> TypeGuard[str]:
    """Return whether *value* is one direct portable path entry."""
    return (
        isinstance(value, str)
        and value != ""
        and value.strip() == value
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def is_normalized_relative_path(value: Any) -> TypeGuard[str]:
    """Return whether *value* is one exactly normalized relative POSIX path."""
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute() and value == path.as_posix() and ".." not in path.parts
    )


def is_immutable_revision(value: Any) -> TypeGuard[str]:
    """Return whether *value* is one full lowercase Git revision."""
    return isinstance(value, str) and _IMMUTABLE_REVISION.fullmatch(value) is not None


def _sha256_hex(value: str) -> str:
    if not is_sha256_hex(value):
        raise ValueError("must be lowercase SHA-256 hex.")
    return value


Sha256Hex = Annotated[str, AfterValidator(_sha256_hex)]
