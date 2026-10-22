"""Read the maintainer-owned definitions of the terms a chart page uses."""

from __future__ import annotations

from pathlib import Path

import yaml

# The terms a chart reading, a results control label or a task table's class
# filter defines.
READER_TERMS = (
    "instance",
    "trial",
    "metric",
    "baseline",
    "tau",
    "performance_ratio",
    "success_rate",
    "privacy_interpretation",
)
READER_TERMS_PATH = Path(__file__).resolve().parents[1] / "content" / "reader-terms.yml"


def load_reader_terms(path: Path = READER_TERMS_PATH) -> dict[str, dict[str, str]]:
    """The term definitions in *path*, each ``{"text": ..., "link"?: ...}``.

    An absent file defines nothing, and a term the file omits renders as plain
    text. A file that exists but is malformed fails the build.
    """
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping of term to definition")
    terms: dict[str, dict[str, str]] = {}
    for term, entry in loaded.items():
        if term not in READER_TERMS:
            raise ValueError(f"{path}: term {term!r} is not one of {READER_TERMS}")
        text = entry.get("text") if isinstance(entry, dict) else None
        link = entry.get("link") if isinstance(entry, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: term {term!r} needs a non-empty text")
        if link is not None and not isinstance(link, str):
            raise ValueError(f"{path}: term {term!r} link must be a string")
        terms[str(term)] = {"text": text.strip()} | ({"link": link} if link else {})
    return terms
