"""Offline parsing checks for the opt-in reference resolver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_references", Path(__file__).parents[2] / "tools" / "check_references.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check_references = importlib.util.module_from_spec(_SPEC)
sys.modules["check_references"] = check_references
_SPEC.loader.exec_module(check_references)

_TEXT = """
        Iyengar, R. (2019). "Towards Practical Differentially Private Convex
        Optimization." IEEE S&P.
        https://doi.org/10.1109/SP.2019.00001.
        Alabi, D. (2020). "Differentially Private Simple Linear Regression."
        https://arxiv.org/abs/2007.05157v1
        See also https://doi.org/10.48550/arXiv.2401.00583 and arxiv.org/pdf/1607.00133.
"""


def test_parse_references_finds_locators_and_nearby_titles() -> None:
    found = check_references.parse_references(_TEXT, "x.py")
    assert [(r.kind, r.identifier) for r in found] == [
        ("doi", "10.1109/SP.2019.00001"),
        ("arxiv", "2007.05157v1"),
        ("arxiv", "2401.00583"),
        ("arxiv", "1607.00133"),
    ]
    assert found[0].cited_title == (
        "Towards Practical Differentially Private Convex Optimization"
    )
    assert found[1].cited_title == "Differentially Private Simple Linear Regression"


def test_title_similarity_separates_matching_from_unrelated_titles() -> None:
    match = check_references.title_similarity(
        "Towards practical differentially private convex optimization",
        "Towards Practical Differentially Private Convex Optimization",
    )
    other = check_references.title_similarity(
        "Towards Practical Differentially Private Convex Optimization",
        "Postcards from the Post-HTTP World",
    )
    assert match > 0.95 > 0.6 > other


def test_bibliography_entries_yield_their_identifiers_and_titles() -> None:
    text = r"""
@article{a2020x,
  author = {A, B},
  title = {A {SQL} Title},
  journal = {J},
  year = {2020},
  doi = {10.1000/a},
  eprint = {2001.00001},
  archivePrefix = {arXiv}
}
@misc{b2021y,
  author = {B, C},
  title = {No Identifier},
  year = {2021},
  url = {https://example.org/}
}
"""
    found = check_references.collect_bibliography(text, "references.bib")
    assert [(r.kind, r.identifier, r.cited_title, r.location) for r in found] == [
        ("doi", "10.1000/a", "A SQL Title", "references.bib:a2020x"),
        ("arxiv", "2001.00001", "A SQL Title", "references.bib:a2020x"),
    ]
