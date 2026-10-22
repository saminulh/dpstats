"""The API reference derives its pages and configured-field helpers from the source."""

from __future__ import annotations

from pathlib import Path

import pytest

from api_reference import configured_base, format_search, reference_pages
from dpstats.algorithm import BaseAlgorithm, Transformation
from dpstats.parameters import search_range, search_values
from dpstats.tasks.mean_estimation.transformations import DataBounds
from tests.unit.docs.conftest import ReferenceMean

REFERENCE = Path(__file__).resolve().parents[3] / "docs" / "reference"


def test_public_api_objects_appear_in_their_category_navigation():
    pages, routes = reference_pages(
        {"domains": REFERENCE / "domains.md"},
    )
    domain = "dpstats.domains.Domain"
    assert routes[domain] in pages
    assert "dpstats.domains" in pages["library/reference/domains/index.md"]
    assert (
        "- [Domain](dpstats.domains.Domain.md)"
        in pages["library/reference/domains/SUMMARY.md"]
    )


def test_split_categories_give_each_shared_module_member_one_home():
    groups = (
        "collections",
        "sweep",
        "algorithm",
        "tasks",
        "instance",
        "metric",
        "analysis",
        "parameters",
    )
    pages, routes = reference_pages(
        {group: REFERENCE / f"{group}.md" for group in groups}
    )
    for identifier, group in (
        ("dpstats.TaskCollection", "collections"),
        ("dpstats.open_collection", "collections"),
        ("dpstats.algorithm.Transformation", "algorithm"),
        ("dpstats.algorithm.BaseAlgorithm", "algorithm"),
        ("dpstats.algorithm.CompositeAlgorithm", "algorithm"),
        ("dpstats.instance.Instance", "instance"),
        ("dpstats.metric.Metric", "metric"),
    ):
        expected = f"library/reference/{group}/{identifier}.md"
        assert routes[identifier] == expected
        assert [path for path in pages if path.endswith(f"/{identifier}.md")] == [
            expected
        ]
    assert "heading: Collections" in pages["library/reference/collections/index.md"]
    assert "TaskCollection.run_sweep" in pages["library/reference/sweep/index.md"]
    assert "Transformation user guide" in pages["library/reference/algorithm/index.md"]


@pytest.mark.parametrize(
    ("model", "base"),
    [(ReferenceMean, BaseAlgorithm), (DataBounds, Transformation), (object, None)],
)
def test_the_excluded_base_is_the_platform_family_the_class_belongs_to(
    model: type, base: type | None
) -> None:
    assert configured_base(model) is base


@pytest.mark.parametrize(
    ("search", "text"),
    [
        (None, ""),
        (search_range(0.1, 10.0, scale="log"), "0.1 – 10.0 (log scale)"),
        (search_values(1, 2, 3), "1, 2, 3"),
    ],
)
def test_a_declared_search_renders_as_short_text(search: object, text: str) -> None:
    assert format_search(search) == text  # type: ignore[arg-type]
