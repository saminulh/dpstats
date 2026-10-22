"""Links into the site become relative, so a build works on any host."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from site_links import PUBLIC_SITE_URL, relative_link

PREVIEW = "https://preview.example.org/"
ROOTS = (PUBLIC_SITE_URL, PREVIEW)


@pytest.mark.parametrize(
    ("url", "page", "expected"),
    [
        (
            PUBLIC_SITE_URL + "linear_regression/transformations/Standardize/",
            "tasks/linear_regression/problem-formulation/",
            "../../../linear_regression/transformations/Standardize/",
        ),
        (
            PREVIEW + "single_coefficient_estimation/metrics/#absolute-error-to-ols",
            "library/tutorials/analyze_results/",
            "../../../single_coefficient_estimation/metrics/#absolute-error-to-ols",
        ),
        (PUBLIC_SITE_URL, "corpus/", "../"),
        (
            PUBLIC_SITE_URL + "corpus/?task=linear_regression",
            "",
            "corpus/?task=linear_regression",
        ),
    ],
)
def test_a_link_into_the_site_is_relative_to_its_page(
    url: str, page: str, expected: str
) -> None:
    assert relative_link(url, page, ROOTS) == expected


def test_a_link_off_the_site_is_left_alone() -> None:
    assert (
        relative_link("https://doi.org/10.1109/FOCS.2007.66", "corpus/", ROOTS) is None
    )


def test_the_public_address_is_the_default_site_url() -> None:
    """The hook's public address and mkdocs.yml's default are one value."""
    config = (Path(__file__).parents[3] / "docs/site/mkdocs.yml").read_text()
    default = re.search(r'site_url: !ENV \[DOCS_SITE_URL, "([^"]+)"\]', config)
    assert default is not None
    assert default[1] == PUBLIC_SITE_URL
