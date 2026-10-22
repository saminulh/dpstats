"""An area landing is the navigation's own list of that area's pages."""

from __future__ import annotations


from area_index import area_pages, branch_for, first_page, page_summary

NAV = [
    {"Leaderboard": [{"DP Benchmark": "index.md"}]},
    {
        "Library": [
            {"Library": "library/index.md"},
            {"Get started": "library/tutorials/getting-started.md"},
            {"Tutorials": "library/tutorials/"},
            {
                "Reference": [
                    {"API": "library/reference/index.md"},
                    {"Budgets and domains": "library/reference/budgets/"},
                ]
            },
        ]
    },
]


def test_the_landing_lists_the_areas_pages_and_not_itself() -> None:
    pages = area_pages(branch_for(NAV, "library/index.md"), "library")
    assert [item.text for item in pages] == ["Get started", "Tutorials", "Reference"]
    assert [item.url for item in pages] == [
        "tutorials/getting-started.md",
        "tutorials/index.md",
        "reference/index.md",
    ]


def test_a_nested_area_is_reached_through_its_own_first_page() -> None:
    """Reference sits inside Library, and its landing lists its own sections."""
    branch = branch_for(NAV, "library/reference/index.md")
    assert [item.url for item in area_pages(branch, "library/reference")] == [
        "budgets/index.md"
    ]
    assert first_page(branch) == "library/reference/index.md"


def test_an_area_the_navigation_does_not_declare_has_no_branch() -> None:
    assert branch_for(NAV, "operate/index.md") == []


def test_a_page_opening_on_markup_declares_no_sentence() -> None:
    """A generated page that opens on a table states nothing a landing can use."""
    assert page_summary("# API\n\n| Section | Description |\n| --- | --- |\n") == ""
    assert page_summary("---\ntitle: X\n---\n\n# X\n\nOne line. Another.\n") == (
        "One line."
    )
    assert page_summary("# X\n\nNo terminator here\n") == "No terminator here"
