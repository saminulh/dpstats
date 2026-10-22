"""T15, the site-wide Methodology page: the release's narrative, verbatim."""

from __future__ import annotations

import pytest

from methodology import (
    methodology_page,
    release_relative_links,
    split_title,
    tuning_fragment,
)
from templates.render import render
from templates.spec import TEMPLATES

NARRATIVE = """# Benchmark protocol

How this benchmark works.

## 3. Privacy budget

We set $\\delta$ for each instance from its sample size $n$:

$$
\\delta = \\min(n^{-1.1}, 0.01),
$$

as [the protocol file](benchmark-protocol.toml) states; see
[weighting](#8-weighting) and [the library](https://example.org/library/).
"""


def test_the_narrative_renders_as_written_under_its_own_title() -> None:
    record = methodology_page(NARRATIVE, "Distribution Release test")
    assert record.title == "Benchmark protocol"
    page = render(TEMPLATES["T15"], record)
    # The math, the headings and every sentence reach the page unchanged; a
    # link to a file inside the release shows as its text.
    expected = NARRATIVE.split("\n", 2)[2].replace(
        "[the protocol file](benchmark-protocol.toml)", "the protocol file"
    )
    assert expected.strip() in page
    assert "$$\n\\delta = \\min(n^{-1.1}, 0.01),\n$$" in page
    assert "[weighting](#8-weighting)" in page
    assert "[the library](https://example.org/library/)" in page
    assert "This release" not in page


def test_a_tuned_page_links_tuning_else_the_roster_section_else_the_page() -> None:
    # Each fragment is the id MkDocs' toc extension gives that heading.
    roster = "# Protocol\n\n## 4. The algorithms roster\n\nText.\n"
    assert tuning_fragment(roster) == "#4-the-algorithms-roster"
    assert tuning_fragment(roster + "\n### Tuning\n\nText.\n") == "#tuning"
    assert tuning_fragment(roster + "\n### 4.2 Tuning\n\nText.\n") == "#42-tuning"
    assert tuning_fragment(roster + "\n### Fine-tuning\n\nText.\n") == (
        "#4-the-algorithms-roster"
    )
    assert tuning_fragment(NARRATIVE) == ""
    assert tuning_fragment(None) == ""


def test_only_links_into_the_release_are_named_for_the_build() -> None:
    assert release_relative_links(NARRATIVE) == ("benchmark-protocol.toml",)


def test_a_narrative_without_its_own_title_is_refused() -> None:
    with pytest.raises(ValueError, match="title"):
        split_title("## 1. Tasks\n\nText.\n")


def test_a_build_reading_no_release_says_so() -> None:
    record = methodology_page(None, "", banner="No release.")
    page = render(TEMPLATES["T15"], record)
    assert record.title == "Methodology"
    assert '<p class="dp-build-source">No release.</p>' in page
