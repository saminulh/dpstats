"""The bibliography renders every citation in one style, from one file."""

from __future__ import annotations

import pytest

from bibliography import (
    CitedWork,
    cited_by,
    cited_works,
    full_citation,
    load,
    short_citation,
    work_link,
)

_FIXTURE = load(
    r"""
@article{one2001title,
  author = {Aum{\"u}ller, Martin},
  title = {A {PLAN} for Means \& Medians},
  journal = {Journal Name},
  volume = {3},
  number = {2},
  pages = {10--20},
  year = {2001},
  doi = {10.1000/one}
}
@inproceedings{two2002title,
  author = {Wang, Yu-Xiang and McMahan, H. Brendan},
  title = {Two Authors},
  booktitle = {Proceedings Name},
  volume = {7},
  pages = {1--2},
  year = {2002},
  url = {https://proceedings.example.org/two.html},
  eprint = {2002.00002},
  archivePrefix = {arXiv}
}
@misc{three2003title,
  author = {Dick, Travis and Gillenwater, Jennifer and Joseph, Matthew},
  title = {Three Authors?},
  year = {2003},
  eprint = {2003.00003},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2003.00003}
}
@misc{four2004title,
  author = {A, Ann and B, Bo and C, Cy and D, Di},
  title = {Four Authors},
  howpublished = {Presented at a Workshop},
  year = {2004},
  url = {https://workshop.example.org/}
}
@software{six2006code,
  author = {Doe, Jane},
  title = {Six (code)},
  url = {https://github.com/example/six},
  version = {abc1234},
  year = {2006}
}
@misc{five2005title,
  author = {Alabi, Daniel and B, Bo and C, Cy and D, Di and E, Ed},
  title = {Five Authors},
  year = {2005},
  eprint = {2005.00005},
  archivePrefix = {arXiv}
}
"""
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("one2001title", "Aumüller (2001)"),
        ("two2002title", "Wang and McMahan (2002)"),
        ("three2003title", "Dick, Gillenwater and Joseph (2003)"),
        ("four2004title", "A, B, C and D (2004)"),
        ("five2005title", "Alabi et al. (2005)"),
    ],
)
def test_the_in_text_form_names_up_to_four_authors(key: str, expected: str) -> None:
    assert short_citation(key, _FIXTURE) == expected


def test_a_journal_entry_renders_its_venue_and_doi() -> None:
    assert full_citation("one2001title", _FIXTURE) == (
        "Aumüller, M. (2001). <span data-verbatim>A PLAN for Means &amp; Medians</span>. "
        "<em>Journal Name</em>, <em>3</em>(2), 10–20. "
        '<a href="https://doi.org/10.1000/one">https://doi.org/10.1000/one</a>'
    )


def test_a_proceedings_entry_ends_with_its_published_page() -> None:
    """A published version without a DOI cites its own page, not the preprint."""
    assert full_citation("two2002title", _FIXTURE) == (
        "Wang, Y.-X., &amp; McMahan, H. B. (2002). <span data-verbatim>Two Authors</span>. "
        "In <em>Proceedings Name</em> (Vol. 7, pp. 1–2). "
        '<a href="https://proceedings.example.org/two.html">'
        "https://proceedings.example.org/two.html</a>"
    )


def test_a_preprint_is_italic_and_ends_with_its_arxiv_page() -> None:
    assert full_citation("three2003title", _FIXTURE) == (
        "Dick, T., Gillenwater, J., &amp; Joseph, M. (2003). <span data-verbatim><em>Three Authors?</em></span> "
        'arXiv. <a href="https://arxiv.org/abs/2003.00003">'
        "https://arxiv.org/abs/2003.00003</a>"
    )


def test_a_software_entry_names_its_version_and_repository() -> None:
    assert full_citation("six2006code", _FIXTURE) == (
        "Doe, J. (2006). <span data-verbatim><em>Six (code)</em></span>. (Version abc1234) [Computer software]. "
        '<a href="https://github.com/example/six">https://github.com/example/six</a>'
    )
    assert work_link("six2006code", _FIXTURE) == "https://github.com/example/six"


def test_an_article_without_a_journal_cites_its_title_and_doi() -> None:
    """A registrar may type a working paper @article without a journal."""
    working_paper = load(
        "@article{wp2022title, title={Working Paper}, author={Doe, Jane},"
        " year={2022}, doi={10.1000/wp}, publisher={Institute}}"
    )
    assert full_citation("wp2022title", working_paper) == (
        "Doe, J. (2022). <span data-verbatim><em>Working Paper</em></span>. Institute. "
        '<a href="https://doi.org/10.1000/wp">https://doi.org/10.1000/wp</a>'
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("one2001title", "https://doi.org/10.1000/one"),
        ("two2002title", "https://proceedings.example.org/two.html"),
        ("three2003title", "https://arxiv.org/abs/2003.00003"),
        ("four2004title", "https://workshop.example.org/"),
    ],
)
def test_a_bare_link_prefers_the_doi_then_the_published_page(
    key: str, expected: str
) -> None:
    assert work_link(key, _FIXTURE) == expected


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(KeyError):
        short_citation("missing2000title", _FIXTURE)


def test_a_reference_block_reads_keys_and_wrapped_locators() -> None:
    block = (
        "- two2002title: Algorithms 2 and 5 (arXiv v1 numbering;\n"
        "  Algorithms 2 and 6 in the published version).\n"
        "- one2001title.\n"
    )
    assert cited_works(block) == [
        CitedWork(
            "two2002title",
            "Algorithms 2 and 5 (arXiv v1 numbering; Algorithms 2 and 6 in the "
            "published version)",
        ),
        CitedWork("one2001title", ""),
    ]


@pytest.mark.parametrize(
    "block",
    [
        'Wang, Y.-X. (2018). "Title." UAI.',
        "- two2002title: Algorithm 2",
        "- Two2002title: Algorithm 2.",
    ],
)
def test_a_reference_block_outside_the_bullet_shape_is_refused(block: str) -> None:
    with pytest.raises(ValueError):
        cited_works(block)


def test_a_docstring_without_a_reference_block_cites_nothing() -> None:
    assert cited_by("Summary.\n\nRequires:\n    - Something.\n") == []
    assert cited_by(
        "Summary.\n\nReference:\n    - one2001title: Theorem 1.\n\nNotes:\n    x\n"
    ) == [CitedWork("one2001title", "Theorem 1")]


def test_the_library_bibliography_parses() -> None:
    assert "wang2018revisiting" in load().entries
