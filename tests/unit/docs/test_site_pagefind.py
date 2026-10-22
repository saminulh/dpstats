"""Search indexes an API page's documented members, not its summary tables."""

from site_pagefind import on_page_content

SUMMARY = (
    '<p><span class="doc-section-title">{title}:</span></p>\n'
    '    <div class="dp-table-scroll" role="region"><table><tr><td>Member</td></tr></table></div>'
)


def test_member_summary_tables_are_kept_out_of_search():
    for title in ("Classes", "Functions", "Methods", "Modules", "Attributes"):
        html = on_page_content(SUMMARY.format(title=title))
        assert html.count("data-pagefind-ignore") == 2, title
        assert '<div data-pagefind-ignore class="dp-table-scroll"' in html


def test_documented_parameters_stay_searchable():
    html = SUMMARY.format(title="Parameters")
    assert on_page_content(html) == html


def test_an_unwrapped_summary_table_is_kept_out_of_search():
    html = '<p><span class="doc-section-title">Classes:</span></p><table></table>'
    assert on_page_content(html) == (
        '<p data-pagefind-ignore><span class="doc-section-title">Classes:</span></p>'
        "<table data-pagefind-ignore></table>"
    )
