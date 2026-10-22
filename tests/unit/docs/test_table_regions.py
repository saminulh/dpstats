"""Every table on a page scrolls in its own region, named by its heading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import site_copy
from table_regions import on_page_content


@pytest.fixture(autouse=True)
def written_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The region label as the maintainer might write it, so its field shows."""
    path = tmp_path / "copy.yml"
    path.write_text(json.dumps({"site": {"table_region": "{heading} table"}}))
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)


class _Page:
    title = "Linear Regression"


def test_a_bare_table_is_wrapped_and_named_by_the_heading_above_it() -> None:
    html = '<h2 id="m">Metrics at a glance<a class="headerlink">¶</a></h2>\n<table><tr><td>a</td></tr></table>'
    out = on_page_content(html, page=_Page())
    assert (
        '<div class="dp-table-scroll table-responsive" role="region" '
        'aria-label="Metrics at a glance table">\n<table>'
    ) in out
    assert out.count("<table") == 1 and out.rstrip().endswith("</table>\n</div>")


def test_an_already_wrapped_table_is_named_but_not_wrapped_again() -> None:
    html = (
        '<div class="dp-table-scroll table-responsive">\n'
        '<table class="dp-table"><tr><td>a</td></tr></table>\n</div>'
    )
    out = on_page_content(html, page=_Page())
    assert out.count("dp-table-scroll") == 1
    assert 'role="region" aria-label="Linear Regression table"' in out


def test_each_table_takes_the_nearest_heading_before_it() -> None:
    html = (
        "<h2>First</h2><table><tr><td>1</td></tr></table>"
        "<h3>Second &amp; last</h3><table><tr><td>2</td></tr></table>"
    )
    out = on_page_content(html, page=_Page())
    assert 'aria-label="First table"' in out
    assert 'aria-label="Second &amp; last table"' in out


def test_the_region_label_is_its_slot_until_the_maintainer_writes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "unwritten.yml"
    path.write_text(json.dumps({"site": {"table_region": "PLACEHOLDER"}}))
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)
    out = on_page_content("<h2>First</h2><table></table>", page=_Page())
    assert 'aria-label="⟨site.table_region⟩"' in out
