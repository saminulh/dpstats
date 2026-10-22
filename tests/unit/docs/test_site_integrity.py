"""The site gate rejects broken links and undeclared published pages."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import site_integrity
from site_integrity import check_site, on_nav, on_page_content

BASE = "https://example.org/benchmark/"


def write(site: Path, name: str, text: str) -> None:
    path = site / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_valid_links_include_downloads_queries_encoded_anchors_and_absolute_urls(
    tmp_path,
):
    write(tmp_path, "guide/index.html", '<h1 id="café">Guide</h1>')
    write(tmp_path, "guide/example.ipynb", "{}")
    write(
        tmp_path,
        "index.html",
        '<a href="guide/?task=mean#caf%C3%A9">Guide</a>'
        '<a href="https://example.org/benchmark/guide/">Absolute</a>'
        '<a href="guide/example.ipynb" download>Notebook</a>'
        '<a href="https://outside.example/missing">External</a>'
        '<a href="mailto:maintainer@example.org">Email</a>',
    )
    write(
        tmp_path,
        "old/index.html",
        '<meta http-equiv="refresh" content="0; url=../guide/">',
    )
    assert check_site(tmp_path, BASE) == ([], 3)


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ('<a href="missing/">Page</a>', "missing destination missing/"),
        (
            '<a href="guide/#old-heading">Section</a>',
            "missing anchor guide/#old-heading",
        ),
        (
            '<a href="missing.ipynb" download>Notebook</a>',
            "missing destination missing.ipynb",
        ),
        ('<img src="missing.png">', "missing destination missing.png"),
        ('<script src="missing.js"></script>', "missing destination missing.js"),
        ('<a href="/guide/">Root</a>', "links outside the site /guide/"),
        (
            '<a href="https://example.org/guide/">Host root</a>',
            "links outside the site https://example.org/guide/",
        ),
    ],
)
def test_broken_page_anchor_or_asset_link_fails(tmp_path, markup, expected):
    write(tmp_path, "index.html", markup)
    write(tmp_path, "guide/index.html", '<h1 id="current">Guide</h1>')
    errors, _ = check_site(tmp_path, BASE)
    assert errors == [f"index.html: {expected}"]


SEARCHABLE = "<article data-pagefind-body>{}</article>"


def test_sitemap_cannot_list_missing_or_unsearchable_pages(tmp_path):
    write(tmp_path, "index.html", SEARCHABLE.format("Home"))
    write(tmp_path, "plain/index.html", "Plain")
    write(
        tmp_path,
        "sitemap.xml",
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{BASE}</loc></url>"
        f"<url><loc>{BASE}removed/</loc></url>"
        f"<url><loc>{BASE}plain/</loc></url></urlset>",
    )
    errors, _ = check_site(tmp_path, BASE)
    assert errors == [
        f"sitemap.xml: indexes a page absent from search {BASE}plain/",
        f"sitemap.xml: missing destination {BASE}removed/",
    ]


def test_search_cannot_index_a_page_the_sitemap_omits(tmp_path):
    write(tmp_path, "index.html", SEARCHABLE.format("Home"))
    write(tmp_path, "templates/t01/index.html", SEARCHABLE.format("Preview"))
    write(
        tmp_path,
        "sitemap.xml",
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{BASE}</loc></url></urlset>",
    )
    errors, _ = check_site(tmp_path, BASE)
    assert errors == [
        "templates/t01/index.html: searchable but absent from the sitemap"
    ]


def test_a_lazily_loaded_search_bundle_must_exist(tmp_path):
    write(tmp_path, "index.html", '<template data-src="pagefind/ui.js"></template>')
    errors, _ = check_site(tmp_path, BASE)
    assert errors == ["index.html: missing destination pagefind/ui.js"]


def test_an_empty_or_wrong_build_directory_is_not_a_pass(tmp_path):
    assert check_site(tmp_path, BASE) == (["site: no HTML pages found"], 0)


def test_hidden_details_cannot_reenter_the_sitemap(tmp_path):
    write(tmp_path, "index.html", "Home")
    write(tmp_path, "symbol/index.html", "Hidden API detail")
    write(
        tmp_path,
        "sitemap.xml",
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{BASE}symbol/</loc></url></urlset>",
    )
    errors, _ = check_site(tmp_path, BASE)
    assert errors == [f"sitemap.xml: indexes a page absent from search {BASE}symbol/"]


def test_only_declared_pages_or_explicitly_hidden_details_are_published():
    declared = SimpleNamespace(file=SimpleNamespace(src_uri="guide.md"), meta={})
    retired = SimpleNamespace(file=SimpleNamespace(src_uri="old.md"), meta={})
    detail = SimpleNamespace(
        file=SimpleNamespace(src_uri="symbol.md"), meta={"search": {"exclude": True}}
    )
    on_nav(SimpleNamespace(pages=[declared]))
    assert on_page_content("guide", declared) == "guide"
    assert on_page_content("detail", detail) == "detail"
    with pytest.raises(
        site_integrity.PluginError, match="Undeclared site page: old.md"
    ):
        on_page_content("old", retired)
