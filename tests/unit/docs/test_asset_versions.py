"""Stylesheets and scripts are fetched afresh when, and only when, they change."""

from __future__ import annotations

from pathlib import Path

import asset_versions


def _render(docs: Path, html: str) -> str:
    asset_versions.on_pre_build({"docs_dir": str(docs)})
    return asset_versions.on_post_page(html, config={"docs_dir": str(docs)})


def test_a_local_asset_reference_carries_its_content_version(tmp_path: Path) -> None:
    """Only a changed file's version moves; other references are left as written."""
    (tmp_path / "javascripts").mkdir()
    (tmp_path / "stylesheets").mkdir()
    (tmp_path / "javascripts/site.js").write_text("one")
    (tmp_path / "stylesheets/extra.css").write_text("body {}")
    html = (
        '<link href="../../stylesheets/extra.css"><script src="javascripts/site.js">'
        '</script><script src="https://cdn.example/x.js"></script>'
        '<script src="javascripts/missing.js"></script><a href="stylesheets/extra.css?v=1">'
    )
    first = _render(tmp_path, html)
    assert 'href="../../stylesheets/extra.css?v=' in first
    assert 'src="javascripts/site.js?v=' in first
    assert 'src="https://cdn.example/x.js"' in first
    # A file the docs do not hold, or a reference already versioned, is left as written.
    assert 'src="javascripts/missing.js"' in first
    assert 'href="stylesheets/extra.css?v=1"' in first

    (tmp_path / "javascripts/site.js").write_text("two")
    second = _render(tmp_path, html)

    def version(text: str, name: str) -> str:
        return text.split(f"{name}?v=")[1][:12]

    assert version(second, "site.js") != version(first, "site.js")
    assert version(second, "extra.css") == version(first, "extra.css")
