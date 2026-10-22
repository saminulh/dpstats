"""Check the finished site's internal links and its declared page inventory."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import logging
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

from mkdocs.exceptions import PluginError
from mkdocs.plugins import event_priority

log = logging.getLogger("mkdocs.site_integrity")
_nav_pages: set[str] = set()


class PageLinks(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: set[str] = set()
        self.searchable = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        if "data-pagefind-body" in attrs:
            self.searchable = True
        # data-src and data-href: assets a script loads on demand, such as search.
        for attr in ("href", "src", "data-src", "data-href"):
            if attrs.get(attr):
                self.links.add(attrs[attr])


def check_site(site: Path, site_url: str) -> tuple[list[str], int]:
    """Check local destinations, anchors, and that search indexes the sitemap.

    External sites are outside this offline check. A link to the site's own
    host that leaves its base path, such as a root-absolute ``/page/`` on a
    project site, is broken once deployed. Query parameters do not change a
    static file's destination; fragments do and must exist in HTML.
    """
    site = site.resolve()
    base = site_url.rstrip("/") + "/"
    origin = urlsplit(base)
    pages = {
        path: PageLinks(path.read_text(encoding="utf-8"))
        for path in site.rglob("*.html")
    }
    errors: set[str] = set()
    if not pages:
        errors.add("site: no HTML pages found")

    def resolve(source: str, href: str):
        url = urlsplit(urljoin(urljoin(base, source), href))
        if (url.scheme, url.netloc) != (origin.scheme, origin.netloc):
            return None
        if not url.path.startswith(origin.path):
            return url.path
        path = (site / unquote(url.path[len(origin.path) :])).resolve()
        if not path.is_relative_to(site):
            return None
        if path.is_dir():
            path /= "index.html"
        return path, unquote(url.fragment.split(":~:text=", 1)[0])

    def check(source: str, href: str, *, index: bool = False):
        target = resolve(source, href)
        if target is None:
            return
        if isinstance(target, str):
            errors.add(f"{source}: links outside the site {href}")
            return
        path, fragment = target
        if not path.is_file():
            errors.add(f"{source}: missing destination {href}")
        elif path in pages:
            page = pages[path]
            if fragment and fragment not in page.ids:
                errors.add(f"{source}: missing anchor {href}")
            if index and path.relative_to(site).parts[0] == "templates":
                errors.add(f"{source}: indexes a preview page {href}")

    for path, page in pages.items():
        source = path.relative_to(site).as_posix()
        for href in page.links:
            check(source, href)
    sitemap = site / "sitemap.xml"
    locations: list[str] = []
    if sitemap.is_file():
        locations = [
            node.text or ""
            for node in ElementTree.parse(sitemap).iter(
                "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
            )
        ]
        if len(locations) != len(set(locations)):
            errors.add("sitemap.xml: duplicate canonical URLs")
        for location in locations:
            check("sitemap.xml", location, index=True)
    # Pagefind indexes exactly the pages that mark their content
    # data-pagefind-body, so search and the sitemap list the same pages.
    listed: set[Path] = set()
    for location in locations:
        target = resolve("sitemap.xml", location)
        if isinstance(target, tuple) and target[0] in pages:
            listed.add(target[0])
            if not pages[target[0]].searchable:
                errors.add(f"sitemap.xml: indexes a page absent from search {location}")
    for path, page in pages.items():
        if page.searchable and path not in listed:
            source = path.relative_to(site).as_posix()
            errors.add(f"{source}: searchable but absent from the sitemap")
    return sorted(errors), len(pages)


def on_nav(nav, **kwargs):
    _nav_pages.clear()
    _nav_pages.update(page.file.src_uri for page in nav.pages)
    return nav


def on_page_content(html, page, **kwargs):
    if page.file.src_uri not in _nav_pages and not page.meta.get("search", {}).get(
        "exclude", False
    ):
        raise PluginError(f"Undeclared site page: {page.file.src_uri}")
    return html


@event_priority(-1000)
def on_post_build(config, **kwargs):
    errors, count = check_site(Path(config.site_dir), config.site_url)
    if errors:
        raise PluginError("Site integrity failed:\n" + "\n".join(errors))
    log.info("Site integrity passed (%s HTML pages)", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--site-url", required=True)
    args = parser.parse_args()
    errors, count = check_site(args.site, args.site_url)
    print(
        "\n".join(errors) if errors else f"Site integrity passed ({count} HTML pages)"
    )
    raise SystemExit(bool(errors))
