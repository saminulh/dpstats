"""Make every link into the site relative, whichever host the build is for.

Authored pages and notebooks link to the site at its public address, and a
build for another host (a preview, a local server) would send those links
there. Each page's links to the public address or to this build's own
``site_url`` become relative to the page, so the built site works wherever it
is served.
"""

from __future__ import annotations

import posixpath
import re
from urllib.parse import urlsplit

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page

# The address authored pages link to; mkdocs.yml's default site_url.
PUBLIC_SITE_URL = "https://saminulh.github.io/dpstats/"

HREF = re.compile(r'href="(?P<url>https?://[^"]+)"')


def relative_link(url: str, page_url: str, roots: tuple[str, ...]) -> str | None:
    """The link from the page at *page_url* to *url*, or ``None`` if it leaves the site."""
    for root in roots:
        if url.startswith(root):
            target = urlsplit(url[len(root) :])
            route = posixpath.relpath(target.path or ".", page_url or ".")
            if target.path.endswith("/") or not target.path:
                route = route.rstrip("/") + "/" if route != "." else "./"
            if target.query:
                route += "?" + target.query
            if target.fragment:
                route += "#" + target.fragment
            return route
    return None


def on_page_content(html: str, *, page: Page, config: MkDocsConfig, **_: object) -> str:
    """Rewrite the page's links into the site as relative links."""
    roots = tuple(
        dict.fromkeys(
            root.rstrip("/") + "/" for root in (PUBLIC_SITE_URL, config["site_url"])
        )
    )

    def replace(match: re.Match[str]) -> str:
        route = relative_link(match["url"], page.url, roots)
        return match[0] if route is None else f'href="{route}"'

    return HREF.sub(replace, html)
