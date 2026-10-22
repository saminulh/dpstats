"""Version every local stylesheet and script a page references by its content.

The site's CSS and JavaScript keep stable file names, so a browser that cached
one keeps serving it after a rebuild or deploy: new pages then run against old
styles and scripts. Each page's references to them, from the theme or from
generated page content alike, gain ``?v=`` and the first twelve hex digits of
the file's SHA-256, so a changed file is fetched afresh and an unchanged one
stays cached.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

# A local reference into the site's own asset directories, without a query.
ASSET = re.compile(
    r'(?P<attr>(?:src|href)=")(?P<path>(?:\.\./|\./)*(?P<file>(?:javascripts|stylesheets)/[^"?#:]+\.(?:js|css)))"'
)

_versions: dict[str, str] = {}


def _version(docs_dir: Path, file: str) -> str | None:
    """The content version of one asset under the docs directory, if it exists."""
    if file not in _versions:
        path = docs_dir / file
        _versions[file] = (
            hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.is_file() else ""
        )
    return _versions[file] or None


def on_pre_build(config: MkDocsConfig, **kwargs) -> None:
    """Forget the previous build's versions, so a rebuild reads the files again."""
    _versions.clear()


def on_post_page(output: str, *, config: MkDocsConfig, **kwargs) -> str:
    """Append each local asset reference's content version to one rendered page."""
    docs_dir = Path(config["docs_dir"])

    def versioned(match: re.Match[str]) -> str:
        version = _version(docs_dir, match["file"])
        if version is None:
            return match[0]
        return f'{match["attr"]}{match["path"]}?v={version}"'

    return ASSET.sub(versioned, output)
