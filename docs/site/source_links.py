"""Resolve authored page and downloaded notebook links to their published homes."""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from urllib.parse import urljoin

from claims import RELOCATED_PAGES
from site_links import PUBLIC_SITE_URL
from tutorials import TUTORIALS_DIRNAME, companion_scripts


def published_page(source: str, *, fragment: str = "") -> str:
    """Where one authored page under ``docs/`` is published in the site.

    Args:
        source: The page's path relative to the authored ``docs/`` directory.
        fragment: The link's fragment, when the caller resolved one. A plain
            task link names the task itself, so it reaches the task overview;
            a fragment names a specific heading the guide split may still
            carry, so it reaches the guide route by name.

    Returns:
        The published page path, relative to the site's document root.
    """
    if source.startswith("content/tasks/") and source.endswith(
        "/problem-formulation.md"
    ):
        return f"{source.split('/')[2]}/index.md"
    head, _, tail = source.partition("/")
    if head == "tasks":
        task_id = tail.removesuffix(".md")
        return f"{task_id}/guide.md" if fragment else f"{task_id}/index.md"
    return RELOCATED_PAGES.get(source, f"library/{source}")


def notebook_links(text: str, stem: str) -> str:
    """Repoint one tutorial notebook's markdown links at their published homes.

    mkdocs-jupyter renders a notebook straight to HTML, so MkDocs never sees
    these links and never rewrites them: the browser resolves them against the
    page's own directory URL, one level below the tutorials index. A sibling
    notebook reaches that notebook's page, and an authored guide reaches
    wherever :func:`published_page` puts it.
    """
    here = f"library/{TUTORIALS_DIRNAME}/{stem}"

    def replace(match: re.Match[str]) -> str:
        link = match[1]
        # A link to the site's own address is made relative on the rendered
        # page, with every other page's, by the site_links hook.
        if "://" in link or link.startswith(("/", "#")):
            return match[0]
        path, marker, fragment = link.partition("#")
        if path.endswith(".ipynb"):
            stem = posixpath.basename(path).removesuffix(".ipynb")
            sibling = f"library/{TUTORIALS_DIRNAME}/{stem}"
            route = posixpath.relpath(sibling, here) + "/"
            return "](" + route + (marker + fragment if marker else "") + ")"
        if not path.endswith(".md"):
            return match[0]
        target = posixpath.normpath(posixpath.join(TUTORIALS_DIRNAME, path))
        route = (
            published_page(target, fragment=fragment)
            .removesuffix(".md")
            .removesuffix(".ipynb")
            .removesuffix("/index")
        )
        page = posixpath.relpath(route, here)
        return "](" + page + "/" + (marker + fragment if marker else "") + ")"

    return re.sub(r"\]\(([^\s)]+)\)", replace, text)


def download_notebook(path: Path, site_url: str) -> bytes:
    """The source notebook a reader downloads, its links aimed at this build.

    A downloaded notebook has no page to be relative to, so each link into the
    site becomes an absolute address on the build it was downloaded from; a
    script the notebook names stays relative, since it downloads beside it.
    """
    root = site_url.rstrip("/") + "/"
    here = f"{root}library/{TUTORIALS_DIRNAME}/{path.stem}/"
    scripts = {script.name for script in companion_scripts(path)}
    document = json.loads(path.read_text(encoding="utf-8"))

    def absolute(match: re.Match[str]) -> str:
        link = match[1]
        if link.startswith(PUBLIC_SITE_URL):
            return "](" + root + link[len(PUBLIC_SITE_URL) :] + ")"
        if "://" in link or link.startswith("#") or link.partition("#")[0] in scripts:
            return match[0]
        return "](" + urljoin(here, link) + ")"

    for cell in document["cells"]:
        if cell["cell_type"] == "markdown":
            text = notebook_links("".join(cell["source"]), path.stem)
            cell["source"] = re.sub(r"\]\(([^\s)]+)\)", absolute, text)
    return (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode()
