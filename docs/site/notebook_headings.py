"""Drop the heading anchors nbconvert writes into a rendered notebook.

A notebook page is converted to HTML by the notebook plugin rather than by
MkDocs, and that conversion appends its own permalink to every heading. The
anchor is invisible on the page, because the theme hides it, but its pilcrow is
part of the heading's text everywhere the text is read again, search results
included. Every other page's
permalinks come from the theme and are already excluded, so this hook exists to
give notebook pages the same headings as the rest of the site. It also carries
authored cell image descriptions into HTML: nbconvert's lab template currently
omits cell ``alt`` metadata from its image tags, and a Plotly figure the page
draws is a ``div`` with no image tag at all.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from mkdocs.structure.pages import Page
from typing import Any

# The id the site gives each Plotly figure it draws on a notebook page.
FIGURE_ID_PREFIX = "dp-notebook-figure-"
# The class the site's chart runtime draws each such figure by.
FIGURE_CLASS = "dp-notebook-figure"

# The anchor nbconvert appends inside each heading it renders.
ANCHOR = re.compile(r'<a\b[^>]*class="anchor-link"[^>]*>.*?</a>', re.S)


def on_page_content(html: str, *, page: Page, **_: Any) -> str:
    """Preserve notebook image descriptions and remove converter permalinks."""
    html = ANCHOR.sub("", html)
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    document = json.loads(Path(page.file.abs_src_path).read_text())
    # A cell with no outputs has no image to describe: an unrun tutorial
    # publishes its source cells alone.
    descriptions = {
        cell["id"]: cell["metadata"]["alt"]
        for cell in document["cells"]
        if cell.get("metadata", {}).get("alt") and cell.get("outputs")
    }
    if not descriptions:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for cell_id, description in descriptions.items():
        cell = soup.find(id=f"cell-id={cell_id}")
        if cell is None:
            raise RuntimeError(
                f"Notebook image description has no rendered cell: {cell_id}"
            )
        images = cell.select(".jp-OutputArea img")
        figures = cell.select(f'.jp-OutputArea [id^="{FIGURE_ID_PREFIX}"]')
        if len(images) + len(figures) != 1:
            raise RuntimeError(
                f"Notebook alt text requires one image in cell {cell_id}"
            )
        if images:
            images[0]["alt"] = description
        else:
            figures[0]["role"] = "img"
            figures[0]["aria-label"] = description
    return str(soup)
