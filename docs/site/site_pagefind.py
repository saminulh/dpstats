"""Index the finished site with Pagefind, before the integrity sweep reads it.

A page's searchable content is its article (``data-pagefind-body`` in
``overrides/main.html``); this hook keeps the parts of an article that are not
prose out of the index, so a result's excerpt never shows flattened table text.
The binary comes from the pinned ``pagefind[bin]`` wheel, so no build needs Node.
"""

import re
import subprocess
import sys

# Markup inside an article that is data or chrome, not prose.
EXCLUDE_SELECTORS = [
    ".headerlink",
    ".doc-labels",  # mkdocstrings "pydantic-model" chips, kept out of titles
    ".dp-histogram-data",
    # The interactive results views: their controls, help and charts are
    # chrome, repeated once per Metric; each page's own prose stays indexed.
    ".dp-results",
    # Results tables, which repeat every configured algorithm's name.
    ".dp-leaderboard",
    ".dp-plotted-rows",
    ".dp-picker",
    ".dp-roster-landing table",
]
# An API page's member summary ("Classes:", "Functions:" ...) and the table
# after it: each member is documented, and indexed, under its own heading.
_MEMBER_SUMMARY = re.compile(
    r'<p(><span class="doc-section-title">'
    r"(?:Classes|Functions|Methods|Modules|Attributes):</span></p>\s*)<(div|table)\b"
)


def on_page_content(html, **kwargs):
    return _MEMBER_SUMMARY.sub(
        r"<p data-pagefind-ignore\1<\2 data-pagefind-ignore", html
    )


def on_post_build(config, **kwargs):
    command = [sys.executable, "-m", "pagefind", "--site", config["site_dir"]]
    for selector in EXCLUDE_SELECTORS:
        command += ["--exclude-selectors", selector]
    subprocess.run([*command, "--quiet"], check=True)
