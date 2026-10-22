"""Keep the literate-nav input files out of the built site.

Every section's navigation is written as a ``SUMMARY.md`` beside its pages, and
literate-nav reads those files to build the nav. They are build input, not
reader content, so once the nav is resolved they leave the file set instead of
being published as pages of their own.

The removal runs after literate-nav's own ``on_files``, which is declared at
priority -100, so the nav is already resolved when this hook sees the files.
"""

from __future__ import annotations

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files

NAV_FILE = "SUMMARY.md"


@event_priority(-200)
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Drop every navigation file, keeping the pages it named."""
    for file in list(files):
        if file.src_uri == NAV_FILE or file.src_uri.endswith(f"/{NAV_FILE}"):
            files.remove(file)
    return files
