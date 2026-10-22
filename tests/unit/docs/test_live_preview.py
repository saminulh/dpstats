"""The preview server also serves the standalone browser regression pages."""

from __future__ import annotations

import sys
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parents[3] / "docs" / "site"
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))

from live_preview import PreviewServer  # noqa: E402


def test_a_page_may_open_a_burst_of_connections_at_once() -> None:
    """A page fetching a dozen scripts must not have some of them reset.

    The standard library's default listen backlog is five, and a browser opens
    that many connections in one go; the refused ones failed the browser
    regression pages at random.
    """
    assert PreviewServer.request_queue_size >= 32
