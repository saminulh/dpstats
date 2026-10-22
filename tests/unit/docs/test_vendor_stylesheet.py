"""The vendored Tabler stylesheet carries no ``:has()`` rule.

While any is present, Chrome restyles the whole page after every DOM change;
``docs/site/trim_vendor_css.py`` removes them, and this keeps a re-vendored
Tabler from bringing them back unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDORED = ROOT / "docs/site/docs/stylesheets/vendor/tabler-1.5.1.min.css"
sys.path.insert(0, str(ROOT / "docs/site"))

import pytest  # noqa: E402

from trim_vendor_css import strip_has  # noqa: E402


def test_the_vendored_tabler_stylesheet_has_no_has_rules() -> None:
    assert ":has(" not in VENDORED.read_text(encoding="utf-8")


def test_trimming_drops_only_rules_using_has() -> None:
    css = (
        "a{color:red}.x:has(.y){color:blue}"
        "@media (min-width:1px){.z{margin:0}.p:has(> .q),.r{padding:0}}"
        '@media print{.s:has(.t){top:0}}@charset "utf-8";'
    )
    trimmed, removed = strip_has(css)
    assert removed == 3
    assert (
        trimmed == 'a{color:red}@media (min-width:1px){.z{margin:0}}@charset "utf-8";'
    )


def test_trimming_refuses_a_stylesheet_it_cannot_parse() -> None:
    """A stray token is reported where it stands, never copied through."""
    with pytest.raises(ValueError, match="line 1"):
        strip_has("a{color:red}/* note */;")
