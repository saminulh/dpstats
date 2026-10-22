"""Remove every rule using ``:has()`` from a vendored minified stylesheet.

Tabler 1.5.1 ships ``:has()`` rules for components this site does not use
(vertical and folded navbars, steps, selectable tables, card tabs). While any
of them is present, Chrome restyles the whole page after every DOM change, so
a chart redrawing its axis labels while panned restyles thousands of elements
per frame. Removing them changes no computed style the site relies on; the one
custom property they set, ``--tblr-sidebar-width``, is restated in
``extra.css``.

    python docs/site/trim_vendor_css.py docs/site/docs/stylesheets/vendor/tabler-1.5.1.min.css
"""

from __future__ import annotations

import sys
from pathlib import Path

import tinycss2

# At-rules whose block holds declarations rather than rules.
_DECLARATION_BLOCKS = frozenset({"font-face", "page", "property"})


def _strip(rules: list) -> tuple[str, int]:
    """*rules* serialized without any style rule using ``:has()``, and how many
    rules went; an at-rule's own rules are walked, and one left empty goes too."""
    removed = 0
    parts: list[str] = []
    for rule in rules:
        if rule.type == "error":
            raise ValueError(
                f"unparseable CSS at line {rule.source_line}, column "
                f"{rule.source_column}: {rule.message}"
            )
        if rule.type == "qualified-rule" and ":has(" in tinycss2.serialize(
            rule.prelude
        ):
            removed += 1
            continue
        if (
            rule.type == "at-rule"
            and rule.content is not None
            and rule.lower_at_keyword not in _DECLARATION_BLOCKS
        ):
            inner, count = _strip(tinycss2.parse_rule_list(rule.content))
            removed += count
            if inner.strip():
                parts.append(
                    f"@{rule.at_keyword}{tinycss2.serialize(rule.prelude)}{{{inner}}}"
                )
            continue
        parts.append(rule.serialize())
    return "".join(parts), removed


def strip_has(css: str) -> tuple[str, int]:
    """*css* without any style rule whose selector uses ``:has()``, and how
    many rules went; at-rule blocks are walked, and one left empty goes too."""
    return _strip(tinycss2.parse_stylesheet(css))


def main() -> None:
    path = Path(sys.argv[1])
    trimmed, removed = strip_has(path.read_text(encoding="utf-8"))
    path.write_text(trimmed, encoding="utf-8")
    print(f"{path}: removed {removed} rules using :has()")


if __name__ == "__main__":
    main()
