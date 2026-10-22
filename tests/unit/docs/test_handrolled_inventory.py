"""Every piece of the site's own machinery is listed with the reason it stays.

``docs/site/handrolled.yml`` names each custom property the stylesheet defines
and each site script; a new one cannot land without a stated reason, and a
removed one leaves no stale entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = yaml.safe_load(
    (ROOT / "docs/site/handrolled.yml").read_text(encoding="utf-8")
)
STYLESHEET = ROOT / "docs/site/docs/stylesheets/extra.css"
SCRIPTS = ROOT / "docs/site/docs/javascripts"


def _listed(section: str) -> list[str]:
    return [name for entry in INVENTORY[section] for name in entry["names"]]


def test_every_custom_property_is_listed_once_with_a_reason() -> None:
    css = re.sub(r"/\*.*?\*/", "", STYLESHEET.read_text(encoding="utf-8"), flags=re.S)
    defined = set(re.findall(r"(--dp-[\w-]+)\s*:", css))
    listed = _listed("css_tokens")
    assert sorted(defined - set(listed)) == [], "custom properties with no entry"
    assert (
        sorted(set(listed) - defined) == []
    ), "entries for properties that no longer exist"
    assert len(listed) == len(set(listed)), "a property listed twice"


def test_every_site_script_is_listed_once_with_a_reason() -> None:
    present = {path.name for path in SCRIPTS.glob("*.js")}
    listed = _listed("scripts")
    assert sorted(present - set(listed)) == [], "site scripts with no entry"
    assert (
        sorted(set(listed) - present) == []
    ), "entries for scripts that no longer exist"
    assert len(listed) == len(set(listed)), "a script listed twice"


def test_every_entry_states_its_reason() -> None:
    entries = [*INVENTORY["css_tokens"], *INVENTORY["scripts"], *INVENTORY["kept"]]
    assert all(str(entry.get("reason", "")).strip() for entry in entries)
