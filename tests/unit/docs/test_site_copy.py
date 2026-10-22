"""Explanatory site text lives in slots: unwritten, blank, or the maintainer's own."""

import html
import json
import re
from pathlib import Path

import pytest
import yaml

import leaderboard
import playground
import site_copy
from panel_data import TaskData
from site_copy import (
    PLACEHOLDER,
    SITE_COPY_KEYS,
    copy_html,
    copy_text,
    load_site_copy,
    inline_html,
    slot_marker,
)
from tests.unit.docs.conftest import parse_html

CONTENT_FILE = (
    Path(site_copy.__file__).resolve().parents[1] / "content" / "site-copy.yml"
)


def _write(path: Path, groups: dict[str, dict[str, str]]) -> Path:
    # JSON is also YAML, and quotes the literal author text unambiguously.
    path.write_text(json.dumps(groups))
    return path


def test_the_content_file_declares_exactly_the_known_slots() -> None:
    loaded = yaml.safe_load(CONTENT_FILE.read_text(encoding="utf-8"))
    assert {group: set(entries) for group, entries in loaded.items()} == SITE_COPY_KEYS


def test_missing_and_blank_copy_render_nothing(tmp_path: Path) -> None:
    path = tmp_path / "site-copy.yml"
    assert load_site_copy("explore", path) == {}
    path.write_text('explore:\n  feature_intro: "  "\n  outcomes_glossary:\n')
    assert load_site_copy("explore", path) == {}
    assert copy_html("explore", "feature_intro", path=path) == ""
    assert copy_text("explore", "outcomes_glossary", path=path) == ""


def test_a_placeholder_renders_a_marker_naming_its_slot(tmp_path: Path) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": PLACEHOLDER}})
    marker = slot_marker("explore", "feature_intro")
    assert marker == "⟨explore.feature_intro⟩"
    rendered = parse_html(copy_html("explore", "feature_intro", path=path))
    slot = rendered.find("data-copy-slot")
    assert slot.attrs["data-copy-slot"] == "explore.feature_intro"
    assert "dp-copy-slot" in slot.attrs["class"].split()
    assert slot.text == marker
    assert copy_text("explore", "feature_intro", path=path) == marker


def test_written_text_renders_escaped_and_unmarked(tmp_path: Path) -> None:
    text = 'A <marker> & its "trials".\nAnother line.'
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": text}})
    rendered = copy_html("explore", "feature_intro", path=path)
    assert html.escape(text, quote=False) in rendered
    assert "dp-copy-slot" not in rendered and "dp-authored-copy" in rendered
    assert copy_text("explore", "feature_intro", path=path) == text


def test_a_markdown_link_renders_as_a_link_and_tooltips_keep_its_text(
    tmp_path: Path,
) -> None:
    text = (
        "See the [protocol](https://saminulh.github.io/dpstats/x/) "
        "and [Explore](../explore/); [this](javascript:alert(1)) stays text."
    )
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": text}})
    rendered = parse_html(copy_html("explore", "feature_intro", path=path))
    links = rendered.find_all("href", tag="a")
    assert [(a.attrs["href"], a.text) for a in links] == [
        ("https://saminulh.github.io/dpstats/x/", "protocol"),
        ("../explore/", "Explore"),
    ]
    assert "[this](javascript:alert(1))" in rendered.text
    assert copy_text("explore", "feature_intro", path=path) == (
        "See the protocol and Explore; [this](javascript:alert(1)) stays text."
    )


@pytest.mark.parametrize(
    "source, message",
    [
        ("- prose\n", "expected a mapping"),
        ("explore: prose\n", "malformed page group"),
        ("explore:\n  typo: prose\n", "unknown text location explore.typo"),
        ("nowhere:\n  intro: prose\n", "unknown or malformed page group 'nowhere'"),
        ("explore:\n  outcomes_glossary: 12\n", "must be text or blank"),
    ],
)
def test_invalid_copy_names_the_problem(
    tmp_path: Path, source: str, message: str
) -> None:
    path = tmp_path / "site-copy.yml"
    path.write_text(source)
    with pytest.raises(ValueError, match=message):
        load_site_copy("explore", path)


@pytest.mark.parametrize("text", ["", PLACEHOLDER, 'A <b> & "c".\nNext line.'])
def test_each_page_slot_follows_its_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measured_release: TaskData,
    text: str,
) -> None:
    path = _write(
        tmp_path / "copy.yml",
        {group: {key: text for key in keys} for group, keys in SITE_COPY_KEYS.items()},
    )
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)
    explore = playground.playground_page(
        measured_release, "fixture", depth=3, provenance="provenance.json"
    ).body
    standings = leaderboard.leaderboard_body(
        "mean_estimation",
        leaderboard.collection_standings(measured_release),
        "fixture",
        depth=2,
        published=True,
        provenance="provenance.json",
        policy=measured_release.index,
        explore_href="../explore/",
        roster_href="../roster/",
        methodology_href="../../methodology/",
    )
    for body, keys in (
        (
            explore,
            (
                "explore.introduction",
                "explore.outcomes_intro",
                "explore.outcomes_glossary",
                "explore.outcomes_bar_help",
                "explore.feature_help",
                "explore.feature_scale_help",
                "explore.point_details_prompt",
                "explore.view_profiles",
                "explore.view_feature_short",
                "explore.add_algorithm",
                "explore.outcome_compare_label",
                "explore.outcome_clear",
                "explore.feature_label",
                "explore.bar_heading",
                "explore.breakdown_caption",
                "explore.retry_loading",
                "explore.reload_page",
                "explore.interpretation",
                "shared.baseline_label",
                "shared.metric_label",
                "shared.quantile_label",
                "shared.full_definition",
                "shared.plotted_heading",
                "shared.show_top",
                "shared.rankings_heading",
                "shared.column_algorithm",
                "leaderboard.column_headline",
            ),
        ),
        (
            standings,
            (
                "leaderboard.intro",
                "leaderboard.intro_corpus",
                "shared.chart_noscript",
                "shared.rankings_heading",
                "shared.quantile_label",
                "shared.plotted_heading",
            ),
        ),
    ):
        page = parse_html(body)
        for key in keys:
            slots = [
                node
                for node in page.find_all("data-copy-slot")
                if node.attrs["data-copy-slot"] == key
            ]
            assert bool(slots) == bool(text), key
            shown = slot_marker(*key.split(".")) if text == PLACEHOLDER else text
            assert all(slot.text == " ".join(shown.split()) for slot in slots)
            assert all(
                ("dp-copy-slot" in slot.attrs["class"]) == (text == PLACEHOLDER)
                for slot in slots
            )
        if text and text != PLACEHOLDER:
            assert html.escape(text, quote=False) in body
            assert not page.find_all(tag="b")

    # Scripts read the raw values; a placeholder becomes a marker in the browser.
    carried = json.loads(
        parse_html(explore).find("data-site-copy").attrs["data-site-copy"]
    )
    assert set(carried) == {"shared", "explore", "component", "leaderboard", "task"}
    standing = json.loads(
        parse_html(standings).find("data-site-copy").attrs["data-site-copy"]
    )
    assert set(standing) == {"shared", "leaderboard", "component", "task"}
    assert carried["shared"] == (
        {key: text for key in SITE_COPY_KEYS["shared"]} if text else {}
    )

    # No rankings heading carries hover text.
    assert not [
        heading
        for heading in parse_html(standings).find_all(tag="th")
        if "title" in heading.attrs
    ]


def test_a_bare_page_name_links_where_the_page_offers_it() -> None:
    """``[Corpus]`` becomes a link only when the rendering page names that page."""
    text = "See [Corpus] for the studies, and [Elsewhere] stays text."
    linked = inline_html(text, {"Corpus": "../../corpus/?task=t&x=1"})
    assert '<a href="../../corpus/?task=t&amp;x=1">Corpus</a>' in linked
    assert "[Elsewhere]" in linked
    assert "<a" not in inline_html(text)


def test_a_page_the_build_lacks_shows_its_bare_name_as_text() -> None:
    """A page named ``None`` leaves its bare name as plain text, in any case."""
    linked = inline_html(
        "See the [roster] and the [explore] page.",
        {"Roster": "../roster/", "Explore": None},
    )
    assert linked == 'See the <a href="../roster/">roster</a> and the explore page.'


def test_a_page_name_resolves_as_a_markdown_reference_link() -> None:
    """The page mapping is ordinary reference definitions: any case, any form."""
    linked = inline_html(
        "Open [the studies][corpus], or [Corpus][].", {"Corpus": "../corpus/"}
    )
    assert linked == (
        'Open <a href="../corpus/">the studies</a>, or <a href="../corpus/">Corpus</a>.'
    )


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "data:text/html,x", "mailto:a@b.org", "ftp://x/"]
)
def test_a_link_to_another_scheme_stays_its_text(url: str) -> None:
    for text, pages in (
        (f"Go [there]({url}) now.", None),
        ("Go [there] now.", {"there": url}),
    ):
        rendered = inline_html(text, pages)
        assert "<a" not in rendered
        assert f"[there]({url})" in html.unescape(rendered)


def test_one_paragraph_renders_inline_with_its_line_breaks() -> None:
    assert inline_html("One <line>\nand [a](#b).") == (
        'One &lt;line&gt;\nand <a href="#b">a</a>.'
    )


@pytest.fixture
def declared_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare one parameterized slot, as SITE_COPY_FIELDS does for real ones."""
    monkeypatch.setitem(
        site_copy.SITE_COPY_FIELDS, ("explore", "feature_intro"), {"share", "name"}
    )


@pytest.mark.usefixtures("declared_fields")
def test_fields_fill_with_escaped_values_and_doubled_braces_stay_literal(
    tmp_path: Path,
) -> None:
    text = "{name} holds {share} of {{all}} [instances](#bins)."
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": text}})
    fields = {"share": "40%", "name": "<b>A & B</b>"}
    rendered = copy_html("explore", "feature_intro", path=path, fields=fields)
    assert rendered.endswith(
        "&lt;b&gt;A &amp; B&lt;/b&gt; holds 40% of {all} "
        '<a href="#bins">instances</a>.</p>'
    )
    assert copy_text("explore", "feature_intro", path=path, fields=fields) == (
        "<b>A & B</b> holds 40% of {all} instances."
    )


@pytest.mark.usefixtures("declared_fields")
def test_a_field_value_is_never_read_as_markdown(tmp_path: Path) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": "Got {name}."}})
    fields = {"share": "", "name": "*x* [y](https://z/)"}
    rendered = copy_html("explore", "feature_intro", path=path, fields=fields)
    assert "*x* [y](https://z/)" in rendered and "<a" not in rendered
    assert "<em>" not in rendered


@pytest.mark.usefixtures("declared_fields")
def test_a_placeholder_with_fields_renders_its_marker(tmp_path: Path) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": PLACEHOLDER}})
    fields = {"share": "40%", "name": "A"}
    marker = slot_marker("explore", "feature_intro")
    rendered = parse_html(
        copy_html("explore", "feature_intro", path=path, fields=fields)
    )
    assert rendered.find("data-copy-slot").text == marker
    assert copy_text("explore", "feature_intro", path=path, fields=fields) == marker


@pytest.mark.usefixtures("declared_fields")
@pytest.mark.parametrize(
    "text, message",
    [
        ("Holds {count}.", r"explore\.feature_intro uses undeclared field \{count\}"),
        ("Holds { share }.", r"explore\.feature_intro has a stray brace"),
        ("Holds {share.", r"explore\.feature_intro has a stray brace"),
    ],
)
def test_the_loader_refuses_an_undeclared_field_or_a_stray_brace(
    tmp_path: Path, text: str, message: str
) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": text}})
    with pytest.raises(ValueError, match=message):
        load_site_copy("explore", path)


def test_a_slot_without_declared_fields_refuses_any_field(tmp_path: Path) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_help": "A {x}."}})
    with pytest.raises(ValueError, match=r"undeclared field \{x\}"):
        load_site_copy("explore", path)


@pytest.mark.usefixtures("declared_fields")
@pytest.mark.parametrize(
    "fields", [None, {"share": "1"}, {"share": "1", "name": "a", "x": "b"}]
)
def test_a_caller_passes_exactly_the_declared_fields(
    tmp_path: Path, fields: dict[str, str] | None
) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": PLACEHOLDER}})
    with pytest.raises(ValueError, match=r"explore\.feature_intro takes fields"):
        copy_html("explore", "feature_intro", path=path, fields=fields)
    with pytest.raises(ValueError, match=r"explore\.feature_intro takes fields"):
        copy_text("explore", "feature_intro", path=path, fields=fields)


def test_every_declared_field_names_a_known_slot() -> None:
    for group, key in site_copy.SITE_COPY_FIELDS:
        assert key in SITE_COPY_KEYS[group], (group, key)


def test_a_label_renders_as_a_bare_span_carrying_its_slot(tmp_path: Path) -> None:
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": PLACEHOLDER}})
    rendered = site_copy.copy_span("explore", "feature_intro", path=path)
    assert rendered == (
        '<span class="dp-authored-copy dp-copy-slot" '
        'data-copy-slot="explore.feature_intro">⟨explore.feature_intro⟩</span>'
    )
    blank = _write(tmp_path / "blank.yml", {"explore": {"feature_intro": ""}})
    assert site_copy.copy_span("explore", "feature_intro", path=blank) == ""


@pytest.mark.usefixtures("declared_fields")
def test_markdown_copy_keeps_its_links_and_fills_fields_as_written(
    tmp_path: Path,
) -> None:
    text = "{name} holds {share} of {{all}}; see [the guide](https://x.org/g/)."
    path = _write(tmp_path / "copy.yml", {"explore": {"feature_intro": text}})
    fields = {"share": "40%", "name": "A & B"}
    assert site_copy.copy_markdown(
        "explore", "feature_intro", path=path, fields=fields
    ) == ("A & B holds 40% of {all}; see [the guide](https://x.org/g/).")
    marked = _write(
        tmp_path / "marked.yml", {"explore": {"feature_intro": PLACEHOLDER}}
    )
    assert (
        site_copy.copy_markdown("explore", "feature_intro", path=marked, fields=fields)
        == "⟨explore.feature_intro⟩"
    )


def test_theme_templates_read_site_labels_as_escaped_text_and_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(
        tmp_path / "copy.yml",
        {"site": {"search": PLACEHOLDER, "copy_code": 'Copy "it" & go'}},
    )
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)

    class Environment:
        globals: dict[str, object] = {}

    env = site_copy.on_env(Environment())
    assert env.globals["site_copy_text"]("copy_code") == "Copy &quot;it&quot; &amp; go"
    assert env.globals["site_copy_span"]("search") == (
        '<span class="dp-authored-copy dp-copy-slot" '
        'data-copy-slot="site.search">⟨site.search⟩</span>'
    )


@pytest.mark.parametrize(
    "render",
    [
        site_copy.copy_text,
        site_copy.copy_html,
        site_copy.copy_span,
        site_copy.copy_markdown,
    ],
)
def test_a_caller_naming_no_known_slot_is_refused(render) -> None:
    """A misspelt slot fails rather than rendering as a blank one."""
    with pytest.raises(ValueError, match=r"explore\.no_such_slot is not a slot"):
        render("explore", "no_such_slot")


def test_the_carried_copy_survives_a_page_s_markdown(tmp_path: Path) -> None:
    """A page whose line opens inline is read as Markdown, attributes and all.

    The carried slots keep their own Markdown: no link, emphasis or code span
    is made of them, so the scripts read back exactly what the file holds.
    """
    import markdown

    text = "See [the paper](https://x.org/p), *not* `code` or _this_ (ok)."
    path = _write(tmp_path / "copy.yml", {"shared": {"threshold_help": text}})
    attribute = site_copy.copy_attribute("shared", path=path)
    page = markdown.markdown(f"<span>Preview</span><div hidden{attribute}></div>")
    carried = re.search(r'data-site-copy="([^"]*)"', page)[1]
    assert json.loads(html.unescape(carried)) == {"shared": {"threshold_help": text}}


def test_a_link_in_copy_survives_a_page_that_passes_through_markdown(
    tmp_path: Path,
) -> None:
    """The scripts' copy rides in an attribute; Markdown must not open a link in it."""
    import markdown
    from html.parser import HTMLParser

    text = "Known as a performance profile ([Dolan and Moré, 2002](https://doi.org/10.1007/s101070100263))."
    path = _write(tmp_path / "copy.yml", {"shared": {"threshold_help": text}})
    rendered = markdown.markdown(
        # An inline element opening the line, as a preview build's badge
        # does, makes the whole line one Markdown paragraph.
        '<span class="badge">Preview</span>'
        f'<div class="x"{site_copy.copy_attribute("shared", path=path)}></div>',
        extensions=["md_in_html"],
    )

    found: list[str] = []

    class Reader(HTMLParser):
        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            found.extend(
                value or "" for name, value in attrs if name == "data-site-copy"
            )

    Reader().feed(rendered)
    assert json.loads(found[0])["shared"]["threshold_help"] == text
