"""Tests for the reader-term definitions loader."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reader_terms import READER_TERMS, READER_TERMS_PATH, load_reader_terms

PANEL_DATA_JS = (
    Path(__file__).resolve().parents[3] / "docs/site/docs/javascripts/panel-data.js"
)


def test_an_absent_file_defines_no_terms(tmp_path: Path) -> None:
    assert load_reader_terms(tmp_path / "missing.yml") == {}


def test_the_default_path_is_the_maintainer_file() -> None:
    assert READER_TERMS_PATH.parts[-3:] == ("docs", "content", "reader-terms.yml")


def test_a_partial_file_defines_only_the_terms_it_lists(tmp_path: Path) -> None:
    path = tmp_path / "terms.yml"
    path.write_text("trial:\n  text: One trial.\n", encoding="utf-8")
    assert load_reader_terms(path) == {"trial": {"text": "One trial."}}


def test_a_well_formed_file_keeps_each_text_and_optional_link(tmp_path: Path) -> None:
    path = tmp_path / "terms.yml"
    path.write_text(
        "instance:\n  text: One dataset.\n  link: mean_estimation/instances/\n"
        "tau:\n  text: The threshold.\n"
        "success_rate:\n  text: The share of successful trials.\n"
        "performance_ratio:\n  text: Error over the baseline.\n",
        encoding="utf-8",
    )
    assert load_reader_terms(path) == {
        "instance": {"text": "One dataset.", "link": "mean_estimation/instances/"},
        "tau": {"text": "The threshold."},
        "success_rate": {"text": "The share of successful trials."},
        "performance_ratio": {"text": "Error over the baseline."},
    }


@pytest.mark.parametrize(
    "body",
    [
        "- trial\n",
        "trial: plain\n",
        "trial:\n  link: x/\n",
        "seed:\n  text: A retired term.\n",
        "run:\n  text: A retired term.\n",
    ],
)
def test_a_malformed_file_fails_loudly(tmp_path: Path, body: str) -> None:
    path = tmp_path / "terms.yml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_reader_terms(path)


def test_the_term_set_is_the_one_the_chart_readings_name() -> None:
    assert READER_TERMS == (
        "instance",
        "trial",
        "metric",
        "baseline",
        "tau",
        "performance_ratio",
        "success_rate",
        "privacy_interpretation",
    )
    named = set(re.findall(r'term: "(\w+)"', PANEL_DATA_JS.read_text("utf-8")))
    assert named <= set(READER_TERMS)
    assert {"instance", "trial", "tau", "success_rate"} <= named
