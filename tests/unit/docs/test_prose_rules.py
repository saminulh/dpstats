"""STYLE.md S38 over the authored problem formulations.

S38 reads the maintainer-owned ``docs/content/tasks/*/problem-formulation.md``
files and never edits them: a task whose file misses a rule is a known gap,
listed below with the fix the maintainer owes, and its test reports xfail so the
entry is removed when the file is fixed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[3] / "docs"
sys.path.insert(0, str(DOCS / "site"))

from site_links import PUBLIC_SITE_URL  # noqa: E402

FORMULATIONS = {
    path.parent.name: path
    for path in sorted((DOCS / "content" / "tasks").glob("*/problem-formulation.md"))
}

# Maintainer-owned files (docs/content is embargoed): each gap names the rule
# the file misses. Remove the entry when the maintainer's diff fixes the file.
S38_GAPS = {
    "covariance_estimation": {"display math", "baseline", "metrics link"},
    "logistic_regression": {"baseline", "metrics link"},
    "m_estimation": {"baseline", "metrics link"},
    "median_estimation": {"baseline", "metrics link"},
    "simple_linear_regression": {"baseline", "metrics link"},
}


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _gap(task: str, check: str) -> None:
    if check in S38_GAPS.get(task, set()):
        pytest.xfail(f"{task} problem formulation misses {check}: maintainer fix")


def test_problem_formulations_are_discovered() -> None:
    assert len(FORMULATIONS) >= 8


@pytest.mark.parametrize("task", FORMULATIONS)
def test_problem_formulation_opens_with_its_heading_and_gives_display_math(
    task: str,
) -> None:
    """S38: opens ``## Problem formulation`` and has a display-math block."""
    text = FORMULATIONS[task].read_text(encoding="utf-8")
    assert text.startswith("## Problem formulation\n"), task
    _gap(task, "display math")
    assert re.search(r"\$\$.+?\$\$", text, re.DOTALL), f"{task} has no display math"


@pytest.mark.parametrize("task", FORMULATIONS)
def test_problem_formulation_names_its_metric_and_baseline(task: str) -> None:
    """S38: one paragraph names the baseline beside the error it normalizes."""
    _gap(task, "baseline")
    text = FORMULATIONS[task].read_text(encoding="utf-8")
    assert any(
        "baseline" in p.lower() and re.search(r"error|distance", p, re.IGNORECASE)
        for p in _paragraphs(text)
    ), f"{task} names no metric and baseline"


@pytest.mark.parametrize("task", FORMULATIONS)
def test_problem_formulation_links_the_tasks_metrics_page(task: str) -> None:
    """S38: a link to ``<site>/<task>/metrics/``."""
    _gap(task, "metrics link")
    text = FORMULATIONS[task].read_text(encoding="utf-8")
    assert f"]({PUBLIC_SITE_URL}{task}/metrics/" in text, task
