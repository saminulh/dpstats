"""Tests for the chart chrome both results surfaces share."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest

import site_copy

from profile_chart import (
    DIAGNOSTIC_FAMILY,
    OTHER_FAMILY,
    absence_body,
    algorithm_family,
    algorithm_baseline_href,
    budget_family_href,
    default_metric,
    family_slot,
    results_section,
)
from tests.unit.docs.conftest import parse_html


def test_algorithm_baseline_help_links_to_its_current_api() -> None:
    assert algorithm_baseline_href(2) == (
        "../../library/reference/analysis/" "dpstats.analysis.compute_baseline/"
    )


def test_a_page_opens_on_the_headline_metric_the_protocol_ranks() -> None:
    assert (
        default_metric(
            "absolute_error_to_ols",
            ["sign_disagreement_with_ols", "absolute_error_to_ols"],
        )
        == "absolute_error_to_ols"
    )
    assert (
        default_metric("residual_rmse", ["residual_rmse", "l2_error_to_ols"])
        == "residual_rmse"
    )


def test_a_root_that_cannot_supply_the_headline_metric_opens_on_the_first_one() -> None:
    """The selector still names every metric it supplies and why the headline
    one is absent, so the page opening elsewhere loses nothing."""
    assert (
        default_metric("absolute_error_to_ols", ["sign_disagreement_with_ols"])
        == "sign_disagreement_with_ols"
    )


def test_a_root_supplying_no_metric_has_no_page_to_open() -> None:
    with pytest.raises(ValueError, match="No metric is supplied"):
        default_metric("absolute_error_to_ols", [])


def test_a_family_takes_one_palette_slot_wherever_its_mechanisms_appear() -> None:
    family = algorithm_family("TukeyEM")
    assert algorithm_family("GridExponentialMechanism") == family
    assert family_slot(family) > 0
    assert family_slot(algorithm_family("AdaSSP")) not in (0, family_slot(family))


def test_an_unmapped_class_is_other_rather_than_guessed_into_a_family() -> None:
    assert algorithm_family("SomeUnreleasedMechanism") == OTHER_FAMILY
    assert family_slot(OTHER_FAMILY) == 0


def test_a_diagnostic_class_takes_no_palette_slot() -> None:
    assert algorithm_family("OLS") == DIAGNOSTIC_FAMILY
    assert family_slot(DIAGNOSTIC_FAMILY) == 0


@pytest.fixture
def written_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The results slots written with every field, so the page shows each value."""
    path = tmp_path / "site-copy.yml"
    path.write_text(
        json.dumps(
            {
                "shared": {
                    "results_unavailable_lead": "{task} cannot be displayed from {identity}.",
                    "profile_question_individual": "How often does each algorithm succeed?",
                    "profile_formula_individual": "{success_rate}: share of {trials} "
                    "with {metric} ≤ {tau}{baseline}",
                    "term_success_rate": "Success rate",
                    "term_trial": "trial",
                    "term_trials": "trials",
                    "term_instances": "problem instances",
                }
            }
        )
    )
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)


@pytest.mark.usefixtures("written_copy")
def test_absence_links_reasons_without_claiming_results_are_missing() -> None:
    """Absent inputs and incomparable measured results share a refusal surface."""
    body = absence_body(
        "Local collection work",
        "median_estimation",
        "unavailable.txt",
    )

    page = parse_html(body)
    assert "median_estimation cannot be displayed" in page.text
    assert "Local collection work" in page.text
    assert [link.attrs["href"] for link in page.find_all(tag="a")] == [
        "unavailable.txt"
    ]
    assert "carries no results" not in page.text


@pytest.mark.usefixtures("written_copy")
def test_absence_escapes_the_identity_it_names() -> None:
    body = absence_body("<b>x</b>", "mean_estimation", 'reasons".txt')

    assert "<b>x</b>" not in body
    assert "&lt;b&gt;x&lt;/b&gt;" in body
    assert 'href="reasons&quot;.txt"' in body


def _section(**options: object) -> str:
    return results_section(
        "chart",
        "l2_error",
        [("PureDPBudget(eps=1)", "<table></table>"), ("PureDPBudget(eps=2)", "")],
        label="L2 error",
        active=True,
        baseline_label="sample_mean_l2_se",
        **options,  # type: ignore[arg-type]
    )


def _text(markup: str, pattern: str) -> str:
    return re.sub(r"<[^>]+>", "", re.search(pattern, markup, re.S)[1])


@pytest.mark.usefixtures("written_copy")
def test_a_chart_heading_is_a_plain_question_above_a_formula_line() -> None:
    markup = _section(terms={})
    assert (
        _text(markup, r"dp-profile-heading[^>]*>(.*?)</h3>")
        == "How often does each algorithm succeed?"
    )
    assert (
        _text(markup, r"dp-profile-formula[^>]*>(.*?)</p>")
        == "Success rate: share of trials with L2 error ≤ τ × sample_mean_l2_se"
    )


@pytest.mark.usefixtures("written_copy")
def test_no_baseline_is_decided_by_its_key_not_its_name() -> None:
    against = _section(terms={}, baseline_key="sample_mean_l2_se")
    alone = _section(terms={}, baseline_key="none")
    formula = r"dp-profile-formula[^>]*>(.*?)</p>"
    assert _text(against, formula).endswith("× sample_mean_l2_se")
    assert "×" not in _text(alone, formula)


def test_reader_terms_ship_in_the_page_with_links_reaching_the_site_root() -> None:
    markup = _section(terms={"trial": {"text": "A trial.", "link": "x/"}}, depth=2)
    shipped = html.unescape(re.search(r'data-reader-terms="([^"]*)"', markup)[1])
    assert json.loads(shipped) == {"trial": {"text": "A trial.", "link": "../../x/"}}


def test_the_budget_family_label_links_to_its_api_page() -> None:
    markup = _section(terms={}, depth=2)
    assert (
        'dp-budget-family"><a href="../../library/reference/budgets/'
        'dpstats.budgets.PureDPBudget/">PureDPBudget</a>'
    ) in markup


def test_an_unknown_budget_family_renders_as_plain_text() -> None:
    assert budget_family_href(1, "NoSuchPolicy") == ""


def test_a_budget_parameter_every_budget_shares_is_stated_once_under_the_buttons() -> (
    None
):
    from profile_chart import budget_choice_labels, budget_chip_group

    fixed = [
        "ApproxDPBudget(eps=0.3, delta=1e-06)",
        "ApproxDPBudget(eps=1.0, delta=1e-06)",
    ]
    labels, shared = budget_choice_labels(fixed)
    assert list(labels.values()) == ["ε = 0.3", "ε = 1.0"] and shared == "δ = 1e-06"
    rendered = budget_chip_group(fixed, fixed[0], scope="t")
    assert 'role="radiogroup"' in rendered and "δ = 1e-06</p>" in rendered
    varying = [
        "ApproxDPBudget(eps=0.3, delta=1e-06)",
        "ApproxDPBudget(eps=1.0, delta=1e-05)",
    ]
    labels, shared = budget_choice_labels(varying)
    assert shared == "" and labels[varying[1]] == "ε = 1.0, δ = 1e-05"
