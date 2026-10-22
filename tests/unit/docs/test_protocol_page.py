"""T23, the per-task protocol-and-baselines page.

The standings statement comes from the protocol's ranking; the catalogue
lists every baseline of the task, from the library's declared per-instance
values and the protocol's external and computed baselines.
"""

from __future__ import annotations

import html
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dpstats.tasks import (
    LinearRegression,
    SingleCoefficientEstimation,
    list_built_in_metric_arguments,
)

import site_copy
from protocol_page import protocol_page
from results_roots import (
    BEST_MEDIAN,
    ExternalBaseline,
    ReleasedBaseline,
    ReleasedComparison,
    ReleasedGrid,
)
from templates.model import Link
from templates.render import render
from templates.spec import TEMPLATES

SITE_METHODOLOGY = Link("Benchmark protocol", "../../methodology/")

POOLED = ReleasedBaseline(
    BEST_MEDIAN, "best_median", (("pool", "standard"), ("aggregation", "median"))
)
PUBLISHED = "The standard error the study published for the coefficient."
SCE = ReleasedComparison(
    headline_metric="absolute_error_to_ols",
    headline_baseline="published_se",
    headline_definition=ReleasedBaseline("published_se", "reference"),
    headline_quantile=0.9,
    taus=(0.0, 10.0),
    grid=ReleasedGrid(0.0, 10.0, "linear", 1001),
    definitions={BEST_MEDIAN: POOLED},
    external={
        "published_se": ExternalBaseline(
            "published_se", PUBLISHED, ("absolute_error_to_ols",)
        )
    },
    weighting="provenance_ref",
    eligibility="standard",
)
METRICS = "../../single_coefficient_estimation/metrics/"


def _page(comparison: ReleasedComparison = SCE) -> str:
    record = protocol_page(
        SingleCoefficientEstimation,
        "single-coefficient estimation: methodology",
        comparison,
        "Distribution Release test",
        methodology=SITE_METHODOLOGY,
        depth=2,
    )
    return render(TEMPLATES["T23"], record)


def _row(page: str, name: str) -> str:
    """One catalogue row, by the baseline it lists."""
    return page.split(f'id="baseline-{name}">', 1)[1].split("</tr>")[0]


@pytest.fixture
def written_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Protocol slots written with every field, so the page shows each value."""
    path = tmp_path / "site-copy.yml"
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "ranking_rule": "rank [{headline}][headline] against "
                    "[{baseline}][baseline] at {percent}%.",
                    "baseline_best_median": "{aggregation} over {pool}",
                    "baseline_best_of": "{aggregation} over algorithms {algorithms}",
                    "baseline_algorithm": "{aggregation} of {uuid}",
                    "role_headline": "Headline baseline",
                    "role_offered": "Offered on Explore",
                },
                "shared": {"built_from": "{identity}"},
            }
        )
    )
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)


@pytest.mark.usefixtures("written_copy")
def test_the_page_states_the_headline_the_catalogue_and_the_grid() -> None:
    page = _page()
    assert [line for line in page.splitlines() if line.startswith("## ")] == [
        "## Standings",
        "## Baselines",
        "## Threshold grid",
    ]
    assert (
        f'rank <a href="{METRICS}#absolute-error-to-ols">Absolute error vs OLS</a> '
        'against <a href="#baseline-published_se">published_se</a>' in page
    )
    assert '<td class="dp-value">1001</td>' in page


def test_the_lead_opens_on_the_site_wide_methodology_page() -> None:
    """The page states the comparison; the site-wide page explains it."""
    lead = _page().split("## Standings", 1)[0]
    first = lead.split("\n\n")[2]
    assert first == '<p><a href="../../methodology/">Benchmark protocol</a></p>'


def test_the_catalogue_lists_every_baseline_headline_first() -> None:
    page = _page()
    order = [part.split('"', 1)[0] for part in page.split('id="baseline-')[1:]]
    assert order == ["published_se", "ols_se", BEST_MEDIAN, "none"]


def test_an_external_baseline_shows_the_protocols_declaration() -> None:
    row = _row(_page(), "published_se")
    assert "⟨protocol.declared_protocol⟩" in row
    assert html.escape(PUBLISHED) in row
    assert f'<a href="{METRICS}#absolute-error-to-ols">' in row
    assert f'<a href="{METRICS}#absolute-deviation-to-median">' not in row
    assert "⟨protocol.role_headline⟩" in row


def test_a_statistical_baseline_applies_to_the_metrics_that_name_it() -> None:
    page = _page()
    registry = list_built_in_metric_arguments(SingleCoefficientEstimation)
    row = _row(page, "ols_se")
    assert "⟨protocol.declared_library⟩" in row
    assert registry["ols_se"].description.rstrip(".") in row
    assert "⟨protocol.role_offered⟩" in row
    for anchor in (
        "absolute-error-to-ols",
        "absolute-error-of-median-to-ols",
        "absolute-deviation-to-median",
    ):
        assert f'<a href="{METRICS}#{anchor}">' in row
    # A declared value no Metric names as its baseline is never offered, so
    # it has no row.
    assert 'id="baseline-ols_coef"' not in page


def test_the_computed_baselines_apply_to_every_metric() -> None:
    page = _page()
    pooled = _row(page, BEST_MEDIAN)
    assert 'data-copy-slot="protocol.baseline_best_median"' in pooled
    # Every library Metric is offered it when no Explore list narrows them.
    assert f'<a href="{METRICS}#sign-disagreement-with-ols">' in pooled
    assert "data-copy-slot" not in _row(page, "none")


def test_the_standings_rules_are_stated_as_the_protocol_names_them() -> None:
    page = _page()
    weighting = "<td>⟨protocol.rule_weighting⟩</td>"
    assert f"{weighting}<td><code>provenance_ref</code></td>" in page
    assert "<td>⟨protocol.rule_eligibility⟩</td><td><code>standard</code></td>" in page
    assert "Pooling" not in page
    page = _page(replace(SCE, weighting="by_size"))
    assert f"{weighting}<td><code>by_size</code></td>" in page


@pytest.mark.usefixtures("written_copy")
def test_a_computed_headline_baseline_is_stated_in_its_own_words() -> None:
    roster = ReleasedBaseline(
        "roster", "best_of", (("algorithms", "a1, a2"), ("aggregation", "median"))
    )
    page = _page(
        replace(
            SCE,
            headline_baseline="roster",
            headline_definition=roster,
            definitions={BEST_MEDIAN: POOLED, "roster": roster},
        )
    )
    row = _row(page, "roster")
    assert "algorithms a1, a2" in row and "Headline baseline" in row
    assert "Offered on Explore" in _row(page, "published_se")


def test_a_statistical_headline_baseline_is_a_library_row() -> None:
    comparison = replace(
        SCE,
        headline_metric="prediction_rmse_to_ols",
        headline_baseline="ols_prediction_se",
        headline_definition=ReleasedBaseline("ols_prediction_se", "reference"),
        external={},
    )
    page = render(
        TEMPLATES["T23"],
        protocol_page(
            LinearRegression,
            "LR",
            comparison,
            "x",
            methodology=SITE_METHODOLOGY,
            depth=2,
        ),
    )
    registry = list_built_in_metric_arguments(LinearRegression)
    row = _row(page, "ols_residual_rmse")
    assert registry["ols_residual_rmse"].description.rstrip(".") in row
    assert "⟨protocol.declared_library⟩" in row
    assert "Residual RMSE" in row
    assert 'id="baseline-response_std"' not in page
    assert "⟨protocol.role_headline⟩" in _row(page, "ols_prediction_se")


def test_a_metric_the_library_does_not_define_is_refused() -> None:
    with pytest.raises(ValueError, match="does not define"):
        _page(replace(SCE, headline_metric="invented"))
    comparison = replace(
        SCE,
        external={
            "published_se": ExternalBaseline("published_se", PUBLISHED, ("invented",))
        },
    )
    with pytest.raises(ValueError, match="does not define"):
        _page(comparison)


def test_a_metric_explore_does_not_list_is_never_advertised() -> None:
    """Only the Metrics Explore lists appear under Applies to."""
    record = protocol_page(
        SingleCoefficientEstimation,
        "SCE",
        SCE,
        "x",
        methodology=SITE_METHODOLOGY,
        depth=2,
        explored=("absolute_error_to_ols",),
    )
    page = render(TEMPLATES["T23"], record)
    assert "#sign-disagreement-with-ols" not in page
    assert "#absolute-deviation-to-median" not in _row(page, "ols_se")
    assert f'<a href="{METRICS}#absolute-error-to-ols">' in _row(page, BEST_MEDIAN)
