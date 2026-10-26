"""T18's baselines table, which a Benchmarked task's Metrics page ends on.

The catalogue lists every baseline of the task, from the library's declared
per-instance values and the protocol's external and computed baselines.
"""

from __future__ import annotations

import html
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dpstats.task import Task
from dpstats.tasks import (
    LinearRegression,
    SingleCoefficientEstimation,
    list_built_in_metric_arguments,
)

import site_copy
from baseline_catalogue import baseline_catalogue
from results_roots import (
    BEST_MEDIAN,
    ExternalBaseline,
    ReleasedBaseline,
    ReleasedComparison,
    ReleasedGrid,
)
from templates import library_source
from templates.render import render
from templates.spec import TEMPLATES

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


def _page(
    comparison: ReleasedComparison = SCE,
    task: Task = SingleCoefficientEstimation,
    explored: tuple[str, ...] | None = None,
) -> str:
    record = replace(
        library_source.metrics(task),
        baselines=baseline_catalogue(task, comparison, explored),
    )
    return render(TEMPLATES["T18"], record)


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
                    "baseline_best_median": "{aggregation} over {pool}",
                    "baseline_best_of": "{aggregation} over algorithms {algorithms}",
                    "baseline_algorithm": "{aggregation} of {uuid}",
                    "role_headline": "Headline baseline",
                    "role_offered": "Offered on Explore",
                },
            }
        )
    )
    monkeypatch.setattr(site_copy, "SITE_COPY_PATH", path)


def test_the_metrics_page_ends_on_the_baselines_table() -> None:
    page = _page()
    assert page.rstrip().rsplit("\n## ", 1)[1].startswith("Baselines\n")


def test_a_page_without_a_release_has_no_baselines_table() -> None:
    page = render(TEMPLATES["T18"], library_source.metrics(SingleCoefficientEstimation))
    assert "## Baselines" not in page
    assert 'id="baseline-' not in page


def test_the_catalogue_lists_every_baseline_headline_first() -> None:
    page = _page()
    order = [part.split('"', 1)[0] for part in page.split('id="baseline-')[1:]]
    assert order == ["published_se", "ols_se", BEST_MEDIAN, "none"]


def test_an_external_baseline_shows_the_protocols_declaration() -> None:
    row = _row(_page(), "published_se")
    assert "⟨protocol.declared_protocol⟩" in row
    assert html.escape(PUBLISHED) in row
    assert '<a href="#absolute-error-to-ols">' in row
    assert '<a href="#absolute-deviation-to-median">' not in row
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
        assert f'<a href="#{anchor}">' in row
    # A declared value no Metric names as its baseline is never offered, so
    # it has no row.
    assert 'id="baseline-ols_coef"' not in page


def test_the_computed_baselines_apply_to_every_metric() -> None:
    page = _page()
    pooled = _row(page, BEST_MEDIAN)
    assert 'data-copy-slot="protocol.baseline_best_median"' in pooled
    # Every library Metric is offered it when no Explore list narrows them.
    assert '<a href="#sign-disagreement-with-ols">' in pooled
    assert "data-copy-slot" not in _row(page, "none")


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
    page = _page(comparison, LinearRegression)
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
    page = _page(explored=("absolute_error_to_ols",))
    table = page.split("## Baselines", 1)[1]
    assert "#sign-disagreement-with-ols" not in table
    assert "#absolute-deviation-to-median" not in _row(page, "ols_se")
    assert '<a href="#absolute-error-to-ols">' in _row(page, BEST_MEDIAN)
