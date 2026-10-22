"""Public behavior of the trial-outcome histogram, against hand-computed oracles."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd
import pytest

from dpstats.analysis import bin_outcomes, plot_outcomes
from dpstats.budgets import PureDPBudget
from dpstats.instance import Instance
from dpstats.results import ResultStatus
from tests.registry_components import SimpleAlgorithm, SimpleMetadata


def _instance(name: str) -> Instance:
    return Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(name.encode()).hexdigest(),
        metric_arguments={},
        label=name,
    )


ALPHA = SimpleAlgorithm(label="Alpha")
BETA = SimpleAlgorithm(label="Beta", scale=2.0)
ONE, TEN, HUNDRED, UNMEASURED = (
    _instance(name) for name in ("one", "ten", "hundred", "none")
)
INSTANCES = [ONE, TEN, HUNDRED, UNMEASURED]
# Each Instance's property, and its weight: the hundred carries two of the
# five units, the others one each; the last has no value to place.
FEATURE = pd.Series(
    {ONE.uuid: 1.0, TEN.uuid: 10.0, HUNDRED.uuid: 100.0, UNMEASURED.uuid: math.nan}
)
WEIGHTS = pd.Series(
    {ONE.uuid: 1.0, TEN.uuid: 1.0, HUNDRED.uuid: 2.0, UNMEASURED.uuid: 1.0}
)
ERROR = ResultStatus.ALGORITHM_ERROR
SUCCESS = ResultStatus.SUCCESS


def _trials(
    *rows: tuple[SimpleAlgorithm, Instance, ResultStatus, float]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "toy",
                "algorithm": algorithm,
                "budget": PureDPBudget(eps=1.0),
                "instance": instance,
                "alg_seed": seed,
                "status": status,
                "cost": cost,
            }
            for seed, (algorithm, instance, status, cost) in enumerate(rows)
        ]
    )


# Alpha: half its trials within a threshold of one on the first Instance;
# one success beside two errors on the ten; beyond it on the hundred.
# Beta: within on the first, half within on the hundred, never run on the ten.
TRIALS = _trials(
    (ALPHA, ONE, SUCCESS, 0.5),
    (ALPHA, ONE, SUCCESS, 2.0),
    (ALPHA, TEN, SUCCESS, 0.5),
    (ALPHA, TEN, ERROR, math.inf),
    (ALPHA, TEN, ERROR, math.inf),
    (ALPHA, HUNDRED, SUCCESS, 3.0),
    (ALPHA, UNMEASURED, SUCCESS, 0.1),
    (BETA, ONE, SUCCESS, 0.5),
    (BETA, ONE, SUCCESS, 0.5),
    (BETA, HUNDRED, SUCCESS, 0.5),
    (BETA, HUNDRED, SUCCESS, 5.0),
)
# Costs read in their own units.
BASELINE = pd.DataFrame(
    {"task": "toy", "instance": INSTANCES, "baseline": [1.0] * len(INSTANCES)}
)


def _bins(**chosen: Any) -> pd.DataFrame:
    settings: dict[str, Any] = {
        "metric": "cost",
        "tau": 1.0,
        "algorithm": ALPHA.uuid,
        "compare_with": None,
        "feature": FEATURE,
        "weights": WEIGHTS,
        "instances": [instance.uuid for instance in INSTANCES],
        "scale": "log",
        "bins": 2,
        "aggregation": "individual",
        **chosen,
    }
    return bin_outcomes(TRIALS, BASELINE, **settings)


def test_each_trial_takes_its_share_of_its_instances_weight() -> None:
    """Of five units: the one's 1 splits within and beyond; the ten's 1 is a
    third within and two thirds errors; the hundred's 2 are beyond; the
    unmeasured one's 1 is in no bin."""
    bins = _bins()
    assert bins["success_within"].tolist() == pytest.approx([0.1, 1 / 15])
    assert bins["success_beyond"].tolist() == pytest.approx([0.1, 0.4])
    assert bins["algorithm_error"].tolist() == pytest.approx([0.0, 2 / 15])
    assert bins["timeout"].tolist() == [0.0, 0.0]
    assert bins["instances"].tolist() == [[ONE.uuid], [TEN.uuid, HUNDRED.uuid]]
    shares = bins.drop(columns=["start", "stop", "lower", "upper", "instances"])
    assert 1 - shares.to_numpy().sum() == pytest.approx(0.2)


def test_an_instance_the_baseline_does_not_cover_is_in_no_bin() -> None:
    """Without a baseline on the ten, its unit of weight leaves the second
    bin: the hundred's 2 of 5 are beyond, and 2 of 5 are in no bin."""
    covered = BASELINE[[instance is not TEN for instance in BASELINE["instance"]]]
    bins = bin_outcomes(
        TRIALS,
        covered,
        metric="cost",
        tau=1.0,
        algorithm=ALPHA.uuid,
        compare_with=None,
        feature=FEATURE,
        weights=WEIGHTS,
        instances=[instance.uuid for instance in INSTANCES],
        scale="log",
        bins=2,
        aggregation="individual",
    )
    assert bins["success_within"].tolist() == pytest.approx([0.1, 0.0])
    assert bins["success_beyond"].tolist() == pytest.approx([0.1, 0.4])
    assert bins["algorithm_error"].tolist() == [0.0, 0.0]
    assert bins["instances"].tolist() == [[ONE.uuid], [HUNDRED.uuid]]


def test_a_log_scale_needs_no_weighted_value_to_rank() -> None:
    """With no Instance weighted, nothing is ranked, and log bins still read."""
    bins = _bins(weights=pd.Series(dtype=float), instances=[ONE.uuid])
    assert bins["instances"].tolist() == [[ONE.uuid], []]


def test_log_bins_split_the_logarithm_of_the_placed_values_evenly() -> None:
    """log10 of 1 and 100 are 0 and 2: two bins meet at 10, which opens the
    second."""
    bins = _bins()
    assert bins["lower"].tolist() == pytest.approx([1.0, 10.0])
    assert bins["upper"].tolist() == pytest.approx([10.0, 100.0])
    assert bins["start"].tolist() == pytest.approx([1.0, 10.0])


def test_linear_bins_split_the_values_evenly() -> None:
    """From 1 to 100 in two: the edge at 50.5 puts the ten in the first bin."""
    bins = _bins(scale="linear")
    assert bins["upper"].tolist() == pytest.approx([50.5, 100.0])
    assert bins["instances"].tolist() == [[ONE.uuid, TEN.uuid], [HUNDRED.uuid]]


def test_a_median_goes_whole_to_one_side_of_the_threshold() -> None:
    """The one's median 1.25 is beyond; the ten's median is infinite, so its
    weight splits by its trials, the success counting as beyond."""
    bins = _bins(aggregation="median")
    assert bins["success_within"].tolist() == [0.0, 0.0]
    assert bins["success_beyond"].tolist() == pytest.approx([0.2, 1 / 15 + 0.4])
    assert bins["algorithm_error"].tolist() == pytest.approx([0.0, 2 / 15])


def test_no_trial_is_within_a_threshold_that_is_not_positive() -> None:
    bins = _bins(tau=0.0)
    assert bins["success_within"].tolist() == [0.0, 0.0]


def test_a_pair_splits_by_the_two_success_rates() -> None:
    """On the one, Alpha's rate is 1/2 and Beta's 1: both 1/2, Beta alone
    1/2. On the hundred, 0 and 1/2: Beta alone 1/2, neither 1/2. The ten,
    which Beta never ran on, is in no bin."""
    bins = _bins(compare_with=BETA.uuid)
    assert list(bins.columns[4:-1]) == ["both", "a_only", "b_only", "neither"]
    assert bins["both"].tolist() == pytest.approx([0.1, 0.0])
    assert bins["a_only"].tolist() == pytest.approx([0.0, 0.0])
    assert bins["b_only"].tolist() == pytest.approx([0.1, 0.2])
    assert bins["neither"].tolist() == pytest.approx([0.0, 0.2])
    assert bins["instances"].tolist() == [[ONE.uuid], [HUNDRED.uuid]]


def test_a_median_pair_reads_each_median_against_the_threshold() -> None:
    """Medians 1.25 and 0.5 on the one, 3 and 2.75 on the hundred: Beta alone
    on the first, neither on the second."""
    bins = _bins(compare_with=BETA.uuid, aggregation="median")
    assert bins["b_only"].tolist() == pytest.approx([0.2, 0.0])
    assert bins["neither"].tolist() == pytest.approx([0.0, 0.4])


def test_quantile_bins_hold_equal_weight_of_the_ranked_instances() -> None:
    """The ranked weight is 1, 1 and 2 of 4, so the tie groups end at 1/4,
    1/2 and 1: two bins meet at 1/2."""
    bins = _bins(scale="quantile")
    assert bins["start"].tolist() == pytest.approx([0.0, 0.5])
    assert bins["stop"].tolist() == pytest.approx([0.5, 1.0])
    assert bins["instances"].tolist() == [[ONE.uuid, TEN.uuid], [HUNDRED.uuid]]
    assert bins["lower"].tolist() == [1.0, 100.0]


def test_a_quantile_edge_moves_to_the_nearest_tie_group_end() -> None:
    """In four bins the edge at 3/4 is as near 1/2 as 1 and moves to the
    first, so the third bin is empty and the hundred is whole in the last."""
    bins = _bins(scale="quantile", bins=4)
    assert bins["start"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.5])
    assert bins["instances"].tolist() == [[ONE.uuid], [TEN.uuid], [], [HUNDRED.uuid]]


# ── The stacked figure ──────────────────────────────────────────────────────

# The figure's texts, as a caller writes them.
TEXTS = {
    "xlabel": "n",
    "ylabel": "share",
    "success_hover": "within: {share}",
    "stack_hover": "stack: {share}",
    "pair_hover": "A {a} · B {b}",
}
CATEGORIES = {
    "success_within": "≤ τ",
    "success_beyond": "> τ",
    "algorithm_error": "Error",
    "not_run": "Not run",
    "both": "Both",
    "b_only": "B only",
}
TEMPLATE = {
    "layout": {
        "meta": {
            "ink": "#111111",
            "outcomes": {
                "colors": {"success_within": "#0000ff", "algorithm_error": "#ff0000"},
                "hatched": ["algorithm_error"],
                "hatch": {"shape": "/"},
                "bargap": 0.05,
            },
        }
    }
}


def test_the_present_categories_stack_in_the_order_named() -> None:
    figure = plot_outcomes(_bins(), categories=CATEGORIES)
    assert [trace.name for trace in figure.data] == ["≤ τ", "&gt; τ", "Error"]
    assert figure.layout.barmode == "stack"
    assert figure.layout.yaxis.tickformat == ".0%"
    assert figure.layout.xaxis.title.text == "property"
    assert figure.layout.yaxis.title.text == "share of problem instances"


def test_a_log_bar_spans_its_bin_around_the_geometric_centre() -> None:
    """The first bin runs from 1 to 10; the axis opens on log10 0 to 2,
    padded by a twentieth of that span."""
    figure = plot_outcomes(_bins(), scale="log", padding=0.05, categories=CATEGORIES)
    bar = figure.data[0]
    assert bar.x[0] == pytest.approx(math.sqrt(10))
    assert bar.width[0] == pytest.approx(9.0)
    assert figure.layout.xaxis.range == pytest.approx((-0.1, 2.1))


def test_the_template_colours_and_hatches_each_category() -> None:
    figure = plot_outcomes(_bins(), categories=CATEGORIES, template=TEMPLATE)
    within, _, error = figure.data
    assert within.marker.color == "#0000ff"
    assert error.marker.pattern.shape == "/"
    assert error.marker.pattern.fgcolor == "#111111"
    assert figure.layout.bargap == 0.05


def test_a_bar_hover_names_its_range_and_shares() -> None:
    """The second bin holds 1/15 within of 0.6: a ninth of the bin."""
    figure = plot_outcomes(_bins(), categories=CATEGORIES, **TEXTS)
    assert (
        figure.data[0].hovertext[1]
        == "10–100<br>≤ τ: 6.7%<br>within: 11.1%<br>stack: 6.7%"
    )


def test_a_pair_hover_names_each_algorithms_share_within() -> None:
    """The first bin is both and Beta alone, half each: Alpha within on half,
    Beta on all of it."""
    figure = plot_outcomes(
        _bins(compare_with=BETA.uuid), categories=CATEGORIES, **TEXTS
    )
    assert "A 50.0% · B 100.0%" in figure.data[0].hovertext[0]


def test_a_quantile_axis_names_each_edge_by_its_value() -> None:
    figure = plot_outcomes(
        _bins(scale="quantile"), scale="quantile", categories=CATEGORIES
    )
    assert figure.layout.xaxis.range == (0, 1)
    assert figure.layout.xaxis.tickvals == pytest.approx((0.0, 0.5, 1.0))
    assert figure.layout.xaxis.ticktext == ("1", "100", "100")
