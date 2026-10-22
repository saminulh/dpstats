"""Public behavior of the profile transforms, against hand-computed oracles."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable
from operator import attrgetter
from typing import Any

import pandas as pd
import pytest

from dpstats.algorithm import PrivacyInterpretation
from dpstats.budgets import NO_PRIVACY, DPBudget, PureDPBudget
from dpstats.instance import Instance
from dpstats.metric import Metric
from tests.registry_components import (
    NonprivateAlgorithm,
    SimpleAlgorithm,
    SimpleMetadata,
    SIMPLE_TASK,
)
from dpstats.analysis import (
    aggregate_trials,
    compute_baseline,
    compute_error_ratio,
    compute_profile_curves,
)


class OtherReference(NonprivateAlgorithm):
    """A second configured reference identity for baseline selection tests."""


def _algorithm(label: str) -> SimpleAlgorithm:
    """One configured algorithm, told apart by the name it goes by."""
    return SimpleAlgorithm(label=label, scale=float(sum(label.encode()) + 1))


def _instance(name: str) -> Instance:
    """One Instance, told apart by the problem its digest stands for."""
    return Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(name.encode()).hexdigest(),
        metric_arguments={},
        label=name,
    )


def _measurement(
    *,
    algorithm: str,
    instance: str,
    value: float,
    alg_seed: int = 0,
    budget: DPBudget = PureDPBudget(eps=1.0),
) -> dict[str, Any]:
    """One measurement row, spelled out independently of the producer."""
    return {
        "task": "toy",
        "metric": "cost",
        "instance": _instance(instance),
        "algorithm": _algorithm(algorithm),
        "budget": budget,
        "alg_seed": alg_seed,
        "value": value,
    }


def _reference_row(
    *,
    instance: str,
    value: float,
    alg_seed: int = 0,
    label: str = "Reference",
) -> dict[str, Any]:
    """One no-privacy row of an algorithm registered as a nonprivate reference."""
    return {
        "task": "toy",
        "metric": "cost",
        "instance": _instance(instance),
        "algorithm": NonprivateAlgorithm(label=label),
        "budget": NO_PRIVACY,
        "alg_seed": alg_seed,
        "value": value,
    }


def _baseline(instance: str, value: float) -> dict[str, Any]:
    return {
        "task": "toy",
        "metric": "cost",
        "instance": _instance(instance),
        "baseline": value,
    }


# The ruled division table: cost, baseline, and the normalized value each pair
# must produce.
_DIVISION_CASES = [
    (0.0, 0.0, 1.0),
    (3.0, 0.0, math.inf),
    (math.inf, 0.0, math.inf),
    (math.inf, 2.0, math.inf),
    (2.0, 4.0, 0.5),
    (0.0, 4.0, 0.0),
    (5.0, math.inf, 0.0),
    # A measured negative infinity is a legitimate best cost, not a failure.
    (-math.inf, 2.0, -math.inf),
    # A NaN carries no order, so it normalizes to the worst score, and so
    # does a quotient of infinities that no order can rank.
    (-math.inf, -math.inf, math.inf),
]


@pytest.mark.parametrize(("cost", "reference", "expected"), _DIVISION_CASES)
def test_normalization_follows_the_ruled_division_table(
    cost: float, reference: float, expected: float
) -> None:
    frame = pd.DataFrame([_measurement(algorithm="A", instance="i1", value=cost)])
    baseline = pd.DataFrame([_baseline("i1", reference)])

    normalized = compute_error_ratio(frame, baseline, metric="value")

    assert normalized["value"].tolist() == [pytest.approx(expected)]


def test_normalization_broadcasts_a_baseline_across_the_keys_it_omits() -> None:
    """A baseline keyed by Instance alone covers every algorithm and budget."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0),
            _measurement(
                algorithm="B",
                instance="i1",
                value=8.0,
                budget=PureDPBudget(eps=2.0),
            ),
        ]
    )

    normalized = compute_error_ratio(
        frame, pd.DataFrame([_baseline("i1", 4.0)]), metric="value"
    )

    assert normalized["value"].tolist() == [pytest.approx(0.5), pytest.approx(2.0)]
    assert normalized["baseline"].tolist() == [4.0, 4.0]


def test_normalization_refuses_rows_already_carrying_a_baseline() -> None:
    frame = pd.DataFrame([_measurement(algorithm="A", instance="i1", value=2.0)])
    baseline = pd.DataFrame([_baseline("i1", 4.0)])
    normalized = compute_error_ratio(frame, baseline, metric="value")

    with pytest.raises(ValueError, match="already carries a baseline"):
        compute_error_ratio(normalized, baseline, metric="value")


def test_a_baseline_must_cover_every_row_exactly_once() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0),
            _measurement(algorithm="A", instance="i2", value=2.0),
        ]
    )

    with pytest.raises(ValueError, match="have no baseline"):
        compute_error_ratio(frame, pd.DataFrame([_baseline("i1", 1.0)]), metric="value")

    twice = pd.DataFrame([_baseline("i1", 1.0), _baseline("i1", 2.0)])
    with pytest.raises(ValueError, match="more than one baseline"):
        compute_error_ratio(frame, twice, metric="value")


def test_trial_aggregation_orders_infinite_costs_instead_of_dropping_them() -> None:
    """The middle of 1, 2, inf is 2; a cell failing half its runs is infinite."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=1),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=2),
            _measurement(algorithm="B", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="B", instance="i1", value=math.inf, alg_seed=1),
        ]
    )

    aggregated = aggregate_trials(frame, metric="value")

    labels = [str(algorithm) for algorithm in aggregated["algorithm"]]
    assert dict(zip(labels, aggregated["value"])) == {"A": 2.0, "B": math.inf}
    assert "alg_seed" not in aggregated.columns


def test_trial_aggregation_takes_any_aggregation_pandas_can_apply() -> None:
    """One failed run of two: every aggregation keeps its infinity in the order."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=1),
        ]
    )

    assert aggregate_trials(frame, aggregation="min", metric="value")[
        "value"
    ].tolist() == [1.0]
    assert aggregate_trials(frame, aggregation="max", metric="value")[
        "value"
    ].tolist() == [math.inf]
    # Aggregations that are not order statistics are accepted just the same.
    assert aggregate_trials(frame, aggregation="mean", metric="value")[
        "value"
    ].tolist() == [math.inf]
    assert aggregate_trials(frame, aggregation="count", metric="value")[
        "value"
    ].tolist() == [2]
    assert aggregate_trials(
        frame, aggregation=lambda costs: costs.min(), metric="value"
    )["value"].tolist() == [1.0]
    with pytest.raises(ValueError, match="is not an aggregation"):
        aggregate_trials(frame, aggregation="not_an_aggregation", metric="value")


def test_baseline_by_algorithm_without_per_budget_takes_one_algorithm_median_on_each_instance() -> (
    None
):
    """One algorithm\'s own runs are the baseline, whatever budget they ran under.

    On i1 that algorithm costs 2, 6 and 99, whose median is 6; another
    algorithm\'s 99 is not its run and does not enter. With ``per_budget=False``
    the baseline is keyed by Instance alone, so it broadcasts across every
    budget compared against it.
    """
    frame = pd.DataFrame(
        [
            _measurement(
                algorithm="OLS",
                instance="i1",
                value=2.0,
                alg_seed=0,
                budget=NO_PRIVACY,
            ),
            _measurement(
                algorithm="OLS",
                instance="i1",
                value=6.0,
                alg_seed=1,
                budget=NO_PRIVACY,
            ),
            _measurement(algorithm="OLS", instance="i2", value=1.0, budget=NO_PRIVACY),
            _measurement(algorithm="OLS", instance="i1", value=99.0),
            _measurement(
                algorithm="Private", instance="i1", value=99.0, budget=NO_PRIVACY
            ),
        ]
    )

    baseline = compute_baseline(
        frame, baseline=_algorithm("OLS"), per_budget=False, metric="value"
    )

    assert set(baseline.columns) == {"task", "instance", "baseline"}
    named = [instance.label for instance in baseline["instance"]]
    assert dict(zip(named, baseline["baseline"])) == {"i1": 6.0, "i2": 1.0}
    with pytest.raises(ValueError, match="no rows for reference algorithm"):
        compute_baseline(frame, baseline=_algorithm("Absent"), metric="value")
    with pytest.raises(TypeError, match="no stored key 'Absent'"):
        compute_baseline(frame, baseline="Absent", metric="value")  # type: ignore[arg-type]


def test_baseline_by_algorithm_matches_one_algorithm_on_instance_and_budget() -> None:
    """Hand-computed: A costs 2 and 6 at eps=1 on i1, so its median there is 4.

    Unlike the per-Instance baseline above, the default ``per_budget=True``
    keys by budget too, so a private budget and a no-privacy one each
    keep their own row.
    """
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=6.0, alg_seed=1),
            _measurement(algorithm="A", instance="i2", value=8.0),
            _measurement(
                algorithm="A",
                instance="i1",
                value=1.0,
                budget=PureDPBudget(eps=2.0),
            ),
            _measurement(algorithm="A", instance="i1", value=0.25, budget=NO_PRIVACY),
            _measurement(algorithm="B", instance="i1", value=99.0),
        ]
    )

    baseline = compute_baseline(frame, baseline=_algorithm("A"), metric="value")

    assert set(baseline.columns) == {
        "task",
        "instance",
        "budget",
        "baseline",
    }
    keyed = {
        (instance.label, getattr(budget, "eps", None)): value
        for instance, budget, value in zip(
            baseline["instance"], baseline["budget"], baseline["baseline"]
        )
    }
    assert keyed == {
        ("i1", 1.0): 4.0,
        ("i2", 1.0): 8.0,
        ("i1", 2.0): 1.0,
        ("i1", None): 0.25,
    }


def test_baseline_by_algorithm_reads_another_algorithm_as_the_baseline() -> None:
    """B against A on the same Instance and budget: 99 over 4, and A against itself."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=6.0, alg_seed=1),
            _measurement(algorithm="B", instance="i1", value=99.0),
        ]
    )

    normalized = compute_error_ratio(
        frame,
        compute_baseline(frame, baseline=_algorithm("A"), metric="value"),
        metric="value",
    )

    assert normalized["value"].tolist() == [
        pytest.approx(0.5),
        pytest.approx(1.5),
        pytest.approx(24.75),
    ]


def test_baseline_of_a_nonprivate_algorithm_is_that_algorithms_own_cost() -> None:
    """Hand-computed: the reference costs 2 and 4 on i1, so its median is 3.

    The ranked pool's own cheaper cost is not the baseline, so a private cost
    of 1 on that Instance reads as a third of what no privacy spends there.
    """
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=2.0, alg_seed=0),
            _reference_row(instance="i1", value=4.0, alg_seed=1),
            _measurement(algorithm="A", instance="i1", value=1.0),
        ]
    )

    baseline = compute_baseline(
        frame, baseline=frame.iloc[0]["algorithm"], per_budget=False, metric="value"
    )

    assert set(baseline.columns) == {"task", "instance", "baseline"}
    assert baseline["baseline"].tolist() == [3.0]
    assert compute_error_ratio(frame, baseline, metric="value")["value"].tolist() == [
        pytest.approx(2 / 3),
        pytest.approx(4 / 3),
        pytest.approx(1 / 3),
    ]


def test_baseline_none_leaves_every_cost_in_its_own_unit() -> None:
    """A cost already expressed in an external reference unit is its own baseline."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0),
            _measurement(algorithm="A", instance="i2", value=6.0),
        ]
    )

    baseline = compute_baseline(frame, baseline=None, metric="value")

    assert set(baseline.columns) == {"task", "instance", "baseline"}
    assert baseline["baseline"].tolist() == [1.0, 1.0]
    assert compute_error_ratio(frame, baseline, metric="value")["value"].tolist() == [
        2.0,
        6.0,
    ]


@pytest.mark.parametrize("spelled", ["unknown_mode", "standard", "published_se"])
def test_compute_baseline_refuses_a_string_and_names_where_a_key_goes(
    spelled: str,
) -> None:
    """A string is neither an algorithm nor a class; a stored key is normalized directly."""
    frame = pd.DataFrame([_measurement(algorithm="A", instance="i1", value=1.0)])

    with pytest.raises(
        TypeError, match=f"no stored key '{spelled}'.*compute_error_ratio"
    ):
        compute_baseline(frame, baseline=spelled, metric="value")  # type: ignore[arg-type]


def test_a_class_baseline_ranks_only_that_class() -> None:
    """A nonprivate reference is never ranked for the standard class's best."""
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=0.5),
            _measurement(algorithm="A", instance="i1", value=3.0),
            _measurement(algorithm="B", instance="i1", value=2.0),
        ]
    )

    baseline = compute_baseline(
        frame, baseline=PrivacyInterpretation.STANDARD, metric="value"
    )

    assert baseline["baseline"].tolist() == [2.0]


def test_a_class_baseline_ranks_the_class_the_caller_names() -> None:
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=0.5),
            _measurement(algorithm="A", instance="i1", value=3.0),
        ]
    )

    nonprivate = compute_baseline(
        frame, baseline=PrivacyInterpretation.NONPRIVATE, metric="value"
    )
    standard = compute_baseline(
        frame, baseline=PrivacyInterpretation.STANDARD, metric="value"
    )

    assert nonprivate["baseline"].tolist() == [0.5]
    assert standard["baseline"].tolist() == [3.0]
    with pytest.raises(ValueError, match="no rows"):
        compute_baseline(
            frame, baseline=PrivacyInterpretation.QUALIFIED, metric="value"
        )


@pytest.mark.parametrize(
    "baseline",
    [_algorithm("A"), [_algorithm("A")], PrivacyInterpretation.STANDARD, None],
    ids=["algorithm", "sequence", "class", "none"],
)
def test_compute_baseline_requires_the_metric(baseline: Any) -> None:
    """Every kind compute_baseline builds ranks or keys measured costs."""
    frame = pd.DataFrame([_measurement(algorithm="A", instance="i1", value=3.0)])

    with pytest.raises(TypeError, match="metric"):
        compute_baseline(frame, baseline=baseline)  # type: ignore[call-arg]


def _referenced_frame() -> pd.DataFrame:
    """Two trials on one Instance that stores ``published_se`` = 2."""
    referenced = Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(b"referenced").hexdigest(),
        metric_arguments={"published_se": 2.0},
        label="referenced",
    )
    return pd.DataFrame(
        [
            {
                **_measurement(
                    algorithm="A", instance="referenced", value=value, alg_seed=seed
                ),
                "instance": referenced,
            }
            for seed, value in enumerate((5.0, 3.0))
        ]
    )


def test_a_stored_key_normalizes_like_the_table_it_names() -> None:
    """Hand-computed: 5 and 3 against a stored 2 are 2.5 and 1.5."""
    frame = _referenced_frame()
    table = frame[["task", "instance"]].drop_duplicates().assign(baseline=2.0)

    ratios = compute_error_ratio(frame, "published_se", metric="value")

    assert ratios["value"].tolist() == [2.5, 1.5]
    assert ratios["baseline"].tolist() == [2.0, 2.0]
    pd.testing.assert_frame_equal(
        ratios, compute_error_ratio(frame, table, metric="value")
    )


def test_best_of_ranks_only_the_named_roster() -> None:
    """Named algorithms set the baseline whatever their class, and no others."""
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=0.5),
            _measurement(algorithm="A", instance="i1", value=3.0),
            _measurement(algorithm="B", instance="i1", value=2.0),
        ]
    )
    reference = frame.iloc[0]["algorithm"]
    first = frame.iloc[1]["algorithm"]

    both = compute_baseline(
        frame, baseline=[reference, first], metric="value", per_budget=False
    )
    only_first = compute_baseline(frame, baseline=(first,), metric="value")

    assert both["baseline"].tolist() == [0.5]
    assert only_first["baseline"].tolist() == [3.0]
    with pytest.raises(ValueError, match="no rows for reference algorithm"):
        compute_baseline(frame, baseline=[first, _algorithm("Absent")], metric="value")
    with pytest.raises(TypeError, match="lists configured algorithms.*'str'"):
        compute_baseline(frame, baseline=[first, first.uuid], metric="value")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one algorithm"):
        compute_baseline(frame, baseline=[], metric="value")


def test_relabelled_references_are_one_baseline_and_one_seed_group() -> None:
    first = _reference_row(instance="i1", value=2.0)
    second = _reference_row(
        instance="i1", value=4.0, alg_seed=1, label="Panel reference"
    )
    second["instance"] = second["instance"].model_copy(
        update={"label": "panel instance"}
    )
    frame = pd.DataFrame([first, second])
    baseline = compute_baseline(frame, baseline=first["algorithm"], metric="value")
    assert baseline["baseline"].tolist() == [3.0]
    assert aggregate_trials(frame, metric="value")["value"].tolist() == [3.0]
    assert compute_error_ratio(frame, baseline, metric="value")[
        "value"
    ].tolist() == pytest.approx([2 / 3, 4 / 3])


@pytest.mark.parametrize("per_budget", [False, True])
def test_baseline_selectors_ignore_colliding_and_uuid_looking_labels(
    per_budget: bool,
) -> None:
    first = _algorithm("A")
    shadow = _algorithm("B").model_copy(update={"label": first.uuid})
    frame = pd.DataFrame(
        [
            {
                **_measurement(
                    algorithm="A", instance="i1", value=2.0, budget=NO_PRIVACY
                ),
                "algorithm": first,
            },
            {
                **_measurement(
                    algorithm="B", instance="i1", value=8.0, budget=NO_PRIVACY
                ),
                "algorithm": shadow,
            },
        ]
    )
    assert compute_baseline(
        frame, metric="value", baseline=first, per_budget=per_budget
    )["baseline"].tolist() == [2.0]
    assert compute_baseline(
        frame, metric="value", baseline=shadow, per_budget=per_budget
    )["baseline"].tolist() == [8.0]
    for spelled in ("A", first.uuid):
        with pytest.raises(TypeError, match="no stored key"):
            compute_baseline(
                frame,
                metric="value",
                baseline=spelled,  # type: ignore[arg-type]
                per_budget=per_budget,
            )
    frame["algorithm"] = [
        value.model_copy(update={"label": "same"}) for value in frame["algorithm"]
    ]
    assert compute_baseline(
        frame, metric="value", baseline=first, per_budget=per_budget
    )["baseline"].tolist() == [2.0]
    assert compute_baseline(
        frame, metric="value", baseline=shadow, per_budget=per_budget
    )["baseline"].tolist() == [8.0]


def test_a_stored_key_no_instance_stores_is_refused_by_name() -> None:
    frame = _referenced_frame()

    with pytest.raises(ValueError, match="have no value named 'reference_missing'"):
        compute_error_ratio(frame, "reference_missing", metric="value")


def test_baseline_by_reference_refuses_a_sibling_instance_lacking_it() -> None:
    """A reference only some Instances store is refused by name, not a bare KeyError."""
    has_it = Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(b"has_it").hexdigest(),
        metric_arguments={"published_se": 2.0},
        label="has_it",
    )
    lacks_it = Instance(
        metadata=SimpleMetadata(n_samples=2, dim=1),
        input_digest=hashlib.sha256(b"lacks_it").hexdigest(),
        metric_arguments={},
        label="lacks_it",
    )
    frame = pd.DataFrame(
        [
            {
                **_measurement(algorithm="A", instance="has_it", value=5.0),
                "instance": has_it,
            },
            {
                **_measurement(algorithm="A", instance="lacks_it", value=5.0),
                "instance": lacks_it,
            },
        ]
    )

    with pytest.raises(ValueError, match="lacks_it.*have no value named"):
        compute_error_ratio(frame, "published_se", metric="value")


def test_selecting_before_a_class_baseline_narrows_what_it_ranks() -> None:
    """A wins i1 alone; B has the better median across the three Instances."""
    costs = {"A": [1.0, 5.0, 6.0], "B": [2.0, 3.0, 4.0], "C": [10.0, 11.0, 12.0]}
    frame = pd.DataFrame(
        [
            _measurement(algorithm=label, instance=instance, value=value)
            for label, values in costs.items()
            for instance, value in zip(("i1", "i2", "i3"), values)
        ]
    )

    whole_pool = compute_baseline(
        frame, metric="value", baseline=PrivacyInterpretation.STANDARD
    )
    selected_ids = {_algorithm("B").uuid, _algorithm("C").uuid}
    without_a = compute_baseline(
        frame[frame["algorithm"].map(attrgetter("uuid")).isin(selected_ids)],
        metric="value",
        baseline=PrivacyInterpretation.STANDARD,
    )

    named = [instance.label for instance in whole_pool["instance"]]
    assert dict(zip(named, whole_pool["baseline"])) == {
        "i1": 1.0,
        "i2": 3.0,
        "i3": 4.0,
    }
    narrowed = [instance.label for instance in without_a["instance"]]
    assert dict(zip(narrowed, without_a["baseline"])) == {
        "i1": 2.0,
        "i2": 3.0,
        "i3": 4.0,
    }


def test_a_class_baseline_ranks_each_budget_on_its_own() -> None:
    """Per budget: the best median, with and without the reference excluded."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=6.0, alg_seed=1),
            _measurement(algorithm="B", instance="i1", value=3.0),
            _measurement(algorithm="OLS", instance="i1", value=0.5),
            _measurement(
                algorithm="A",
                instance="i1",
                value=1.0,
                budget=PureDPBudget(eps=2.0),
            ),
            _measurement(
                algorithm="B",
                instance="i1",
                value=5.0,
                budget=PureDPBudget(eps=2.0),
            ),
        ]
    )

    unfiltered = compute_baseline(
        frame, metric="value", baseline=PrivacyInterpretation.STANDARD
    )
    unfiltered_offers = [budget.eps for budget in unfiltered["budget"]]
    assert dict(zip(unfiltered_offers, unfiltered["baseline"])) == {1.0: 0.5, 2.0: 1.0}
    baseline = compute_baseline(
        frame[frame["algorithm"].map(str) != "OLS"],
        metric="value",
        baseline=PrivacyInterpretation.STANDARD,
    )

    offered = [budget.eps for budget in baseline["budget"]]
    assert dict(zip(offered, baseline["baseline"])) == {1.0: 3.0, 2.0: 1.0}

    # That baseline joins back onto the rows by the budget value itself.
    normalized = compute_error_ratio(frame, baseline, metric="value")

    assert normalized["value"].tolist() == [
        pytest.approx(2 / 3),
        pytest.approx(2.0),
        pytest.approx(1.0),
        pytest.approx(1 / 6),
        pytest.approx(1.0),
        pytest.approx(5.0),
    ]


def test_profile_curves_preserve_an_empty_selection() -> None:
    empty = pd.DataFrame([_measurement(algorithm="a", instance="i1", value=1.0)]).iloc[
        :0
    ]
    curves = compute_profile_curves(empty, [0.0, 1.0], metric="value")
    assert curves.empty
    assert list(curves.columns) == ["algorithm", "metric", "tau", "fraction"]


def test_trial_groups_and_curves_carry_the_label_columns() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=3.0, alg_seed=1),
        ]
    ).assign(algorithm_label="A", instance_label="i1")

    medians = aggregate_trials(frame, metric="value")
    curves = compute_profile_curves(medians, [2.0], metric="value")

    assert medians[["algorithm_label", "instance_label", "value"]].values.tolist() == [
        ["A", "i1", 2.0]
    ]
    assert list(curves.columns) == [
        "algorithm",
        "algorithm_label",
        "metric",
        "tau",
        "fraction",
    ]
    assert curves["algorithm_label"].tolist() == ["A"]


def test_named_metric_missing_rows_are_excluded_from_analysis_denominators() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=math.nan, alg_seed=1),
        ]
    ).rename(columns={"value": "error"})
    frame["other_metric"] = [100.0, 200.0]
    with pytest.warns(UserWarning, match="missing 'error'"):
        medians = aggregate_trials(frame, metric="error")
    assert medians["error"].tolist() == [1.0]
    assert "value" not in medians and "other_metric" not in medians
    with pytest.warns(UserWarning, match="missing 'error'"):
        curves = compute_profile_curves(frame, [1.0], metric="error")
    assert curves["fraction"].tolist() == [1.0]
    assert curves["metric"].tolist() == ["error"]
    with pytest.warns(UserWarning, match="missing 'error'"):
        normalized = compute_error_ratio(
            frame,
            pd.DataFrame([{"instance": frame.iloc[0]["instance"], "baseline": 2.0}]),
            metric="error",
        )
    assert normalized["error"].tolist() == [0.5]


def test_missing_reference_measurement_does_not_become_a_unit_baseline() -> None:
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=math.nan),
            _measurement(algorithm="A", instance="i1", value=2.0),
        ]
    )
    with pytest.warns(UserWarning, match="missing"):
        baseline = compute_baseline(
            frame, baseline=frame.iloc[0]["algorithm"], metric="value"
        )
    assert baseline.empty
    with pytest.warns(UserWarning, match="missing"):
        with pytest.raises(ValueError, match="no baseline"):
            compute_error_ratio(frame, baseline, metric="value")


def test_baseline_best_uses_configured_identity_and_aggregates_before_best() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=9.0, alg_seed=1),
            _measurement(algorithm="B", instance="i1", value=4.0, alg_seed=0),
            _measurement(algorithm="B", instance="i1", value=4.0, alg_seed=1),
            _measurement(
                algorithm="A", instance="i1", value=12.0, budget=PureDPBudget(eps=2.0)
            ),
        ]
    )
    original = frame.copy(deep=True)
    assert compute_baseline(
        frame, metric="value", baseline=PrivacyInterpretation.STANDARD
    )["baseline"].tolist() == [4.0, 12.0]
    assert compute_baseline(
        frame,
        metric="value",
        baseline=PrivacyInterpretation.STANDARD,
        aggregation="min",
    )["baseline"].tolist() == [1.0, 12.0]
    algorithm = frame.iloc[0]["algorithm"]
    for selected in (algorithm, algorithm.model_copy(update={"label": "renamed"})):
        assert compute_baseline(frame, metric="value", baseline=selected)[
            "baseline"
        ].tolist() == [5.0, 12.0]
    ambiguous = frame.copy()
    ambiguous["algorithm"] = [
        algorithm.model_copy(update={"label": "same"})
        for algorithm in frame["algorithm"]
    ]
    with pytest.raises(TypeError, match="no stored key 'same'"):
        compute_baseline(ambiguous, metric="value", baseline="same")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is not an aggregation"):
        compute_baseline(
            frame,
            metric="value",
            baseline=PrivacyInterpretation.STANDARD,
            aggregation="not_an_aggregation",
        )
    pd.testing.assert_frame_equal(frame, original)


def test_profile_curves_preserve_numeric_algorithms_and_threshold_order() -> None:
    frame = pd.DataFrame({"algorithm": [2, 1], "instance": [0, 0], "value": [1.0, 2.0]})
    curves = compute_profile_curves(frame, [2.0, 0.0, 1.0], metric="value")
    expected = pd.DataFrame(
        {
            "algorithm": [2, 1, 2, 1, 2, 1],
            "metric": ["value"] * 6,
            "tau": [2.0, 2.0, 0.0, 0.0, 1.0, 1.0],
            "fraction": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(curves, expected)


def test_full_coverage_stays_within_the_profile_fraction_domain() -> None:
    frame = pd.DataFrame(
        {"algorithm": ["A"] * 30, "instance": [0, 1, 2] * 10, "value": [1.0] * 30}
    )
    curves = compute_profile_curves(frame, [1.0], metric="value")
    fraction = curves["fraction"].item()
    assert 0.0 <= fraction <= 1.0
    assert fraction == pytest.approx(1.0)


def test_profile_curves_weight_instances_as_the_caller_supplies() -> None:
    """i1 weighs 3 and i2 weighs 1 over runs 1, 2 and 4, inf."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=1),
            _measurement(algorithm="A", instance="i2", value=4.0, alg_seed=0),
            _measurement(algorithm="A", instance="i2", value=math.inf, alg_seed=1),
        ]
    )
    weights = pd.DataFrame(
        [
            {"instance": _instance("i1"), "weight": 3.0},
            {"instance": _instance("i2"), "weight": 1.0},
        ]
    )

    curves = compute_profile_curves(
        frame, [1.0, 2.0, 4.0], weights=weights, metric="value"
    )

    assert dict(zip(curves["tau"], curves["fraction"])) == {
        1.0: pytest.approx(0.375),
        2.0: pytest.approx(0.75),
        4.0: pytest.approx(0.875),
    }


def test_profile_curves_weigh_instances_equally_by_default() -> None:
    """The same runs, each Instance counting once."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=2.0, alg_seed=1),
            _measurement(algorithm="A", instance="i2", value=4.0, alg_seed=0),
            _measurement(algorithm="A", instance="i2", value=math.inf, alg_seed=1),
        ]
    )

    curves = compute_profile_curves(frame, [1.0, 2.0, 4.0], metric="value")

    assert dict(zip(curves["tau"], curves["fraction"])) == {
        1.0: pytest.approx(0.25),
        2.0: pytest.approx(0.5),
        4.0: pytest.approx(0.75),
    }


def test_profile_curves_refuse_zero_total_weight_for_a_curve() -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["weighted", "weighted", "zero", "zero"],
            "instance": [0, 1, 0, 1],
            "value": [1.0, 2.0, 1.0, 2.0],
        }
    )
    weights = pd.DataFrame({"algorithm": ["weighted", "zero"], "weight": [1.0, 0.0]})

    with pytest.raises(ValueError, match="zero curve has zero total weight"):
        compute_profile_curves(frame, [1.0], weights=weights, metric="value")


@pytest.mark.parametrize("row_index", [[0, 1, 2], [9, 4, 9]])
def test_profile_threshold_order_ties_and_unequal_instance_sizes(
    row_index: list[int],
) -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=1),
            _measurement(algorithm="A", instance="i2", value=math.inf),
        ],
        index=row_index,
    )
    weights = pd.DataFrame(
        [
            {"instance": _instance("i1"), "weight": 1.0},
            {"instance": _instance("i2"), "weight": 3.0},
        ]
    )
    thresholds = [2.0, 0.0, 1.0, 1.0, math.inf]
    curves = compute_profile_curves(frame, thresholds, weights=weights, metric="value")
    assert curves["tau"].tolist() == thresholds
    assert curves["fraction"].tolist() == pytest.approx([0.25, 0.0, 0.25, 0.25, 0.25])


def test_a_nan_threshold_is_refused_rather_than_scored() -> None:
    """An infinity is a threshold every finite cost falls within; a NaN is none."""
    frame = pd.DataFrame(
        [_measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0)]
    )

    with pytest.raises(ValueError, match="NaN names no threshold"):
        compute_profile_curves(frame, [1.0, math.nan], metric="value")
    assert compute_profile_curves(frame, [math.inf], metric="value")[
        "fraction"
    ].tolist() == [1.0]


def test_profile_thresholds_are_refused_before_any_measurement() -> None:
    """The thresholds are read before a column is touched."""
    absent_metric = pd.DataFrame(
        [_measurement(algorithm="A", instance="i1", value=1.0)]
    ).drop(columns=["value"])

    with pytest.raises(ValueError, match="at least one threshold"):
        compute_profile_curves(absent_metric, [], metric="value")


def test_an_infinite_cost_keeps_the_curve_below_one_at_every_threshold() -> None:
    """Two of three runs are measurable, so no threshold reaches full coverage."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=1),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=2),
        ]
    )

    curves = compute_profile_curves(frame, [1.0, 1e9], metric="value")

    assert curves["fraction"].tolist() == [pytest.approx(2 / 3), pytest.approx(2 / 3)]
    assert bool((curves["fraction"] < 1.0).all())


def test_the_share_at_an_infinite_threshold_is_the_share_of_finite_errors() -> None:
    """An infinite cost is never within a threshold, the infinite one included."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=1e300, alg_seed=1),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=2),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=3),
        ]
    )

    curves = compute_profile_curves(frame, [1.0, math.inf], metric="value")

    assert curves["fraction"].tolist() == [0.25, 0.5]


def test_a_negative_infinite_cost_is_within_every_threshold() -> None:
    """Negative infinity is a measured best cost, not a failure to release."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=-math.inf, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=1),
        ]
    )

    curves = compute_profile_curves(frame, [-math.inf, 0.0, math.inf], metric="value")

    assert curves["fraction"].tolist() == [0.5, 0.5, 0.5]


def test_a_curve_lacking_a_listed_instance_is_refused_by_its_label() -> None:
    """A never-attempted coordinate leaves no row; the denominator stays checked."""
    absent = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0),
            _measurement(algorithm="A", instance="i2", value=1.0),
            _measurement(algorithm="B", instance="i1", value=1.0),
        ]
    )

    with pytest.raises(
        ValueError, match="The B curve has no trial on Instance\\(s\\) i2"
    ):
        compute_profile_curves(absent, [1.0], metric="value")
    with pytest.raises(ValueError, match="no trial on Instance\\(s\\) i3"):
        compute_profile_curves(
            absent, [1.0], metric="value", instances=[_instance("i1"), _instance("i3")]
        )


def test_listed_instances_are_the_denominator_and_other_rows_are_left_out() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=0.5),
            _measurement(algorithm="A", instance="i2", value=2.0),
            _measurement(algorithm="B", instance="i1", value=2.0),
        ]
    )

    curves = compute_profile_curves(
        frame, [1.0], metric="value", instances=[_instance("i1")]
    )

    assert [algorithm.label for algorithm in curves["algorithm"]] == ["A", "B"]
    assert curves["fraction"].tolist() == [1.0, 0.0]


def test_unequal_trial_counts_share_each_instance_within_itself() -> None:
    """A has two trials on i1 and B one; each Instance still counts once."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0, alg_seed=0),
            _measurement(algorithm="A", instance="i1", value=3.0, alg_seed=1),
            _measurement(algorithm="B", instance="i1", value=1.0, alg_seed=0),
        ]
    )

    curves = compute_profile_curves(frame, [2.0], metric="value")

    assert curves["fraction"].tolist() == [0.5, 1.0]


def test_profile_curves_refuse_an_instance_no_weight_covers() -> None:
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=1.0),
            _measurement(algorithm="A", instance="i2", value=1.0),
        ]
    )
    weights = pd.DataFrame([{"instance": _instance("i1"), "weight": 1.0}])

    with pytest.raises(ValueError, match="have no weight"):
        compute_profile_curves(frame, [1.0], weights=weights, metric="value")


def test_profile_curves_pool_whatever_budgets_they_are_given() -> None:
    """One algorithm under two budgets draws one curve over both of its trials."""
    frame = pd.DataFrame(
        [
            _measurement(algorithm="A", instance="i1", value=4.0),
            _measurement(
                algorithm="A",
                instance="i1",
                value=1.0,
                budget=PureDPBudget(eps=2.0),
            ),
        ]
    )

    curves = compute_profile_curves(frame, [2.0], metric="value")

    assert "budget" not in curves.columns
    assert curves["fraction"].tolist() == [0.5]


def test_baseline_aggregates_label_aliases_as_one_identity() -> None:
    algorithm = _algorithm("A")
    renamed = algorithm.model_copy(update={"label": "renamed"})
    frame = pd.DataFrame(
        [
            {
                **_measurement(algorithm="A", instance="i1", value=1.0),
                "algorithm": algorithm,
            },
            {
                **_measurement(algorithm="A", instance="i1", value=9.0, alg_seed=1),
                "algorithm": renamed,
            },
        ]
    )
    assert algorithm.uuid == renamed.uuid
    for selected in (algorithm, renamed, PrivacyInterpretation.STANDARD):
        assert compute_baseline(frame, metric="value", baseline=selected)[
            "baseline"
        ].tolist() == [5.0]


def test_infinite_baseline_ratio_uses_declared_policy_without_runtime_warning() -> None:
    frame = pd.DataFrame([_measurement(algorithm="A", instance="i1", value=math.inf)])
    baseline = pd.DataFrame([_baseline("i1", math.inf)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        normalized = compute_error_ratio(frame, baseline, metric="value")
        curves = compute_profile_curves(normalized, [1.0], metric="value")
    assert normalized["value"].tolist() == [math.inf]
    assert curves["fraction"].tolist() == [0.0]


def test_every_measured_data_helper_reads_a_metric_as_the_column_it_names() -> None:
    """A Metric renders as the column it produces, and a name is already one."""
    metric = Metric(task=SIMPLE_TASK, name="value", calculation=lambda output: 0.0)
    frame = pd.DataFrame(
        [
            _reference_row(instance="i1", value=2.0),
            _reference_row(instance="i1", value=2.0, alg_seed=1),
            _measurement(algorithm="A", instance="i1", value=4.0),
            _measurement(algorithm="A", instance="i1", value=math.inf, alg_seed=1),
        ]
    ).assign(status="success")
    baseline = pd.DataFrame([{"instance": frame.iloc[0]["instance"], "baseline": 2.0}])
    operations: list[Callable[[Metric | str], pd.DataFrame]] = [
        lambda selected: aggregate_trials(frame, metric=selected),
        lambda selected: compute_error_ratio(frame, baseline, metric=selected),
        lambda selected: compute_baseline(
            frame, metric=selected, baseline=NonprivateAlgorithm(), per_budget=False
        ),
        lambda selected: compute_baseline(frame, metric=selected, baseline=None),
        lambda selected: compute_baseline(
            frame, metric=selected, baseline=PrivacyInterpretation.STANDARD
        ),
        lambda selected: compute_profile_curves(frame, [1.0], metric=selected),
    ]

    assert str(metric) == "value"
    # The full declaration stays available for a diagnostic that asks for it.
    assert repr(metric).startswith("Metric(task=")
    for operation in operations:
        pd.testing.assert_frame_equal(operation(metric), operation(metric.name))
