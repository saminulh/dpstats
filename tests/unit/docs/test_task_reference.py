"""Task navigation and the catalogue read declarations, never authored text."""

from __future__ import annotations


from task_reference import outer_tasks, task_statuses

# Every built-in task, as the catalogue and task navigation read them.
EVERY_TASK = [
    "linear_regression",
    "logistic_regression",
    "m_estimation",
    "mean_estimation",
    "median_estimation",
    "simple_linear_regression",
    "single_coefficient_estimation",
]


def test_an_outer_task_is_one_a_transformation_prepares_this_task_for():
    """ToLinearRegression and ToLinearRegression consume their task and prepare linear regression."""
    reachable = outer_tasks(EVERY_TASK)
    assert reachable["linear_regression"] == [
        "simple_linear_regression",
        "single_coefficient_estimation",
    ]
    # Logistic regression exports no transformation and none prepares for it.
    assert reachable["logistic_regression"] == []


def test_outer_tasks_are_closed_through_an_intermediate_task():
    """M-estimation reaches SCE only through linear regression's own hop."""
    one_hop = outer_tasks(["m_estimation", "linear_regression", "mean_estimation"])
    assert "linear_regression" in one_hop["m_estimation"]
    assert "single_coefficient_estimation" not in one_hop["m_estimation"]
    closed = outer_tasks(EVERY_TASK)
    assert closed["m_estimation"] == [
        "linear_regression",
        "mean_estimation",
        "median_estimation",
        "simple_linear_regression",
        "single_coefficient_estimation",
    ]


def test_a_task_never_reaches_itself_through_its_own_transformations():
    """DataBounds and Standardize consume and prepare the same task."""
    assert outer_tasks(["mean_estimation"])["mean_estimation"] == []


def test_the_release_this_build_reads_decides_which_tasks_are_benchmarked():
    statuses = task_statuses(outer_tasks(EVERY_TASK), {"linear_regression"})
    assert statuses["linear_regression"] == "Benchmarked"
    # A task the release does not declare has no standings, however runnable.
    assert statuses["median_estimation"] == "Library task"


def test_a_task_other_tasks_reach_through_a_transformation_is_auxiliary_core():
    statuses = task_statuses(outer_tasks(EVERY_TASK), set())
    assert statuses["m_estimation"] == "Auxiliary core"
    # ResponseMean prepares MeanEstimation for LinearRegression.
    assert statuses["mean_estimation"] == "Auxiliary core"
    assert statuses["logistic_regression"] == "Library task"
    assert statuses["median_estimation"] == "Library task"


def test_every_documented_task_is_given_exactly_one_status():
    statuses = task_statuses(outer_tasks(EVERY_TASK), {"linear_regression"})
    assert sorted(statuses) == sorted(EVERY_TASK)
    assert set(statuses.values()) <= {"Benchmarked", "Auxiliary core", "Library task"}
