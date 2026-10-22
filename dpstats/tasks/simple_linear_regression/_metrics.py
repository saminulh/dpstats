"""Post-run metrics and metric arguments for the simple_linear_regression task."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import SimpleLinearRegression
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionOutput,
)


ground_truth = MetricArgument(
    name="ground_truth",
    description="Ground-truth slope and intercept, in that order.",
)

ols_coef = MetricArgument(
    name="ols_coef",
    description="OLS slope and intercept, in that order.",
)


def _l2_error_to_ground_truth(
    output: SimpleLinearRegressionOutput, *, ground_truth: Any
) -> float:
    r"""L2 error between the output coefficients and ground truth.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Ground-truth slope and intercept, in that order.

    Notes:
        Computes :math:`\lVert(\hat a,\hat b)-(a^\star,b^\star)\rVert_2`,
        the :math:`\ell_2` error to the ground-truth slope and intercept.
    """
    ground_truth = np.asarray(ground_truth, dtype=float)
    return math.hypot(
        output.slope - float(ground_truth[0]),
        output.intercept - float(ground_truth[1]),
    )


l2_error_to_ground_truth = Metric(
    task=SimpleLinearRegression,
    name="l2_error_to_ground_truth",
    calculation=_l2_error_to_ground_truth,
    requires=(ground_truth,),
    formula=r"\lVert (\hat{a}, \hat{b}) - (a^{\star}, b^{\star}) \rVert_2",
    label="L2 error vs ground truth",
)


def _l2_error_to_ols(output: SimpleLinearRegressionOutput, *, ols_coef: Any) -> float:
    r"""L2 error between the output coefficients and OLS.

    Args:
        output: The algorithm output to evaluate.
        ols_coef (array_like):
            OLS slope and intercept, in that order.

    Notes:
        Computes :math:`\lVert(\hat a,\hat b)-(\hat a_{\mathrm{OLS}},\hat b_{\mathrm{OLS}})\rVert_2`,
        the :math:`\ell_2` error to the OLS slope and intercept.
    """
    ols_coef = np.asarray(ols_coef, dtype=float)
    return math.hypot(
        output.slope - float(ols_coef[0]),
        output.intercept - float(ols_coef[1]),
    )


l2_error_to_ols = Metric(
    task=SimpleLinearRegression,
    name="l2_error_to_ols",
    calculation=_l2_error_to_ols,
    requires=(ols_coef,),
    formula=r"\lVert (\hat{a}, \hat{b}) - "
    r"(\hat{a}_{\mathrm{OLS}}, \hat{b}_{\mathrm{OLS}}) \rVert_2",
    label="L2 error vs OLS",
)


METRICS: tuple[Metric, ...] = (
    l2_error_to_ground_truth,
    l2_error_to_ols,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
