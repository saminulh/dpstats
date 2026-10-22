"""Post-run Metrics and metric arguments for logistic regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import LogisticRegression
from dpstats.tasks.logistic_regression import LogisticRegressionOutput


ground_truth = MetricArgument(
    name="ground_truth",
    description="Ground-truth coefficient vector.",
)

mle_coef = MetricArgument(
    name="mle_coef",
    description="Maximum likelihood coefficient vector fitted on the instance.",
)


def _l2_error_to_ground_truth(
    output: LogisticRegressionOutput, *, ground_truth: Any
) -> float:
    r"""L2 error between the output coefficients and ground truth.

    Args:
        output: The algorithm output to evaluate.
        ground_truth (array_like):
            Ground-truth coefficient vector.

    Notes:
        Computes :math:`\lVert\hat\theta-\theta^\star\rVert_2`, the
        :math:`\ell_2` error to the ground-truth coefficient vector
        :math:`\theta^\star`.
    """
    return float(np.linalg.norm(output.theta - np.asarray(ground_truth, dtype=float)))


l2_error_to_ground_truth = Metric(
    task=LogisticRegression,
    name="l2_error_to_ground_truth",
    calculation=_l2_error_to_ground_truth,
    requires=(ground_truth,),
    formula=r"\lVert \hat{\theta} - \theta^{\star} \rVert_2",
    label="L2 error vs ground truth",
)


def _l2_error_to_mle(output: LogisticRegressionOutput, *, mle_coef: Any) -> float:
    r"""L2 error between the output coefficients and the maximum likelihood estimate.

    Args:
        output: The algorithm output to evaluate.
        mle_coef (array_like):
            Maximum likelihood coefficient vector fitted on the instance.

    Notes:
        Computes :math:`\lVert\hat\theta-\hat\theta_{\mathrm{MLE}}\rVert_2`,
        the :math:`\ell_2` error to the reference maximum likelihood coefficient
        vector :math:`\hat\theta_{\mathrm{MLE}}`.
    """
    return float(np.linalg.norm(output.theta - np.asarray(mle_coef, dtype=float)))


l2_error_to_mle = Metric(
    task=LogisticRegression,
    name="l2_error_to_mle",
    calculation=_l2_error_to_mle,
    requires=(mle_coef,),
    formula=r"\lVert \hat{\theta} - \hat{\theta}_{\mathrm{MLE}} \rVert_2",
    label="L2 error vs MLE",
)


METRICS: tuple[Metric, ...] = (
    l2_error_to_ground_truth,
    l2_error_to_mle,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
