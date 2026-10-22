"""Post-run metrics and metric arguments for the single-coefficient-estimation task."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from collections.abc import Sequence

from dpstats.metric import Metric, MetricArgument
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationOutput,
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
)


def _ols_se_arguments(
    metadata: SingleCoefficientEstimationMetadata,
    input: SingleCoefficientEstimationInput,
) -> dict[str, Any]:
    """Fit OLS once and derive the standard error of the coefficient of interest.

    Raises:
        ValueError: If the design has no residual degrees of freedom, or is
            rank-deficient so ``(X'X)^-1`` does not exist.
    """
    n, d = input.X.shape
    if n <= d:
        raise ValueError(
            f"design has n={n} rows and d={d} columns: "
            "no residual degrees of freedom."
        )
    rank = np.linalg.matrix_rank(input.X)
    if rank < d:
        raise ValueError(f"design is rank-deficient: rank {rank} < {d} columns.")
    beta, *_ = np.linalg.lstsq(input.X, input.y, rcond=None)
    rss = float(np.sum((input.X @ beta - input.y) ** 2))
    gram_inverse = np.linalg.inv(input.X.T @ input.X)
    index = metadata.coefficient_of_interest_index
    residual_variance = rss / (n - d)
    return {"ols_se": math.sqrt(residual_variance * float(gram_inverse[index, index]))}


ols_coef = MetricArgument(
    name="ols_coef",
    description="The OLS coefficient of interest.",
)

ols_se = MetricArgument(
    name="ols_se",
    description=(
        "The OLS standard error of the coefficient of interest, with "
        "residual variance RSS/(n - d)."
    ),
    producer=_ols_se_arguments,
)


def _sign_disagreement_with_ols(
    output: SingleCoefficientEstimationOutput, *, ols_coef: float
) -> float:
    """Whether the output and OLS coefficients differ in sign.

    Args:
        output: The algorithm output to evaluate.
        ols_coef (float):
            The OLS coefficient of interest.

    Notes:
        Computes 0 for matching signs and 1 otherwise; zero has its own sign.
    """
    return 0.0 if np.sign(output.coefficient) == np.sign(ols_coef) else 1.0


sign_disagreement_with_ols = Metric(
    task=SingleCoefficientEstimation,
    name="sign_disagreement_with_ols",
    calculation=_sign_disagreement_with_ols,
    requires=(ols_coef,),
    formula=r"\mathbf{1}\{\operatorname{sign}(\hat{\beta}_j) \neq "
    r"\operatorname{sign}(\hat{\beta}_{j,\mathrm{OLS}})\}",
    label="Sign disagreement vs OLS",
)


def _absolute_error_to_ols(
    output: SingleCoefficientEstimationOutput, *, ols_coef: float
) -> float:
    r"""Absolute error between the output and OLS coefficients.

    Args:
        output: The algorithm output to evaluate.
        ols_coef (float):
            The OLS coefficient of interest.

    Notes:
        Computes :math:`\lvert\hat\beta_j-\hat\beta_{j,\mathrm{OLS}}\rvert`, the
        absolute error between the output and OLS coefficients.

        Divided by a standard error of the coefficient, it has the form of a
        :math:`|t|` statistic for the hypothesis that the coefficient equals
        :math:`\hat\beta_j`: the one-coefficient case of the linear-regression
        task's F-statistic reading.
    """
    return abs(output.coefficient - ols_coef)


absolute_error_to_ols = Metric(
    task=SingleCoefficientEstimation,
    name="absolute_error_to_ols",
    calculation=_absolute_error_to_ols,
    requires=(ols_coef,),
    formula=r"\lvert \hat{\beta}_j - \hat{\beta}_{j,\mathrm{OLS}} \rvert",
    label="Absolute error vs OLS",
    baseline=ols_se,
)


def _median_coefficient(outputs: Sequence[SingleCoefficientEstimationOutput]) -> float:
    """The median output coefficient over one Result group's successful Outputs."""
    return float(np.median([output.coefficient for output in outputs]))


def _absolute_error_of_median_to_ols(
    output: SingleCoefficientEstimationOutput,
    *,
    outputs: Sequence[SingleCoefficientEstimationOutput],
    ols_coef: float,
) -> float:
    r"""Absolute error between the group's median output coefficient and OLS.

    Args:
        output: The algorithm output; the value is the same for every Output
            of its group.
        outputs (Sequence[SingleCoefficientEstimationOutput]):
            The group's successful Outputs, whose median coefficient is read.
        ols_coef (float):
            The OLS coefficient of interest.

    Notes:
        Computes :math:`\lvert\operatorname{med}_t\hat\beta_{j,t}-\hat\beta_{j,\mathrm{OLS}}\rvert`:
        how far the algorithm's centre sits from OLS, its bias with the noise
        averaged out. Read as one median per instance, a failed trial counting
        :math:`+\infty`, it is infinite unless more than half the trials succeed.
    """
    del output
    return abs(_median_coefficient(outputs) - ols_coef)


absolute_error_of_median_to_ols = Metric(
    task=SingleCoefficientEstimation,
    name="absolute_error_of_median_to_ols",
    calculation=_absolute_error_of_median_to_ols,
    requires=("outputs", ols_coef),
    formula=r"\lvert \operatorname{med}_t \hat{\beta}_{j,t} - "
    r"\hat{\beta}_{j,\mathrm{OLS}} \rvert",
    label="Error of median",
    baseline=ols_se,
)


def _absolute_deviation_to_median(
    output: SingleCoefficientEstimationOutput,
    *,
    outputs: Sequence[SingleCoefficientEstimationOutput],
) -> float:
    r"""Absolute deviation of the output coefficient from its group's median.

    Args:
        output: The algorithm output to evaluate.
        outputs (Sequence[SingleCoefficientEstimationOutput]):
            The group's successful Outputs, whose median coefficient is read.

    Notes:
        Computes :math:`\lvert\hat\beta_{j,t}-\operatorname{med}_t\hat\beta_{j,t}\rvert`.
        Read as one median per instance it is the median absolute deviation, the
        spread of the outputs about their own centre; with the error of median
        it brackets the median absolute error to OLS from above.
    """
    return abs(output.coefficient - _median_coefficient(outputs))


absolute_deviation_to_median = Metric(
    task=SingleCoefficientEstimation,
    name="absolute_deviation_to_median",
    calculation=_absolute_deviation_to_median,
    requires=("outputs",),
    formula=r"\lvert \hat{\beta}_{j,t} - \operatorname{med}_t \hat{\beta}_{j,t} \rvert",
    label="Spread",
    baseline=ols_se,
)


METRICS: tuple[Metric, ...] = (
    absolute_error_to_ols,
    sign_disagreement_with_ols,
    absolute_error_of_median_to_ols,
    absolute_deviation_to_median,
)

# Produced values no Metric names, kept in the Task's catalogue.
ARGUMENTS: tuple[MetricArgument, ...] = ()
