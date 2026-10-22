"""Built-in components refuse configured values outside their declared ranges.

Each row states one component's contract value explicitly: a failure
probability inside ``(0, 1)``, a positive scale, rate, multiplier or radius, a
budget share strictly inside ``(0, 1)``, and so on. A row fails if the
declaration that enforces it is loosened or removed.
"""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from dpstats.tasks.linear_regression.algorithms import (
    AdaOPS,
    AdaSSP,
    BoostedAdaSSP,
    DPAMBSSGD,
    GridExponentialMechanism,
    OnePassDPGD,
    TailRobustDPGD,
)
from dpstats.tasks.linear_regression.transformations import (
    ApproxBounds as LRApproxBounds,
    ApproxBoundsAll as LRApproxBoundsAll,
    DPKendall,
    ExactKendallSelection,
    ToMEstimation as LinearRegressionToMEstimation,
)
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    HuberResidualLoss,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    HuberNormLoss,
)
from dpstats.tasks.m_estimation.algorithms import (
    AMP,
    DPSGD,
)
from dpstats.tasks.mean_estimation.algorithms import (
    CoinPress,
)
from dpstats.tasks.mean_estimation.transformations import (
    ApproxBounds as MeanApproxBounds,
)
from dpstats.tasks.median_estimation.algorithms import (
    UnboundedQuantileMechanism,
    WideExponentialMechanism,
)
from dpstats.tasks.median_estimation.transformations import ToMEstimation
from dpstats.tasks.simple_linear_regression.algorithms import (
    DPGD,
    DPTheilSen,
)
from dpstats.tasks.single_coefficient_estimation.algorithms import (
    LocalOutputPerturbation,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ApproxBounds as SCEApproxBounds,
    ApproxBoundsAll as SCEApproxBoundsAll,
    DPKendall,
)

nan, inf = math.nan, math.inf

REFUSED: list[tuple[type[Any], str, Any]] = [
    # Failure probabilities lie strictly inside (0, 1).
    *[
        (component, "failure_probability", value)
        for component in (AdaOPS, AdaSSP, DPAMBSSGD, CoinPress)
        for value in (0.0, 1.0, nan)
    ],
    # Budget and selection shares lie strictly inside (0, 1).
    *[
        (component, field, value)
        for component, field in (
            (AMP, "output_budget_fraction"),
            (CoinPress, "final_budget_fraction"),
            (DPKendall, "selection_share"),
            (DPKendall, "selection_share"),
        )
        for value in (0.0, 1.0)
    ],
    # Private-bound scales are positive and finite.
    *[
        (transformation, "scale", value)
        for transformation in (
            MeanApproxBounds,
            LRApproxBounds,
            LRApproxBoundsAll,
            SCEApproxBounds,
            SCEApproxBoundsAll,
        )
        for value in (0.0, inf, nan)
    ],
    # A parameter ball has a positive finite radius; ``None`` spells unbounded.
    *[
        (LinearRegressionToMEstimation, "parameter_radius", value)
        for value in (0.0, inf)
    ],
    # Positive clipping thresholds, multipliers, rates and counts.
    *[
        (component, field, 0)
        for component, field in (
            (DPGD, "tau"),
            (DPGD, "num_steps"),
            (BoostedAdaSSP, "tau"),
            (BoostedAdaSSP, "num_rounds"),
            (OnePassDPGD, "c"),
            (OnePassDPGD, "scale"),
            (TailRobustDPGD, "learning_rate"),
            (AMP, "gradient_tolerance"),
            (GridExponentialMechanism, "residual_cap_mult"),
            (GridExponentialMechanism, "coef_box_mult"),
            (HuberResidualLoss, "threshold"),
            (HuberNormLoss, "threshold_per_sqrt_dim"),
            (DPKendall, "num_features"),
            (ExactKendallSelection, "num_features"),
            (DPKendall, "num_features"),
            (CoinPress, "num_iters"),
            (DPTheilSen, "degrees_of_freedom"),
        )
    ],
    (AMP, "objective_noise_ratio", inf),
    (DPSGD, "subsample_rate", 0.0),
    (DPSGD, "subsample_rate", 1.5),
    (DPSGD, "C_grad_clip", "lipschitz"),
    (DPTheilSen, "degrees_of_freedom", nan),
    (DPTheilSen, "degrees_of_freedom", True),
    (UnboundedQuantileMechanism, "beta", 1.0),
    (WideExponentialMechanism, "width", -0.1),
    # A ridge weight is nonnegative and finite.
    *[(LocalOutputPerturbation, "regularization", value) for value in (-0.1, inf, nan)],
]


@pytest.mark.parametrize(
    ("component", "field", "value"),
    REFUSED,
    ids=[
        f"{component.__module__.split('.')[2]}.{component.__name__}.{field}={value}"
        for component, field, value in REFUSED
    ],
)
def test_out_of_range_parameter_is_refused_at_construction(
    component: type[Any], field: str, value: Any
) -> None:
    with pytest.raises(ValidationError) as refusal:
        component(**{field: value})

    assert {error["loc"][0] for error in refusal.value.errors()} == {field}
