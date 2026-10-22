"""Non-private ridge-regression baseline."""

from __future__ import annotations

from typing import ClassVar

import math

import numpy as np
from scipy.linalg import lstsq

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_range
from dpstats.task import Array1D, Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


class Ridge(BaseAlgorithm):
    """Fit ridge regression exactly, with no privacy, as a reference point.

    An implementation of the method in Friedman, Hastie and Tibshirani (2010).
    Minimizes the sum of squared residuals plus a penalty on the squared size
    of the coefficients, and outputs the fit exactly. It adds no noise, so it
    gives no privacy guarantee; it is a reference point for the private
    methods.

    Requires:
        - An unconstrained privacy budget.

    Parametrization notes:
        - ``regularization_multiplier``: The penalty is this multiplier times
          ``n_samples``, glmnet's scaling for the Gaussian loss. glmnet fixes
          only that scaling, not the multiplier.

    Implementation notes:
        - Unlike glmnet, every coefficient is penalized, including a declared
          intercept, which is shrunk toward zero.

    Reference:
        - friedman2010regularization.
    """

    TASK = LinearRegression

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.NONPRIVATE
    )

    regularization_multiplier: float = parameter_field(
        default=0.01,
        gt=0.0,
        search=search_range(1e-6, 10.0, scale="log"),
        description="Strictly positive L2 penalty on the averaged-loss scale.",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        if not budget.is_unconstrained:
            raise InvalidRegime("Ridge requires an unconstrained privacy budget.")
        X = input.X
        y = input.y
        penalty = self.regularization_multiplier * metadata.n_samples
        augmented_design: Array2D = np.concatenate(
            [X, math.sqrt(penalty) * np.eye(metadata.dim)]
        )
        augmented_response: Array1D = np.concatenate([y, np.zeros(metadata.dim)])
        beta, *_ = lstsq(
            augmented_design,
            augmented_response,
            cond=0.0,
            lapack_driver="gelsy",
        )
        return LinearRegressionOutput(beta=beta)
