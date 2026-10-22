import numpy as np

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field, search_values
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)

from my_regression_script import noisy_regression


class MyRegression(BaseAlgorithm):
    """Bridge my_regression_script into the linear-regression contract.

    The same wrapper as my_regression.py, with the type annotations kept.
    """

    # Declare the task that this algorithm is for
    TASK = LinearRegression

    # The regularization hyperparameter
    ridge: float = parameter_field(
        default=1.0,
        ge=0.0,
        search=search_values(0.1, 1.0, 5.0, 10.0),
        description="Ridge term added to the noisy X'X before solving.",
    )

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        # Extract the bounds on the domain of X and y from the metadata
        x_bound = metadata.design_domain.l2_norm_bound
        y_bound = metadata.response_domain.l2_norm_bound

        # Unbounded domains are not supported by noisy_regression, so we
        # raise an error if the bounds are not finite
        if not np.isfinite(x_bound) or not np.isfinite(y_bound):
            raise InvalidRegime(
                "noisy_regression requires finite covariate and response bounds."
            )

        # Convert the budget to a sufficient pure-DP budget and extract epsilon
        sufficient_pure_dp_budget = accounting.spend_as_pure(budget)
        eps = sufficient_pure_dp_budget.eps

        # Extract the inputs X and y
        X, y = input.X, input.y

        # Call the script with the extracted arguments and the sweep's seed
        beta = noisy_regression(X, y, eps, x_bound, y_bound, self.ridge, seed=seed)

        # Convert to the output type expected by the contract and return it
        return LinearRegressionOutput(beta=beta)
