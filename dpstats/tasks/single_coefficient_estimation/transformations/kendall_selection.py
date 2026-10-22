"""Private control selection that always retains the coefficient of interest."""

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import Transformation
from dpstats.exceptions import NotApplicable
from dpstats.budgets import DPBudget
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.linear_regression.transformations import (
    DPKendall as LinearRegressionDPKendall,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)


class DPKendall(Transformation):
    """Privately select which controls to keep, always keeping the coefficient of interest.

    An adaptation of Algorithm 1 from Dick, Gillenwater and Joseph (2023).
    Privately picks which control columns to keep by how strongly they are
    rank-correlated with the response, always keeping the
    coefficient-of-interest column and any declared intercept without scoring
    them, and passes the kept columns and the response to the next step.

    Requires:
        - A selectable control besides the coefficient of interest; otherwise
          ``NotApplicable``.
        - At least two declared samples.
        - A ``num_features`` within the selectable columns; otherwise
          ``NotApplicable``.

    Parametrization notes:
        - ``num_features``: Corresponds to $k$ in the paper, counted here
          without the coefficient of interest and the intercept. Set to the
          value Section 4.4 of the paper suggests as the more relevant setting.
        - ``selection_share``: The fraction of the budget spent on selection.
          Set to the share Section 4.2 of the paper allocates to selection.

    Reference:
        - dick2023better: Algorithm 1 and Section 4.2.
        - dick2023code: Listing ``dp_kendall_feature_selection``.
    """

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

    num_features: int = parameter_field(
        default=5,
        gt=0,
        description=(
            "Number of selected controls, excluding the coefficient of "
            "interest and the intercept."
        ),
    )
    selection_share: float = parameter_field(
        default=0.05,
        gt=0,
        lt=1,
        description="Fraction of the offered budget spent on selection.",
    )

    @override
    def prepare(
        self,
        metadata: SingleCoefficientEstimationMetadata,
        input: SingleCoefficientEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        SingleCoefficientEstimationMetadata,
        SingleCoefficientEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        coi = metadata.coefficient_of_interest_index
        without_coi = np.delete(np.arange(metadata.dim), coi)
        if without_coi.size == 0:
            raise NotApplicable("DPKendall has no selectable controls.")
        selector_metadata = LinearRegressionMetadata(
            n_samples=metadata.n_samples,
            design_domain=metadata.design_domain.select_coordinates(without_coi),
            response_domain=metadata.response_domain,
        )
        selector_input = LinearRegressionInput(X=input.X[:, without_coi], y=input.y)
        _, _, remaining_budget, child_seed, selection = LinearRegressionDPKendall(
            num_features=self.num_features,
            selection_share=self.selection_share,
        ).prepare(selector_metadata, selector_input, budget, seed)

        selected = np.sort(np.append(without_coi[selection.selected], coi))
        prepared_metadata = SingleCoefficientEstimationMetadata(
            n_samples=metadata.n_samples,
            design_domain=metadata.design_domain.select_coordinates(selected),
            response_domain=metadata.response_domain,
            coefficient_of_interest_index=int(np.searchsorted(selected, coi)),
        )
        prepared_input = SingleCoefficientEstimationInput(
            X=input.X[:, selected], y=input.y
        )
        return prepared_metadata, prepared_input, remaining_budget, child_seed, None

    @override
    def restore(
        self, output: SingleCoefficientEstimationOutput, restoration: None
    ) -> SingleCoefficientEstimationOutput:
        return output
