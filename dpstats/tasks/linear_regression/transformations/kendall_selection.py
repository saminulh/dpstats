"""Kendall feature-selection transformations for :class:`CompositeAlgorithm` compositions.

Kendall feature selection appears twice: the released Dick-Gillenwater-Joseph
mechanism, and the exact selector that serves as its oracle in the
value-of-privacy diagnostic. They share a reduction and a restoration but carry
different privacy arguments, so they are separate classes rather than one class
with a mode flag. The Kendall mathematics lives in the task's ``kendall``
module; transformations reuse it rather than restating it.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import PrivacyInterpretation, Transformation
from dpstats.exceptions import InvalidRegime, NotApplicable

from typing import ClassVar

import numpy as np

from dpstats.budgets import accounting
from dpstats.budgets import DPBudget
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.task import IntArray1D
from dpstats.utils import split_seed
from dpstats.tasks.linear_regression.transformations._selector_utils import (
    SelectorRestoration,
    prepend_intercept,
    reduce_to_selection,
    scatter_selected,
)
from dpstats.tasks.linear_regression.transformations.kendall import (
    _select_features_exactly,
    _select_features_privately,
)
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)


def _selectable_columns(metadata: LRMetadata, num_features: int) -> IntArray1D:
    """Return the non-intercept columns a selector may choose among."""
    selectable = np.arange(1 if metadata.has_intercept else 0, metadata.dim)
    if num_features > selectable.size:
        raise NotApplicable(
            f"num_features={num_features} exceeds the {selectable.size} "
            "selectable non-intercept columns."
        )
    return selectable


class DPKendall(Transformation):
    """Privately select the features most rank-correlated (Kendall's τ) with the response, and pass only those to the next step. Upon receiving its coefficients for the selected features, fill in zero for every feature that was dropped.

    An implementation of Algorithm 1 from Dick, Gillenwater and Joseph (2023).
    Privately picks a fixed number of features that are strongly
    rank-correlated with the response and not redundant with each other,
    always keeping any declared intercept without scoring it, and passes only
    those columns to the next step.

    Requires:
        - At least two declared samples.
        - A ``num_features`` within the selectable non-intercept columns;
          otherwise ``NotApplicable``.

    Parametrization notes:
        - ``num_features``: Corresponds to $k$ in the paper. Set to the value
          Section 4.4 suggests as the more relevant setting; the paper's other
          studied value is also configurable.
        - ``selection_share``: The fraction of the budget spent on selection.
          Set to the share the paper allocates to selection in Section 4.2.

    Reference:
        - dick2023better: Algorithm 1.
        - dick2023code: Listing ``dp_kendall_feature_selection``.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression
    num_features: int = parameter_field(
        default=5, gt=0, description="Number of non-intercept columns to retain."
    )
    selection_share: float = parameter_field(
        default=0.05,
        gt=0,
        lt=1,
        description="Fraction of the offered budget spent on selection.",
    )

    @override
    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ):
        if metadata.n_samples < 2:
            raise InvalidRegime("DPKendall requires at least two samples.")
        selectable = _selectable_columns(metadata, self.num_features)

        selection_seed, downstream_seed = split_seed(seed, 2)
        selection_budget, downstream_budget = budget.split(
            (self.selection_share, 1.0 - self.selection_share)
        )
        selected_within_candidates = _select_features_privately(
            input.X[:, selectable],
            input.y,
            self.num_features,
            accounting.spend_as_pure(selection_budget),
            selection_seed,
        )
        selected = prepend_intercept(
            selectable[selected_within_candidates], metadata.has_intercept
        )
        reduced_metadata, reduced_input, restoration = reduce_to_selection(
            metadata, input, selected
        )
        return (
            reduced_metadata,
            reduced_input,
            downstream_budget,
            downstream_seed,
            restoration,
        )

    @override
    def restore(self, output: LROutput, restoration: SelectorRestoration):
        return scatter_selected(output, restoration)


class ExactKendallSelection(Transformation):
    """Select the features the private Kendall selection aims for, without privacy, and pass only those to the next step. Upon receiving its coefficients for the selected features, fill in zero for every feature that was dropped.

    An adaptation of Algorithm 1 from Dick, Gillenwater and Joseph (2023)
    without its noisy maximizer. Picks the same features the private Kendall
    selection aims for, but without noise, and passes only those columns to the
    next step. It spends no privacy budget and is not private; set against the
    private selection, it shows what privacy costs.

    Requires:
        - At least two declared samples.
        - A ``num_features`` within the selectable non-intercept columns;
          otherwise ``NotApplicable``.

    Parametrization notes:
        - ``num_features``: Corresponds to $k$ in the paper. Set to the larger
          of the two values the paper studies in Section 4.2, not the smaller
          one it recommends for :class:`DPKendall`.

    Reference:
        - dick2023better: Algorithm 1 and Section 4.2.
        - dick2023code: Listing ``dp_kendall_feature_selection``.
    """

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.DIAGNOSTIC
    )

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression
    num_features: int = parameter_field(
        default=10, gt=0, description="Number of non-intercept columns to retain."
    )

    @override
    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ):
        if metadata.n_samples < 2:
            raise InvalidRegime("ExactKendallSelection requires at least two samples.")
        selectable = _selectable_columns(metadata, self.num_features)

        tie_seed, downstream_seed = split_seed(seed, 2)
        selected_within_candidates = _select_features_exactly(
            input.X[:, selectable], input.y, self.num_features, tie_seed
        )
        selected = prepend_intercept(
            selectable[selected_within_candidates], metadata.has_intercept
        )
        reduced_metadata, reduced_input, restoration = reduce_to_selection(
            metadata, input, selected
        )
        return (
            reduced_metadata,
            reduced_input,
            budget,
            downstream_seed,
            restoration,
        )

    @override
    def restore(self, output: LROutput, restoration: SelectorRestoration):
        return scatter_selected(output, restoration)
