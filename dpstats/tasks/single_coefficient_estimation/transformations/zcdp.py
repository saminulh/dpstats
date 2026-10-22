"""Convert an SCE composition's remaining offer to a shared zCDP budget."""

from typing import ClassVar

from typing_extensions import override

from dpstats.algorithm import Transformation
from dpstats.budgets import DPBudget, ZCDPBudget, accounting
from dpstats.task import Task
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)


class ToZCDP(Transformation):
    """Convert the remaining privacy budget to zCDP for subsequent stages.

    Finds, with the shared accountant, the zCDP spend the offered budget
    admits, which later transformations and the child algorithm allocate as a
    common budget. Leaves the task, metadata, input, seed, and output
    unchanged.

    Requires:
        - A zCDP, approximate-DP, or unconstrained budget.
    """

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

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
        ZCDPBudget,
        int,
        None,
    ]:
        return metadata, input, accounting.spend_as_zcdp(budget), seed, None

    @override
    def restore(
        self, output: SingleCoefficientEstimationOutput, restoration: None
    ) -> SingleCoefficientEstimationOutput:
        return output
