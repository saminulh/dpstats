"""Example Algorithm declaration for template_task."""

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.budgets import accounting
from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.parameters import parameter_field, search_range
from dpstats.budgets import DPBudget

from .. import TemplateTask
from ..types import TemplateTaskInput, TemplateTaskMetadata, TemplateTaskOutput


class LaplaceMean(BaseAlgorithm):
    """Example private mean algorithm for ``TemplateTask``.

    Outputs the empirical mean of the scalar observations with Laplace noise
    calibrated to the add/remove-one sensitivity of the centered public-``n``
    mean. Requires a bounded domain in metadata.
    """

    TASK = TemplateTask

    # Keep the template informal until the complete mechanism has an accepted
    # review supporting a stronger class.
    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.INFORMAL
    )

    noise_multiplier: float = parameter_field(
        default=1.0,
        ge=1.0,
        search=search_range(1.0, 4.0),
        description="Dimensionless conservative multiplier on the minimum Laplace scale; values at least one preserve the charged guarantee.",
    )

    @override
    def run(
        self,
        metadata: TemplateTaskMetadata,
        input: TemplateTaskInput,
        budget: DPBudget,
        seed: int,
    ) -> TemplateTaskOutput:
        rng = np.random.default_rng(seed)
        _, radius = metadata.domain.l2_enclosing_ball()
        mean = float(np.mean(input.data))
        # Sensitivity of the centered public-``n`` mean, ``c + sum(x - c) / n``
        # at the domain's enclosing-ball center ``c``: adding or removing one
        # admitted record changes it by one centered contribution, whose
        # magnitude is at most the domain radius. On the platform's
        # ``input.data.shape[0] == n_samples`` manifold, that expression is the
        # empirical mean output above.
        sensitivity = radius / metadata.n_samples

        scale = self.noise_multiplier * accounting.calibrate_laplace_scale(
            budget, sensitivity
        )
        noisy_mean = mean + rng.laplace(0, scale)
        return TemplateTaskOutput(mean_est=noisy_mean)
