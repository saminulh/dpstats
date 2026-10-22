"""RKW objective perturbation for configured M-estimation losses."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace
from dpstats.task import Array1D
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks.m_estimation.algorithms._objective_perturbation_utils import (
    admit_loss,
    calibrate_rkw_objective_ridge,
    solve_perturbed_objective,
)
from dpstats.parameters import parameter_field


class ObjectivePerturbationRedberg(BaseAlgorithm):
    """Objective Perturbation. Minimize the loss after adding a random Gaussian linear term and a ridge penalty.

    An implementation of the method in Definition 2.1 of Redberg, Koskela and
    Wang (2023), with the privacy analysis of its Theorem 3.1. Adds a random
    Gaussian linear term and a ridge penalty to the average loss, minimizes the
    result numerically, and outputs the minimizer with no further noise. It is
    an approximate-DP mechanism.

    Requires:
        - A ``EuclideanSpace`` parameter domain.
        - A convex generalized-linear loss with finite bounds on the declared
          domains.
        - A strictly positive ``gradient_lipschitz_constant``.
        - An ``ApproxDPBudget`` or an unconstrained budget.
        - A budget and ``objective_noise_ratio`` whose Section 5 ridge rule has
          a finite solution.

    Parametrization notes:
        - ``objective_noise_ratio``: Corresponds to $f$ in Redberg, Koskela and
          Wang (2023). Set as in their Section 5 (p. 9).
        - Resolved $λ$: The regularization is resolved at run time as the least
          ridge above $β$ whose objective profile meets $(ε, δ)$ at $σ = f$
          times the exact Gaussian mechanism's scale, following RKW's Section 5
          rule. It is not a hyperparameter.
        - Fixed $C$ and $β$: The gradient and smoothness bounds are the
          configured loss's properties and are not exposed.

    Implementation notes:
        - The privacy guarantee is for the exact minimizer, but a numerical
          root is output, so the class carries the theoretical privacy label
          rather than the standard one.
        - The privacy argument needs a continuously differentiable loss. That
          is assumed, not checked, because no declared property of a loss
          records it.

    Reference:
        - redberg2023improving: Definition 2.1 and Theorem 3.1.
    """

    TASK = MEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.THEORETICAL
    )

    objective_noise_ratio: float = parameter_field(
        default=1.3,
        gt=0.0,
        allow_inf_nan=False,
        description="Objective Gaussian deviation relative to the exact Gaussian mechanism.",
    )

    @override
    def run(
        self,
        metadata: MEstimationMetadata,
        input: MEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MEstimationOutput:
        domain = metadata.parameter_domain
        if not isinstance(domain, EuclideanSpace):
            raise InvalidRegime(
                "ObjectivePerturbationRedberg minimizes over the whole parameter "
                f"space; the declared {type(domain).__name__} feasible set is "
                "outside its objective profile."
            )
        properties = metadata.loss_properties
        admit_loss(metadata.loss, properties)
        gradient_bound = properties.loss_lipschitz_constant
        smoothness_bound = properties.gradient_lipschitz_constant
        if smoothness_bound <= 0.0:
            raise InvalidRegime(
                "ObjectivePerturbationRedberg requires a strictly positive loss "
                f"smoothness bound; {type(metadata.loss).__name__} gives "
                f"{smoothness_bound} on the declared domains, and the least "
                "ridge above it is no positive ridge."
            )
        curvature, objective_noise_scale = calibrate_rkw_objective_ridge(
            budget,
            gradient_bound,
            smoothness_bound,
            self.objective_noise_ratio,
        )
        rng = np.random.default_rng(seed)
        perturbation: Array1D = rng.normal(size=domain.dim) * objective_noise_scale
        # Each record contributes at most C to the summed score; the sampled
        # perturbation is independent of data and completes this gradient scale.
        gradient_scale = metadata.n_samples * gradient_bound + float(
            np.linalg.norm(perturbation)
        )
        theta = solve_perturbed_objective(
            metadata.loss,
            input.data,
            curvature,
            perturbation,
            domain.dim,
            gradient_scale,
            type(self).__name__,
        )
        return MEstimationOutput(theta=theta)
