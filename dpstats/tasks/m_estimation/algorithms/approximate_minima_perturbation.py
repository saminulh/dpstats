"""Approximate minima perturbation under a configured M-estimation loss."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import ApproxDPBudget, DPBudget, accounting
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace
from dpstats.parameters import parameter_field
from dpstats.task import Array1D, Array2D
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks.m_estimation.algorithms._objective_perturbation_utils import (
    admit_loss,
    calibrate_rkw_amp,
    perturbed_gradient,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss


def _fixed_accelerated_gradient_steps(
    n: int,
    gradient_bound: float,
    smoothness_bound: float,
    regularization: float,
    gradient_tolerance: float,
) -> int:
    """Return a sufficient number of Nesterov gradient updates.

    Bubeck (2015), Theorem 3.18 bounds the gradient-step iterate after K
    updates by A * exp(-K / sqrt(kappa)) in squared gradient norm, with
    A = M * (lambda + M) * (n * C / lambda)**2 and kappa = M / lambda.
    """
    smoothness = n * smoothness_bound + regularization
    initial_distance_bound = n * gradient_bound / regularization
    gradient_bound_squared = (
        smoothness * (regularization + smoothness) * initial_distance_bound**2
    )
    if gradient_bound_squared <= gradient_tolerance**2:
        return 0

    required_reduction = gradient_bound_squared / gradient_tolerance**2
    root_condition_number = math.sqrt(smoothness / regularization)
    return math.ceil(root_condition_number * math.log(required_reduction))


def _run_fixed_accelerated_gradient(
    loss: MEstimationLoss,
    data: Array2D,
    regularization: float,
    perturbation: Array1D,
    smoothness_bound: float,
    n: int,
    num_steps: int,
) -> Array1D:
    """Run Nesterov updates and return the gradient-step iterate y.

    The initializer cancels the ridge and perturbation in the gradient.
    Bubeck (2015), Theorem 3.18 certifies y, not the extrapolated iterate x.
    """
    smoothness = n * smoothness_bound + regularization
    root_condition_number = math.sqrt(smoothness / regularization)
    momentum = (root_condition_number - 1.0) / (root_condition_number + 1.0)
    x: Array1D = -perturbation / regularization
    y: Array1D = x.copy()
    for _ in range(num_steps):
        y_next = (
            x
            - perturbed_gradient(loss, data, x, regularization, perturbation)
            / smoothness
        )
        x = (1.0 + momentum) * y_next - momentum * y
        y = y_next
    return y


class AMP(BaseAlgorithm):
    """Approximate Minima Perturbation. Minimize a randomly perturbed, regularized loss approximately, then add Gaussian noise to the result.

    An implementation of Algorithm 1 from Iyengar et al. (2019), with the
    privacy analysis of Theorems 3.2 and 4.1 in Redberg, Koskela and Wang
    (2023). Adds a random linear term and a ridge penalty to the average loss,
    runs a fixed number of accelerated optimization steps on it, and adds
    Gaussian noise to the result to cover the gap to the exact minimizer. It is
    an approximate-DP mechanism.

    Requires:
        - A ``EuclideanSpace`` parameter domain.
        - A convex generalized-linear loss with finite bounds on the declared
          domains.
        - A strictly positive ``gradient_lipschitz_constant``.
        - For finite execution, an ``ApproxDPBudget`` and a positive gradient
          bound.
        - For an unconstrained budget, an ``objective_noise_ratio`` above 1.

    Parametrization notes:
        - ``objective_noise_ratio``: Corresponds to $f$ in Redberg, Koskela
          and Wang (2023). Set as in their Section 5 evaluation.
        - ``gradient_tolerance``: Corresponds to the tolerance $τ$ on the
          summed gradient in Redberg, Koskela and Wang (2023). The default
          ``"inverse_n"`` sets $τ = 1/n$, the tolerance $γ = 1/n^2$ on the
          averaged objective in the code of Iyengar et al. (2019); a number
          fixes a constant tolerance.
        - ``output_budget_fraction``: Corresponds to $η$ in Iyengar et al.
          (2019), the share of the budget spent on the output noise. Set as in
          their simulations.

    Implementation notes:
        - The ridge penalty is the smallest one for which the Rényi-DP bound
          of Theorem 3.2 of Redberg, Koskela and Wang (2023) meets the
          objective's share of the budget, with the objective noise set to
          ``objective_noise_ratio`` times the noise a plain Gaussian mechanism
          would need on the same share.
        - The output share of the budget is taken from both epsilon and delta.
        - The privacy argument needs a continuously differentiable loss. That
          is assumed, not checked, because no declared property of a loss
          records it.
        - The half-Huber loss is differentiable only once but is admitted: the
          points where its second derivative jumps form sets of measure zero,
          which can be removed from the change-of-variables argument of
          Redberg, Koskela and Wang.
        - The minimizer is approximated by a number of accelerated gradient
          steps fixed in advance by Theorem 3.18 of Bubeck (2015), starting
          from ``-b / lambda``,
          where ``b`` is the random linear term. In exact arithmetic the result
          then meets the resolved tolerance; there is no stopping rule based
          on the residual and no exact-solve mode.

    Reference:
        - iyengar2019towards: Algorithm 1.
        - iyengar2019code: Listing ``amp_run_classification``.
        - redberg2023improving: Theorems 3.2 and 4.1.
        - canonne2022discrete: Proposition 2.11.
        - bubeck2015convex: Theorem 3.18.
    """

    TASK = MEstimation

    objective_noise_ratio: float = parameter_field(
        default=1.3,
        gt=0.0,
        allow_inf_nan=False,
        description="Objective Gaussian deviation relative to the C-sensitive Gaussian.",
    )
    gradient_tolerance: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | Literal["inverse_n"]
    ) = parameter_field(
        default="inverse_n",
        description=(
            "Summed-gradient tolerance of the approximate minimizer, "
            "or 'inverse_n' for 1 / n_samples."
        ),
    )
    output_budget_fraction: float = parameter_field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description="Fraction of epsilon and delta reserved for the output Gaussian.",
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
                "AMP minimizes over the whole "
                f"parameter space; the declared {type(domain).__name__} "
                "feasible set is outside its objective profile."
            )

        properties = metadata.loss_properties
        admit_loss(metadata.loss, properties)
        gradient_bound = properties.loss_lipschitz_constant
        smoothness_bound = properties.gradient_lipschitz_constant
        if smoothness_bound <= 0.0:
            raise InvalidRegime(
                "AMP requires a strictly positive "
                f"loss smoothness bound; {type(metadata.loss).__name__} gives "
                f"{smoothness_bound} on the declared domains, and the least "
                "ridge above it is no positive ridge."
            )

        gradient_tolerance = (
            1.0 / metadata.n_samples
            if self.gradient_tolerance == "inverse_n"
            else self.gradient_tolerance
        )
        if budget.is_unconstrained:
            if self.objective_noise_ratio <= 1.0:
                raise InvalidRegime(
                    "AMP has no unconstrained limit at objective_noise_ratio <= 1: "
                    "its calibration is infeasible at every epsilon."
                )
            regularization = smoothness_bound
            objective_noise_scale = output_noise_scale = 0.0
        else:
            if not isinstance(budget, ApproxDPBudget):
                raise InvalidRegime(
                    "Finite RKW AMP execution requires an ApproxDPBudget."
                )
            if gradient_bound <= 0.0:
                raise InvalidRegime(
                    "Finite RKW AMP execution requires a positive gradient bound."
                )
            objective_budget, output_budget = budget.split(
                (1.0 - self.output_budget_fraction, self.output_budget_fraction)
            )
            regularization, objective_noise_scale = calibrate_rkw_amp(
                objective_budget,
                gradient_bound,
                smoothness_bound,
                self.objective_noise_ratio,
            )
            # Sensitivity 2 * gradient_tolerance / lambda covers the distance to the
            # exact minimizer; the two releases compose as approximate DP.
            output_noise_scale = accounting.calibrate_gaussian_scale(
                output_budget, 2.0 * gradient_tolerance / regularization
            )

        rng = np.random.default_rng(seed)
        perturbation: Array1D = rng.normal(size=domain.dim) * objective_noise_scale
        num_steps = _fixed_accelerated_gradient_steps(
            metadata.n_samples,
            gradient_bound,
            smoothness_bound,
            regularization,
            gradient_tolerance,
        )
        theta = _run_fixed_accelerated_gradient(
            metadata.loss,
            input.data,
            regularization,
            perturbation,
            smoothness_bound,
            metadata.n_samples,
            num_steps,
        )
        return MEstimationOutput(
            theta=theta + rng.normal(size=domain.dim) * output_noise_scale
        )
