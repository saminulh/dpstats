"""KST objective perturbation for configured M-estimation losses."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.optimize import minimize

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm, PrivacyInterpretation
from dpstats.budgets import ApproxDPBudget, DPBudget, PureDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.task import Array1D, Array2D
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks.m_estimation.algorithms._objective_perturbation_utils import (
    admit_loss,
    kst_objective_terms,
    perturbed_gradient,
    solve_perturbed_objective,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss


def _solve_on_ball(
    loss: MEstimationLoss,
    data: Array2D,
    domain: LpBallDomain,
    curvature: float,
    perturbation: Array1D,
    smoothness: float,
    gradient_scale: float,
) -> Array1D:
    """Minimize KST's summed perturbed objective on the declared L2 ball."""

    def objective(theta: Array1D) -> float:
        return (
            float(
                loss(data, theta, reduction="sum")
                + 0.5 * curvature * (theta @ theta)
                + perturbation @ theta
            )
            / smoothness
        )

    def gradient(theta: Array1D) -> Array1D:
        return perturbed_gradient(loss, data, theta, curvature, perturbation)

    try:
        result = minimize(
            objective,
            x0=domain.project(np.zeros(domain.dim)),
            jac=lambda theta: gradient(theta) / smoothness,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda theta: domain.radius**2
                - np.sum((theta - domain.center) ** 2),
                "jac": lambda theta: -2.0 * (theta - domain.center),
            },
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        theta: Array1D = result.x
        # A boundary optimum has a zero projected gradient, not a zero gradient.
        residual = smoothness * (
            theta - domain.project(theta - gradient(theta) / smoothness)
        )
        feasible = np.linalg.norm(theta - domain.center) <= (1.0 + 1e-7) * domain.radius
        stationary = np.linalg.norm(residual) <= 1e-7 * gradient_scale
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise RuntimeError(
            "ObjectivePerturbationKifer: the constrained solve failed."
        ) from error
    if not np.all(np.isfinite(theta)) or not feasible or not stationary:
        raise RuntimeError(
            "ObjectivePerturbationKifer: the constrained optimizer did not reach "
            "the required feasibility and first-order tolerances."
        )
    return theta


class ObjectivePerturbationKifer(BaseAlgorithm):
    """Objective Perturbation. Minimize the loss after adding a random linear term and a ridge penalty.

    An implementation of Algorithm 1 from Kifer, Smith and Thakurta (2012).
    Adds a random linear term and a ridge penalty to the average loss,
    minimizes the result numerically, and outputs the minimizer with no further
    noise. The random term is Gaussian under approximate differential privacy
    or comes from a radial law under pure differential privacy. It is a
    pure-DP or approximate-DP mechanism, matching the offered budget.

    Requires:
        - A finite ``PureDPBudget`` or ``ApproxDPBudget``.
        - A convex, generalized-linear C2 loss.
        - Finite domain gradient bounds and finite, positive ambient
          smoothness.
        - An ``EuclideanSpace`` or positive-radius L2-ball parameter domain.

    Implementation notes:
        - The ridge penalty stays centered at zero even when the parameter
          domain is a ball centered elsewhere.
        - The minimizer is found numerically. Over the whole space this is a
          root of the gradient. Over a ball it is SLSQP on the objective
          divided by its public smoothness bound (``ftol=1e-14``,
          ``maxiter=1000``), checked for feasibility and projected gradient at
          a relative ``1e-7``; numerical failure raises ``RuntimeError``. The
          privacy guarantee is for the exact minimizer, which neither route
          certifies.

    Reference:
        - kifer2012private: Algorithm 1.
    """

    TASK = MEstimation

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.THEORETICAL
    )

    @override
    def run(
        self,
        metadata: MEstimationMetadata,
        input: MEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MEstimationOutput:
        if budget.is_unconstrained:
            raise InvalidRegime(
                "ObjectivePerturbationKifer requires a finite privacy budget; it "
                "does not define a no-privacy numerical limit."
            )
        if not isinstance(budget, (PureDPBudget, ApproxDPBudget)):
            raise InvalidRegime(
                "ObjectivePerturbationKifer supports finite PureDPBudget and "
                "ApproxDPBudget offers only."
            )
        domain = metadata.parameter_domain
        if not (
            isinstance(domain, EuclideanSpace)
            or isinstance(domain, LpBallDomain)
            and domain.p == 2
            and domain.radius > 0
        ):
            raise InvalidRegime(
                "ObjectivePerturbationKifer requires EuclideanSpace or a "
                "positive-radius L2 ball as its parameter domain."
            )
        properties = metadata.loss_properties
        admit_loss(metadata.loss, properties)
        if not properties.continuously_twice_differentiable:
            raise InvalidRegime(
                "ObjectivePerturbationKifer requires a continuously twice "
                f"differentiable loss; {type(metadata.loss).__name__} is not one."
            )
        gradient_bound = properties.loss_lipschitz_constant
        smoothness_bound = properties.gradient_lipschitz_constant
        if smoothness_bound <= 0:
            raise InvalidRegime(
                "ObjectivePerturbationKifer requires positive loss smoothness "
                "to define a strongly convex perturbed objective."
            )
        curvature, perturbation = kst_objective_terms(
            gradient_bound,
            smoothness_bound,
            domain.dim,
            budget,
            seed,
        )
        gradient_scale = metadata.n_samples * gradient_bound + float(
            np.linalg.norm(perturbation)
        )
        if isinstance(domain, EuclideanSpace):
            theta = solve_perturbed_objective(
                metadata.loss,
                input.data,
                curvature,
                perturbation,
                domain.dim,
                gradient_scale,
                type(self).__name__,
            )
        else:
            theta = _solve_on_ball(
                metadata.loss,
                input.data,
                domain,
                curvature,
                perturbation,
                metadata.n_samples * smoothness_bound + curvature,
                gradient_scale + curvature * domain.l2_norm_bound,
            )
        return MEstimationOutput(theta=theta)
