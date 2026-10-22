"""Linear regression as M-estimation: the residual losses and their transformations.

Each record stacks as ``[x_i | y_i]``, so the leading ``parameter_domain.dim``
coordinates are the design row and the final coordinate is the response, and a
problem declaring one of these losses declares a data domain one dimension
wider than its parameter domain. Least squares is M-estimation under the
squared residual loss and robust regression under the Huber loss; the map is
rowwise and public and the fitted parameter is the coefficient vector.
``ToMEstimation`` declares the coefficient domain from ``parameter_radius``.
"""

from __future__ import annotations

from dpstats.algorithm import Transformation

from typing import Annotated, ClassVar, Literal
from typing_extensions import override

import math

import numpy as np
from pydantic import Field

from dpstats.domains import (
    Domain,
    EuclideanSpace,
    LpBallDomain,
    ProductDomain,
)
from dpstats.budgets import DPBudget
from dpstats.parameters import parameter_field
from dpstats.task import Array1D, Array2D, ArrayND, Task
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
    Reduction,
)
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput as MInput,
    MEstimationMetadata as MMetadata,
    MEstimationOutput as MOutput,
)


class SquaredResidualLoss(MEstimationLoss):
    """Squared error of the residual ``r = y - x @ theta``."""

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        _, residuals = _design_and_residuals(data, theta)
        costs = residuals**2
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        X, residuals = _design_and_residuals(data, theta)
        if reduction == "none":
            return -2 * residuals[:, None] * X
        gradient_sum = -2 * (X.T @ residuals)
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(X)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        design_domain = data_domain.select_coordinates(np.arange(parameter_domain.dim))
        response_domain = data_domain.select_coordinates(
            np.array([parameter_domain.dim])
        )
        design_bound = design_domain.l2_norm_bound
        residual_bound = (
            response_domain.l2_norm_bound
            + design_bound * parameter_domain.l2_norm_bound
        )
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=2 * design_bound * residual_bound,
            gradient_lipschitz_constant=2 * design_bound**2,
        )


class AbsoluteResidualLoss(MEstimationLoss):
    """Absolute error of the residual ``r = y - x @ theta``."""

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        _, residuals = _design_and_residuals(data, theta)
        costs = np.abs(residuals)
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        X, residuals = _design_and_residuals(data, theta)
        if reduction == "none":
            return -np.sign(residuals)[:, None] * X
        gradient_sum = -(X.T @ np.sign(residuals))
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(X)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        design_domain = data_domain.select_coordinates(np.arange(parameter_domain.dim))
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=False,
            loss_lipschitz_constant=design_domain.l2_norm_bound,
            gradient_lipschitz_constant=math.inf,
        )


class HuberResidualLoss(MEstimationLoss):
    """Huber residual loss at ``r = y - x @ theta``."""

    threshold: float = Field(
        gt=0,
        description="Positive residual magnitude at which the loss turns linear.",
    )

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        _, residuals = _design_and_residuals(data, theta)
        magnitudes = np.abs(residuals)
        costs = np.where(
            magnitudes <= self.threshold,
            0.5 * residuals**2,
            self.threshold * magnitudes - 0.5 * self.threshold**2,
        )
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        X, residuals = _design_and_residuals(data, theta)
        clipped = np.clip(residuals, -self.threshold, self.threshold)
        if reduction == "none":
            return -clipped[:, None] * X
        gradient_sum = -(X.T @ clipped)
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(X)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        design_bound = data_domain.select_coordinates(
            np.arange(parameter_domain.dim)
        ).l2_norm_bound
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=False,
            loss_lipschitz_constant=self.threshold * design_bound,
            gradient_lipschitz_constant=design_bound**2,
        )


class RobHytResidualLoss(MEstimationLoss):
    """RobHyt residual loss ``(k^2/2) log cosh(2r/k)``, quadratic then linear.

    Implements Definition 1 of Slavkovic and Molinari (2022) on the residual,
    summed as in its Definition 2. Its derivative ``k tanh(2r/k)`` and second
    derivative ``2 sech^2(2r/k)`` give ``xi_k = kR``, ``lambda_k = 2R^2``.

    Reference:
        - slavkovic2022perturbed: Definitions 1 and 2.
    """

    k: float = Field(gt=0, description="Positive tuning constant in response units.")

    @override
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        _, residuals = _design_and_residuals(data, theta)
        scaled = 2.0 * residuals / self.k
        # log(cosh(u)) = |u| + log1p(exp(-2|u|)) - log(2).
        magnitudes = np.abs(scaled)
        costs = (self.k**2 / 2.0) * (
            magnitudes + np.log1p(np.exp(-2.0 * magnitudes)) - math.log(2.0)
        )
        if reduction == "none":
            return costs
        if reduction == "sum":
            return float(np.sum(costs))
        if reduction == "mean":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        X, residuals = _design_and_residuals(data, theta)
        scores = self.k * np.tanh(2.0 * residuals / self.k)
        if reduction == "none":
            return -scores[:, None] * X
        gradient_sum = -(X.T @ scores)
        if reduction == "sum":
            return gradient_sum
        if reduction == "mean":
            return gradient_sum / len(X)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    def hessian(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        """Return RobHyt Hessians without per-record matrices when reducing.

        Args:
            data: Two-dimensional batch of stacked ``[x | y]`` records.
            theta: Coefficient vector where the residual curvature is evaluated.
            reduction: Record reduction: ``none``, ``sum``, or ``mean``.

        Returns:
            Per-record or reduced d-by-d Hessian matrices.

        Example:
            >>> RobHytResidualLoss(k=1.0).hessian(
            ...     np.array([[1.0, 0.0, 0.0]]), np.zeros(2), reduction="sum"
            ... ).shape
            (2, 2)
        """
        X, residuals = _design_and_residuals(data, theta)
        weights = 2.0 * (1.0 - np.tanh(2.0 * residuals / self.k) ** 2)
        if reduction == "none":
            return weights[:, None, None] * X[:, :, None] * X[:, None, :]
        summed_hessian = X.T @ (weights[:, None] * X)
        if reduction == "sum":
            return summed_hessian
        if reduction == "mean":
            return summed_hessian / len(X)
        raise ValueError(f"Unknown reduction: {reduction!r}")

    @override
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        design_bound = data_domain.select_coordinates(
            np.arange(parameter_domain.dim)
        ).l2_norm_bound
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=self.k * design_bound,
            gradient_lipschitz_constant=2 * design_bound**2,
        )


def _stacked_problem(
    metadata: LRMetadata,
    input: LRInput,
    loss: MEstimationLoss,
    parameter_radius: float | None,
) -> tuple[MMetadata, MInput]:
    """Stack rows as ``[x_i | y_i]`` records under the given loss.

    The coefficient vector is the M-estimation parameter over the domain the
    radius selects, and the declared design and response domains transfer as
    the record domain.
    """
    parameter_domain = (
        EuclideanSpace(dim=metadata.dim)
        if parameter_radius is None
        else LpBallDomain.at_origin(p=2, radius=parameter_radius, dim=metadata.dim)
    )
    prepared_metadata = MMetadata(
        parameter_domain=parameter_domain,
        n_samples=metadata.n_samples,
        loss=loss,
        data_domain=ProductDomain(
            parts=(metadata.design_domain, metadata.response_domain)
        ),
    )
    prepared_input = MInput(data=np.column_stack((input.X, input.y)))
    return prepared_metadata, prepared_input


class ToMEstimation(Transformation):
    """Fit the coefficients by minimizing a configured loss over the records, such as the Huber loss.

    Passes each record's features and response together to a general
    loss-minimization algorithm, with the configured loss of the residual, and
    reports the parameter it returns as the coefficient vector. This spends no
    privacy budget.

    Parametrization notes:
        - ``loss``: The default loss gives the ordinary least-squares
          estimand, so the conversion alone changes no estimand.
        - ``parameter_radius``: The default leaves the parameter in unbounded
          Euclidean space, so the unconstrained estimand is preserved. The
          value ``"sqrt_dim"``, the square root of the coefficient dimension,
          gives the squared-loss regression recipe of Appendix H of Kifer,
          Smith and Thakurta (2012).

    Implementation notes:
        - A finite ``parameter_radius`` is part of the estimand: for a design
          that is not isotropic, the minimizer inside the ball is not the
          projection of the unconstrained fit onto it, so the radius is
          declared on the regression's own scale.

    Reference:
        - kifer2012private: Appendix H.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = MEstimation

    loss: MEstimationLoss = parameter_field(
        default=SquaredResidualLoss(), description="Configured stacked-record loss."
    )
    parameter_radius: (
        Annotated[float, Field(gt=0, allow_inf_nan=False)] | Literal["sqrt_dim"] | None
    ) = parameter_field(
        default=None,
        description=(
            "L2 coefficient radius in original units: a positive scalar, "
            "'sqrt_dim' for the square root of the coefficient dimension, or "
            "None for unbounded Euclidean space; the ball is part of the estimand."
        ),
    )

    @override
    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ):
        radius = (
            math.sqrt(metadata.dim)
            if self.parameter_radius == "sqrt_dim"
            else self.parameter_radius
        )
        prepared_metadata, prepared_input = _stacked_problem(
            metadata,
            input,
            self.loss,
            radius,
        )
        return prepared_metadata, prepared_input, budget, seed, None

    @override
    def restore(self, output: MOutput, restoration: None):
        return LROutput(beta=output.theta)


def _design_and_residuals(data: Array2D, theta: Array1D) -> tuple[Array2D, Array1D]:
    """Split stacked ``[x | y]`` records into design rows and residuals."""
    X = data[:, :-1]
    return X, data[:, -1] - X @ theta


__all__ = [
    "AbsoluteResidualLoss",
    "HuberResidualLoss",
    "RobHytResidualLoss",
    "SquaredResidualLoss",
    "ToMEstimation",
]
