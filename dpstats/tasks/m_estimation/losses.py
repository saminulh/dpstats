"""The M-estimation loss interface and the properties a loss states.

Every loss evaluates batches of records and derives its mathematical
properties over the public data and parameter domains. Concrete
losses live with the task that presents its problem as M-estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from dpstats.domains import Domain
from dpstats.task import Array1D, Array2D, ArrayND
from dpstats.transport.configuration import PolymorphicModel


Reduction = Literal["none", "sum", "mean"]


@dataclass(frozen=True)
class LossProperties:
    """The properties of a loss function ``l(theta; z)``.

    Attributes:
        convex: Whether the loss is convex in theta.
        generalized_linear: Whether the loss depends on theta only through a
            linear predictor ``l(theta; z) = f(x^T theta; y)``.
        continuously_twice_differentiable: Whether the per-record loss is C^2
            in theta everywhere.
        loss_lipschitz_constant: L2 Lipschitz constant of the per-record loss in
            theta over the parameter domain.
        gradient_lipschitz_constant: L2 Lipschitz constant of the per-record
            gradient in theta over the parameter domain.
    """

    convex: bool
    generalized_linear: bool
    continuously_twice_differentiable: bool
    loss_lipschitz_constant: float
    gradient_lipschitz_constant: float

    def __post_init__(self) -> None:
        if not self.loss_lipschitz_constant >= 0:
            raise ValueError(
                "loss_lipschitz_constant must be nonnegative, either finite or "
                f"infinite; the declared domains give {self.loss_lipschitz_constant}."
            )
        if not self.gradient_lipschitz_constant >= 0:
            raise ValueError(
                "gradient_lipschitz_constant must be nonnegative, either finite "
                f"or infinite; the declared domains give "
                f"{self.gradient_lipschitz_constant}."
            )


class MEstimationLoss(PolymorphicModel, ABC):
    """An M-estimation loss function.

    A loss function ``l(theta; z)`` is a nonnegative cost for a record ``z``
    at a parameter vector ``theta``. This class defines the interface for
    evaluating the loss, its gradient, and optionally its Hessian, as well as
    for computing its mathematical properties over the public data and parameter domains.
    """

    @abstractmethod
    def __call__(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> float | Array1D:
        """Evaluate the loss for a batch at ``theta``.

        Args:
            data: Two-dimensional batch, with one record per row.
            theta: Parameter vector where the loss is evaluated.
            reduction: If ``none``, return the per-record costs; if ``sum`` or ``mean``, return the scalar sum or mean.

        Returns:
            Per-record costs or their scalar sum or mean.

        Example:
            >>> import numpy as np
            >>> from dpstats.tasks.mean_estimation.transformations.m_estimation import (
            ...     SquaredErrorLoss,
            ... )
            >>> SquaredErrorLoss()(np.array([[1.0]]), np.zeros(1), reduction="sum")
            1.0
        """

    @abstractmethod
    def gradient(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND:
        """Evaluate a gradient or selected subgradient for each record.

        Args:
            data: Two-dimensional batch, with one record per row.
            theta: Parameter vector where the gradient is evaluated.
            reduction: If ``none``, return the per-record gradients; if ``sum`` or ``mean``, return the scalar sum or mean.

        Returns:
            Per-record gradients or their vector sum or mean.

        Example:
            >>> import numpy as np
            >>> from dpstats.tasks.mean_estimation.transformations.m_estimation import (
            ...     SquaredErrorLoss,
            ... )
            >>> SquaredErrorLoss().gradient(
            ...     np.array([[1.0]]), np.zeros(1), reduction="sum"
            ... ).tolist()
            [-2.0]
        """

    def hessian(
        self, data: Array2D, theta: Array1D, *, reduction: Reduction = "none"
    ) -> ArrayND | None:
        """Return an analytic parameter Hessian, if supplied.

        ``none`` returns one d-by-d Hessian per record, while
        ``sum`` and ``mean`` reduce over records. A loss that does not
        supply this optional operation returns ``None``. Availability is fixed
        by the configured loss implementation, rather than the records or a
        solver outcome.

        Args:
            data: Two-dimensional batch, with one record per row.
            theta: Parameter vector where the Hessian is evaluated.
            reduction: If ``none``, return the per-record Hessians; if ``sum`` or ``mean``, return the scalar sum or mean.

        Returns:
            Per-record or reduced Hessian, or ``None`` when unavailable.

        Example:
            >>> import numpy as np
            >>> from dpstats.tasks.mean_estimation.transformations.m_estimation import (
            ...     SquaredErrorLoss,
            ... )
            >>> SquaredErrorLoss().hessian(np.zeros((1, 1)), np.zeros(1)) is None
            True
        """
        return None

    @abstractmethod
    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        """Return properties about this loss on the declared public domains."""


__all__ = ["LossProperties", "MEstimationLoss", "Reduction"]
