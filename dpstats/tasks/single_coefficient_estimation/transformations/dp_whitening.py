"""Privately whiten nuisance coordinates with Gaussian mean and covariance estimates."""

import math
from typing import ClassVar

import numpy as np

from dpstats.algorithm import CompositeAlgorithm, Transformation
from dpstats.budgets import DPBudget, accounting
from dpstats.domains import LpBallDomain, ProductDomain
from dpstats.exceptions import NotApplicable
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks import (
    CovarianceEstimation,
    MeanEstimation,
    SingleCoefficientEstimation,
)
from dpstats.tasks.covariance_estimation.transformations import (
    CovarianceToMeanEstimation,
)
from dpstats.tasks.mean_estimation.algorithms import GaussianMechanism
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationMetadata,
    SingleCoefficientEstimationOutput,
)
from dpstats.utils import split_seed


class DPWhitening(Transformation):
    """Privately rescale the nuisance columns, keeping the coefficient of interest unchanged.

    Leaves the coefficient-of-interest column, the response and any declared
    intercept as they are, and privately whitens the other columns: when an
    intercept is declared and the coefficient of interest is not the intercept,
    it centers them by their private mean and scales them by their private
    covariance; otherwise it scales them by their private second moment. The
    least-squares coefficient of interest is unchanged.

    Requires:
        - A nuisance column other than the declared intercept; otherwise ``NotApplicable``.

    Parametrization notes:
        - ``preparation_share``: The fraction of the budget spent on the mean
          and covariance; the rest goes to the next step. Within it, the
          covariance gets four times the mean's share.
        - Fixed eigenvalue floor: the larger of ``sigma_cov*sqrt(d)`` and
          ``1e-8/n``, where ``sigma_cov`` is the covariance noise scale. It
          keeps the scaling stable where the noisy covariance has small or
          zero eigenvalues.

    Implementation notes:
        - Centering is used only when a declared intercept can absorb the
          shift; other constant columns are whitened like any nuisance column.
        - The nuisance columns' declared domain is mapped through the same
          transformation, so the next step receives valid bounds.
    """

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

    preparation_share: float = parameter_field(
        default=0.25,
        gt=0,
        lt=1,
        description="Incoming privacy share for mean and covariance estimation.",
    )

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
        """Prepare transformed nuisance rows and the remaining privacy budget."""
        coi_ind = metadata.coefficient_of_interest_index
        center_nuisances = metadata.has_intercept and coi_ind != 0
        retained_inds = np.array([0, coi_ind] if center_nuisances else [coi_ind])
        nuisance_inds = np.delete(np.arange(metadata.dim), retained_inds)
        d = len(nuisance_inds)
        if d == 0:
            raise NotApplicable("No nuisance coordinates to whiten.")
        preparation_budget, child_budget = budget.split(
            (self.preparation_share, 1 - self.preparation_share)
        )
        covariance_budget, mean_budget = preparation_budget.split((4, 1))
        mean_seed, covariance_seed, child_seed = split_seed(seed, 3)
        domain = metadata.design_domain.select_coordinates(nuisance_inds)
        Z = input.X[:, nuisance_inds]

        mean = (
            GaussianMechanism()
            .run(
                MeanEstimation.Metadata(n_samples=metadata.n_samples, domain=domain),
                MeanEstimation.Input(X=Z),
                mean_budget,
                mean_seed,
            )
            .mean_est
        )
        covariance_algorithm = CompositeAlgorithm(
            transformations=(CovarianceToMeanEstimation(),),
            algorithm=GaussianMechanism(),
        )
        covariance = covariance_algorithm.run(
            CovarianceEstimation.Metadata(n_samples=metadata.n_samples, domain=domain),
            CovarianceEstimation.Input(X=Z),
            covariance_budget,
            covariance_seed,
        ).covariance_est

        bound = domain.l2_norm_bound
        sensitivity = math.hypot(bound, bound * bound) / metadata.n_samples
        sigma = accounting.calibrate_gaussian_scale(covariance_budget, sensitivity)
        floor = max(1e-8 / metadata.n_samples, sigma * math.sqrt(d))
        moment = covariance if center_nuisances else covariance + np.outer(mean, mean)
        values, vectors = np.linalg.eigh(moment)
        whitening = vectors / np.sqrt(np.maximum(values, floor))
        center, radius = domain.l2_enclosing_ball()
        if center_nuisances:
            Z = Z - mean
            center = center - mean
        child_metadata = SingleCoefficientEstimation.Metadata(
            n_samples=metadata.n_samples,
            coefficient_of_interest_index=int(center_nuisances),
            design_domain=ProductDomain(
                parts=(
                    metadata.design_domain.select_coordinates(retained_inds),
                    LpBallDomain(
                        p=2,
                        center=center @ whitening,
                        radius=radius * np.linalg.norm(whitening, ord=2),
                    ),
                ),
            ),
            response_domain=metadata.response_domain,
        )
        child_input = SingleCoefficientEstimation.Input(
            X=np.column_stack((input.X[:, retained_inds], Z @ whitening)), y=input.y
        )
        return child_metadata, child_input, child_budget, child_seed, None

    def restore(
        self, output: SingleCoefficientEstimationOutput, restoration: None
    ) -> SingleCoefficientEstimationOutput:
        """Return the child's coefficient of interest unchanged."""
        return output
