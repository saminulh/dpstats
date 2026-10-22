"""Blockwise Laplace histograms and one Gaussian numerical mean release."""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
)
from dpstats.domains import Domain, OneHotDomain, ProductDomain
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D, IntArray1D
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms.dp_histogram import (
    DPHistogram,
)
from dpstats.tasks.mean_estimation.algorithms.gaussian_mechanism import (
    GaussianMechanism,
)
from dpstats.utils import split_seed


class HistogramGaussian(BaseAlgorithm):
    """Estimate categorical frequencies and a numerical mean.

    Runs ``DPHistogram``, the vector Laplace mechanism (Dwork and Roth, 2014),
    on each nonconstant one-hot block and ``GaussianMechanism``, the analytic
    Gaussian mechanism (Balle and Wang, 2018), on all remaining numerical
    coordinates together. It is an approximate-DP mechanism.

    Requires:
        - A domain and child offer supported by ``GaussianMechanism`` when
          numerical coordinates are active.
        - Pure or approximate DP for active blocks, unless the offer is unconstrained.

    Implementation notes:
        - Each categorical child receives one original ``OneHotDomain``
          part and a pure-DP epsilon share. The numerical child receives the
          domain's exact coordinate selection, preserving its geometry.
        - Coordinates with equal public lower and upper bounds are returned
          exactly and receive no budget share. Coordinates keep their original
          order, without projection, clipping, or renormalization.

    Reference:
        - dwork2014algorithmic: Theorems 3.6 and 3.16.
        - balle2018improving: Theorem 8.
    """

    TASK = MeanEstimation

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        mean_est, histograms, numerical = _partition_domain(metadata.domain)
        num_releases = len(histograms) + int(numerical is not None)
        if num_releases == 0:
            return MeanEstimationOutput(mean_est=mean_est)
        histogram_budget, numerical_budget = _child_budgets(budget, num_releases)
        child_seeds = split_seed(seed, num_releases)
        if numerical is not None:
            coordinates, domain = numerical
            output = GaussianMechanism().run(
                MeanEstimationMetadata(n_samples=metadata.n_samples, domain=domain),
                MeanEstimationInput(X=input.X[:, coordinates]),
                budget=numerical_budget,
                seed=child_seeds[-1],
            )
            mean_est[coordinates] = output.mean_est
        for index, (coordinates, domain) in enumerate(histograms):
            output = DPHistogram().run(
                MeanEstimationMetadata(n_samples=metadata.n_samples, domain=domain),
                MeanEstimationInput(X=input.X[:, coordinates]),
                budget=histogram_budget,
                seed=child_seeds[index],
            )
            mean_est[coordinates] = output.mean_est
        return MeanEstimationOutput(mean_est=mean_est)


def _partition_domain(
    domain: Domain,
) -> tuple[Array1D, list[tuple[slice, OneHotDomain]], tuple[IntArray1D, Domain] | None]:
    """Resolve public constants and active blocks before any child release."""
    parts = domain.parts if isinstance(domain, ProductDomain) else (domain,)
    histograms: list[tuple[slice, OneHotDomain]] = []
    numerical_coordinates: list[int] = []
    mean_est: Array1D = np.zeros(domain.dim)
    start = 0
    for part in parts:
        end = start + part.dim
        if isinstance(part, OneHotDomain) and (part.dim > 1 or part.allow_zero):
            histograms.append((slice(start, end), part))
        else:
            # Bounds identify constants only; the numerical child retains the
            # original domain's exact coordinate image below, not this box.
            bounds = part.bounding_box().bounds
            constant = bounds[:, 0] == bounds[:, 1]
            coordinates = np.arange(start, end)
            mean_est[coordinates[constant]] = bounds[constant, 0]
            numerical_coordinates.extend(int(i) for i in coordinates[~constant])
        start = end

    numerical: IntArray1D = np.array(numerical_coordinates, dtype=np.int64)
    if numerical.size:
        return mean_est, histograms, (numerical, domain.select_coordinates(numerical))
    return mean_est, histograms, None


def _child_budgets(
    budget: DPBudget, num_releases: int
) -> tuple[PureDPBudget, DPBudget]:
    """Allocate equal epsilon shares and keep all delta on the numerical child.

    An unconstrained offer, including approximate DP with delta one, leaves
    every child unconstrained.
    """
    if budget.is_unconstrained:
        unconstrained = PureDPBudget(eps=np.inf)
        return unconstrained, unconstrained
    if isinstance(budget, (ApproxDPBudget, PureDPBudget)):
        epsilon = budget.eps
    else:
        raise InvalidRegime(
            f"HistogramGaussian cannot spend an offered {type(budget).__name__}."
        )
    child_epsilon = epsilon / num_releases
    histogram_budget = PureDPBudget(eps=child_epsilon)
    numerical_budget = (
        ApproxDPBudget(eps=child_epsilon, delta=budget.delta)
        if isinstance(budget, ApproxDPBudget)
        else histogram_budget
    )
    return histogram_budget, numerical_budget
