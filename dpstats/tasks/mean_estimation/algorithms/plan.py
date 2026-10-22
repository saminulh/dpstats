"""Variance-aware private mean estimation from the PLAN family."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, ZCDPBudget, accounting
from dpstats.exceptions import InvalidRegime


import numpy as np

from dpstats.parameters import parameter_field, search_values
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.utils import split_seed
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


def _binary_private_quantile(
    values: Array1D,
    quantile: float,
    lower: float,
    upper: float,
    budget: ZCDPBudget,
    seed: int,
    *,
    num_rounds: int,
    population: int,
) -> float:
    """Run PLAN's binary private-quantile mechanism.

    Each round releases a sensitivity-one count with an equal share of the
    supplied zCDP budget. The final interval midpoint is post-processing of
    those releases. ``population`` is the public stage population — the
    declared ``n`` for row stages and the public number of groups for the
    variance stage — so the target rank never reads the realized private array
    length.
    """
    rng = np.random.default_rng(seed)
    target_rank = quantile * population
    round_budget = ZCDPBudget(rho=budget.rho / num_rounds)
    count_noise_scale = accounting.calibrate_gaussian_scale(
        round_budget, sensitivity=1.0
    )

    for _ in range(num_rounds):
        midpoint = (lower + upper) / 2
        count = np.count_nonzero(values <= midpoint)
        noisy_count = count + rng.normal(0, count_noise_scale)
        if noisy_count <= target_rank:
            lower = midpoint
        else:
            upper = midpoint

    return (lower + upper) / 2


class PLANBinarySearchGaussian(BaseAlgorithm):
    """PLAN. Estimate a mean privately, with noise shaped to each coordinate's spread.

    An implementation of Algorithm 1 from Aumüller, Lebeda, Nelson and Pagh
    (2024). Privately re-centers the data, estimates each coordinate's spread
    from differences between pairs of records, rescales by it, clips records at
    a radius found by private binary search, and adds Gaussian noise to the
    mean. It is a zCDP mechanism.

    Requires:
        - A finite coordinate bounding box on ``metadata.domain``.
        - ``metadata.n_samples`` of at least ``2 * variance_group_size``.
        - A public sample size and budget admitting a positive norm quantile.
        - A budget family admitting a zCDP implication.

    Parametrization notes:
        - ``variance_group_size``: Corresponds to $g$ in the paper. Set to the
          value of the Gaussian recipe; Appendix D also evaluates groups of
          four.
        - ``num_rounds``: Corresponds to $T$ in the paper. Set as in Section 7
          and the authors' artifact, which also uses ten rounds.
        - ``failure_probability``: Corresponds to $β$ in the paper. Set as in
          the authors' artifact.
        - ``preprocessing_budget_fraction``: Corresponds to $a$ in the paper.
          Set as in Section 7 and the authors' artifact.
        - ``centering_budget_fraction``: Corresponds to $b$ in the paper. Set
          as in the authors' artifact.
        - ``threshold_budget_fraction``: Corresponds to $c$ in the paper. Set
          as in Section 7 and the authors' artifact.

    Implementation notes:
        - The guarantee is zCDP for adding or removing one record at the declared
          ``n``, proved through a randomized embedding of every dataset into
          length ``n``. On data of the declared size the embedding has the same
          distribution as running the paper's algorithm directly, which is what
          runs; the padding exists only in the proof.
        - Of the paper's options, this fixes the L2 norm and the binary-search
          quantile.
        - The group-size-dependent Wilson-Hilferty correction and the range
          searched for the norm follow the final paper. The authors' code
          triples the data in a step its accounting does not cover; that step
          is left out.

    Reference:
        - aumuller2024plan: Algorithm 1.
    """

    TASK = MeanEstimation

    variance_group_size: int = parameter_field(
        default=1,
        ge=1,
        search=search_values(1, 4),
        description="Number of coordinate-difference pairs averaged together when estimating each marginal Gaussian scale.",
    )
    num_rounds: int = parameter_field(
        default=20,
        ge=1,
        search=search_values(10, 20),
        description="Number of binary-search rounds used for the coordinate-centering and clipping-threshold searches.",
    )
    failure_probability: float = parameter_field(
        default=0.1,
        gt=0,
        lt=1,
        description="Failure probability governing the high-probability tail bound used to select the norm-clipping radius.",
    )
    preprocessing_budget_fraction: float = parameter_field(
        default=0.25,
        gt=0,
        lt=1,
        description="Fraction of the zCDP budget assigned to preprocessing (centering and variance estimation) rather than the final release.",
    )
    centering_budget_fraction: float = parameter_field(
        default=0.25,
        gt=0,
        lt=1,
        description="Fraction of the preprocessing budget assigned to coordinate centering rather than variance estimation.",
    )
    threshold_budget_fraction: float = parameter_field(
        default=0.25,
        gt=0,
        lt=1,
        description="Fraction of the release budget assigned to the clipping-threshold search rather than the final noisy mean.",
    )

    def _estimate_gaussian_variances(
        self,
        X: Array2D,
        *,
        coordinate_bound: float,
        budget: ZCDPBudget,
        seed: int,
    ) -> Array1D:
        """Privately estimate Gaussian coordinate variances from grouped pairs."""
        n_samples, dim = X.shape
        group_size = self.variance_group_size
        num_groups = n_samples // (2 * group_size)
        permutation_seed, *quantile_seeds = split_seed(seed, dim + 1)
        coordinate_budget = ZCDPBudget(rho=budget.rho / dim)

        permutation_rng = np.random.default_rng(permutation_seed)
        grouped_rows = X[
            permutation_rng.permutation(n_samples)[: 2 * group_size * num_groups]
        ].reshape(num_groups, group_size, 2, dim)
        grouped_pair_differences = np.mean(
            (grouped_rows[:, :, 0, :] - grouped_rows[:, :, 1, :]) ** 2 / 2,
            axis=1,
        )
        private_group_medians = np.array(
            [
                _binary_private_quantile(
                    grouped_pair_differences[:, coordinate],
                    0.5,
                    0,
                    2 * coordinate_bound**2,
                    coordinate_budget,
                    quantile_seeds[coordinate],
                    num_rounds=self.num_rounds,
                    population=num_groups,
                )
                for coordinate in range(dim)
            ]
        )

        correction = (1 - 2 / (9 * group_size)) ** 3
        return private_group_medians / correction

    def _plan_tail_count(self, n: int, rho_2: ZCDPBudget) -> float:
        """Return Algorithm 1's instantiated number of clipped records ``k``."""
        T = self.num_rounds
        beta = self.failure_probability
        return np.sqrt(n) + np.sqrt(T * np.log(3 * T / beta) / (2 * rho_2.rho))

    def _run_plan(
        self,
        X: Array2D,
        sigma_squared_hat: Array1D,
        *,
        M: float,
        p: float,
        rho_1: ZCDPBudget,
        rho_2: ZCDPBudget,
        rho_3: ZCDPBudget,
        seed: int,
    ) -> Array1D:
        """Run Algorithm 1 with supplied variance estimates and stage budgets."""
        n, d = X.shape
        T = self.num_rounds
        beta = self.failure_probability
        k = self._plan_tail_count(n, rho_2)
        q = (n - k) / n

        center_seed, threshold_seed, final_seed = split_seed(seed, 3)
        center_quantile_seeds = split_seed(center_seed, d)
        coordinate_budget = ZCDPBudget(rho=rho_1.rho / d)

        mu_tilde = np.array(
            [
                _binary_private_quantile(
                    X[:, coordinate],
                    0.5,
                    -M,
                    M,
                    coordinate_budget,
                    center_quantile_seeds[coordinate],
                    num_rounds=T,
                    population=n,
                )
                for coordinate in range(d)
            ]
        )

        # Store the diagonal matrix Sigma_hat as its diagonal vector.
        Sigma_hat = sigma_squared_hat
        Sigma_hat_power = Sigma_hat ** (1 / (p + 2))
        y = (X - mu_tilde) / Sigma_hat_power

        # Section 5 bounds the transformed norm universe with log(n), rather
        # than the log(d) transcription used in the accompanying artifact.
        U = np.sqrt(np.log(n) * np.log(1 / beta) * np.sum(np.sqrt(Sigma_hat)))
        transformed_norms = np.minimum(np.linalg.norm(y, axis=1), U)
        C = _binary_private_quantile(
            transformed_norms,
            q,
            0,
            U,
            rho_2,
            threshold_seed,
            num_rounds=T,
            population=n,
        )

        clipped_y = clip_rows(y, C)
        eta_scale = accounting.calibrate_gaussian_scale(rho_3, sensitivity=2 * C)
        final_rng = np.random.default_rng(final_seed)
        eta = final_rng.normal(0, eta_scale, size=(d,))
        return mu_tilde + ((np.sum(clipped_y, axis=0) + eta) / n) * Sigma_hat_power

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        n = metadata.n_samples
        d = metadata.dim
        minimum_samples = 2 * self.variance_group_size
        if n < minimum_samples:
            raise InvalidRegime(
                "PLANBinarySearchGaussian requires n_samples >= "
                "2 * variance_group_size."
            )

        bounds = metadata.domain.bounding_box().bounds
        if not np.all(np.isfinite(bounds)):
            raise InvalidRegime(
                "PLANBinarySearchGaussian requires a finite coordinate bounding box."
            )
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        if np.array_equal(lower_bounds, upper_bounds):
            return MeanEstimationOutput(mean_est=lower_bounds.copy())

        available_budget = accounting.spend_as_zcdp(budget)
        preprocessing_budget, release_budget = available_budget.split(
            (
                self.preprocessing_budget_fraction,
                1 - self.preprocessing_budget_fraction,
            ),
        )
        rho_1, variance_budget = preprocessing_budget.split(
            (
                self.centering_budget_fraction,
                1 - self.centering_budget_fraction,
            ),
        )
        rho_2, rho_3 = release_budget.split(
            (
                self.threshold_budget_fraction,
                1 - self.threshold_budget_fraction,
            ),
        )
        k = self._plan_tail_count(n, rho_2)
        if k >= n:
            raise InvalidRegime(
                "PLANBinarySearchGaussian requires a positive norm quantile for "
                "the public n_samples and privacy budget."
            )
        variance_seed, plan_seed = split_seed(seed, 2)

        coordinate_bound = np.max(np.abs(bounds))
        sigma_squared_hat = self._estimate_gaussian_variances(
            input.X,
            coordinate_bound=coordinate_bound,
            budget=variance_budget,
            seed=variance_seed,
        )
        sigma_hat = np.sqrt(sigma_squared_hat)
        sigma_hat = sigma_hat + np.sum(sigma_hat) / d
        mean_est = self._run_plan(
            input.X,
            sigma_hat**2,
            M=coordinate_bound,
            p=2,
            rho_1=rho_1,
            rho_2=rho_2,
            rho_3=rho_3,
            seed=plan_seed,
        )
        return MeanEstimationOutput(mean_est=mean_est)
