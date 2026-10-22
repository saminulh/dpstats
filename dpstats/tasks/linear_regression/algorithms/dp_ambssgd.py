"""Adaptive-clipping private mini-batch SGD for linear regression."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm

import math

import numpy as np

from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import parameter_field
from dpstats.budgets import DPBudget, ZCDPBudget
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.utils import split_seed
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _dp_stat(
    X: Array2D,
    y: Array1D,
    w: Array1D,
    domain_size: float,
    failure_probability: float,
    budget: ZCDPBudget,
    seed: int,
) -> float:
    """Privately estimate an approximate maximum residual (Algorithm 3).

    The proof's ``ceil(log2(B / Delta))`` query count and threshold formula
    identify the tested grid as ``2 Delta, ..., 2**q Delta``. This resolves the
    displayed pseudocode's inconsistent zero-based loop without changing the
    search range. ``budget`` is the typed accounting representation of the
    paper's noise multiplier ``alpha``. The house failure probability ``beta``
    computes the displayed ``Delta`` as ``B * beta`` and sets the query
    count to ``ceil(log2(1 / beta))``.

    This transcribes Algorithm 3's displayed stopping rule. Lemma 12's claimed
    upper quantile bound does not follow from that rule: failure to stop at the
    preceding threshold gives only an upper bound of ``s + Gamma`` on its true
    count, not the stated ``s - Gamma``. That affects the paper's utility
    theorem, not the sensitivity-one count releases or their privacy.
    """
    rng = np.random.default_rng(seed)
    discretization_width = domain_size * failure_probability
    n_queries = math.ceil(math.log2(1.0 / failure_probability))
    query_budget = ZCDPBudget(rho=budget.rho / n_queries)
    count_noise_sd = accounting.calibrate_gaussian_scale(query_budget, sensitivity=1.0)
    residuals = np.abs(X @ w - y)
    gamma = discretization_width
    for _ in range(n_queries):
        gamma *= 2.0
        count = int(np.count_nonzero(residuals <= gamma))
        private_count = count + rng.normal(scale=count_noise_sd)
        if private_count >= y.size:
            return gamma
    return gamma


def _run_algorithm_2(
    X: Array2D,
    y: Array1D,
    n: int,
    d: int,
    eta: float,
    R_x: float,
    B: float,
    failure_probability: float,
    a: float,
    b: int,
    s: int,
    budget: ZCDPBudget,
    seed: int,
) -> Array1D:
    """Run the paper's complete DP-AMBSSGD mechanism (Algorithm 2)."""
    block_size = s + b
    T = n // block_size
    if T < 2:
        raise InvalidRegime(
            "DP-AMBSSGD requires enough rows for two complete iteration blocks."
        )
    effective_n = T * block_size

    # Steps 2-4: shuffle once, keep the largest complete-block prefix so that
    # T = N/(b+s) is exact, and initialize. Statistics and gradient slices are
    # disjoint within and across iterations, so both children get the full rho.
    stat_budget = budget
    gradient_budget = budget
    child_seeds = split_seed(seed, 1 + 2 * T)
    shuffle_rng = np.random.default_rng(child_seeds[0])
    order = shuffle_rng.permutation(n)[:effective_n]
    X = X[order]
    y = y[order]

    w = np.zeros(d, dtype=np.float64)
    iterates: list[Array1D] = []
    log_n = math.log(effective_n)
    # Steps 5-11: estimate each private threshold and take one noisy update.
    for t in range(T):
        tau = (b + s) * t
        gamma_t = _dp_stat(
            X[tau : tau + s],
            y[tau : tau + s],
            w,
            B,
            failure_probability,
            stat_budget,
            child_seeds[1 + 2 * t],
        )
        zeta_t = R_x * gamma_t * log_n**a
        X_batch = X[tau + s : tau + s + b]
        y_batch = y[tau + s : tau + s + b]
        gradients = X_batch * (X_batch @ w - y_batch)[:, None]
        clipped = clip_rows(gradients, zeta_t)
        # Replacing one record moves the mean of b rows clipped at zeta_t by at
        # most 2 * zeta_t / b.
        sensitivity = 2.0 * zeta_t / b
        noise_scale = accounting.calibrate_gaussian_scale(gradient_budget, sensitivity)
        gradient_rng = np.random.default_rng(child_seeds[2 + 2 * t])
        noisy_gradient = clipped.mean(axis=0) + noise_scale * gradient_rng.normal(
            size=d
        )
        w = w - eta * noisy_gradient
        iterates.append(w.copy())

    # Step 12: average the final half of the private iterates.
    beta: Array1D = np.mean(iterates[T // 2 :], axis=0)
    return beta


class DPAMBSSGD(BaseAlgorithm):
    """DP-Adaptive-Mini-Batch-Shuffled-SGD. Fit linear regression by private stochastic gradient descent with an adaptive clipping threshold.

    An implementation of Algorithms 2 and 3 from Varshney, Thakurta and Jain
    (2022). Shuffles the records once and uses each in at most one step. In
    each step, privately estimates the typical residual size on a small batch
    to set the clipping threshold, then takes a clipped, noisy gradient step on
    a separate batch. Outputs the average of the last half of the steps. It is
    a zCDP mechanism.

    Requires:
        - A budget family admitting a zCDP implication.
        - Enough records for two complete iteration blocks of
          ``batch_size + stat_sample_size``.

    Parametrization notes:
        - ``learning_rate_multiplier``: Scales Corollary 11's step size
          $η = 1/(4d)$. The default reproduces that corollary.
        - ``covariate_moment_bound``: Corresponds to $R_x$ in the paper. Set to
          unit scale because the paper gives no way to compute it for general
          data.
        - ``domain_size``: Corresponds to $B$ in the paper. Set to unit scale
          for the same reason.
        - ``failure_probability``: A confidence parameter $β$ for the
          residual-scale search, which sets its grid through
          $Δ = B β$. Algorithm 2 effectively uses $β = N^{-200}$, a proof
          device rather than a usable confidence level, so an ordinary one is
          used instead.
        - ``tail_exponent``: Corresponds to $a$ in the paper. The default is
          the Gaussian case of Corollary 11.
        - ``batch_size``: Corresponds to $b$ in the paper. The smallest batch
          size for which Theorem 10's ratio $s = b/10$ gives a whole number
          of statistics records.
        - ``stat_sample_size``: Corresponds to $s$ in the paper. Gives Theorem
          10's ratio $s = b/10$ at the default ``batch_size``.

    Implementation notes:
        - The records are shuffled once and each record used takes part in exactly
          one iteration. When the records do not fill a whole number of blocks,
          the shuffled order is cut to the longest whole-block prefix, and the
          algorithm runs at that smaller public sample size.
        - The paper proves privacy for replacing one record in a fixed-size
          dataset. The guarantee here is for adding or removing one record at the
          declared ``n``. It holds without a group-privacy factor because a
          randomized embedding, used only in the proof, pads or truncates any
          dataset to length ``n`` so that neighbors differ in one position. On
          inputs of the declared size that embedding is exactly the single
          shuffle and prefix executed here, so no padding is ever generated.

    Reference:
        - varshney2022nearly: Algorithms 2 and 3.
    """

    TASK = LinearRegression

    learning_rate_multiplier: float = parameter_field(
        default=1.0,
        gt=0.0,
        description="Multiplier on the Corollary 11 learning-rate policy ``eta = 1 / (4d)``.",
    )
    covariate_moment_bound: float = parameter_field(
        default=1.0,
        gt=0.0,
        description="Constructed bound ``R_x`` satisfying ``E[||x||^2 xx^T] <= R_x^2 E[xx^T]``.",
    )
    domain_size: float = parameter_field(
        default=1.0,
        gt=0.0,
        description="Algorithm 2's public DP-STAT search bound ``B``.",
    )
    failure_probability: float = parameter_field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="House confidence parameter ``beta`` controlling the DP-STAT grid through ``Delta = B * beta``.",
    )
    tail_exponent: float = parameter_field(
        default=0.5,
        gt=0.0,
        description="Public tail exponent ``a``; Gaussian covariates and noise use ``a = 1/2``.",
    )
    batch_size: int = parameter_field(
        default=10,
        gt=0,
        description="Algorithm 2's gradient mini-batch size ``b``.",
    )
    stat_sample_size: int = parameter_field(
        default=1,
        gt=0,
        description="Algorithm 2's statistics-batch size ``s``.",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        available = accounting.spend_as_zcdp(budget)
        n = metadata.n_samples
        d = metadata.dim
        eta = self.learning_rate_multiplier / (4.0 * d)
        beta = _run_algorithm_2(
            X=input.X,
            y=input.y,
            n=n,
            d=d,
            eta=eta,
            R_x=self.covariate_moment_bound,
            B=self.domain_size,
            failure_probability=self.failure_probability,
            a=self.tail_exponent,
            b=self.batch_size,
            s=self.stat_sample_size,
            budget=available,
            seed=seed,
        )
        return LinearRegressionOutput(beta=beta)
