"""The ``CoinPress`` algorithm."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, ZCDPBudget, accounting
from dpstats.exceptions import InvalidRegime


import numpy as np

from dpstats.parameters import parameter_field, search_values
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


def _mvm(
    d: int,
    n: int,
    X: Array2D,
    c: Array1D,
    r: float,
    budget: ZCDPBudget,
    beta: float,
    rng: np.random.Generator,
) -> tuple[Array1D, float]:
    """Run one CoinPress mean-refinement step (Biswas et al. Alg. 1, MVM).

    The released quantity is ``c + sum_i(X_proj_i - c) / n``, where every
    recentered term has norm at most ``r + gamma_1`` and ``n`` is public.
    Its add/remove sensitivity is therefore ``(r + gamma_1) / n``. The center
    ``c`` is a prior release and may be reused by post-processing.
    The next radius uses the Gaussian scale actually released, preserving the
    paper's concentration argument under this add/remove calibration.
    """
    gamma_1 = np.sqrt(d + 2 * np.sqrt(d * np.log(n / beta)) + 2 * np.log(n / beta))
    gamma_2 = np.sqrt(d + 2 * np.sqrt(d * np.log(1 / beta)) + 2 * np.log(1 / beta))
    recentered = clip_rows(X - c[None, :], r + gamma_1)
    sensitivity = (r + gamma_1) / n
    sigma = accounting.calibrate_gaussian_scale(budget, sensitivity)
    Z = c + recentered.sum(axis=0) / n + rng.normal(0, sigma, size=(d,))
    new_r = gamma_2 * np.sqrt(1 / n + sigma**2)
    return (Z, new_r)


def _mvm_rec(
    d: int,
    n: int,
    X: Array2D,
    c: Array1D,
    r: float,
    t: int,
    budgets: list[ZCDPBudget],
    beta: float,
    rng: np.random.Generator,
) -> Array1D:
    """Run CoinPress Algorithm 2 (MVMRec) with one budget per MVM call.

    ``beta`` is the overall failure budget. Each MVM call has two tail
    events: the earlier calls receive ``beta / (4 * (t - 1))`` per event and
    the final call receives ``beta / 4`` per event, as in Algorithm 2.
    """
    for i in range(t - 1):
        c, r = _mvm(d, n, X, c, r, budgets[i], beta / (4 * (t - 1)), rng)
    c, _ = _mvm(d, n, X, c, r, budgets[t - 1], beta / 4, rng)
    return c


class CoinPress(BaseAlgorithm):
    """Refine a private mean estimate over progressively smaller radii.

    An implementation of Algorithm 2 (MVMRec) from Biswas, Dong, Kamath and
    Ullman (2020). Starts from a large ball known to contain the mean, then in
    each round clips the data to the current ball, privately estimates the
    mean, and shrinks the ball around that estimate. Outputs the last round's
    estimate. It is a zCDP mechanism.

    Requires:
        - A finite L2 enclosing radius on ``metadata.domain``.
        - A budget family admitting a zCDP implication.

    Parametrization notes:
        - ``num_iters``: Corresponds to $t$ in the paper. Set to the value
          Section 4.1 reports most often; Section 4.1 also finds ten rounds
          robust.
        - ``final_budget_fraction``: The share of the budget the last round
          spends. Set to Section 4.1's practical split; the theorem gives the
          last round half.
        - ``failure_probability``: Corresponds to $β$ in the paper. The paper
          allows any value in $(0, 1)$ and gives none for Algorithm 2.

    Reference:
        - biswas2020coinpress: Algorithm 2.
    """

    TASK = MeanEstimation

    num_iters: int = parameter_field(
        default=2,
        ge=1,
        search=search_values(1, 2, 3, 4, 10),
        description="Number of CoinPress refinement iterations.",
    )
    final_budget_fraction: float = parameter_field(
        default=0.75,
        gt=0.0,
        lt=1.0,
        search=search_values(0.5, 0.75),
        description="Fraction of the zCDP budget assigned to the final refinement round.",
    )
    failure_probability: float = parameter_field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Overall probability of estimation failure.",
    )

    @override
    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        c, r = metadata.domain.l2_enclosing_ball()
        if not np.isfinite(r):
            raise InvalidRegime("CoinPress requires a finite initial enclosing radius.")
        rng = np.random.default_rng(seed)
        rho_budget = accounting.spend_as_zcdp(budget)
        t = self.num_iters
        if t == 1:
            step_budgets = [rho_budget]
        else:
            final_rho = self.final_budget_fraction * rho_budget.rho
            early_rho = (1 - self.final_budget_fraction) * rho_budget.rho / (t - 1)
            step_budgets = [
                *[ZCDPBudget(rho=early_rho) for _ in range(t - 1)],
                ZCDPBudget(rho=final_rho),
            ]
        beta = self.failure_probability
        d = metadata.dim
        n = metadata.n_samples
        X = input.X
        mean_est = _mvm_rec(d, n, X, c, r, t, step_budgets, beta, rng)
        return MeanEstimationOutput(mean_est=mean_est)
