"""Papernot & Steinke DP private hyperparameter selection."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from dp_accounting import dp_event, rdp

from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.budgets import ApproxDPBudget, DPBudget, ZCDPBudget, accounting
from dpstats.parameters import parameter_field, search_range
from dpstats.task import AnyArray1D
from dpstats.utils import bisect_monotone_boundary
from dpstats.utils import split_seed
from dpstats.dp_tuning.base import DPHyperparameterTuned, best_candidate

TMetadata = Any
TInput = Any


def _sample_tnb(eta: float, gamma: float, seed: int) -> int:
    """Draw a single truncated-negative-binomial repeat count ``K >= 1``.

    The TNB(``eta``, ``gamma``) distribution of Papernot & Steinke. The realized
    value is the secret repeat count and must NEVER be released; only the
    distribution family and its parameters are public.
    """
    rng = np.random.default_rng(seed)
    if eta == 0.0:
        return int(rng.logseries(1.0 - gamma))
    while True:
        k = int(rng.negative_binomial(eta, gamma))
        if k >= 1:
            return k


_PS_RDP_ORDERS: tuple[float, ...] = (
    *(i / 10 for i in range(11, 111)),
    *range(12, 64),
    128.0,
    256.0,
    512.0,
    1024.0,
)


def _expected_repeats_tnb(eta: float, gamma: float) -> float:
    """Return the expected truncated-negative-binomial repeat count."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1).")
    if eta == 0.0:
        return (1.0 / gamma - 1.0) / math.log(1.0 / gamma)
    return eta * (1.0 - gamma) / (gamma * (1.0 - gamma**eta))


def _zcdp_base_rdp_curve(
    budget: ZCDPBudget,
) -> tuple[AnyArray1D, AnyArray1D]:
    """Return the explicit RDP curve of a zCDP base mechanism."""
    orders: AnyArray1D = np.asarray(_PS_RDP_ORDERS, dtype=float)
    rdp_eps: AnyArray1D = budget.rho * orders
    return orders, rdp_eps


def _largest_fitting_rho(
    charged_epsilon: Callable[[float], float],
    eps: float,
    rho_upper_bound: float | None = None,
) -> float:
    """Return the largest per-run ``rho`` a route may charge within ``eps``.

    A route's charged epsilon increases with the per-run ``rho`` it hands the
    tuned algorithm, so the budgets that fit are an interval anchored at
    zero and bisection finds its upper end. ``rho_upper_bound`` carries a
    theorem's own cap on the per-run budget. Numerical failures inside
    ``charged_epsilon`` propagate.
    """
    return bisect_monotone_boundary(
        lambda rho: rho == 0.0 or charged_epsilon(rho) <= eps,
        0.0,
        1.0,
        satisfying_endpoint="lower",
        upper_bound=rho_upper_bound,
    )


def _poisson_repeat_rdp_bound(
    order: float,
    base_rdp: float,
    delta_hat: float,
    mu: float,
) -> float:
    """Evaluate the corrected Poisson-repeat RDP bound at one Renyi order.

    Papernot--Steinke Theorem 6 omits the common ``K = 0`` mass. Here that
    fixed, data-independent law contributes ``exp(-mu)`` once to the Renyi
    power sum.
    """
    log_zero_repeats = -mu
    log_positive_repeats = math.log(mu) + (order - 1.0) * (base_rdp + mu * delta_hat)
    return np.logaddexp(log_zero_repeats, log_positive_repeats) / (order - 1.0)


def _ps_epsilon_poisson(per_run_budget: ZCDPBudget, mu: float, delta: float) -> float:
    """Convert the corrected Poisson-repeat RDP curve to epsilon at ``delta``."""
    orders, base_rdp = _zcdp_base_rdp_curve(per_run_budget)
    base_accountant = rdp.RdpAccountant()
    base_accountant.compose(dp_event.ZCDpEvent(per_run_budget.rho))
    tuned_orders: list[float] = []
    tuned_rdp: list[float] = []
    for lam, eps_lam in zip(orders, base_rdp):
        eps_hat = math.log(1.0 + 1.0 / (lam - 1.0))
        delta_hat = base_accountant.get_delta(eps_hat)
        tuned_orders.append(lam)
        tuned_rdp.append(
            _poisson_repeat_rdp_bound(
                order=lam,
                base_rdp=eps_lam,
                delta_hat=delta_hat,
                mu=mu,
            )
        )
    epsilon, _ = rdp.compute_epsilon(list(tuned_orders), tuned_rdp, delta)
    return epsilon


def _ps_epsilon_tnb(
    per_run_budget: ZCDPBudget,
    eta: float,
    gamma: float,
    delta: float,
) -> float:
    """Evaluate Papernot--Steinke Corollary 4 for a zCDP base mechanism."""
    expected_k = _expected_repeats_tnb(eta, gamma)
    log_ek = math.log(expected_k)
    log_inv_gamma = math.log(1.0 / gamma)
    rho = per_run_budget.rho
    surcharge = 2.0 * (1.0 + eta) * math.sqrt(rho * log_inv_gamma) - eta * rho
    threshold = 1.0 + math.sqrt(log_ek / rho)
    orders, _ = _zcdp_base_rdp_curve(per_run_budget)
    tuned_rdp: list[float] = []
    for lam in orders:
        if lam <= threshold:
            eps_prime = 2.0 * math.sqrt(rho * log_ek) + surcharge
        else:
            eps_prime = rho * (lam - 1.0) + log_ek / (lam - 1.0) + surcharge
        tuned_rdp.append(eps_prime)
    epsilon, _ = rdp.compute_epsilon(list(orders), tuned_rdp, delta)
    return epsilon


@lru_cache(maxsize=256)
def _calibrate_poisson_rho(eps: float, delta: float, mu: float) -> float:
    """Calibrate the corrected-Theorem-6 per-run ``rho``, cached by scalars."""
    return _largest_fitting_rho(
        lambda rho: _ps_epsilon_poisson(ZCDPBudget(rho=rho), mu=mu, delta=delta),
        eps,
    )


@lru_cache(maxsize=256)
def _calibrate_tnb_rho(eps: float, delta: float, eta: float, gamma: float) -> float:
    """Calibrate the Corollary 4 per-run ``rho``, cached by scalars.

    Corollary 4 holds only for a per-run ``rho`` at most ``log(1/gamma)``.
    """
    return _largest_fitting_rho(
        lambda rho: _ps_epsilon_tnb(
            ZCDPBudget(rho=rho), eta=eta, gamma=gamma, delta=delta
        ),
        eps,
        rho_upper_bound=math.log(1.0 / gamma),
    )


class _PapernotSteinkeSelection(DPHyperparameterTuned, ABC):
    """Repeat-and-select body shared by the Papernot-Steinke routes.

    A concrete route supplies the two things its own privacy argument decides:
    the per-run budget the theorem calibrates and the public distribution
    the secret repeat count ``K`` is drawn from. Each concrete class documents
    both. The realized ``K`` is secret and is never released; only the
    distribution family and its parameters are public.
    """

    def tune(
        self,
        tuning_meta: TMetadata,
        tuning_data: TInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Select the best hyperparameters via Papernot-Steinke private selection."""
        budget = accounting.spend_as_approx_dp(budget)
        per_run_budget = self._calibrated_per_run_budget(budget)

        repeat_seed, loop_seed = split_seed(seed, 2)
        num_repeats = self._draw_repeats(repeat_seed)

        param_keys = list(self.alg_hyperparam_grid.keys())
        param_values = list(self.alg_hyperparam_grid.values())
        candidates: list[tuple[float, dict[str, Any], Any | None]] = []

        for iter_seed in split_seed(loop_seed, num_repeats):
            param_seed, alg_seed, selection_cost_seed = split_seed(iter_seed, 3)
            param_seeds = split_seed(param_seed, len(param_keys))
            params = {
                key: self.sample_param(values, child_seed)
                for key, values, child_seed in zip(
                    param_keys, param_values, param_seeds
                )
            }
            candidates.append(
                self.evaluate_candidate(
                    tuning_meta,
                    tuning_data,
                    params,
                    per_run_budget,
                    alg_seed,
                    selection_cost_seed,
                    earlier_candidate_ran=bool(candidates),
                )
            )

        return best_candidate(candidates)

    @abstractmethod
    def _calibrated_per_run_budget(self, budget: ApproxDPBudget) -> ZCDPBudget:
        """Return the per-run budget this route's theorem calibrates.

        Both routes yield an approximate-DP guarantee, so the offered
        budget is an ``ApproxDPBudget`` and the calibrated ``ZCDPBudget``
        is handed unchanged to the tuned algorithm and its selection cost.
        """

    @abstractmethod
    def _draw_repeats(self, seed: int) -> int:
        """Draw the secret repeat count from this route's public distribution.

        A route whose distribution admits a zero count releases that bottom
        outcome here, because no candidate the accounting covers was drawn.
        """


class DPSelectionPapernotSteinkeTNB(_PapernotSteinkeSelection):
    """Tune hyperparameters privately: try a random number of random settings and output the best.

    An implementation of the method in Corollary 4 of Papernot and Steinke
    (2022). Draws a secret number of runs from a truncated negative binomial
    distribution, tries a randomly chosen hyperparameter setting with the
    wrapped algorithm in each run, and outputs the best result. It is an
    approximate-DP mechanism.

    Requires:
        - An ``ApproxDPBudget`` or an unconstrained budget.

    Parametrization notes:
        - ``eta``: Corresponds to the shape parameter $η$ of Corollary 4's
          truncated negative binomial. The default makes the number of runs
          geometric, the repetition law of Liu and Talwar (2019) and one of the
          laws Figure 3 of Papernot and Steinke (2022) compares.
        - ``gamma``: Corresponds to that distribution's success parameter $γ$.
          Papernot and Steinke (2022) prescribe no value.

    Reference:
        - papernot2022hyperparameter: Corollary 4.
        - liu2019private: Algorithm 2.
    """

    eta: float = parameter_field(
        default=1.0,
        ge=0.0,
        description="Shape of the truncated negative binomial; 0 is logarithmic.",
    )
    gamma: float = parameter_field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Success parameter in (0, 1); smaller values draw more repeats "
            "and cap the per-run budget higher."
        ),
    )

    def _calibrated_per_run_budget(self, budget: ApproxDPBudget) -> ZCDPBudget:
        """Return the Corollary 4 per-run budget for this outer budget."""
        return ZCDPBudget(
            rho=_calibrate_tnb_rho(budget.eps, budget.delta, self.eta, self.gamma)
        )

    def _draw_repeats(self, seed: int) -> int:
        """Draw ``K >= 1`` from ``TNB(eta, gamma)``."""
        return _sample_tnb(self.eta, self.gamma, seed)


class DPSelectionPapernotSteinkePoisson(_PapernotSteinkeSelection):
    """Tune hyperparameters privately: try a Poisson-distributed number of random settings and output the best.

    An implementation of the method in Theorem 6 of Papernot and Steinke
    (2022). Draws a secret number of runs from a Poisson distribution, tries a
    randomly chosen hyperparameter setting with the wrapped algorithm in each
    run, and outputs the best result. It is an approximate-DP mechanism.

    Requires:
        - A finite ``ApproxDPBudget``.

    Parametrization notes:
        - ``mu``: Corresponds to the Poisson rate $μ$ of Theorem 6, the
          expected number of runs. Papernot and Steinke (2022) prescribe no
          value.

    Implementation notes:
        - Theorem 6 as published omits the probability of drawing zero runs.
          The accounting includes it, and a draw of zero runs outputs nothing.

    Reference:
        - papernot2022hyperparameter: Theorem 6.
    """

    mu: float = parameter_field(
        default=10.0,
        gt=0.0,
        search=search_range(4.0, 128.0, scale="log"),
        description="Poisson rate of the repeat count.",
    )

    def _calibrated_per_run_budget(self, budget: ApproxDPBudget) -> ZCDPBudget:
        """Return the corrected-Theorem-6 per-run budget for this budget."""
        if budget.is_unconstrained:
            raise InvalidRegime(
                "DPSelectionPapernotSteinkePoisson refuses an unconstrained "
                "offer: no accepted record states the limit of its "
                "corrected Theorem 6 charge."
            )
        return ZCDPBudget(rho=_calibrate_poisson_rho(budget.eps, budget.delta, self.mu))

    def _draw_repeats(self, seed: int) -> int:
        """Draw ``K`` from ``Poisson(mu)``, releasing bottom when it is zero."""
        num_repeats = int(np.random.default_rng(seed).poisson(self.mu))
        if num_repeats == 0:
            raise NoRelease(
                "Papernot-Steinke Poisson selection drew no candidates (the "
                "Poisson bottom outcome); no result is released for this run."
            )
        return num_repeats
