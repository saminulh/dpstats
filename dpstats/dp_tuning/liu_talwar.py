"""Liu and Talwar DP private selection algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np

from dpstats.parameters import parameter_field, search_range
from dpstats.budgets import accounting
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.utils import split_seed
from dpstats.dp_tuning.base import DPHyperparameterTuned, best_candidate

TMetadata = Any
TInput = Any


class DPSelectionLiuTalwar(DPHyperparameterTuned):
    """Random Stopping. Tune hyperparameters privately: try random settings until a coin flip says stop, and output the best.

    An implementation of Algorithm 2 from Liu and Talwar (2019). Tries a
    randomly chosen hyperparameter setting with the wrapped algorithm in each
    round, flips a biased coin after each round to decide whether to stop, and
    outputs the best result. It is a pure-DP or approximate-DP mechanism,
    matching the offered budget.

    Requires:
        - Under an ``ApproxDPBudget``, an epsilon of at least
          ``1.5 * delta / ceil(-log(delta / 2) / gamma)``.

    Parametrization notes:
        - ``gamma``: Corresponds to the stopping probability $γ$ of Algorithm
          2. The expected number of trials is $1/γ$; Liu and Talwar (2019)
          prescribe no value.

    Reference:
        - liu2019private: Algorithm 2.
    """

    gamma: float = parameter_field(
        default=0.05,
        gt=0.0,
        le=1.0,
        search=search_range(0.01, 0.1),
        description="Probability of stopping in each iteration.",
    )

    def tune(
        self,
        tuning_meta: TMetadata,
        tuning_data: TInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Select the best hyperparameters via iterative private evaluation.

        Each iteration:
        1. Sample a random hyperparameter configuration from the grid.
        2. Train the algorithm on the full tuning data with the
           ``1 - scoring_share`` part of the per-run budget.
        3. Score it with the DP selection cost on the same data at the
           ``scoring_share`` part.
        4. Flip a biased coin (probability gamma) to decide whether to stop.

        The class docstring carries the per-iteration budget each family
        receives.

        Args:
            tuning_meta: Metadata for the tuning data.
            tuning_data: Input data for the tuning.
            budget: Total privacy budget for tuning.
            seed: Single int seed; the per-iteration seed stream is built from it.

        Returns:
            The already-trained winning output and its hyperparameters.

        Raises:
            InvalidRegime: If an approximate-DP eps is below the reserve its
                delta sets aside.
            NoRelease: If every evaluated candidate released bottom.
        """
        candidates: list[tuple[float, dict[str, Any], Any | None]] = []
        param_keys = list(self.alg_hyperparam_grid.keys())
        param_values = list(self.alg_hyperparam_grid.values())

        if isinstance(budget, ApproxDPBudget):
            # Theorem 3.4: half of delta caps the trials, the rest sets each trial's
            # delta_base, and each trial gets a third of eps after the reserve.
            eps = budget.eps
            delta = budget.delta
            delta_2 = delta / 2
            max_trials = float(np.ceil(-np.log(delta_2) / self.gamma))
            delta_base = (((delta - delta_2) / max_trials) ** 2) / 2
            min_eps = 3 * np.sqrt(2 * delta_base)
            if eps < min_eps:
                raise InvalidRegime(f"eps must be at least {min_eps}, got {eps}.")
            run_budget: DPBudget = ApproxDPBudget(
                eps=(eps - min_eps) / 3,
                delta=delta_base,
            )
        else:
            pure_budget = (
                budget
                if isinstance(budget, PureDPBudget)
                else accounting.spend_as_pure(budget)
            )
            max_trials = np.inf
            # Pure-DP random stopping costs three trials' epsilon, with no cap.
            run_budget = PureDPBudget(eps=pure_budget.eps / 3)

        iter_count = 0
        while True:
            iter_seed = split_seed(seed, iter_count + 1)[-1]
            param_seed, alg_seed, selection_cost_seed, coin_seed = split_seed(
                iter_seed, 4
            )

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
                    run_budget,
                    alg_seed,
                    selection_cost_seed,
                    earlier_candidate_ran=bool(candidates),
                )
            )
            iter_count += 1
            coin_rng = np.random.default_rng(coin_seed)
            if coin_rng.random() < self.gamma or iter_count >= max_trials:
                break

        return best_candidate(candidates)
