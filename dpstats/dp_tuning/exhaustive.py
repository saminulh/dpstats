"""Exhaustive-composition DP hyperparameter selection."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, cast

from pydantic import field_validator

from dpstats.budgets import DPBudget
from dpstats.utils import split_seed
from dpstats.dp_tuning.base import DPHyperparameterTuned, best_candidate

TMetadata = Any
TInput = Any


class DPExhaustiveCompose(DPHyperparameterTuned):
    """Tune hyperparameters privately: try every setting on a grid and output the best.

    Runs the wrapped algorithm once for every combination of the listed
    hyperparameter values and outputs the best result.

    Parametrization notes:
        - ``alg_hyperparam_grid``: Takes discrete candidate lists only, not
          search ranges, since every candidate is trained.

    Implementation notes:
        - Each of the ``m`` candidates gets ``1/m`` of the offered budget.

    Reference:
        - dwork2014algorithmic: Theorem 3.16.
        - bun2016concentrated: Lemma 7.
        - dong2022gaussian: Corollary 2.
    """

    @field_validator("alg_hyperparam_grid", mode="before")
    @classmethod
    def _validate_discrete_grid(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        discrete_grid: dict[str, list[Any]] = {}
        for key, values in cast(Mapping[str, Any], value).items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(
                    "DPExhaustiveCompose requires finite, non-empty discrete "
                    f"candidate lists; invalid candidates for {key!r}."
                )
            discrete_grid[key] = list(values)
        return discrete_grid

    def tune(
        self,
        tuning_meta: TMetadata,
        tuning_data: TInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Select the best hyperparameters by exhaustively composing over the grid.

        Enumerates the Cartesian product of the per-parameter candidate lists,
        privately evaluates every candidate at the per-candidate budget, and
        returns the configuration with the lowest selection cost.

        Privacy budgets are divided by ``m`` in the offered family's
        composition parameter.

        Args:
            tuning_meta: Metadata for the tuning data.
            tuning_data: Input data for the tuning.
            budget: Total privacy budget for tuning.
            seed: Single int seed; one independent child seed per candidate is
                derived from it via ``split_seed``.

        Returns:
            The already-trained winning output and its hyperparameters.

        Raises:
            ValueError: If the hyperparameter grid is empty.
            NoRelease: If every evaluated candidate released bottom.
        """
        param_keys = list(self.alg_hyperparam_grid.keys())
        value_lists = [self.alg_hyperparam_grid[k] for k in param_keys]
        candidates = [
            dict(zip(param_keys, combo)) for combo in itertools.product(*value_lists)
        ]
        m = len(candidates)
        if m == 0:
            raise ValueError("Hyperparameter grid yields no candidates to evaluate.")

        # Sequential composition over m candidates on the same data: Dwork and
        # Roth (2014), Theorem 3.16; Bun and Steinke (2016), Lemma 7; Dong, Roth
        # and Su (2022), Corollary 2. Picking the lowest cost is post-processing.
        run_budget = budget.split([1.0] * m)[0]
        cand_seeds = split_seed(seed, m)

        evaluated: list[tuple[float, dict[str, Any], Any | None]] = []
        for params, cand_seed in zip(candidates, cand_seeds):
            alg_seed, selection_cost_seed = split_seed(cand_seed, 2)
            evaluated.append(
                self.evaluate_candidate(
                    tuning_meta,
                    tuning_data,
                    params,
                    run_budget,
                    alg_seed,
                    selection_cost_seed,
                    earlier_candidate_ran=bool(evaluated),
                )
            )

        return best_candidate(evaluated)
