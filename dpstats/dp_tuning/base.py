"""Base class for DP hyperparameter tuned algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Annotated, Any, TypeVar, cast

import numpy as np
from pydantic import Field, TypeAdapter, model_validator

from dpstats.algorithm import (
    BaseAlgorithm,
    PrivacyInterpretation,
    select_weakest_privacy_interpretation,
)
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.budgets import DPBudget
from dpstats.parameters import Importable, parameter_field
from dpstats.parameters import RangeSearch, parse_search_range
from dpstats.task import Task

TMetadata = Any
TInput = Any
TOutput = Any


TCost = TypeVar("TCost", bound=Callable[..., Any])


def declares_privacy_interpretation(
    privacy_interpretation: PrivacyInterpretation,
) -> Callable[[TCost], TCost]:
    """Record on a selection cost the class its release carries.

    A tuner reads its class as the weakest of its candidate's and its selection
    cost's; a cost with no declaration reads ``DIAGNOSTIC``.
    """

    def decorate(cost: TCost) -> TCost:
        setattr(cost, "PRIVACY_INTERPRETATION", privacy_interpretation)
        return cost

    return decorate


def best_candidate(
    evaluated_candidates: list[tuple[float, dict[str, Any], TOutput | None]],
) -> tuple[TOutput, dict[str, Any]]:
    """Return the output and hyperparameters with the lowest selection cost.

    Only a candidate with an output can win, and an infinite cost is
    a valid score: among candidates with an output the lowest cost wins, the first
    evaluated on a tie.

    Args:
        evaluated_candidates: Each candidate's ``(cost, hyperparameters,
            output)``, where ``output`` is ``None`` for a candidate that
            released bottom.

    Returns:
        The winning candidate's already-trained output and hyperparameters.

    Raises:
        NoRelease: If every candidate released bottom.

    Example:
        >>> best_output, best_params = best_candidate(
        ...     [(0.5, {"c": 0.1}, "trained-a"), (0.2, {"c": 0.2}, "trained-b")]
        ... )
        >>> best_output, best_params
        ('trained-b', {'c': 0.2})
    """
    released = [
        candidate for candidate in evaluated_candidates if candidate[2] is not None
    ]
    if not released:
        raise NoRelease(
            "Hyperparameter selection releases bottom: every evaluated "
            "candidate released bottom."
        )
    _, best_params, best_output = min(released, key=lambda candidate: candidate[0])
    return best_output, best_params


class DPHyperparameterTuned(BaseAlgorithm, ABC):
    """Abstract immutable configuration for DP hyperparameter tuning.

    Parametrization notes:
        - ``scoring_share``: The fraction of each candidate's budget the
          selection cost spends.

    Implementation notes:
        - Training and scoring use the same data rather than a train/validation
          split: each candidate trains on all of it with ``1 - scoring_share``
          of its budget, and the selection cost scores it on all of it with the
          rest, so no split size needs to be chosen or disclosed.
        - The tuned result carries the weaker of two privacy-interpretation
          labels: the tuned algorithm's and the selection cost's.
        - A candidate that releases nothing never wins and does not abort the
          tuning run, so its siblings still count.
        - An infinite selection cost is a valid score, and a cost that is not
          a number or that releases nothing scores as positive infinity.
          Nothing is released only when every evaluated candidate released
          nothing.
        - A candidate refusing its regime after an earlier candidate ran, or a
          selection cost refusing after its candidate trained, is an execution
          failure, since budget was already spent.
    """

    alg_class: Importable[type[BaseAlgorithm]] = Field(
        description=(
            "Concrete task-native BaseAlgorithm subclass to tune, or its "
            "dotted import path; wrapper classes and nonprivate candidates "
            "are refused."
        )
    )
    alg_hyperparam_grid: dict[str, list[Any] | dict[str, Any]] = Field(
        description=(
            "For each declared parameter of alg_class, a non-empty list of "
            "discrete candidates or a numeric search range."
        )
    )
    selection_cost: Importable[Callable[..., Any]] = Field(
        description="Importable DP selection-cost callable; lower is better."
    )
    selection_cost_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit keyword arguments passed to selection_cost.",
    )

    scoring_share: float = parameter_field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of each candidate's budget spent on its selection "
            "cost; the rest trains the candidate."
        ),
    )

    @model_validator(mode="after")
    def _validate_tuner(self) -> DPHyperparameterTuned:
        if not hasattr(self.alg_class, "TASK"):
            raise TypeError(
                f"{self.alg_class.__name__} declares no class-level TASK. "
                "alg_class must be a concrete task-native Algorithm; to "
                "tune within a processed task, compose the tuner inside "
                "CompositeAlgorithm (transformations transport the task inward; the "
                "tuner tunes the leaf in the inner task's space)."
            )
        parameter_info = self.alg_class.parameter_info()
        unknown_keys = set(self.alg_hyperparam_grid) - set(parameter_info)
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Search space contains parameters not declared by "
                f"{self.alg_class.__name__}: {unknown}."
            )
        normalized_grid: dict[str, list[Any] | dict[str, Any]] = {}
        for key, values in self.alg_hyperparam_grid.items():
            if isinstance(values, dict):
                declared_search = parameter_info[key].search
                parsed = parse_search_range(
                    values,
                    name=key,
                    default_scale=(
                        declared_search.scale
                        if isinstance(declared_search, RangeSearch)
                        else "linear"
                    ),
                )
                normalized_grid[key] = {
                    "lower": parsed.lower,
                    "upper": parsed.upper,
                    "scale": parsed.scale,
                }
            elif len(values) == 0:
                raise ValueError(
                    f"Search space for {key!r} must be a non-empty discrete "
                    "candidate list or a {'lower': x, 'upper': y} range. "
                    "Pass alg_class.default_tuning_grid() explicitly to use "
                    "the algorithm's declared search space."
                )
            else:
                declared_field = self.alg_class.model_fields[key]
                candidate_type: Any = Annotated[
                    declared_field.annotation, declared_field
                ]
                # Keyed by parameter name so a rejected candidate's error
                # locates the parameter and not only its position.
                validated = TypeAdapter(
                    dict[str, list[candidate_type]],
                    config=self.alg_class.model_config,
                ).validate_python({key: deepcopy(list(values))})
                normalized_grid[key] = validated[key]
        object.__setattr__(self, "alg_hyperparam_grid", normalized_grid)
        object.__setattr__(self, "selection_cost_args", dict(self.selection_cost_args))
        return self

    def build_candidate(self, params: Mapping[str, Any]) -> BaseAlgorithm:
        """Construct one configured candidate of the tuned class."""
        candidate = self.alg_class.model_validate(dict(params))
        if (
            candidate.privacy_interpretation
            is not self.alg_class.PRIVACY_INTERPRETATION
        ):
            raise TypeError(
                f"{type(candidate).__name__} reads {candidate.privacy_interpretation} "
                f"but declares PRIVACY_INTERPRETATION "
                f"{self.alg_class.PRIVACY_INTERPRETATION}; a tuner classifies its "
                "candidate by the declared class."
            )
        if candidate.privacy_interpretation is PrivacyInterpretation.NONPRIVATE:
            raise TypeError(
                f"A DP tuner cannot tune nonprivate candidate "
                f"{type(candidate).__name__}."
            )
        return candidate

    @property
    def task(self) -> Task:
        """Return the task the tuned Algorithm is native to."""
        return self.alg_class.TASK

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        """Return the weaker of the candidate's and selection cost's readings."""
        cost_interpretation = cast(
            PrivacyInterpretation,
            getattr(
                self.selection_cost,
                "PRIVACY_INTERPRETATION",
                PrivacyInterpretation.DIAGNOSTIC,
            ),
        )
        return select_weakest_privacy_interpretation(
            [self.alg_class.PRIVACY_INTERPRETATION, cost_interpretation]
        )

    def run(
        self,
        metadata: TMetadata,
        input: TInput,
        budget: DPBudget,
        seed: int,
    ) -> TOutput:
        """Run private selection and output its already-trained winner."""
        selected_output, _ = self.tune(metadata, input, budget, seed)
        return selected_output

    @abstractmethod
    def tune(
        self,
        tuning_meta: TMetadata,
        tuning_data: TInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[TOutput, dict[str, Any]]:
        """Perform the private hyperparameter search a subclass implements.

        Args:
            tuning_meta: Metadata for the tuning data.
            tuning_data: Input data for the tuning.
            budget: Total privacy budget for tuning.
            seed: Single int seed a subclass derives its own child seeds from.

        Returns:
            The already-trained winning output and its hyperparameters.
        """

    def evaluate_candidate(
        self,
        tuning_meta: TMetadata,
        tuning_data: TInput,
        params: dict[str, Any],
        run_budget: DPBudget,
        alg_seed: int,
        selection_cost_seed: int,
        *,
        earlier_candidate_ran: bool,
    ) -> tuple[float, dict[str, Any], TOutput | None]:
        """Privately train one candidate and compute its selection cost.

        Args:
            tuning_meta: Metadata for the tuning data.
            tuning_data: Input data for the tuning.
            params: The candidate's hyperparameter values.
            run_budget: Privacy budget for this candidate, divided
                between its training and its selection cost by
                ``scoring_share``.
            alg_seed: Seed for the candidate's own training run.
            selection_cost_seed: Seed for the selection-cost calculation.
            earlier_candidate_ran: Whether this tuning run already evaluated
                another candidate, so part of its budget is spent.

        Returns:
            The candidate's selection cost, its hyperparameters, and its
            trained output, or ``None`` for the output when the candidate
            released bottom. A selection cost releasing bottom scores its
            candidate positive infinity.

        Raises:
            InvalidRegime: If the candidate refuses its regime before any
                candidate has run.
            RuntimeError: If the candidate refuses its regime after an earlier
                candidate ran, or the selection cost refuses after the
                candidate trained.
        """
        training_budget, scoring_budget = run_budget.split(
            (1.0 - self.scoring_share, self.scoring_share)
        )
        alg = self.build_candidate(params)
        try:
            output: TOutput = alg.run(
                tuning_meta, tuning_data, budget=training_budget, seed=alg_seed
            )
        except NoRelease:
            return np.inf, params, None
        except InvalidRegime as error:
            if earlier_candidate_ran:
                raise RuntimeError(
                    f"{type(alg).__name__} refused its regime after an earlier "
                    "candidate spent budget."
                ) from error
            raise
        try:
            cost = float(
                self.selection_cost(
                    output,
                    evaluation_metadata=tuning_meta,
                    evaluation_input=tuning_data,
                    budget=scoring_budget,
                    seed=selection_cost_seed,
                    **self.selection_cost_args,
                )
            )
        except NoRelease:
            return np.inf, params, output
        except InvalidRegime as error:
            raise RuntimeError(
                "The selection cost refused its regime after "
                f"{type(alg).__name__} spent budget."
            ) from error
        if np.isnan(cost):
            cost = np.inf
        return cost, params, output

    def sample_param(
        self,
        values: list[Any] | dict[str, Any],
        seed: int,
    ) -> Any:
        """Sample one hyperparameter value from a discrete list or range.

        Args:
            values: A non-empty discrete candidate list, or a normalized
                ``{"lower", "upper", "scale"}`` range dict.
            seed: Single int seed for this one draw.

        Returns:
            The sampled value, typed like its declared candidates or bounds.
        """
        rng = np.random.default_rng(seed)
        if isinstance(values, dict):
            lo = values["lower"]
            hi = values["upper"]
            if values["scale"] == "log":
                sample = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                if isinstance(lo, int) and isinstance(hi, int):
                    return int(round(sample))
                return sample
            if isinstance(lo, int) and isinstance(hi, int):
                return int(rng.integers(lo, hi + 1))
            return float(rng.uniform(lo, hi))

        if len(values) == 1:
            return values[0]
        return values[int(rng.integers(len(values)))]
