"""Enumerate configured algorithms, the glossary's noun for one complete configured executable identity."""

from __future__ import annotations

import itertools
from typing import Any, Sequence

from dpstats.algorithm import BaseAlgorithm
from dpstats.parameters import ParameterInfo, ValuesSearch


def grid(
    alg_class: type[BaseAlgorithm], **values: Sequence[Any]
) -> list[BaseAlgorithm]:
    """Configure one algorithm per point of a field grid.

    ``values`` maps configured field names to the candidate values that field
    takes, and the result is their Cartesian product, varying the last named
    field fastest. Registers nothing.
    """
    for name, candidates in values.items():
        if not candidates:
            raise ValueError(f"grid() received no candidate values for {name!r}.")
    names = list(values)
    return [
        alg_class(**dict(zip(names, point)))
        for point in itertools.product(*values.values())
    ]


def _linspace(lower: float, upper: float, n_points: int) -> list[float]:
    if n_points == 1:
        return [float(lower)]
    step = (upper - lower) / (n_points - 1)
    values = [lower + step * index for index in range(n_points)]
    values[0], values[-1] = float(lower), float(upper)
    return values


def _geomspace(lower: float, upper: float, n_points: int) -> list[float]:
    if n_points == 1:
        return [float(lower)]
    ratio = (upper / lower) ** (1.0 / (n_points - 1))
    values = [lower * ratio**index for index in range(n_points)]
    values[0], values[-1] = float(lower), float(upper)
    return values


def _build_tuning_grid(  # pyright: ignore[reportUnusedFunction]
    parameters: dict[str, ParameterInfo],
    *,
    n_points: int,
) -> dict[str, list[Any]]:
    """Discretize declared, data-independent tuning policy.

    Reached through :meth:`BaseAlgorithm.default_tuning_grid`.

    Args:
        parameters: One configured class's parameter introspection, from
            :func:`inspect_parameters`; a parameter declaring no search
            contributes no key.
        n_points: Points to take from each declared range; ignored for a
            declared discrete list, which is returned as given.

    Returns:
        One candidate list per parameter that declares a search, keyed by
        parameter name.

    Raises:
        ValueError: If ``n_points`` is not positive.

    Example:
        >>> from dpstats.algorithm import BaseAlgorithm
        >>> from dpstats.tasks.linear_regression.algorithms import (
        ...     OnePassDPGD,
        ... )
        >>> grid = OnePassDPGD.default_tuning_grid(n_points=5)
    """
    if n_points < 1:
        raise ValueError("n_points must be positive.")

    grid: dict[str, list[Any]] = {}
    for name, parameter in parameters.items():
        search = parameter.search
        if search is None:
            continue
        if isinstance(search, ValuesSearch):
            grid[name] = list(search.values)
            continue

        numeric_values = (
            _geomspace(search.lower, search.upper, n_points)
            if search.scale == "log"
            else _linspace(search.lower, search.upper, n_points)
        )
        if parameter.annotation is int:
            rounded = [int(round(value)) for value in numeric_values]
            values = list(dict.fromkeys(rounded))
        else:
            values = numeric_values
        grid[name] = values
    return grid


__all__ = [
    "grid",
]
