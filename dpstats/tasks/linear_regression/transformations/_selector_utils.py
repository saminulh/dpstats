"""Shared reduction and restoration for feature-selection transformations.

Every selector releases a public support, restricts the problem to it, and
scatters the child's coefficients back into the original coordinates. The
record and maps here carry no unreleased private value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpstats.task import IntArray1D
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)


@dataclass(frozen=True)
class SelectorRestoration:
    """Map reduced design coordinates back to the original design.

    Attributes:
        selected: Original column indices retained, ascending.
        original_dim: Width of the original design.
    """

    selected: IntArray1D
    original_dim: int


def prepend_intercept(selected: IntArray1D, has_intercept: bool) -> IntArray1D:
    """Sort a selection and prepend the intercept column when present."""
    ordered = np.sort(selected)
    if has_intercept:
        return np.concatenate((np.array([0], dtype=np.int64), ordered))
    return ordered


def reduce_to_selection(
    metadata: LRMetadata,
    input: LRInput,
    selected: IntArray1D,
) -> tuple[LRMetadata, LRInput, SelectorRestoration]:
    """Restrict the problem to ``selected`` columns and record the inverse map."""
    reduced_metadata = LRMetadata(
        n_samples=metadata.n_samples,
        design_domain=metadata.design_domain.select_coordinates(selected),
        response_domain=metadata.response_domain,
    )
    reduced_input = LRInput(X=input.X[:, selected], y=input.y)
    restoration = SelectorRestoration(selected=selected, original_dim=metadata.dim)
    return reduced_metadata, reduced_input, restoration


def scatter_selected(output: LROutput, restoration: SelectorRestoration) -> LROutput:
    """Place reduced coefficients at their original indices, zero elsewhere."""
    beta = np.zeros(restoration.original_dim)
    beta[restoration.selected] = output.beta
    return LROutput(beta=beta)
