"""Builders for the built-in Task Metadata the task tests construct by hand.

Each Metadata builder defaults every domain to the unbounded space of the right
dimension, so a test spells out only the geometry its assertion is about. A
domain argument given as an ``int`` is that dimension's unbounded space.
``Unreadable`` stands in for a Metadata or Input a mechanism must not read.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dpstats.domains import BoxDomain, Domain, EuclideanSpace
from dpstats.tasks.linear_regression import LinearRegressionMetadata
from dpstats.tasks.m_estimation import (
    MEstimationLoss,
    MEstimationMetadata,
)
from dpstats.tasks.mean_estimation import MeanEstimationMetadata
from dpstats.tasks.median_estimation import MedianEstimationMetadata
from dpstats.tasks.simple_linear_regression import (
    SimpleLinearRegressionMetadata,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationMetadata,
)


def box(*intervals: Sequence[float]) -> BoxDomain:
    """Return the box with one ``(lower, upper)`` interval per coordinate."""
    return BoxDomain(bounds=np.array(intervals, dtype=float))


def _domain(domain: Domain | int) -> Domain:
    return EuclideanSpace(dim=domain) if isinstance(domain, int) else domain


def lr_metadata(
    n_samples: int, design: Domain | int = 1, response: Domain | int = 1
) -> LinearRegressionMetadata:
    """Return linear-regression Metadata over the given design and response."""
    return LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=_domain(design),
        response_domain=_domain(response),
    )


def slr_metadata(
    n_samples: int, x: Domain | int = 1, y: Domain | int = 1
) -> SimpleLinearRegressionMetadata:
    """Return simple-linear-regression Metadata over the given x and y."""
    return SimpleLinearRegressionMetadata(
        n_samples=n_samples, x_domain=_domain(x), y_domain=_domain(y)
    )


def sce_metadata(
    n_samples: int,
    coefficient_of_interest_index: int,
    design: Domain | int,
    response: Domain | int = 1,
) -> SingleCoefficientEstimationMetadata:
    """Return single-coefficient Metadata reporting one design coordinate."""
    return SingleCoefficientEstimationMetadata(
        n_samples=n_samples,
        coefficient_of_interest_index=coefficient_of_interest_index,
        design_domain=_domain(design),
        response_domain=_domain(response),
    )


def mean_metadata(n_samples: int, domain: Domain | int = 1) -> MeanEstimationMetadata:
    """Return mean-estimation Metadata over the given record domain."""
    return MeanEstimationMetadata(n_samples=n_samples, domain=_domain(domain))


def median_metadata(
    n_samples: int, domain: Domain | int = 1
) -> MedianEstimationMetadata:
    """Return median-estimation Metadata over the given scalar domain."""
    return MedianEstimationMetadata(n_samples=n_samples, domain=_domain(domain))


def m_estimation_metadata(
    n_samples: int,
    loss: MEstimationLoss,
    parameter: Domain | int = 1,
    data: Domain | int = 1,
) -> MEstimationMetadata:
    """Return M-estimation Metadata for ``loss`` over the given domains."""
    return MEstimationMetadata(
        n_samples=n_samples,
        loss=loss,
        parameter_domain=_domain(parameter),
        data_domain=_domain(data),
    )


class Unreadable:
    """A Metadata or Input stand-in whose every attribute read fails.

    Passing one to ``run()`` proves the algorithm decides without reading it.
    """

    def __init__(self, what: str) -> None:
        self._what = what

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"{self._what}.{name} was read")
