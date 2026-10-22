"""Shared bounded-domain transformations for numeric estimation Tasks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    Domain,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D

NumericBlockTransform = Callable[[Domain, Array2D], tuple[Domain, Array2D]]


@dataclass(frozen=True)
class AffineTransform:
    """A coordinatewise map ``(value - shift) / scale`` and its image."""

    domain: Domain
    shift: Array1D
    scale: Array1D

    def apply(self, values: Array1D | Array2D) -> Array1D | Array2D:
        """Map values into the transformed coordinates."""
        return (values - self.shift) / self.scale

    def restore(self, values: Array1D | Array2D) -> Array1D | Array2D:
        """Map transformed values back to the original coordinates."""
        return self.shift + self.scale * values


def transform_numeric_blocks(
    source_domain: Domain,
    values: Array2D,
    transform: NumericBlockTransform,
) -> tuple[Domain, Array2D]:
    """Transform ordinary numeric blocks while preserving structural blocks."""
    if isinstance(source_domain, (InterceptDomain, OneHotDomain)):
        return source_domain, values
    if isinstance(source_domain, ProductDomain):
        domains: list[Domain] = []
        transformed_values: list[Array2D] = []
        start = 0
        for part in source_domain.parts:
            end = start + part.dim
            domain, block = transform_numeric_blocks(
                part,
                values[:, start:end],
                transform,
            )
            domains.append(domain)
            transformed_values.append(block)
            start = end
        return ProductDomain(parts=tuple(domains)), np.concatenate(
            transformed_values, axis=1
        )
    return transform(source_domain, values)


def standardize_domain(domain: Domain, *, center: bool) -> AffineTransform:
    """Build a public affine transform for one bounded domain.

    Numeric coordinates are centered when the caller's estimand permits it and
    otherwise only scaled. Structural intercept and one-hot blocks pass
    through. Finite singleton numeric domains also pass through so the map
    never divides by zero. Unbounded or unknown ordinary numeric domains are
    refused rather than silently receiving data-dependent bounds.
    """
    if isinstance(domain, (InterceptDomain, OneHotDomain)):
        return AffineTransform(
            domain=domain,
            shift=np.zeros(domain.dim),
            scale=np.ones(domain.dim),
        )

    if isinstance(domain, ProductDomain):
        parts = tuple(standardize_domain(part, center=center) for part in domain.parts)
        return AffineTransform(
            domain=ProductDomain(parts=tuple(part.domain for part in parts)),
            shift=np.concatenate(tuple(part.shift for part in parts)),
            scale=np.concatenate(tuple(part.scale for part in parts)),
        )

    if isinstance(domain, BoxDomain):
        if not np.all(np.isfinite(domain.bounds)):
            raise InvalidRegime("Standardize requires finite public domain bounds.")
        lo = domain.bounds[:, 0]
        hi = domain.bounds[:, 1]
        widths = hi - lo
        positive_width = widths > 0
        shift = np.zeros(domain.dim)
        scale = np.ones(domain.dim)
        scale[positive_width] = widths[positive_width] / 2
        if center:
            shift[positive_width] = (lo[positive_width] + hi[positive_width]) / 2
        transformed_bounds = np.column_stack(
            ((lo - shift) / scale, (hi - shift) / scale)
        )
        return AffineTransform(
            domain=BoxDomain(bounds=transformed_bounds),
            shift=shift,
            scale=scale,
        )

    if isinstance(domain, LpBallDomain):
        if domain.radius == 0:
            return AffineTransform(
                domain=domain,
                shift=np.zeros(domain.dim),
                scale=np.ones(domain.dim),
            )
        # The p-norm of the all-ones vector gives each coordinate unit scale.
        target_radius = 1.0 if math.isinf(domain.p) else domain.dim ** (1 / domain.p)
        shift = domain.center if center else np.zeros(domain.dim)
        scale = np.full(domain.dim, domain.radius / target_radius)
        return AffineTransform(
            domain=LpBallDomain(
                p=domain.p,
                radius=target_radius,
                center=(domain.center - shift) / scale,
            ),
            shift=shift,
            scale=scale,
        )

    raise InvalidRegime(
        "Standardize requires bounded BoxDomain or LpBallDomain numeric blocks; "
        f"received {type(domain).__name__}."
    )


def derive_empirical_domain(source_domain: Domain, values: Array2D) -> Domain:
    """Return one domain with exact empirical boxes for its numeric blocks."""
    if isinstance(source_domain, (InterceptDomain, OneHotDomain)):
        return source_domain
    if isinstance(source_domain, ProductDomain):
        # Only metadata changes: do not concatenate unchanged record blocks.
        parts: list[Domain] = []
        start = 0
        for part in source_domain.parts:
            end = start + part.dim
            parts.append(derive_empirical_domain(part, values[:, start:end]))
            start = end
        return ProductDomain(parts=tuple(parts))
    return BoxDomain(bounds=np.column_stack((values.min(axis=0), values.max(axis=0))))


__all__ = [
    "AffineTransform",
    "derive_empirical_domain",
    "standardize_domain",
]
