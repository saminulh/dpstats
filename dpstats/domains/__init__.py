"""Domain types for specifying public data constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import ClassVar

import numpy as np
from pydantic import Field, model_validator

from typing_extensions import override

from dpstats.transport.configuration import PolymorphicModel
from dpstats.task import (
    Array1D,
    Array2D,
    BoolArray1D,
    IntArray1D,
)

# LpBallDomain.project can return a point whose norm is marginally above the
# radius under float rounding; 16 ULP absorbs that overshoot so a projected
# point still counts as a member (see Domain.contains for the membership
# contract this constant serves).
_NORM_MEMBERSHIP_SLACK = 16 * float(np.finfo(np.float64).eps)


class Domain(PolymorphicModel, ABC):
    """Abstract base class for domains."""

    dim: ClassVar[int]
    center: ClassVar[Array1D]

    @abstractmethod
    def l2_diameter(self) -> float:
        """Return the L2 diameter."""

    @abstractmethod
    def l1_diameter(self) -> float:
        """Return the L1 diameter."""

    @abstractmethod
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        """Project point(s) onto the domain."""

    @abstractmethod
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        """Return one membership decision per supplied row."""

    @abstractmethod
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        """Return the image of this domain on selected coordinates."""

    def _validate_coordinate_selection(self, indices: IntArray1D) -> None:
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("Domain coordinate selection must be nonempty and 1D.")
        if np.any(indices < 0) or np.any(indices >= self.dim):
            raise ValueError("Domain coordinate selection is out of bounds.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Domain coordinate selection must not repeat coordinates.")

    @property
    def l2_norm_bound(self) -> float:
        """Return a bound on the L2 norm of points in the domain."""
        center, radius = self.l2_enclosing_ball()
        return float(np.linalg.norm(center)) + radius

    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        """Return a public center and sound L2 radius around that center."""
        return self.center, math.inf

    def bounding_box(self) -> BoxDomain:
        """Return a bounding box for the domain."""
        return BoxDomain(
            bounds=np.tile(np.array([[-math.inf, math.inf]]), (self.dim, 1))
        )

    def convex_enclosure(self) -> Domain:
        """Return a public convex domain containing this one.

        A convex domain returns itself; any other returns its bounding box, a
        convex superset that need not be the convex hull.
        """
        return self.bounding_box()


class EuclideanSpace(Domain):
    """Unbounded Euclidean space of a declared dimension."""

    dim: int = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        description="Positive number of coordinates."
    )

    @model_validator(mode="after")
    def _validate_dim(self) -> EuclideanSpace:
        if self.dim <= 0:
            raise ValueError("EuclideanSpace.dim must be positive.")
        return self

    @property
    def center(self) -> Array1D:  # pyright: ignore[reportIncompatibleVariableOverride]
        return np.zeros(self.dim)

    @override
    def l2_diameter(self) -> float:
        return math.inf

    @override
    def l1_diameter(self) -> float:
        return math.inf

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        return x

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        return np.atleast_1d(np.all(np.isfinite(values), axis=-1))

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        return EuclideanSpace(dim=int(indices.size))

    @override
    def convex_enclosure(self) -> Domain:
        return self


class BoxDomain(Domain):
    """Axis-aligned box with one lower and upper bound per coordinate.

    Examples:
        >>> import numpy as np
        >>> from dpstats.domains import BoxDomain
        >>> domain = BoxDomain(bounds=np.array([[-2.0, 2.0], [0.0, 10.0]]))
        >>> domain.dim
        2
    """

    bounds: Array2D = Field(
        description=(
            "Lower and upper bound per coordinate, shape (dim, 2): row i "
            "holds [lower_i, upper_i]."
        )
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> BoxDomain:
        if self.bounds.shape[0] == 0 or self.bounds.shape[1] != 2:
            raise ValueError(
                "BoxDomain.bounds must have shape (dim, 2) with dim positive."
            )
        if np.any(np.isnan(self.bounds)):
            raise ValueError("BoxDomain.bounds must not contain NaN.")
        if np.any(np.isposinf(self.bounds[:, 0])) or np.any(
            np.isneginf(self.bounds[:, 1])
        ):
            raise ValueError(
                "BoxDomain lower bounds cannot be positive infinity and upper "
                "bounds cannot be negative infinity."
            )
        if np.any(self.bounds[:, 0] > self.bounds[:, 1]):
            raise ValueError("BoxDomain lower bounds must not exceed upper bounds.")
        return self

    @property
    def dim(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        return int(self.bounds.shape[0])

    @property
    def center(self) -> Array1D:  # pyright: ignore[reportIncompatibleVariableOverride]
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        center = np.zeros(self.dim)
        bounded = np.isfinite(lo) & np.isfinite(hi)
        center[bounded] = (lo[bounded] + hi[bounded]) / 2
        return np.minimum(np.maximum(center, lo), hi)

    @override
    def l2_diameter(self) -> float:
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        return float(np.sqrt(np.sum(widths**2)))

    @override
    def l1_diameter(self) -> float:
        return float(np.sum(self.bounds[:, 1] - self.bounds[:, 0]))

    @property
    @override
    def l2_norm_bound(self) -> float:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return float(np.sqrt(np.sum(np.maximum(lo**2, hi**2))))

    @override
    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        return self.center, self.l2_diameter() / 2

    @override
    def bounding_box(self) -> BoxDomain:
        return self

    @override
    def convex_enclosure(self) -> Domain:
        return self

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        return np.atleast_1d(
            np.all(
                np.isfinite(values)
                & (values >= self.bounds[:, 0])
                & (values <= self.bounds[:, 1]),
                axis=-1,
            )
        )

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        return BoxDomain(bounds=self.bounds[indices])


class LpBallDomain(Domain):
    """Ball of a given Lp-norm radius."""

    p: float = Field(description="Norm order: 1, 2, or infinity.")
    radius: float = Field(description="Nonnegative, finite ball radius.")
    center: Array1D = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        description="Ball center, a nonempty finite 1-D array."
    )

    @model_validator(mode="after")
    def _validate_ball(self) -> LpBallDomain:
        if self.p not in (1, 2, np.inf):
            raise ValueError("LpBallDomain.p must be 1, 2, or infinity.")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("LpBallDomain.radius must be finite and nonnegative.")
        if self.center.size == 0:
            raise ValueError("LpBallDomain.center must be a nonempty vector.")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("LpBallDomain.center must be finite.")
        return self

    @property
    def dim(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        return int(self.center.shape[0])

    @classmethod
    def at_origin(cls, p: float, radius: float, dim: int) -> LpBallDomain:
        if dim <= 0:
            raise ValueError("LpBallDomain.dim must be positive.")
        return cls(p=p, radius=radius, center=np.zeros(dim))

    @override
    def l2_diameter(self) -> float:
        if self.p in (1, 2):
            return 2 * self.radius
        if self.p == np.inf:
            return 2 * np.sqrt(self.dim) * self.radius
        raise ValueError(f"Unsupported p: {self.p}")

    @override
    def l1_diameter(self) -> float:
        if self.p == 1:
            return 2 * self.radius
        if self.p == 2:
            return 2 * np.sqrt(self.dim) * self.radius
        if self.p == np.inf:
            return 2 * self.dim * self.radius
        raise ValueError(f"Unsupported p: {self.p}")

    @override
    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        if self.p in (1, 2):
            return self.center, self.radius
        return self.center, math.sqrt(self.dim) * self.radius

    @override
    def bounding_box(self) -> BoxDomain:
        return BoxDomain(
            bounds=np.column_stack(
                (self.center - self.radius, self.center + self.radius)
            )
        )

    @override
    def convex_enclosure(self) -> Domain:
        return self

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        c = self.center
        is_vector = x.ndim == 1
        points = x[np.newaxis, :] if is_vector else x
        centered = points - c
        norms = np.linalg.norm(centered, ord=self.p, axis=1)
        outside = norms > self.radius

        if not np.any(outside):
            return x

        projected = points.copy()
        if self.p == 2:
            scale = self.radius / norms[outside]
            projected[outside] = centered[outside] * scale[:, np.newaxis] + c
        elif self.p == np.inf:
            projected[outside] = np.clip(
                points[outside], c - self.radius, c + self.radius
            )
        elif self.p == 1:
            if self.radius == 0:
                projected[outside] = c
            else:
                outside_values: Array2D = centered[outside]
                magnitudes: Array2D = np.abs(outside_values)
                sorted_magnitudes = np.sort(magnitudes, axis=1)[:, ::-1]
                cumulative = np.cumsum(sorted_magnitudes, axis=1)
                positions = np.arange(1, self.dim + 1)
                active = sorted_magnitudes > (cumulative - self.radius) / positions
                support_size = np.sum(active, axis=1)
                row_indices: IntArray1D = np.arange(magnitudes.shape[0])
                theta = (
                    cumulative[row_indices, support_size - 1] - self.radius
                ) / support_size
                projected[outside] = (
                    np.sign(outside_values)
                    * np.maximum(magnitudes - theta[:, np.newaxis], 0)
                    + c
                )
        else:
            raise ValueError(f"Unsupported p: {self.p}")

        return projected[0] if is_vector else projected

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        norms = np.linalg.norm(values - self.center, ord=self.p, axis=-1)
        return np.atleast_1d(
            (norms <= self.radius)
            | np.isclose(norms, self.radius, rtol=_NORM_MEMBERSHIP_SLACK, atol=0)
        )

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        return LpBallDomain(
            p=self.p,
            radius=self.radius,
            center=self.center[indices],
        )


class OneHotDomain(Domain):
    """One-hot categorical block, optionally including the all-zero code."""

    dim: int = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        description="Positive number of one-hot codes."
    )
    allow_zero: bool = Field(
        default=False, description="Whether the all-zero code is also a member."
    )

    @model_validator(mode="after")
    def _validate_dim(self) -> OneHotDomain:
        if self.dim <= 0:
            raise ValueError("OneHotDomain.dim must be positive.")
        return self

    @property
    def center(self) -> Array1D:  # pyright: ignore[reportIncompatibleVariableOverride]
        if self.allow_zero and self.dim == 1:
            return np.full(1, 0.5)
        return np.full(self.dim, 1 / self.dim)

    @override
    def l2_diameter(self) -> float:
        if self.dim == 1:
            return 1.0 if self.allow_zero else 0.0
        return math.sqrt(2)

    @override
    def l1_diameter(self) -> float:
        if self.dim == 1:
            return 1.0 if self.allow_zero else 0.0
        return 2.0

    @property
    @override
    def l2_norm_bound(self) -> float:
        return 1.0

    @override
    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        center = self.center
        points = np.eye(self.dim)
        if self.allow_zero:
            points = np.vstack((np.zeros(self.dim), points))
        return center, float(np.max(np.linalg.norm(points - center, axis=1)))

    @override
    def bounding_box(self) -> BoxDomain:
        lower = 0.0 if self.allow_zero or self.dim > 1 else 1.0
        return BoxDomain(bounds=np.tile(np.array([[lower, 1.0]]), (self.dim, 1)))

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        is_vector = x.ndim == 1
        rows = x[np.newaxis, :] if is_vector else x
        projected = np.zeros_like(rows)
        winner = np.argmax(rows, axis=1)
        row_indices = np.arange(rows.shape[0])
        projected[row_indices, winner] = 1.0
        if self.allow_zero:
            winner_values = rows[row_indices, winner]
            zero_distance = np.sum(rows**2, axis=1)
            winner_distance = zero_distance - 2 * winner_values + 1
            projected[zero_distance <= winner_distance] = 0.0
        return projected[0] if is_vector else projected

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        rows = values if values.ndim == 2 else values[None, :]
        finite = np.all(np.isfinite(rows), axis=1)
        binary = np.all((rows == 0.0) | (rows == 1.0), axis=1)
        totals = np.sum(rows, axis=1)
        allowed_total = totals == 1.0
        if self.allow_zero:
            allowed_total |= totals == 0.0
        return finite & binary & allowed_total

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        return OneHotDomain(
            dim=int(indices.size),
            allow_zero=self.allow_zero or indices.size < self.dim,
        )


class InterceptDomain(Domain):
    """Singleton domain for one regression-intercept coordinate fixed at 1."""

    dim: ClassVar[int] = 1

    @property
    def center(self) -> Array1D:  # pyright: ignore[reportIncompatibleVariableOverride]
        return np.ones(1)

    @override
    def l2_diameter(self) -> float:
        return 0.0

    @override
    def l1_diameter(self) -> float:
        return 0.0

    @property
    @override
    def l2_norm_bound(self) -> float:
        return 1.0

    @override
    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        return self.center, 0.0

    @override
    def bounding_box(self) -> BoxDomain:
        return BoxDomain(bounds=np.array([[1.0, 1.0]]))

    @override
    def convex_enclosure(self) -> Domain:
        return self

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        return np.ones_like(x)

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        return np.atleast_1d(np.all(values == 1.0, axis=-1))

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        return self


class ProductDomain(Domain):
    """Cartesian product of domains.

    Examples:
        An intercept, two bounded continuous columns, and a four-level
        reference-coded category:

        >>> import numpy as np
        >>> from dpstats.domains import (
        ...     BoxDomain, InterceptDomain, OneHotDomain, ProductDomain,
        ... )
        >>> design_domain = ProductDomain(
        ...     parts=(
        ...         InterceptDomain(),
        ...         BoxDomain(bounds=np.array([[-2.0, 2.0], [0.0, 10.0]])),
        ...         OneHotDomain(dim=3, allow_zero=True),
        ...     )
        ... )
    """

    parts: tuple[Domain, ...] = Field(
        description="Component domains, one per coordinate group, in order."
    )

    @model_validator(mode="after")
    def _validate_parts(self) -> ProductDomain:
        if not self.parts:
            raise ValueError("ProductDomain.parts must not be empty.")
        if any(isinstance(part, ProductDomain) for part in self.parts):
            object.__setattr__(
                self,
                "parts",
                tuple(
                    nested
                    for part in self.parts
                    for nested in (
                        part.parts if isinstance(part, ProductDomain) else (part,)
                    )
                ),
            )
        return self

    @property
    def dim(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        return sum(part.dim for part in self.parts)

    @property
    def center(self) -> Array1D:  # pyright: ignore[reportIncompatibleVariableOverride]
        return np.concatenate([part.center for part in self.parts])

    @override
    def l2_diameter(self) -> float:
        return math.sqrt(sum(part.l2_diameter() ** 2 for part in self.parts))

    @override
    def l1_diameter(self) -> float:
        return sum(part.l1_diameter() for part in self.parts)

    @property
    @override
    def l2_norm_bound(self) -> float:
        return math.sqrt(sum(part.l2_norm_bound**2 for part in self.parts))

    @override
    def l2_enclosing_ball(self) -> tuple[Array1D, float]:
        centers: list[Array1D] = []
        squared_radius = 0.0
        for part in self.parts:
            center, radius = part.l2_enclosing_ball()
            centers.append(center)
            squared_radius += radius**2
        return np.concatenate(centers), math.sqrt(squared_radius)

    @override
    def bounding_box(self) -> BoxDomain:
        bounds = [part.bounding_box().bounds for part in self.parts]
        return BoxDomain(bounds=np.concatenate(bounds))

    @override
    def convex_enclosure(self) -> Domain:
        return ProductDomain(
            parts=tuple(part.convex_enclosure() for part in self.parts)
        )

    @override
    def project(self, x: Array1D | Array2D) -> Array1D | Array2D:
        projected_parts: list[Array1D | Array2D] = []
        start = 0
        for part in self.parts:
            end = start + part.dim
            projected_parts.append(part.project(x[..., start:end]))
            start = end
        return np.concatenate(projected_parts, axis=-1)

    @override
    def contains(self, values: Array1D | Array2D) -> BoolArray1D:
        rows = values if values.ndim == 2 else values[None, :]
        members: BoolArray1D = np.ones(rows.shape[0], dtype=np.bool_)
        start = 0
        for part in self.parts:
            end = start + part.dim
            members &= part.contains(rows[:, start:end])
            start = end
        return members

    @override
    def select_coordinates(self, indices: IntArray1D) -> Domain:
        self._validate_coordinate_selection(indices)
        offsets = np.cumsum(np.array((0, *(part.dim for part in self.parts))))
        part_indices = np.searchsorted(offsets[1:], indices, side="right")
        if np.any(part_indices[1:] < part_indices[:-1]):
            raise ValueError(
                "ProductDomain coordinate selection cannot reorder product blocks."
            )

        selected_parts: list[Domain] = []
        for part_index, part in enumerate(self.parts):
            in_part = part_indices == part_index
            if np.any(in_part):
                local_indices = indices[in_part] - offsets[part_index]
                selected_parts.append(part.select_coordinates(local_indices))
        return ProductDomain(parts=tuple(selected_parts))


__all__ = [
    "Domain",
    "EuclideanSpace",
    "BoxDomain",
    "LpBallDomain",
    "OneHotDomain",
    "InterceptDomain",
    "ProductDomain",
]
