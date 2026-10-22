"""Unit-square coordinate transport shared by the simple-regression algorithms.

Public-metadata readers and the transport itself live here.

The bounds-assuming mechanisms all run on records affinely transported from
their public scalar domains onto the unit square and release a line there;
these helpers own that transport in one spelling so every algorithm normalizes
and de-normalizes identically.
"""

from __future__ import annotations


import math

from dpstats.domains import Domain
from dpstats.task import Array1D
from dpstats.exceptions import InvalidRegime


def require_scalar_interval(domain: Domain, field_name: str) -> tuple[float, float]:
    """Return the public interval enclosing a scalar field's declared domain.

    The interval comes from the domain's L2 enclosing ball, which in dimension
    one is the bounding interval ``[center - radius, center + radius]``. Any
    declared scalar domain shape therefore supplies it, and a box supplies its
    own endpoints. The interval is a sound enclosure, so clipping to it and
    calibrating noise from its width are valid for the declared domain.

    An unbounded field has no such interval. That is decidable from public
    metadata alone, so it is refused before any input is read.

    Args:
        domain: Public domain declared for the field.
        field_name: Field named in the error messages, ``"x"`` or ``"y"``.

    Returns:
        The ``(lower, upper)`` endpoints of the enclosing interval.

    Raises:
        ValueError: If the field is unbounded or has no positive-width interval.
    """
    center, radius = domain.l2_enclosing_ball()
    if not math.isfinite(radius):
        raise InvalidRegime(
            f"The {field_name} interval requires a sound L2 enclosing ball."
        )
    lower = float(center[0]) - radius
    upper = float(center[0]) + radius
    if upper <= lower:
        raise InvalidRegime(f"{field_name}_domain must have positive width.")
    return lower, upper


def to_unit_interval(values: Array1D, interval: tuple[float, float]) -> Array1D:
    """Affinely map values from a public interval onto ``[0, 1]``.

    ``interval`` is a positive-width ``(lower, upper)`` enclosure (from
    :func:`require_scalar_interval`), so admitted values land in the unit
    interval and the map is invertible.
    """
    lower, upper = interval
    return (values - lower) / (upper - lower)


def line_from_unit_square(
    normalized_slope: float,
    normalized_intercept: float,
    x_interval: tuple[float, float],
    y_interval: tuple[float, float],
) -> tuple[float, float]:
    """Invert the unit-square normalization on a released line.

    Given ``y_unit = normalized_slope * x_unit + normalized_intercept`` on the
    unit square, returns the ``(slope, intercept)`` of the same line in the
    original coordinates, where ``x_unit = (x - x_lower)/x_width`` and
    ``y = y_lower + y_width * y_unit``. Deterministic post-processing of the
    released normalized line.
    """
    x_lower, x_upper = x_interval
    y_lower, y_upper = y_interval
    x_width = x_upper - x_lower
    y_width = y_upper - y_lower
    slope = normalized_slope * y_width / x_width
    intercept = y_lower + y_width * normalized_intercept - slope * x_lower
    return slope, intercept


def line_from_quartile_predictions(
    prediction_25: float,
    prediction_75: float,
) -> tuple[float, float]:
    """Convert released predictions at ``x = 0.25`` and ``x = 0.75`` to a line.

    The unique line through the two prediction points has slope
    ``(p75 - p25) / 0.5`` and intercept ``p25 - 0.25 * slope``; both are
    deterministic post-processing of the released pair. Returns the
    ``(slope, intercept)`` in the same (normalized) coordinates as the
    predictions.
    """
    slope = 2 * (prediction_75 - prediction_25)
    intercept = prediction_25 - 0.25 * slope
    return slope, intercept
