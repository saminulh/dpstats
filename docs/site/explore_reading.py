"""The constants Explore's error-ratio and trial-outcome readings are taken with.

The page reads them from its payload and the notebook from its parameter
cell, so the two read the same chart; the library takes each as an argument.
"""

from __future__ import annotations

# The running median's window, as a share of the benchmark's weight, and how
# many evenly spaced positions in weight-quantile space it is read at.
RUNNING_WINDOW = 0.15
RUNNING_POSITIONS = 41

# The opening ratio range reaches this many interquartile ranges beyond each
# weighted quartile: Tukey's far-out fences.
RANGE_FENCE = 3.0

# The share of each axis's span, property then ratio, left empty beyond the
# opening range, where values beyond the view sit.
RANGE_PADDING = [0.08, 0.05]

# How many bins the trial outcomes are cut into: equal spans of the property
# on a logarithmic or linear scale, equal shares of the weight on a quantile
# one; and the share of the property axis's span left empty beyond the bins.
OUTCOME_BINS = 20
QUANTILE_BINS = 10
OUTCOME_PADDING = 0.05


def reading_parameters() -> dict[str, float | int | list[float]]:
    """The error-ratio constants by the names the page and the notebook read."""
    return {
        "RUNNING_WINDOW": RUNNING_WINDOW,
        "RUNNING_POSITIONS": RUNNING_POSITIONS,
        "RANGE_FENCE": RANGE_FENCE,
        "RANGE_PADDING": list(RANGE_PADDING),
    }


def outcome_parameters() -> dict[str, float | int]:
    """The trial-outcome constants by the names the page and the notebook read."""
    return {
        "OUTCOME_BINS": OUTCOME_BINS,
        "QUANTILE_BINS": QUANTILE_BINS,
        "OUTCOME_PADDING": OUTCOME_PADDING,
    }
