"""Measure observed Results and turn their values into comparable conclusions.

Task collections acquire Results. ``measure_results`` consumes that retained
frame and, given the collection, reads each Instance's Input once to produce
every per-Instance value its Task declares. The remaining operations take
frames and values without reading storage; each operation states which
columns it reads.

Costs are extended-real and lower is better: a trial whose algorithm failed
carries an infinite cost, and a declared coordinate that was never attempted
carries no row at all.
"""

from dpstats.analysis._error_ratios import (
    place_error_ratio_points,
    compute_robust_range,
    trace_running_median,
)
from dpstats.analysis._measurements import (
    get_metric_availability,
    measure_results,
)
from dpstats.analysis._outcomes import bin_outcomes, summarize_outcomes
from dpstats.analysis._plots import (
    plot_error_ratios,
    plot_outcomes,
    plot_profiles,
)
from dpstats.analysis._profiles import (
    aggregate_trials,
    compute_baseline,
    compute_error_ratio,
    compute_profile_curves,
)

__all__ = [
    "get_metric_availability",
    "measure_results",
    "aggregate_trials",
    "compute_baseline",
    "summarize_outcomes",
    "compute_error_ratio",
    "plot_profiles",
    "plot_error_ratios",
    "plot_outcomes",
    "compute_profile_curves",
    "place_error_ratio_points",
    "compute_robust_range",
    "trace_running_median",
    "bin_outcomes",
]
