"""Independent privacy-accounting oracles for checking the budget seams.

Each oracle evaluates a published conversion directly, through
``dp_accounting`` or ``scipy.stats.norm``, and never through the library's
accounting modules, so a test that inverts or calibrates with the library can
check the result against a separate derivation.
"""

from __future__ import annotations

import math

import numpy as np
from dp_accounting import dp_event, rdp
from scipy.stats import norm

# Renyi orders for the zCDP oracle: dense near one, where small-epsilon and
# large-delta targets are optimized, and log-spaced out to very large orders,
# where tiny-delta targets are optimized. Every order yields a sound bound, so
# a finite grid can only overstate epsilon: by under one part in 1e5 at
# ordinary targets, and by up to a few parts in 1e3 where epsilon is tiny and
# delta is large, since ``dp_accounting`` skips orders below 1.01.
_ZCDP_ORDERS = np.unique(
    np.concatenate(
        [
            1.0 + np.logspace(-2, 0, 400),
            np.logspace(math.log10(2.0), 8.0, 8000),
        ]
    )
)


def epsilon_from_zcdp(rho: float, delta: float) -> float:
    """Return an approximate-DP epsilon certified for a ``rho``-zCDP mechanism.

    The smaller of two sound conversions: ``dp_accounting``'s Renyi-DP
    conversion (Canonne, Kamath & Steinke 2020, Proposition 12) of the zCDP
    curve ``alpha * rho`` over a dense order grid, and the Bun & Steinke 2016
    closed form ``rho + 2 * sqrt(rho * log(1 / delta))`` (Proposition 1.3).
    """
    accountant = rdp.RdpAccountant(orders=[float(order) for order in _ZCDP_ORDERS])
    accountant.compose(dp_event.ZCDpEvent(rho))
    by_rdp = float(accountant.get_epsilon(delta))
    by_closed_form = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    return min(by_rdp, by_closed_form)


def delta_from_gdp(mu: float, eps: float) -> float:
    """Return the exact approximate-DP delta of a ``mu``-GDP mechanism at ``eps``.

    Dong, Roth & Su 2019, Corollary 2.13, equivalently Balle & Wang 2018,
    Theorem 8 for a Gaussian release with sensitivity-to-scale ratio ``mu``:
    ``Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)``.
    """
    return float(
        norm.cdf(-eps / mu + mu / 2.0) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0)
    )


def epsilon_for_subsampled_gaussian(
    noise_multiplier: float, sampling_rate: float, num_steps: int, delta: float
) -> float:
    """Return ``dp_accounting``'s RDP epsilon for Poisson-subsampled Gaussians."""
    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.SelfComposedDpEvent(
            dp_event.PoissonSampledDpEvent(
                sampling_rate, dp_event.GaussianDpEvent(noise_multiplier)
            ),
            num_steps,
        )
    )
    return float(accountant.get_epsilon(delta))
