"""Gaussian/GDP profiles, composition, and sampled-Gaussian accounting."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
import dp_accounting
from dp_accounting import dp_event, mechanism_calibration, rdp

from dpstats.budgets.accounting._gdp import (
    _delta_from_gdp,  # pyright: ignore[reportPrivateUsage]
    _epsilon_from_gdp,  # pyright: ignore[reportPrivateUsage]
    spend_as_gdp,
)

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    NoPrivacy,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime


def calibrate_gaussian_scale(budget: DPBudget, sensitivity: float) -> float:
    """Calibrate a Gaussian release to an offered budget.

    Approximate DP uses the exact Balle--Wang analytic calibration through the
    GDP profile. zCDP and GDP use their defining Gaussian guarantees. Finite
    pure DP is unsupported for positive sensitivity.

    Args:
        budget: Offered budget of any family.
        sensitivity: Nonnegative finite L2 sensitivity of the released vector.

    Returns:
        The Gaussian noise scale; zero when sensitivity is zero or the offer
        is unconstrained.

    Raises:
        InvalidRegime: If sensitivity is positive and the offer is a finite
            pure-DP budget.

    Example:
        >>> budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        >>> round(calibrate_gaussian_scale(budget, 1.0), 6)
        3.730632
    """
    if not math.isfinite(sensitivity) or sensitivity < 0:
        raise ValueError("sensitivity must be finite and nonnegative.")
    if sensitivity == 0:
        return 0.0
    return sensitivity / spend_as_gdp(budget).mu


def _gaussian_gdp_profile(scale: float, sensitivity: float) -> GDPBudget:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive.")
    return GDPBudget(mu=sensitivity / scale)


def compute_delta_from_gaussian_mechanism(
    scale: float,
    sensitivity: float,
    eps: float,
) -> float:
    """Return the exact approximate-DP delta of a Gaussian release.

    Args:
        scale: Positive finite standard deviation of the added noise.
        sensitivity: Positive finite L2 sensitivity of the released vector.
        eps: Nonnegative finite approximate-DP epsilon to evaluate at.

    Returns:
        The smallest delta at which the release is ``(eps, delta)``-DP.

    Example:
        >>> round(compute_delta_from_gaussian_mechanism(1.0, 1.0, 1.0), 6)
        0.126937
    """
    return _delta_from_gdp(_gaussian_gdp_profile(scale, sensitivity), eps)


def compute_epsilon_from_gaussian_mechanism(
    scale: float,
    sensitivity: float,
    delta: float,
) -> float:
    """Return the exact approximate-DP epsilon of a Gaussian release.

    Args:
        scale: Positive finite standard deviation of the added noise.
        sensitivity: Positive finite L2 sensitivity of the released vector.
        delta: Target approximate-DP delta in (0, 1).

    Returns:
        The smallest epsilon at which the release is ``(epsilon, delta)``-DP.

    Example:
        >>> round(compute_epsilon_from_gaussian_mechanism(1.0, 1.0, 1e-5), 6)
        4.377178
    """
    return _epsilon_from_gdp(_gaussian_gdp_profile(scale, sensitivity), delta)


def compute_noise_multiplier_for_composed_gaussian(
    budget: DPBudget, num_steps: int
) -> float:
    """Calibrate a common multiplier for composed Gaussian mechanisms.

    Approximate DP uses exact GDP composition, zCDP uses its native Gaussian
    guarantee, and GDP adds squared mu. Finite pure DP is unsupported.

    Args:
        budget: Offered budget of any family, spent over the whole composition.
        num_steps: Positive number of composed Gaussian releases.

    Returns:
        The noise multiplier (scale per unit sensitivity) shared by every step.

    Raises:
        InvalidRegime: If the offer is a finite pure-DP budget.

    Example:
        >>> compute_noise_multiplier_for_composed_gaussian(GDPBudget(mu=1.0), 4)
        2.0
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    return math.sqrt(num_steps) / spend_as_gdp(budget).mu


@lru_cache(maxsize=None)
def compute_noise_multiplier_for_subsampled_gaussian(
    budget: DPBudget,
    sampling_rate: float,
    num_steps: int,
) -> float:
    """Calibrate Gaussian noise for Poisson-subsampled composition.

    Approximate DP retains sampled-Gaussian RDP amplification. zCDP and GDP
    use the unamplified composed-Gaussian route. Finite pure DP is unsupported.

    Args:
        budget: Offered budget of any family, spent over the whole composition.
        sampling_rate: Poisson inclusion probability of each record, in (0, 1].
        num_steps: Positive number of composed subsampled releases.

    Returns:
        The noise multiplier (scale per unit sensitivity) shared by every step.

    Raises:
        InvalidRegime: If the offer is a finite pure-DP budget.

    Example:
        >>> budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        >>> round(compute_noise_multiplier_for_subsampled_gaussian(budget, 0.01, 100), 3)
        1.08
    """
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1].")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if budget.is_unconstrained:
        return 0.0
    match budget:
        case ZCDPBudget() | GDPBudget():
            return compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
        case PureDPBudget():
            raise InvalidRegime(
                "Cannot spend a PureDPBudget as subsampled Gaussian composition: "
                "no finite-noise Gaussian guarantee implies pure DP."
            )
        case ApproxDPBudget():
            pass
        case NoPrivacy():
            return 0.0
        case _:
            raise InvalidRegime(
                f"Cannot spend an offered {type(budget).__name__}: accounting "
                "implements no implication from that budget family."
            )
    if sampling_rate == 1.0:
        return compute_noise_multiplier_for_composed_gaussian(budget, num_steps)

    def make_fresh_accountant() -> rdp.RdpAccountant:
        return rdp.RdpAccountant()

    def make_event_from_param(noise_multiplier: float) -> dp_event.DpEvent:
        return dp_event.SelfComposedDpEvent(
            dp_event.PoissonSampledDpEvent(
                sampling_rate, dp_event.GaussianDpEvent(noise_multiplier)
            ),
            num_steps,
        )

    # The finite RDP order grid has an epsilon floor. The unamplified exact-GDP
    # route is sound for internal data-independent sampling and supplies the
    # tighter fallback when that grid cannot resolve the target.
    unamplified_multiplier = compute_noise_multiplier_for_composed_gaussian(
        budget, num_steps
    )
    # Conservative sampled-RDP convergence chatter is silenced for this call.
    absl_logger = logging.getLogger("absl")
    previous_level = absl_logger.level
    absl_logger.setLevel(logging.ERROR)
    try:
        amplified_multiplier = float(
            dp_accounting.calibrate_dp_mechanism(
                make_fresh_accountant,
                make_event_from_param,
                budget.eps,
                budget.delta,
            )
        )
    except mechanism_calibration.NoBracketIntervalFoundError:
        return unamplified_multiplier
    finally:
        absl_logger.setLevel(previous_level)
    return min(amplified_multiplier, unamplified_multiplier)
