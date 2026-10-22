"""Final-iterate one-pass DP-GD schedule family of Bombari et al."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm

import math
from typing import Literal, cast

import numpy as np
from pydantic import model_validator

from dpstats.budgets import accounting
from dpstats.parameters import parameter_field, search_range, search_values
from dpstats.budgets import DPBudget, ZCDPBudget
from dpstats.task import Array1D, Array2D
from dpstats.utils import split_seed
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)


def _harmonic_learning_rates(
    n: int,
    d: int,
    scale: float,
    tau_over_gamma: float,
) -> Array1D:
    """Generate the harmonic learning rates from Equation (5.18).

    ``eta_k = scale / (k + d * tau_over_gamma)``: the paper's continuous
    ``eta~(t) = scale/(t+tau)`` at ``eta_k = eta~(k/n)/n`` gives
    ``scale / (k + n*tau)``, and ``n*tau == d*(tau/gamma)`` since
    ``gamma = d/n``, i.e. ``tau_over_gamma = tau/gamma``.
    """
    positions = np.arange(1, n + 1)
    return scale / (positions + d * tau_over_gamma)


def _polynomial_learning_rates(
    n: int,
    learning_rate_scale: float,
    alpha: float,
) -> Array1D:
    """Generate the polynomial learning rates from Equations (4.4)-(5.1)."""
    initial_rate = learning_rate_scale / n
    if alpha == 0.0:
        learning_rates = np.full(n, initial_rate)
    else:
        remaining_fractions = (n - np.arange(1, n + 1)) / n
        learning_rates = initial_rate * remaining_fractions**alpha

    return learning_rates


def _equation_4_2_noise_multipliers(
    learning_rates: Array1D,
    budget: ZCDPBudget,
) -> Array1D:
    """Construct the Equation (4.2) noise schedule from a zCDP budget."""
    paper_rho = math.sqrt(2.0 * budget.rho)
    next_rates = np.append(learning_rates[1:], 0.0)
    return np.sqrt(learning_rates**2 - next_rates**2) / paper_rho


def _run_algorithm_1(
    X: Array2D,
    y: Array1D,
    learning_rates: Array1D,
    noise_multipliers: Array1D,
    clipping_norm: float,
    seed: int,
) -> Array1D:
    """Run Bombari et al. Algorithm 1 on one fixed-length sequence."""
    rng = np.random.default_rng(seed)
    theta: Array1D = np.zeros(X.shape[1])

    for k in range(X.shape[0]):
        x_k = X[k]
        gradient: Array1D = x_k * (x_k @ theta - y[k])
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > clipping_norm:
            gradient = gradient * (clipping_norm / gradient_norm)

        row_norm_squared = x_k @ x_k
        # The source's ``min(eta_k, 2 / ||x_k||^2)`` cap takes the limit
        # ``eta_k`` at a zero row.
        with np.errstate(divide="ignore"):
            adaptive_learning_rate = min(learning_rates[k], 2.0 / row_norm_squared)

        parameter_noise = (
            2.0 * clipping_norm * noise_multipliers[k] * rng.normal(size=theta.size)
        )
        theta = theta - adaptive_learning_rate * gradient + parameter_noise

    return theta


class OnePassDPGD(BaseAlgorithm):
    """One-Pass DP Gradient Descent. Fit linear regression with one pass of private gradient descent, one record per step.

    An implementation of Algorithm 1 from Bombari, Luo, Seroussi and Mondelli
    (2026). Shuffles the records and visits each once: every step clips that
    record's gradient to a fixed norm, takes a gradient step, and adds Gaussian
    noise. Outputs the last iterate. It is a zCDP mechanism.

    Requires:
        - A budget family admitting a zCDP implication.

    Parametrization notes:
        - ``c``: Corresponds to $c$ in the paper's clipping norm
          $C_{clip} = c√d$ of Equation 4.3. Set as in the paper's simulations
          in Section 7.
        - ``scale``: Corresponds to $β$ for the harmonic schedule and $η(0)$
          for the polynomial one in the paper, and to the authors' searched
          coordinate $v$ in their code. The search range 1 to 1000 comes from
          their code. The paper does not disclose the values
          its Figures 7 and 8 optimize, so the default was picked by a public
          calibration study inside that range; both schedules share it.
        - ``schedule``: The authors' schedule switch; their code selects the
          harmonic schedule with $α = -1$. The default is the polynomial
          schedule, which needs no further setting, while the harmonic one
          requires ``tau_over_gamma``.
        - ``alpha``: Corresponds to $α$ in Equation 5.1 of the paper. The
          search values come from the authors' code; the default is one of
          them, picked by the same calibration study as ``scale``.
        - ``tau_over_gamma``: Corresponds to the paper's $τ$ divided by
          $γ = d/n$, the authors' searched coordinate. Its range, 16 to 1000,
          comes from the authors' code. It has no class default, since
          only ``schedule="harmonic"`` requires it.
        - Fixed $σ_k$: Equation 4.2 fixes it from the schedule and $ρ$, and it
          is not exposed.
        - Fixed $A$, $λ_{min}$ and $λ_{max}$: They are the theorem witnesses of
          Appendix D.3 and D.105, which need private quantities, and are not
          exposed. D.3 writes the harmonic ``scale`` as $α$.

    Implementation notes:
        - The paper proves zCDP for replacing one record in a fixed-length
          sequence. Adding or removing a record is covered by a construction used
          only in the proof: truncate above the declared ``n`` or pad below it
          with records of zero gradient, then order the slots at random, so
          neighbors differ in at most one slot. On inputs of the declared size
          this is exactly a random permutation of the records followed by
          Algorithm 1, which is what runs. The random order changes the
          output's distribution for a fixed ordered dataset, but not the
          trajectory or risk the paper analyzes for exchangeable data.

    Reference:
        - bombari2026high: Algorithm 1 of arXiv v2.
    """

    TASK = LinearRegression

    c: float = parameter_field(
        default=0.1,
        gt=0.0,
        description="Clipping coordinate in ``C_clip = c sqrt(d)``.",
    )
    schedule: Literal["polynomial", "harmonic"] = parameter_field(
        default="polynomial",
        description="Which of the paper's two rate schedules to run.",
    )
    scale: float = parameter_field(
        default=30.0,
        gt=0.0,
        search=search_range(1, 1000, scale="log"),
        description="Per-step rate scale/n (Eq. 4.4): eta~(0) for the "
        "polynomial schedule, beta for the harmonic schedule.",
    )
    alpha: float | None = parameter_field(
        default=0.5,
        search=search_values(0, 0.5, 1, 2),
        description="Polynomial learning-rate exponent (Eq. 5.1); required "
        "iff schedule=='polynomial', where it must be 0, 0.5, or at least 1; "
        "refused if explicitly supplied for schedule=='harmonic'.",
    )
    tau_over_gamma: float | None = parameter_field(
        default=None,
        gt=0.0,
        description="Harmonic offset tau/gamma with gamma=d/n (Eq. 5.18), "
        "required iff schedule=='harmonic' and absent from the default tuning "
        "grid, which belongs to the polynomial schedule.",
    )

    @model_validator(mode="after")
    def _validate_schedule_fields(self) -> "OnePassDPGD":
        if self.schedule == "polynomial":
            if self.alpha is None:
                raise ValueError("alpha is required when schedule='polynomial'.")
            if self.tau_over_gamma is not None:
                raise ValueError(
                    "tau_over_gamma must not be set when schedule='polynomial'."
                )
            if self.alpha not in (0.0, 0.5) and self.alpha < 1.0:
                raise ValueError("alpha must be 0, 0.5, or at least 1.")
        else:
            if self.tau_over_gamma is None:
                raise ValueError("tau_over_gamma is required when schedule='harmonic'.")
            # Accept an explicit None like an omitted alpha: a full field dump, as in
            # `type(x)(**dict(x))`, carries alpha=None for a harmonic instance.
            if "alpha" in self.model_fields_set and self.alpha is not None:
                raise ValueError("alpha must not be set when schedule='harmonic'.")
            # alpha's polynomial default must never enter a harmonic identity.
            object.__setattr__(self, "alpha", None)
        return self

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        available = accounting.spend_as_zcdp(budget)
        n = metadata.n_samples
        d = metadata.dim
        if self.schedule == "polynomial":
            # The constructor validator makes alpha non-None here.
            learning_rates = _polynomial_learning_rates(
                n=n,
                learning_rate_scale=self.scale,
                alpha=cast(float, self.alpha),
            )
        else:
            # The constructor validator makes tau_over_gamma non-None here.
            learning_rates = _harmonic_learning_rates(
                n=n,
                d=d,
                scale=self.scale,
                tau_over_gamma=cast(float, self.tau_over_gamma),
            )
        noise_multipliers = _equation_4_2_noise_multipliers(
            learning_rates,
            available,
        )

        projection_seed, algorithm_seed = split_seed(seed, 2)
        order = np.random.default_rng(projection_seed).permutation(n)
        clipping_norm = self.c * math.sqrt(d)
        theta = _run_algorithm_1(
            X=input.X[order],
            y=input.y[order],
            learning_rates=learning_rates,
            noise_multipliers=noise_multipliers,
            clipping_norm=clipping_norm,
            seed=algorithm_seed,
        )
        return LinearRegressionOutput(beta=theta)
