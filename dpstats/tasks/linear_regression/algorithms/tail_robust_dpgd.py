"""Tail-robust private linear regression from Chang et al. (2026).

The registered estimators implement the complete low- and high-dimensional
recipes in Sections 5.1.1 and 5.1.2. Their paper algorithms remain explicit
private boundaries shared with the recipe-level classes below.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import (
    BaseAlgorithm,
)
from dpstats.budgets import accounting
from dpstats.exceptions import InvalidRegime

import math

import numpy as np

from dpstats.parameters import parameter_field
from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
)
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D
from dpstats.utils import clip_rows
from dpstats.utils import split_seed
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)

# The paper specifies an exact ridge-Huber minimizer. These house solver
# controls are not experimental parameters; the sensitivity accounts for the
# declared gradient tolerance.
INITIALIZER_GRAD_TOL = 1e-6
INITIALIZER_MAX_ITERS = 1000


def _huber_score(residual: Array1D, tau: float) -> Array1D:
    """Evaluate the half-Huber score ``psi_tau(u) = clip(u, -tau, tau)``.

    Chang et al. use the ``u^2 / 2`` quadratic convention, so the score has
    unit slope inside the quadratic region and magnitude at most ``tau``.
    """
    score: Array1D = np.clip(residual, -tau, tau)
    return score


def _run_algorithm_1(
    X: Array2D,
    y: Array1D,
    n: int,
    beta_init: Array1D,
    tau: float,
    gamma: float,
    learning_rate: float,
    num_steps: int,
    budget: ApproxDPBudget,
    seed: int,
) -> Array1D:
    """Run Chang et al. Algorithm 1 from ``beta_init`` and return the final iterate.

    Each of ``num_steps`` steps releases the public-``n`` mean clipped Huber
    score-gradient (per-record L2 contribution ``<= gamma * tau``, so add/remove-one
    L2 sensitivity ``gamma * tau / n``) under Gaussian noise; the steps are composed
    to ``budget`` through the library's RDP accountant and the iterate updates are
    post-processing. Owns its accounting -- callers pass the budget, never a
    pre-calibrated noise scale.
    """
    sensitivity = gamma * tau / n
    noise_scale = (
        accounting.compute_noise_multiplier_for_composed_gaussian(budget, num_steps)
        * sensitivity
    )
    clipped_X = clip_rows(X, gamma)
    rng = np.random.default_rng(seed)
    beta = beta_init.copy()
    dim = beta.shape[0]
    for _ in range(num_steps):
        residual = y - X @ beta
        gradient_mean = (clipped_X.T @ _huber_score(residual, tau)) / n
        noise = rng.normal(0.0, 1.0, size=dim) * noise_scale
        beta = beta + learning_rate * (gradient_mean + noise)
    return beta


def _solve_ridge_huber(
    X_tilde: Array2D,
    y: Array1D,
    tau0: float,
    lambda_0: float,
    grad_tol: float,
    max_iters: int,
) -> Array1D:
    """Full-batch GD to ``grad_tol`` on the lambda_0-strongly-convex half-Huber ridge objective.

    Minimizes ``(1/n) sum_i rho_{tau0}(y_i - x_tilde_i^T beta) + (lambda_0/2)||beta||^2``
    (raw ``y``; the Huber score clips the residual). A non-private inner solve --
    its approximate minimizer is privatized by the caller's output Gaussian. The
    objective is ``(B^2 + lambda_0)``-smooth (``B^2 = 1 + p/36``), so a fixed public
    step ``1/(B^2 + lambda_0)`` converges linearly; the returned point has gradient
    norm ``<= grad_tol``.

    Raises:
        RuntimeError: If ``grad_tol`` is not reached within ``max_iters``. This
            is an unexpected execution failure, not a mechanism-defined bottom
            release.
    """
    n, dim = X_tilde.shape
    step = 1.0 / (1.0 + dim / 36.0 + lambda_0)
    beta: Array1D = np.zeros(dim, dtype=np.float64)
    for iteration in range(max_iters + 1):
        gradient = -(X_tilde.T @ _huber_score(y - X_tilde @ beta, tau0)) / n + (
            lambda_0 * beta
        )
        if np.linalg.norm(gradient) <= grad_tol:
            return beta
        if iteration < max_iters:
            beta = beta - step * gradient
    raise RuntimeError(
        "ridge-Huber initializer did not reach the gradient-norm tolerance "
        f"grad_tol = {grad_tol:.2e} within {max_iters} iterations."
    )


def _private_initializer(
    X: Array2D,
    y: Array1D,
    metadata: LinearRegressionMetadata,
    budget: ApproxDPBudget,
    seed: int,
) -> tuple[Array1D, float]:
    """Run the private initializer from Chang et al. Section 5.1.1.

    The clipped-moment scale estimate spends one quarter of ``budget`` and the
    ridge-Huber output perturbation spends the remaining three quarters. The
    latter uses add/remove sensitivity ``B * tau_0 / (lambda_0 * n)`` plus the
    declared optimization tolerance.
    """
    n = metadata.n_samples
    dim = metadata.dim
    eps_init = budget.eps
    output_budget = ApproxDPBudget(eps=0.75 * eps_init, delta=budget.delta)
    scale_seed, warm_seed = split_seed(seed, 2)

    # (eps_init/4, 0) scale estimate: two Laplace moment releases, each eps_init/8.
    log_n = math.log(n)
    y_clipped = np.clip(y, -log_n, log_n)
    eps_each = eps_init / 8.0
    # Under add/remove with public n, each moment is c + sum(v - c) / n, with
    # c = 0 for clipped y and c = log(n)^2 / 2 for its square, so the
    # sensitivities are log(n) / n and log(n)^2 / (2n).
    b1 = (log_n / n) / eps_each
    b2 = (log_n**2 / (2.0 * n)) / eps_each
    scale_rng = np.random.default_rng(scale_seed)
    m1 = np.sum(y_clipped) / n + scale_rng.laplace(0.0, b1)
    m2 = np.sum(y_clipped**2) / n + scale_rng.laplace(0.0, b2)
    variance = m2 - m1**2
    # Chang et al. Section 5.1.1 use 2 when the noisy variance is nonpositive.
    tau_0_hat = math.sqrt(variance) if variance > 0.0 else 2.0

    # Equation (12): preserve the canonical intercept in column zero and clip
    # the remaining coordinates to radius sqrt(p) / 6.
    X_tilde = X.copy()
    X_tilde[:, 1:] = clip_rows(X_tilde[:, 1:], math.sqrt(dim) / 6.0)
    ridge_penalty = 0.2  # Chang et al. Equation (12), lambda_0.
    beta_hat = _solve_ridge_huber(
        X_tilde,
        y,
        tau_0_hat,
        ridge_penalty,
        INITIALIZER_GRAD_TOL,
        INITIALIZER_MAX_ITERS,
    )
    design_bound = math.sqrt(1.0 + dim / 36.0)
    output_sensitivity = design_bound * tau_0_hat / (ridge_penalty * n) + (
        2.0 * INITIALIZER_GRAD_TOL / ridge_penalty
    )
    sigma = accounting.calibrate_gaussian_scale(output_budget, output_sensitivity)
    warm_rng = np.random.default_rng(warm_seed)
    beta_0: Array1D = beta_hat + sigma * warm_rng.normal(0.0, 1.0, size=dim)
    return beta_0, tau_0_hat


def _private_sparse_support(
    X: Array2D,
    y: Array1D,
    n: int,
    dim: int,
    sparsity: int,
    budget: PureDPBudget,
    seed: int,
) -> list[int]:
    """Recover the private support from Chang et al. Section 5.1.2.

    The canonical column-zero intercept is retained without spending privacy.
    The remaining ``sparsity - 1`` coordinates are selected by repeated Report
    Noisy Max.
    Each clipped signed product has magnitude at most ``sqrt(log(p * n))``;
    division by public ``n`` therefore gives add/remove sensitivity
    ``sqrt(log(p * n)) / n``.
    """
    candidate_indices = list(range(1, dim))
    clip_radius = math.sqrt(math.log(dim * n))
    scores = np.empty(len(candidate_indices), dtype=np.float64)
    for position, index in enumerate(candidate_indices):
        products = np.clip(y * X[:, index], -clip_radius, clip_radius)
        scores[position] = abs(np.sum(products) / n)

    rng = np.random.default_rng(seed)
    selected = np.zeros(len(candidate_indices), dtype=bool)
    num_selected = sparsity - 1
    # Report Noisy Max uses Laplace scale 2 * sensitivity / epsilon.
    # Each of the num_selected peeling rounds receives an equal pure-DP share.
    scale = 2.0 * num_selected * clip_radius / (n * budget.eps)
    for _ in range(num_selected):
        noisy_scores = scores + rng.laplace(0.0, scale, size=scores.size)
        noisy_scores[selected] = -np.inf
        chosen = int(np.argmax(noisy_scores))
        selected[chosen] = True

    return [0] + [candidate_indices[index] for index in np.flatnonzero(selected)]


def _run_algorithm_2(
    vector: Array1D,
    sparsity: int,
    budget: ApproxDPBudget,
    sensitivity: float,
    seed: int,
) -> Array1D:
    """Run Chang et al. Algorithm 2."""
    # Chang et al. Algorithm 2, using the sensitivity parameter lambda.
    scale = (
        2.0
        * sensitivity
        * math.sqrt(5.0 * sparsity * math.log(1.0 / budget.delta))
        / budget.eps
    )

    rng = np.random.default_rng(seed)
    selected = np.zeros(vector.size, dtype=bool)
    for _ in range(sparsity):
        magnitudes = np.abs(vector)
        scores = magnitudes + rng.laplace(0.0, scale, size=vector.size)
        scores[selected] = -np.inf
        chosen = int(np.argmax(scores))
        selected[chosen] = True

    output = np.zeros(vector.size, dtype=np.float64)
    output_noise = rng.laplace(0.0, scale, size=vector.size)
    output[selected] = vector[selected] + output_noise[selected]
    return output


def _noisy_ht_step_schedules(
    budget: ApproxDPBudget,
    num_steps: int,
) -> tuple[ApproxDPBudget, ...]:
    """Return the per-step NoisyHT budgets Chang et al. Section 3.3 authorizes.

    Section 3.3 authorizes the standard schedule when each of the ``num_steps``
    steps meets Algorithm 2's premise, and the advanced schedule when the whole
    offer also meets the strong-composition premise. Both premises read only the
    offered budget and the public step count, so an offer neither authorizes is
    a regime limit decidable before any Input is read.
    """
    schedules: list[ApproxDPBudget] = []
    standard = ApproxDPBudget(
        eps=budget.eps / num_steps,
        delta=budget.delta / num_steps,
    )
    if standard.eps <= 0.5 and standard.delta <= 0.011:
        schedules.append(standard)

    if budget.eps <= 1.0 and budget.delta <= 0.01:
        advanced = ApproxDPBudget(
            eps=budget.eps
            * math.sqrt(2.0 / (5.0 * num_steps * math.log(2.0 / budget.delta))),
            delta=budget.delta / (2.0 * num_steps),
        )
        if advanced.eps <= 0.5 and advanced.delta <= 0.011:
            schedules.append(advanced)

    if not schedules:
        raise InvalidRegime(
            "No paper-valid NoisyHT composition is available: each step requires "
            "epsilon <= 0.5 and delta <= 0.011; the advanced schedule additionally "
            "requires total epsilon <= 1 and delta <= 0.01."
        )
    return tuple(schedules)


def _noisy_ht_step_budget(
    budget: ApproxDPBudget,
    schedules: tuple[ApproxDPBudget, ...],
    sparsity: int,
    sensitivity: float,
) -> ApproxDPBudget:
    """Choose the lower-noise schedule among those Section 3.3 authorizes.

    ``schedules`` are the authorized per-step budgets for ``budget``.
    """
    best_budget = schedules[0]
    best_scale = math.inf
    for candidate in schedules:
        candidate_scale = (
            2.0
            * sensitivity
            * math.sqrt(5.0 * sparsity * math.log(1.0 / candidate.delta))
            / candidate.eps
        )
        if candidate_scale < best_scale:
            best_budget, best_scale = candidate, candidate_scale
    return best_budget


def _run_algorithm_3(
    X: Array2D,
    y: Array1D,
    n: int,
    beta_init: Array1D,
    tau: float,
    gamma: float,
    learning_rate: float,
    num_steps: int,
    sparsity: int,
    budget: ApproxDPBudget,
    step_schedules: tuple[ApproxDPBudget, ...],
    seed: int,
) -> Array1D:
    """Run Chang et al. Algorithm 3 from a released initial value.

    The intermediate update is a public-``n`` average of contributions with
    infinity norm at most ``gamma * tau``. Its add/remove sensitivity is
    therefore ``learning_rate * gamma * tau / n``, half the paper's replace-one
    bound, and Algorithm 2 receives exactly that value as Lemma D.1's
    ``lambda``: the lemma's proof consumes only non-strict domination of each
    coordinate difference by ``lambda``. Algorithm 2 privatizes every adaptive
    update, and this boundary owns the paper's standard-versus-advanced
    composition choice.

    Both uses of ``n`` here are plain ``1/n`` divisions. ``n > 1`` is required
    only by the caller, whose ``num_steps`` and ``tau`` come from the Section
    5.1.2 ``log(n)`` recipe.
    """
    sensitivity = learning_rate * gamma * tau / n
    step_budget = _noisy_ht_step_budget(
        budget,
        step_schedules,
        sparsity,
        sensitivity,
    )

    covariate_norms = np.max(np.abs(X), axis=1)
    clip_weights = np.ones_like(covariate_norms)
    nonzero = covariate_norms > 0.0
    clip_weights[nonzero] = np.minimum(1.0, gamma / covariate_norms[nonzero])
    clipped_X = X * clip_weights[:, None]

    beta = beta_init.copy()
    for iteration_seed in split_seed(seed, num_steps):
        residual = y - X @ beta
        huber_scores = _huber_score(residual, tau)
        intermediate = beta + (learning_rate / n) * (clipped_X.T @ huber_scores)
        beta = _run_algorithm_2(
            intermediate,
            sparsity,
            step_budget,
            sensitivity,
            iteration_seed,
        )
    return beta


class TailRobustDPGD(BaseAlgorithm):
    """Fit a linear regression that tolerates heavy-tailed noise, by private gradient descent on the Huber loss.

    An implementation of the method in Section 5.1.1 of Chang, Yang, Zha and
    Zhou (2026). Computes a private ridge-Huber starting fit and an estimate of
    the response scale, then runs private gradient descent on the Huber loss
    from that start. It is an approximate-DP mechanism.

    Requires:
        - A finite ``ApproxDPBudget``.
        - A declared column-zero intercept.

    Parametrization notes:
        - ``learning_rate``: Corresponds to $η_0$ in the paper. Set as in the
          practical recipe of Section 5.1.1.

    Implementation notes:
        - The response scale comes from the first two moments of responses
          clipped at ``log(n)``, released with Laplace noise at the declared
          ``n``.
        - The paper's starting point is an exact ridge-Huber minimizer. Here it
          is solved to a fixed gradient-norm tolerance, and its sensitivity is
          increased by twice that tolerance divided by the ridge penalty to
          cover the gap.
        - Equation (12) clips the covariates but leaves the intercept column
          unchanged.

    Reference:
        - chang2026adapting: Section 5.1.1.
    """

    TASK = LinearRegression

    learning_rate: float = parameter_field(
        default=0.2,
        gt=0.0,
        description="Gradient-descent learning rate ``eta_0`` (Theorem 1's theorem-ranged free parameter, ``eta_0 = eta/(2 lambda_1)``).",
    )

    @override
    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        # Metadata-decidable scope refusals, before any Input access.
        budget = accounting.spend_as_approx_dp(budget)
        if not math.isfinite(budget.eps):
            raise InvalidRegime(
                "TailRobustDPGD refuses an infinite epsilon: its limit keeps "
                "nonzero per-step noise (tau grows as sqrt(eps) while the GDP "
                "calibration's mu grows as sqrt(2 * eps), so their ratio tends "
                "to a nonzero constant instead of vanishing), and no accepted "
                "accounting record states that limiting noise scale."
            )
        if not metadata.has_intercept:
            raise InvalidRegime(
                "TailRobustDPGD's Section 5.1.1 initializer requires a declared "
                "column-zero intercept (`InterceptDomain`) in "
                "metadata.design_domain."
            )
        p = metadata.dim
        n = metadata.n_samples

        # Section 5.1.1 split: a sixth to the initializer, the rest to the main loop.
        init_budget = ApproxDPBudget(eps=budget.eps / 6.0, delta=budget.delta / 6.0)
        main_budget = ApproxDPBudget(
            eps=budget.eps - init_budget.eps,
            delta=budget.delta - init_budget.delta,
        )
        init_seed, main_seed = split_seed(seed, 2)

        # Private warm start + private noise-scale estimate (owns its eps/6 accounting).
        beta_0, tau_0_hat = _private_initializer(
            input.X, input.y, metadata, init_budget, init_seed
        )

        # Chang et al. Section 5.1.1 fixes gamma's leading constant at 0.5.
        gamma = 0.5 * math.sqrt(p + math.log(n))
        # Its tau formula uses the overall epsilon, not the 5eps/6 main share.
        tau = (
            0.04  # Chang et al. Section 5.1.1 leading constant.
            * tau_0_hat
            * math.sqrt(budget.eps * n / (p + math.log(n)))
        )
        # Chang et al. Section 5.1.1 uses T = ceil(2 log n).
        num_steps = max(1, math.ceil(2.0 * math.log(n)))

        beta = _run_algorithm_1(
            input.X,
            input.y,
            n,
            beta_init=beta_0,
            tau=tau,
            gamma=gamma,
            learning_rate=self.learning_rate,
            num_steps=num_steps,
            budget=main_budget,
            seed=main_seed,
        )
        return LinearRegressionOutput(beta=beta)


class TailRobustSparseHuberIHT(BaseAlgorithm):
    """Fit a sparse linear regression that tolerates heavy-tailed noise, privately keeping only a few coefficients nonzero.

    An implementation of the method in Section 5.1.2 of Chang, Yang, Zha and
    Zhou (2026). Privately picks a small set of features, fits a private robust
    starting point on them, then runs private gradient descent on the Huber
    loss with noisy hard thresholding, which keeps only the largest
    coefficients. It is an approximate-DP mechanism.

    Requires:
        - A finite ``ApproxDPBudget``.
        - A declared column-zero intercept (``metadata.has_intercept``).
        - A ``sparsity`` no greater than ``metadata.dim``.
        - ``metadata.n_samples`` above one.
        - An offer meeting Section 3.3's NoisyHT composition premise at
          ``T = ceil(2 log n)`` steps.

    Parametrization notes:
        - ``sparsity``: Corresponds to $s$ in the paper. Set to $⌈1.2 s^*⌉$ at
          the true sparsity $s^* = 10$ of the paper's simulations in Sections
          5.1.2 and 5.3. The floor of 10 is Algorithm 2's privacy-lemma
          condition.
        - ``learning_rate``: Corresponds to $η_0$ in the paper. Set as in the
          practical recipe of Section 5.1.2.

    Implementation notes:
        - The paper assumes a constant-ones intercept column that never enters
          support selection. The task puts a declared intercept in column zero,
          and it is always kept.

    Reference:
        - chang2026adapting: Section 5.1.2.
    """

    TASK = LinearRegression

    sparsity: int = parameter_field(
        default=12,
        ge=10,
        description=(
            "Working sparsity ``s`` used by support recovery and NoisyHT, "
            "bounded below at 10 because Algorithm 2's privacy lemma assumes "
            "s >= 10."
        ),
    )
    learning_rate: float = parameter_field(
        default=0.01,
        gt=0.0,
        description=(
            "Gradient learning rate ``eta_0`` (the paper's theorem-ranged "
            "free parameter)."
        ),
    )

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        """Output the Section 5.1.2 sparse private Huber estimate."""
        budget = accounting.spend_as_approx_dp(budget)
        if budget.is_unconstrained:
            raise InvalidRegime(
                "TailRobustSparseHuberIHT has no unconstrained limit: its "
                "noisy-hard-thresholding schedule refuses every epsilon above a "
                "bound set by the step count."
            )
        total_epsilon = budget.eps
        main_budget = ApproxDPBudget(
            eps=budget.eps / 3.0,
            delta=budget.delta / 2.0,
        )
        support_budget = PureDPBudget(eps=budget.eps / 3.0)
        coefficient_init_budget = ApproxDPBudget(
            eps=budget.eps / 3.0,
            delta=budget.delta / 2.0,
        )
        if not metadata.has_intercept:
            raise InvalidRegime(
                "TailRobustSparseHuberIHT's Section 5.1.2 initializer requires "
                "a declared column-zero intercept (`InterceptDomain`) in "
                "metadata.design_domain."
            )
        if self.sparsity > metadata.dim:
            raise InvalidRegime("sparsity cannot exceed the public feature dimension.")
        if metadata.n_samples <= 1:
            raise InvalidRegime(
                "TailRobustSparseHuberIHT requires n_samples > 1 because the "
                "paper's recipe uses log(n)."
            )

        n = metadata.n_samples
        dim = metadata.dim
        # Chang et al. Section 5.1.2 uses T = ceil(2 log n) Algorithm 3 steps.
        num_steps = math.ceil(2.0 * math.log(n))
        # Section 3.3's composition premises read only the offer and that public
        # step count, so an offer they authorize no schedule for is refused here
        # rather than after the initializer has spent part of it on the Input.
        step_schedules = _noisy_ht_step_schedules(main_budget, num_steps)
        support_seed, coefficient_seed, main_seed = split_seed(seed, 3)

        selected_indices = _private_sparse_support(
            input.X,
            input.y,
            n,
            dim,
            self.sparsity,
            support_budget,
            support_seed,
        )
        # Equation (12)'s clip radius is fixed by the paper, so no declared covariate
        # bound enters the initializer's sensitivity; the selected coordinates stay
        # unbounded.
        selected_metadata = LinearRegressionMetadata(
            n_samples=n,
            design_domain=ProductDomain(
                parts=(InterceptDomain(), EuclideanSpace(dim=self.sparsity - 1))
            ),
            response_domain=metadata.response_domain,
        )
        beta_selected, tau_0_hat = _private_initializer(
            input.X[:, selected_indices],
            input.y,
            selected_metadata,
            coefficient_init_budget,
            coefficient_seed,
        )
        beta_0 = np.zeros(dim, dtype=np.float64)
        beta_0[selected_indices] = beta_selected

        # Chang et al. Section 5.1.2 fixes these practical-recipe constants.
        gamma = 0.5 * math.sqrt(math.log(dim * n))
        tau = (
            0.04
            * tau_0_hat
            * math.sqrt(
                n * total_epsilon / (self.sparsity * math.log(dim) + math.log(n))
            )
        )
        beta = _run_algorithm_3(
            input.X,
            input.y,
            n,
            beta_init=beta_0,
            tau=tau,
            gamma=gamma,
            learning_rate=self.learning_rate,
            num_steps=num_steps,
            sparsity=self.sparsity,
            budget=main_budget,
            step_schedules=step_schedules,
            seed=main_seed,
        )
        return LinearRegressionOutput(beta=beta)
