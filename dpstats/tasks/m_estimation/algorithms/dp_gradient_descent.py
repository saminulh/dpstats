"""Differentially private gradient descent for M-estimation."""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget, accounting
from dpstats.exceptions import InvalidRegime

import math
from typing import Any, Literal, cast

import numpy as np

from dpstats.parameters import parameter_field, search_range, search_values
from dpstats.task import Array2D, ArrayND
from dpstats.utils import clip_rows
from dpstats.tasks import MEstimation
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)


_IterateAggregation = Literal["last", "tail_average", "polyak"]


class DPSGD(BaseAlgorithm):
    """Differentially Private Stochastic Gradient Descent. Minimize a loss by private gradient descent on random batches, or on the full batch as a special case.

    An implementation of Algorithm 1 from Abadi et al. (2016). Runs gradient
    descent on the average loss, at each step clipping every record's gradient
    to a fixed norm, adding Gaussian noise to their sum, and keeping the
    iterate inside the allowed parameter range. It is a Gaussian-DP, zCDP or
    approximate-DP mechanism, matching the offered budget.

    Requires:
        - A budget family admitting a GDP implication.
        - A finite ``metadata.loss_properties.loss_lipschitz_constant`` for
          ``C_grad_clip="loss_lipschitz"``.

    Parametrization notes:
        - ``C_grad_clip``: A number sets the clipping norm to
          ``C_grad_clip * sqrt(d)``, so one value means the same
          per-coordinate clip in any dimension; Abadi et al. (2016) prescribe no
          universal clip. ``"loss_lipschitz"`` sets it to the loss's declared
          Lipschitz constant, its largest possible gradient size.
        - ``eta``: The step size. Equals the initial learning rate of the
          MNIST experiments in Section 5.2 of Abadi et al. (2016).
        - ``num_steps``: In a public synthetic audit across tasks and budgets,
          this many steps beat ten steps in every tested setting. The paper
          prescribes no value.
        - ``subsample_rate``: Corresponds to $q$ in the paper. The default is
          full batch, the setting Brown et al. (2024) analyze.
        - ``iterate_aggregation``: Outputs the last iterate. In the same
          audit, averaging the later iterates was not better in every
          setting, and outputting only the last iterate leaves room for a
          later analysis that does not publish the earlier ones.

    Implementation notes:
        - With ``C_grad_clip="loss_lipschitz"``, every record's gradient is
          still clipped at the Lipschitz constant ``L`` before the sum is
          released.
        - With ``subsample_rate`` below one, each step draws a fresh batch in
          which every record is included independently with probability ``q``.
          Both the batch sum and its noise are divided by the expected batch
          size, so the step size means the same as in full-batch training and
          the realized private batch size is never used.
        - Iteration starts at the projection of zero onto the parameter domain,
          and each noisy update is projected back onto it.

    Reference:
        - abadi2016deep: Algorithm 1.
        - brown2024private: Lemma 2.1.
        - mironov2019renyi: Section 3.3.
    """

    TASK = MEstimation

    C_grad_clip: float | Literal["loss_lipschitz"] = parameter_field(
        default=1.0,
        search=search_range(0.01, 1000.0, scale="log"),
        description=(
            "Per-example gradient clipping constant: a number clips to norm "
            "``C_grad_clip * sqrt(d)``, and ``loss_lipschitz`` clips at the "
            "loss's declared Lipschitz constant."
        ),
    )
    eta: float = parameter_field(
        default=0.1,
        search=search_range(0.001, 2.0, scale="log"),
        description="Learning rate applied to each noisy gradient step.",
    )
    num_steps: int = parameter_field(
        default=100,
        search=search_range(2, 100),
        description="Number of gradient steps.",
    )
    subsample_rate: float = parameter_field(
        default=1.0,
        gt=0.0,
        le=1.0,
        search=search_range(0.01, 1.0, scale="log"),
        description="Poisson lot rate ``q`` in ``(0, 1]``; ``q=1.0`` runs full batch.",
    )
    iterate_aggregation: _IterateAggregation = parameter_field(
        default="last",
        search=search_values("last", "tail_average", "polyak"),
        description="Which function of the iterate trajectory to release (``last``, ``tail_average``, or ``polyak``).",
    )

    @override
    def run(
        self,
        metadata: MEstimationMetadata,
        input: MEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MEstimationOutput:
        n, d = metadata.n_samples, metadata.parameter_domain.dim
        if self.C_grad_clip == "loss_lipschitz":
            grad_clip = metadata.loss_properties.loss_lipschitz_constant
            if not math.isfinite(grad_clip):
                raise InvalidRegime(
                    "C_grad_clip 'loss_lipschitz' requires a finite loss "
                    "Lipschitz constant on the declared domains; use a "
                    "numeric C_grad_clip."
                )
        else:
            grad_clip = self.C_grad_clip * math.sqrt(d)
        q = self.subsample_rate
        num_steps = self.num_steps
        domain = metadata.parameter_domain
        theta = domain.project(np.zeros(d))
        # Iterates theta_1, ..., theta_T; aggregating them is post-processing.
        iterates: list[ArrayND] = []
        rng = np.random.default_rng(seed)

        def _apply_step(
            theta: ArrayND,
            grads: Array2D,
            normalizer: float,
            noise_std: float,
        ) -> ArrayND:
            """Clip rows, normalize the noisy sum by a public constant, and update.

            The released quantity is the clipped-gradient SUM, whose add/remove L2
            sensitivity is ``grad_clip`` (one clipped term of norm <= grad_clip;
            the replace-one bound 2*grad_clip is intentionally dropped). Dividing
            by ``normalizer`` is public post-processing. ``noise_std`` is already
            on that normalized scale, so never divide the noise a second time.
            Projecting the noisy iterate onto the declared parameter domain is
            deterministic post-processing of the already-released noisy step.
            """
            clipped = clip_rows(grads, grad_clip)
            avg_grad = np.sum(clipped, axis=0) / normalizer
            noise = rng.normal(loc=0.0, scale=noise_std, size=d)
            theta = theta - self.eta * (avg_grad + noise)
            return domain.project(theta)

        if q >= 1.0:
            # Every step uses all n records, so the calibration composes unamplified
            # Gaussian steps.
            noise_std = (
                accounting.compute_noise_multiplier_for_composed_gaussian(
                    cast(Any, budget), num_steps
                )
                * grad_clip
                / n
            )
            for _ in range(num_steps):
                grads: Array2D = metadata.loss.gradient(input.data, theta)
                theta = _apply_step(theta, grads, n, noise_std)
                iterates.append(theta)
        else:
            # Each record joins each lot independently with public probability q.
            # Approximate DP uses sampled-Gaussian amplification; zCDP and GDP
            # retain the sound unamplified route.
            expected_lot_size = q * n
            noise_std = (
                accounting.compute_noise_multiplier_for_subsampled_gaussian(
                    cast(Any, budget), q, num_steps
                )
                * grad_clip
                / expected_lot_size
            )
            for _ in range(num_steps):
                mask = rng.random(input.data.shape[0]) < q
                lot_grads: Array2D = metadata.loss.gradient(input.data[mask], theta)
                theta = _apply_step(theta, lot_grads, expected_lot_size, noise_std)
                iterates.append(theta)

        if self.iterate_aggregation == "last":
            return MEstimationOutput(theta=iterates[-1])
        if self.iterate_aggregation == "tail_average":
            tail = iterates[-math.ceil(num_steps / 2) :]
            return MEstimationOutput(theta=np.mean(tail, axis=0))
        return MEstimationOutput(theta=np.mean(iterates, axis=0))
