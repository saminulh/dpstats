"""Importable components for the complete simulated-instance sweep example."""

from __future__ import annotations

import time

import numpy as np
from pydantic import JsonValue

from dpstats.domains import BoxDomain
from dpstats.parameters import parameter_field
from dpstats.budgets import DPBudget
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class DemonstrationGaussianMechanism(GaussianMechanism):
    """Gaussian mean release with a visible delay for resume demonstrations."""

    delay_seconds: float = parameter_field(
        default=5.0,
        description="Artificial sleep before releasing, to demonstrate a resumable run.",
    )

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        if seed != 0:
            time.sleep(self.delay_seconds)
        return super().run(metadata, input, budget, seed)


def demonstration_instance(
    *, seed: int, n_samples: int = 8
) -> tuple[MeanEstimationMetadata, MeanEstimationInput, dict[str, JsonValue]]:
    """Produce one bounded mean-estimation payload before collection."""
    data = np.clip(np.random.default_rng(seed).normal(size=(n_samples, 1)), -1.0, 1.0)
    metadata = MeanEstimationMetadata(
        n_samples=n_samples,
        domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
    )
    return metadata, MeanEstimationInput(X=data), {"ground_truth": [0.0]}
