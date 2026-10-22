"""Create the registries and authored sweep for the simulated-instance example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from components import DemonstrationGaussianMechanism, demonstration_instance
from dpstats import init_workspace, open_collection
from dpstats.domains import BoxDomain
from dpstats.budgets import ApproxDPBudget
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)


STORAGE_ROOT = Path("benchmark-storage")


def main() -> None:
    collection = open_collection(init_workspace(STORAGE_ROOT), MeanEstimation)
    collection.add_algorithm(
        DemonstrationGaussianMechanism(label="delayed-gaussian"),
        registry="simulated",
    )
    collection.add_instance(
        MeanEstimationMetadata(
            n_samples=3,
            domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
        ),
        MeanEstimationInput(X=np.array([[-0.5], [0.0], [0.5]])),
        {"ground_truth": [0.0]},
        label="small-direct-input",
        registry="real",
    )
    for seed in [0]:
        collection.add_instance(
            *demonstration_instance(seed=seed, n_samples=8), registry="simulated"
        )

    print(
        collection.save_sweep(
            "simulated",
            {
                "budgets": [ApproxDPBudget(eps=1.0, delta=1e-5)],
                "seeds": [0, 1, 2],
                "time_limit": None,
                "memory_limit_gb": None,
            },
        )
    )


if __name__ == "__main__":
    main()
