"""Clean-wheel smoke for importable downstream configuration and OBO seams."""

from __future__ import annotations

import json

import numpy as np

from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
from dpstats.budgets import NO_PRIVACY, ApproxDPBudget, ZCDPBudget
from dpstats.domains import BoxDomain
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression.algorithms import SSP
from dpstats.tasks.single_coefficient_estimation.transformations import (
    ToLinearRegression,
    ToZCDP,
)

from external_configured_components import (
    ExternalAlgorithm,
    ExternalExhaustiveTuner,
    external_instance,
    ExternalTransformation,
    external_selection_cost,
)


def main() -> None:
    """Exercise portable configuration, tuner, and OBO imports."""
    metadata, input_data, _ = external_instance(seed=7)

    composition = CompositeAlgorithm(
        transformations=(ExternalTransformation(),),
        algorithm=ExternalAlgorithm(scale=2.0),
    )
    composition_record = composition.dump_config()
    restored = BaseAlgorithm.load_record(composition_record)
    assert isinstance(restored, CompositeAlgorithm)
    output = restored.run(
        metadata=metadata,
        input=input_data,
        budget=NO_PRIVACY,
        seed=9,
    )
    assert output.estimate.shape == (2,)

    sce = SingleCoefficientEstimation
    metadata = sce.Metadata(
        n_samples=4,
        coefficient_of_interest_index=0,
        design_domain=BoxDomain(bounds=np.array([[-1, 1], [-1, 1]])),
        response_domain=BoxDomain(bounds=np.array([[-1, 1]])),
    )
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    input_data = sce.Input(X=X, y=0.5 * X[:, 0])
    adapted = CompositeAlgorithm(
        transformations=(ToZCDP(), ToLinearRegression()), algorithm=SSP()
    )
    restored_adapter = BaseAlgorithm.load_record(adapted.dump_config())
    assert isinstance(restored_adapter, CompositeAlgorithm)
    offer = ApproxDPBudget(eps=2.0, delta=1e-6)
    prepared = restored_adapter.process(metadata, input_data, offer, 9)
    assert isinstance(prepared[2], ZCDPBudget)
    assert restored_adapter.task is sce
    assert np.isfinite(restored_adapter.run(metadata, input_data, offer, 9).coefficient)

    tuner = ExternalExhaustiveTuner(
        alg_class=ExternalAlgorithm,
        alg_hyperparam_grid={"scale": [1.0, 2.0]},
        selection_cost=external_selection_cost,
    )
    restored_tuner = BaseAlgorithm.load_record(tuner.dump_config())
    assert isinstance(restored_tuner, ExternalExhaustiveTuner)

    print(
        json.dumps(
            {
                "algorithm": type(restored.algorithm).__name__,
                "transformation": type(restored.transformations[0]).__name__,
                "tuner": type(restored_tuner).__name__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
