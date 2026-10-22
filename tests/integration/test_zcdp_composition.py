"""Conservative NoRelease outcomes after zCDP budget conversion."""

import numpy as np
import pytest

from dpstats import init_workspace, open_collection
from dpstats.algorithm import CompositeAlgorithm
from dpstats.budgets import ApproxDPBudget
from dpstats.exceptions import NoRelease
from dpstats.results import ResultStatus
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression.algorithms import SSP
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    DPWhitening,
    ToLinearRegression,
    ToZCDP,
)
from tests.unit.tasks.task_builders import box, sce_metadata


@pytest.mark.parametrize(
    "preparation",
    [
        (ToZCDP(),),
        (DPWhitening(), ToZCDP()),
        (ToZCDP(), DPWhitening()),
    ],
)
def test_direct_and_collected_refusals_keep_the_conservative_composition_outcome(
    tmp_path, preparation
):
    metadata = sce_metadata(4, 0, box((-1, 1), (-1, 1)))
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    input_data = SingleCoefficientEstimationInput(X=X, y=0.5 * X[:, 0])
    composition = CompositeAlgorithm(
        transformations=(*preparation, ToLinearRegression()), algorithm=SSP()
    )
    budget = ApproxDPBudget(eps=2.0, delta=1e-6)

    # SSP needs a bounded response domain; whitening changes only the design.
    with pytest.raises(NoRelease):
        composition.run(metadata, input_data, budget, 17)

    workspace = init_workspace(tmp_path / "workspace")
    collection = open_collection(workspace, SingleCoefficientEstimation)
    collection.add_instance(metadata, input_data, {})
    collection.add_algorithm(composition)
    sweep = collection.save_sweep("conversion", {"budgets": [budget], "seeds": [17]})
    collection.run_sweep(sweep)
    (result,) = collection.load_results(sweep).to_dict("records")
    assert result["status"] is ResultStatus.NO_RELEASE
