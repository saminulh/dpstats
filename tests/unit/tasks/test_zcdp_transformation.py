"""SCE budget conversion and its placement in a private composition."""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
    accounting,
)
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression.algorithms import OLS, SSP
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput,
    SingleCoefficientEstimationOutput,
)
from dpstats.tasks.single_coefficient_estimation.algorithms import (
    ZeroCoefficient,
)
from dpstats.tasks.single_coefficient_estimation.transformations import (
    DPWhitening,
    ToLinearRegression,
    ToZCDP,
)
from tests.registry_components import recording_transformation
from tests.unit.tasks.task_builders import Unreadable, box, sce_metadata


@pytest.mark.parametrize(
    "offered",
    [
        ApproxDPBudget(eps=2.0, delta=1e-6),
        ZCDPBudget(rho=0.25),
        NO_PRIVACY,
        PureDPBudget(eps=math.inf),
        GDPBudget(mu=math.inf),
    ],
)
def test_only_the_budget_changes_and_conversion_reads_no_input(offered):
    metadata, input_data = Unreadable("metadata"), Unreadable("input")
    transformation = ToZCDP()
    prepared = transformation.prepare(metadata, input_data, offered, 17)

    # Delegation oracle: the accountant, already tested independently, owns
    # which zCDP spend each offered family admits.
    assert prepared[0] is metadata
    assert prepared[1] is input_data
    assert prepared[2] == accounting.spend_as_zcdp(offered)
    assert prepared[3] == 17
    assert prepared[4] is None
    output = SingleCoefficientEstimationOutput(coefficient=0.5)
    assert transformation.restore(output, prepared[4]) is output


@pytest.mark.parametrize("budget", [PureDPBudget(eps=2.0), GDPBudget(mu=1.0)])
def test_an_unsupported_offer_is_refused_without_reading_the_problem(budget):
    with pytest.raises(InvalidRegime):
        ToZCDP().prepare(Unreadable("metadata"), Unreadable("input"), budget, 17)


def test_the_sce_chain_round_trips_and_runs_through_a_private_child():
    metadata = sce_metadata(4, 0, box((-1, 1), (-1, 1)), box((-1, 1)))
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    input_data = SingleCoefficientEstimationInput(X=X, y=0.5 * X[:, 0])
    composition = CompositeAlgorithm(
        transformations=(ToZCDP(), ToLinearRegression()), algorithm=SSP()
    )
    restored = BaseAlgorithm.load_record(composition.dump_config())

    assert restored.task is SingleCoefficientEstimation
    assert restored.uuid == composition.uuid
    budget = ZCDPBudget(rho=0.5)
    output = restored.run(metadata, input_data, budget, 17)
    direct = CompositeAlgorithm(
        transformations=(ToLinearRegression(),), algorithm=SSP()
    ).run(metadata, input_data, budget, 17)
    assert output.coefficient == direct.coefficient

    nonprivate = CompositeAlgorithm(
        transformations=(ToZCDP(), ToLinearRegression()), algorithm=OLS()
    )
    assert nonprivate.privacy_interpretation is PrivacyInterpretation.NONPRIVATE


def test_conversion_follows_outer_spending_and_precedes_private_splits(monkeypatch):
    metadata = sce_metadata(4, 0, box((-1, 1), (-1, 1)))
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    input_data = SingleCoefficientEstimationInput(X=X, y=0.5 * X[:, 0])
    remainder = ApproxDPBudget(eps=2.0, delta=5e-6)
    outer = recording_transformation(
        SingleCoefficientEstimation,
        SingleCoefficientEstimation,
        child_budget=remainder,
    )()
    convert = Mock(return_value=ZCDPBudget(rho=8.0))
    output = SingleCoefficientEstimationOutput(coefficient=0.0)
    run = Mock(return_value=output)
    monkeypatch.setattr(accounting, "spend_as_zcdp", convert)
    monkeypatch.setattr(ZeroCoefficient, "run", run)
    composition = CompositeAlgorithm(
        transformations=(outer, ToZCDP(), DPWhitening(preparation_share=0.25)),
        algorithm=ZeroCoefficient(),
    )

    result = composition.run(
        metadata, input_data, budget=ApproxDPBudget(eps=4.0, delta=1e-5), seed=17
    )

    convert.assert_called_once_with(remainder)
    assert run.call_args.kwargs["budget"] == ZCDPBudget(rho=6.0)
    assert result is output


def test_composition_conservatively_reports_a_stage_refusal_after_conversion():
    composition = CompositeAlgorithm(
        transformations=(ToZCDP(), DPWhitening()), algorithm=ZeroCoefficient()
    )
    metadata = sce_metadata(4, 0, 2)
    input_data = SingleCoefficientEstimationInput(X=np.zeros((4, 2)), y=np.zeros(4))

    with pytest.raises(NoRelease):
        composition.run(metadata, input_data, ApproxDPBudget(eps=2.0, delta=1e-6), 17)
