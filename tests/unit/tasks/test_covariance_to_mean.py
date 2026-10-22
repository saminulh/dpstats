"""Public behavior of the joint-moment covariance Transformation."""

import math
from typing import Any

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
    DPBudget,
    PureDPBudget,
)
from dpstats.domains import EuclideanSpace, LpBallDomain
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.tasks import MeanEstimation
from dpstats.tasks.covariance_estimation import (
    CovarianceEstimationInput,
    CovarianceEstimationMetadata,
)
from dpstats.tasks.covariance_estimation.transformations import (
    CovarianceToMeanEstimation,
)
from dpstats.tasks.covariance_estimation.algorithms import SampleCovariance
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
    SampleMean,
)
from tests.unit.tasks.task_builders import box


def test_sample_mean_composition_recovers_centered_covariance():
    metadata = CovarianceEstimationMetadata(n_samples=3, domain=EuclideanSpace(dim=2))
    input = CovarianceEstimationInput(X=[[2.0, 3.0], [4.0, 7.0], [6.0, 5.0]])
    algorithm = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),), algorithm=SampleMean()
    )

    output = algorithm.run(metadata, input, NO_PRIVACY, seed=7)

    # Centered rows (-2,-2), (0,2), (2,0) give this ddof=0 covariance.
    np.testing.assert_allclose(output.covariance_est, [[8 / 3, 4 / 3], [4 / 3, 8 / 3]])
    assert algorithm.privacy_interpretation == PrivacyInterpretation.NONPRIVATE


def test_packing_is_isometric_and_uses_the_origin_bound_of_a_shifted_box():
    metadata = CovarianceEstimationMetadata(
        n_samples=2, domain=box((1, 2), (2, 3), (3, 4))
    )
    input = CovarianceEstimationInput(X=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    budget = ApproxDPBudget(eps=1.0, delta=1e-6)

    child_metadata, child_input, child_budget, child_seed, state = (
        CovarianceToMeanEstimation().prepare(metadata, input, budget, 19)
    )

    # Diagonal entries of the rank-one matrix precede (0,1), (0,2), (1,2).
    np.testing.assert_allclose(
        child_input.X[0],
        [1, 2, 3, 1, 4, 9, 2 * math.sqrt(2), 3 * math.sqrt(2), 6 * math.sqrt(2)],
    )
    # ||x||^2=14; ||xx^T||_F^2=196. The farthest box corner has ||x||^2=29.
    assert np.linalg.norm(child_input.X[0, 3:]) == pytest.approx(14)
    assert np.linalg.norm(child_input.X[0]) == pytest.approx(math.sqrt(210))
    assert isinstance(child_metadata.domain, LpBallDomain)
    assert child_metadata.domain.radius == pytest.approx(math.sqrt(870))
    np.testing.assert_array_equal(child_metadata.domain.center, np.zeros(9))
    child_metadata.validate_input(child_input)
    assert child_metadata.n_samples == 2
    assert child_budget is budget
    assert child_seed == 19
    assert state == 3


def test_deleting_a_source_row_deletes_only_its_joint_moment_row():
    input = CovarianceEstimationInput(X=[[1, 2], [3, -4], [-2, 1]])
    metadata = CovarianceEstimationMetadata(n_samples=3, domain=EuclideanSpace(dim=2))
    transformation = CovarianceToMeanEstimation()
    full = transformation.prepare(metadata, input, NO_PRIVACY, 0)[1]
    # The public reference n stays fixed on the add/remove closure.
    removed = transformation.prepare(
        metadata, CovarianceEstimationInput(X=input.X[[0, 2]]), NO_PRIVACY, 0
    )[1]
    np.testing.assert_array_equal(removed.X, full.X[[0, 2]])


@pytest.mark.parametrize("rows", [[[0.1, 0.3]], [[1.0], [4.0], [7.0]], [[0.0]]])
def test_sample_mean_matches_sample_covariance_at_small_dimensions(rows):
    input = CovarianceEstimationInput(X=rows)
    metadata = CovarianceEstimationMetadata(
        n_samples=len(rows), domain=EuclideanSpace(dim=len(rows[0]))
    )
    algorithm = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),), algorithm=SampleMean()
    )
    expected = SampleCovariance().run(metadata, input, NO_PRIVACY, 0)
    actual = algorithm.run(metadata, input, NO_PRIVACY, 0)
    np.testing.assert_allclose(
        actual.covariance_est, expected.covariance_est, atol=1e-14
    )


def test_gaussian_composition_matches_direct_child_with_same_budget_and_seed():
    metadata = CovarianceEstimationMetadata(n_samples=2, domain=box((1, 3), (2, 4)))
    input = CovarianceEstimationInput(X=[[1, 2], [3, 4]])
    budget = ApproxDPBudget(eps=1, delta=1e-6)
    child = GaussianMechanism()
    composition = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),), algorithm=child
    )

    actual = composition.run(metadata, input, budget, 31)
    # This is a delegation oracle, not a second Gaussian mechanism test.
    direct = child.run(
        MeanEstimationMetadata(
            n_samples=2,
            domain=LpBallDomain.at_origin(p=2, radius=math.sqrt(650), dim=5),
        ),
        MeanEstimationInput(
            X=[[1, 2, 1, 4, 2 * math.sqrt(2)], [3, 4, 9, 16, 12 * math.sqrt(2)]]
        ),
        budget,
        31,
    ).mean_est
    expected = [
        [direct[2] - direct[0] ** 2, direct[4] / math.sqrt(2) - direct[0] * direct[1]],
        [direct[4] / math.sqrt(2) - direct[0] * direct[1], direct[3] - direct[1] ** 2],
    ]
    np.testing.assert_allclose(actual.covariance_est, expected)
    assert composition.privacy_interpretation == PrivacyInterpretation.STANDARD


def test_restoration_retains_indefinite_output_without_debiasing():
    # m=(1,2), M=[[0,1],[1,0]] gives [[-1,-1],[-1,-4]].
    released = MeanEstimationOutput(mean_est=[1, 2, 0, 0, math.sqrt(2)])
    actual = CovarianceToMeanEstimation().restore(released, 2)
    np.testing.assert_allclose(actual.covariance_est, [[-1, -1], [-1, -4]])


class _RecordingMeanChild(BaseAlgorithm):
    TASK = MeanEstimation
    record: dict[str, Any]

    def run(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> MeanEstimationOutput:
        self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        self.record["calls"] = self.record.get("calls", 0) + 1
        raise self.record["exception"]


@pytest.mark.parametrize(
    "exception",
    [
        NoRelease("child bottom"),
        InvalidRegime("child regime"),
        RuntimeError("child failure"),
    ],
)
def test_child_outcomes_propagate_from_exactly_one_full_budget_call(exception):
    metadata = CovarianceEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    input = CovarianceEstimationInput(X=[[1]])
    budget = ApproxDPBudget(eps=1, delta=1e-6)
    child = _RecordingMeanChild(record={"exception": exception})
    with pytest.raises(type(exception)) as raised:
        CompositeAlgorithm(
            transformations=(CovarianceToMeanEstimation(),), algorithm=child
        ).run(metadata, input, budget, 73)
    assert raised.value is exception
    assert child.record["calls"] == 1
    assert child.record["budget"] is budget
    assert child.record["seed"] == 73


@pytest.mark.parametrize(
    "domain,budget",
    [
        (EuclideanSpace(dim=1), ApproxDPBudget(eps=1, delta=1e-6)),
        (
            LpBallDomain.at_origin(p=2, radius=1e200, dim=1),
            ApproxDPBudget(eps=1, delta=1e-6),
        ),
        (LpBallDomain.at_origin(p=2, radius=1, dim=1), PureDPBudget(eps=1)),
    ],
)
def test_child_restrictions_are_not_replaced_by_a_fallback(domain, budget):
    algorithm = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),),
        algorithm=GaussianMechanism(),
    )
    with pytest.raises(InvalidRegime):
        algorithm.run(
            CovarianceEstimationMetadata(n_samples=1, domain=domain),
            CovarianceEstimationInput(X=[[0.5]]),
            budget,
            0,
        )


def test_zero_radius_keeps_the_child_singleton_law():
    metadata = CovarianceEstimationMetadata(
        n_samples=2, domain=LpBallDomain.at_origin(p=2, radius=0, dim=2)
    )
    algorithm = CompositeAlgorithm(
        transformations=(CovarianceToMeanEstimation(),),
        algorithm=GaussianMechanism(),
    )
    output = algorithm.run(
        metadata, CovarianceEstimationInput(X=[[0, 0], [0, 0]]), PureDPBudget(eps=1), 0
    )
    np.testing.assert_array_equal(output.covariance_est, np.zeros((2, 2)))


@pytest.mark.parametrize(
    "domain", [EuclideanSpace(dim=1), LpBallDomain.at_origin(p=2, radius=1e200, dim=1)]
)
def test_infinite_joint_radius_is_delegated_to_the_mean_child(domain):
    metadata = CovarianceEstimationMetadata(n_samples=2, domain=domain)
    input = CovarianceEstimationInput(X=[[1.0], [3.0]])
    transformation = CovarianceToMeanEstimation()
    child_metadata, child_input, *_ = transformation.prepare(
        metadata, input, NO_PRIVACY, 0
    )
    assert child_metadata.domain == EuclideanSpace(dim=2)
    assert child_metadata.domain.l2_norm_bound == math.inf
    child_metadata.validate_input(child_input)

    output = CompositeAlgorithm(
        transformations=(transformation,), algorithm=SampleMean()
    ).run(metadata, input, NO_PRIVACY, 0)
    # The centered rows are -1 and 1, hence ddof=0 variance is 1.
    np.testing.assert_allclose(output.covariance_est, [[1.0]])


def test_nonfinite_joint_moment_is_an_execution_error():
    metadata = CovarianceEstimationMetadata(n_samples=1, domain=EuclideanSpace(dim=1))
    with pytest.raises(RuntimeError, match="Nonfinite joint moments"):
        CovarianceToMeanEstimation().prepare(
            metadata, CovarianceEstimationInput(X=[[1e200]]), NO_PRIVACY, 0
        )
