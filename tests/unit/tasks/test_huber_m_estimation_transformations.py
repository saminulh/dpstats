"""Tests for Huber transformations and loss-Lipschitz DP-SGD."""

from collections.abc import Callable
from functools import partial
import math
import pickle
from typing import Any

import numpy as np
import pytest
from pydantic import Field

from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    LpBallDomain,
)
from dpstats.algorithm import BaseAlgorithm, CompositeAlgorithm
from dpstats.algorithm import PrivacyInterpretation, Transformation
from dpstats.budgets import NO_PRIVACY, DPBudget, ZCDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.task import Array1D, ArrayND, TaskInput, TaskMetadata
from dpstats.tasks.linear_regression.transformations import Standardize
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    AbsoluteResidualLoss,
    HuberResidualLoss,
    ToMEstimation as LinearRegressionToMEstimation,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from dpstats.tasks.m_estimation.algorithms.dp_gradient_descent import (
    DPSGD,
)
from dpstats.tasks.m_estimation.algorithms.erm import (
    ERM,
)
from dpstats.tasks.m_estimation.algorithms.gradient_descent import (
    GD,
)
from dpstats.tasks.m_estimation.losses import (
    LossProperties,
    MEstimationLoss,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    HuberNormLoss,
    SquaredErrorLoss,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks import MEstimation
from dpstats.tasks.mean_estimation import (
    transformations as mean_transformations,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)
from dpstats.tasks.median_estimation import (
    transformations as median_transformations,
)
from dpstats.tasks.median_estimation import (
    MedianEstimationInput,
    MedianEstimationMetadata,
)


class TestHuberResidualLossAndGradient:
    """Hand-computed values, signs, and the quadratic/linear boundary."""

    def test_loss_matches_hand_computed_values_across_the_boundary(self):
        loss = HuberResidualLoss(threshold=2.0)
        data = np.array([[1.0, 0.0, 3.0], [1.0, 0.0, 1.5], [1.0, 0.0, 2.0]])
        theta = np.zeros(2)  # residual r = 3

        np.testing.assert_allclose(loss(data, theta), [4.0, 1.125, 2.0])

    def test_gradient_is_minus_x_times_clipped_residual(self):
        loss = HuberResidualLoss(threshold=1.0)
        theta = np.zeros(2)

        np.testing.assert_allclose(
            loss.gradient(
                np.array([[2.0, -1.0, 0.5], [2.0, -1.0, 7.0], [2.0, -1.0, -7.0]]),
                theta,
            ),
            [[-1.0, 0.5], [-2.0, 1.0], [2.0, -1.0]],
        )


class TestHuberNormLossAndGradient:
    """Hand-computed values across the quadratic/linear boundary at d = 4."""

    # d = 4, so the effective threshold is twice threshold_per_sqrt_dim.
    z = np.array([[3.0, 4.0, 0.0, 0.0]])  # deviation norm 5 at theta = 0
    theta = np.zeros(4)

    def test_loss_matches_hand_computed_values_across_the_boundary(self):
        # Thresholds 10 and 5 give 5**2 / 2; threshold 1 gives 1 * 5 - 1 / 2.
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=5.0)(self.z, self.theta), [12.5]
        )
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=2.5)(self.z, self.theta), [12.5]
        )
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=0.5)(self.z, self.theta), [4.5]
        )

    def test_gradient_is_deviation_scaled_to_the_threshold(self):
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=5.0).gradient(self.z, self.theta),
            [[-3.0, -4.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=0.5).gradient(self.z, self.theta),
            [[-0.6, -0.8, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            HuberNormLoss(threshold_per_sqrt_dim=0.5).gradient(
                self.theta[None, :], self.theta
            ),
            [[0.0, 0.0, 0.0, 0.0]],
        )

    def test_the_threshold_grows_with_the_square_root_of_the_dimension(self):
        loss = HuberNormLoss(threshold_per_sqrt_dim=1.0)

        # A deviation of norm 2 is past the threshold 1 at d = 1 and exactly at
        # the threshold 2 at d = 4.
        np.testing.assert_allclose(loss(np.array([[2.0]]), np.zeros(1)), [1.5])
        np.testing.assert_allclose(
            loss(np.array([[2.0, 0.0, 0.0, 0.0]]), np.zeros(4)), [2.0]
        )

    def test_the_declared_lipschitz_constant_grows_with_the_square_root_of_the_dimension(
        self,
    ):
        loss = HuberNormLoss(threshold_per_sqrt_dim=1.5)

        constants = [
            loss.properties(
                BoxDomain(bounds=np.array([[-1.0, 1.0]] * dim)), EuclideanSpace(dim=dim)
            ).loss_lipschitz_constant
            for dim in (1, 4)
        ]

        assert constants == pytest.approx([1.5, 3.0])


def _lr_instance(n: int = 60, response_half_range: float = 50.0, outlier: bool = True):
    """Planted line inside declared bounds, optionally with one gross outlier."""
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    beta_true = np.array([2.0, -1.0])
    y = X @ beta_true + rng.normal(0.0, 0.05, size=n)
    if outlier:
        y[0] = 40.0
    metadata = LinearRegressionMetadata(
        n_samples=n,
        design_domain=BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]])),
        response_domain=BoxDomain(
            bounds=np.array([[-response_half_range, response_half_range]])
        ),
    )
    return metadata, LinearRegressionInput(X=X, y=y), beta_true


class TestHuberResidualLossThreshold:
    """Outlier damping, the carried loss, and the declared Lipschitz constant."""

    def test_outlier_moves_the_robust_fit_less_than_least_squares(self):
        metadata, task_input, beta_true = _lr_instance()

        robust = CompositeAlgorithm(
            transformations=(
                LinearRegressionToMEstimation(loss=HuberResidualLoss(threshold=0.5)),
            ),
            algorithm=GD(num_steps=400),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)
        least_squares = CompositeAlgorithm(
            transformations=(LinearRegressionToMEstimation(),),
            algorithm=GD(num_steps=400),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        robust_error = float(np.linalg.norm(robust.beta - beta_true))
        least_squares_error = float(np.linalg.norm(least_squares.beta - beta_true))
        assert robust_error < least_squares_error
        assert robust_error < 0.2

    def test_generic_transformation_carries_the_selected_configured_loss(self):
        metadata, task_input, _ = _lr_instance(outlier=False)
        loss = AbsoluteResidualLoss()

        prepared_metadata, prepared_input, _, _, _ = LinearRegressionToMEstimation(
            loss=loss
        ).prepare(metadata, task_input, budget=NO_PRIVACY, seed=42)

        assert prepared_metadata.loss is loss
        np.testing.assert_array_equal(
            prepared_input.data, np.column_stack((task_input.X, task_input.y))
        )

        restored = Transformation.load_record(
            LinearRegressionToMEstimation(loss=loss).dump_config()
        )
        assert type(restored) is LinearRegressionToMEstimation
        assert restored.loss == loss

    def test_declares_the_exact_loss_lipschitz_constant(self):
        metadata, task_input, _ = _lr_instance(outlier=False)
        transformation = LinearRegressionToMEstimation(
            loss=HuberResidualLoss(threshold=0.1)
        )
        prepared_metadata, _, _, _, _ = transformation.prepare(
            metadata, task_input, budget=NO_PRIVACY, seed=42
        )
        assert prepared_metadata.loss_properties.loss_lipschitz_constant == (
            pytest.approx(0.1 * math.sqrt(2))
        )

    def test_response_rescaling_rescales_the_release_after_standardizing(self):
        """No radius is declared: equivariance is a claim about the fit."""
        metadata, task_input, _ = _lr_instance(outlier=False)
        s = 3.0
        scaled_metadata = LinearRegressionMetadata(
            n_samples=metadata.n_samples,
            design_domain=metadata.design_domain,
            response_domain=BoxDomain(bounds=s * np.array([[-50.0, 50.0]])),
        )
        scaled_input = LinearRegressionInput(X=task_input.X, y=s * task_input.y)
        composition = CompositeAlgorithm(
            transformations=(
                Standardize(),
                LinearRegressionToMEstimation(loss=HuberResidualLoss(threshold=0.1)),
            ),
            algorithm=DPSGD(C_grad_clip="loss_lipschitz"),
        )

        base = composition.run(metadata, task_input, budget=ZCDPBudget(rho=0.2), seed=5)
        scaled = composition.run(
            scaled_metadata, scaled_input, budget=ZCDPBudget(rho=0.2), seed=5
        )

        np.testing.assert_allclose(scaled.beta, s * base.beta, rtol=1e-10)


class _LipschitzTestLoss(MEstimationLoss):
    """Configured linear loss used to expose DP-gradient clipping behavior."""

    lipschitz_constant: float = Field(ge=0)

    def __call__(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> float | Array1D:
        del theta
        values = np.zeros(len(np.atleast_2d(data)))
        if data.ndim == 1:
            return float(values[0])
        if reduction == "none":
            return values
        return float(np.sum(values) if reduction == "sum" else np.mean(values))

    def gradient(
        self, data: ArrayND, theta: Array1D, *, reduction: str = "none"
    ) -> ArrayND:
        del theta
        if self.lipschitz_constant == 0:
            if data.ndim == 1 or reduction != "none":
                return np.zeros(data.shape[-1])
            return np.zeros_like(data)
        if data.ndim == 1 or reduction == "none":
            return np.atleast_2d(data) if data.ndim == 2 else data
        gradient_sum = np.sum(data, axis=0)
        return gradient_sum if reduction == "sum" else gradient_sum / len(data)

    def properties(
        self, data_domain: Domain, parameter_domain: Domain
    ) -> LossProperties:
        del data_domain, parameter_domain
        return LossProperties(
            convex=True,
            generalized_linear=True,
            continuously_twice_differentiable=True,
            loss_lipschitz_constant=self.lipschitz_constant,
            gradient_lipschitz_constant=0.0,
        )


def _lipschitz_problem(
    loss_lipschitz_constant: float,
    *,
    parameter_domain=None,
    data=None,
):
    domain = parameter_domain or EuclideanSpace(dim=4)
    records = np.ones((3, domain.dim)) if data is None else data
    return MEstimationMetadata(
        parameter_domain=domain,
        n_samples=3,
        loss=_LipschitzTestLoss(lipschitz_constant=loss_lipschitz_constant),
        data_domain=EuclideanSpace(dim=domain.dim),
    ), MEstimationInput(data=records)


class TestConfiguredLossLipschitzContract:
    def test_metadata_pickle_round_trip_preserves_the_configured_loss(self):
        metadata, _ = _lipschitz_problem(math.inf)

        restored = pickle.loads(pickle.dumps(metadata))

        assert restored.loss == metadata.loss


class TestLossLipschitzClip:
    @pytest.mark.parametrize("subsample_rate", [1.0, 0.5])
    @pytest.mark.parametrize("iterate_aggregation", ["last", "tail_average", "polyak"])
    def test_loss_lipschitz_matches_the_numeric_clip_at_the_same_norm(
        self, subsample_rate, iterate_aggregation
    ):
        # L=3 in d=4 equals the generic C=1.5 policy's clip 1.5*sqrt(4)=3.
        metadata, input_data = _lipschitz_problem(3.0)
        common = dict(
            eta=0.07,
            num_steps=4,
            subsample_rate=subsample_rate,
            iterate_aggregation=iterate_aggregation,
        )
        budget = ZCDPBudget(rho=0.4)

        lipschitz = DPSGD(C_grad_clip="loss_lipschitz", **common).run(
            metadata, input_data, budget=budget, seed=11
        )
        generic = DPSGD(C_grad_clip=1.5, **common).run(
            metadata, input_data, budget=budget, seed=11
        )

        np.testing.assert_array_equal(lipschitz.theta, generic.theta)

    def test_the_option_survives_a_configuration_round_trip(self):
        algorithm = DPSGD(C_grad_clip="loss_lipschitz", num_steps=7)

        restored = BaseAlgorithm.load_record(algorithm.dump_config())

        assert restored == algorithm
        assert restored.C_grad_clip == "loss_lipschitz"

    def test_unknown_constant_is_refused_before_input_access(self):
        metadata, _ = _lipschitz_problem(math.inf)

        with pytest.raises(InvalidRegime, match="loss Lipschitz constant"):
            DPSGD(C_grad_clip="loss_lipschitz", num_steps=1).run(
                metadata, object(), budget=ZCDPBudget(rho=0.4), seed=0
            )

    def test_unbounded_parameter_domain_leaves_the_squared_loss_incompatible(self):
        metadata = MEstimationMetadata(
            n_samples=3,
            parameter_domain=EuclideanSpace(dim=1),
            data_domain=BoxDomain(bounds=np.array([[-2.0, 2.0]])),
            loss=SquaredErrorLoss(),
        )

        assert math.isinf(metadata.loss_properties.loss_lipschitz_constant)

        with pytest.raises(InvalidRegime, match="loss Lipschitz constant"):
            DPSGD(C_grad_clip="loss_lipschitz", num_steps=1).run(
                metadata, object(), budget=ZCDPBudget(rho=0.4), seed=0
            )

    def test_the_declared_data_domain_makes_that_constant_finite(self):
        mean_metadata = MeanEstimationMetadata(
            n_samples=3,
            domain=BoxDomain(bounds=np.array([[-2.0, 2.0]])),
        )
        mean_input = MeanEstimationInput(X=np.array([[-1.0], [0.0], [1.0]]))

        (
            prepared_metadata,
            prepared_input,
            _,
            _,
            _,
        ) = mean_transformations.ToMEstimation().prepare(
            mean_metadata, mean_input, budget=NO_PRIVACY, seed=0
        )

        # The parameter domain is the data box, so the widest deviation on the
        # one coordinate is |-2 - 2| = 4, and the squared-error loss is
        # 2-Lipschitz per unit of deviation.
        assert prepared_metadata.loss_properties.loss_lipschitz_constant == (
            pytest.approx(8.0)
        )
        output = DPSGD(C_grad_clip="loss_lipschitz", num_steps=1).run(
            prepared_metadata, prepared_input, budget=ZCDPBudget(rho=0.4), seed=0
        )
        assert np.all(np.isfinite(output.theta))

    @pytest.mark.parametrize("budget", [NO_PRIVACY, ZCDPBudget(rho=0.4)])
    @pytest.mark.parametrize("iterate_aggregation", ["last", "tail_average", "polyak"])
    def test_zero_constant_returns_data_independent_projected_initialization(
        self, budget, iterate_aggregation
    ):
        domain = BoxDomain(bounds=np.array([[1.0, 2.0], [-3.0, -2.0]]))
        first = np.ones((3, 2))
        second = np.full((3, 2), 1e6)
        metadata, first_input = _lipschitz_problem(
            0.0, parameter_domain=domain, data=first
        )
        _, second_input = _lipschitz_problem(0.0, parameter_domain=domain, data=second)
        algorithm = DPSGD(
            C_grad_clip="loss_lipschitz",
            num_steps=4,
            subsample_rate=0.5,
            iterate_aggregation=iterate_aggregation,
        )

        first_output = algorithm.run(metadata, first_input, budget=budget, seed=7)
        second_output = algorithm.run(metadata, second_input, budget=budget, seed=7)
        expected = domain.project(np.zeros(2))

        np.testing.assert_array_equal(first_output.theta, expected)
        np.testing.assert_array_equal(second_output.theta, expected)


def _mean_instance(n: int = 41):
    """Symmetric cloud around a known center plus one gross outlier."""
    rng = np.random.default_rng(9)
    center = np.array([1.0, -2.0])
    X = center + rng.normal(0.0, 0.1, size=(n, 2))
    X[0] = np.array([90.0, 90.0])
    metadata = MeanEstimationMetadata(
        n_samples=n,
        domain=BoxDomain(bounds=np.array([[-100.0, 100.0], [-100.0, 100.0]])),
    )
    return metadata, MeanEstimationInput(X=X), center


class TestHuberNormLossThreshold:
    """Outlier damping and the declared Lipschitz constant through the public composition."""

    def test_declares_the_exact_loss_lipschitz_constant(self):
        metadata, task_input, _ = _mean_instance()

        prepared_metadata, _, _, _, _ = mean_transformations.ToMEstimation(
            loss=HuberNormLoss(threshold_per_sqrt_dim=0.5)
        ).prepare(metadata, task_input, budget=NO_PRIVACY, seed=0)

        assert prepared_metadata.loss_properties.loss_lipschitz_constant == (
            pytest.approx(0.5 * math.sqrt(2))
        )

    def test_outlier_moves_the_robust_estimate_less_than_the_mean(self):
        metadata, task_input, center = _mean_instance()

        robust = CompositeAlgorithm(
            transformations=(
                mean_transformations.ToMEstimation(
                    loss=HuberNormLoss(threshold_per_sqrt_dim=1.0)
                ),
            ),
            algorithm=GD(num_steps=400),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)
        squared = CompositeAlgorithm(
            transformations=(mean_transformations.ToMEstimation(),),
            algorithm=GD(num_steps=400),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        robust_error = float(np.linalg.norm(robust.mean_est - center))
        squared_error = float(np.linalg.norm(squared.mean_est - center))
        assert robust_error < squared_error
        assert robust_error < 0.2


def _recording_algorithm(record: dict[str, Any]) -> type[BaseAlgorithm]:
    """Build an M-estimation algorithm that records the problem it is offered."""

    class RecordingAlgorithm(BaseAlgorithm):
        TASK = MEstimation

        @property
        def privacy_interpretation(self) -> PrivacyInterpretation:
            return PrivacyInterpretation.NONPRIVATE

        def run(
            self,
            metadata: MEstimationMetadata,
            input: MEstimationInput,
            budget: DPBudget,
            seed: int,
        ) -> MEstimationOutput:
            record["metadata"] = metadata
            return MEstimationOutput(theta=np.zeros(metadata.parameter_domain.dim))

    return RecordingAlgorithm


def _problem_prepared_by(
    transformation: Transformation, metadata: TaskMetadata, task_input: TaskInput
) -> MEstimationMetadata:
    """Return the M-estimation problem one public composition run prepares."""
    record: dict[str, Any] = {}
    CompositeAlgorithm(
        transformations=(transformation,),
        algorithm=_recording_algorithm(record)(),
    ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)
    return record["metadata"]


def _assert_origin_ball(domain: Domain, radius: float, dim: int) -> None:
    """Assert a domain is the L2 ball of that radius centered at the origin."""
    assert isinstance(domain, LpBallDomain)
    assert domain.p == 2
    assert domain.radius == radius
    assert domain.dim == dim
    np.testing.assert_array_equal(domain.center, np.zeros(dim))


_LossBearingTransformation = (
    LinearRegressionToMEstimation
    | mean_transformations.ToMEstimation
    | median_transformations.ToMEstimation
)


def _unit_scale_regression():
    """A one-column regression on |x| <= 1, |y| <= 3, inside both bounds."""
    metadata = LinearRegressionMetadata(
        n_samples=3,
        design_domain=BoxDomain(bounds=np.array([[-1.0, 1.0]])),
        response_domain=BoxDomain(bounds=np.array([[-3.0, 3.0]])),
    )
    task_input = LinearRegressionInput(
        X=np.array([[0.5], [-0.5], [0.0]]), y=np.array([1.0, -1.0, 0.0])
    )
    return metadata, task_input


def _unit_scale_location():
    """A one-coordinate location problem on |z| <= 1."""
    metadata = MeanEstimationMetadata(
        n_samples=3, domain=BoxDomain(bounds=np.array([[-1.0, 1.0]]))
    )
    return metadata, MeanEstimationInput(X=np.array([[0.5], [-0.5], [0.0]]))


def _unit_scale_median():
    """A scalar median problem on |z| <= 2."""
    metadata = MedianEstimationMetadata(
        n_samples=3, domain=BoxDomain(bounds=np.array([[-2.0, 2.0]]))
    )
    return metadata, MedianEstimationInput(X=np.array([-1.0, 0.0, 1.0]))


class TestDeclaredParameterBall:
    """Every loss-bearing transformation publishes the parameter domain it declares.

    Each case reads the prepared problem through one public composition run.
    The fixtures are sized so the loss facts the ball unlocks are whole
    numbers: a design row of norm at most 1, a response of magnitude at most 3,
    and a radius-2 ball, so a residual reaches 3 + 2 = 5 and a coordinate
    deviation reaches 1 + 2 = 3.
    """

    @pytest.mark.parametrize(
        ("transformation", "build_problem", "lipschitz", "smoothness"),
        [
            # d(r**2)/dtheta = -2*r*x: widest slope 2 * 5 * 1 = 10, and the
            # curvature is the constant 2*x**2 <= 2.
            pytest.param(
                LinearRegressionToMEstimation(parameter_radius=2.0),
                _unit_scale_regression,
                10.0,
                2.0,
                id="lr-squared-residual",
            ),
            # d|r|/dtheta = -sign(r)*x, of magnitude at most 1; the kink at
            # r = 0 leaves no finite curvature.
            pytest.param(
                LinearRegressionToMEstimation(
                    loss=AbsoluteResidualLoss(), parameter_radius=2.0
                ),
                _unit_scale_regression,
                1.0,
                math.inf,
                id="lr-absolute-residual",
            ),
            # The Huber slope saturates at the threshold 1.5 times the design
            # norm bound 1, and the curvature is x**2 <= 1.
            pytest.param(
                LinearRegressionToMEstimation(
                    loss=HuberResidualLoss(threshold=1.5),
                    parameter_radius=2.0,
                ),
                _unit_scale_regression,
                1.5,
                1.0,
                id="lr-huber-residual",
            ),
        ],
    )
    def test_the_ball_unlocks_the_hand_derived_loss_constants(
        self,
        transformation: Transformation,
        build_problem: Callable[[], tuple[TaskMetadata, TaskInput]],
        lipschitz: float,
        smoothness: float,
    ) -> None:
        metadata, task_input = build_problem()

        prepared = _problem_prepared_by(transformation, metadata, task_input)

        _assert_origin_ball(prepared.parameter_domain, 2.0, 1)
        properties = prepared.loss_properties
        assert properties.loss_lipschitz_constant == pytest.approx(lipschitz)
        assert properties.gradient_lipschitz_constant == pytest.approx(smoothness)

    @pytest.mark.parametrize(
        "transformation, lipschitz, smoothness",
        [
            # d((z-theta)**2)/dtheta = 2*(theta-z), of magnitude at most
            # 2 * (1 + 1) = 4 on the box |z|, |theta| <= 1, and the curvature
            # is the constant 2.
            pytest.param(mean_transformations.ToMEstimation(), 4.0, 2.0, id="squared"),
            # The threshold is 0.5 at d = 1; the Huber slope saturates there and
            # its curvature is 1 below the threshold.
            pytest.param(
                mean_transformations.ToMEstimation(
                    loss=HuberNormLoss(threshold_per_sqrt_dim=0.5)
                ),
                0.5,
                1.0,
                id="huber-norm",
            ),
        ],
    )
    def test_the_mean_parameter_domain_is_its_declared_domains_enclosure(
        self, transformation: Transformation, lipschitz: float, smoothness: float
    ):
        """The mean lies in the convex data domain, which is the parameter domain."""
        metadata, task_input = _unit_scale_location()

        prepared = _problem_prepared_by(transformation, metadata, task_input)

        assert prepared.parameter_domain == metadata.domain
        properties = prepared.loss_properties
        assert properties.loss_lipschitz_constant == pytest.approx(lipschitz)
        assert properties.gradient_lipschitz_constant == pytest.approx(smoothness)

    def test_the_median_parameter_domain_is_its_declared_domain(self):
        """The median lies in the declared interval, which is the parameter domain."""
        metadata, task_input = _unit_scale_median()

        prepared = _problem_prepared_by(
            median_transformations.ToMEstimation(), metadata, task_input
        )

        assert prepared.parameter_domain == metadata.domain
        # d|z-theta|/dtheta = sign(theta-z) on the one coordinate, so the slope
        # is 1 and the kink leaves no finite curvature.
        properties = prepared.loss_properties
        assert properties.loss_lipschitz_constant == pytest.approx(1.0)
        assert properties.gradient_lipschitz_constant == math.inf

    def test_the_radius_is_part_of_the_configured_identity(self):
        radii = (None, 1.0, 4.0)
        compositions = [
            CompositeAlgorithm(
                transformations=(
                    LinearRegressionToMEstimation(parameter_radius=radius),
                ),
                algorithm=GD(),
            )
            for radius in radii
        ]

        assert len({composition.uuid for composition in compositions}) == len(radii)
        for radius, composition in zip(radii, compositions):
            restored = Transformation.load_record(
                composition.transformations[0].dump_config()
            )
            assert type(restored) is LinearRegressionToMEstimation
            assert restored.parameter_radius == radius

    def test_the_ball_spans_every_parameter_coordinate(self):
        metadata, task_input, _ = _lr_instance(outlier=False)

        prepared = _problem_prepared_by(
            LinearRegressionToMEstimation(parameter_radius=2.0), metadata, task_input
        )

        _assert_origin_ball(prepared.parameter_domain, 2.0, 2)

    def test_an_unconstrained_algorithm_refuses_the_declared_ball(self):
        metadata, task_input = _unit_scale_regression()

        with pytest.raises(InvalidRegime, match="unconstrained optimization only"):
            CompositeAlgorithm(
                transformations=(LinearRegressionToMEstimation(parameter_radius=2.0),),
                algorithm=ERM(),
            ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)


class TestUnboundedParameterDomain:
    """``parameter_radius=None`` declares no ball, so no ball is prepared.

    ``None`` is the default and the configured value naming the unbounded
    parameter domain, the geometry an unconstrained algorithm requires, so a
    composition reaches such an algorithm instead of being refused by its own
    declaration. Declaring no ball is also what leaves a squared loss no finite
    Lipschitz constant, so an algorithm calibrating from one refuses.
    """

    @pytest.mark.parametrize(
        "radius_kwargs", [{}, {"parameter_radius": None}], ids=["omitted", "explicit"]
    )
    @pytest.mark.parametrize(
        "build, build_problem",
        [
            (LinearRegressionToMEstimation, _unit_scale_regression),
            (
                partial(
                    LinearRegressionToMEstimation,
                    loss=HuberResidualLoss(threshold=0.5),
                ),
                _unit_scale_regression,
            ),
        ],
    )
    def test_every_transformation_declares_euclidean_space(
        self,
        build: Callable[..., _LossBearingTransformation],
        build_problem: Callable[[], tuple[TaskMetadata, TaskInput]],
        radius_kwargs: dict[str, None],
    ) -> None:
        """Omitting the radius and naming ``None`` are the same declaration.

        Nothing in a transformation needs a finite parameter diameter, so the
        unconfigured estimand is already the unconstrained minimizer.
        """
        transformation = build(**radius_kwargs)
        metadata, task_input = build_problem()

        assert transformation.parameter_radius is None
        prepared = _problem_prepared_by(transformation, metadata, task_input)
        assert isinstance(prepared.parameter_domain, EuclideanSpace)
        assert prepared.parameter_domain.dim == 1

    def test_a_diameter_calibrated_algorithm_refuses_the_default(self):
        """The squared loss has no finite Lipschitz constant without a ball."""
        metadata, task_input = _unit_scale_regression()

        with pytest.raises(InvalidRegime, match="loss Lipschitz constant"):
            CompositeAlgorithm(
                transformations=(LinearRegressionToMEstimation(),),
                algorithm=DPSGD(C_grad_clip="loss_lipschitz", num_steps=1),
            ).run(metadata, task_input, budget=ZCDPBudget(rho=0.4), seed=0)

    def test_an_unconstrained_algorithm_accepts_the_composition(self):
        """The exactly fitted line is recovered where the ball is refused."""
        metadata = LinearRegressionMetadata(
            n_samples=3,
            design_domain=BoxDomain(bounds=np.array([[-4.0, 4.0]])),
            response_domain=BoxDomain(bounds=np.array([[-8.0, 8.0]])),
        )
        task_input = LinearRegressionInput(
            X=np.array([[1.0], [2.0], [3.0]]), y=np.array([2.0, 4.0, 6.0])
        )

        output = CompositeAlgorithm(
            transformations=(LinearRegressionToMEstimation(parameter_radius=None),),
            algorithm=ERM(),
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        # y = 2x holds on every row, so the unconstrained least-squares
        # minimizer is the exact slope 2 and the residual sum is zero.
        np.testing.assert_allclose(output.beta, np.array([2.0]), atol=1e-5)

    def test_it_spans_every_parameter_coordinate(self):
        """The declared dimension is the problem's, not the scalar case."""
        lr_metadata, lr_input, _ = _lr_instance(outlier=False)
        mean_metadata, mean_input, _ = _mean_instance()

        lr_prepared = _problem_prepared_by(
            LinearRegressionToMEstimation(parameter_radius=None), lr_metadata, lr_input
        )
        mean_prepared = _problem_prepared_by(
            mean_transformations.ToMEstimation(), mean_metadata, mean_input
        )

        assert lr_prepared.parameter_domain == EuclideanSpace(dim=2)
        # The mean's parameter domain is its box, which spans both coordinates.
        assert mean_prepared.parameter_domain == mean_metadata.domain


class TestHuberLossIdentity:
    """A Huber threshold is part of the configured identity."""

    def test_the_stored_threshold_is_the_configured_field(self):
        assert HuberResidualLoss(threshold=0.5).dump_config().config == {
            "threshold": 0.5
        }
        assert HuberNormLoss(threshold_per_sqrt_dim=0.5).dump_config().config == {
            "threshold_per_sqrt_dim": 0.5
        }

    def test_default_mean_estimation_record_is_empty(self):
        """The default loss is excluded, and no other field remains to record."""
        transformation = mean_transformations.ToMEstimation()
        record = transformation.dump_config()
        assert record.config == {}
        assert Transformation.load_record(record) == transformation

    def test_a_non_default_mean_loss_changes_identity_and_round_trips(self):
        transformation = mean_transformations.ToMEstimation(
            loss=HuberNormLoss(threshold_per_sqrt_dim=0.1)
        )

        assert transformation.dump_config().config["loss"]["config"] == {
            "threshold_per_sqrt_dim": 0.1
        }
        assert (
            Transformation.load_record(transformation.dump_config()) == transformation
        )
