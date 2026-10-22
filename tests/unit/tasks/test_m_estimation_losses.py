"""Shape and value contracts for the configured M-estimation losses."""

import math

import numpy as np
import pytest

from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    LpBallDomain,
    ProductDomain,
)
from dpstats.budgets import NO_PRIVACY
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    AbsoluteResidualLoss,
    HuberResidualLoss,
    RobHytResidualLoss,
    SquaredResidualLoss,
)
from dpstats.tasks.m_estimation.losses import MEstimationLoss
from dpstats.tasks.median_estimation.transformations.to_m_estimation import (
    AbsoluteErrorLoss,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    HuberNormLoss,
    SquaredErrorLoss,
)
from dpstats.tasks.m_estimation import MEstimationMetadata
from dpstats.tasks.median_estimation.transformations import ToMEstimation
from dpstats.tasks.median_estimation import MedianEstimationInput
from tests.unit.tasks.task_builders import median_metadata

_BATCH = np.array([[2.0, -1.0, 3.0], [1.0, 0.0, 0.5]])  # residuals 2, -0.5
_THETA = np.array([1.0, 1.0])

_DESIGN_DOMAIN = BoxDomain(
    bounds=np.array(
        [
            [-2.0, 2.0],
            [-1.0, 1.0],
        ]
    )
)
_RESPONSE_DOMAIN = BoxDomain(bounds=np.array([[-3.0, 3.0]]))
_STACKED_DOMAIN = ProductDomain(parts=(_DESIGN_DOMAIN, _RESPONSE_DOMAIN))
_PARAMETER_DOMAIN = BoxDomain(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]))

_LOCATION_THETA = np.zeros(2)


def _median_loss_metadata() -> MEstimationMetadata:
    metadata, *_ = ToMEstimation().prepare(
        median_metadata(2),
        MedianEstimationInput(X=np.array([3.0, 0.5])),
        budget=NO_PRIVACY,
        seed=0,
    )
    return metadata


class TestSharedRegressionLosses:
    """Hand-derived values and reductions over the record axis."""

    def test_robhyt_reductions_are_hand_derived(self):
        """At tanh(log(3))=0.8, the two row weights are 0.72 and 2."""
        loss = RobHytResidualLoss(k=2.0)
        theta = np.array([1.0, 1.0])
        data = np.array(
            [
                [2.0, -1.0, 1.0 + math.log(3.0)],
                [1.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            loss(data, theta, reduction="none"), [2.0 * math.log(5.0 / 3.0), 0.0]
        )
        assert loss(data, theta, reduction="sum") == pytest.approx(
            2.0 * math.log(5.0 / 3.0)
        )
        assert loss(data, theta, reduction="mean") == pytest.approx(math.log(5.0 / 3.0))

        np.testing.assert_allclose(
            loss.gradient(data, theta, reduction="none"), [[-3.2, 1.6], [0.0, 0.0]]
        )
        np.testing.assert_allclose(
            loss.gradient(data, theta, reduction="sum"), [-3.2, 1.6]
        )
        np.testing.assert_allclose(
            loss.gradient(data, theta, reduction="mean"), [-1.6, 0.8]
        )

        np.testing.assert_allclose(
            loss.hessian(data, theta, reduction="none"),
            [[[2.88, -1.44], [-1.44, 0.72]], [[2.0, 0.0], [0.0, 0.0]]],
        )
        np.testing.assert_allclose(
            loss.hessian(data, theta, reduction="sum"),
            [[4.88, -1.44], [-1.44, 0.72]],
        )
        np.testing.assert_allclose(
            loss.hessian(data, theta, reduction="mean"),
            [[2.44, -0.72], [-0.72, 0.36]],
        )

    def test_losses_without_an_analytic_hessian_report_its_absence(self):
        """An unsupported optional derivative has a stable ``None`` result."""
        assert SquaredResidualLoss().hessian(_BATCH, _THETA, reduction="sum") is None

    @pytest.mark.parametrize(
        ("loss", "data", "theta"),
        [
            (SquaredResidualLoss(), _BATCH, _THETA),
            (AbsoluteResidualLoss(), _BATCH, _THETA),
            (HuberResidualLoss(threshold=1.5), _BATCH, _THETA),
            (RobHytResidualLoss(k=1.5), _BATCH, _THETA),
            (
                SquaredErrorLoss(),
                np.array([[3.0, 4.0], [0.0, 1.0]]),
                _LOCATION_THETA,
            ),
            (
                AbsoluteErrorLoss(),
                np.array([[3.0, 4.0], [0.0, 1.0]]),
                _LOCATION_THETA,
            ),
            (
                HuberNormLoss(threshold_per_sqrt_dim=1 / math.sqrt(2)),
                np.array([[3.0, 4.0], [0.0, 1.0]]),
                _LOCATION_THETA,
            ),
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_all_native_losses_reduce_values_and_gradients_over_records(
        self,
        loss: MEstimationLoss,
        data: np.ndarray,
        theta: np.ndarray,
        batch_size: int,
    ) -> None:
        """Every built-in loss reduces only over the record axis."""
        data = data[:batch_size]
        per_record_values = loss(data, theta, reduction="none")
        per_record_gradients = loss.gradient(data, theta, reduction="none")

        assert isinstance(per_record_values, np.ndarray)
        assert per_record_values.shape == (len(data),)
        assert per_record_gradients.shape == (len(data), len(theta))
        assert loss(data, theta, reduction="sum") == pytest.approx(
            np.sum(per_record_values)
        )
        assert loss(data, theta, reduction="mean") == pytest.approx(
            np.mean(per_record_values)
        )
        np.testing.assert_allclose(
            loss.gradient(data, theta, reduction="sum"),
            np.sum(per_record_gradients, axis=0),
        )
        np.testing.assert_allclose(
            loss.gradient(data, theta, reduction="mean"),
            np.mean(per_record_gradients, axis=0),
        )


@pytest.mark.parametrize(
    ("loss", "expected_value", "expected_gradient", "expected_properties"),
    [
        (
            SquaredResidualLoss(),
            4.0,
            [-8.0, 4.0],
            (True, True, True, 2 * math.sqrt(5) * (3 + math.sqrt(10)), 10.0),
        ),
        (
            AbsoluteResidualLoss(),
            2.0,
            [-2.0, 1.0],
            (True, True, False, math.sqrt(5), math.inf),
        ),
        (
            HuberResidualLoss(threshold=1.5),
            1.875,
            [-3.0, 1.5],
            (True, True, False, 1.5 * math.sqrt(5), 5.0),
        ),
        (
            # rho_k(2) = (k^2/2) log cosh(4/k) and rho_k'(2) = k tanh(4/k)
            # against the record's design row, at k = 1.5.
            RobHytResidualLoss(k=1.5),
            2.2256277962922,
            [-2.971171482677043, 1.4855857413385214],
            (True, True, True, 1.5 * math.sqrt(5), 10.0),
        ),
    ],
)
def test_configured_regression_losses_publish_coherent_behavior_and_properties(
    loss: MEstimationLoss,
    expected_value: float,
    expected_gradient: list[float],
    expected_properties: tuple[bool, bool, bool, float, float],
) -> None:
    np.testing.assert_allclose(loss(_BATCH[:1], _THETA), [expected_value])
    np.testing.assert_allclose(loss.gradient(_BATCH[:1], _THETA), [expected_gradient])

    properties = loss.properties(_STACKED_DOMAIN, _PARAMETER_DOMAIN)

    assert properties.convex is expected_properties[0]
    assert properties.generalized_linear is expected_properties[1]
    assert properties.continuously_twice_differentiable is expected_properties[2]
    assert properties.loss_lipschitz_constant == pytest.approx(expected_properties[3])
    assert properties.gradient_lipschitz_constant == pytest.approx(
        expected_properties[4]
    )
    assert MEstimationLoss.load_record(loss.dump_config()) == loss


@pytest.mark.parametrize(
    ("loss", "expected_value", "expected_gradient", "expected_properties"),
    [
        (
            SquaredErrorLoss(),
            25.0,
            [-6.0, -8.0],
            (True, False, True, 2 * math.sqrt(41), 2.0),
        ),
        (
            AbsoluteErrorLoss(),
            7.0,
            [-1.0, -1.0],
            (True, False, False, math.sqrt(2), math.inf),
        ),
        (
            HuberNormLoss(threshold_per_sqrt_dim=1 / math.sqrt(2)),
            4.5,
            [-0.6, -0.8],
            (True, False, False, 1.0, 1.0),
        ),
    ],
)
def test_configured_location_losses_publish_coherent_behavior_and_properties(
    loss: MEstimationLoss,
    expected_value: float,
    expected_gradient: list[float],
    expected_properties: tuple[bool, bool, bool, float, float],
) -> None:
    data = np.array([[3.0, 4.0]])
    theta = np.zeros(2)
    domain = BoxDomain(bounds=np.array([[-3.0, 3.0], [-4.0, 4.0]]))

    np.testing.assert_allclose(loss(data, theta), [expected_value])
    np.testing.assert_allclose(loss.gradient(data, theta), [expected_gradient])

    properties = loss.properties(domain, _PARAMETER_DOMAIN)

    assert properties.convex is expected_properties[0]
    assert properties.generalized_linear is expected_properties[1]
    assert properties.continuously_twice_differentiable is expected_properties[2]
    assert properties.loss_lipschitz_constant == pytest.approx(expected_properties[3])
    assert properties.gradient_lipschitz_constant == pytest.approx(
        expected_properties[4]
    )
    assert MEstimationLoss.load_record(loss.dump_config()) == loss


def test_robhyt_loss_reaches_its_tail_where_cosh_would_overflow() -> None:
    """``rho_k(r) -> k|r| - (k^2/2) log 2`` far past the overflow of cosh."""
    loss = RobHytResidualLoss(k=2.0)
    data = np.array([[1.0, 0.0, 1e6]])  # residual 1e6 at the origin

    assert loss(data, np.zeros(2), reduction="sum") == pytest.approx(
        2.0 * 1e6 - 2.0 * math.log(2.0), rel=1e-12
    )
    np.testing.assert_allclose(loss.gradient(data, np.zeros(2)), [[-2.0, 0.0]])


class TestMedianL1Contract:
    """The median transformation supplies a per-record L1 loss."""

    def test_batch_returns_per_sample_costs(self):
        batch = _median_loss_metadata().loss(np.array([[3.0], [0.5]]), np.array([1.0]))
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (2,)
        np.testing.assert_allclose(batch, [2.0, 0.5])

    def test_gradient_is_per_record(self):
        loss = _median_loss_metadata().loss
        batch = loss.gradient(np.array([[3.0], [0.5]]), np.array([1.0]))
        np.testing.assert_allclose(batch, [[-1.0], [1.0]])
