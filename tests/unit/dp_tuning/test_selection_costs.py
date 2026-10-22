"""Built-in DP selection costs: accounting routes, released values, and identity."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dpstats.budgets import DPBudget, PureDPBudget, ZCDPBudget
from dpstats.exceptions import InvalidRegime
from dpstats.domains import (
    BoxDomain,
    EuclideanSpace,
    LpBallDomain,
    OneHotDomain,
)
from dpstats.dp_tuning.selection_costs.linear_regression import (
    gaussian_clipped_sum_of_squared_errors,
    laplace_clipped_sum_of_squared_errors,
    unbounded_quantile_median_squared_error,
)
from dpstats.dp_tuning.selection_costs.logistic_regression import (
    laplace_classification_error_rate,
)
from dpstats.dp_tuning.selection_costs.m_estimation import (
    laplace_clipped_empirical_loss,
    unbounded_quantile_median_empirical_loss,
)
from dpstats.dp_tuning.selection_costs.mean_estimation import (
    laplace_l2_error_to_sample_mean,
)
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from dpstats.tasks.logistic_regression import (
    LogisticRegressionInput,
    LogisticRegressionMetadata,
    LogisticRegressionOutput,
)
from dpstats.tasks.m_estimation import (
    MEstimationInput,
    MEstimationMetadata,
    MEstimationOutput,
)
from dpstats.tasks.mean_estimation.transformations.m_estimation import (
    SquaredErrorLoss,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)
from tests.fixtures.accounting_spy import AccountingSpy
from tests.registry_components import ExternalBudget
from tests.fixtures.randomness_fake import FakeRandomness, zero_draw

# One evaluation problem per task: (output, evaluation metadata, evaluation input).
Problem = tuple[Any, Any, Any]


def _linear_problem() -> Problem:
    return (
        LinearRegressionOutput(beta=np.array([1.0])),
        LinearRegressionMetadata(
            n_samples=3,
            design_domain=EuclideanSpace(dim=1),
            response_domain=BoxDomain(bounds=np.array([[0.0, 4.0]])),
        ),
        LinearRegressionInput(X=np.ones((3, 1)), y=np.array([1.0, 2.0, 3.0])),
    )


def _m_estimation_problem() -> Problem:
    return (
        MEstimationOutput(theta=np.array([1.0])),
        MEstimationMetadata(
            parameter_domain=EuclideanSpace(dim=1),
            n_samples=3,
            loss=SquaredErrorLoss(),
            data_domain=EuclideanSpace(dim=1),
        ),
        MEstimationInput(data=np.array([[1.0], [2.0], [3.0]])),
    )


def _mean_problem() -> Problem:
    return (
        MeanEstimationOutput(mean_est=np.array([2.0])),
        MeanEstimationMetadata(
            n_samples=3, domain=LpBallDomain.at_origin(p=2, radius=2.0, dim=1)
        ),
        MeanEstimationInput(X=np.array([[1.0], [2.0], [3.0]])),
    )


def _logistic_problem() -> Problem:
    return (
        LogisticRegressionOutput(theta=np.array([1.0])),
        LogisticRegressionMetadata(dim=1, n_samples=3),
        LogisticRegressionInput(
            X=np.array([[-1.0], [1.0], [2.0]]), y=np.array([0.0, 1.0, 1.0])
        ),
    )


@pytest.mark.parametrize(
    ("selection_cost", "problem", "arguments", "budget", "entry_point"),
    [
        pytest.param(
            laplace_clipped_sum_of_squared_errors,
            _linear_problem,
            {},
            ZCDPBudget(rho=2.0),
            "spend_as_pure",
            id="linear-laplace",
        ),
        pytest.param(
            gaussian_clipped_sum_of_squared_errors,
            _linear_problem,
            {},
            ZCDPBudget(rho=2.0),
            "calibrate_gaussian_scale",
            id="linear-gaussian",
        ),
        pytest.param(
            laplace_clipped_empirical_loss,
            _m_estimation_problem,
            {"clipping_bound": 4.0},
            PureDPBudget(eps=1.0),
            "spend_as_pure",
            id="m-estimation-laplace",
        ),
        pytest.param(
            laplace_l2_error_to_sample_mean,
            _mean_problem,
            {},
            ZCDPBudget(rho=2.0),
            "spend_as_pure",
            id="mean-laplace",
        ),
        pytest.param(
            laplace_classification_error_rate,
            _logistic_problem,
            {},
            PureDPBudget(eps=1.0),
            "spend_as_pure",
            id="logistic-laplace",
        ),
    ],
)
def test_each_noised_cost_spends_its_budget_through_one_route(
    accounting_spy: AccountingSpy,
    selection_cost: Callable[..., float],
    problem: Callable[[], Problem],
    arguments: dict[str, Any],
    budget: DPBudget,
    entry_point: str,
) -> None:
    output, metadata, input_data = problem()

    selection_cost(output, metadata, input_data, budget, 0, **arguments)

    (call,) = accounting_spy.calls
    assert (call.entry_point, call.budget) == (entry_point, budget)
    if entry_point == "calibrate_gaussian_scale":
        assert call.sensitivity == pytest.approx(16.0)


@pytest.mark.parametrize(
    ("selection_cost", "problem", "budget", "seed"),
    [
        pytest.param(
            unbounded_quantile_median_squared_error,
            _linear_problem,
            ZCDPBudget(rho=2.0),
            11,
            id="linear",
        ),
        pytest.param(
            unbounded_quantile_median_empirical_loss,
            _m_estimation_problem,
            PureDPBudget(eps=1.0),
            13,
            id="m-estimation",
        ),
    ],
)
def test_a_median_cost_hands_its_whole_budget_and_seed_to_the_quantile(
    monkeypatch: pytest.MonkeyPatch,
    selection_cost: Callable[..., float],
    problem: Callable[[], Problem],
    budget: DPBudget,
    seed: int,
) -> None:
    from dpstats.tasks.median_estimation import algorithms

    observed: list[tuple[DPBudget, int]] = []

    class RecordingQuantile:
        def __init__(self, beta: float) -> None:
            self.beta = beta

        def run(self, metadata: Any, input: Any, budget: DPBudget, seed: int) -> Any:
            observed.append((budget, seed))
            return type("Output", (), {"median_est": 1.0})()

    monkeypatch.setattr(algorithms, "UnboundedQuantileMechanism", RecordingQuantile)
    output, metadata, input_data = problem()

    selection_cost(output, metadata, input_data, budget, seed, beta=2.0)

    assert observed == [(budget, seed)]


@pytest.mark.parametrize(
    ("selection_cost", "problem"),
    [
        pytest.param(
            unbounded_quantile_median_squared_error, _linear_problem, id="linear"
        ),
        pytest.param(
            unbounded_quantile_median_empirical_loss,
            _m_estimation_problem,
            id="m-estimation",
        ),
    ],
)
def test_a_median_cost_passes_the_declared_evaluation_size_to_the_quantile(
    monkeypatch: pytest.MonkeyPatch,
    selection_cost: Callable[..., float],
    problem: Callable[[], Problem],
) -> None:
    from dpstats.tasks.median_estimation import algorithms

    declared: list[int] = []

    class RecordingQuantile:
        def __init__(self, beta: float) -> None:
            pass

        def run(self, metadata: Any, input: Any, budget: DPBudget, seed: int) -> Any:
            declared.append(metadata.n_samples)
            return type("Output", (), {"median_est": 1.0})()

    monkeypatch.setattr(algorithms, "UnboundedQuantileMechanism", RecordingQuantile)
    output, metadata, input_data = problem()
    metadata = metadata.model_copy(update={"n_samples": 3})

    selection_cost(output, metadata, input_data, PureDPBudget(eps=1.0), 0, beta=2.0)

    assert declared == [3]


@pytest.mark.parametrize(
    "selection_cost",
    [laplace_clipped_sum_of_squared_errors, gaussian_clipped_sum_of_squared_errors],
)
def test_the_linear_cost_refuses_an_unbounded_response_domain(
    selection_cost: Callable[..., float],
) -> None:
    output, metadata, input_data = _linear_problem()
    unbounded = metadata.model_copy(update={"response_domain": EuclideanSpace(dim=1)})

    with pytest.raises(InvalidRegime, match="unbounded"):
        selection_cost(output, unbounded, input_data, ZCDPBudget(rho=2.0), 0)


def test_the_linear_laplace_cost_clips_predictions_and_scales_by_squared_width(
    fake_randomness: FakeRandomness,
) -> None:
    """Predictions of 10 clip to 4, so each error is (4 - y)**2 at most 16."""
    fake_randomness.script("laplace", zero_draw)
    _, metadata, input_data = _linear_problem()

    cost = laplace_clipped_sum_of_squared_errors(
        LinearRegressionOutput(beta=np.array([10.0])),
        metadata,
        input_data,
        PureDPBudget(eps=2.0),
        seed=1,
    )

    assert cost == pytest.approx(9.0 + 4.0 + 1.0)
    assert fake_randomness.parameters_of("laplace", "scale") == [
        pytest.approx(16.0 / 2.0)
    ]


def test_one_added_record_moves_the_off_origin_reference_by_at_most_radius_over_n(
    fake_randomness: FakeRandomness,
) -> None:
    """The box's centre is far from the origin, so the raw sum would move more."""
    fake_randomness.script("laplace", zero_draw)
    bounds = np.array([[-2.0, 5.0], [0.0, 1.0], [-1.0, 3.0]])
    metadata = MeanEstimationMetadata(n_samples=4, domain=BoxDomain(bounds=bounds))
    _, radius = metadata.domain.l2_enclosing_ball()
    rows = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    corner = np.array([5.0, 1.0, 3.0])
    estimate = MeanEstimationOutput(mean_est=np.zeros(3))

    def cost(records: Any) -> float:
        return laplace_l2_error_to_sample_mean(
            estimate,
            metadata,
            MeanEstimationInput(X=records),
            PureDPBudget(eps=1.0),
            seed=0,
        )

    def reference(records: Any) -> Any:
        centre = bounds.mean(axis=1)
        return centre + np.sum(records - centre, axis=0) / 4

    with_corner = np.vstack([rows, corner])
    assert (
        np.linalg.norm(reference(with_corner) - reference(rows)) <= radius / 4 + 1e-12
    )
    assert cost(with_corner) == pytest.approx(np.linalg.norm(reference(with_corner)))


def test_the_mean_cost_measures_against_the_sum_over_the_declared_count(
    fake_randomness: FakeRandomness,
) -> None:
    """Two of three declared rows are present, so the reference is their sum / 3."""
    fake_randomness.script("laplace", zero_draw)
    metadata = MeanEstimationMetadata(
        n_samples=3, domain=LpBallDomain.at_origin(p=2, radius=2.0, dim=1)
    )

    cost = laplace_l2_error_to_sample_mean(
        MeanEstimationOutput(mean_est=np.array([0.0])),
        metadata,
        MeanEstimationInput(X=np.array([[1.0], [2.0]])),
        PureDPBudget(eps=1.0),
        seed=0,
    )

    assert cost == pytest.approx(1.0)


def test_the_mean_cost_adds_laplace_noise_scaled_by_a_box_radius(
    fake_randomness: FakeRandomness,
) -> None:
    """Box half-widths 3.5, 0.5 and 2 give the centred radius sqrt(16.5)."""
    fake_randomness.script("laplace", zero_draw)
    metadata = MeanEstimationMetadata(
        n_samples=4,
        domain=BoxDomain(bounds=np.array([[-2.0, 5.0], [0.0, 1.0], [-1.0, 3.0]])),
    )
    # The rows average to (1, 0.5, 1), which the estimate misses by (3, 0, 4).
    rows = MeanEstimationInput(
        X=np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 2.0], [0.0, 1.0, 0.0], [2.0, 0.0, 2.0]])
    )
    estimate = MeanEstimationOutput(mean_est=np.array([4.0, 0.5, 5.0]))

    cost = laplace_l2_error_to_sample_mean(
        estimate, metadata, rows, PureDPBudget(eps=2.0), seed=17
    )

    assert cost == pytest.approx(5.0)
    (draw,) = fake_randomness.draws_of("laplace")
    assert draw.seed == 17
    assert draw.parameters["scale"] == pytest.approx(math.sqrt(16.5) / (4 * 2.0))


def test_the_mean_cost_adds_laplace_noise_scaled_by_a_ball_radius(
    fake_randomness: FakeRandomness,
) -> None:
    fake_randomness.script("laplace", zero_draw)
    metadata = MeanEstimationMetadata(
        n_samples=5, domain=LpBallDomain.at_origin(p=2, radius=3.0, dim=2)
    )
    rows = MeanEstimationInput(
        X=np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
    )
    estimate = MeanEstimationOutput(mean_est=np.array([0.6, 0.8]))

    cost = laplace_l2_error_to_sample_mean(
        estimate, metadata, rows, PureDPBudget(eps=1.5), seed=3
    )

    assert cost == pytest.approx(1.0)
    assert fake_randomness.parameters_of("laplace", "scale") == [
        pytest.approx(3.0 / (5 * 1.5))
    ]


def test_the_mean_cost_on_a_single_point_domain_draws_at_scale_zero(
    fake_randomness: FakeRandomness,
) -> None:
    """A zero radius spends the offered budget on a Laplace draw of scale zero."""
    metadata = MeanEstimationMetadata(n_samples=2, domain=OneHotDomain(dim=1))

    cost = laplace_l2_error_to_sample_mean(
        MeanEstimationOutput(mean_est=np.array([0.75])),
        metadata,
        MeanEstimationInput(X=np.ones((2, 1))),
        budget=PureDPBudget(eps=1.0),
        seed=0,
    )

    assert cost == pytest.approx(0.25)
    assert fake_randomness.parameters_of("laplace", "scale") == [0.0]


def test_the_classification_cost_is_the_error_rate_over_the_disclosed_count(
    fake_randomness: FakeRandomness,
) -> None:
    """Two of four rows are misclassified, so the rate is 1/2 at scale 1/(4 eps)."""
    fake_randomness.script("laplace", zero_draw)
    metadata = LogisticRegressionMetadata(dim=1, n_samples=4)
    # theta = 1 predicts class 1 wherever x >= 0: (0, 1, 1, 1) against (0, 1, 0, 0).
    rows = LogisticRegressionInput(
        X=np.array([[-1.0], [1.0], [2.0], [3.0]]), y=np.array([0, 1, 0, 0])
    )

    cost = laplace_classification_error_rate(
        LogisticRegressionOutput(theta=np.array([1.0])),
        metadata,
        rows,
        PureDPBudget(eps=2.0),
        seed=5,
    )

    assert cost == pytest.approx(0.5)
    (draw,) = fake_randomness.draws_of("laplace")
    assert draw.seed == 5
    assert draw.parameters["scale"] == pytest.approx(1.0 / (4 * 2.0))


@pytest.mark.parametrize(
    ("evaluate", "named"),
    [
        pytest.param(
            lambda: gaussian_clipped_sum_of_squared_errors(
                *_linear_problem(), PureDPBudget(eps=1.0), 0
            ),
            "PureDPBudget",
            id="gaussian-under-pure",
        ),
        pytest.param(
            lambda: gaussian_clipped_sum_of_squared_errors(
                *_linear_problem(), ExternalBudget(credits=1.0), 0
            ),
            "ExternalBudget",
            id="gaussian-under-unknown-family",
        ),
        pytest.param(
            lambda: laplace_l2_error_to_sample_mean(
                MeanEstimationOutput(mean_est=np.zeros(2)),
                MeanEstimationMetadata(n_samples=2, domain=EuclideanSpace(dim=2)),
                MeanEstimationInput(X=np.zeros((2, 2))),
                PureDPBudget(eps=1.0),
                0,
            ),
            "radius",
            id="mean-on-unbounded-domain",
        ),
    ],
)
def test_a_cost_refuses_an_budget_or_domain_it_cannot_calibrate(
    evaluate: Callable[[], float], named: str
) -> None:
    with pytest.raises(InvalidRegime, match=named):
        evaluate()


def test_builtin_selection_cost_has_one_persisted_identity() -> None:
    from dpstats.dp_tuning import DPExhaustiveCompose
    from dpstats.tasks.linear_regression.algorithms import OLS

    tuner = DPExhaustiveCompose(
        alg_class=OLS,
        alg_hyperparam_grid={},
        selection_cost=laplace_clipped_sum_of_squared_errors,
    )

    assert tuner.dump_config().config["selection_cost"] == (
        "dpstats.dp_tuning.selection_costs.linear_regression:"
        "laplace_clipped_sum_of_squared_errors"
    )


def test_importing_generic_dp_tuning_does_not_load_selection_costs() -> None:
    script = textwrap.dedent(
        """
        import sys

        import dpstats.dp_tuning

        assert not any(
            module_name.startswith("dpstats.dp_tuning.selection_costs.")
            for module_name in sys.modules
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
