"""Public release-law tests for the HistogramGaussian mean estimator."""

import numpy as np
import pytest

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.domains import (
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.exceptions import InvalidRegime
from dpstats.tasks import list_built_in_algorithms
from dpstats.tasks.mean_estimation.algorithms import (
    DPHistogram,
    GaussianMechanism,
)
from dpstats.utils import split_seed
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.fixtures.randomness_fake import FakeRandomness, draws_in_order, zero_draw
from tests.unit.tasks.task_builders import Unreadable, box, mean_metadata


@pytest.fixture
def gaussian_refusals(monkeypatch):
    """Record the actual child exception and its payload before forwarding."""
    refusals = []
    run = GaussianMechanism.run

    def record_run(self, metadata, input, budget, seed):
        try:
            return run(self, metadata, input, budget, seed)
        except InvalidRegime as error:
            refusals.append((error, str(error)))
            raise

    monkeypatch.setattr(GaussianMechanism, "run", record_run)
    return refusals


def test_mixed_blocks_share_epsilon_and_reconstruct_unprojected_coordinates(
    mean_estimation_task,
    accounting_spy: AccountingSpy,
    fake_randomness: FakeRandomness,
):
    domain = ProductDomain(
        parts=(
            box((7.0, 13.0), (4.0, 4.0)),
            OneHotDomain(dim=3),
            ProductDomain(
                parts=(
                    OneHotDomain(dim=1),
                    box((-4.0, 4.0)),
                    OneHotDomain(dim=1, allow_zero=True),
                    InterceptDomain(),
                )
            ),
        )
    )
    X = np.array(
        [
            [7, 4, 1, 0, 0, 1, -4, 0, 1],
            [9, 4, 0, 1, 0, 1, 0, 1, 1],
            [11, 4, 1, 0, 0, 1, 4, 0, 1],
            [13, 4, 0, 1, 0, 1, 0, 1, 1],
        ],
        dtype=float,
    )
    fake_randomness.script(
        "laplace", draws_in_order([np.array([1.0, -2.0, 3.0]), np.array([-1.0])])
    )
    fake_randomness.script("normal", draws_in_order([np.array([20.0, -30.0])]))

    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    output = algorithm.run(
        mean_metadata(4, domain),
        mean_estimation_task.Input(X=X),
        budget=ApproxDPBudget(eps=6.0, delta=1e-5),
        seed=17,
    )

    # Two whole histograms and one numerical vector: epsilon*=2, not 6/6.
    laplace_calls = [
        c for c in accounting_spy.calls if c.entry_point == "calibrate_laplace_scale"
    ]
    assert len(laplace_calls) == 2
    for laplace in laplace_calls:
        assert_budget_close(laplace.budget, PureDPBudget(eps=2.0))
        assert laplace.sensitivity == pytest.approx(0.25)
    (gaussian,) = [
        c for c in accounting_spy.calls if c.entry_point == "calibrate_gaussian_scale"
    ]
    assert_budget_close(gaussian.budget, ApproxDPBudget(eps=2.0, delta=1e-5))
    # The two separated numerical intervals have joint radius sqrt(3²+4²)=5.
    assert gaussian.sensitivity == pytest.approx(1.25)
    assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx(
        [0.125, 0.125]
    )
    assert fake_randomness.parameters_of("laplace", "size") == [(3,), (1,)]
    assert fake_randomness.parameters_of("normal", "size") == [(2,)]
    # Independent root of Balle–Wang equation (6), Delta=1.25, eps=2, delta=1e-5.
    assert fake_randomness.parameters_of("normal", "scale") == pytest.approx(
        [2.492265557054729]
    )
    np.testing.assert_allclose(
        output.mean_est, [30.0, 4.0, 1.5, -1.5, 3.0, 1.0, -30.0, -0.5, 1.0]
    )


def test_subroutines_receive_original_domains_selected_inputs_and_child_offers(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    monkeypatch,
):
    numeric_domain = LpBallDomain(p=2, center=np.array([12.0, -8.0]), radius=5.0)
    domain = ProductDomain(parts=(box((4.0, 4.0)), OneHotDomain(dim=2), numeric_domain))
    calls = []
    histogram_calls = []
    gaussian_run = GaussianMechanism.run
    histogram_run = DPHistogram.run

    def record_run(self, metadata, input, budget, seed):
        calls.append((metadata, input, budget, seed))
        return gaussian_run(self, metadata, input, budget, seed)

    def record_histogram_run(self, metadata, input, budget, seed):
        histogram_calls.append((metadata, input, budget, seed))
        return histogram_run(self, metadata, input, budget, seed)

    monkeypatch.setattr(GaussianMechanism, "run", record_run)
    monkeypatch.setattr(DPHistogram, "run", record_histogram_run)
    fake_randomness.script("laplace", zero_draw)
    fake_randomness.script("normal", draws_in_order([np.array([0.5, -0.25])]))
    output = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]().run(
        mean_metadata(4, domain),
        mean_estimation_task.Input(
            X=np.array(
                [
                    [4, 1, 0, 15, -4],
                    [4, 0, 1, 9, -4],
                    [4, 1, 0, 12, -8],
                    [4, 0, 1, 12, -8],
                ],
                dtype=float,
            )
        ),
        budget=ApproxDPBudget(eps=4.0, delta=1e-5),
        seed=19,
    )

    (call,) = calls
    child_metadata, child_input, child_budget, child_seed = call
    assert child_metadata.n_samples == 4
    assert isinstance(child_metadata.domain, ProductDomain)
    (child_domain,) = child_metadata.domain.parts
    assert isinstance(child_domain, LpBallDomain)
    assert child_domain.p == 2
    assert child_domain.radius == 5.0
    np.testing.assert_array_equal(child_domain.center, [12.0, -8.0])
    np.testing.assert_array_equal(
        child_input.X, [[15, -4], [9, -4], [12, -8], [12, -8]]
    )
    assert_budget_close(child_budget, ApproxDPBudget(eps=2.0, delta=1e-5))
    assert child_seed == split_seed(19, 2)[1]
    (histogram_call,) = histogram_calls
    histogram_metadata, histogram_input, histogram_budget, histogram_seed = (
        histogram_call
    )
    assert histogram_metadata.n_samples == 4
    assert isinstance(histogram_metadata.domain, OneHotDomain)
    assert histogram_metadata.domain.dim == 2
    assert not histogram_metadata.domain.allow_zero
    np.testing.assert_array_equal(histogram_input.X, [[1, 0], [0, 1], [1, 0], [0, 1]])
    assert_budget_close(histogram_budget, PureDPBudget(eps=2.0))
    assert histogram_seed == split_seed(19, 2)[0]
    np.testing.assert_allclose(output.mean_est, [4.0, 0.5, 0.5, 12.5, -6.25])


@pytest.mark.parametrize("allow_zero", [False, True])
@pytest.mark.parametrize(
    "budget",
    [
        PureDPBudget(eps=2.0),
        ApproxDPBudget(eps=2.0, delta=0.2),
    ],
)
def test_one_categorical_block_spends_pure_epsilon_and_leaves_delta_unused(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
    budget,
    allow_zero,
):
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0 if allow_zero else 1, 0], [1, 0, 0]])
    fake_randomness.script("laplace", zero_draw)
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    output = algorithm.run(
        mean_metadata(4, OneHotDomain(dim=3, allow_zero=allow_zero)),
        mean_estimation_task.Input(X=X),
        budget=budget,
        seed=8,
    )

    (call,) = [
        c for c in accounting_spy.calls if c.entry_point == "calibrate_laplace_scale"
    ]
    assert_budget_close(call.budget, PureDPBudget(eps=2.0))
    assert call.sensitivity == pytest.approx(0.25)
    assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx([0.125])
    assert fake_randomness.parameters_of("laplace", "size") == [(3,)]
    assert fake_randomness.draws_of("normal") == ()
    np.testing.assert_allclose(output.mean_est, [0.5, 0.25 if allow_zero else 0.5, 0.0])


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=np.inf),
        ApproxDPBudget(eps=np.inf, delta=1e-5),
        ApproxDPBudget(eps=2.0, delta=1.0),
        ZCDPBudget(rho=np.inf),
        GDPBudget(mu=np.inf),
    ],
)
@pytest.mark.parametrize("seed", [0, 73])
def test_unconstrained_limits_are_exact(mean_estimation_task, budget, seed):
    domain = ProductDomain(
        parts=(box((2.0, 6.0)), OneHotDomain(dim=2), OneHotDomain(dim=1))
    )
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    output = algorithm.run(
        mean_metadata(2, domain),
        mean_estimation_task.Input(X=np.array([[2, 1, 0, 1], [6, 0, 1, 1]])),
        budget=budget,
        seed=seed,
    )

    np.testing.assert_array_equal(output.mean_est, [4.0, 0.5, 0.5, 1.0])


def test_numerical_only_uses_the_domain_ball_and_the_whole_budget(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
):
    domain = LpBallDomain(p=2, center=np.array([12.0, -8.0]), radius=5.0)
    fake_randomness.script("normal", draws_in_order([np.array([0.5, -0.25])]))
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    output = algorithm.run(
        mean_metadata(4, domain),
        mean_estimation_task.Input(X=np.array([[15, -4], [9, -4], [12, -8], [12, -8]])),
        budget=ApproxDPBudget(eps=2.0, delta=1e-5),
        seed=29,
    )

    (call,) = accounting_spy.calls
    assert call.entry_point == "calibrate_gaussian_scale"
    assert_budget_close(call.budget, ApproxDPBudget(eps=2.0, delta=1e-5))
    assert call.sensitivity == pytest.approx(1.25)
    assert fake_randomness.draws_of("laplace") == ()
    assert fake_randomness.parameters_of("normal", "scale") == pytest.approx(
        [2.492265557054729]
    )
    np.testing.assert_allclose(output.mean_est, [12.5, -6.25])


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        PureDPBudget(eps=1.0),
        ApproxDPBudget(eps=1.0, delta=1e-5),
        ZCDPBudget(rho=1.0),
        GDPBudget(mu=1.0),
    ],
)
def test_public_constants_need_neither_input_accounting_nor_randomness(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
    budget,
):
    domain = ProductDomain(
        parts=(
            box((0.1, 0.1)),
            OneHotDomain(dim=1),
            LpBallDomain(p=2, center=np.array([2.0, 3.0]), radius=0.0),
            InterceptDomain(),
        )
    )
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    output = algorithm.run(
        mean_metadata(11, domain), Unreadable("Input"), budget=budget, seed=0
    )

    np.testing.assert_array_equal(output.mean_est, [0.1, 1.0, 2.0, 3.0, 1.0])
    assert accounting_spy.calls == ()
    assert fake_randomness.seeds == ()


@pytest.mark.parametrize(
    "budget",
    [
        NO_PRIVACY,
        ApproxDPBudget(eps=1.0, delta=1e-5),
        ApproxDPBudget(eps=1.0, delta=1.0),
        PureDPBudget(eps=np.inf),
        ApproxDPBudget(eps=np.inf, delta=1e-5),
        ZCDPBudget(rho=np.inf),
        GDPBudget(mu=np.inf),
    ],
)
def test_unbounded_numerical_domain_refusal_comes_from_gaussian_before_any_release(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    gaussian_refusals,
    budget,
):
    domain = ProductDomain(parts=(OneHotDomain(dim=2), EuclideanSpace(dim=1)))
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    with pytest.raises(InvalidRegime) as raised:
        algorithm.run(
            mean_metadata(2, domain),
            mean_estimation_task.Input(X=np.array([[1, 0, -1], [0, 1, 1]])),
            budget=budget,
            seed=0,
        )
    ((child_error, child_message),) = gaussian_refusals
    assert raised.value is child_error
    assert str(raised.value) == child_message
    assert fake_randomness.draws_of("laplace") == ()
    assert fake_randomness.draws_of("normal") == ()


@pytest.mark.parametrize(
    "domain",
    [
        OneHotDomain(dim=2),
        box((-1.0, 1.0)),
        ProductDomain(parts=(OneHotDomain(dim=2), box((-1.0, 1.0)))),
    ],
)
@pytest.mark.parametrize("budget", [ZCDPBudget(rho=1.0), GDPBudget(mu=1.0)])
def test_finite_concentrated_offers_refuse_any_active_release(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
    domain,
    budget,
):
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    with pytest.raises(InvalidRegime, match=type(budget).__name__):
        algorithm.run(
            mean_metadata(2, domain), Unreadable("Input"), budget=budget, seed=0
        )
    assert accounting_spy.calls == ()
    assert fake_randomness.seeds == ()


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1, 0, -1], [0, 1, 1]]),
        np.array([[0, 1, 0.5], [1, 0, -0.5]]),
    ],
)
def test_finite_pure_refusal_comes_from_gaussian_before_any_histogram_release(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    accounting_spy: AccountingSpy,
    gaussian_refusals,
    X,
):
    domain = ProductDomain(parts=(OneHotDomain(dim=2), box((-1.0, 1.0))))
    algorithm = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]()
    with pytest.raises(InvalidRegime) as raised:
        algorithm.run(
            mean_metadata(2, domain),
            mean_estimation_task.Input(X=X),
            budget=PureDPBudget(eps=1.0),
            seed=0,
        )
    ((child_error, child_message),) = gaussian_refusals
    assert raised.value is child_error
    assert str(raised.value) == child_message
    (gaussian,) = accounting_spy.calls
    assert gaussian.entry_point == "calibrate_gaussian_scale"
    assert_budget_close(gaussian.budget, PureDPBudget(eps=0.5))
    assert fake_randomness.draws_of("laplace") == ()
    assert fake_randomness.draws_of("normal") == ()


@pytest.mark.parametrize(
    "budget", [PureDPBudget(eps=2.0), ApproxDPBudget(eps=2.0, delta=1e-5)]
)
def test_zero_radius_residual_reaches_gaussian_and_keeps_its_child_share(
    mean_estimation_task,
    fake_randomness: FakeRandomness,
    monkeypatch,
    budget,
):
    class LooseBoxPoint(LpBallDomain):
        def bounding_box(self):
            return box((1.0, 3.0))

    domain = ProductDomain(
        parts=(
            OneHotDomain(dim=2),
            LooseBoxPoint(p=2, center=np.array([2.0]), radius=0.0),
        )
    )
    calls = []
    gaussian_run = GaussianMechanism.run

    def record_run(self, metadata, input, budget, seed):
        calls.append((metadata, budget, seed))
        return gaussian_run(self, metadata, input, budget, seed)

    monkeypatch.setattr(GaussianMechanism, "run", record_run)
    fake_randomness.script("laplace", zero_draw)
    output = list_built_in_algorithms(mean_estimation_task)["HistogramGaussian"]().run(
        mean_metadata(2, domain),
        mean_estimation_task.Input(X=np.array([[1, 0, 2], [0, 1, 2]])),
        budget=budget,
        seed=17,
    )

    (call,) = calls
    child_metadata, child_budget, child_seed = call
    center, radius = child_metadata.domain.l2_enclosing_ball()
    np.testing.assert_array_equal(center, [2.0])
    assert radius == 0.0
    expected_budget = (
        ApproxDPBudget(eps=1.0, delta=budget.delta)
        if isinstance(budget, ApproxDPBudget)
        else PureDPBudget(eps=1.0)
    )
    assert_budget_close(child_budget, expected_budget)
    assert child_seed == split_seed(17, 2)[1]
    assert fake_randomness.parameters_of("laplace", "scale") == pytest.approx([0.5])
    np.testing.assert_allclose(output.mean_est, [0.5, 0.5, 2.0])
