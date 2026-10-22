from __future__ import annotations

# pyright: reportPrivateUsage=false

from typing import Any, ClassVar

import numpy as np
import pytest

import dpstats.tasks.linear_regression.transformations.kendall as kendall_module
import dpstats.tasks.linear_regression.transformations.kendall_selection as transformation_module
from dpstats.algorithm import (
    BaseAlgorithm,
    CompositeAlgorithm,
    PrivacyInterpretation,
)
from dpstats.exceptions import NotApplicable

from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    DPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime, NoRelease
from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Array1D, Array2D, IntArray1D
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.linear_regression.algorithms.ssp import (
    SSP,
)
from dpstats.tasks.linear_regression.transformations import DPKendall
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
    LinearRegressionOutput,
)
from tests.registry_components import ExternalBudget
from tests.fixtures.accounting_spy import (
    AccountingCall,
    AccountingSpy,
    assert_budget_close,
)


def _metadata(
    *,
    dim: int,
    n_samples: int,
    design_domain: Domain | None = None,
    response_domain: Domain | None = None,
) -> LinearRegressionMetadata:
    return LinearRegressionMetadata(
        n_samples=n_samples,
        design_domain=(
            EuclideanSpace(dim=dim) if design_domain is None else design_domain
        ),
        response_domain=(
            EuclideanSpace(dim=1) if response_domain is None else response_domain
        ),
    )


class RecordingRegressor(BaseAlgorithm):
    TASK = LinearRegression

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    calls: ClassVar[
        list[
            tuple[
                LinearRegressionMetadata,
                LinearRegressionInput,
                DPBudget,
                int,
            ]
        ]
    ] = []

    def __init__(self, **_kwargs: Any) -> None:
        super().__init__()

    def run(
        self,
        metadata: LinearRegressionMetadata,
        input: LinearRegressionInput,
        budget: DPBudget,
        seed: int,
    ) -> LinearRegressionOutput:
        self.calls.append((metadata, input, budget, seed))
        return LinearRegressionOutput(beta=np.arange(1, metadata.dim + 1, dtype=float))


def _dp_kendall(
    *,
    algorithm_arg: BaseAlgorithm | None = None,
    num_features: int = 5,
    selection_share: float = 0.05,
) -> CompositeAlgorithm:
    return CompositeAlgorithm(
        transformations=(
            DPKendall(
                num_features=num_features,
                selection_share=selection_share,
            ),
        ),
        algorithm=SSP() if algorithm_arg is None else algorithm_arg,
    )


def test_public_run_selects_dominant_feature_at_large_epsilon() -> None:
    rng = np.random.default_rng(4)
    n = 200
    signal = np.arange(n, dtype=float)
    X = np.column_stack((rng.normal(size=n), signal, rng.normal(size=n)))
    y = signal
    output = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=1).run(
        _metadata(dim=3, n_samples=n),
        LinearRegressionInput(X=X, y=y),
        PureDPBudget(eps=1e9),
        seed=7,
    )

    assert np.flatnonzero(output.beta).tolist() == [1]


def test_unconstrained_tie_selection_stays_on_exact_maximizers() -> None:
    signal = np.arange(20, dtype=float)
    X = np.column_stack((signal, signal, np.random.default_rng(1).normal(size=20)))
    output = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=1).run(
        _metadata(dim=3, n_samples=20),
        LinearRegressionInput(X=X, y=signal),
        NO_PRIVACY,
        seed=11,
    )

    assert np.flatnonzero(output.beta).item() in (0, 1)


def test_two_row_score_is_exact() -> None:
    ranks = np.arange(2, dtype=np.int64)
    assert kendall_module._scaled_kendall_score(ranks, ranks) == 1.0


@pytest.mark.parametrize(
    "budget,selection_budget,expected_downstream",
    [
        (PureDPBudget(eps=2.0), PureDPBudget(eps=0.1), PureDPBudget(eps=1.9)),
        (
            ApproxDPBudget(eps=2.0, delta=1e-5),
            ApproxDPBudget(eps=0.1, delta=5e-7),
            ApproxDPBudget(eps=1.9, delta=9.5e-6),
        ),
        (ZCDPBudget(rho=2.0), ZCDPBudget(rho=0.1), ZCDPBudget(rho=1.9)),
        (
            GDPBudget(mu=2.0),
            GDPBudget(mu=2.0 * np.sqrt(0.05)),
            GDPBudget(mu=2.0 * np.sqrt(0.95)),
        ),
    ],
)
def test_wrapper_composes_typed_budget_and_restores_output(
    monkeypatch: pytest.MonkeyPatch,
    accounting_spy: AccountingSpy,
    budget: DPBudget,
    selection_budget: DPBudget,
    expected_downstream: DPBudget,
) -> None:
    RecordingRegressor.calls.clear()
    selection_budgets: list[PureDPBudget] = []

    def record_selection(
        X: Array2D,
        y: Array1D,
        k: int,
        selection_budget: PureDPBudget,
        seed: int,
    ) -> IntArray1D:
        del X, y, seed
        selection_budgets.append(selection_budget)
        return np.arange(k, dtype=np.int64)

    monkeypatch.setattr(
        transformation_module, "_select_features_privately", record_selection
    )
    X = np.column_stack(
        (
            np.linspace(-2.0, 2.0, 30),
            np.linspace(-1, 1, 30),
            np.sin(np.arange(30)),
            np.cos(np.arange(30)),
        )
    )
    y = X[:, 0]
    bounds = np.asarray([[-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0], [-5.0, 5.0]])
    metadata = _metadata(
        dim=4,
        n_samples=30,
        design_domain=BoxDomain(bounds=bounds),
    )
    algorithm = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=2)

    output = algorithm.run(
        metadata,
        LinearRegressionInput(X=X, y=y),
        budget=budget,
        seed=3,
    )

    assert len(accounting_spy.calls) == 1
    assert accounting_spy.calls[0].entry_point == "spend_as_pure"
    assert_budget_close(accounting_spy.calls[0].budget, selection_budget)
    assert len(RecordingRegressor.calls) == 1
    reduced_metadata, reduced_input, downstream_budget, downstream_seed = (
        RecordingRegressor.calls[0]
    )
    assert reduced_metadata.dim == 2
    assert reduced_input.X.shape == (30, 2)
    assert isinstance(reduced_metadata.design_domain, BoxDomain)
    assert reduced_metadata.design_domain.bounds.shape == (2, 2)
    assert isinstance(downstream_seed, int)
    assert len(selection_budgets) == 1
    assert_budget_close(downstream_budget, expected_downstream)
    assert np.count_nonzero(output.beta) == 2


def test_default_downstream_refuses_pure_budget_through_gaussian_seam(
    accounting_spy: AccountingSpy,
) -> None:
    rng = np.random.default_rng(91)
    metadata = _metadata(
        dim=2,
        n_samples=20,
        design_domain=BoxDomain(
            bounds=np.asarray([[-3.0, 3.0], [-3.0, 3.0]], dtype=float)
        ),
        response_domain=BoxDomain(bounds=np.asarray([[-3.0, 3.0]], dtype=float)),
    )
    input_data = LinearRegressionInput(
        X=rng.normal(size=(20, 2)),
        y=rng.normal(size=20),
    )
    budget = PureDPBudget(eps=2.0)
    with pytest.raises(NoRelease) as raised:
        _dp_kendall(num_features=1).run(metadata, input_data, budget=budget, seed=0)
    assert isinstance(raised.value.__cause__, InvalidRegime)
    assert len(accounting_spy.calls) == 2
    assert accounting_spy.calls[0] == AccountingCall(
        "spend_as_pure", PureDPBudget(eps=0.1)
    )
    refusal = accounting_spy.calls[1]
    assert refusal.entry_point == "spend_as_gdp"


def test_an_unknown_budget_family_is_refused_as_an_invalid_regime() -> None:
    rng = np.random.default_rng(92)
    input_data = LinearRegressionInput(
        X=rng.normal(size=(20, 2)), y=rng.normal(size=20)
    )

    with pytest.raises(InvalidRegime, match="ExternalBudget"):
        _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=1).run(
            _metadata(dim=2, n_samples=20),
            input_data,
            budget=ExternalBudget(credits=1.0),
            seed=0,
        )


def test_num_features_above_public_dimension_is_refused_before_accounting(
    accounting_spy: AccountingSpy,
) -> None:
    metadata = _metadata(dim=2, n_samples=3)
    input = LinearRegressionInput(X=np.ones((3, 2)), y=np.ones(3))

    with pytest.raises(NotApplicable, match="exceeds"):
        DPKendall(num_features=3).prepare(metadata, input, PureDPBudget(eps=1.0), 0)
    assert accounting_spy.calls == ()


def test_declared_intercept_is_excluded_then_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingRegressor.calls.clear()
    selector_inputs: list[Array2D] = []

    def record_selection(
        X: Array2D,
        y: Array1D,
        k: int,
        budget: PureDPBudget,
        seed: int,
    ) -> IntArray1D:
        del y, budget, seed
        selector_inputs.append(X)
        assert k == 2
        return np.array([1, 2], dtype=np.int64)

    monkeypatch.setattr(
        transformation_module, "_select_features_privately", record_selection
    )
    X = np.column_stack(
        (
            np.ones(12),
            np.arange(12, dtype=float),
            np.linspace(-1, 1, 12),
            np.arange(12, dtype=float) ** 2,
        )
    )
    output = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=2).run(
        _metadata(
            dim=4,
            n_samples=12,
            design_domain=ProductDomain(
                parts=(InterceptDomain(), EuclideanSpace(dim=3))
            ),
        ),
        LinearRegressionInput(X=X, y=np.arange(12, dtype=float)),
        PureDPBudget(eps=2.0),
        seed=4,
    )

    assert len(selector_inputs) == 1
    np.testing.assert_array_equal(selector_inputs[0], X[:, [1, 2, 3]])
    child_metadata, child_input, _, _ = RecordingRegressor.calls[0]
    assert child_metadata.dim == 3
    assert child_metadata.has_intercept
    np.testing.assert_array_equal(child_input.X, X[:, [0, 2, 3]])
    assert np.flatnonzero(output.beta).tolist() == [0, 2, 3]


def test_selected_set_returns_to_source_order_across_domain_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingRegressor.calls.clear()

    def select_out_of_order(
        X: Array2D,
        y: Array1D,
        k: int,
        budget: PureDPBudget,
        seed: int,
    ) -> IntArray1D:
        del X, y, k, budget, seed
        return np.array([3, 0, 2], dtype=np.int64)

    monkeypatch.setattr(
        transformation_module, "_select_features_privately", select_out_of_order
    )
    X = np.column_stack(
        (
            np.ones(12),
            np.linspace(-2.0, 2.0, 12),
            np.linspace(-3.0, 3.0, 12),
            np.tile([1.0, 0.0], 6),
            np.tile([0.0, 1.0], 6),
        )
    )
    metadata = _metadata(
        dim=5,
        n_samples=12,
        design_domain=ProductDomain(
            parts=(
                InterceptDomain(),
                BoxDomain(bounds=np.array([[-2.0, 2.0], [-3.0, 3.0]])),
                OneHotDomain(dim=2),
            )
        ),
    )

    output = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=3).run(
        metadata,
        LinearRegressionInput(X=X, y=np.arange(12, dtype=float)),
        PureDPBudget(eps=2.0),
        seed=4,
    )

    child_metadata, child_input, _, _ = RecordingRegressor.calls[0]
    np.testing.assert_array_equal(child_input.X, X[:, [0, 1, 3, 4]])
    assert isinstance(child_metadata.design_domain, ProductDomain)
    intercept, box, category = child_metadata.design_domain.parts
    assert isinstance(intercept, InterceptDomain)
    assert isinstance(box, BoxDomain)
    np.testing.assert_array_equal(box.bounds, [[-2.0, 2.0]])
    assert category == OneHotDomain(dim=2)
    assert np.flatnonzero(output.beta).tolist() == [0, 1, 3, 4]


def test_dim_two_declared_intercept_is_retained_and_estimated() -> None:
    n = 12
    x = np.linspace(-1.0, 1.0, n)
    X = np.column_stack((np.ones(n), x))
    y = 3.0 + 2.0 * x

    output = _dp_kendall(algorithm_arg=OLS(), num_features=1).run(
        _metadata(
            dim=2,
            n_samples=n,
            design_domain=ProductDomain(
                parts=(InterceptDomain(), EuclideanSpace(dim=1))
            ),
        ),
        LinearRegressionInput(X=X, y=y),
        NO_PRIVACY,
        seed=5,
    )

    np.testing.assert_allclose(output.beta, [3.0, 2.0])


@pytest.mark.parametrize(
    "budget,selection_budget,expected_downstream",
    [
        (NO_PRIVACY, NO_PRIVACY, NO_PRIVACY),
        (PureDPBudget(eps=np.inf), PureDPBudget(eps=np.inf), PureDPBudget(eps=np.inf)),
        (
            ApproxDPBudget(eps=np.inf, delta=1e-5),
            ApproxDPBudget(eps=np.inf, delta=5e-7),
            ApproxDPBudget(eps=np.inf, delta=9.5e-6),
        ),
        (ZCDPBudget(rho=np.inf), ZCDPBudget(rho=np.inf), ZCDPBudget(rho=np.inf)),
        (GDPBudget(mu=np.inf), GDPBudget(mu=np.inf), GDPBudget(mu=np.inf)),
    ],
)
def test_unconstrained_offer_reaches_the_child_without_constraint(
    accounting_spy: AccountingSpy,
    budget: DPBudget,
    selection_budget: DPBudget,
    expected_downstream: DPBudget,
) -> None:
    RecordingRegressor.calls.clear()
    n = 20
    signal = np.arange(n, dtype=float)
    X = np.column_stack((signal, np.linspace(-1.0, 1.0, n)))
    y = signal

    output = _dp_kendall(algorithm_arg=RecordingRegressor(), num_features=1).run(
        _metadata(dim=2, n_samples=n),
        LinearRegressionInput(X=X, y=y),
        budget,
        seed=13,
    )

    assert len(accounting_spy.calls) == 1
    assert accounting_spy.calls[0].entry_point == "spend_as_pure"
    assert_budget_close(accounting_spy.calls[0].budget, selection_budget)
    assert len(RecordingRegressor.calls) == 1
    assert_budget_close(RecordingRegressor.calls[0][2], expected_downstream)
    assert np.count_nonzero(output.beta) == 1


def test_non_private_child_refuses_a_constrained_offer() -> None:
    n = 20
    rng = np.random.default_rng(2)
    metadata = _metadata(dim=2, n_samples=n)
    input = LinearRegressionInput(X=rng.normal(size=(n, 2)), y=rng.normal(size=n))

    with pytest.raises(NoRelease) as raised:
        _dp_kendall(algorithm_arg=OLS(), num_features=1).run(
            metadata,
            input,
            PureDPBudget(eps=2.0),
            seed=0,
        )
    assert isinstance(raised.value.__cause__, InvalidRegime)


def test_selection_size_defaults_to_the_paper_k() -> None:
    assert DPKendall().num_features == 5
