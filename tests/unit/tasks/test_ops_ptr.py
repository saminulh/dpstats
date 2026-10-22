"""Tests for generalized-PTR one-posterior-sample regression."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpstats.budgets import (
    ApproxDPBudget,
    DPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.parameters import ValuesSearch
from dpstats.tasks.linear_regression.algorithms import (
    ops_ptr as ops_ptr_module,
)
from dpstats.tasks.linear_regression.algorithms._ops import (
    solve_ops_scaled_temperature,
)
from dpstats.tasks.linear_regression.algorithms.ops_ptr import OPSPTR
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionMetadata,
)
from tests.fixtures.accounting_spy import AccountingSpy, assert_budget_close
from tests.unit.tasks.task_builders import box, lr_metadata


def _bounded_instance() -> tuple[LinearRegressionMetadata, LinearRegressionInput]:
    """Build a small bounded regression instance."""
    X = np.array([[-1.0, 0.2], [-0.4, 0.7], [0.1, -0.3], [0.8, 0.5], [1.0, -0.2]])
    y = np.array([-1.2, -0.1, 0.3, 1.1, 1.4])
    metadata = lr_metadata(5, design=box((-1, 1), (-1, 1)), response=box((-2, 2)))
    return metadata, LinearRegressionInput(X=X, y=y)


class TestOPSPTRCalibration:
    def test_scaled_temperature_saturates_privacy_bound(self) -> None:
        h = 100.0
        budget = ApproxDPBudget(eps=0.5, delta=1e-6)
        t = solve_ops_scaled_temperature(h, feature_radius=1.0, budget=budget)
        log_term = math.log(2.0 / budget.delta)
        privacy_bound = (
            t * math.sqrt(log_term / h)
            + t**2 / (2.0 * (h + 1.0))
            + (1.0 + log_term) / (2.0 * h)
        )
        assert privacy_bound == pytest.approx(budget.eps, rel=1e-12)

    def test_structural_term_can_make_temperature_infeasible(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            solve_ops_scaled_temperature(
                1.0,
                feature_radius=1.0,
                budget=ApproxDPBudget(eps=0.01, delta=1e-6),
            )


class TestOPSPTRRegularizationDefault:
    def test_default_is_the_papers_30_point_grid_point_i_14(self) -> None:
        """The default is a point of the paper's own lambda grid, not its midpoint."""
        # 2.5 ** (1 + 9 * 14 / 29), computed independently of the production
        # expression: Section 5.1's grid point i=14.
        assert OPSPTR().regularization == pytest.approx(133.94322025429884)

    def test_search_values_are_the_papers_30_point_grid(self) -> None:
        """The 30 candidates are pinned by endpoints, count and constant log-spacing.

        A geometric sequence is uniquely determined by its first term, last
        term and length, so fixing those three (Section 5.1's stated
        endpoints and k=30) independently pins the whole sequence without
        recomputing its closed form.
        """
        search = OPSPTR.parameter_info()["regularization"].search
        assert isinstance(search, ValuesSearch)
        values = search.values
        assert len(values) == 30
        assert values[0] == pytest.approx(2.5)
        assert values[-1] == pytest.approx(2.5**10)
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        assert ratios == pytest.approx([ratios[0]] * len(ratios))
        # The declared default is that grid's index-14 point.
        assert values[14] == OPSPTR().regularization

    def test_new_default_is_a_new_configured_identity(self) -> None:
        """A changed default changes content-addressed identity, as intended."""
        assert OPSPTR().uuid != OPSPTR(regularization=2.5**5.5).uuid


class TestOPSPTRAlgorithm:
    def test_infeasible_temperature_is_an_execution_failure(self) -> None:
        metadata, input_data = _bounded_instance()
        with pytest.raises(RuntimeError, match="temperature"):
            OPSPTR(regularization=2.5).run(
                metadata,
                input_data,
                ApproxDPBudget(eps=0.01, delta=1e-6),
                seed=0,
            )

    def test_requires_public_bounds(self) -> None:
        """Reject an instance lacking the bounds used by the privacy proof."""
        metadata, input_data = _bounded_instance()
        unbounded = lr_metadata(5, design=2)
        with pytest.raises(InvalidRegime, match="finite"):
            OPSPTR().run(
                unbounded,
                input_data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

        missing_response = lr_metadata(5, design=metadata.design_domain)
        with pytest.raises(InvalidRegime, match="finite"):
            OPSPTR().run(
                missing_response,
                input_data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )

    @pytest.mark.parametrize("budget", [PureDPBudget(eps=1.0), ZCDPBudget(rho=1.0)])
    def test_refuses_non_approx_budget_before_accounting(
        self, accounting_spy: AccountingSpy, budget: DPBudget
    ) -> None:
        """Refuse families not implied by the paper's approximate-DP guarantee."""
        metadata, input_data = _bounded_instance()
        with pytest.raises(InvalidRegime, match="Cannot spend"):
            OPSPTR().run(metadata, input_data, budget, seed=0)
        assert accounting_spy.calls == ()

    def test_zero_response_bound_is_rejected_before_accounting(
        self, accounting_spy: AccountingSpy
    ) -> None:
        metadata = lr_metadata(2, design=box((-1, 1)), response=box((0, 0)))
        input_data = LinearRegressionInput(X=np.zeros((2, 1)), y=np.zeros(2))

        with pytest.raises(InvalidRegime, match="response norm"):
            OPSPTR().run(
                metadata,
                input_data,
                ApproxDPBudget(eps=1.0, delta=1e-5),
                seed=0,
            )
        assert accounting_spy.calls == ()

    def test_feature_rescaling_maps_coefficients_back(self) -> None:
        """Changing public feature units inversely rescales the released beta."""
        X = np.array([[-0.5], [0.0], [0.25], [0.75]])
        y = np.array([-1.0, 0.2, 0.6, 1.4])
        unit_metadata = lr_metadata(4, design=box((-1, 1)), response=box((-2, 2)))
        doubled_metadata = lr_metadata(4, design=box((-2, 2)), response=box((-2, 2)))
        algorithm = OPSPTR()
        unit = algorithm.run(
            unit_metadata,
            LinearRegressionInput(X=X, y=y),
            ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=7,
        )
        doubled = algorithm.run(
            doubled_metadata,
            LinearRegressionInput(X=2.0 * X, y=y),
            ApproxDPBudget(eps=1.0, delta=1e-5),
            seed=7,
        )
        assert doubled.beta == pytest.approx(unit.beta / 2.0)

    def test_accounting_composition_matches_offered_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
        accounting_spy: AccountingSpy,
    ) -> None:
        """Route both Gaussian leaves and preserve the paper's full split."""
        posterior_budgets: list[ApproxDPBudget] = []
        real_solver = solve_ops_scaled_temperature

        def record_solver(
            regularized_curvature_lower_bound: float,
            feature_radius: float,
            budget: ApproxDPBudget,
        ) -> float:
            posterior_budgets.append(budget)
            return real_solver(
                regularized_curvature_lower_bound, feature_radius, budget
            )

        monkeypatch.setattr(
            ops_ptr_module, "solve_ops_scaled_temperature", record_solver
        )
        metadata, input_data = _bounded_instance()
        budget = ApproxDPBudget(eps=1.0, delta=1e-5)
        OPSPTR().run(metadata, input_data, budget, seed=0)

        scalar_budget = ApproxDPBudget(eps=budget.eps / 4.0, delta=budget.delta / 6.0)
        posterior_budget = ApproxDPBudget(
            eps=budget.eps / 2.0, delta=budget.delta / 6.0
        )
        assert len(accounting_spy.calls) == 2
        assert accounting_spy.calls[0].entry_point == "calibrate_gaussian_scale"
        assert_budget_close(accounting_spy.calls[0].budget, scalar_budget)
        assert accounting_spy.calls[0].sensitivity == pytest.approx(1.0)
        assert accounting_spy.calls[1].entry_point == "calibrate_gaussian_scale"
        assert_budget_close(accounting_spy.calls[1].budget, scalar_budget)
        assert accounting_spy.calls[1].sensitivity is not None
        assert accounting_spy.calls[1].sensitivity > 0.0
        assert len(posterior_budgets) == 1
        assert_budget_close(posterior_budgets[0], posterior_budget)
        assert 2.0 * scalar_budget.eps + posterior_budget.eps == pytest.approx(
            budget.eps
        )
        # The three mechanisms spend delta/2; Theorem C.1's confidence events
        # spend the other delta/2, giving the offered approximate-DP budget.
        assert (
            2.0 * scalar_budget.delta + posterior_budget.delta + budget.delta / 2.0
        ) == pytest.approx(budget.delta)
