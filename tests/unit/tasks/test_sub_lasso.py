"""Unit tests for the SubLasso selection transformation through explicit compositions."""

from typing import Any, cast

import numpy as np
import pytest

from dpstats.algorithm import PrivacyInterpretation
from dpstats.exceptions import InvalidRegime, NotApplicable

from dpstats.domains import InterceptDomain, ProductDomain
from dpstats.algorithm import BaseAlgorithm
from dpstats.algorithm import CompositeAlgorithm
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
)
from dpstats.tasks.linear_regression.transformations import (
    sub_lasso as sub_lasso_module,
)
from dpstats.tasks.linear_regression.transformations.sub_lasso import (
    SubLasso,
    _block_vote,
    _peel,
)
from dpstats.task import Array1D, Array2D
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput,
    LinearRegressionOutput,
)
from tests.unit.tasks.task_builders import Unreadable, box, lr_metadata


def _instance(n: int = 120, selectable: int = 4):
    """Build an intercept-first instance with a planted sparse signal."""
    rng = np.random.default_rng(11)
    covariates = rng.normal(size=(n, selectable))
    X = np.column_stack((np.ones(n), covariates))
    # Strong signal on the first two selectable columns only.
    y = 30.0 * covariates[:, 0] - 25.0 * covariates[:, 1] + rng.normal(size=n)
    metadata = lr_metadata(
        n,
        design=ProductDomain(
            parts=(InterceptDomain(), box(*[(-6.0, 6.0)] * selectable))
        ),
        response=box((-300.0, 300.0)),
    )
    return metadata, LinearRegressionInput(X=X, y=y)


class TestSupportVote:
    """Votes drop the intercept and totalize ties in source-column order."""

    def test_votes_k_largest_absolute_non_intercept_coefficients(self):
        vote = _block_vote(np.array([9.0, 0.5, -2.0, 1.0, 0.0]), num_features=2)
        np.testing.assert_array_equal(vote, [0, 1, 1, 0])

    def test_exact_ties_totalize_in_source_column_order(self):
        vote = _block_vote(np.array([9.0, 1.0, -1.0, 1.0]), num_features=2)
        np.testing.assert_array_equal(vote, [1, 1, 0])


class TestPeel:
    """Finite epsilon perturbs with fresh Gumbels; infinity picks exact maxima."""

    def test_infinite_epsilon_selects_uniformly_among_exact_maxima(self):
        votes = np.array([3, 3, 0, 3], dtype=np.int64)
        winners = {frozenset(_peel(votes, 2, float("inf"), seed)) for seed in range(60)}

        for selection in winners:
            assert selection <= {0, 1, 3}
        assert len(winners) == 3  # every 2-subset of the tied maxima appears

    def test_finite_epsilon_prefers_high_votes_but_is_random(self):
        votes = np.array([40, 0, 0, 0], dtype=np.int64)
        first_choices = {_peel(votes, 1, 5.0, seed)[0] for seed in range(30)}
        assert 0 in first_choices

    def test_rounds_never_repeat_a_winner(self):
        votes = np.array([5, 4, 3, 2], dtype=np.int64)
        selection = _peel(votes, 4, 0.5, seed=7)
        assert sorted(selection) == [0, 1, 2, 3]


class _SpyChild(BaseAlgorithm):
    """Stand-in capturing the reduced task a composition hands its child."""

    TASK = LinearRegression

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return PrivacyInterpretation.STANDARD

    record: dict[str, Any] | None = None

    def run(self, metadata, input, budget, seed):
        if self.record is not None:
            self.record.update(metadata=metadata, input=input, budget=budget, seed=seed)
        return LinearRegressionOutput(beta=np.full(metadata.dim, 7.0))


class TestCompositionSurface:
    """Formulas, splits, reduction, and restoration through the public run."""

    def test_selection_reduces_and_restores_with_zeros(self):
        metadata, task_input = _instance()
        record: dict[str, Any] = {}
        child = _SpyChild(record=record)
        composition = CompositeAlgorithm(
            transformations=(SubLasso(num_features=2),), algorithm=child
        )
        output = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=2.0, delta=1e-5), seed=3
        )
        assert composition.algorithm.record is not None
        record = composition.algorithm.record

        reduced = record["metadata"]
        assert reduced.dim == 3  # intercept + k
        assert reduced.has_intercept
        assert record["input"].X.shape == (metadata.n_samples, 3)
        # 95% of the offered budget reaches the child.
        assert record["budget"].eps == pytest.approx(2.0 * 0.95)
        assert record["budget"].delta == pytest.approx(1e-5 * 0.95)
        # Restoration scatters the child's vector; omitted coordinates zero.
        assert np.count_nonzero(output.beta) == 3
        assert set(np.flatnonzero(output.beta)) >= {0}
        assert np.all(output.beta[np.flatnonzero(output.beta)] == 7.0)

    def test_planted_signal_columns_are_selected_at_infinite_budget(self):
        metadata, task_input = _instance()
        record: dict[str, Any] = {}
        child = _SpyChild(record=record)

        output = CompositeAlgorithm(
            transformations=(SubLasso(num_features=2),), algorithm=child
        ).run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        # Columns 1 and 2 carry the planted signal; the intercept is retained.
        np.testing.assert_array_equal(np.flatnonzero(output.beta), [0, 1, 2])

    def test_seed_determinism_through_the_public_surface(self):
        metadata, task_input = _instance()
        record: dict[str, Any] = {}
        child = _SpyChild(record=record)
        composition = CompositeAlgorithm(
            transformations=(SubLasso(num_features=2),), algorithm=child
        )

        first = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=9
        )
        second = composition.run(
            metadata, task_input, budget=ApproxDPBudget(eps=1.0, delta=1e-5), seed=9
        )

        np.testing.assert_array_equal(first.beta, second.beta)


class _RecordingLasso:
    """Stand-in for the frozen scikit-learn Lasso that records each block fit."""

    fits: list[tuple[Any, Any]] = []

    def __init__(self, **_: Any) -> None:
        self.coef_ = None

    def fit(self, X: Array2D, y: Array1D) -> "_RecordingLasso":
        _RecordingLasso.fits.append((np.array(X), np.array(y)))
        self.coef_ = np.zeros(X.shape[1])
        return self


class TestPublicBlockLaw:
    """Block count, block size and kept length follow the declared ``n_samples``."""

    @pytest.fixture(autouse=True)
    def recording_lasso(self, monkeypatch: pytest.MonkeyPatch):
        _RecordingLasso.fits = []
        monkeypatch.setattr(sub_lasso_module, "Lasso", _RecordingLasso)

    def test_blocks_are_floor_n_over_d_of_size_floor_n_over_m_dropping_the_rest(self):
        # n = 123, d = 5: 24 blocks of 5 rows; the 3 remaining rows are dropped.
        metadata, task_input = _instance(n=123)

        SubLasso(num_features=2).prepare(metadata, task_input, NO_PRIVACY, 0)

        assert len(_RecordingLasso.fits) == 24
        assert {X.shape for X, _ in _RecordingLasso.fits} == {(5, 5)}
        fitted = {tuple(row) for X, _ in _RecordingLasso.fits for row in X}
        assert len(fitted) == 120
        assert fitted < {tuple(row) for row in task_input.X}


class TestPublicRefusals:
    """Structural limits raise from public metadata before private execution."""

    def test_missing_intercept_is_refused(self):
        metadata, task_input = _instance()
        no_intercept = lr_metadata(
            metadata.n_samples,
            design=box(*[(-6.0, 6.0)] * 5),
            response=metadata.response_domain,
        )

        with pytest.raises(NotApplicable, match="intercept"):
            SubLasso().prepare(no_intercept, task_input, NO_PRIVACY, 0)

    def test_num_features_beyond_selectable_columns_is_refused(self):
        metadata, task_input = _instance()

        with pytest.raises(NotApplicable, match="num_features"):
            SubLasso(num_features=5).prepare(metadata, task_input, NO_PRIVACY, 0)

    def test_too_few_rows_for_one_selector_block_is_refused_unread(self):
        metadata, _ = _instance()
        tiny = lr_metadata(
            3, design=metadata.design_domain, response=metadata.response_domain
        )

        with pytest.raises(InvalidRegime, match="block") as refused:
            SubLasso(num_features=2).prepare(
                tiny, cast(LinearRegressionInput, Unreadable("Input")), NO_PRIVACY, 0
            )
        assert refused.type is InvalidRegime

    def test_too_few_rows_refuses_the_whole_composition(self):
        # A NotApplicable stage would be passed through, sending every column on.
        metadata, task_input = _instance(n=3)
        composition = CompositeAlgorithm(
            transformations=(SubLasso(num_features=2),), algorithm=_SpyChild()
        )

        with pytest.raises(InvalidRegime, match="block") as refused:
            composition.run(metadata, task_input, NO_PRIVACY, 0)
        assert refused.type is InvalidRegime
