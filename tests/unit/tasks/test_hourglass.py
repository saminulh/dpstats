"""Unit tests for the scalar Hourglass mean estimator."""

import math

import numpy as np
import pytest

import dpstats.tasks.mean_estimation.algorithms.hourglass as module
from tests.fixtures.accounting_spy import AccountingCall, AccountingSpy
from dpstats.domains import BoxDomain, EuclideanSpace, LpBallDomain
from dpstats.budgets import (
    NO_PRIVACY,
    ApproxDPBudget,
    GDPBudget,
    PureDPBudget,
    ZCDPBudget,
)
from dpstats.exceptions import InvalidRegime
from dpstats.utils import split_seed
from dpstats.tasks.mean_estimation.algorithms.hourglass import (
    Hourglass,
    _draw_hourglass,
    _staircase_gamma,
)
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
)


def _scalar_cell(lower: float = 2.0, upper: float = 6.0, values=None):
    metadata = MeanEstimationMetadata(
        n_samples=5, domain=BoxDomain(bounds=np.array([[lower, upper]]))
    )
    if values is None:
        values = [2.0, 3.0, 4.0, 5.0, 6.0]
    task_input = MeanEstimationInput(X=np.array(values, dtype=float)[:, None])
    return metadata, task_input


class TestTaskMapping:
    """Scalar mapping, interval normalization, bounds, and determinism."""

    @pytest.mark.parametrize("budget", [NO_PRIVACY, PureDPBudget(eps=math.inf)])
    def test_unconstrained_budget_releases_the_exact_sample_mean(self, budget):
        # At epsilon = inf the staircase gamma is zero and the sampler's joint
        # zero-parameter limit draws no noise.
        metadata, task_input = _scalar_cell()
        output = Hourglass().run(metadata, task_input, budget=budget, seed=3)

        assert output.mean_est[0] == pytest.approx(4.0)

    def test_private_release_lies_in_the_public_interval(self):
        metadata, task_input = _scalar_cell()
        for seed in range(20):
            output = Hourglass().run(
                metadata, task_input, budget=PureDPBudget(eps=1.0), seed=seed
            )
            assert 2.0 <= output.mean_est[0] <= 6.0

    def test_release_is_interval_equivariant_not_unit_bound(self):
        # The same normalized data on a shifted, scaled interval maps the
        # release by the same affine map at a fixed seed.
        base_md, base_in = _scalar_cell(0.0, 1.0, [0.25, 0.5, 0.75])
        wide_md, wide_in = _scalar_cell(10.0, 30.0, [15.0, 20.0, 25.0])

        base = Hourglass().run(base_md, base_in, budget=PureDPBudget(eps=1.0), seed=7)
        wide = Hourglass().run(wide_md, wide_in, budget=PureDPBudget(eps=1.0), seed=7)

        assert wide.mean_est[0] == pytest.approx(10.0 + 20.0 * base.mean_est[0])

    def test_interval_resolves_through_the_bounding_box(self):
        # A one-dimensional L2 ball is an interval: the mechanism resolves
        # [c - r, c + r] through the domain's bounding box rather than
        # requiring a literal box type.
        metadata = MeanEstimationMetadata(
            n_samples=3,
            domain=LpBallDomain(p=2, radius=2.0, center=np.array([4.0])),
        )
        task_input = MeanEstimationInput(X=np.array([[3.0], [4.0], [5.0]]))

        output = Hourglass().run(metadata, task_input, budget=NO_PRIVACY, seed=0)

        assert output.mean_est[0] == 4.0
        private = Hourglass().run(
            metadata, task_input, budget=PureDPBudget(eps=1.0), seed=5
        )
        assert 2.0 <= private.mean_est[0] <= 6.0


class TestPublicRefusals:
    """Unsupported public cells raise before private input is read."""

    def test_vector_dimension_is_refused(self):
        metadata = MeanEstimationMetadata(
            n_samples=3,
            domain=BoxDomain(bounds=np.array([[0.0, 1.0], [0.0, 1.0]])),
        )
        with pytest.raises(InvalidRegime, match="dim"):
            Hourglass().run(metadata, object(), budget=PureDPBudget(eps=1.0), seed=0)

    def test_unbounded_domain_is_refused(self):
        metadata = MeanEstimationMetadata(n_samples=3, domain=EuclideanSpace(dim=1))
        with pytest.raises(InvalidRegime, match="interval"):
            Hourglass().run(metadata, object(), budget=PureDPBudget(eps=1.0), seed=0)

    def test_infinite_bounds_are_refused(self):
        metadata = MeanEstimationMetadata(
            n_samples=3, domain=BoxDomain(bounds=np.array([[0.0, math.inf]]))
        )
        with pytest.raises(InvalidRegime, match="interval"):
            Hourglass().run(metadata, object(), budget=PureDPBudget(eps=1.0), seed=0)

    def test_degenerate_bounds_are_refused(self):
        metadata = MeanEstimationMetadata(
            n_samples=3, domain=BoxDomain(bounds=np.array([[1.0, 1.0]]))
        )
        with pytest.raises(InvalidRegime, match="nondegenerate"):
            Hourglass().run(metadata, object(), budget=PureDPBudget(eps=1.0), seed=0)


class TestAccountingSeam:
    """One pure spend through the seam; every accepted family boundary."""

    def test_spends_the_offered_budget_once_as_pure(
        self, accounting_spy: AccountingSpy
    ):
        metadata, task_input = _scalar_cell()
        offered = ZCDPBudget(rho=0.5)

        Hourglass().run(metadata, task_input, budget=offered, seed=0)

        assert accounting_spy.calls == (AccountingCall("spend_as_pure", offered),)

    @pytest.mark.parametrize(
        "budget",
        [
            PureDPBudget(eps=1.0),
            ApproxDPBudget(eps=1.0, delta=1e-5),
            ZCDPBudget(rho=0.5),
            GDPBudget(mu=1.0),
            NO_PRIVACY,
            PureDPBudget(eps=math.inf),
        ],
    )
    def test_accepted_budget_families_execute(self, budget):
        metadata, task_input = _scalar_cell()
        output = Hourglass().run(metadata, task_input, budget=budget, seed=1)
        assert 2.0 <= output.mean_est[0] <= 6.0


class TestStaircaseGamma:
    """Equation (50) against its raw transcription and both limits."""

    @pytest.mark.parametrize("epsilon", [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0])
    def test_matches_the_raw_published_form(self, epsilon):
        b = math.exp(-epsilon)
        raw = -b / (1 - b) + (b - 2 * b**2 + 2 * b**4 - b**5) ** (1 / 3) / (
            2 ** (1 / 3) * (1 - b) ** 2
        )
        assert _staircase_gamma(epsilon) == pytest.approx(raw, rel=1e-12)

    def test_small_epsilon_limit_is_one_half(self):
        # gamma = 1/2 - epsilon/12 + O(epsilon^2).
        for epsilon in (1e-3, 1e-5):
            assert _staircase_gamma(epsilon) == pytest.approx(
                0.5 - epsilon / 12, abs=1e-7
            )

    def test_large_epsilon_limit_vanishes(self):
        assert _staircase_gamma(40.0) == pytest.approx(
            (math.exp(-40.0) / 2) ** (1 / 3), rel=1e-6
        )
        assert 0.0 < _staircase_gamma(40.0) < 1e-5

    def test_gamma_stays_in_the_unit_interval(self):
        for epsilon in np.geomspace(1e-4, 50.0, 40):
            assert 0.0 < _staircase_gamma(float(epsilon)) <= 0.5


def _hourglass_draws(epsilon: float, count: int, seed: int = 1):
    gamma = _staircase_gamma(epsilon)
    draws = [
        _draw_hourglass(epsilon, gamma, child_seed)
        for child_seed in split_seed(seed, count)
    ]
    z1 = np.array([d[0] for d in draws])
    k = np.array([d[1] for d in draws])
    return gamma, z1, k


@pytest.fixture(scope="module")
def unit_epsilon_draws():
    """One shared 200,000-draw sample at epsilon = 1 for the distributional laws."""
    return _hourglass_draws(1.0, 200_000)


# The shared 200,000-draw fixture alone takes several seconds, which leaves a
# loaded four-worker run no margin under the default ten-second budget.
@pytest.mark.timeout(60)
class TestExactSampler:
    """Independent distributional oracles for the hourglass draw."""

    def test_line_index_is_an_exact_integer(self):
        _, _, k = _hourglass_draws(1.0, 5000)
        assert k.dtype.kind == "i"

    def test_zero_gamma_at_finite_epsilon_keeps_the_sampler_tails(self):
        z1, line_index = _draw_hourglass(1.0, 0.0, 0)

        assert z1 != 0.0
        assert isinstance(line_index, int)

    def test_joint_zero_parameter_limit_is_deterministic(self):
        assert _draw_hourglass(math.inf, 0.0, 0) == (0.0, 0)

    def test_marginal_matches_the_staircase_law(self, unit_epsilon_draws):
        b = math.exp(-1.0)
        gamma, z1, _ = unit_epsilon_draws

        central = float(np.mean(np.abs(z1) < gamma))
        predicted_central = gamma / (gamma + b / (1 - b))
        assert central == pytest.approx(predicted_central, abs=5e-3)

        step_one = float(np.mean((np.abs(z1) >= gamma) & (np.abs(z1) < 1 + gamma)))
        step_two = float(np.mean((np.abs(z1) >= 1 + gamma) & (np.abs(z1) < 2 + gamma)))
        assert step_two / step_one == pytest.approx(b, abs=2e-2)

    def test_central_symmetry(self, unit_epsilon_draws):
        _, z1, _ = unit_epsilon_draws
        assert float(np.mean(np.sign(z1))) == pytest.approx(0.0, abs=8e-3)
        # The Z1 marginal is symmetric; compare its symmetric tail masses.
        assert float(np.mean(z1 > 1.0)) == pytest.approx(
            float(np.mean(z1 < -1.0)), abs=5e-3
        )

    def test_conditional_offset_is_two_sided_geometric(self, unit_epsilon_draws):
        b = math.exp(-1.0)
        gamma, z1, k = unit_epsilon_draws
        center = np.where(
            z1 >= 0.0,
            np.floor(z1 + 1.0 - gamma),
            -np.floor(-z1 + 1.0 - gamma),
        ).astype(int)
        offset = k - center

        zero = float(np.mean(offset == 0))
        one = float(np.mean(np.abs(offset) == 1))
        two = float(np.mean(np.abs(offset) == 2))
        # P(|j|=1)/P(j=0) = 2b and P(|j|=2)/P(|j|=1) = b for the
        # (1-b)/(1+b) * b^|j| law.
        assert one / zero == pytest.approx(2 * b, abs=2e-2)
        assert two / one == pytest.approx(b, abs=2e-2)


class TestZeroDenominatorAtom:
    """The source-omission atom uses the exact integer lattice index."""

    # At epsilon = 0.5, seed 44 draws line index k = -1 while the float
    # denominator reconstructs to ~5.6e-17 rather than 0.0.
    _EPSILON = 0.5
    _SEED = 44

    def _one_row_cell(self):
        metadata = MeanEstimationMetadata(
            n_samples=1, domain=BoxDomain(bounds=np.array([[0.0, 3.0]]))
        )
        return metadata, MeanEstimationInput(X=np.array([[1.0]]))

    @pytest.mark.parametrize(
        ("z1", "expected"),
        [(-1.0, 0.0), (1.0, 3.0), (-1.0 / 3.0, 3.0)],
        ids=["negative-numerator", "positive-numerator", "zero-numerator"],
    )
    def test_exact_zero_atom_uses_the_signed_endpoint_completion(
        self, monkeypatch: pytest.MonkeyPatch, z1: float, expected: float
    ):
        monkeypatch.setattr(
            module,
            "_draw_hourglass",
            lambda eps, gamma, seed: (z1, -1),
        )
        metadata, task_input = self._one_row_cell()
        output = Hourglass().run(
            metadata,
            task_input,
            budget=PureDPBudget(eps=self._EPSILON),
            seed=self._SEED,
        )
        assert output.mean_est[0] == expected

    def test_detection_is_exact_integer_not_float(self):
        # The lattice index is exactly -n while its reconstructed float sum is
        # nonzero, so the public release must follow the completed atom.
        gamma = _staircase_gamma(self._EPSILON)
        z1, k = _draw_hourglass(self._EPSILON, gamma, self._SEED)
        z2 = k - z1
        t = 1.0 / 3.0
        float_denominator = (t + z1) + ((1.0 - t) + z2)

        assert k == -1
        assert float_denominator != 0.0

        metadata, task_input = self._one_row_cell()
        output = Hourglass().run(
            metadata,
            task_input,
            budget=PureDPBudget(eps=self._EPSILON),
            seed=self._SEED,
        )
        assert output.mean_est[0] == 3.0

    def test_nonzero_lattice_denominator_is_not_reconstructed_as_a_float(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The sampler's integer line index is the authoritative noisy count.
        # At this supported extreme, reconstructed float sums cancel to zero
        # even though the conceptual lattice denominator is n + k = 2.
        monkeypatch.setattr(
            module,
            "_draw_hourglass",
            lambda eps, gamma, seed: (1e16, 1),
        )
        metadata, task_input = self._one_row_cell()

        output = Hourglass().run(
            metadata,
            task_input,
            budget=PureDPBudget(eps=self._EPSILON),
            seed=self._SEED,
        )

        assert output.mean_est[0] == 3.0

    def test_negative_noisy_count_clips_as_algorithm_3_defines(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            module,
            "_draw_hourglass",
            lambda eps, gamma, seed: (0.25, -2),
        )
        metadata, task_input = self._one_row_cell()

        output = Hourglass().run(
            metadata,
            task_input,
            budget=PureDPBudget(eps=self._EPSILON),
            seed=self._SEED,
        )

        assert output.mean_est[0] == 0.0

    def test_empty_off_manifold_input_uses_the_nonnegative_completion(self):
        metadata = MeanEstimationMetadata(
            n_samples=1, domain=BoxDomain(bounds=np.array([[0.0, 1.0]]))
        )
        empty = MeanEstimationInput(X=np.empty((0, 1)))
        output = Hourglass().run(metadata, empty, budget=NO_PRIVACY, seed=0)
        assert output.mean_est[0] == 1.0


@pytest.mark.parametrize("epsilon", [1e-5, 100.0])
def test_staircase_gamma_stays_in_domain_at_the_privacy_parameter_envelope(epsilon):
    # The supported numerical envelope is eps in [1e-5, 100]; below about 1e-9
    # gamma loses precision, and it leaves (0, 1] near 1e-16.
    gamma = _staircase_gamma(epsilon)
    assert math.isfinite(gamma)
    assert 0.0 < gamma <= 0.5
