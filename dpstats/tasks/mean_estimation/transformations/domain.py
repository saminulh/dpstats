"""Domain transformations for mean estimation."""

from __future__ import annotations

from typing import ClassVar

from pydantic import model_validator

from dpstats.algorithm import PrivacyInterpretation, Transformation
from dpstats.budgets import DPBudget
from dpstats.domains.approx_bounds import (
    estimate_bounds_and_clip,
    validate_approx_bounds_grid,
)
from dpstats.domains.transforms import (
    AffineTransform,
    derive_empirical_domain,
    standardize_domain,
)
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks import MeanEstimation
from dpstats.tasks.mean_estimation import (
    MeanEstimationInput,
    MeanEstimationMetadata,
    MeanEstimationOutput,
)


class DataBounds(Transformation):
    """Bound every coordinate by the data's own minimum and maximum, without privacy.

    Replaces the declared bounds of each coordinate with its observed minimum
    and maximum, and passes the data unchanged to the next step. Those bounds
    come from the private data without noise, so the result has no privacy
    guarantee; it is a diagnostic.
    """

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.DIAGNOSTIC
    )

    OUTER_TASK: ClassVar[Task] = MeanEstimation
    INNER_TASK: ClassVar[Task] = MeanEstimation

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        """Replace numeric metadata blocks with realized sample bounds."""
        prepared_metadata = MeanEstimationMetadata(
            n_samples=metadata.n_samples,
            domain=derive_empirical_domain(metadata.domain, input.X),
        )
        return prepared_metadata, input, budget, seed, None

    def restore(
        self, output: MeanEstimationOutput, restoration: None
    ) -> MeanEstimationOutput:
        """Return the child's output mean unchanged."""
        return output


class ApproxBounds(Transformation):
    """Privately bound all unbounded coordinates and clip the data to them.

    An adaptation of the method in Section 5.1.1 of Wilson et al. (2020).
    Privately estimates a range for each coordinate that has no known bound,
    clips the data to those ranges, passes the clipped data to the next step,
    and outputs that step's estimate unchanged.

    Parametrization notes:
        - ``bounds_share``: The fraction of the budget spent on estimating
          bounds; the next step receives the rest, so the total spend stays
          the offered budget. Follows flexsynth's domain-budget default.
        - ``scale``: The upper edge of the first magnitude bin; each later bin
          edge doubles it. Matches the base-2 histogram of Section 5.1.1 of
          Wilson et al. (2020), which puts a value $x$ in bin $⌈log_2 x⌉$, so
          its first edge is one.
        - ``num_bins``: The number of magnitude bins on each side of zero. Set
          to the bin count of the paper's example in Section 5.1.1.

    Implementation notes:
        - Each private interval is intersected with any declared public side,
          so a release never widens a public bound. An interval that misses the
          declared domain entirely releases nothing.
        - If no histogram bin's noisy count clears the threshold, the threshold
          is retried down a fixed schedule of failure probabilities. If none
          clears it, the coordinate keeps its declared bounds, so it stays
          unbounded, and a later step that needs finite bounds refuses. The
          paper leaves this case to the caller.
        - The bounds budget is divided equally across the coordinates that need
          an estimate. A coordinate with two finite public sides costs nothing,
          and a domain needing no estimate passes the whole budget to the next
          step.

    Reference:
        - wilson2020differentially: Section 5.1.1.
    """

    OUTER_TASK: ClassVar[Task] = MeanEstimation
    INNER_TASK: ClassVar[Task] = MeanEstimation

    bounds_share: float = parameter_field(
        default=0.1,
        gt=0,
        lt=1,
        description="Fraction of the offered budget spent on bound selection.",
    )
    scale: float = parameter_field(
        default=1.0,
        gt=0,
        allow_inf_nan=False,
        description="Right edge of the first positive and negative magnitude bins.",
    )
    num_bins: int = parameter_field(
        default=64,
        gt=0,
        description="Number of base-two magnitude bins on each side of zero.",
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> "ApproxBounds":
        validate_approx_bounds_grid(self.scale, self.num_bins)
        return self

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        None,
    ]:
        """Select bounds, clip the sample, and hand the residual budget on."""
        domains, values, downstream_budget, downstream_seed = estimate_bounds_and_clip(
            ((metadata.domain, input.X),),
            budget,
            seed,
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
        )
        prepared_metadata = MeanEstimationMetadata(
            n_samples=metadata.n_samples,
            domain=domains[0],
        )
        prepared_input = MeanEstimationInput(X=values[0])
        return (
            prepared_metadata,
            prepared_input,
            downstream_budget,
            downstream_seed,
            None,
        )

    def restore(
        self, output: MeanEstimationOutput, restoration: None
    ) -> MeanEstimationOutput:
        """Return the child's output mean unchanged."""
        return output


class Standardize(Transformation):
    """Center and scale the data to unit scale in each coordinate, using only its public bounds: a box becomes [−1, 1] per coordinate, and a ball the ball through the all-ones vector. Upon receiving the mean returned by the next step, convert it back into the original units.

    Rescales each coordinate to unit scale using its declared bounds, which are
    known before any data are read, and passes the result to the next step. This
    spends no privacy budget.

    Requires:
        - Every ordinary numeric block a ``BoxDomain`` with finite bounds or an
          ``LpBallDomain``.

    Implementation notes:
        - Structural blocks and coordinates of zero width pass through
          unscaled.
    """

    OUTER_TASK: ClassVar[Task] = MeanEstimation
    INNER_TASK: ClassVar[Task] = MeanEstimation

    def prepare(
        self,
        metadata: MeanEstimationMetadata,
        input: MeanEstimationInput,
        budget: DPBudget,
        seed: int,
    ) -> tuple[
        MeanEstimationMetadata,
        MeanEstimationInput,
        DPBudget,
        int,
        AffineTransform,
    ]:
        """Prepare standardized public metadata and private data."""
        transform = standardize_domain(metadata.domain, center=True)
        prepared_metadata = MeanEstimationMetadata(
            n_samples=metadata.n_samples,
            domain=transform.domain,
        )
        prepared_input = MeanEstimationInput(X=transform.apply(input.X))
        return prepared_metadata, prepared_input, budget, seed, transform

    def restore(
        self,
        output: MeanEstimationOutput,
        restoration: AffineTransform,
    ) -> MeanEstimationOutput:
        """Return the mean in the original coordinates."""
        return MeanEstimationOutput(mean_est=restoration.restore(output.mean_est))
