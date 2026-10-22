"""Transformations that change the declared linear-regression domains.

:class:`DataBounds` replaces numeric domain blocks with exact coordinatewise
bounds of the realized sample. :class:`ApproxBounds` privately completes
missing bounds and clips the data to make those bounds truthful.
:class:`ApproxBoundsAll` instead privately estimates every numeric
coordinate regardless of an existing public side, so its cost against
:class:`ApproxBounds` measures whether spending budget on range estimation is
worthwhile when bounds are already public.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
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
from dpstats.tasks import LinearRegression
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
    LinearRegressionOutput as LROutput,
)


class DataBounds(Transformation):
    """Bound every coordinate by the data's own minimum and maximum, without privacy.

    Replaces the declared bounds of each column and of the response with their
    observed minimum and maximum, and passes the data unchanged to the next step.
    Those bounds come from the private data without noise, so the result has no
    privacy guarantee; it is a diagnostic.

    Implementation notes:
        - Intercept and one-hot blocks keep their declared domains.
    """

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.DIAGNOSTIC
    )

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ) -> tuple[LRMetadata, LRInput, DPBudget, int, None]:
        """Replace numeric metadata domains with realized sample bounds."""
        design_domain = derive_empirical_domain(metadata.design_domain, input.X)
        response_domain = derive_empirical_domain(
            metadata.response_domain, input.y[:, None]
        )
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=design_domain,
            response_domain=response_domain,
        )
        return prepared_metadata, input, budget, seed, None

    def restore(self, output: LROutput, restoration: None) -> LROutput:
        """Return the child's output coefficients unchanged."""
        return output


class ApproxBounds(Transformation):
    """Privately bound all unbounded coordinates and clip the data to them.

    An adaptation of the method in Section 5.1.1 of Wilson et al. (2020).
    Privately estimates a range for each column that has no known bound, clips
    the data to those ranges, passes the clipped data to the next step, and
    outputs that step's estimate unchanged.

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
          an estimate. Intercept, one-hot and already two-sided coordinates
          cost nothing, and a problem needing no estimate passes the whole
          budget to the next step.

    Reference:
        - wilson2020differentially: Section 5.1.1.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

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
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ) -> tuple[LRMetadata, LRInput, DPBudget, int, None]:
        """Select bounds, clip both data blocks, and pass the residual budget."""
        domains, values, downstream_budget, downstream_seed = estimate_bounds_and_clip(
            (
                (metadata.design_domain, input.X),
                (metadata.response_domain, input.y[:, None]),
            ),
            budget,
            seed,
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
        )
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=domains[0],
            response_domain=domains[1],
        )
        prepared_input = LRInput(X=values[0], y=values[1][:, 0])
        return (
            prepared_metadata,
            prepared_input,
            downstream_budget,
            downstream_seed,
            None,
        )

    def restore(self, output: LROutput, restoration: None) -> LROutput:
        """Return the child's output coefficients unchanged."""
        return output


class ApproxBoundsAll(Transformation):
    """APPROX_BOUNDS. Privately bound every coordinate, even one that is already bounded, and clip the data to them.

    An adaptation of the method in Section 5.1.1 of Wilson et al. (2020).
    Privately estimates a range for every column, even one whose bounds are
    already known, clips the data to those ranges, and passes the clipped data
    to the next step. Set against the variant that only bounds open columns, it
    shows what range estimation costs when the bounds were already known.

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
        - The bounds budget is divided across every declared coordinate,
          including intercept and one-hot columns, which are then passed
          through without an estimate. Each estimate therefore spends less than
          an equal share, and the unused remainder is not passed on.
        - Each estimate is intersected with any declared public side before
          clipping, and an interval that misses the declared domain entirely
          releases nothing.

    Reference:
        - wilson2020differentially: Section 5.1.1.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

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
    def _validate_grid(self) -> "ApproxBoundsAll":
        validate_approx_bounds_grid(self.scale, self.num_bins)
        return self

    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ) -> tuple[LRMetadata, LRInput, DPBudget, int, None]:
        """Estimate every coordinate, clip both data blocks, and pass the residual."""
        domains, values, downstream_budget, downstream_seed = estimate_bounds_and_clip(
            (
                (metadata.design_domain, input.X),
                (metadata.response_domain, input.y[:, None]),
            ),
            budget,
            seed,
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
            estimate_all=True,
        )
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=domains[0],
            response_domain=domains[1],
        )
        prepared_input = LRInput(X=values[0], y=values[1][:, 0])
        return (
            prepared_metadata,
            prepared_input,
            downstream_budget,
            downstream_seed,
            None,
        )

    def restore(self, output: LROutput, restoration: None) -> LROutput:
        """Return the child's output coefficients unchanged."""
        return output


class Standardize(Transformation):
    """Center and scale the data to unit scale in each coordinate, using only its public bounds: a box becomes [−1, 1] per coordinate, and a ball the ball through the all-ones vector. Upon receiving the coefficients returned by the next step, convert them back into the original units.

    Rescales each column and the response to unit scale using their declared
    bounds, which are known before any data are read, and passes the result to
    the next step. This spends no privacy budget.

    Requires:
        - Every ordinary numeric block a ``BoxDomain`` with finite bounds or an
          ``LpBallDomain``.

    Implementation notes:
        - With an intercept, numeric columns and the response are centered and
          scaled; without one, they are only scaled, so a model through the
          origin stays one.
        - Intercept and one-hot blocks, and coordinates of zero width, pass
          through unscaled.
    """

    OUTER_TASK: ClassVar[Task] = LinearRegression
    INNER_TASK: ClassVar[Task] = LinearRegression

    @dataclass(frozen=True)
    class Restoration:
        """Public affine maps needed to restore original coefficients."""

        design: AffineTransform
        response: AffineTransform
        centered: bool

        def restore_nonintercept_coefficient(
            self, coefficient: float, index: int
        ) -> float:
            """Restore one coefficient that does not require coupled slopes."""
            if self.centered and index == 0:
                raise ValueError(
                    "Restoring a centered intercept requires the other slopes."
                )
            return (
                float(self.response.scale[0])
                * coefficient
                / float(self.design.scale[index])
            )

    def prepare(
        self, metadata: LRMetadata, input: LRInput, budget: DPBudget, seed: int
    ) -> tuple[LRMetadata, LRInput, DPBudget, int, Standardize.Restoration]:
        """Prepare standardized public metadata and private data."""
        return self.prepare_with_centering(
            metadata, input, budget, seed, center=metadata.has_intercept
        )

    def prepare_with_centering(
        self,
        metadata: LRMetadata,
        input: LRInput,
        budget: DPBudget,
        seed: int,
        *,
        center: bool,
    ) -> tuple[LRMetadata, LRInput, DPBudget, int, Standardize.Restoration]:
        """Prepare as ``prepare`` does, centering only when *center* is set.

        Centering needs a declared intercept to absorb the shift; without
        centering, every coefficient, the intercept included, restores alone.
        """
        design = standardize_domain(metadata.design_domain, center=center)
        response = standardize_domain(metadata.response_domain, center=center)
        prepared_metadata = LRMetadata(
            n_samples=metadata.n_samples,
            design_domain=design.domain,
            response_domain=response.domain,
        )
        prepared_input = LRInput(
            X=design.apply(input.X),
            y=response.apply(input.y),
        )
        restoration = Standardize.Restoration(
            design=design,
            response=response,
            centered=center,
        )
        return prepared_metadata, prepared_input, budget, seed, restoration

    def restore(
        self, output: LROutput, restoration: Standardize.Restoration
    ) -> LROutput:
        """Return coefficients in the original design and response units."""
        response_scale = float(restoration.response.scale[0])
        beta = response_scale * output.beta / restoration.design.scale
        if restoration.centered:
            beta[0] = (
                float(restoration.response.shift[0])
                + response_scale * output.beta[0]
                - float(np.dot(beta, restoration.design.shift))
            )
        return LROutput(beta=beta)
