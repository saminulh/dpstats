"""SCE adapters over the linear-regression domain transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import model_validator

from dpstats.algorithm import PrivacyInterpretation, Transformation
from dpstats.budgets import DPBudget
from dpstats.domains.approx_bounds import validate_approx_bounds_grid
from dpstats.parameters import parameter_field
from dpstats.task import Task
from dpstats.tasks import SingleCoefficientEstimation
from dpstats.tasks.linear_regression import (
    LinearRegressionInput as LRInput,
    LinearRegressionMetadata as LRMetadata,
)
from dpstats.tasks.linear_regression.transformations import (
    ApproxBounds as LRApproxBounds,
    ApproxBoundsAll as LRApproxBoundsAll,
    DataBounds as LRDataBounds,
    Standardize as LRStandardize,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput as SCEInput,
    SingleCoefficientEstimationMetadata as SCEMetadata,
    SingleCoefficientEstimationOutput as SCEOutput,
)


def _as_sce_problem(
    metadata: LRMetadata,
    input: LRInput,
    coefficient_of_interest_index: int,
) -> tuple[SCEMetadata, SCEInput]:
    """Rehydrate one prepared LR-shaped problem with its SCE task identity."""
    prepared_metadata = SCEMetadata(
        n_samples=metadata.n_samples,
        coefficient_of_interest_index=coefficient_of_interest_index,
        design_domain=metadata.design_domain,
        response_domain=metadata.response_domain,
    )
    prepared_input = (
        input if isinstance(input, SCEInput) else SCEInput(X=input.X, y=input.y)
    )
    return prepared_metadata, prepared_input


class DataBounds(Transformation):
    """Bound every coordinate by the data's own minimum and maximum, without privacy.

    Replaces the declared bounds of each column and of the response with their
    observed minimum and maximum, and passes the data unchanged to the next step.
    Those bounds come from the private data without noise, so the result has no
    privacy guarantee; it is a diagnostic.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return LRDataBounds().privacy_interpretation

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

    def prepare(
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ) -> tuple[SCEMetadata, SCEInput, DPBudget, int, None]:
        """Delegate preparation and retain the coefficient-of-interest index."""
        prepared_metadata, prepared_input, budget, seed, restoration = (
            LRDataBounds().prepare(metadata, input, budget, seed)
        )
        sce_metadata, sce_input = _as_sce_problem(
            prepared_metadata,
            prepared_input,
            metadata.coefficient_of_interest_index,
        )
        return sce_metadata, sce_input, budget, seed, restoration

    def restore(self, output: SCEOutput, restoration: None) -> SCEOutput:
        """Return the child's output coefficient unchanged."""
        return output


class ApproxBounds(Transformation):
    """Privately bound all unbounded coordinates and clip the data to them.

    An adaptation of the method in Section 5.1.1 of Wilson et al. (2020).
    Privately estimates a range for each column that has no known bound, clips
    the data to those ranges, passes the clipped data to the next step, and
    outputs its estimate of the coefficient unchanged.

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
        - The work is done by the linear-regression
          :class:`~dpstats.tasks.linear_regression.transformations.domain.ApproxBounds`.
        - Coordinates already bounded on both sides get no budget. If a private
          interval falls entirely outside a coordinate's declared domain, the
          algorithm outputs nothing.

    Reference:
        - wilson2020differentially: Section 5.1.1.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return LRApproxBounds(
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
        ).privacy_interpretation

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

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
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ) -> tuple[SCEMetadata, SCEInput, DPBudget, int, None]:
        """Delegate clipping while retaining the coefficient-of-interest index."""
        prepared_metadata, prepared_input, budget, seed, restoration = LRApproxBounds(
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
        ).prepare(metadata, input, budget, seed)
        sce_metadata, sce_input = _as_sce_problem(
            prepared_metadata,
            prepared_input,
            metadata.coefficient_of_interest_index,
        )
        return sce_metadata, sce_input, budget, seed, restoration

    def restore(self, output: SCEOutput, restoration: None) -> SCEOutput:
        """Return the child's output coefficient unchanged."""
        return output


class ApproxBoundsAll(Transformation):
    """APPROX_BOUNDS. Privately bound every coordinate, even one that is already bounded, and clip the data to them.

    An adaptation of the method in Section 5.1.1 of Wilson et al. (2020).
    Privately estimates a range for every column, even one whose bounds are
    already known, clips the data to those ranges, passes the clipped data to
    the next step, and outputs its estimate of the coefficient unchanged.

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
        - The work is done by the linear-regression
          :class:`~dpstats.tasks.linear_regression.transformations.domain.ApproxBoundsAll`.
        - The bounds budget is divided across every declared coordinate,
          including intercept and one-hot columns, which are then passed
          through without an estimate. Each estimate therefore spends less than
          an equal share, and the unused remainder is not passed on.
        - An interval that misses the declared domain entirely releases
          nothing.

    Reference:
        - wilson2020differentially: Section 5.1.1.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return LRApproxBoundsAll(
            bounds_share=self.bounds_share,
            scale=self.scale,
            num_bins=self.num_bins,
        ).privacy_interpretation

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

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
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ) -> tuple[SCEMetadata, SCEInput, DPBudget, int, None]:
        """Delegate re-estimation while retaining the coefficient-of-interest index."""
        prepared_metadata, prepared_input, budget, seed, restoration = (
            LRApproxBoundsAll(
                bounds_share=self.bounds_share,
                scale=self.scale,
                num_bins=self.num_bins,
            ).prepare(metadata, input, budget, seed)
        )
        sce_metadata, sce_input = _as_sce_problem(
            prepared_metadata,
            prepared_input,
            metadata.coefficient_of_interest_index,
        )
        return sce_metadata, sce_input, budget, seed, restoration

    def restore(self, output: SCEOutput, restoration: None) -> SCEOutput:
        """Return the child's output coefficient unchanged."""
        return output


class Standardize(Transformation):
    """Center and scale the data to unit scale in each coordinate, using only its public bounds: a box becomes [−1, 1] per coordinate, and a ball the ball through the all-ones vector. Upon receiving the coefficient returned by the next step, convert it back into the original units.

    Rescales each column and the response to unit scale using their declared
    bounds, which are known before any data are read, and passes the result to
    the next step. This spends no privacy budget.

    Requires:
        - Every ordinary numeric block a ``BoxDomain`` with finite bounds or an
          ``LpBallDomain``.

    Implementation notes:
        - The output coefficient is restored with the public scale factors of
          the response and of its own column, so no other coefficient is
          needed. A centered intercept cannot be restored this way: it needs
          the slopes this task does not report. When the coefficient of
          interest is the declared intercept, columns and the response are
          therefore only scaled, not centered.
    """

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        return LRStandardize().privacy_interpretation

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

    @dataclass(frozen=True)
    class Restoration:
        """LR restoration plus the scalar coordinate retained by SCE."""

        linear_regression: LRStandardize.Restoration
        coefficient_of_interest_index: int

    def prepare(
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ) -> tuple[SCEMetadata, SCEInput, DPBudget, int, Standardize.Restoration]:
        """Delegate preparation for a scalar-output child."""
        index = metadata.coefficient_of_interest_index
        (
            prepared_metadata,
            prepared_input,
            budget,
            seed,
            lr_restoration,
        ) = LRStandardize().prepare_with_centering(
            metadata,
            input,
            budget,
            seed,
            center=metadata.has_intercept and index != 0,
        )
        sce_metadata, sce_input = _as_sce_problem(
            prepared_metadata,
            prepared_input,
            index,
        )
        restoration = Standardize.Restoration(
            linear_regression=lr_restoration,
            coefficient_of_interest_index=index,
        )
        return sce_metadata, sce_input, budget, seed, restoration

    def restore(
        self, output: SCEOutput, restoration: Standardize.Restoration
    ) -> SCEOutput:
        """Return the targeted coefficient in its original units."""
        coefficient = restoration.linear_regression.restore_nonintercept_coefficient(
            output.coefficient,
            restoration.coefficient_of_interest_index,
        )
        return SCEOutput(coefficient=coefficient)
