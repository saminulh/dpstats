"""One-way Frisch-Waugh-Lovell partialling-out as a composition transformation.

Regressing the group-demeaned retained columns on the group-demeaned
response reproduces the retained coefficients of the full regression on
covariates, intercept, and group indicators. Fixed effects are the
motivating application, but exact FWL needs only a nuisance categorical
block.
"""

from __future__ import annotations

from typing_extensions import override

from dpstats.algorithm import PrivacyInterpretation, Transformation
from dpstats.exceptions import NotApplicable
from dpstats.budgets import DPBudget

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    LpBallDomain,
    OneHotDomain,
    ProductDomain,
)
from dpstats.task import Task
from dpstats.task import Array2D, IntArray1D
from dpstats.tasks import (
    SingleCoefficientEstimation,
)
from dpstats.tasks.single_coefficient_estimation import (
    SingleCoefficientEstimationInput as SCEInput,
    SingleCoefficientEstimationMetadata as SCEMetadata,
    SingleCoefficientEstimationOutput as SCEOutput,
)


def _centered_difference(part: Domain) -> Domain:
    """Origin-centered enclosure of a domain point minus a convex combination.

    Sound blockwise: every coordinate block of a demeaned row lies in the
    difference set of its raw block's convex hull. Bounded geometries keep
    their sharpest supported representation; unboundedness is preserved
    honestly; a geometry without a sharper representation falls back to the
    origin-centered L2 ball at its public L2 diameter.
    """
    if isinstance(part, ProductDomain):
        return ProductDomain(
            parts=tuple(_centered_difference(inner) for inner in part.parts)
        )
    if isinstance(part, BoxDomain):
        widths = part.bounds[:, 1] - part.bounds[:, 0]
        return BoxDomain(bounds=np.column_stack((-widths, widths)))
    if isinstance(part, LpBallDomain):
        return LpBallDomain.at_origin(
            p=part.p,
            radius=2 * part.radius,
            dim=part.dim,
        )
    if isinstance(part, OneHotDomain):
        # Demeaned encodings are not one-hot; the sharpest supported
        # enclosure is the L2 ball at the block's public diameter.
        return LpBallDomain.at_origin(p=2, radius=part.l2_diameter(), dim=part.dim)
    if isinstance(part, EuclideanSpace):
        return part
    diameter = part.l2_diameter()
    if not math.isfinite(diameter):
        return EuclideanSpace(dim=part.dim)
    return LpBallDomain.at_origin(p=2, radius=diameter, dim=part.dim)


class LargestOneHotFWLDemean(Transformation):
    """Remove the largest set of group-indicator controls exactly by demeaning within groups, without privacy.

    An implementation of the method in Frisch and Waugh (1933) and Lovell
    (1963). Picks the largest block of group-indicator columns, drops it, and
    subtracts each group's mean from the remaining columns and the response. By
    the Frisch-Waugh-Lovell theorem the least-squares coefficient of interest
    on this data equals the one with the indicators included. The group means
    come from the private data without noise, so this step has no privacy
    guarantee.

    Requires:
        - A ``ProductDomain`` design; otherwise ``NotApplicable``.
        - A coefficient of interest other than the intercept; otherwise
          ``NotApplicable``.
        - An eligible top-level ``OneHotDomain`` part; otherwise
          ``NotApplicable``.

    Implementation notes:
        - The block removed is the eligible one-hot part with the most columns,
          ties broken by position, chosen from the declared domains alone. A
          part is eligible when it does not contain the coefficient of
          interest; a reference-coded part (``allow_zero``) only when an
          intercept is declared in column zero, and then a record that is zero
          across the block forms its own reference group.
        - Each remaining block's declared domain is replaced by one containing
          every demeaned value.

    Reference:
        - frisch1933partial.
        - lovell1963seasonal.
    """

    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.DIAGNOSTIC
    )

    OUTER_TASK: ClassVar[Task] = SingleCoefficientEstimation
    INNER_TASK: ClassVar[Task] = SingleCoefficientEstimation

    @override
    def prepare(
        self, metadata: SCEMetadata, input: SCEInput, budget: DPBudget, seed: int
    ):
        design = metadata.design_domain
        if not isinstance(design, ProductDomain):
            raise NotApplicable(
                "LargestOneHotFWLDemean requires a ProductDomain design to "
                f"locate a nuisance one-hot block; got {type(design).__name__}."
            )
        parts = design.parts
        offsets = np.cumsum([0] + [part.dim for part in parts])
        coi = metadata.coefficient_of_interest_index
        coi_part = int(np.searchsorted(offsets[1:], coi, side="right"))
        if isinstance(parts[coi_part], InterceptDomain):
            raise NotApplicable(
                "The coefficient of interest is the intercept coordinate, "
                "which within-demeaning removes."
            )

        eligible = [
            position
            for position, part in enumerate(parts)
            if isinstance(part, OneHotDomain)
            and position != coi_part
            and (not part.allow_zero or metadata.has_intercept)
        ]
        if not eligible:
            raise NotApplicable(
                "LargestOneHotFWLDemean found no eligible nuisance one-hot "
                "block: a block must not contain the coefficient of interest, "
                "and a reference-coded block requires a declared intercept."
            )
        selected = max(eligible, key=lambda position: (parts[position].dim, -position))

        retained_positions = [
            position
            for position, part in enumerate(parts)
            if position != selected and not isinstance(part, InterceptDomain)
        ]
        retained_columns: IntArray1D = np.concatenate(
            [
                np.arange(offsets[position], offsets[position + 1], dtype=np.int64)
                for position in retained_positions
            ]
        )
        remapped_coi = int(np.flatnonzero(retained_columns == coi)[0])
        retained_domain = design.select_coordinates(retained_columns)

        prepared_metadata = SCEMetadata(
            n_samples=metadata.n_samples,
            coefficient_of_interest_index=remapped_coi,
            design_domain=_centered_difference(retained_domain),
            response_domain=_centered_difference(metadata.response_domain),
        )

        # Realized one-way groups from the selected block; an all-zero
        # reference-coded record belongs to its own reference group.
        block = input.X[:, offsets[selected] : offsets[selected + 1]]
        hot = np.argmax(block, axis=1)
        group: IntArray1D = np.asarray(
            np.where(block.max(axis=1) > 0, hot, block.shape[1]),
            dtype=np.int64,
        )

        combined = np.column_stack((input.X[:, retained_columns], input.y))
        group_means: Array2D = (
            pd.DataFrame(combined)
            .groupby(group, sort=False)
            .transform("mean")
            .to_numpy()
        )
        centered: Array2D = combined - group_means

        prepared_input = SCEInput(X=centered[:, :-1], y=centered[:, -1])
        return prepared_metadata, prepared_input, budget, seed, None

    @override
    def restore(self, output: SCEOutput, restoration: None):
        return output
