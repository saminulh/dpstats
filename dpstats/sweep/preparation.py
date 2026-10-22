"""Prepare one composite Sweep cell for its child algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dpstats.algorithm import Transformation
from dpstats.budgets import DPBudget
from dpstats.task import TaskInput, TaskMetadata
from dpstats.coordinates import CellCoordinate


@dataclass(frozen=True, slots=True)
class PreparedCell:
    """One transient composite preparation completed for a representative cell.

    The prepared child problem and restoration values exist only while the
    parent Sweep schedules its missing leaves. Process-backed schedulers transport this
    value, so their transformation implementations and restoration values must be
    serializable; serial and thread-backed schedulers need no such property.
    """

    coordinate: CellCoordinate
    attempt_number: int
    metadata: TaskMetadata
    input: TaskInput
    budget: DPBudget
    seed: int
    restorations: tuple[tuple[Transformation, Any], ...]
    runtime_s: float

    def __post_init__(self) -> None:
        if not 0 <= self.runtime_s < math.inf:
            raise ValueError("PreparedCell runtime_s must be finite and nonnegative.")
