"""Portable Sweep authoring, validation, and per-Instance budget resolution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import model_validator

from dpstats.instance import Instance
from dpstats.transport.configuration import PolymorphicModel
from dpstats.transport.canonical import decode_json_document
from dpstats.transport.documents import write_json_atomic
from dpstats.budgets import DPBudget
from dpstats.sweep.budget_policies import BudgetPolicy
from dpstats.coordinates import AlgorithmSeed


class Sweep(PolymorphicModel):
    """The sole reader and writer for an authored Sweep."""

    task: str
    algorithms: tuple[str, ...] | None = None
    instances: tuple[str, ...] | None = None
    budgets: tuple[DPBudget | BudgetPolicy, ...] | None = None
    seeds: tuple[AlgorithmSeed, ...] | None = None
    time_limit: float | None = None
    memory_limit_gb: float | None = None

    @model_validator(mode="after")
    def _validate_sweep(self) -> Sweep:
        if self.task in {"", ".", ".."} or self.task != Path(self.task).name:
            raise ValueError("A Sweep must name exactly one Task.")
        limits: tuple[tuple[str, float | None], ...] = (
            ("time_limit", self.time_limit),
            ("memory_limit_gb", self.memory_limit_gb),
        )
        for field, value in limits:
            if value is not None and not 0 < value < math.inf:
                raise ValueError(
                    f"Sweep {field!r} must be positive and finite, or null for unlimited."
                )
        return self

    def settings(self) -> dict[str, Any]:
        """Project this Sweep as ordinary editable settings.

        Each selection and the seeds become a list; an omitted selection or
        an unlimited resource stays ``None``.
        """

        def listed(values: tuple[Any, ...] | None) -> list[Any] | None:
            return None if values is None else list(values)

        return {
            "task": self.task,
            "algorithms": listed(self.algorithms),
            "instances": listed(self.instances),
            "budgets": listed(self.budgets),
            "seeds": listed(self.seeds),
            "time_limit": self.time_limit,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def budget_groups_for(
        self, instance: Instance
    ) -> tuple[tuple[DPBudget, DPBudget | BudgetPolicy], ...]:
        """Resolve each distinct concrete budget beside the entry declaring it.

        Two declared entries resolving to one concrete budget on this Instance
        are one budget, named by the first entry that reached it.
        """
        distinct: dict[DPBudget, tuple[DPBudget, DPBudget | BudgetPolicy]] = {}
        for declared in self.budgets or ():
            if isinstance(declared, DPBudget):
                budget = declared
            else:
                try:
                    budget = declared.resolve(instance.metadata)
                except ValueError as error:
                    raise ValueError(
                        f"Budget policy {type(declared).__name__} offered no valid "
                        f"budget for Instance {instance.uuid!r}."
                    ) from error
            distinct.setdefault(budget, (budget, declared))
        return tuple(distinct.values())

    def budgets_for(self, instance: Instance) -> tuple[DPBudget, ...]:
        """Resolve this Sweep's distinct concrete budgets for one Instance."""
        return tuple(budget for budget, _ in self.budget_groups_for(instance))


def read_sweep(path: str | Path) -> Sweep:
    """Read and validate one portable Sweep.

    Reconstructing the declared budgets loads the classes they name, exactly as
    a configured registry entry does, so a Sweep naming a class this
    installation cannot import is refused here rather than at resolution.
    """
    document = decode_json_document(Path(path))
    try:
        return Sweep.load_config(document)
    except Exception as error:
        raise ValueError(f"Sweep {path} is invalid.") from error


def write_sweep(path: str | Path, sweep: Sweep, *, overwrite: bool) -> None:
    """Atomically write one portable Sweep.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is false.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    write_json_atomic(path, sweep.dump_config().config)


def read_sweep_task(path: str | Path) -> str:
    """Read the Task one portable Sweep declares, and nothing else.

    Settling an abandoned store must not depend on code the Sweep names, so
    this validates the declared Task through the Sweep model without
    reconstructing any configured component the rest of the document holds.
    """
    document = decode_json_document(Path(path))
    try:
        sweep = Sweep.load_config({"task": document.get("task")})
    except Exception as error:
        raise ValueError(f"Sweep {path} declares no readable Task.") from error
    return sweep.task
