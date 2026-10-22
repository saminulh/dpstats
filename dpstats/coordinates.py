"""Scientific-cell coordinates and their stable persistence identity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property, lru_cache, total_ordering
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
import pyarrow as pa
from typing_extensions import Self

from dpstats.algorithm import BaseAlgorithm
from dpstats.budgets import DPBudget
from dpstats.budgets.families import canonical_budget_config
from dpstats.instance import Instance
from dpstats.task import Task
from dpstats.transport.configuration import (
    evict_undeclared_copy_state,
    ConfigurationRecord,
)
from dpstats.transport.canonical import (
    canonical_json_bytes,
    coordinate_identity,
    decode_json_document,
)


FACTOR_KEYS = ("instance_id", "alg_id", "budget_path", "budget_config")

_ALGORITHM_SEED_LIMIT = 2**64


def _validated_algorithm_seed(value: int) -> int:
    """Refuse a seed outside the valid range."""
    if not 0 <= value < _ALGORITHM_SEED_LIMIT:
        raise ValueError(
            "An algorithm seed must be a non-negative integer below "
            f"{_ALGORITHM_SEED_LIMIT}."
        )
    return value


AlgorithmSeed = Annotated[int, AfterValidator(_validated_algorithm_seed)]
"""The seed for an algorithm, in the range [0, 2**64)."""


def budget_key(budget: DPBudget) -> tuple[str, str]:
    """Return the budget factor columns one typed budget keys."""
    record = budget.identity_record()
    return record.path, canonical_json_bytes(record.config).decode("ascii")


def canonical_budget_text(config: str) -> str:
    """Return the budget factor one recorded configuration keys.

    A stored row holds the parameters its Attempt was executed with; the
    factor it joins on is what those parameters name.
    """
    decoded = decode_json_document(config.encode("ascii"))
    return canonical_json_bytes(canonical_budget_config(decoded)).decode("ascii")


@lru_cache(maxsize=8192)
def _budget_identity(budget: DPBudget) -> dict[str, Any]:
    """Return one offer's part of a coordinate identity, derived once per offer.

    A selection repeats few distinct budgets across many coordinates; a cut's
    offers number in the thousands, which the cache bound covers. Callers
    only read the returned record.
    """
    record = budget.identity_record()
    return {"path": record.path, "config": record.config}


def coordinate_record_identity(record: Mapping[str, Any]) -> str:
    """Derive the identity one persisted coordinate record names.

    The budget contributes its canonical parameters, so a cell keeps one
    identity wherever its budget was resolved.
    """
    budget = record["budget"]
    return coordinate_identity(
        {
            **record,
            "budget": {
                "path": budget["path"],
                "config": canonical_budget_config(budget["config"]),
            },
        }
    )


def factor_selection_table(
    groups: Iterable[tuple[Instance, BaseAlgorithm, tuple[DPBudget, ...]]],
) -> pa.Table:
    """Encode selected scientific-cell factors without expanding the seed axis."""
    columns: dict[str, list[str]] = {name: [] for name in FACTOR_KEYS}
    budgets: dict[DPBudget, tuple[str, str]] = {}
    for instance, algorithm, selected in groups:
        for budget in selected:
            if budget not in budgets:
                budgets[budget] = budget_key(budget)
            path, config = budgets[budget]
            columns["instance_id"].append(instance.uuid)
            columns["alg_id"].append(algorithm.uuid)
            columns["budget_path"].append(path)
            columns["budget_config"].append(config)
    return pa.table(columns)


class CoordinateRecord(BaseModel):
    """The stable references one persisted trial's coordinate names.

    This is the coordinate as a persisted record rather than as runtime values: it
    validates a persisted record without importing the Instance or algorithm
    it points at, so a reader can judge a stored record it cannot execute.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_uuid: Annotated[str, Field(min_length=1)]
    alg_uuid: Annotated[str, Field(min_length=1)]
    budget: ConfigurationRecord
    alg_seed: AlgorithmSeed


@total_ordering
class CellCoordinate(BaseModel):
    """One trial's coordinate, independent of any selecting Sweep.

    Runtime code carries the real Instance, configured algorithm, and typed
    privacy budget. Stable identifiers appear only at persistence and join
    boundaries, where ``CoordinateRecord`` is the same point in its persisted form.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance: Instance = Field(description="The Instance this cell runs against.")
    algorithm: BaseAlgorithm = Field(
        description="The configured algorithm this cell runs."
    )
    budget: DPBudget = Field(description="The typed privacy budget offered to it.")
    alg_seed: AlgorithmSeed = Field(description="The algorithm's own random seed.")

    @model_validator(mode="after")
    def _validate_algorithm_task(self) -> Self:
        """Refuse an algorithm that does not implement the Instance's Task."""
        if self.algorithm.task is not self.instance.task:
            raise ValueError(
                f"Algorithm {type(self.algorithm).__name__!r} belongs to task "
                f"{self.algorithm.task.id!r}, not {self.instance.task.id!r}."
            )
        return self

    @property
    def task(self) -> Task:
        """This coordinate's Task, from its Instance."""
        return self.instance.task

    @property
    def instance_uuid(self) -> str:
        """This coordinate's Instance identity."""
        return self.instance.uuid

    @property
    def alg_uuid(self) -> str:
        """This coordinate's algorithm identity."""
        return self.algorithm.uuid

    def to_record(self) -> dict[str, Any]:
        """Project runtime values to stable persisted references."""
        return {
            "instance_uuid": self.instance_uuid,
            "alg_uuid": self.alg_uuid,
            "budget": self.budget.dump_config().model_dump(),
            "alg_seed": self.alg_seed,
        }

    @cached_property
    def uuid(self) -> str:
        """Return this coordinate's content identity.

        One short ASCII token of the family the Instance and algorithm UUIDs
        beside it already use. A cell is addressed and joined by this token
        wherever it has to survive a command line, a job array, a file of work,
        or a persisted observation. The model is frozen, so the token is
        derived once and reused by every hash, equality, and ordering
        comparison.
        """
        return coordinate_identity(
            {
                "instance_uuid": self.instance_uuid,
                "alg_uuid": self.alg_uuid,
                "budget": _budget_identity(self.budget),
                "alg_seed": self.alg_seed,
            }
        )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy this coordinate without carrying the original's cached identity."""
        copied = super().model_copy(update=update, deep=deep)
        evict_undeclared_copy_state(copied)
        return copied

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        instances: Mapping[str, Instance],
        algorithms: Mapping[str, BaseAlgorithm],
    ) -> CellCoordinate:
        """Resolve stable references into a real runtime coordinate."""
        parsed = CoordinateRecord.model_validate(record)
        instance_uuid = parsed.instance_uuid
        alg_uuid = parsed.alg_uuid
        try:
            instance = instances[instance_uuid]
        except KeyError as error:
            raise ValueError(
                f"Cell coordinate names unknown Instance {instance_uuid!r}."
            ) from error
        try:
            algorithm = algorithms[alg_uuid]
        except KeyError as error:
            raise ValueError(
                f"Cell coordinate names unknown algorithm {alg_uuid!r}."
            ) from error
        coordinate = cls(
            instance=instance,
            algorithm=algorithm,
            budget=DPBudget.load_record(parsed.budget),
            alg_seed=parsed.alg_seed,
        )
        if canonical_json_bytes(coordinate.to_record()) != canonical_json_bytes(record):
            raise ValueError("Cell coordinate record does not match its references.")
        return coordinate

    def __hash__(self) -> int:
        # Hash the content identity rather than the fields: the held Instance
        # and algorithm have no field-wise hash, and equal fields derive one
        # identity anyway, so this stays consistent with equality.
        return hash(self.uuid)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CellCoordinate):
            return NotImplemented
        return self.uuid == other.uuid

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CellCoordinate):
            return NotImplemented
        return self.uuid < other.uuid
