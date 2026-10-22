"""Typed privacy budgets offered to algorithms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from decimal import Context, Decimal, ROUND_HALF_EVEN
from functools import cached_property
from typing import Any, cast, final

from pydantic import Field, field_validator
from typing_extensions import Self, override

from dpstats.transport.canonical import canonical_json_bytes
from dpstats.transport.configuration import (
    ConfigurationRecord,
    PolymorphicModel,
)

BUDGET_IDENTITY_DIGITS = 12
_IDENTITY_CONTEXT = Context(prec=BUDGET_IDENTITY_DIGITS, rounding=ROUND_HALF_EVEN)


def canonical_budget_parameter(value: float | Decimal) -> float:
    """Return the canonical value naming one resolved budget parameter.

    Two hosts evaluating one rule can disagree in a float's last places, and
    neither result need be the correctly rounded one, so a parameter is named
    by its quantization to ``BUDGET_IDENTITY_DIGITS`` significant decimal
    digits, half to even. A float is quantized from its exact decimal value,
    so a recorded parameter and a freshly resolved one reach one name.

    Args:
        value: One budget parameter, as resolved in decimal or as recorded.

    Returns:
        The nearest float to the quantized decimal. A non-finite value is
        returned unchanged.

    Example:
        >>> canonical_budget_parameter(0.0017735147777247586)
        0.00177351477772
    """
    return float(_IDENTITY_CONTEXT.plus(Decimal(value)))


def _canonical_value(value: Any) -> Any:
    """Canonicalize one configuration value, recursing into nested containers."""
    if isinstance(value, float):
        return canonical_budget_parameter(value)
    if isinstance(value, Mapping):
        nested = cast("Mapping[str, Any]", value)
        return {name: _canonical_value(item) for name, item in nested.items()}
    if isinstance(value, list):
        return [_canonical_value(item) for item in cast("list[Any]", value)]
    return value


def canonical_budget_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return one budget configuration holding the parameters that name it.

    Args:
        config: One budget's configuration, as ``dump_config`` returns it or as
            a stored row recorded it.

    Returns:
        The same configuration with every float parameter, at any nesting
        depth, replaced by its canonical value.

    Example:
        >>> canonical_budget_config({"eps": 1.0, "delta": 0.0017735147777247586})
        {'eps': 1.0, 'delta': 0.00177351477772}
    """
    return {name: _canonical_value(value) for name, value in config.items()}


def _validated_split_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Return positive finite allocation weights."""
    values = tuple(weights)
    if not values:
        raise ValueError("weights must be nonempty.")
    if any(not math.isfinite(weight) or weight <= 0 for weight in values):
        raise ValueError("weights must be finite and positive.")
    return values


class DPBudget(PolymorphicModel, ABC):
    """Base class for one offered differential-privacy budget.

    Custom families declare their own ``split`` composition law and whether an
    offer is unconstrained. Two offers are one budget when their canonical
    identities agree, so a rule resolved on two hosts names one budget.
    """

    @property
    @abstractmethod
    def is_unconstrained(self) -> bool:
        """Whether this offer imposes no privacy constraint."""

    @abstractmethod
    def split(self, weights: Sequence[float]) -> tuple[Self, ...]:
        """Allocate this offer among positive relative weights.

        Args:
            weights: Positive relative shares of this offer, one per part.

        Returns:
            One offer of this family per weight, in order.
        """

    def identity_record(self) -> ConfigurationRecord:
        """Return this offer's class path beside its canonical parameters.

        Every parameter is already canonicalized at construction, so this
        recanonicalizes what is already canonical: a deliberate no-op that
        keeps the identity derivation correct even for a family whose
        parameters reach this record by another route.

        Returns:
            The class path and canonical parameters that identify this offer.
        """
        record = self.dump_config()
        return ConfigurationRecord(
            path=record.path, config=canonical_budget_config(record.config)
        )

    @cached_property
    def _identity(self) -> bytes:
        """Derive this frozen offer's canonical identity bytes once."""
        return canonical_json_bytes(self.identity_record().model_dump())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DPBudget):
            return NotImplemented
        return self._identity == other._identity

    def __hash__(self) -> int:
        return hash(self._identity)


@final
class PureDPBudget(DPBudget):
    """Represent a pure differential privacy budget."""

    eps: float = Field(
        description="Positive epsilon parameter; infinite removes the constraint."
    )

    @field_validator("eps")
    @classmethod
    def _validate_eps(cls, value: float) -> float:
        if not value > 0:
            raise ValueError("eps must be positive, either finite or infinite.")
        return canonical_budget_parameter(value)

    @property
    @override
    def is_unconstrained(self) -> bool:
        return self.eps == math.inf

    @override
    def split(self, weights: Sequence[float]) -> tuple[PureDPBudget, ...]:
        values = _validated_split_weights(weights)
        total = sum(values)
        # Constructed through the class, not model_copy: model_copy bypasses
        # field validators, which is how a split part would otherwise skip
        # canonicalization.
        return tuple(PureDPBudget(eps=self.eps * weight / total) for weight in values)


@final
class ApproxDPBudget(DPBudget):
    """Represent an approximate differential privacy budget.

    Delta 1 constrains nothing: every outcome distribution is already
    (0, 1)-indistinguishable from every other, so the offer is unconstrained
    independent of epsilon.
    """

    eps: float = Field(
        description="Positive epsilon parameter; infinite removes the constraint."
    )
    delta: float = Field(description="Failure probability, in (0, 1].")

    @field_validator("eps")
    @classmethod
    def _validate_eps(cls, value: float) -> float:
        if not value > 0:
            raise ValueError("eps must be positive, either finite or infinite.")
        return canonical_budget_parameter(value)

    @field_validator("delta")
    @classmethod
    def _validate_delta(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("delta must be in (0, 1].")
        return canonical_budget_parameter(value)

    @property
    @override
    def is_unconstrained(self) -> bool:
        return self.eps == math.inf or self.delta == 1.0

    @override
    def split(self, weights: Sequence[float]) -> tuple[ApproxDPBudget, ...]:
        values = _validated_split_weights(weights)
        total = sum(values)
        # A whole delta of 1 is vacuous: every part keeps delta 1 rather than
        # dividing into a constrained one, so an unconstrained offer always
        # splits into unconstrained parts. Every part is itself canonicalized
        # by the field validator on construction, so the parts' sum may
        # exceed the whole by rounding at BUDGET_IDENTITY_DIGITS significant
        # digits; that overspend is bounded at roughly 1e-12 relative and is
        # accepted, not corrected.
        divide_delta = self.delta != 1.0
        # Constructed through the class, not model_copy: see PureDPBudget.split.
        return tuple(
            ApproxDPBudget(
                eps=self.eps * weight / total,
                delta=self.delta * weight / total if divide_delta else 1.0,
            )
            for weight in values
        )


@final
class ZCDPBudget(DPBudget):
    """Represent a zero-concentrated differential privacy budget."""

    rho: float = Field(
        description="Positive rho parameter; infinite removes the constraint."
    )

    @field_validator("rho")
    @classmethod
    def _validate_rho(cls, value: float) -> float:
        if not value > 0:
            raise ValueError("rho must be positive, either finite or infinite.")
        return canonical_budget_parameter(value)

    @property
    @override
    def is_unconstrained(self) -> bool:
        return self.rho == math.inf

    @override
    def split(self, weights: Sequence[float]) -> tuple[ZCDPBudget, ...]:
        values = _validated_split_weights(weights)
        total = sum(values)
        # Constructed through the class, not model_copy: see PureDPBudget.split.
        return tuple(ZCDPBudget(rho=self.rho * weight / total) for weight in values)


@final
class GDPBudget(DPBudget):
    """Represent a Gaussian differential privacy budget."""

    mu: float = Field(
        description="Positive mu parameter; infinite removes the constraint."
    )

    @field_validator("mu")
    @classmethod
    def _validate_mu(cls, value: float) -> float:
        if not value > 0:
            raise ValueError("mu must be positive, either finite or infinite.")
        return canonical_budget_parameter(value)

    @property
    @override
    def is_unconstrained(self) -> bool:
        return self.mu == math.inf

    @override
    def split(self, weights: Sequence[float]) -> tuple[GDPBudget, ...]:
        values = _validated_split_weights(weights)
        total = sum(values)
        # Constructed through the class, not model_copy: see PureDPBudget.split.
        return tuple(
            GDPBudget(mu=self.mu * math.sqrt(weight / total)) for weight in values
        )


@final
class NoPrivacy(DPBudget):
    """Represent an explicitly unconstrained release."""

    @property
    @override
    def is_unconstrained(self) -> bool:
        return True

    @override
    def split(self, weights: Sequence[float]) -> tuple[Self, ...]:
        values = _validated_split_weights(weights)
        return tuple(self for _ in values)


NO_PRIVACY = NoPrivacy()
