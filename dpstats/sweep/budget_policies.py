"""Configured policies that offer one privacy budget per Instance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Context, Decimal

from pydantic import Field

from typing_extensions import override

from dpstats.transport.configuration import PolymorphicModel
from dpstats.budgets import ApproxDPBudget, DPBudget
from dpstats.budgets.families import canonical_budget_parameter
from dpstats.task import TaskMetadata

# Decimal128 working precision: exact integer arithmetic at far more digits
# than a budget identity keeps, so a resolved parameter cannot depend on the
# host's floating-point library.
_RESOLUTION_CONTEXT = Context(prec=34)


class BudgetPolicy(PolymorphicModel, ABC):
    """Base class for a budget an authored Sweep states as a rule.

    Implementation notes:
        A Sweep declares a policy wherever it declares a concrete budget, and
        every Sweep operation resolves it against each installed Instance. The
        Sweep pins the policy class and its parameters, so the rule that produced
        a cell's budget is part of the authored document rather than of the
        invocation.

        A rule sees one Instance's public Metadata and nothing else: a privacy
        budget derived from private input would not be a privacy budget.
    """

    @abstractmethod
    def resolve(self, metadata: TaskMetadata) -> DPBudget:
        """Return the budget this policy offers for one Instance's Metadata."""


class SubinverseSampleSizeDelta(BudgetPolicy):
    """Offer approximate DP with a delta below the inverse sample size.

    Implementation notes:
        The delta is ``min(n_samples ** -1.1, 0.01)``. An exponent above one keeps
        delta asymptotically smaller than ``1 / n_samples``, the scale at which a
        mechanism could publish one randomly chosen record outright and still
        satisfy the definition. The ceiling bounds the smallest Instances.

        The power is taken in decimal at a fixed precision and canonicalized
        from that exact Decimal before construction: rounding the Decimal to
        a plain ``float`` first can cross the half-even boundary the
        canonical value itself would land on, giving a different last digit
        than canonicalizing the Decimal directly. ``ApproxDPBudget``'s own
        validator then canonicalizes that already-canonical float, a no-op,
        so the offered delta is the same number wherever the Sweep is
        resolved.
    """

    eps: float = Field(description="Positive epsilon offered at every sample size.")

    @override
    def resolve(self, metadata: TaskMetadata) -> ApproxDPBudget:
        delta = _RESOLUTION_CONTEXT.power(Decimal(metadata.n_samples), Decimal("-1.1"))
        return ApproxDPBudget(
            eps=self.eps,
            delta=canonical_budget_parameter(min(delta, Decimal("0.01"))),
        )
