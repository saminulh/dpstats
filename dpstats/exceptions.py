"""Non-release outcomes an extension signals to the execution engine.

Every ``PolymorphicModel`` extension kind — algorithms, transformations, budgets
and budget policies, domains, task types, losses — signals a non-release
outcome by raising one of these types, and execution classifies each cell by
them: :class:`NoRelease` is an accounted bottom release, :class:`InvalidRegime`
a declared public limit, and :class:`NotApplicable` a stage the public problem
gives nothing to do. Any other exception is an execution failure.
"""

from __future__ import annotations

__all__ = [
    "InvalidRegime",
    "NoRelease",
    "NotApplicable",
]


class InvalidRegime(ValueError):
    """Refuse a cell on a publicly discernible limit this algorithm declares.

    Raising this is a declared limit, not a crash: collection records a
    terminal ``invalid_regime`` result retaining the message. That result
    tells the reader no budget was spent, and it is never retried, because
    the refusal depends on public information alone.

    Requires:
        - Both the refusal event and its message depend only on public
          information: the metadata the stage is handed, its configuration
          and the offered budget. No privacy budget has been spent. Internal
          Input access is not itself a privacy expenditure; the mechanism's
          analysis establishes that its observable refusal is public-only.
        - An authored message, exactly as a ``NoRelease`` message is: written
          deliberately at the raise site, naming the family mismatch in public
          terms, never composed from values that could carry a private
          quantity.

    Implementation notes:
        - Prefer public checks before Input access when straightforward. This
          is a design preference, not a condition of this outcome; a caller
          need not duplicate a child's check before preparing its input.
        - The limit is an offered budget family admitting no implication the
          algorithm realizes, a public domain lacking a shape or bound the
          analysis requires, or another failed public precondition.
        - Unexpected execution failure is not a regime limit; it raises an
          ordinary exception and is recorded as ``algorithm_error`` with its
          type name alone.
        - A composition converts a refusal after an earlier stage spent
          budget into :class:`NoRelease`, recorded as ``no_release`` with the
          stage's message: the metadata that stage refused is the earlier
          stage's release, so the refusal is post-processing covered by the
          budget already spent. A tuner instead converts a candidate's
          refusal after an earlier candidate spent budget into
          :class:`RuntimeError`, recorded as ``algorithm_error``: every
          candidate sees the same public metadata, so that refusal was
          knowable before any spend.
        - This class extends :class:`ValueError`, so a caller refusing
          invalid offers generically still refuses a declared regime limit.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NoRelease(Exception):
    """Raise the accounted, intentional bottom outcome from an algorithm.

    Raising this is a release, not a crash. Bottom is an outcome the mechanism
    defines, and the algorithm's charged privacy budget must cover it.
    Consumers may therefore observe and post-process it freely: collection
    records a ``no_release`` result and DP selection tuners score a bottom
    candidate ``+inf`` without another privacy charge.

    Raise it only for that intentional, accounted bottom: the mechanism reached
    a branch its privacy analysis covers as a release, and the raise site states
    that argument. A Propose-Test-Release check declining to release is the
    canonical case.

    Unsupported regimes and unexpected execution failure are not bottom.
    A budget family, domain shape, or other public precondition this algorithm
    declares no realization for raises :class:`InvalidRegime`; a configuration
    malformed in itself raises :class:`ValueError` at construction. An
    unaccounted numerical or solver failure raises :class:`RuntimeError` unless
    the mechanism explicitly defines and accounts for bottom there. An unexpected
    failure of an input-dependent precondition is an execution error with no
    privacy guarantee, so its reason stays at the raise site and does not
    travel with the recorded outcome. An intended data-dependent refusal is a
    mechanism branch and needs a joint privacy argument; changing its exception
    type does not supply one. Information source alone does not decide the
    class: bottom must be an outcome the mechanism intends and charges for.

    The persisted message may contain only public quantities or values already
    covered by the mechanism's release; never interpolate an unaccounted private
    statistic.
    """

    def __init__(self, message: str = "No result released") -> None:
        super().__init__(message)


class NotApplicable(InvalidRegime):
    """Refuse a stage the public problem gives nothing to do.

    Structural inapplicability — no eligible nuisance block, no selectable
    column — rather than a budget family, a bound or a sample size the
    mechanism cannot realize. The authored-message and
    public-information contracts of :class:`InvalidRegime` hold unchanged.

    A :class:`CompositeAlgorithm` passes a same-task stage raising this
    through: the chain advances with the metadata, input, budget and seed it
    already had, and the stage leaves no restoration behind. Raise it only
    where that pass-through is an identity on all four. A stage transporting
    one task into another cannot be omitted, so the composition re-raises it,
    and a transformation used on its own refuses exactly as its base class does.
    A re-raised refusal after an earlier stage spent budget is the
    composition's :class:`NoRelease`, as for :class:`InvalidRegime`.
    """
