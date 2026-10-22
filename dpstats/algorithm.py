"""Author and compose configured algorithms for one task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from pydantic import field_validator, model_validator

from typing_extensions import override

from dpstats.transport.configuration import (
    IDENTITY_CONTEXT,
    STRUCTURAL_COMPARISON_CONTEXT,
    PolymorphicModel,
    identity_payload,
)
from dpstats.parameters import (
    ParameterInfo,
    inspect_parameters,
)
from dpstats.budgets import DPBudget
from dpstats.exceptions import InvalidRegime, NoRelease, NotApplicable
from dpstats.transport.canonical import algorithm_identity
from dpstats.transport.registry import (
    load_entries,
    partition_verified_entries,
)
from dpstats.workspace.registries import file_registry_entry
from dpstats.utils import split_seed
from dpstats.workspace.layout import task_collection_directory

if TYPE_CHECKING:
    from dpstats.task import Task

TMetadata = Any
TInput = Any
TOutput = Any


class PrivacyInterpretation(str, Enum):
    """How to interpret the privacy guarantees of an Algorithm or Transformation.

    In this library, we adopt the add/remove differential privacy definition.
    Not all algorithms satisfy this definition, and some satisfy it only in a theoretical sense.
    Additionally, the library includes nonprivate algorithms purely for comparison purposes, which should not be interpreted
    as providing any privacy guarantees.
    To account for all of this, we have the following taxonomy of privacy interpretations:

    - ``STANDARD``: the standard add/remove differential privacy guarantee holds on all inputs and budgets.
    - ``QUALIFIED``: some other privacy guarantee holds, such as replace adjacency.
    - ``THEORETICAL``: a theoretical version of the method satisfies the guarantee but the implementation does not.
    - ``INFORMAL``: the mechanism pursues privacy-like behavior without any formal guarantees.
    - ``DIAGNOSTIC``: the algorithm knowingly includes an unaccounted or privacy-invalid component for comparison.
    - ``NONPRIVATE``: an explicit control, unnoised output, or
      ``NO_PRIVACY`` execution.

    The default privacy interpretation is ``STANDARD`` unless the algorithm or transformation declares a different one in its
    ``PRIVACY_INTERPRETATION``. A CompositeAlgorithm takes the weakest privacy interpretation among its transformation and algorithm components.
    """

    STANDARD = "standard"
    QUALIFIED = "qualified"
    THEORETICAL = "theoretical"
    INFORMAL = "informal"
    DIAGNOSTIC = "diagnostic"
    NONPRIVATE = "nonprivate"


def select_weakest_privacy_interpretation(
    interpretations: Sequence["PrivacyInterpretation"],
) -> "PrivacyInterpretation":
    """Return the weakest of ``interpretations`` according to declaration order.

    Args:
        interpretations: The privacy interpretations of the parts of one
            composed release.

    Returns:
        The weakest of them, the one declared last in ``PrivacyInterpretation``.

    Example:
        >>> select_weakest_privacy_interpretation(
        ...     [PrivacyInterpretation.STANDARD, PrivacyInterpretation.INFORMAL]
        ... )
        <PrivacyInterpretation.INFORMAL: 'informal'>
    """
    return max(interpretations, key=tuple(PrivacyInterpretation).index)


class BaseAlgorithm(PolymorphicModel, ABC):
    """Abstract class for implementing algorithms.

    Every algorithm in this library is a subclass of BaseAlgorithm.
    Every algorithm subclass defines the `run` method for running the algorithm on a task instance
    and defines hyperparameters as parameter_field attributes.
    """

    TASK: ClassVar[Task]
    """The task this algorithm is native to."""
    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.STANDARD
    )
    """The privacy interpretation for this algorithm."""

    label: str | None = None

    @property
    def task(self) -> Task:
        """Return the task this configured algorithm runs on."""
        return self.TASK

    def get_core_algorithm(self) -> BaseAlgorithm:
        """Return the core algorithm: the ``algorithm`` attribute of a CompositeAlgorithm, and ``self`` otherwise.

        This removes only a transformation composition, not tuning, boosting, or
        other algorithm internals. Keep the original algorithm for execution,
        Result identity, and privacy interpretation: its core may run on a
        different task and does not represent the full configured treatment.

        Returns:
            This object when it is not a composition; otherwise the exact
            configured algorithm held by the composition, not its class.

        Example:
            >>> from dpstats.tasks.linear_regression.algorithms import OLS
            >>> algorithm = OLS()
            >>> algorithm.get_core_algorithm() is algorithm
            True
        """
        return self

    @cached_property
    def uuid(self) -> str:
        """Return this configured algorithm's content identity.

        The model is frozen, so this content address is derived on first read
        and reused for the life of the value. Nested configuration must not be
        mutated afterwards, which the frozen contract already forbids. The
        identity covers the portable configuration only, so this algorithm's
        label never changes it.
        """
        return algorithm_identity(identity_payload(self, exclude={"label"}))

    @cached_property
    def _comparison_record(self) -> dict[str, Any]:
        """Derive live structural equality without an importable class path."""
        excluded = frozenset({"label"})
        return self.model_dump(
            mode="python",
            exclude=set(excluded),
            exclude_computed_fields=True,
            context={
                IDENTITY_CONTEXT: excluded,
                STRUCTURAL_COMPARISON_CONTEXT: True,
            },
        )

    def __eq__(self, other: object) -> bool:
        """Compare configured values without their labels."""
        if self is other:
            return True
        if not isinstance(other, BaseAlgorithm) or type(self) is not type(other):
            return NotImplemented
        return self._comparison_record == other._comparison_record

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __str__(self) -> str:
        """Return the label, or a name derived from the configuration.

        An unlabelled algorithm prints as its class followed by the configured
        fields that depart from their declared defaults: ``Ridge`` for the
        default configuration and ``Ridge[regularization_multiplier=0.5]`` for
        a variant. A field with no default always appears. Identity remains the
        content address.
        """
        if self.label:
            return self.label
        variants = [
            f"{name}={getattr(self, name)!r}"
            for name, parameter in self.parameter_info().items()
            if parameter.required or getattr(self, name) != parameter.default
        ]
        if not variants:
            return type(self).__name__
        return f"{type(self).__name__}[{', '.join(variants)}]"

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("BaseAlgorithm.label must be nonempty when supplied.")
        return value

    @classmethod
    def parameter_info(cls) -> dict[str, ParameterInfo]:
        """Describe configured fields and their optional platform policies.

        The dictionary displays readable text in a terminal and a field table
        in notebooks. Indexing returns ParameterInfo records; ordinary
        dictionary operations remain available.
        """
        return inspect_parameters(cls, BaseAlgorithm)

    @classmethod
    def default_tuning_grid(cls, n_points: int = 7) -> dict[str, list[Any]]:
        """Discretize the class's recommended data-independent search space."""
        # Imported lazily: configure imports algorithm for CompositeAlgorithm
        # and BaseAlgorithm, so a module-level import here would cycle.
        from dpstats.configure import (
            _build_tuning_grid,  # pyright: ignore[reportPrivateUsage]
        )

        return _build_tuning_grid(cls.parameter_info(), n_points=n_points)

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        """Return the reading of this component's output.

        ``STANDARD`` unless the subclass declares ``PRIVACY_INTERPRETATION``
        otherwise: an author declares it only where the output is not
        strictly private.
        """
        return type(self).PRIVACY_INTERPRETATION

    @abstractmethod
    def run(
        self,
        metadata: TMetadata,
        input: TInput,
        budget: DPBudget,
        seed: int,
    ) -> TOutput:
        """Run the algorithm on one instance in ``self.task`` format."""


class Transformation(PolymorphicModel, ABC):
    """Transform an instance into another, possibly consuming some of the privacy budget.

    A transformation converts one problem instance (consisting of metadata, input, and budget) into another,
    possibly of a different task. Transformations are applied sequentially in CompositeAlgorithm to prepare the problem for its core algorithm.
    To implement a Transformation, define the OUTER_TASK and INNER_TASK class variables, which specify the tasks whose instances the transformation transforms from and to, respectively.
    Then implement the prepare method, which transforms the outer task instance into an inner task instance along with optional restoration context.
    Finally, implement the restore method, which transforms the inner task output back into an outer task output using the restoration context.

    ``prepare`` refuses a problem it cannot serve from public metadata alone.
    A limit that makes the whole cell unanswerable — a budget family the stage
    cannot realize, a missing bound its analysis needs, a privacy-providing
    stage with nothing to spend on — raises :class:`InvalidRegime`. A
    structural limit that merely leaves this stage nothing to do raises
    :class:`NotApplicable`, which a composition passes through; raise it only
    when omitting the stage is an identity on problem, budget and seed. A
    refusal reaching a composition after an earlier stage spent budget is
    that composition's accounted bottom, :class:`NoRelease`: the metadata the
    stage refused is the earlier stage's release, so the refusal is
    post-processing of it.
    """

    OUTER_TASK: ClassVar[Task]
    """The task this transformation accepts from its caller."""
    INNER_TASK: ClassVar[Task]
    """The task this transformation transports its caller into."""
    PRIVACY_INTERPRETATION: ClassVar[PrivacyInterpretation] = (
        PrivacyInterpretation.STANDARD
    )
    """The privacy interpretation for this transformation."""

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        """Return the privacy interpretation for this transformation."""
        return type(self).PRIVACY_INTERPRETATION

    @abstractmethod
    def prepare(
        self,
        metadata: Any,
        input: Any,
        budget: DPBudget,
        seed: int,
    ) -> tuple[Any, Any, DPBudget, int, Any]:
        """Construct the inner task problem and restoration state."""

    @abstractmethod
    def restore(self, output: Any, restoration: Any) -> Any:
        """Map an inner-task output back to the outer-task output format."""


class CompositeAlgorithm(BaseAlgorithm):
    """Run a chain of transformations then run a configured algorithm.

    This algorithm runs a sequence of transformations on a task instance,
    then runs an algorithm on the transformed instance, and finally
    restores the algorithm's output back to the original task's output format
    using the transformations' restore method.
    """

    transformations: tuple[Transformation, ...]
    algorithm: BaseAlgorithm

    @override
    def get_core_algorithm(self) -> BaseAlgorithm:
        return self.algorithm

    @model_validator(mode="after")
    def _validate_composition(self) -> "CompositeAlgorithm":
        if not self.transformations:
            raise ValueError(
                "CompositeAlgorithm requires at least one transformation; the "
                "algorithm alone already represents that configured algorithm."
            )

        transformations = self.transformations
        algorithm = self.algorithm
        while isinstance(algorithm, CompositeAlgorithm):
            transformations = transformations + algorithm.transformations
            algorithm = algorithm.algorithm
        object.__setattr__(self, "transformations", transformations)
        object.__setattr__(self, "algorithm", algorithm)

        for earlier, later in zip(self.transformations, self.transformations[1:]):
            if earlier.INNER_TASK is not later.OUTER_TASK:
                raise ValueError(
                    f"{type(earlier).__name__} prepares {earlier.INNER_TASK.name}, "
                    f"but {type(later).__name__} expects {later.OUTER_TASK.name}."
                )
        prepared_task = self.transformations[-1].INNER_TASK
        if self.algorithm.task is not prepared_task:
            raise ValueError(
                f"{type(self.algorithm).__name__} is native to "
                f"{self.algorithm.task.name}, but the chain prepares "
                f"{prepared_task.name}."
            )
        return self

    @property
    def task(self) -> Task:
        """Return the task this algorithm accepts."""
        return self.transformations[0].OUTER_TASK

    def __str__(self) -> str:
        """Return the label, or the chain and the child joined by ``" + "``.

        An unlabelled composition prints as, for example,
        ``ApproxBounds + Standardize + AdaSSP``.
        """
        if self.label:
            return self.label
        stages = [
            type(transformation).__name__ for transformation in self.transformations
        ]
        return " + ".join([*stages, str(self.algorithm)])

    @property
    def privacy_interpretation(self) -> PrivacyInterpretation:
        interpretations = [
            transformation.privacy_interpretation
            for transformation in self.transformations
        ]
        interpretations.append(self.algorithm.privacy_interpretation)
        return select_weakest_privacy_interpretation(interpretations)

    def process(
        self,
        metadata: Any,
        input: Any,
        budget: DPBudget,
        seed: int,
    ) -> tuple[Any, Any, DPBudget, int, tuple[tuple[Transformation, Any], ...]]:
        """Prepare the exact child call and retain its restoration values.

        A stage raising :class:`NotApplicable` on this problem is passed
        through unchanged and contributes no restoration, so :meth:`restore`
        omits it. Only a stage whose inner task is its outer task may be
        omitted; a transporting stage's refusal propagates.

        Raises:
            NoRelease: If a stage refuses with :class:`InvalidRegime` after an
                earlier stage reduced the offered budget. The refusal depends
                only on that stage's released metadata and public
                configuration, so it is post-processing covered by the budget
                already spent; the message names the stage and its reason.
        """
        offered = budget
        restorations: list[tuple[Transformation, Any]] = []
        for transformation in self.transformations:
            try:
                metadata, input, child_budget, seed, restoration = (
                    transformation.prepare(metadata, input, budget, seed)
                )
            except InvalidRegime as error:
                if (
                    isinstance(error, NotApplicable)
                    and transformation.OUTER_TASK is transformation.INNER_TASK
                ):
                    continue
                if budget != offered:
                    raise NoRelease(
                        f"{type(transformation).__name__} refused its regime "
                        f"after an earlier stage spent budget: {error}"
                    ) from error
                raise
            if metadata.task is not transformation.INNER_TASK:
                raise ValueError(
                    f"{type(transformation).__name__} returned metadata for "
                    f"{metadata.task.name}, not its declared inner task "
                    f"{transformation.INNER_TASK.name}."
                )
            metadata.validate_input(input)
            restorations.append((transformation, restoration))
            budget = child_budget

        return metadata, input, budget, seed, tuple(restorations)

    def restore(
        self,
        output: Any,
        restorations: tuple[tuple[Transformation, Any], ...],
    ) -> Any:
        """Restore one child output through the prepared transformations."""
        for transformation, restoration in reversed(restorations):
            output = transformation.restore(output, restoration)
        return output

    @override
    def run(
        self,
        metadata: Any,
        input: Any,
        budget: DPBudget,
        seed: int,
    ) -> Any:
        offered = budget
        metadata, input, budget, seed, restorations = self.process(
            metadata, input, budget, seed
        )

        try:
            output = self.algorithm.run(metadata, input, budget=budget, seed=seed)
        except InvalidRegime as error:
            if budget != offered:
                raise NoRelease(
                    f"{type(self.algorithm).__name__} refused its regime after "
                    f"an earlier stage spent budget: {error}"
                ) from error
            raise
        return self.restore(output, restorations)


def register_algorithm(
    task_collections_root: str | Path,
    algorithm: BaseAlgorithm,
    *,
    registry: str = "default",
) -> str:
    """Register a configured algorithm in a named registry group."""
    file_registry_entry(
        task_collection_directory(task_collections_root, algorithm.task.id),
        task_id=algorithm.task.id,
        kind="algorithm",
        uuid=algorithm.uuid,
        record=algorithm.dump_config().model_dump(),
        registry=registry,
    )
    return algorithm.uuid


def load_algorithms(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    strict: bool = True,
    sources: Mapping[str, Path] | None = None,
) -> dict[str, BaseAlgorithm]:
    """Rebuild the configured Algorithms one registry's entries name.

    The rebuilt value's own ``task`` settles which task it is native to. An
    entry this library cannot rebuild refuses the group; ``strict=False`` skips
    it with one warning naming it instead. ``sources`` maps an entry to the
    registry file holding it, so a refusal or warning names that file.
    """
    return load_entries(
        entries,
        kind="algorithm",
        task_id=task_id,
        strict=strict,
        reconstruct=BaseAlgorithm.load_record,
        derive_uuid=lambda algorithm: algorithm.uuid,
        native_task_id=lambda algorithm: algorithm.task.id,
        sources=sources,
    )


def load_algorithms_partitioned(
    entries: Mapping[str, dict[str, Any]],
    *,
    task_id: str,
    sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, BaseAlgorithm], dict[str, str]]:
    """Load valid Algorithms and every rejected key's stated cause."""
    return partition_verified_entries(
        entries,
        task_id=task_id,
        reconstruct=BaseAlgorithm.load_record,
        derive_uuid=lambda algorithm: algorithm.uuid,
        native_task_id=lambda algorithm: algorithm.task.id,
        sources=sources,
    )


__all__ = [
    "BaseAlgorithm",
    "CompositeAlgorithm",
    "PrivacyInterpretation",
    "Transformation",
    "split_seed",
    "select_weakest_privacy_interpretation",
]
