"""Task authoring contracts and common typed NumPy array values.

Custom Task authors use these values to declare the public Metadata, private
Input, and Algorithm Output triple. The array aliases centralize Pydantic's
NumPy conversion, dimension checks, and JSON handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Generic, Literal

import numpy as np
from pydantic import Field, PlainSerializer, PlainValidator, field_validator
from typing_extensions import TypeVar as TypeVarWithDefault

from dpstats.parameters import FieldSymbol
from dpstats.transport.configuration import PolymorphicModel

_Shape = tuple[int, ...]
_DtypeFamily = Literal["floating", "integer", "boolean", "any"]
_Array = np.ndarray[_Shape, np.dtype[Any]]


def _validate_array(ndim: int | None, family: _DtypeFamily):
    def validate(value: Any) -> _Array:
        array: _Array = np.asarray(value)
        if family == "floating" and not np.issubdtype(array.dtype, np.floating):
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("expected an array of real numbers.")
            array = array.astype(float)

        if ndim is not None and array.ndim != ndim:
            raise ValueError(f"expected a {ndim}D array, got {array.ndim}D.")

        if family == "integer":
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"expected an integer array, got {array.dtype}.")
            # The evidence store persists this family as int64, and uint64 is the
            # only numpy integer dtype wider than that range.
            if array.size and int(array.max()) > np.iinfo(np.int64).max:
                raise ValueError(
                    "expected integers within the int64 range, got a value of "
                    f"{int(array.max())}"
                )
        if family == "boolean" and array.dtype != np.dtype(bool):
            raise ValueError(f"expected a boolean array, got {array.dtype}")

        return array

    return validate


_ARRAY_SERIALIZER = PlainSerializer(lambda array: array.tolist(), return_type=list)

Array1D = Annotated[
    np.ndarray[_Shape, np.dtype[np.floating[Any]]],
    PlainValidator(_validate_array(1, "floating")),
    _ARRAY_SERIALIZER,
]
"""A 1d float array."""
Array2D = Annotated[
    np.ndarray[_Shape, np.dtype[np.floating[Any]]],
    PlainValidator(_validate_array(2, "floating")),
    _ARRAY_SERIALIZER,
]
"""A 2d float array."""
_Array3D = Annotated[
    np.ndarray[_Shape, np.dtype[np.floating[Any]]],
    PlainValidator(_validate_array(3, "floating")),
    _ARRAY_SERIALIZER,
]
ArrayND = Annotated[
    np.ndarray[_Shape, np.dtype[np.floating[Any]]],
    PlainValidator(_validate_array(None, "floating")),
    _ARRAY_SERIALIZER,
]
"""A float array of any dimension."""
IntArray1D = Annotated[
    np.ndarray[_Shape, np.dtype[np.integer[Any]]],
    PlainValidator(_validate_array(1, "integer")),
    _ARRAY_SERIALIZER,
]
"""A 1d integer array."""
IntArray2D = Annotated[
    np.ndarray[_Shape, np.dtype[np.integer[Any]]],
    PlainValidator(_validate_array(2, "integer")),
    _ARRAY_SERIALIZER,
]
"""A 2d integer array."""
BoolArray1D = Annotated[
    np.ndarray[_Shape, np.dtype[np.bool_]],
    PlainValidator(_validate_array(1, "boolean")),
    _ARRAY_SERIALIZER,
]
"""A 1d boolean array."""
_BoolArray2D = Annotated[
    np.ndarray[_Shape, np.dtype[np.bool_]],
    PlainValidator(_validate_array(2, "boolean")),
    _ARRAY_SERIALIZER,
]
AnyArray1D = Annotated[
    np.ndarray[_Shape, np.dtype[Any]],
    PlainValidator(_validate_array(1, "any")),
    _ARRAY_SERIALIZER,
]
"""A 1d array of any dtype."""
_AnyArray2D = Annotated[
    np.ndarray[_Shape, np.dtype[Any]],
    PlainValidator(_validate_array(2, "any")),
    _ARRAY_SERIALIZER,
]
_AnyArrayND = Annotated[
    np.ndarray[_Shape, np.dtype[Any]],
    PlainValidator(_validate_array(None, "any")),
    _ARRAY_SERIALIZER,
]

# The dtype family and rank a columnar store types each fixed-rank alias by;
# ``ArrayND`` and the ``Any``-dtype aliases have no fixed column type.
ARRAY_COLUMN_SHAPES: dict[Any, tuple[_DtypeFamily, int]] = {
    Array1D: ("floating", 1),
    Array2D: ("floating", 2),
    _Array3D: ("floating", 3),
    IntArray1D: ("integer", 1),
    IntArray2D: ("integer", 2),
    BoolArray1D: ("boolean", 1),
    _BoolArray2D: ("boolean", 2),
}


class TaskMetadata(PolymorphicModel, ABC):
    """Base class for one task's metadata: the public information available to an algorithm.

    The sample size is public and is stored in the ``n_samples`` field.
    Task-specific subclasses add the rest of the task's public structure and
    validate the corresponding private input.
    """

    task: ClassVar[Task[TaskMetadata, TaskInput, TaskOutput]]
    n_samples: Annotated[int, FieldSymbol("n")] = Field(
        description="Number of samples."
    )

    @field_validator("n_samples")
    @classmethod
    def _validate_n_samples(cls, value: int) -> int:
        """Require every Task Instance to contain at least one sample."""
        if value < 1:
            raise ValueError("n_samples must be positive.")
        return value

    @abstractmethod
    def validate_input(self, input: TaskInput) -> None:
        """Validate that the input satisfies the constraints imposed by this metadata."""


class TaskInput(PolymorphicModel):
    """Base class for one Task's private input data."""


InputT = TypeVarWithDefault(
    "InputT", bound=TaskInput, covariant=True, default=TaskInput
)


class TaskOutput(PolymorphicModel):
    """Base class for a Task's output."""


MetadataT = TypeVarWithDefault(
    "MetadataT", bound=TaskMetadata, covariant=True, default=TaskMetadata
)
OutputT = TypeVarWithDefault(
    "OutputT", bound=TaskOutput, covariant=True, default=TaskOutput
)


class Task(Generic[MetadataT, InputT, OutputT]):
    """A statistical estimation task.

    Each task defines the classes of its metadata, input and output.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        metadata_cls: type[MetadataT],
        input_cls: type[InputT],
        output_cls: type[OutputT],
    ):
        """Construct one Task from its Metadata, Input, and Output classes.

        Args:
            id: String id for the task.
            name: Short human-readable name.
            description: Short description of the task.
            metadata_cls: The Metadata class.
            input_cls: The Input class.
            output_cls: The Output class.
        """
        self.id = id
        self.name = name
        self.description = description
        self.Metadata: type[MetadataT] = metadata_cls
        self.Input: type[InputT] = input_cls
        self.Output: type[OutputT] = output_cls
        metadata_cls.task = self

    def __repr__(self) -> str:
        return f"Task(id={self.id!r}, name={self.name!r})"


__all__ = [
    "Task",
    "TaskInput",
    "TaskMetadata",
    "TaskOutput",
    "AnyArray1D",
    "Array1D",
    "Array2D",
    "ArrayND",
    "BoolArray1D",
    "IntArray1D",
    "IntArray2D",
]
