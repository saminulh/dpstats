"""Unit tests for dpstats.task.Task."""

from dpstats.task import (
    Task,
    TaskInput,
    TaskMetadata,
    TaskOutput,
)


class _Metadata(TaskMetadata):
    n: int
    d: int

    def validate_input(self, input: TaskInput) -> None:
        _Input.model_validate(input)


class _Input(TaskInput):
    value: int


class _Output(TaskOutput):
    value: int


def test_constructing_a_task_binds_its_metadata_class_to_it() -> None:
    task = Task(
        id="simple_test_task",
        name="Simple Test Task",
        description="A simple task for unit testing",
        metadata_cls=_Metadata,
        input_cls=_Input,
        output_cls=_Output,
    )

    assert getattr(_Metadata, "task") is task
