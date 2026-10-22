"""Template task contract and value types for ``generate-task``."""

from dpstats.task import Task

from .types import TemplateTaskInput, TemplateTaskMetadata, TemplateTaskOutput

# Naming (STYLE.md): the Task id is the snake_case package name and the symbol is
# its CapWords form. Algorithm and Transformation classes are CapWords named for the
# scientific method, once, in the shortest recognizable form; they do not repeat
# the Task. A Metric's ``name`` is snake_case and equals its exported symbol.
TemplateTask = Task(
    id="template_task",
    name="Template Task",
    description="Privately estimate the quantity this new task defines.",
    metadata_cls=TemplateTaskMetadata,
    input_cls=TemplateTaskInput,
    output_cls=TemplateTaskOutput,
)

# To reuse an algorithm native to another task on this task, present it as a
# transformation composed by ``CompositeAlgorithm``; the built-in tasks' ``transformations/``
# subpackages show the pattern.
