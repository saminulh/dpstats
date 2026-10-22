#!/usr/bin/env python3
"""Scaffold a new task package from the bundled template."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path


TASK_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


def generate_task(task_name: str, task_dir: Path):
    """Scaffold a new task directory from the template."""
    if not TASK_ID.fullmatch(task_name):
        print(
            f"Error: Task name {task_name!r} must be snake_case: lowercase words "
            "joined by single underscores, naming the estimand (for example "
            "'median_estimation')."
        )
        sys.exit(1)
    task_dir = task_dir.expanduser().resolve()
    target_dir = task_dir / task_name

    if target_dir.exists():
        print(f"Error: Task directory '{target_dir}' already exists.")
        sys.exit(1)

    template_resource = resources.files("dpstats").joinpath("templates/template_task")
    with resources.as_file(template_resource) as template_dir:
        shutil.copytree(
            template_dir,
            target_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    for file_path in target_dir.rglob("*"):
        if file_path.is_file():
            content = file_path.read_text()
            content = content.replace(
                "TemplateTask", task_name.title().replace("_", "")
            )
            content = content.replace(
                "Template Task", task_name.replace("_", " ").title()
            )
            content = content.replace("template_task", task_name)
            lines = content.splitlines()
            filtered_lines = [
                line for line in lines if not line.strip().startswith("from templates.")
            ]
            file_path.write_text("\n".join(filtered_lines))
            if "template_task" in file_path.name:
                new_name = file_path.name.replace("template_task", task_name)
                file_path.rename(file_path.parent / new_name)

    print(f"✅ New task '{task_name}' created at {target_dir}")


def parser() -> argparse.ArgumentParser:
    """Build the ``generate-task`` command's argument parser."""
    command = argparse.ArgumentParser(
        description="Generate a new dpstats task from the template."
    )
    command.add_argument(
        "--task-name",
        required=True,
        help=(
            "Name of the new task: snake_case, naming the estimand; it becomes the "
            "package name, the Task id, and (as CapWords) the Task symbol. "
            "Algorithm classes are CapWords named for the method, not the task "
            "(required)."
        ),
    )
    command.add_argument(
        "--task-dir",
        default=Path.cwd(),
        help="Directory where the new task folder will be created (path; default: current directory)",
    )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``generate-task`` command."""
    parsed = parser().parse_args(argv)

    generate_task(parsed.task_name, Path(parsed.task_dir))
