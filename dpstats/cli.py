from typing import Callable

import argparse
import os
import sys
from difflib import get_close_matches

from dpstats.scripts import (
    fetch,
    generate_task,
    init,
    measure_sweep,
    get_sweep_status,
    get_sweep_coordinates,
    reconcile_sweep,
    run_sweep,
    submission,
)

COMMANDS: dict[str, Callable[..., None]] = {
    "init": init.main,
    "measure-sweep": measure_sweep.main,
    "fetch": fetch.main,
    "get-sweep-status": get_sweep_status.main,
    "get-sweep-coordinates": get_sweep_coordinates.main,
    "run-sweep": run_sweep.main,
    "reconcile-sweep": reconcile_sweep.main,
    "generate-task": generate_task.main,
    "prepare-submission": submission.prepare_main,
    "verify-submission": submission.verify_main,
}

# Each command's parser, reachable without executing the command, so the
# documentation build can render the CLI reference from the argument
# declarations themselves.
COMMAND_PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "init": init.parser,
    "measure-sweep": measure_sweep.parser,
    "fetch": fetch.parser,
    "get-sweep-status": get_sweep_status.parser,
    "get-sweep-coordinates": get_sweep_coordinates.parser,
    "run-sweep": run_sweep.parser,
    "reconcile-sweep": reconcile_sweep.parser,
    "generate-task": generate_task.parser,
    "prepare-submission": submission.prepare_parser,
    "verify-submission": submission.verify_parser,
}

COMMAND_DESCRIPTIONS: dict[str, str] = {
    "init": "Create <path>/local/, idempotently, as a fresh or existing workspace",
    "measure-sweep": "Measure the Results a sweep selects with one or more task metrics",
    "fetch": "Ensure a Distribution Release is installed at <workspace>/official-release/",
    "get-sweep-status": "Summarize selected Sweep coverage as JSON",
    "get-sweep-coordinates": "Export selected Sweep coordinates as CSV",
    "run-sweep": "Fill missing sweep cells and return all selected Results",
    "reconcile-sweep": "Record the cells a killed run never reported, executing none",
    "generate-task": "Scaffold a new task directory from the template",
    "prepare-submission": "Prepare one verifiable method-submission bundle",
    "verify-submission": "Verify one method-submission bundle",
}


def print_usage() -> None:
    """Print the usage line and the list of available commands."""
    print("Usage: dpstats <command> [<args>]\n")
    print("Run 'dpstats <command> --help' for command-specific options.\n")
    print("Available commands:")
    width = max(len(cmd) for cmd in COMMANDS)
    for cmd in COMMANDS:
        print(f"  {cmd:<{width}}  {COMMAND_DESCRIPTIONS[cmd]}")


def main():

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_usage()
        sys.exit(0)

    cmd, *args = sys.argv[1:]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        suggestions = get_close_matches(cmd, COMMANDS, n=1)
        if suggestions:
            print(f"Did you mean: {suggestions[0]}?")
        print()
        print_usage()
        sys.exit(1)

    # Make the user's current working directory importable so commands can load the
    # user's own task packages. Done here, at command-execution time, rather than as an
    # import side-effect of this module.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # CLI error boundary: user mistakes (bad config value, missing file) are
    # reported as one-line errors, not tracebacks. Programming errors still
    # propagate with a full traceback.
    try:
        COMMANDS[cmd](args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
