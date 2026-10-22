"""Execute the site-selected notebooks from a built wheel in a fresh interpreter.

The tutorials are the library's user-facing journeys, so they have to keep
working against the *installed* package rather than against this checkout. This
module builds the wheel, installs it non-editably into a fresh interpreter, and
executes each notebook there with ``nbclient``, in the site's lesson order with
fresh kernels and one shared working directory outside the repository. A sibling
script the notebook names is copied beside it. Each notebook's
``result``-tagged cells carry its success criterion.

The execution tests are marked ``slow`` (wheel build, fresh install, real
mechanisms): run them with ``pytest tests/integration/test_tutorials.py -m slow
-n 0``. The source-only, portability, and public-import checks are ordinary
tests and run in the default gate.

A notebook that needs a real benchmark release tags its fetch cell
``fetch-guard``. When that cell is the first to error (the release is private
or unreachable rather than the tutorial itself being broken), the run is
reported as a clean skip with the fetch error as the reason, not a failure.
Every notebook runs as an anonymous reader: no Hub token and no cached
credentials. The copies under ``_executed/`` therefore show exactly
what a reader without access sees, and a private release becomes the clean
skip above rather than published outputs. With ``DOCS_LOCAL_RELEASE_ROOT`` set,
the harness pre-installs that local cut as each notebook's release and writes
to the gitignored ``_executed_local/`` instead, which only a site build that
reads a local release renders.

This harness is also the site's one notebook executor. Each notebook that runs
to completion is written to the gitignored ``tutorials/_executed/``, keyed
by the source notebook's content hash, after its outputs are held against the
same portability rules the sources obey. The site renders those copies and
never executes a notebook; a missing or stale one stops the site build by name.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple, cast

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIRECTORY = REPOSITORY_ROOT / "tutorials"

sys.path.insert(0, str(REPOSITORY_ROOT / "docs" / "site"))

from site_links import PUBLIC_SITE_URL  # noqa: E402
from tutorials import (  # noqa: E402
    EXECUTED_DIRNAME,
    GUARDED_SKIP_SUFFIX,
    LOCAL_EXECUTED_DIRNAME,
    SKIP_REASON_KEY,
    SOURCE_DIGEST_KEY,
    companion_scripts,
    published_tutorials,
    tutorial_sources,
)
from dpstats.workspace.layout import workspace_release_root

_tutorial_sources = tutorial_sources(TUTORIALS_DIRECTORY)
TUTORIALS = published_tutorials(REPOSITORY_ROOT / "docs", TUTORIALS_DIRECTORY)

KERNEL_NAME = "dp-tutorials"
CELL_TIMEOUT = 900  # seconds per notebook cell

# When set, this harness pre-installs that tree as each notebook's own
# `official-release/`, so the fetch cell never reaches the Hub, and writes the
# copies under the local-release directory instead of `_executed/`.
DOCS_LOCAL_RELEASE_ROOT_ENV = "DOCS_LOCAL_RELEASE_ROOT"
LOCAL_RELEASE_ROOT = os.environ.get(DOCS_LOCAL_RELEASE_ROOT_ENV)

# Where this harness writes the executed copy the site renders. The site module
# imported above owns the directory names and the metadata key.
EXECUTED_DIRECTORY = TUTORIALS_DIRECTORY / (
    LOCAL_EXECUTED_DIRNAME if LOCAL_RELEASE_ROOT else EXECUTED_DIRNAME
)

# What one notebook may spend. A notebook over either figure is an escalation
# to the maintainer, not a reason to move these lines.
WALL_BUDGET_SECONDS = 120.0
MEMORY_BUDGET_BYTES = 4 * 1024 * 1024 * 1024

# What one notebook's outputs may weigh on the page that carries them: a plot
# or a frame is fine, an embedded dataset is not.
OUTPUT_BYTES_CEILING = 512 * 1024
NOTEBOOK_OUTPUT_BYTES_CEILING = 4 * 1024 * 1024

# Execution-derived state that may not appear in a committed notebook, mirroring
# the workspace source-only rules for authored notebooks.
CELL_EXECUTION_METADATA = frozenset(
    {"ExecuteTime", "collapsed", "execution", "scrolled", "trusted"}
)
# Removes outputs, execution state and the kernel and interpreter an editor
# records, keeping cell ids so a commit shows only the cells that changed.
STRIP_COMMAND = (
    'nbstripout --keep-id --extra-keys "metadata.kernelspec metadata.language_info" '
    "tutorials/*.ipynb"
)

# Substrings each notebook's ``result``-tagged cells must print. They are row and
# column labels the notebook itself chooses, so they assert that the journey
# reached its answer without pinning a noisy numerical value.
SUCCESS_CRITERIA: dict[str, tuple[str, ...]] = {
    "tasks.ipynb": (
        "first input: valid",
        "second input: MeanEstimation X has shape (4, 1)",
        "third input: MeanEstimation X leaves its declared domain.",
        "output is a MeanEstimationOutput object: True",
        "mean_est:",
        "output1 error:",
        "output2 error:",
    ),
    "algorithms.ipynb": (
        "GaussianMechanism",
        "CoinPress",
        "num_iters",
        "num_iters=4",
        "InvalidRegime:",
    ),
    "metrics_and_baselines.ipynb": (
        "standard error of the sample mean:",
        "trials: 360",
        "l2_error_to_sample_mean reads",
        "max_coordinate_error",
    ),
    "domains.ipynb": ("n_samples=4", "original noise scale:", "wider noise scale:"),
    "privacy_budgets_and_accounting.ipynb": (
        "InvalidRegime:",
        "GDP budget available to the mechanism:",
        "split budgets:",
    ),
    "install_and_fetch.ipynb": (
        "manifest",
        "SCE release:",
        "seed 0 (success):",
        "seed 1 (success):",
        "seed 2 (success):",
    ),
    "run_your_own_script.ipynb": ("ridge", "attempts, statuses"),
    "extend_with_transformations.ipynb": (
        "same estimate: True",
        "InvalidRegime:",
        "trials: 360",
        "ApproxBounds + ToMEstimation + DPSGD",
    ),
    "compare_on_the_benchmark.ipynb": (
        "invalid_regime",
        "absolute_error_to_ols",
    ),
    "tuning_headroom.ipynb": (
        "algorithm_label",
        "shared choice",
        "per-instance oracle",
        "oracle on its own trials",
    ),
    "ssp_hard_instances.ipynb": (
        "SSP, scaled",
        "condition number",
        "n / d",
    ),
}

# Machine- and workspace-specific spellings that must not reach a tutorial: a
# reader's checkout has none of them.
UNPORTABLE_TEXT = ("/Users/", "/home/", "benchmark-storage", "custom_code")
# A pasted hub read token: the private-release notebook has a slot for one.
HUB_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")

# The absolute paths an execution prints although the notebook names none: the
# temporary directory it runs in, the release it installs there, and this
# machine's home. They are this run's, not a reader's, so they never reach a
# page.
MACHINE_PATH = re.compile(
    r"/(?:private/)?(?:Users|home|tmp|var/folders)/[^\s'\"<>)\]]*"
)
REDACTED_PATH = "/.../"
REDACTED_TOKEN = "<token>"
# The Hub's per-request identifier in a fetch error, which would make every
# run rewrite an otherwise unchanged skip record.
HUB_REQUEST_ID = re.compile(r" \(Request ID: [^)]*\)")

# The output fields that carry text a reader sees.
TEXT_OUTPUT_KEYS = ("text", "ename", "evalue")
TEXT_MIME_TYPES = ("text/plain", "text/html", "text/markdown", "text/latex")


class JourneyEnvironment(NamedTuple):
    """The fresh interpreter one journey run executes in, and how it was built."""

    root: Path
    interpreter: Path
    kernels: Path
    wheel: str
    wheel_sha256: str
    install_seconds: float


def _notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_tutorial_has_one_position_in_the_site_journey() -> None:
    """Every notebook in ``tutorials/`` is published, and every script serves one."""
    assert len(TUTORIALS) == len(set(TUTORIALS))
    assert set(_tutorial_sources) == {path.name for path in TUTORIALS}
    served = {script for path in TUTORIALS for script in companion_scripts(path)}
    assert set(TUTORIALS_DIRECTORY.glob("*.py")) == served


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


FETCH_GUARD_TAG = "fetch-guard"

# The Hub credentials a maintainer's shell may carry. The kernel runs without
# them, and with a fresh Hub home, so a token file on this machine cannot
# reach a notebook either.
HUB_CREDENTIAL_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")
_WORKSPACE_DIRECTORY_NAME = "dp-benchmark"


def _preinstall_local_release(workdir: Path) -> None:
    """Symlink ``DOCS_LOCAL_RELEASE_ROOT`` in as the notebook's installed release."""
    if not LOCAL_RELEASE_ROOT:
        return
    workspace = workdir / _WORKSPACE_DIRECTORY_NAME
    workspace.mkdir(parents=True, exist_ok=True)
    release = workspace_release_root(workspace)
    if not release.exists():
        release.symlink_to(Path(LOCAL_RELEASE_ROOT).resolve(), target_is_directory=True)


def _user_code_lines(notebook: dict[str, Any]) -> int:
    return sum(
        1
        for cell in _code_cells(notebook)
        for line in _source(cell).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _library_imports(source: str, origin: str) -> list[tuple[str, str]]:
    """Return the (module, name) pairs one code cell imports from the library."""
    imports: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source, filename=origin)):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, "")
                for alias in node.names
                if alias.name.split(".")[0] == "dpstats"
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == "dpstats":
                imports.extend((module, alias.name) for alias in node.names)
    return imports


def _run(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one setup step, failing with its exact command and streams."""
    completed = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if completed.returncode != 0:
        pytest.fail(
            f"{' '.join(command)} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _error_outputs(notebook: Any) -> list[str]:
    """Describe the error outputs of an executed ``nbformat.NotebookNode``."""
    return [
        f"cell {index}: {output['ename']}: {output['evalue']}"
        for index, cell in enumerate(notebook.cells)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def _fetch_guard_reason(executed: Any) -> str | None:
    """Return the fetch guard's stated reason if it is the run's first error.

    ``allow_errors=True`` lets execution continue past a raised exception, so a
    failed guard cell leaves every later cell referencing undefined names. Only
    the first erroring cell is diagnostic: if that one cell is tagged
    ``fetch-guard``, the precondition failed and every later error is noise
    from cells that never had a chance to run; any other first failure is a
    genuine defect.
    """
    for cell in executed.cells:
        errors = [
            output
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if not errors:
            continue
        if FETCH_GUARD_TAG in cell.get("metadata", {}).get("tags", []):
            return f"{errors[0]['ename']}: {errors[0]['evalue']}"
        return None
    return None


def _result_text(notebook: Any) -> str:
    """Read printed and displayed text from the result-tagged cells."""
    tagged = [
        cell
        for cell in notebook.cells
        if "result" in cell.get("metadata", {}).get("tags", [])
    ]
    return "".join(
        output.get("text", output.get("data", {}).get("text/plain", ""))
        for cell in (tagged or notebook.cells)
        for output in cell.get("outputs", [])
    )


@pytest.mark.parametrize("output_type", ["stream", "execute_result", "display_data"])
def test_result_text_includes_printed_and_displayed_answers(output_type: str) -> None:
    content = (
        {"text": "answer"}
        if output_type == "stream"
        else {"data": {"text/plain": "answer"}}
    )
    notebook = nbformat.from_dict(
        {
            "cells": [
                {
                    "metadata": {"tags": ["result"]},
                    "outputs": [{"output_type": output_type, **content}],
                },
                {"metadata": {}, "outputs": [{"text": "not a result"}]},
            ]
        }
    )
    assert _result_text(notebook) == "answer"


def test_tutorials_are_discovered() -> None:
    """Guard the glob: an empty list would silently skip every check below."""
    assert TUTORIALS, f"no tutorial notebooks found under {TUTORIALS_DIRECTORY}"
    assert set(_tutorial_sources) == set(
        SUCCESS_CRITERIA
    ), "every authored tutorial needs a declared success criterion"


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_source_carries_no_execution_state(path: Path) -> None:
    """A committed tutorial is source only: no outputs, counts, timings or editor state."""
    notebook = _notebook(path)
    assert notebook["metadata"] == {}, (
        f"{path.name} carries notebook metadata {sorted(notebook['metadata'])}; "
        f"run `{STRIP_COMMAND}`"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert (
            cell["outputs"] == []
        ), f"cell {index} carries outputs; run `{STRIP_COMMAND}`"
        assert (
            cell["execution_count"] is None
        ), f"cell {index} carries an execution count"
        assert not CELL_EXECUTION_METADATA & set(cell["metadata"]), f"cell {index}"


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_code_parses_and_imports_only_the_public_surface(path: Path) -> None:
    """Every cell compiles, and every library import resolves.

    Each ``from dpstats... import X`` must name a public module and
    a name that module (or a submodule of it) actually defines.
    """
    for index, cell in enumerate(_code_cells(_notebook(path))):
        origin = f"{path.name}[code cell {index}]"
        source = TransformerManager().transform_cell(_source(cell))
        compile(source, origin, "exec")
        for module, name in _library_imports(source, origin):
            private = [part for part in module.split(".") if part.startswith("_")]
            assert not private, f"{origin} imports the private module {module}"
            assert not name.startswith("_"), f"{origin} imports the private name {name}"
            assert importlib.util.find_spec(module) is not None, f"{origin}: {module}"
            if not name:
                continue
            resolved = hasattr(importlib.import_module(module), name) or (
                importlib.util.find_spec(f"{module}.{name}") is not None
            )
            assert resolved, (
                f"{origin} imports {name!r} from {module}, which the installed "
                "package does not define"
            )


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_paths_are_portable(path: Path) -> None:
    """No tutorial names a machine path or a maintainer-only storage root."""
    text = path.read_text(encoding="utf-8")
    for spelling in UNPORTABLE_TEXT:
        assert spelling not in text, f"{path.name} names {spelling}"
    assert not HUB_TOKEN.search(text), f"{path.name} carries a Hugging Face token"


# STYLE.md S33-S36: the tutorial voice, which a tool can decide in part. S34's
# "names the step", S35's "aphorisms" and "commentary not needed to run the
# cell", and S36's "says what the cell does and how to read its output" need a
# reader; only the shapes below are mechanical.
WALKTHROUGH_PHRASE = "we walk through"  # the maintainer's tutorial 01 register: "We walk through how to …"
BARE_NOUN_HEADINGS = frozenset(
    {
        "overview",
        "introduction",
        "background",
        "summary",
        "conclusion",
        "discussion",
        "notes",
    }
)
NUMBERED_STEP = re.compile(r"^\d+\. ", re.MULTILINE)
BOLD_CALLOUT = re.compile(r"\*\*\s*(Warning|Note)\b", re.IGNORECASE)
TIMING_PHRASE = re.compile(
    r"\b(takes|take|taking)\s+(about|around|roughly|approximately)\s+\w+\s+"
    r"(seconds?|minutes?)\b",
    re.IGNORECASE,
)
MARKDOWN_LINK_TARGET = re.compile(r"\]\(([^)\s]+)")
SIBLING_FILE = re.compile(r"^[\w.-]+\.(ipynb|py)(#.*)?$")


# Tutorial prose is the maintainer's register (STYLE S33-S36 came from the
# maintainer's rewrite of run_your_own_script). A notebook that misses a rule is
# listed here and reports xfail; the entry must go when the maintainer's diff
# fixes it (a listed notebook that passes fails the test). The maintainer's own
# notebooks are embargoed outright; the others wait for the tutorial-prose lane. Note for the rule owner: the exemplar itself misses
# S34 (no numbered step list) and S36 (two consecutive code cells), so those
# two rules are proposals the maintainer has not yet confirmed against it.
MAINTAINER_TUTORIAL_GAPS = {
    "S33": {
        "algorithms.ipynb",
        "domains.ipynb",
        "metrics_and_baselines.ipynb",
        "privacy_budgets_and_accounting.ipynb",
        "tasks.ipynb",
    },
    "S34": {
        "domains.ipynb",
        "extend_with_transformations.ipynb",
        "metrics_and_baselines.ipynb",
        "privacy_budgets_and_accounting.ipynb",
        "tasks.ipynb",
    },
    "S36": {
        "algorithms.ipynb",
        "domains.ipynb",
        "extend_with_transformations.ipynb",
        "metrics_and_baselines.ipynb",
        "privacy_budgets_and_accounting.ipynb",
        "run_your_own_script.ipynb",
    },
}


def _maintainer_gap(rule: str, path: Path) -> None:
    if path.name in MAINTAINER_TUTORIAL_GAPS.get(rule, set()):
        pytest.xfail(f"{path.name} misses {rule}: maintainer-owned notebook")


def _markdown_cells(notebook: dict[str, Any]) -> list[str]:
    return [
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    ]


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_opens_in_the_walkthrough_voice(path: Path) -> None:
    """S33: the first markdown cell says "In this notebook, we walk through"."""
    _maintainer_gap("S33", path)
    first = _markdown_cells(_notebook(path))[0]
    assert WALKTHROUGH_PHRASE in " ".join(first.split()).lower(), path.name


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_headings_are_not_bare_nouns_and_steps_are_numbered(
    path: Path,
) -> None:
    """S34: no section is a bare "Overview"; the steps are a numbered list."""
    _maintainer_gap("S34", path)
    markdown = _markdown_cells(_notebook(path))
    headings = [
        line.lstrip("#").strip()
        for cell in markdown
        for line in cell.splitlines()
        if line.startswith("##")
    ]
    bare = [h for h in headings if h.lower().rstrip(".:") in BARE_NOUN_HEADINGS]
    assert not bare, f"{path.name} has headings that name no step: {bare}"
    assert any(
        NUMBERED_STEP.search(cell) for cell in markdown
    ), f"{path.name} lists its steps in no numbered list"


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_markdown_has_no_em_dash_bold_callout_or_timing(path: Path) -> None:
    """S35: no em-dash, no bold Warning or Note, no "takes about N seconds"."""
    for index, cell in enumerate(_markdown_cells(_notebook(path))):
        where = f"{path.name} markdown cell {index}"
        assert "\u2014" not in cell, f"{where} has an em-dash"
        assert not BOLD_CALLOUT.search(cell), f"{where} has a bold callout"
        assert not TIMING_PHRASE.search(cell), f"{where} comments on timing"


@pytest.mark.parametrize("path", _tutorial_sources.values(), ids=lambda path: path.name)
def test_tutorial_code_cells_follow_markdown_and_link_reference_pages_absolutely(
    path: Path,
) -> None:
    """S36: markdown precedes every code cell; reference links are absolute."""
    _maintainer_gap("S36", path)
    cells = _notebook(path)["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert (
                index > 0 and cells[index - 1]["cell_type"] == "markdown"
            ), f"{path.name} code cell {index} is not preceded by markdown"
    for index, cell in enumerate(_markdown_cells(_notebook(path))):
        for target in MARKDOWN_LINK_TARGET.findall(cell):
            local = target.startswith("#") or SIBLING_FILE.match(target)
            absolute = target.startswith(PUBLIC_SITE_URL) or target.startswith("http")
            assert (
                local or absolute
            ), f"{path.name} markdown cell {index} links {target}, not an absolute URL"


@pytest.fixture(scope="module")
def journey_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> JourneyEnvironment:
    """Return a fresh interpreter carrying the library wheel and a kernel for it."""
    if shutil.which("uv") is None:
        pytest.skip("uv builds the wheel and is not on PATH")

    root = tmp_path_factory.mktemp("tutorial-journeys")
    started = time.perf_counter()
    distribution = root / "dist"
    constraints = root / "constraints.txt"
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(distribution)],
        cwd=REPOSITORY_ROOT,
    )
    _run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--output-file",
            str(constraints),
        ],
        cwd=REPOSITORY_ROOT,
    )
    wheels = sorted(distribution.glob("*.whl"))
    assert len(wheels) == 1, [wheel.name for wheel in wheels]

    venv = root / "venv"
    _run([sys.executable, "-m", "venv", str(venv)])
    interpreter = venv / "bin" / "python"
    _run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--constraint",
            str(constraints),
            str(wheels[0]),
            "ipykernel",
            # A notebook frontend has it, so tqdm's autonotebook import finds
            # the widget it looks for instead of warning onto stderr.
            "ipywidgets",
            # A notebook frontend has it too, and Plotly's figure.show() needs
            # it to hand the figure to the notebook.
            "nbformat",
        ]
    )
    install_seconds = time.perf_counter() - started

    # An explicit kernelspec: the stock one launches whichever `python` PATH
    # resolves, which in this suite would be the editable checkout.
    kernels = root / "kernels"
    specification = kernels / KERNEL_NAME
    specification.mkdir(parents=True)
    (specification / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    str(interpreter),
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "dpstats tutorials",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )

    digest = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
    print(f"\nwheel {wheels[0].name} sha256:{digest}")
    print(f"venv {venv} installed in {install_seconds:.1f}s")
    return JourneyEnvironment(
        root=root,
        interpreter=interpreter,
        kernels=kernels,
        wheel=wheels[0].name,
        wheel_sha256=digest,
        install_seconds=round(install_seconds, 1),
    )


@pytest.fixture(scope="module")
def journey_records(
    journey_environment: JourneyEnvironment,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[list[dict[str, object]]]:
    """Collect one record per tutorial and write the report on teardown."""
    records: list[dict[str, object]] = []
    yield records
    report = {
        "python": sys.version.split()[0],
        "wheel": journey_environment.wheel,
        "wheel_sha256": journey_environment.wheel_sha256,
        "install_seconds": journey_environment.install_seconds,
        "tutorials": records,
    }
    path = tmp_path_factory.getbasetemp() / "tutorial-journeys.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\njourneys report {path}\n{json.dumps(report, indent=2)}")


def _redact(text: str) -> str:
    """Replace this machine's paths and any pasted token in one output string."""
    return HUB_TOKEN.sub(REDACTED_TOKEN, MACHINE_PATH.sub(REDACTED_PATH, text))


def _scrub(executed: Any) -> list[str]:
    """Redact an executed notebook's outputs in place, naming what was removed.

    A source notebook carries no machine path and no token, but running one
    prints the directory it ran in and the release it installed there. Those
    are properties of this run, so they are replaced before the copy that the
    site renders is written.

    Returns:
        One line per cell whose outputs changed, for the run's report.
    """
    stripped: list[str] = []
    for index, cell in enumerate(executed.cells):
        for output in cell.get("outputs", []):
            before = json.dumps(output, sort_keys=True)
            for key in TEXT_OUTPUT_KEYS:
                if isinstance(output.get(key), str):
                    output[key] = _redact(output[key])
            traceback: list[str] = output.get("traceback", [])
            if traceback:
                output["traceback"] = [_redact(line) for line in traceback]
            data = output.get("data", {})
            for mime in TEXT_MIME_TYPES:
                held: str | list[str] | None = data.get(mime)
                if isinstance(held, str):
                    data[mime] = _redact(held)
                elif held is not None:
                    data[mime] = [_redact(line) for line in held]
            if json.dumps(output, sort_keys=True) != before:
                stripped.append(f"cell {index}: machine paths or tokens removed")
    return stripped


def _output_bytes(executed: Any) -> tuple[int, list[str]]:
    """The weight of one executed notebook's outputs, and every output over the cap."""
    total = 0
    heavy: list[str] = []
    for index, cell in enumerate(executed.cells):
        for output in cell.get("outputs", []):
            size = len(json.dumps(output).encode("utf-8"))
            total += size
            if size > OUTPUT_BYTES_CEILING:
                heavy.append(f"cell {index}: one output of {size // 1024} KiB")
    return total, heavy


def _write_executed(path: Path, executed: Any) -> Path:
    """Write one executed copy, keyed by the source notebook it was run from."""
    EXECUTED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    executed.metadata[SOURCE_DIGEST_KEY] = hashlib.sha256(path.read_bytes()).hexdigest()
    copy = EXECUTED_DIRECTORY / path.name
    nbformat.write(executed, copy)
    (EXECUTED_DIRECTORY / f"{path.name}{GUARDED_SKIP_SUFFIX}").unlink(missing_ok=True)
    return copy


def _write_skipped(path: Path, reason: str) -> Path:
    """Record that this environment could not meet one notebook's precondition.

    The site reads this instead of an executed copy and says on the page why
    the cells carry no output, so a guarded skip is a stated absence rather
    than a missing file or a published error.
    """
    EXECUTED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    record = EXECUTED_DIRECTORY / f"{path.name}{GUARDED_SKIP_SUFFIX}"
    record.write_text(
        json.dumps(
            {
                SOURCE_DIGEST_KEY: hashlib.sha256(path.read_bytes()).hexdigest(),
                SKIP_REASON_KEY: HUB_REQUEST_ID.sub("", _redact(reason)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (EXECUTED_DIRECTORY / path.name).unlink(missing_ok=True)
    return record


def _sample_children(pid: int, stop: threading.Event, samples: list[int]) -> None:
    """Record the resident memory of one process's children until *stop* is set.

    The kernel a notebook runs in is a child of this process, so its footprint
    is measured from outside rather than from the notebook's own arithmetic.
    """
    while not stop.wait(0.25):
        children = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        ).stdout.split()
        if not children:
            continue
        resident = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(children)],
            capture_output=True,
            text=True,
        ).stdout.split()
        samples.append(sum(int(value) for value in resident) * 1024)


def _execute(
    path: Path, environment: JourneyEnvironment
) -> tuple[Any, float, float, list[int]]:
    """Execute one notebook copy and return it, its timings, and its memory samples.

    The notebook is an ``nbformat.NotebookNode``, which that package leaves
    unannotated. Timings are seconds; the samples are the kernel's resident
    bytes, and an empty list means nothing was measured rather than nothing
    was used.
    """
    workdir = environment.root / "shared-workspace"
    workdir.mkdir(exist_ok=True)
    _preinstall_local_release(workdir)
    copy = Path(shutil.copy(path, workdir / path.name))
    # A script the notebook wraps sits beside it in the checkout, so it sits
    # beside the copy here; the notebook reads and imports it from its own
    # directory, exactly as a reader who downloaded both would.
    for script in companion_scripts(path):
        shutil.copy(script, workdir / script.name)
    # nbformat leaves read() unannotated; the value is an nbformat.NotebookNode.
    notebook = cast(Any, nbformat.read(copy, as_version=4))

    manager = KernelManager(kernel_name=KERNEL_NAME)
    manager.kernel_spec_manager = KernelSpecManager(
        kernel_dirs=[str(environment.kernels)]
    )
    started = time.perf_counter()
    first_output_at: list[float] = []

    def on_cell_executed(cell: Any, **_: object) -> None:
        if cell.get("outputs") and not first_output_at:
            first_output_at.append(time.perf_counter())

    client = NotebookClient(
        notebook,
        manager,
        timeout=CELL_TIMEOUT,
        allow_errors=True,
        resources={"metadata": {"path": str(workdir)}},
        on_cell_executed=on_cell_executed,
    )
    resident: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_children, args=(os.getpid(), stop, resident), daemon=True
    )
    sampler.start()
    try:
        client.execute(cleanup_kc=True)
    finally:
        stop.set()
        sampler.join(timeout=5)
    finished = time.perf_counter()
    time_to_first_result = (
        first_output_at[0] if first_output_at else finished
    ) - started
    return notebook, finished - started, time_to_first_result, resident


@pytest.mark.slow
@pytest.mark.parametrize(
    "path",
    TUTORIALS
    + [
        _tutorial_sources[name]
        for name in (
            "install_and_fetch.ipynb",
            "run_your_own_script.ipynb",
            "extend_with_transformations.ipynb",
            "compare_on_the_benchmark.ipynb",
        )
    ],
    ids=lambda path: path.name,
)
def test_tutorial_executes_from_the_wheel(
    path: Path,
    journey_environment: JourneyEnvironment,
    journey_records: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one tutorial against the installed wheel and check its own criterion.

    A run that reaches its answer also leaves the executed copy the site
    renders, so the page and the journey are the same execution.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "1")
    for variable in HUB_CREDENTIAL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HF_HOME", str(journey_environment.root / "hf-home"))
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    executed, seconds, time_to_first_result, sampled = _execute(
        path, journey_environment
    )
    resident = max(sampled, default=0)
    guard_reason = _fetch_guard_reason(executed)
    if guard_reason is not None:
        journey_records.append(
            {
                "notebook": path.name,
                "status": "skipped",
                "reason": guard_reason,
                "skip_record": str(_write_skipped(path, guard_reason)),
            }
        )
        pytest.skip(guard_reason)

    errors = _error_outputs(executed)
    result_text = _result_text(executed)
    stripped = _scrub(executed)
    output_bytes, heavy = _output_bytes(executed)
    record: dict[str, object] = {
        "notebook": path.name,
        "run": 1 + sum(record["notebook"] == path.name for record in journey_records),
        "status": "executed" if not errors else "failed",
        "seconds": round(seconds, 1),
        "time_to_first_result_seconds": round(time_to_first_result, 1),
        "peak_rss_mib": round(resident / (1024 * 1024), 1),
        "output_kib": round(output_bytes / 1024, 1),
        "stripped": stripped,
        "user_code_lines": _user_code_lines(_notebook(path)),
        "code_cells": len(_code_cells(_notebook(path))),
        "error_cells": len(errors),
    }
    journey_records.append(record)
    assert not errors, "\n".join(errors)
    missing = [
        expected
        for expected in SUCCESS_CRITERIA[path.name]
        if expected not in result_text
    ]
    assert not missing, f"result cells did not report {missing}:\n{result_text}"

    # The run's own paths and any pasted token are stripped above, because they
    # belong to this machine. These spellings are not: a tutorial that reaches
    # a maintainer-only store or a private package has a defect to repair, and
    # removing the evidence would hide it.
    body = json.dumps(executed)
    for spelling in UNPORTABLE_TEXT:
        assert spelling not in body, f"the executed {path.name} names {spelling}"
    assert not HUB_TOKEN.search(body), f"the executed {path.name} carries a token"
    record["executed_copy"] = str(_write_executed(path, executed))
    assert not heavy, (
        f"{path.name} embeds an output over {OUTPUT_BYTES_CEILING // 1024} KiB: "
        f"{heavy}. Shrink the figure or the frame rather than the ceiling."
    )
    assert output_bytes <= NOTEBOOK_OUTPUT_BYTES_CEILING, (
        f"{path.name} carries {output_bytes // 1024} KiB of outputs, over "
        f"{NOTEBOOK_OUTPUT_BYTES_CEILING // 1024} KiB."
    )
    assert seconds <= WALL_BUDGET_SECONDS, (
        f"{path.name} ran for {seconds:.0f}s, over the {WALL_BUDGET_SECONDS:.0f}s "
        "budget. Escalate the notebook rather than the budget."
    )
    assert sampled, (
        f"no resident-memory sample was taken while {path.name} ran, so its "
        "memory budget is unverified rather than met."
    )
    assert resident <= MEMORY_BUDGET_BYTES, (
        f"{path.name} peaked at {resident / (1024 * 1024):.0f} MiB, over the "
        f"{MEMORY_BUDGET_BYTES // (1024 * 1024)} MiB budget. Escalate the "
        "notebook rather than the budget."
    )


@pytest.mark.slow
def test_the_journey_interpreter_carries_only_the_installed_library(
    journey_environment: JourneyEnvironment,
) -> None:
    """The kernel resolves the wheel, not this checkout, and has no operator app."""
    root = journey_environment.root
    # Every notebook runs from a directory under this root, so an interpreter
    # started here cannot reach the checkout through the working directory.
    assert not root.is_relative_to(REPOSITORY_ROOT)
    probe = "import dpstats; print(dpstats.__file__)"
    observed = _run(
        [str(journey_environment.interpreter), "-c", probe], cwd=root
    ).stdout
    module = observed.strip()
    assert Path(module).is_relative_to(root), module
