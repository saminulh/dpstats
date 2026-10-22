"""Read the executed tutorial notebooks the journeys harness wrote.

The notebooks under the repository's top-level ``tutorials/`` are committed
source-only, and the journeys harness is the one executor: it runs each of
them from a clean wheel and writes the executed copy under the gitignored
``tutorials/_executed/``, keyed by the source notebook's content hash. The
site renders that copy through mkdocs-jupyter and never executes a notebook
itself, so a missing or stale copy stops the build by name rather than
publishing a page with no outputs.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from site_copy import copy_html

TUTORIALS_DIRNAME = "tutorials"

# Where the harness writes its executed copies (gitignored), and the notebook
# metadata key carrying the digest of the source notebook each copy was run from.
EXECUTED_DIRNAME = "_executed"
SOURCE_DIGEST_KEY = "dp_source_sha256"

# Copies the harness ran against a local release. Only a build that itself
# reads a local release renders them: their outputs carry that release's
# contents.
LOCAL_EXECUTED_DIRNAME = "_executed_local"

# A notebook whose precondition this environment cannot meet — a release it
# cannot reach — leaves this record instead of an executed copy, carrying the
# reason its fetch guard gave. The two never coexist: the harness writes one
# and removes the other.
GUARDED_SKIP_SUFFIX = ".skipped.json"
SKIP_REASON_KEY = "reason"

# The class the page's own notice carries, so a reader and the built-site
# check find the unrun state in the same place.
UNRUN_NOTICE_CLASS = "dp-tutorial-unrun"

# The one command that produces the executed copies, named in every refusal so
# a build that lacks them says how to get them.
HARNESS_COMMAND = (
    "pytest tests/integration/test_tutorials.py -m slow -n 0 "
    "(with DOCS_LOCAL_RELEASE_ROOT set for a copy under _executed_local/)"
)


def tutorial_sources(tutorials: Path) -> dict[str, Path]:
    """Find the tutorial notebooks in one directory, by file name."""
    return {path.name: path for path in sorted(tutorials.glob("*.ipynb"))}


def published_tutorials(docs: Path, tutorials: Path) -> list[Path]:
    """Return the notebooks selected by site navigation, in reading order."""
    sources = tutorial_sources(tutorials)
    names = re.findall(
        r"^\s+- [^:\n]+: library/tutorials/([a-z_]+\.ipynb)$",
        (docs / "site" / "mkdocs.yml").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return [sources[name] for name in names]


def _names_script(text: str, script: Path) -> bool:
    """One text mentions *script* either by its filename or its import name.

    Prose (the notebook's own markdown and code) spells a companion by its
    filename, ``my_script.py``; a Python import spells the same file by its
    bare module name, ``my_script``. Both count as naming it, but the bare
    name must appear as its own identifier (never as a run of characters
    inside a longer word), or an unrelated sibling whose stem happens to be
    a substring of the text — ``data`` inside ``metadata``, or a sibling
    literally named ``a.py`` matching the word ``data`` — would be pulled in.
    """
    if script.name in text:
        return True
    return re.search(rf"(?<![\w.]){re.escape(script.stem)}(?!\w)", text) is not None


def companion_scripts(notebook: Path) -> list[Path]:
    """Return the sibling ``.py`` files one committed notebook needs, transitively.

    A tutorial that wraps a script keeps that script beside it as an ordinary
    module, so the notebook can read and import it. A wrapped script may
    itself import a further sibling the notebook never names directly (a
    script it wraps in turn); that closure is included too, so the harness
    stages everything the notebook's own imports actually reach and the site
    offers all of it beside the notebook's download. A script neither the
    notebook nor an included script names is neither run nor published.

    Args:
        notebook: The committed source notebook.

    Returns:
        The sibling scripts the notebook needs directly or transitively,
        sorted, each included once.

    Example:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     notebook = Path(root) / "00_wrap.ipynb"
        ...     _ = notebook.write_text('print(open("my_script.py").read())')
        ...     _ = (Path(root) / "my_script.py").write_text("x = 1")
        ...     [path.name for path in companion_scripts(notebook)]
        ['my_script.py']

        An import inside an included script, naming a further sibling only
        by its bare module name, still reaches that sibling:

        >>> with TemporaryDirectory() as root:
        ...     notebook = Path(root) / "00_wrap.ipynb"
        ...     _ = notebook.write_text('from outer import Wrapper  # outer.py')
        ...     _ = (Path(root) / "outer.py").write_text(
        ...         "from inner import helper"
        ...     )
        ...     _ = (Path(root) / "inner.py").write_text("def helper(): ...")
        ...     [path.name for path in companion_scripts(notebook)]
        ['inner.py', 'outer.py']
    """
    available = tuple(notebook.parent.glob("*.py"))
    included: dict[str, Path] = {}
    pending = [notebook.read_text(encoding="utf-8")]
    while pending:
        text = pending.pop()
        for script in available:
            if script.name not in included and _names_script(text, script):
                included[script.name] = script
                pending.append(script.read_text(encoding="utf-8"))
    return sorted(included.values())


def tutorial_metadata(path: Path) -> tuple[str, str]:
    """Read the notebook's title and first introductory sentence."""
    document = json.loads(path.read_text(encoding="utf-8"))
    opening = next(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "markdown"
    )
    heading, _, prose = opening.partition("\n")
    if not heading.startswith("# "):
        raise ValueError(f"{path.name}: start the notebook with its title")
    paragraph = prose.strip().split("\n\n", 1)[0]
    sentence = re.split(r"(?<=[.!?])\s+", " ".join(paragraph.split()), maxsplit=1)[0]
    return heading.removeprefix("# ").strip(), sentence


def executed_directory(tutorials: Path, name: str, *, local_release: bool) -> Path:
    """Return where one tutorial's executed copy is read from.

    A build reading a local release renders the copy run against a local
    release when one exists; every other build renders the committed copy.
    """
    local = tutorials / LOCAL_EXECUTED_DIRNAME
    if local_release and any(
        (local / f"{name}{suffix}").is_file() for suffix in ("", GUARDED_SKIP_SUFFIX)
    ):
        return local
    return tutorials / EXECUTED_DIRNAME


class ExecutedTutorial(NamedTuple):
    """One tutorial as this build has it: the cells, and why they carry no output."""

    document: dict[str, Any]
    unrun_reason: str


def executed_notebook(source: Path, executed: Path) -> ExecutedTutorial:
    """Return one tutorial's executed copy, refusing a missing or stale one.

    A notebook the harness could not run because its precondition failed —
    a release this environment cannot reach — comes back as its own source
    with the guard's reason, so the page says why it shows no output instead
    of showing an error or nothing at all.

    Args:
        source: The committed source-only notebook.
        executed: The directory the harness writes its executed copies to.

    Returns:
        The executed notebook and an empty reason, or the source notebook and
        the reason its fetch guard gave.

    Raises:
        RuntimeError: This notebook has neither an executed copy nor a skip
            record, both are present, the record on disk was made from a
            different version of the source, or the copy carries cells that
            never ran.
    """
    copy = executed / source.name
    skipped = executed / f"{source.name}{GUARDED_SKIP_SUFFIX}"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if copy.is_file() and skipped.is_file():
        raise RuntimeError(
            f"{source.name} has both an executed copy at {copy} and a skip "
            f"record at {skipped}, so which one the page shows is undecided. "
            f"Run {HARNESS_COMMAND}."
        )
    if skipped.is_file():
        record = json.loads(skipped.read_text(encoding="utf-8"))
        if record.get(SOURCE_DIGEST_KEY) != digest:
            raise RuntimeError(
                f"The skip record for {source.name} at {skipped} was made from "
                f"a different version of that notebook. Run {HARNESS_COMMAND}."
            )
        return ExecutedTutorial(
            json.loads(source.read_text(encoding="utf-8")),
            record[SKIP_REASON_KEY],
        )
    if not copy.is_file():
        raise RuntimeError(
            f"{source.name} has no executed copy at {copy}, so its page would "
            f"carry no outputs. Run {HARNESS_COMMAND}."
        )
    document = json.loads(copy.read_text(encoding="utf-8"))
    recorded = document["metadata"].get(SOURCE_DIGEST_KEY)
    if recorded != digest:
        raise RuntimeError(
            f"The executed copy of {source.name} at {copy} was run from a "
            f"different version of that notebook ({recorded} rather than "
            f"{digest}). Run {HARNESS_COMMAND}."
        )
    # The digest says which notebook the copy came from, not that it ran. A
    # code cell that never ran carries no execution count, so a source notebook
    # relabelled as an executed one is refused here rather than published as a
    # tutorial with no outputs.
    unrun = [
        index
        for index, cell in enumerate(document["cells"])
        if cell["cell_type"] == "code" and cell.get("execution_count") is None
    ]
    if unrun:
        raise RuntimeError(
            f"The executed copy of {source.name} at {copy} has cells {unrun} "
            f"that never ran, so its page would be missing their outputs. "
            f"Run {HARNESS_COMMAND}."
        )
    return ExecutedTutorial(document, "")


def tutorials_index(groups: dict[str, Sequence[Path]]) -> str:
    """List notebooks in reading order using only their own opening text."""
    entries = []
    for heading, notebooks in groups.items():
        entries.append(f"## {heading}\n")
        for number, path in enumerate(notebooks, 1):
            title, description = tutorial_metadata(path)

            def index_link(match: re.Match[str]) -> str:
                target = match[1]
                if "://" in target or target.startswith(("/", "#")):
                    return match[0]
                rebased = posixpath.normpath(posixpath.join(TUTORIALS_DIRNAME, target))
                if target.endswith("/"):
                    rebased += "/"
                return "](" + rebased + ")"

            description = re.sub(r"\]\(([^\s)]+)\)", index_link, description)
            entries.append(
                f"{number}. [{title}](tutorials/{path.name}) — {description}"
            )
        entries.append("")
    intro = copy_html("tutorials", "index_intro")
    return (
        "# User guide\n\n"
        + (f"{intro}\n\n" if intro else "")
        + "\n".join(entries)
        + "\n"
    )
