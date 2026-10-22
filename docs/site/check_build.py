"""Run the strict documentation build across every root configuration.

`pytest tests/unit` renders pages at the renderer-function level over fixture
data and never builds a real site (see the `tests/unit/docs` guard test), so
this script is the end-to-end build gate: a genuine `mkdocs build --strict`
in each root configuration the site supports. It is not a pytest test —
invoke it directly with `python docs/site/check_build.py`. Wired into
`.github/workflows/docs.yml` before the deploy build, and into the website
skill's gate procedure.

After the builds it checks every page against its template and compares the
result to `templates/conformance-baseline.json`, the violations this site
already carries. A violation that file does not list fails the run, and so does
a listed violation that does not occur, so a burn-down deletes its lines in
the same change that repairs the pages. `--record` rewrites the file from the
current build; it is the burn-down's tool, not a way to absorb a new defect.

The executed tutorial copies under the gitignored tutorials/_executed/ are
written by the journeys harness, and this script renders them. Run it first,
and again after editing a notebook:

    pytest tests/integration/test_tutorials.py -m slow -n 0

The journeys harness is the one notebook executor, and it needs what this
script's environment does not have: `uv`, a built wheel, a fresh non-editable
interpreter, and the Hub for the notebooks that fetch a release. This
script stays importable from the docs CI job's `[docs]`-only environment, which
has none of them and no pytest either, so running the harness from here would
make the gate depend on an environment it does not declare. A missing or stale
executed copy fails every configuration below, naming the notebook and this
command.

Up to three configurations are exercised. The first builds against the release
the committed `docs/benchmark.lock` pins, as the public site does.
When `DOCS_LOCAL_RELEASE_ROOT` names a local Distribution Release, the second
builds against it; the reference workload is the one to name, and its timing
belongs to the site quality run (`docs/site/quality/`), not to this gate. The
third builds the page-template previews. Before the previews, this script also
runs the whole-build sweeps in `test_built_site.py` against the last release
build; see `_built_site_test_pass` for why and how.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

SITE_DIR = Path(__file__).resolve().parent
DOCS_ROOT = SITE_DIR.parent
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))

from dpstats.tasks import BUILT_IN_TASKS  # noqa: E402

from results_roots import LOCAL_RELEASE_ENV  # noqa: E402
from templates.conformance import (  # noqa: E402
    BASELINE_PATH,
    baseline_findings,
    check_site,
    group_by_check,
    load_baseline,
)

# The design preview of the page templates: the flag a build asks for it with,
# the generator that writes the snapshot it renders, and where its pages land.
TEMPLATES_PREVIEW_ENV = "DOCS_SERVE_TEMPLATES"
FILLER_GENERATOR = SITE_DIR / "templates" / "filler_results.py"
TEMPLATES_PREFIX = "templates"

# The env var names test_built_site.py itself reads its two builds from.
BUILT_SITE_ENV = "DOCS_BUILT_SITE"
PUBLIC_SITE_ENV = "DOCS_PUBLIC_SITE"
BUILT_SITE_TEST_PATH = "tests/unit/docs/test_built_site.py"


def build_strict_site(site_dir: Path) -> None:
    """Run one strict `mkdocs build` into *site_dir*, replacing it if present."""
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    shutil.rmtree(site_dir, ignore_errors=True)
    # mkdocs ships no complete type information; the loaded config's exact
    # shape is this library's own, not this script's, so it is honestly `Any`.
    config: Any = load_config(
        config_file=str(SITE_DIR / "mkdocs.yml"), site_dir=str(site_dir), strict=True
    )
    build(config)


def _built_site_test_pass(site: Path, run: list[str]) -> list[str]:
    """Run `test_built_site.py`'s whole-build sweeps against *site*.

    That module's checks read a finished build only and never rebuild one, so
    the last release configuration's own site (built without the
    template-preview flag, hence both `DOCS_BUILT_SITE` and `DOCS_PUBLIC_SITE`)
    already answers them; this is the one place that exercises them, instead of a second,
    separately remembered `pytest ...` invocation a contributor has to know
    to run. It needs nothing this script's own environment cannot already
    have (no wheel, no fresh interpreter, no live release) unlike the
    notebook harness the module docstring names, but it does need pytest and
    the test package, which the docs CI job's `[docs]`-only environment lacks;
    skipped there by name rather than failing on an undeclared dependency.
    """
    if importlib.util.find_spec("pytest") is None:
        print(
            "check_build: skipping the built-site test suite: pytest is not "
            "installed in this environment.",
            flush=True,
        )
        return []
    print("--- check_build: the built-site test suite ---", flush=True)
    run.append("the built-site test suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", BUILT_SITE_TEST_PATH],
        cwd=DOCS_ROOT.parent,
        env={**os.environ, BUILT_SITE_ENV: str(site), PUBLIC_SITE_ENV: str(site)},
    )
    if result.returncode != 0:
        return ["the built-site test suite"]
    return []


def _preview_configuration(site: Path, observed: set[str], run: list[str]) -> list[str]:
    """Build and check the design preview of the templates, where it exists.

    The preview renders the template pages from a snapshot of invented objects,
    so it is the one build whose pages carry placeholder text on purpose. Its
    generator writes that snapshot and is not in every checkout; without it the
    configuration says so rather than passing silently.
    """
    if not FILLER_GENERATOR.is_file():
        print(
            f"check_build: skipping the page-template previews: "
            f"{FILLER_GENERATOR.name} is not in this checkout, so no snapshot "
            "can be written for them.",
            flush=True,
        )
        return []
    print("--- check_build: the page-template previews ---", flush=True)
    run.append("the page-template previews")
    try:
        subprocess.run([sys.executable, str(FILLER_GENERATOR)], check=True)
        os.environ[TEMPLATES_PREVIEW_ENV] = "1"
        build_strict_site(site)
    except Exception:
        traceback.print_exc()
        return ["the page-template previews"]
    finally:
        os.environ.pop(TEMPLATES_PREVIEW_ENV, None)
    if not (site / TEMPLATES_PREFIX / "index.html").exists():
        print(
            f"check_build: the preview build wrote no {TEMPLATES_PREFIX}/ pages.",
            file=sys.stderr,
        )
        return ["the page-template previews"]
    observed.update(check_site(site, sorted(BUILT_IN_TASKS), preview=True))
    return []


def main(record: bool = False) -> int:
    """Build every configuration, check it, and return the exit status."""
    os.environ.setdefault("DISABLE_MKDOCS_2_WARNING", "true")
    os.environ.setdefault("NO_MKDOCS_2_WARNING", "true")
    with tempfile.TemporaryDirectory(prefix="dp-docs-check-build-") as scratch:
        scratch_path = Path(scratch)
        named = os.environ.get(LOCAL_RELEASE_ENV)
        configurations: list[tuple[str, dict[str, str]]] = [
            ("the release the committed lock pins", {}),
        ]
        if named:
            configurations.append(
                (
                    f"the local Distribution Release at {named}",
                    {LOCAL_RELEASE_ENV: named},
                )
            )
        failures: list[str] = []
        observed: set[str] = set()
        run: list[str] = [label for label, _ in configurations]
        for label, env in configurations:
            print(f"--- check_build: {label} ---", flush=True)
            os.environ.pop(LOCAL_RELEASE_ENV, None)
            os.environ.update(env)
            site = scratch_path / "site"
            try:
                build_strict_site(site)
            except Exception:
                traceback.print_exc()
                failures.append(label)
                continue
            observed.update(check_site(site, sorted(BUILT_IN_TASKS)))
        os.environ.pop(LOCAL_RELEASE_ENV, None)
        public = scratch_path / "site"
        if (public / TEMPLATES_PREFIX).exists():
            print(
                "check_build: a build that did not ask for the page-template "
                f"previews wrote {TEMPLATES_PREFIX}/ anyway.",
                file=sys.stderr,
            )
            failures.append("the public build writes no page-template preview")
        if not failures and named:
            failures += _built_site_test_pass(public, run)
        elif not failures:
            print(
                "check_build: skipping the built-site test suite: its sweeps "
                f"read a local release build, and {LOCAL_RELEASE_ENV} names none.",
                flush=True,
            )
        if not failures:
            failures += _preview_configuration(scratch_path / "preview", observed, run)
    if not failures and record:
        BASELINE_PATH.write_text(
            json.dumps(group_by_check(sorted(observed)), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"check_build: recorded {len(observed)} violation(s) in {BASELINE_PATH}.")
        return 0
    if not failures:
        drift = baseline_findings(sorted(observed), load_baseline(BASELINE_PATH))
        for line in drift:
            print(f"conformance: {line}", file=sys.stderr)
        if drift:
            print(
                f"check_build: {len(drift)} template-conformance change(s) against "
                f"{BASELINE_PATH.name}. A repair deletes its line from that file; "
                "a new violation is fixed, not recorded.",
                file=sys.stderr,
            )
            return 1
    if failures:
        print(
            f"check_build: {len(failures)} of {len(run)} "
            f"configuration(s) failed: {failures}",
            file=sys.stderr,
        )
        return 1
    print(f"check_build: {len(run)} configuration(s) passed: {run}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--record" in sys.argv[1:]))
