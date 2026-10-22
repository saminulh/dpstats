"""Reading the executed tutorial notebooks, and the tutorials index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


from tutorials import (
    GUARDED_SKIP_SUFFIX,
    SKIP_REASON_KEY,
    SOURCE_DIGEST_KEY,
    companion_scripts,
    executed_directory,
    executed_notebook,
    tutorial_metadata,
    tutorial_sources,
    tutorials_index,
)
from source_links import download_notebook


def test_download_links_follow_the_serving_site_and_keep_companions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "starter.ipynb"
    script = tmp_path / "wrapper.py"
    script.write_text("def estimate():\n    return 1\n")
    code = {"cell_type": "code", "source": ["from wrapper import estimate\n"]}
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "[Task](https://saminulh.github.io/dpstats/mean_estimation/#task)\n"
                            "[Next](next.ipynb) [Code](wrapper.py) [External](https://python.org/)\n"
                        ],
                    },
                    code,
                ]
            }
        )
    )
    document = json.loads(download_notebook(path, "https://beta.example/"))
    prose = document["cells"][0]["source"]
    assert "https://beta.example/mean_estimation/#task" in prose
    assert "https://beta.example/library/tutorials/next/" in prose
    assert "[Code](wrapper.py)" in prose
    assert "[External](https://python.org/)" in prose
    assert document["cells"][1] == code


def test_tutorial_index_rebases_links_in_source_description(tmp_path: Path) -> None:
    path = tmp_path / "domains.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# Domains\n\nA [Domain](../reference/domains/) describes public constraints."
                        ],
                    }
                ]
            }
        )
    )
    assert "[Domain](reference/domains/)" in tutorials_index(
        {"Using the library": [path]}
    )


def test_tutorial_sources_are_the_directory_notebooks_by_name(tmp_path: Path) -> None:
    first = _write_source(tmp_path, "first.ipynb")
    second = _write_source(tmp_path, "second.ipynb")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    assert tutorial_sources(tmp_path) == {first.name: first, second.name: second}


def _write_source(directory: Path, name: str, cell: str = "print(1)\n") -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "code", "source": [cell], "outputs": []}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_executed(directory: Path, source: Path, digest: str | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["cells"][0]["outputs"] = [{"output_type": "stream", "text": "1\n"}]
    document["cells"][0]["execution_count"] = 1
    if digest is not None:
        document["metadata"][SOURCE_DIGEST_KEY] = digest
    path = directory / source.name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_tutorial_metadata_comes_from_content_not_filename(tmp_path: Path) -> None:
    path = tmp_path / "unrelated_filename.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": [
                            "# Read results\n\n",
                            "Inspect a sweep. More detail follows.\n",
                        ],
                    }
                ]
            }
        )
    )
    assert tutorial_metadata(path) == ("Read results", "Inspect a sweep.")


def test_the_executed_copy_of_the_committed_notebook_is_returned(
    tmp_path: Path,
) -> None:
    """The page is the copy the harness wrote, outputs and all."""
    source = _write_source(tmp_path, "01_first.ipynb")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_executed(tmp_path / "_executed", source, digest)
    document, unrun = executed_notebook(source, tmp_path / "_executed")
    assert unrun == ""
    assert document["cells"][0]["outputs"] == [{"output_type": "stream", "text": "1\n"}]


def test_a_notebook_that_was_never_executed_stops_the_build_by_name(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, "01_first.ipynb")
    with pytest.raises(RuntimeError) as refusal:
        executed_notebook(source, tmp_path / "_executed")
    assert "01_first.ipynb" in str(refusal.value)
    assert "test_tutorials.py" in str(refusal.value)


def test_a_copy_executed_from_an_earlier_notebook_stops_the_build_by_name(
    tmp_path: Path,
) -> None:
    """The key is the source's content, so an edited notebook fails the build."""
    source = _write_source(tmp_path, "01_first.ipynb")
    _write_executed(tmp_path / "_executed", source, "0" * 64)
    with pytest.raises(RuntimeError) as refusal:
        executed_notebook(source, tmp_path / "_executed")
    assert "different version" in str(refusal.value)
    assert "01_first.ipynb" in str(refusal.value)


def test_a_copy_that_never_ran_stops_the_build_by_name(tmp_path: Path) -> None:
    """A source notebook relabelled with the right digest is not an execution."""
    source = _write_source(tmp_path, "01_first.ipynb")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    relabelled = json.loads(source.read_text(encoding="utf-8"))
    relabelled["metadata"][SOURCE_DIGEST_KEY] = digest
    (tmp_path / "_executed").mkdir()
    (tmp_path / "_executed" / source.name).write_text(
        json.dumps(relabelled), encoding="utf-8"
    )
    with pytest.raises(RuntimeError) as refusal:
        executed_notebook(source, tmp_path / "_executed")
    assert "never ran" in str(refusal.value)
    assert "01_first.ipynb" in str(refusal.value)


def _write_skip(directory: Path, source: Path, digest: str, reason: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{source.name}{GUARDED_SKIP_SUFFIX}"
    record.write_text(
        json.dumps({SOURCE_DIGEST_KEY: digest, SKIP_REASON_KEY: reason}),
        encoding="utf-8",
    )
    return record


def test_a_guarded_skip_returns_the_source_with_its_stated_reason(
    tmp_path: Path,
) -> None:
    """A precondition this environment cannot meet is a page, not a build failure."""
    source = _write_source(tmp_path, "02_release.ipynb")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_skip(tmp_path / "_executed", source, digest, "GatedRepoError: no access")
    document, unrun = executed_notebook(source, tmp_path / "_executed")
    assert unrun == "GatedRepoError: no access"
    assert document["cells"][0]["outputs"] == []


def test_a_skip_record_from_an_earlier_notebook_stops_the_build_by_name(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, "02_release.ipynb")
    _write_skip(tmp_path / "_executed", source, "0" * 64, "GatedRepoError: no access")
    with pytest.raises(RuntimeError) as refusal:
        executed_notebook(source, tmp_path / "_executed")
    assert "different version" in str(refusal.value)
    assert "02_release.ipynb" in str(refusal.value)


def test_an_executed_copy_and_a_skip_record_together_stop_the_build(
    tmp_path: Path,
) -> None:
    """Only one of them can be what the page shows, so both is undecided."""
    source = _write_source(tmp_path, "02_release.ipynb")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_executed(tmp_path / "_executed", source, digest)
    _write_skip(tmp_path / "_executed", source, digest, "GatedRepoError: no access")
    with pytest.raises(RuntimeError) as refusal:
        executed_notebook(source, tmp_path / "_executed")
    assert "undecided" in str(refusal.value)


def test_tutorials_index_preserves_reading_order_and_uses_content(
    tmp_path: Path,
) -> None:
    notebooks = []
    for name, title in (("z", "First"), ("a", "Second")):
        path = tmp_path / f"{name}.ipynb"
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": [
                                f"# {title}\n\nAn authored sentence. Not a second summary.\n"
                            ],
                        }
                    ]
                }
            )
        )
        notebooks.append(path)
    page = tutorials_index({"Running experiments": notebooks})
    assert "1. [First](tutorials/z.ipynb) — An authored sentence." in page
    assert "2. [Second](tutorials/a.ipynb) — An authored sentence." in page
    assert "Not a second summary" not in page
    assert "## Running experiments" in page
    grouped = tutorials_index(
        {"Running experiments": notebooks[:1], "Using the library": notebooks[1:]}
    )
    assert grouped.index("## Running experiments") < grouped.index(
        "## Using the library"
    )
    assert "1. [Second]" in grouped


def test_companion_scripts_are_the_sibling_modules_the_notebook_names(
    tmp_path: Path,
) -> None:
    """A script is carried only where a notebook reads or imports it."""
    named = _write_source(
        tmp_path, "00_wrap.ipynb", 'print(open("my_script.py").read())\n'
    )
    silent = _write_source(tmp_path, "01_other.ipynb")
    (tmp_path / "my_script.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("y = 2\n", encoding="utf-8")
    assert companion_scripts(named) == [tmp_path / "my_script.py"]
    assert companion_scripts(silent) == []


def test_companion_scripts_reach_an_import_named_only_inside_another_script(
    tmp_path: Path,
) -> None:
    """A wrapped script's own import is staged even though the notebook never names it.

    The notebook names ``my_regression.py`` in prose; ``my_regression.py`` imports
    ``my_regression_script`` (no notebook ever spells that filename), and the
    closure still reaches it.
    """
    wrapping = _write_source(
        tmp_path,
        "02_wrap_a_script.ipynb",
        "from my_regression import Wrapper  # see my_regression.py\n",
    )
    (tmp_path / "my_regression.py").write_text(
        "from my_regression_script import run\n", encoding="utf-8"
    )
    (tmp_path / "my_regression_script.py").write_text(
        "def run(): ...\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.py").write_text("z = 3\n", encoding="utf-8")

    assert companion_scripts(wrapping) == [
        tmp_path / "my_regression.py",
        tmp_path / "my_regression_script.py",
    ]


def test_companion_scripts_do_not_match_a_stem_inside_a_longer_word(
    tmp_path: Path,
) -> None:
    """A sibling's bare module name must appear as its own identifier.

    ``a.py``'s stem, ``a``, is a substring of ``data``; a notebook that
    merely talks about data must not pull in an unrelated sibling literally
    named ``a.py``.
    """
    talks_about_data = _write_source(
        tmp_path, "03_uses_data.ipynb", "data = load_frame()\n"
    )
    (tmp_path / "a.py").write_text("q = 1\n", encoding="utf-8")

    assert companion_scripts(talks_about_data) == []


@pytest.mark.parametrize(
    ("local_release", "local_copy", "expected"),
    [
        (True, "02_release.ipynb", "_executed_local"),
        (True, f"02_release.ipynb{GUARDED_SKIP_SUFFIX}", "_executed_local"),
        (True, None, "_executed"),
        (False, "02_release.ipynb", "_executed"),
    ],
)
def test_only_a_local_release_build_reads_a_local_copy(
    tmp_path: Path, local_release: bool, local_copy: str | None, expected: str
) -> None:
    """A copy run against a local release never reaches a build that reads none."""
    if local_copy is not None:
        (tmp_path / "_executed_local").mkdir()
        (tmp_path / "_executed_local" / local_copy).write_text("{}", encoding="utf-8")
    directory = executed_directory(
        tmp_path, "02_release.ipynb", local_release=local_release
    )
    assert directory == tmp_path / expected
