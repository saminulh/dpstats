"""The distribution declares its MIT license and ships the license text.

The declaration and the shipped file are checked together; the holder line
names the authors as a group, whom the package metadata and the git history
identify.
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_root_license_file_is_mit_text_with_the_authors_holder_line() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert text.startswith("MIT License\n")
    assert "Copyright (c) 2026 The DP Statistics Benchmark authors" in text


def test_pyproject_declares_spdx_expression_and_license_file() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["license"] == "MIT"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
