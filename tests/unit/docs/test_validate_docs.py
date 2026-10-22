"""The API-reference coverage checker reads what ``docs/reference`` owns."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_docs", REPO_ROOT / "tools" / "validate_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_reference_pages_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main([]) == 0
    assert "Validation PASSED" in capsys.readouterr().out


def test_only_a_losses_module_is_owed_to_the_reference_pages() -> None:
    checker = _load()
    assert checker.required_modules_for_task("m_estimation") == [
        "dpstats.tasks.m_estimation.losses"
    ]
    assert checker.required_modules_for_task("mean_estimation") == []


def test_a_directive_naming_a_deleted_module_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checker = _load()
    (tmp_path / "gone.md").write_text("::: dpstats.tasks.gone\n")
    monkeypatch.setattr(checker, "REFERENCE_DIR", tmp_path)
    assert checker.main([]) == 1
    assert "deleted module dpstats.tasks.gone" in capsys.readouterr().out


def test_a_losses_module_without_a_directive_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checker = _load()
    (tmp_path / "tasks.md").write_text("::: dpstats.tasks\n")
    monkeypatch.setattr(checker, "REFERENCE_DIR", tmp_path)
    assert checker.main(["--task", "m_estimation"]) == 1
    assert "no ':::' directive for dpstats.tasks.m_estimation.losses" in (
        capsys.readouterr().out
    )
