"""The one reader of where a tree's Instances came from."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpstats.workspace.layout import provenance_directory
from dpstats.workspace.provenance import (
    InstanceFacts,
    read_provenance,
    study_paths,
    study_ref,
)

ALPHA = "benchmark-info/provenance/alpha.json"
BETA = "benchmark-info/provenance/beta.json"
SOURCE = {"variables": ["income", "age"], "coefficient": "income"}


def _write(root: Path, ref: str, record: dict[str, Any]) -> None:
    path = root / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "local"
    _write(
        root,
        ALPHA,
        {
            "study_id": "alpha",
            "title": "Alpha",
            "discipline": "economics",
            "data_license": "CC0 1.0",
            "instances": {
                "i5-A": {"sample_unit": "individual", "source": SOURCE},
                "i5-B": {"sample_unit": "household", "source": {}},
            },
        },
    )
    _write(
        root,
        BETA,
        {
            "study_id": "beta",
            "data_license": "CC-BY 4.0",
            "instances": {"i5-C": {"sample_unit": "individual", "source": {}}},
        },
    )
    (root / "input-artifacts").mkdir()
    return root


def test_the_tree_is_read_into_studies_and_instance_facts(tree: Path) -> None:
    provenance = read_provenance(tree)

    assert provenance.instances["i5-A"] == InstanceFacts(
        study=ALPHA, sample_unit="individual", source=SOURCE
    )
    assert provenance.instances["i5-B"] == InstanceFacts(
        study=ALPHA, sample_unit="household", source={}
    )
    assert provenance.study_of == {"i5-A": ALPHA, "i5-B": ALPHA, "i5-C": BETA}
    assert set(provenance.studies) == {ALPHA, BETA}
    assert provenance.studies[ALPHA].record["title"] == "Alpha"
    assert provenance.studies[ALPHA].record["data_license"] == "CC0 1.0"
    assert set(provenance.studies[ALPHA].instances) == {"i5-A", "i5-B"}
    assert provenance.files_read == (ALPHA, BETA)


def test_a_tree_with_no_study_documents_links_nothing(tmp_path: Path) -> None:
    (tmp_path / "input-artifacts").mkdir()
    provenance = read_provenance(tmp_path)
    assert provenance.instances == {} and provenance.studies == {}
    assert study_paths(tmp_path) == []


def test_only_json_files_directly_under_the_provenance_directory_are_studies(
    tree: Path,
) -> None:
    directory = provenance_directory(tree)
    (directory / "algorithms").mkdir()
    (directory / "algorithms" / "note.json").write_text("[]", encoding="utf-8")
    (directory / ".incomplete-alpha.json").write_text("[]", encoding="utf-8")
    (directory / "readme.md").write_text("not a study", encoding="utf-8")
    assert [path.name for path in study_paths(tree)] == ["alpha.json", "beta.json"]
    assert study_ref("alpha.json") == ALPHA


def test_one_instance_named_by_two_documents_is_refused(tree: Path) -> None:
    record = json.loads((tree / BETA).read_text(encoding="utf-8"))
    record["instances"]["i5-A"] = {"sample_unit": "individual", "source": {}}
    _write(tree, BETA, record)
    with pytest.raises(ValueError, match="i5-A.*named by both"):
        read_provenance(tree)


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"study_id": "beta"},
        {"instances": []},
        {"instances": {"i5-C": "individual"}},
        {"instances": {"i5-C": {}}},
        {"instances": {"i5-C": {"sample_unit": "individual"}}},
        {"instances": {"i5-C": {"sample_unit": "individual", "source": []}}},
    ],
)
def test_a_document_no_consumer_could_read_is_refused(tree: Path, record: Any) -> None:
    (tree / BETA).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="beta.json"):
        read_provenance(tree)


def test_a_symlinked_study_document_is_refused(tree: Path) -> None:
    target = tree / BETA
    elsewhere = tree.parent / "elsewhere.json"
    elsewhere.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    target.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="not an ordinary file"):
        read_provenance(tree)


def test_a_document_the_manifest_does_not_declare_is_refused_before_it_is_read(
    tree: Path,
) -> None:
    with pytest.raises(
        ValueError, match="does not declare 'benchmark-info/provenance/beta.json'"
    ):
        read_provenance(tree, declared={ALPHA})
    assert set(read_provenance(tree, declared={ALPHA, BETA}).studies) == {ALPHA, BETA}
