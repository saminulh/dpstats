"""Corpus provenance: which study each Instance came from, and what is recorded about it.

A tree (a Working store's ``local/``, a Distribution or Evaluation Release, a
producer's candidate bundle) keeps one study document per source at
``benchmark-info/provenance/<study>.json``: the study's own facts (title,
authors, DOIs, BibTeX, abstract, ``discipline``, ``data_license``) and an
``instances`` object keyed by Instance UUID, each entry holding the Instance's
``sample_unit`` and a ``source`` block copied verbatim from the producer's
records (Stata variable names, the coefficient of interest, the routed claims
with their evidence digests). An absent value is an absent key.

This module is the one reader of that link and those facts: every
application, maintenance and site consumer asks :func:`read_provenance`
instead of parsing the files itself. The reader interprets nothing: a licence
policy, a weighting or a lane is a consumer's decision over the facts
returned here, and the producer's full shape check is the operator
application's.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dpstats.transport.documents import STAGING_PREFIX
from dpstats.workspace.layout import (
    BENCHMARK_INFO_DIRECTORY,
    PROVENANCE_DIRECTORY,
    provenance_directory,
)


@dataclass(frozen=True)
class InstanceFacts:
    """What the producer recorded about one Instance.

    Attributes:
        study: The tree-root-relative ref of the study document the Instance
            came from (``benchmark-info/provenance/<study>.json``).
        sample_unit: The protected unit one row of the Instance's data stands for.
        source: The verbatim source block (``variables``, ``coefficient``,
            ``claims``, each present only where the producer recorded it).
    """

    study: str
    sample_unit: str
    source: Mapping[str, Any]


@dataclass(frozen=True)
class Study:
    """One study document and the Instances that came from it.

    Attributes:
        ref: The tree-root-relative ref of the study document.
        record: The document as its producer wrote it, ``instances`` included.
        instances: The facts of every Instance the document names, keyed by
            Instance UUID.
    """

    ref: str
    record: Mapping[str, Any]
    instances: Mapping[str, InstanceFacts]


@dataclass(frozen=True)
class Provenance:
    """Everything one tree records about where its Instances came from.

    Attributes:
        studies: Every study document in the tree, keyed by ref.
        instances: Every named Instance's facts, keyed by Instance UUID.
        files_read: The tree-root-relative paths this read opened, in order.
    """

    studies: Mapping[str, Study]
    instances: Mapping[str, InstanceFacts]
    files_read: tuple[str, ...]

    @property
    def study_of(self) -> dict[str, str]:
        """Return each Instance UUID mapped to the ref of its study."""
        return {uuid: facts.study for uuid, facts in self.instances.items()}


def study_ref(name: str) -> str:
    """Return the tree-root-relative ref of the study document file *name*."""
    return f"{BENCHMARK_INFO_DIRECTORY}/{PROVENANCE_DIRECTORY}/{name}"


def study_paths(tree_root: Path) -> list[Path]:
    """List a tree's study documents: the JSON files directly under its provenance directory.

    Algorithm notes and other material live in subdirectories, and staging
    residue is an interrupted write; neither is a study document.
    """
    directory = provenance_directory(tree_root)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.startswith(STAGING_PREFIX)
    )


def read_provenance(
    tree_root: Path, *, declared: Collection[str] | None = None
) -> Provenance:
    """Read every study document in a tree, with each named Instance's facts.

    The read is whole or refused: no document and no entry is skipped.

    Args:
        tree_root: A ``local/`` store, a release bundle or a candidate bundle.
        declared: When given, the tree-root-relative paths this read may open
            (a release manifest's file list); a study document outside it is
            refused before it is opened.

    Returns:
        The tree's studies and Instance facts.

    Raises:
        ValueError: A study document is a symlink, is not one JSON object, has
            no ``instances`` object, names an Instance whose entry is not an
            object with a ``sample_unit`` string and a ``source`` object, names
            an Instance another document also names, or is outside *declared*.

    Example:
        >>> provenance = read_provenance(Path("official-release"))  # doctest: +SKIP
        >>> provenance.study_of  # doctest: +SKIP
        {'i5-...': 'benchmark-info/provenance/<study>.json'}
    """
    files_read: list[str] = []
    instances: dict[str, InstanceFacts] = {}
    studies: dict[str, Study] = {}
    for path in study_paths(tree_root):
        ref = study_ref(path.name)
        if declared is not None and ref not in declared:
            raise ValueError(
                f"The release manifest does not declare {ref!r}, so it is not read."
            )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Study document {ref} is not an ordinary file.")
        files_read.append(ref)
        record = _read_document(path, ref)
        members: dict[str, InstanceFacts] = {}
        for uuid, entry in record["instances"].items():
            if uuid in instances:
                raise ValueError(
                    f"Instance {uuid!r} is named by both {instances[uuid].study} "
                    f"and {ref}; one Instance belongs to one study."
                )
            facts = InstanceFacts(
                study=ref,
                sample_unit=entry["sample_unit"],
                source=entry["source"],
            )
            instances[uuid] = facts
            members[uuid] = facts
        studies[ref] = Study(ref=ref, record=record, instances=members)
    return Provenance(
        studies=studies, instances=instances, files_read=tuple(files_read)
    )


def _read_document(path: Path, ref: str) -> dict[str, Any]:
    """Read one study document, refusing a shape no consumer could read."""
    record: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("instances"), dict):
        raise ValueError(
            f"Study document {ref} is not one JSON object with an instances object."
        )
    # A decoded JSON object's names are strings and its values any JSON value.
    document = cast(dict[str, Any], record)
    for uuid, entry in cast(dict[str, Any], document["instances"]).items():
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("sample_unit"), str)
            or not isinstance(entry.get("source"), dict)
        ):
            raise ValueError(
                f"Study document {ref}: the entry for {uuid!r} is not an object "
                "with a sample_unit string and a source object."
            )
    return document
