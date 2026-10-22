"""Hold the results build to the payload and assembly budgets it was given.

The site emits data, not views: one index per surface and, per panel, one binary
matrix of every run's cost. What matters is therefore what a fresh page load
fetches — the index and the one matrix its opening panel reads — and how many
files a root writes at all. Disk is not the budget; a request is.
"""

from __future__ import annotations

import json


from dpstats.tasks import MeanEstimation
from leaderboard import collection_standings, leaderboard_body
from panel_data import TaskData
from playground import playground_page
from profile_chart import embed
from tests.unit.docs.conftest import parse_html

# The budget, as the results build's own contract states it: one file per panel.
MAXIMUM_FILES_PER_PANEL = 1

# What a fresh page load may fetch: its index and the one matrix its opening
# panel reads.
MAXIMUM_INDEX_BYTES = 1024 * 1024
MAXIMUM_MATRIX_BYTES = 3 * 1024 * 1024
MAXIMUM_INLINE_BYTES = 50 * 1024

# Where a surface's pages reach the files its root wrote once.
PROVENANCE = "../../data/working/provenance.json"


def test_one_root_writes_one_file_per_panel(measured_release: TaskData) -> None:
    panels = len(measured_release.panels)
    assert panels >= 1
    assert len(measured_release.files) == MAXIMUM_FILES_PER_PANEL * panels


def test_a_fresh_page_load_fetches_one_index_and_one_matrix(
    measured_release: TaskData,
) -> None:
    index = len(embed(measured_release.index).encode("utf-8"))
    assert index <= MAXIMUM_INDEX_BYTES
    for name, payload in measured_release.files.items():
        assert len(payload) <= MAXIMUM_MATRIX_BYTES, name


def test_neither_page_carries_its_data_inline(measured_release: TaskData) -> None:
    pages = [
        leaderboard_body(
            MeanEstimation.id,
            collection_standings(measured_release),
            "the fixture store",
            depth=3,
            published=False,
            provenance=PROVENANCE,
            policy=measured_release.index,
        ),
        playground_page(
            measured_release,
            "the fixture store",
            depth=3,
            provenance=PROVENANCE,
        ).body,
    ]
    for body in pages:
        for script in parse_html(body).find_all(tag="script", type="application/json"):
            blob = script.raw_text
            assert len(blob.encode("utf-8")) <= MAXIMUM_INLINE_BYTES
            # Whatever a page does inline is a panel list, not a payload.
            json.loads(blob)
