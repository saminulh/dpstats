"""Tests for the Leaderboard page and the standings it opens at."""

from __future__ import annotations

import math

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dpstats.tasks import MeanEstimation
from leaderboard import collection_standings, leaderboard_body, rank_of, slice_selector
from profile_chart import metrics_page_href
from panel_data import (
    RegisteredAlgorithm,
    TaskData,
    baseline_label,
    released_standings,
)
from results_roots import ReleasedComparison, ReleasedSlice
from templates.roster import eligible_rows
from tests.unit.docs.conftest import (
    COMPARISON,
    FIXTURE_CURVES,
    QUALIFIED_ALGORITHM,
    VIEW_BASELINE,
    WriteCopy,
    parse_html,
    write_view,
)
from release_files import (
    withheld_profiles_relative,
    write_profile_view,
)

REFERENCE_TASK = "single_coefficient_estimation"

TASK = "mean_estimation"
METRIC = "l2_error_to_ground_truth"
LOW = "eps=1.0"

# The scalars the fixture release states: one headline Metric, read at one.


def registered(interpretation: str = "standard") -> dict[str, RegisteredAlgorithm]:
    """What a store registers for each fixture curve's algorithm."""
    return {
        uuid: RegisteredAlgorithm(
            estimator="AdaSSP",
            privacy_interpretation=interpretation,
            transformations=(),
            data_bounds=False,
            fwl_demean=False,
        )
        for _, _, uuid, _, _, _ in FIXTURE_CURVES
    }


def written(
    tmp_path: Path,
    metrics: list[str] = [METRIC],
    baseline: str = VIEW_BASELINE,
) -> Path:
    return write_view(
        tmp_path, withheld_profiles_relative(TASK), metrics, TASK, baseline
    )


def rewritten(view: Path, edit: Callable[[pd.DataFrame], pd.DataFrame]) -> Path:
    """Rewrite one released view as *edit* leaves its rows, in the released shape."""
    write_profile_view(view, edit(pd.read_parquet(view)).itertuples(index=False))
    return view


def ingested(
    view: Path,
    facts: dict[str, RegisteredAlgorithm] | None = None,
    *,
    task_id: str = TASK,
    comparison: ReleasedComparison = COMPARISON,
) -> TaskData:
    """Read one released view as the curves-only panels a page reports."""
    return released_standings(
        view,
        task_id,
        comparison=comparison,
        registered=registered() if facts is None else facts,
        identity="fixture release",
        available=set(registered()),
    )


def test_component_pages_use_the_pool_the_leaderboard_ranks(
    tmp_path: Path,
) -> None:
    """A registration the protocol does not rank is a comparison, not a count."""
    facts = registered()
    for position, uuid in enumerate(facts):
        if position % 2:
            facts[uuid] = replace(facts[uuid], privacy_interpretation="theoretical")
    data = ingested(written(tmp_path), facts)

    counted = eligible_rows(data.index)
    ranked = [row for row in collection_standings(data)[0].rows if not row.diagnostic]

    assert len(counted) == len(ranked)
    assert 0 < len(counted) < len(data.index["algorithms"])


def test_a_built_site_exposes_tasks_in_navigation(
    built_site: Path,
) -> None:
    """Tasks are global navigation, with Home's picker and each task's switcher."""
    home = parse_html((built_site / "index.html").read_text(encoding="utf-8"))
    task = parse_html(
        (built_site / REFERENCE_TASK / "index.html").read_text(encoding="utf-8")
    )
    assert home.find_all(id="task-menu")
    assert home.find_all(data_task_id=REFERENCE_TASK)
    assert task.find(id="task-picker", tag="button").text == "Tasks"
    assert task.find_all(data_task_id=REFERENCE_TASK, aria_current="true")


def test_a_release_publishes_one_curve_file_per_panel(tmp_path: Path) -> None:
    data = ingested(written(tmp_path))
    standings = collection_standings(data)

    assert {panel.budget for panel in standings} == {"eps=1.0", "eps=4.0"}
    assert len(data.files) == len(standings)
    for panel, listed in zip(data.panels, data.index["panels"]):
        # A published panel ships no run matrix: its curves are the Results.
        assert panel.payload == b""
        assert listed["runs"] is None
        assert listed["curves"] in data.files
        published = json.loads(data.files[listed["curves"]])
        assert published["taus"] == [1.0, 10.0, 100.0]
        assert set(published["curves"]) == set(panel.headline)


def test_pending_evaluation_has_no_score_or_curves(tmp_path: Path) -> None:
    view = rewritten(written(tmp_path), lambda frame: frame.iloc[:0])
    data = released_standings(
        view,
        TASK,
        comparison=COMPARISON,
        registered=registered(),
        identity="fixture release",
        available=set(),
    )
    assert data.panels == []
    assert data.files == {}
    assert data.index["algorithms"] == []


def test_a_curve_without_admitted_evaluation_evidence_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="without admitted Evaluation evidence"):
        released_standings(
            written(tmp_path),
            TASK,
            comparison=COMPARISON,
            registered=registered(),
            identity="fixture release",
            available=set(),
        )


def test_admitted_evaluation_must_have_its_curve(tmp_path: Path) -> None:
    view = rewritten(written(tmp_path), lambda frame: frame.iloc[:0])
    with pytest.raises(ValueError, match="different set of algorithms"):
        ingested(view)


def test_admitted_evaluation_must_have_every_published_budget(tmp_path: Path) -> None:
    view = rewritten(
        written(tmp_path),
        lambda frame: frame[
            ~((frame["alg_uuid"] == "aaaa-1111") & (frame["budget"] == LOW))
        ],
    )
    with pytest.raises(ValueError, match="omits admitted Evaluation curves"):
        ingested(view)


def _stated(readings: dict[str, tuple[float | None, float, float | None]]):
    """Edit a view so each named algorithm states its headline reading."""

    def edit(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for name, values in readings.items():
            chosen = frame["algorithm"] == name
            for column, value in zip(
                ("tau_at_headline", "share_reached", "tau_at_share_reached"), values
            ):
                frame.loc[chosen, column] = math.nan if value is None else value
        return frame

    return edit


def test_the_released_headline_reading_is_the_standing(tmp_path: Path) -> None:
    """A release's standings are its own numbers: a lower τ at the quantile
    ranks first, and one that never reaches it ranks after every one that does,
    by the share it reaches, then by the τ at which it reaches that share."""
    view = rewritten(
        written(tmp_path),
        _stated(
            {
                "Alg Alpha": (None, 0.8, 4.0),
                "Alg Beta": (None, 0.8, 3.0),
                "Alg Gamma": (2.5, 1.0, 9.0),
            }
        ),
    )
    standings = collection_standings(ingested(view))
    low = [panel for panel in standings if panel.budget == LOW][0]

    assert [(row.label, row.headline.tau) for row in low.rows] == [
        ("Alg Gamma", 2.5),
        ("Alg Beta", math.inf),
        ("Alg Alpha", math.inf),
    ]
    ranked = [row for row in low.rows if not row.diagnostic]
    assert [rank_of(row, ranked) for row in ranked] == [1, 2, 3]


def test_a_view_stating_two_readings_for_one_curve_is_refused(tmp_path: Path) -> None:
    def split(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame.loc[frame.index[0], "share_reached"] = 0.123
        return frame

    with pytest.raises(ValueError, match="two headline readings"):
        ingested(rewritten(written(tmp_path), split))


@pytest.mark.parametrize(
    ("page", "depth", "expected"),
    [
        ("the site root Home", 0, "mean_estimation/metrics/#l2-error-to-ground-truth"),
        (
            "a task page or a design preview, both climbing 2 levels to root",
            2,
            "../../mean_estimation/metrics/#l2-error-to-ground-truth",
        ),
    ],
)
def test_the_metrics_link_always_addresses_from_the_site_root(
    page: str, depth: int, expected: str
) -> None:
    """One formula reaches the Metrics page from anywhere it is linked from.

    The site root Home is depth 0; a real task page (``{task}/leaderboard/``)
    and a design preview (``templates/t04-leaderboard/``) both climb 2 levels
    to reach the site root — the borrowed depth a preview shares with a real
    task page is numerically identical, so there is no second, page-kind-aware
    code path left to distinguish them: every caller climbs *depth*, then
    descends by the task's own id.
    """
    assert metrics_page_href(depth, TASK, METRIC) == expected, page


def test_the_leaderboard_page_threads_task_id_and_depth_into_that_link(
    tmp_path: Path,
) -> None:
    """The rendered page's own metric-definition link is that same formula."""
    data = ingested(written(tmp_path))
    body = leaderboard_body(
        TASK,
        collection_standings(data),
        "fixture",
        depth=2,
        published=True,
        provenance="provenance.json",
        policy=data.index,
    )
    href = parse_html(body).find("data-definition-href").attrs["data-definition-href"]
    assert href == metrics_page_href(2, TASK, METRIC)


@pytest.mark.parametrize(
    # Each a baseline the library or the protocol declares, as a release
    # view may only name one.
    "label",
    ["sample_mean_l2_se", "none", VIEW_BASELINE],
)
def test_the_page_names_the_baseline_the_view_was_taken_against(
    tmp_path: Path, label: str
) -> None:
    """The release names it; the page shows the site's name for it."""
    data = ingested(written(tmp_path, baseline=label))
    shown = baseline_label(label)

    body = leaderboard_body(
        TASK,
        collection_standings(data),
        "fixture",
        depth=3,
        published=True,
        provenance="provenance.json",
        policy=data.index,
    )

    page = parse_html(body)
    # A fixed value reads as text, not as a control the reader cannot use.
    control = page.find("data-official-baseline")
    assert not control.find_all(tag="select") and not control.find_all(tag="input")
    assert page.find("data-official-baseline-label").text == shown
    assert not page.find_all("data-baseline-algorithm-trigger")
    charted = {
        chart.attrs["data-baseline-label"]
        for chart in page.find_all("data-baseline-label")
    }
    assert charted == {shown}
    (top,) = page.find_all("data-top-n", tag="select")
    assert [option.attrs["value"] for option in top.find_all("selected")] == ["10"]


def test_shared_labels_remain_distinguishable_across_budgets(tmp_path: Path) -> None:
    view = rewritten(
        written(tmp_path),
        lambda rows: rows.replace(
            {"algorithm": {"Alg Alpha": "Shared", "Alg Beta": "Shared"}}
        ),
    )
    standings = collection_standings(ingested(view))
    for panel in standings:
        labels = {row.label for row in panel.rows}
        assert {"Shared (aaa-1111)", "Shared (bbb-2222)"} <= labels


def test_a_released_standing_carries_its_own_coverage_counts(tmp_path: Path) -> None:
    data = ingested(written(tmp_path))
    standings = collection_standings(data)
    low = [panel for panel in standings if panel.budget == LOW][0]
    counts = {row.label: (row.declared, row.measured) for row in low.rows}

    assert counts["Alg Beta"] == (8, 6)
    assert data.index["coverage"]["all"][f"{METRIC}/{LOW}"][low.rows[0].uuid] == [
        low.rows[0].declared,
        low.rows[0].measured,
    ]


def test_a_comparison_claim_is_listed_after_the_standings(tmp_path: Path) -> None:
    facts = registered()
    first = FIXTURE_CURVES[0][2]
    facts[first] = RegisteredAlgorithm(
        estimator="OLS",
        privacy_interpretation="nonprivate",
        transformations=(),
        data_bounds=False,
        fwl_demean=False,
    )
    standings = collection_standings(ingested(written(tmp_path), facts))
    low = [panel for panel in standings if panel.budget == LOW][0]

    assert low.rows[-1].uuid == first
    assert low.rows[-1].diagnostic


def test_a_view_publishing_another_task_is_refused(tmp_path: Path) -> None:
    view = written(tmp_path)
    with pytest.raises(ValueError, match="publishes rows for task"):
        ingested(view, task_id="median_estimation")


def test_a_view_that_is_not_the_released_shape_is_refused(tmp_path: Path) -> None:
    """A view is read against the columns this build reads, not whatever a
    file happens to carry: the library binds its bytes and nothing more."""
    view = written(tmp_path)
    pd.read_parquet(view)[["task", "algorithm"]].to_parquet(view, index=False)

    with pytest.raises(ValueError, match="does not carry the columns and types"):
        ingested(view)


def test_a_view_storing_a_threshold_as_text_is_refused(tmp_path: Path) -> None:
    """Each column is read as the type the release stores it as, never coerced."""
    view = written(tmp_path)
    rows = pd.read_parquet(view)
    rows.assign(tau=rows["tau"].map(repr)).to_parquet(view, index=False)

    with pytest.raises(ValueError, match="does not carry the columns and types"):
        ingested(view)


def test_a_view_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    """A view in any other format, CSV included, is refused rather than parsed."""
    view = written(tmp_path)
    pd.read_parquet(view).to_csv(view, index=False)

    with pytest.raises(ValueError, match="not readable Parquet"):
        ingested(view)


def test_a_written_view_stores_no_pandas_metadata(tmp_path: Path) -> None:
    """A synthetic view's bytes follow its rows, not the writing environment."""
    view = written(tmp_path)

    assert pq.read_schema(view).metadata is None
    assert b"pandas" not in view.read_bytes()


def test_a_view_repeating_one_threshold_is_refused(tmp_path: Path) -> None:
    view = rewritten(written(tmp_path), lambda rows: pd.concat([rows, rows.iloc[[0]]]))

    with pytest.raises(ValueError, match="repeats"):
        ingested(view)


def test_a_view_leaving_one_curve_short_of_a_threshold_is_refused(
    tmp_path: Path,
) -> None:
    view = rewritten(written(tmp_path), lambda rows: rows.iloc[1:])

    with pytest.raises(ValueError, match="without thresholds"):
        ingested(view)


# ── A measured store ─────────────────────────────────────────────────────────


def test_a_measured_standing_reads_its_panel_run_matrix(
    measured_release: TaskData,
) -> None:
    standings = collection_standings(measured_release)
    assert standings
    assert len(standings) == len(measured_release.panels)
    for panel, listed in zip(standings, measured_release.index["panels"]):
        assert (panel.metric, panel.budget) == (listed["metric"], listed["budget"])
        # A measured panel derives every reading from one run matrix.
        assert listed["runs"] in measured_release.files
        assert listed.get("curves") is None
        assert all(0.0 <= row.headline.share <= 1.0 for row in panel.rows)


def test_only_the_ruled_eligibility_class_is_ranked(measured_release: TaskData) -> None:
    """A class the protocol does not rule eligible takes no rank.

    The fixture store registers the same stand-in twice, once at a standard
    interpretation and once at a qualified one, so the two differ in nothing a
    ranking could read except the claim their privacy is interpreted under.
    """
    standings = collection_standings(measured_release)

    for panel in standings:
        ranked = {row.label for row in panel.rows if not row.diagnostic}
        assert QUALIFIED_ALGORITHM not in ranked
        assert "Alg Beta" in ranked
    qualified = [
        row
        for panel in standings
        for row in panel.rows
        if row.label == QUALIFIED_ALGORITHM
    ]
    assert qualified and all(row.diagnostic for row in qualified)


def test_a_view_naming_two_baselines_for_one_metric_is_refused(
    tmp_path: Path,
) -> None:
    """One Metric is read against one baseline, whatever the release calls it."""

    def renamed(rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        rows.loc[rows.index[0], "baseline"] = "another baseline"
        return rows

    view = rewritten(written(tmp_path), renamed)

    with pytest.raises(ValueError, match="against both"):
        ingested(view)


def test_the_measured_standing_is_the_panel_headline_reading(
    measured_release: TaskData,
) -> None:
    standings = collection_standings(measured_release)
    for panel, data in zip(standings, measured_release.panels):
        for row in panel.rows:
            assert row.headline == data.headline[row.uuid]


def test_the_page_names_its_datasets_and_inlines_no_curve(
    measured_release: TaskData,
) -> None:
    body = leaderboard_body(
        MeanEstimation.id,
        collection_standings(measured_release),
        "the fixture store",
        depth=3,
        published=False,
        provenance="../../data/working/provenance.json",
        policy=measured_release.index,
    )
    page = parse_html(body)
    root = page.find("data-index")
    assert root.attrs["data-index"] == "index.json"
    assert root.attrs["data-algorithm-provenance"] == (
        "../../data/working/provenance.json"
    )
    assert root.attrs["data-task"] == MeanEstimation.id
    sections = page.find_all("data-results-section", tag="script")
    assert sections
    for section in sections:
        listed = json.loads(section.raw_text)
        assert listed["chart"]
        assert listed["budgets"]
    for entry in measured_release.index["panels"]:
        assert entry["runs"] not in body
    scripts = {
        Path(script.attrs["src"]).name for script in page.find_all("src", tag="script")
    }
    assert {"leaderboard.js", "panel-data.js"} <= scripts


def test_the_introduction_links_its_bare_page_names(
    measured_release: TaskData, write_copy: WriteCopy
) -> None:
    """Each bare page name links where the build has that page."""
    write_copy(
        {
            "leaderboard": {
                "intro": "The [roster], the [Methodology] page, the [Submit] page.",
                "intro_corpus": "The [public corpus], the [Explore] page.",
            }
        }
    )

    def intro(explore: str | None) -> list[tuple[str, str]]:
        body = leaderboard_body(
            MeanEstimation.id,
            collection_standings(measured_release),
            "the fixture store",
            depth=2,
            published=True,
            provenance="provenance.json",
            policy=measured_release.index,
            explore_href=explore,
            roster_href="../roster/",
            methodology_href="../../methodology/",
        )
        nodes = [
            node
            for node in parse_html(body).find_all("data-copy-slot")
            if node.attrs["data-copy-slot"]
            in ("leaderboard.intro", "leaderboard.intro_corpus")
        ]
        assert len(nodes) == 2
        assert all("[" not in node.text for node in nodes)
        return [
            (link.text, link.attrs["href"])
            for node in nodes
            for link in node.find_all(tag="a")
        ]

    assert intro("../public-results/") == [
        ("roster", "../roster/"),
        ("Methodology", "../../methodology/"),
        ("Submit", "../../submit/"),
        ("public corpus", "../../corpus/?task=mean_estimation"),
        ("Explore", "../public-results/"),
    ]
    assert intro(None) == [
        ("roster", "../roster/"),
        ("Methodology", "../../methodology/"),
        ("Submit", "../../submit/"),
        ("public corpus", "../../corpus/?task=mean_estimation"),
    ]


def test_the_page_ships_each_class_reading_from_the_library(
    measured_release: TaskData,
) -> None:
    """Badges drawn in the browser read the readings PrivacyInterpretation documents."""
    body = leaderboard_body(
        MeanEstimation.id,
        collection_standings(measured_release),
        "the fixture store",
        depth=2,
        published=True,
        provenance="provenance.json",
        policy=measured_release.index,
    )
    (host,) = parse_html(body).find_all("data-privacy-interpretations")
    readings = json.loads(host.attrs["data-privacy-interpretations"])
    assert set(readings) == {
        "standard",
        "qualified",
        "theoretical",
        "informal",
        "diagnostic",
        "nonprivate",
    }
    assert readings["qualified"] == (
        "some other privacy guarantee holds, such as replace adjacency."
    )
    assert readings["nonprivate"] == (
        "an explicit control, unnoised output, or NO_PRIVACY execution."
    )


def test_the_page_opens_on_a_ranked_table(measured_release: TaskData) -> None:
    body = leaderboard_body(
        MeanEstimation.id,
        collection_standings(measured_release),
        "the fixture store",
        depth=3,
        published=False,
        provenance="../../data/working/provenance.json",
        policy=measured_release.index,
    )
    page = parse_html(body)
    headers = [header.text for header in page.find_all(tag="th", scope="col")]
    assert headers == [
        "⟨leaderboard.column_rank⟩",
        "Algorithm",
        "⟨leaderboard.column_headline⟩",
    ] * (len(headers) // 3)
    assert page.find_all(tag="input", type="range")


@pytest.mark.parametrize(
    "interpretation",
    [
        "standard",
        "qualified",
        "theoretical",
        "informal",
        "diagnostic",
        "nonprivate",
        None,
    ],
)
def test_official_pages_admit_only_the_class_the_protocol_ranks(
    tmp_path: Path, interpretation: str | None
) -> None:
    facts = registered()
    first = FIXTURE_CURVES[0][2]
    if interpretation is None:
        # A store that registers nothing about an algorithm registers no entry.
        del facts[first]
    else:
        facts[first] = RegisteredAlgorithm(
            estimator="AdaSSP",
            privacy_interpretation=interpretation,
            transformations=(),
            data_bounds=False,
            fwl_demean=False,
        )
    data = ingested(written(tmp_path), facts)
    standings = collection_standings(data)
    entry = next(entry for entry in data.index["algorithms"] if entry["uuid"] == first)
    assert entry["diagnostic"] is (interpretation != "standard")
    assert data.index["visible_classes"] == ["standard"]
    # Excluded comparisons remain in the shared source for Explore.
    assert any(first in json.loads(value)["curves"] for value in data.files.values())
    for home in (False, True):
        body = leaderboard_body(
            TASK,
            standings,
            "fixture",
            depth=0,
            published=True,
            provenance="provenance.json",
            policy=data.index,
            home=home,
        )
        label = next(
            row.label for panel in standings for row in panel.rows if row.uuid == first
        )
        cells = [cell.text for cell in parse_html(body).find_all(tag="td")]
        assert (label in cells) is (interpretation == "standard")
        if interpretation is None:
            assert "⟨leaderboard.missing_interpretation⟩" in parse_html(body).text


SLICED = replace(COMPARISON, slices=(ReleasedSlice("d-le-10", "d", 10.0, False),))


def _sliced(frame: pd.DataFrame) -> pd.DataFrame:
    """The view's curves for all of the task (8 Instances, 4 studies), and
    again for the d ≤ 10 slice (3 Instances, 2 studies), where Alpha alone
    reaches 90%."""
    whole = frame.assign(slice="all", slice_instance_count=8, slice_study_count=4)
    part = frame.assign(slice="d-le-10", slice_instance_count=3, slice_study_count=2)
    part["tau_at_headline"] = [
        2.0 if name == "Alg Alpha" else math.nan for name in part["algorithm"]
    ]
    part["share_reached"] = 0.5
    part["tau_at_share_reached"] = 4.0
    part.loc[part["algorithm"] == "Alg Alpha", "share_reached"] = 1.0
    return pd.concat([whole, part])


def test_each_slice_ranks_by_its_own_readings(tmp_path: Path) -> None:
    data = ingested(rewritten(written(tmp_path), _sliced), comparison=SLICED)
    assert {panel.slice for panel in data.panels} == {"all", "d-le-10"}
    assert [entry["key"] for entry in data.index["slices"]] == ["all", "d-le-10"]
    assert data.index["slices"][1] == {
        "key": "d-le-10",
        "feature": "d",
        "at_most": 10.0,
        "instances": 3,
        "studies": 2,
    }
    assert (
        data.index["slices"][0]["instances"],
        data.index["slices"][0]["studies"],
    ) == (8, 4)
    part = data.index["headline"]["d-le-10"][f"{METRIC}/{LOW}"]
    assert part[FIXTURE_CURVES[0][2]] == [2.0, 1.0, 4.0]
    assert part[FIXTURE_CURVES[1][2]] == ["Infinity", 0.5, 4.0]
    files = {entry["curves"] for entry in data.index["panels"]}
    assert len(files) == len(data.panels)
    # The pages open on the whole task.
    assert {panel.slice for panel in data.panels if panel.headline} >= {"all"}
    assert all(panel.rows for panel in collection_standings(data))


def test_a_view_whose_slices_are_not_the_protocols_is_refused(tmp_path: Path) -> None:
    view = rewritten(written(tmp_path), _sliced)
    with pytest.raises(ValueError, match="not the protocol's"):
        ingested(view)


def test_two_sizes_for_one_slice_are_refused(tmp_path: Path) -> None:
    def uneven(frame: pd.DataFrame) -> pd.DataFrame:
        frame = _sliced(frame)
        frame.iloc[0, frame.columns.get_loc("slice_study_count")] = 9
        return frame

    with pytest.raises(ValueError, match="two sizes"):
        ingested(rewritten(written(tmp_path), uneven), comparison=SLICED)


def test_the_slice_selector_offers_each_slice_and_opens_on_all() -> None:
    slices = [
        {"key": "all", "instances": 85, "studies": 43},
        {
            "key": "d-le-10",
            "feature": "d",
            "at_most": 10,
            "instances": 37,
            "studies": 23,
        },
        {
            "key": "nd-gt-100",
            "feature": "n_over_d",
            "above": 100,
            "instances": 37,
            "studies": 22,
        },
    ]
    page = parse_html(slice_selector(slices, scope="linear_regression"))
    radios = page.find_all(tag="input", type="radio")
    assert [radio.attrs["value"] for radio in radios] == ["all", "d-le-10", "nd-gt-100"]
    assert "checked" in radios[0].attrs
    assert page.find_all("data-slice-select")
    assert slice_selector(slices[:1], scope="linear_regression") == ""


def test_the_slice_selector_names_each_slice_and_states_each_size_once(
    write_copy: WriteCopy,
) -> None:
    write_copy(
        {
            "leaderboard": {
                "slice_all": "All",
                "slice_above": "{feature} > {value}",
                "slice_size": "{instances} of {studies} many",
                "slice_size_one": "{instances} of {studies} one",
                "slice_size_one_instance": "{instances} of {studies} lone",
            }
        }
    )
    slices = [
        {"key": "all", "instances": 85, "studies": 43},
        {"key": "d-gt-10", "feature": "d", "above": 10, "instances": 4, "studies": 1},
        {
            "key": "nd-gt-100",
            "feature": "n_over_d",
            "above": 100,
            "instances": 1,
            "studies": 1,
        },
    ]
    page = parse_html(slice_selector(slices, scope="linear_regression"))
    labels = [label.text.strip() for label in page.find_all(tag="label")]
    assert labels == ["All", "d > 10", "n/d > 100"]
    (size,) = page.find_all("data-slice-sizes")
    assert size.text.strip() == "85 of 43 many"
    assert json.loads(size.attrs["data-slice-sizes"]) == {
        "all": "85 of 43 many",
        "d-gt-10": "4 of 1 one",
        "nd-gt-100": "1 of 1 lone",
    }
