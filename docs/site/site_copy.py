"""Read the site's explanatory text from the maintainer's content file.

Each entry is one slot: blank or omitted renders nothing, ``PLACEHOLDER``
renders a visible marker naming the slot, and any other text renders as
Markdown with raw HTML escaped, line breaks kept, and ``[text](url)`` a link
only to a web address, a site path or a fragment. A slot's text may name the
fields ``SITE_COPY_FIELDS`` declares for it as ``{name}``; the renderer fills
each with a value it passes, as escaped text, and ``{{``/``}}`` are literal
braces.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from markdown import Markdown
from markdown.treeprocessors import Treeprocessor

SITE_COPY_PATH = Path(__file__).resolve().parents[1] / "content" / "site-copy.yml"
PLACEHOLDER = "PLACEHOLDER"
_SCHEME = re.compile(r"^[A-Za-z][\w+.-]*:")
# A doubled brace, a field, or a brace belonging to neither.
_FIELD = re.compile(r"\{\{|\}\}|\{([A-Za-z_]\w*)\}|[{}]")
# Private-use stand-ins carry a field through Markdown untouched.
_TOKEN = re.compile("\ue000(\\d+)\ue001")
SITE_COPY_KEYS = {
    "home": {
        "task_no_standings",
        "no_release",
        "tabs_label",
        "task_link",
        "source_summary",
        "board_link",
        "board_link_title",
        "explore_link",
    },
    "leaderboard": {
        "intro",
        "intro_corpus",
        "empty_panel",
        "results_load_error",
        "missing_interpretation",
        "column_rank",
        "column_headline",
        "unreached",
        "slice_label",
        "slice_all",
        "slice_at_most",
        "slice_above",
        "slice_size",
        "slice_size_one",
        "slice_size_one_instance",
    },
    "explore": {
        "introduction",
        "trials_label",
        "filter_instances",
        "instance_count",
        "download_notebook",
        "feature_intro",
        "outcomes_intro",
        "outcomes_glossary",
        "outcomes_bar_help",
        "feature_help",
        "feature_scale_help",
        "outcomes_empty",
        "point_details_prompt",
        "empty_no_instances",
        "empty_no_algorithms",
        "empty_no_feature_values",
        "baseline_no_trials",
        "feature_unavailable_epsilon",
        "feature_unavailable_condition_number",
        "feature_unavailable_constant",
        "picker_full",
        "picker_full_placeholder",
        "histogram_excluded_reason",
        "histogram_snapped_note",
        "histogram_paired_note",
        "scatter_click_hint",
        "range_hint_whole",
        "range_hint_positive",
        "loading",
        "load_error",
        "results_loading",
        "results_load_error",
        "plot_label",
        "picker_enter_hint",
        "picker_no_match",
        "picker_transformations",
        "picker_more",
        "range_invalid_whole",
        "range_invalid_positive",
        "range_endpoint",
        "comparison_loading",
        "comparison_draw_error",
        "histogram_label",
        "histogram_label_no_threshold",
        "histogram_excluded_share",
        "bin_share",
        "bin_empty",
        "bin_instances_heading",
        "bin_show_all",
        "point_details_heading",
        "point_reading",
        "trial_count",
        "trial_count_one",
        "median_point_count",
        "median_point_count_one",
        "point_median",
        "point_on_instances",
        "point_more_instances",
        "notebook_no_view",
        "notebook_nothing_selected",
        "notebook_build_error",
        "picker_placeholder",
        "picker_count",
        "picker_suggestions",
        "picker_suggestions_one",
        "picker_browse_transformations",
        "add_algorithm",
        "selected_count",
        "outcome_algorithm_label",
        "outcome_compare_label",
        "outcome_algorithm_search",
        "outcome_compare_search",
        "outcome_clear",
        "outcome_clear_aria",
        "outcome_none",
        "view_tabs_label",
        "view_profiles",
        "view_profiles_short",
        "view_feature",
        "view_feature_short",
        "view_outcomes",
        "view_outcomes_short",
        "interpretation",
        "column_family",
        "comparison_algorithms",
        "feature_label",
        "scale_label",
        "scale_log",
        "scale_linear",
        "y_scale_label",
        "scale_quantile",
        "outcomes_guide",
        "bar_heading",
        "bin",
        "bin_label",
        "breakdown_caption",
        "column_outcome",
        "column_this_bin",
        "column_all_instances",
        "histogram_data",
        "retry_loading",
        "retry_results",
        "reload_page",
        "baseline_forget_aria",
        "baseline_algorithm_trigger",
        "baseline_picker_label",
        "baseline_algorithm_option",
        "each_trial",
        "median_per_instance",
        "each_trial_description",
        "median_per_instance_description",
        "range_samples_per_dimension",
        "range_reset",
        "range_reset_aria",
        "range_from_aria",
        "range_to_aria",
        "range_from_value_aria",
        "range_to_value_aria",
        "plotted_baseline",
        "outcome_released",
        "outcome_no_release",
        "outcome_invalid_regime",
        "outcome_timeout",
        "outcome_memory_limit",
        "outcome_algorithm_error",
        "outcome_not_run",
        "pair_both",
        "pair_a_only",
        "pair_b_only",
        "pair_neither",
        "bar_hover_success",
        "bar_hover_stack",
        "histogram_x_quantile",
        "histogram_y_axis",
        "scatter_x_quantile",
        "scatter_y_axis",
        "running_median",
        "point_hover",
        "point_beyond_view",
        "status_selected_algorithms",
        "status_selected_algorithms_one",
        "status_instances",
        "status_instances_one",
        "outcomes_lead",
        "outcome_count_title",
        "trials_recorded",
        "feature_sample_size",
        "feature_samples_per_dimension",
        "feature_vif",
    },
    "corpus": {
        "data_page_link",
        "source_heading",
        "source_variables",
        "source_coefficient",
        "source_claim",
        "source_page",
        "source_no_quote",
        "missing_citation",
        "empty_selection",
        "no_release",
        "no_sources",
        "search_label",
        "task_filter",
        "all_tasks",
        "all_fields",
        "clear_filters",
        "summary_counts",
        "source_count_one",
        "source_count_many",
        "instance_count_one",
        "instance_count_many",
        "summary_task",
        "task_instance_count_one",
        "task_instance_count_many",
        "copy_citation",
        "fact_study",
        "fact_data_licence",
        "fact_sample_unit",
        "fact_samples",
        "fact_dimension",
        "fact_samples_per_dimension",
        "fact_coefficient",
        "coefficient_column",
        "fact_weight",
        "domain_covariates",
        "domain_response",
        "domain_data",
        "columns_summary",
        "column_intercept",
        "column_range",
        "column_indicator_group",
        "indicator_zero_or_one",
        "indicator_at_most_one",
        "indicator_exactly_one",
        "coefficient_mark",
        "summary_intercept",
        "summary_groups_one",
        "summary_groups_many",
        "summary_columns_one",
        "summary_columns_many",
        "summary_ranges_one",
        "summary_ranges_many",
    },
    "task": {
        "metric_arguments_intro",
        "metric_column",
        "class_filter_all",
        "hands_off_column",
        "required_arguments",
        "baselines_label",
        "metric_arguments_heading",
        "metric_argument_column",
        "used_by_column",
        "term_more",
    },
    "component": {
        "no_implementation_docs",
        "no_structured_configuration",
        "no_provenance",
        "provenance_loading",
        "provenance_load_error",
        "details_button",
        "details_aria",
        "default_value_title",
        "changed_default",
        "full_details",
        "implementation_notes",
        "glance_label",
        "glance_native_task",
        "glance_task",
        "glance_interpretation",
        "glance_requires",
        "glance_source",
        "parameter_default",
        "parameter_required",
        "roster_algorithm_one",
        "roster_algorithm_many",
        "roster_transformation_one",
        "roster_transformation_many",
        "roster_also_in",
        "api_parameter_column",
        "api_type_column",
        "semantics_link",
    },
    "roster": {
        "tuned_source",
        "ranking_note",
        "empty_filter",
        "rank",
        "not_ranked",
        "column_at_budget",
        "search_label",
        "sort_label",
        "filters_summary",
        "core_label",
        "core_any",
        "transformation_label",
        "transformation_hint",
        "other_task_class",
        "active_filters",
        "active_class",
        "active_core",
        "active_transformation",
        "active_search",
        "remove_filter",
        "clear_all",
        "showing",
        "showing_one",
        "details_link",
        "details_aria",
        "exact_configuration",
    },
    "tutorials": {
        "index_intro",
        "notebook_unrun",
    },
    "protocol": {
        "ranking_rule",
        "baseline_best_median",
        "baseline_best_of",
        "baseline_algorithm",
        "declared_library",
        "declared_protocol",
        "role_headline",
        "role_offered",
        "rule_weighting",
        "rule_eligibility",
    },
    "site": {
        "table_show_all",
        "footer_data_link",
        "copy_selected_hint",
        "search_load_error",
        "skip_to_content",
        "open_navigation",
        "search",
        "search_shortcut",
        "search_placeholder",
        "close_search",
        "theme_light",
        "theme_dark",
        "sidebar_collapse",
        "sidebar_expand",
        "section_navigation",
        "nav_overview",
        "breadcrumb",
        "on_this_page",
        "download_notebook",
        "page_not_found",
        "not_found_home",
        "tasks_menu",
        "switch_task",
        "benchmarked_tasks",
        "other_tasks",
        "copy_code",
        "copied",
        "table_region",
        "untitled_page",
        "fact_field",
        "card_samples",
        "card_dimension",
        "fact_condition_number",
    },
    "shared": {
        "filter_algorithms",
        "privacy_budget_help",
        "algorithm_baseline",
        "chart_noscript",
        "results_unavailable_note",
        "results_unavailable_lead",
        "profile_question_individual",
        "profile_formula_individual",
        "profile_question_median",
        "profile_formula_median",
        "threshold_help",
        "quantile_handle_hint",
        "quantile_handle_offscreen_hint",
        "empty_no_curves",
        "empty_no_cores",
        "empty_no_transformation_match",
        "built_from",
        "preview_notice",
        "preview_missing_trials",
        "checklist_no_core_match",
        "checklist_no_transformation_match",
        "baseline_label",
        "preview_badge",
        "term_success_rate",
        "term_trial",
        "term_trials",
        "term_instances",
        "baseline_noscript",
        "rankings_heading",
        "plotted_heading",
        "standings_heading",
        "show_top",
        "top_all",
        "top_custom",
        "threshold_label",
        "quantile_label",
        "quantile_aria",
        "metric_label",
        "privacy_budget_label",
        "full_definition",
        "not_reported",
        "unavailability_link",
        "interpretation_line",
        "interpretation_aria_reading",
        "remove_aria",
        "close_details",
        "dialog_budget_lead",
        "dialog_rank",
        "dialog_headline",
        "privacy_interpretation",
        "filter_heading",
        "filter_dialog_aria",
        "close",
        "column_algorithm",
        "strip_summary",
        "strip_summary_one",
        "strip_summary_filtered",
        "strip_summary_filtered_one",
        "unranked_title",
        "hide_curve_title",
        "plot_curve_title",
        "hide_curve_aria",
        "plot_curve_aria",
        "headline_title",
        "headline_title_median",
        "cores_label",
        "transformations_label",
        "select_all",
        "deselect_all",
        "find_core",
        "find_transformation",
        "checklist_count",
        "select_all_cores",
        "select_all_transformations",
        "curve_hover_reading",
        "ranked_at_quantile",
        "quantile_guide",
        "threshold_guide",
        "pager_previous",
        "pager_next",
        "pager_previous_aria",
        "pager_next_aria",
        "pager_status",
    },
    "contact": {
        "title",
        "introduction",
        "addresses",
    },
    "reference": {
        "section_column",
        "task_transformations_column",
    },
    "notebook": {
        "title",
        "intro",
        "source_local",
        "source_hosted",
        "open_heading",
        "open_token",
        "read_heading",
        "read_lead",
        "filter_heading",
        "condition_note",
        "baseline_heading",
        "compare_heading",
        "compare_note",
        "curves_heading",
        "feature_heading",
        "feature_note",
        "outcomes_heading",
        "outcomes_note",
    },
}


# The fields each parameterized slot may name; every other slot takes none.
# The success-rate question and formula name the reading's terms, each a
# definable term or a name the page fills.
_READING_FIELDS = {
    "success_rate",
    "trial",
    "trials",
    "instances",
    "metric",
    "tau",
    "baseline",
}
# What each baseline is called wherever a reader meets it, by the key the
# release names it with; a key with no entry is shown as that key.
SITE_COPY_KEYS["baselines"] = {
    "ols_se",
    "ols_l2_se",
    "published_se",
    "ols_prediction_se",
    "ols_residual_rmse",
    "response_std",
    "sample_mean_l2_se",
    "sample_mean_mahalanobis_se",
    "best_median",
    "none",
}
SITE_COPY_FIELDS: dict[tuple[str, str], set[str]] = {
    ("explore", "instance_count"): {"selected", "total"},
    ("shared", "results_unavailable_lead"): {"task", "identity"},
    ("shared", "profile_question_individual"): _READING_FIELDS,
    ("shared", "profile_formula_individual"): _READING_FIELDS,
    ("shared", "profile_question_median"): _READING_FIELDS,
    ("shared", "profile_formula_median"): _READING_FIELDS,
    ("leaderboard", "results_load_error"): {"error"},
    ("leaderboard", "missing_interpretation"): {"names"},
    ("explore", "results_load_error"): {"error"},
    ("explore", "picker_transformations"): {"transformations"},
    ("explore", "picker_more"): {"count"},
    ("explore", "range_endpoint"): {"value"},
    ("explore", "comparison_draw_error"): {"error"},
    ("explore", "histogram_label"): {"bins", "categories", "feature", "tau"},
    ("explore", "histogram_label_no_threshold"): {"bins", "categories", "feature"},
    ("explore", "histogram_excluded_share"): {"share"},
    ("explore", "bin_share"): {"share"},
    ("explore", "bin_instances_heading"): {"count"},
    ("explore", "bin_show_all"): {"count"},
    ("site", "table_show_all"): {"count"},
    ("explore", "point_reading"): {"property", "x", "y"},
    ("explore", "trial_count"): {"count"},
    ("explore", "trial_count_one"): {"count"},
    ("explore", "median_point_count"): {"count"},
    ("explore", "median_point_count_one"): {"count"},
    ("explore", "point_on_instances"): {"count"},
    ("explore", "point_more_instances"): {"count"},
    ("explore", "notebook_build_error"): {"error"},
    ("roster", "ranking_note"): {"budget", "metric", "percent"},
    ("protocol", "ranking_rule"): {"baseline", "headline", "percent"},
    ("leaderboard", "column_headline"): {"percent"},
    ("leaderboard", "unreached"): {"percent"},
    ("leaderboard", "slice_at_most"): {"feature", "value"},
    ("leaderboard", "slice_above"): {"feature", "value"},
    ("leaderboard", "slice_size"): {"instances", "studies"},
    ("leaderboard", "slice_size_one"): {"instances", "studies"},
    ("leaderboard", "slice_size_one_instance"): {"instances", "studies"},
    ("protocol", "baseline_best_median"): {"aggregation", "pool"},
    ("protocol", "baseline_best_of"): {"aggregation", "algorithms"},
    ("protocol", "baseline_algorithm"): {"aggregation", "uuid"},
    ("tutorials", "notebook_unrun"): {"precondition"},
    ("component", "provenance_load_error"): {"error"},
    ("shared", "built_from"): {"identity"},
    ("shared", "preview_missing_trials"): {"expected", "missing"},
    ("site", "table_region"): {"heading"},
    ("component", "details_aria"): {"algorithm"},
    ("component", "changed_default"): {"default"},
    ("roster", "rank"): {"rank"},
    ("roster", "column_at_budget"): {"column", "budget"},
    ("roster", "other_task_class"): {"name", "task"},
    ("roster", "active_class"): {"value"},
    ("roster", "active_core"): {"value"},
    ("roster", "active_transformation"): {"value"},
    ("roster", "active_search"): {"value"},
    ("roster", "remove_filter"): {"filter"},
    ("roster", "showing"): {"shown", "total"},
    ("roster", "showing_one"): {"shown", "total"},
    ("roster", "details_aria"): {"algorithm"},
    ("component", "parameter_default"): {"default"},
    ("component", "roster_algorithm_one"): {"count", "task", "name"},
    ("component", "roster_algorithm_many"): {"count", "task", "name"},
    ("component", "roster_transformation_one"): {"count", "task", "name"},
    ("component", "roster_transformation_many"): {"count", "task", "name"},
    ("corpus", "summary_counts"): {"sources", "instances"},
    ("corpus", "source_page"): {"page"},
    ("corpus", "source_no_quote"): {"table"},
    ("corpus", "source_count_one"): {"count"},
    ("corpus", "source_count_many"): {"count"},
    ("corpus", "instance_count_one"): {"count"},
    ("corpus", "instance_count_many"): {"count"},
    ("corpus", "summary_task"): {"task", "instances"},
    ("corpus", "task_instance_count_one"): {"count"},
    ("corpus", "task_instance_count_many"): {"count"},
    ("corpus", "coefficient_column"): {"column"},
    ("corpus", "coefficient_mark"): {"kind"},
    ("corpus", "summary_groups_one"): {"count"},
    ("corpus", "summary_groups_many"): {"count"},
    ("corpus", "summary_columns_one"): {"count"},
    ("corpus", "summary_columns_many"): {"count"},
    ("corpus", "summary_ranges_one"): {"count"},
    ("corpus", "summary_ranges_many"): {"count"},
    ("notebook", "intro"): {"version", "source"},
    ("notebook", "source_hosted"): {"repository", "revision"},
    ("explore", "selected_count"): {"selected", "cap"},
    ("explore", "picker_count"): {"selected", "limit"},
    ("explore", "picker_suggestions"): {"count"},
    ("explore", "picker_suggestions_one"): {"count"},
    ("explore", "outcome_clear_aria"): {"label"},
    ("explore", "bin_label"): {"feature"},
    ("explore", "baseline_algorithm_option"): {"algorithm"},
    ("explore", "range_reset_aria"): {"axis"},
    ("explore", "range_from_aria"): {"axis"},
    ("explore", "range_to_aria"): {"axis"},
    ("explore", "range_from_value_aria"): {"axis"},
    ("explore", "range_to_value_aria"): {"axis"},
    ("explore", "plotted_baseline"): {"algorithm"},
    ("explore", "bar_hover_success"): {"share"},
    ("explore", "bar_hover_stack"): {"share"},
    ("explore", "histogram_x_quantile"): {"feature"},
    ("explore", "scatter_x_quantile"): {"feature"},
    ("explore", "scatter_y_axis"): {"baseline"},
    ("explore", "running_median"): {"algorithm"},
    ("explore", "point_hover"): {"property", "x", "y"},
    ("explore", "status_selected_algorithms"): {"count"},
    ("explore", "status_selected_algorithms_one"): {"count"},
    ("explore", "status_instances"): {"count"},
    ("explore", "status_instances_one"): {"count"},
    ("explore", "outcome_count_title"): {"count", "recorded"},
    ("explore", "trials_recorded"): {"recorded", "declared"},
    ("shared", "baseline_noscript"): {"baseline"},
    ("shared", "interpretation_line"): {"class"},
    ("shared", "interpretation_aria_reading"): {"class", "reading"},
    ("shared", "remove_aria"): {"algorithm"},
    ("shared", "dialog_budget_lead"): {"budget"},
    ("shared", "dialog_rank"): {"rank", "count"},
    ("shared", "dialog_headline"): {"value", "percent"},
    ("shared", "strip_summary"): {"ranked", "plotted"},
    ("shared", "strip_summary_one"): {"ranked", "plotted"},
    ("shared", "strip_summary_filtered"): {"shown", "ranked", "plotted"},
    ("shared", "strip_summary_filtered_one"): {"shown", "ranked", "plotted"},
    ("shared", "hide_curve_title"): {"algorithm"},
    ("shared", "plot_curve_title"): {"algorithm"},
    ("shared", "hide_curve_aria"): {"algorithm"},
    ("shared", "plot_curve_aria"): {"algorithm"},
    ("shared", "headline_title"): {"value", "percent"},
    ("shared", "headline_title_median"): {"value", "percent"},
    ("shared", "checklist_count"): {"selected", "total"},
    ("shared", "curve_hover_reading"): {"tau", "share"},
    ("shared", "ranked_at_quantile"): {"percent"},
    ("shared", "quantile_guide"): {"percent"},
    ("shared", "threshold_guide"): {"tau"},
    ("shared", "pager_status"): {"first", "last", "total"},
}


def _fields_in(place: str, text: str) -> set[str]:
    """The fields *text* names; a stray brace fails, naming *place*."""
    names: set[str] = set()
    for match in _FIELD.finditer(text):
        if match.group(0) in ("{", "}"):
            raise ValueError(f"{place} has a stray brace; write {{{{ or }}}} for one")
        if match.group(1):
            names.add(match.group(1))
    return names


@lru_cache(maxsize=8)
def _read(path: Path, modified: int) -> dict[str, dict[str, str]]:
    """Every nonblank entry in *path*, validated; *modified* keys the cache."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping of page groups to text")
    result: dict[str, dict[str, str]] = {}
    for page, entries in loaded.items():
        if page not in SITE_COPY_KEYS or not isinstance(entries, dict):
            raise ValueError(f"{path}: unknown or malformed page group {page!r}")
        for key, value in entries.items():
            if key not in SITE_COPY_KEYS[page]:
                raise ValueError(f"{path}: unknown text location {page}.{key}")
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{path}: {page}.{key} must be text or blank")
            if value and value.strip() and value.strip() != PLACEHOLDER:
                place = f"{path}: {page}.{key}"
                declared = SITE_COPY_FIELDS.get((page, key), set())
                undeclared = sorted(_fields_in(place, value) - declared)
                if undeclared:
                    raise ValueError(
                        f"{place} uses undeclared field {{{undeclared[0]}}}"
                    )
            if value and value.strip():
                result.setdefault(page, {})[key] = value.strip()
    return result


def load_site_copy(group: str, path: Path | None = None) -> dict[str, str]:
    """Return *group*'s nonblank entries; unknown keys and nontext values fail."""
    path = SITE_COPY_PATH if path is None else path
    if not path.is_file():
        return {}
    return dict(_read(path, path.stat().st_mtime_ns).get(group, {}))


def _linkable(url: str) -> bool:
    """Whether *url* may be a link target: http(s), a site path or a fragment."""
    return not _SCHEME.match(url) or url.startswith(("http://", "https://"))


class _RefuseOtherSchemes(Treeprocessor):
    """Show a link to any other scheme as the Markdown its author wrote."""

    def run(self, root):
        for parent in list(root.iter()):
            for index, child in reversed(list(enumerate(parent))):
                href = child.get("href", "")
                if child.tag != "a" or _linkable(href):
                    continue
                literal = f"[{''.join(child.itertext())}]({href})" + (child.tail or "")
                if index:
                    parent[index - 1].tail = (parent[index - 1].tail or "") + literal
                else:
                    parent.text = (parent.text or "") + literal
                parent.remove(child)


def _markdown() -> Markdown:
    """Python-Markdown with raw HTML shown as text and other schemes refused."""
    markdown = Markdown()
    markdown.preprocessors.deregister("html_block")
    markdown.inlinePatterns.deregister("html")
    # After inline links exist (20), before the serializer's tidy-up (10).
    markdown.treeprocessors.register(_RefuseOtherSchemes(markdown), "schemes", 15)
    return markdown


_MARKDOWN = _markdown()


def inline_html(text: str, pages: Mapping[str, str | None] | None = None) -> str:
    """*text* rendered as Markdown; one paragraph renders without its ``<p>``.

    *pages* are reference-link definitions, so a bare ``[Page name]`` links
    where the rendering page names that page; a page named ``None`` does not
    exist on this build, so its bare name shows as plain text. Names match
    regardless of case, and the link text stays as written.
    """
    for name, url in (pages or {}).items():
        if url is None:
            text = re.sub(
                rf"\[({re.escape(name)})\](?![(\[:])",
                lambda match: match.group(1),
                text,
                flags=re.IGNORECASE,
            )
    definitions = "".join(
        f"\n[{name}]: {url}" for name, url in (pages or {}).items() if url is not None
    )
    rendered = _MARKDOWN.reset().convert(text + "\n" + definitions)
    one_paragraph = rendered.startswith("<p>") and rendered.count("<p>") == 1
    return rendered[3:-4] if one_paragraph and rendered.endswith("</p>") else rendered


def slot_marker(group: str, key: str) -> str:
    """The visible stand-in for an unwritten slot."""
    return f"⟨{group}.{key}⟩"


def _checked(
    group: str, key: str, fields: Mapping[str, str] | None
) -> Mapping[str, str]:
    """*fields*, which must be exactly the ones the slot declares.

    A slot the content file cannot hold fails, rather than rendering blank.
    """
    if key not in SITE_COPY_KEYS.get(group, ()):
        raise ValueError(f"{group}.{key} is not a slot")
    declared = SITE_COPY_FIELDS.get((group, key), set())
    given = dict(fields or {})
    if set(given) != declared:
        raise ValueError(
            f"{group}.{key} takes fields {sorted(declared)}, not {sorted(given)}"
        )
    # A field is text and is escaped where it lands; a rendered slot passed as
    # one would show its markup to the reader.
    rendered = sorted(
        name for name, value in given.items() if "data-copy-slot=" in value
    )
    if rendered:
        raise ValueError(
            f"{group}.{key} was given rendered slot HTML in {rendered}; pass plain text"
        )
    return given


def _tokenized(text: str, fields: Mapping[str, str]) -> tuple[str, list[str]]:
    """*text* with each field a stand-in token, and the values in token order."""
    values: list[str] = []

    def replace(match: re.Match[str]) -> str:
        if match.group(1) is None:
            return match.group(0)[0]
        values.append(fields[match.group(1)])
        return f"\ue000{len(values) - 1}\ue001"

    return _FIELD.sub(replace, text), values


def copy_text(
    group: str,
    key: str,
    path: Path | None = None,
    *,
    fields: Mapping[str, str] | None = None,
) -> str:
    """One slot as plain text, for attributes and tooltips: links keep their text."""
    given = _checked(group, key, fields)
    value = load_site_copy(group, path).get(key, "")
    if value == PLACEHOLDER:
        return slot_marker(group, key)
    text, values = _tokenized(value, given)
    shown = BeautifulSoup(inline_html(text), "html.parser").get_text()
    return _TOKEN.sub(lambda match: values[int(match.group(1))], shown)


def render_slot(
    group: str,
    key: str,
    value: str,
    *,
    tag: str = "p",
    classes: str = "dp-note",
    attributes: str = "",
    pages: Mapping[str, str | None] | None = None,
    fields: Mapping[str, str] | None = None,
) -> str:
    """One slot as an element, or ``""`` where *value* is blank."""
    given = _checked(group, key, fields)
    if not value:
        return ""
    placeholder = value == PLACEHOLDER
    names = f"{classes} dp-authored-copy" + (" dp-copy-slot" if placeholder else "")
    if placeholder:
        content = html.escape(slot_marker(group, key))
    else:
        text, values = _tokenized(value, given)
        content = _TOKEN.sub(
            lambda match: html.escape(values[int(match.group(1))], quote=True),
            inline_html(text, pages),
        )
    return (
        f'<{tag} class="{names.strip()}" data-copy-slot="{group}.{key}"{attributes}>'
        f"{content}</{tag}>"
    )


def copy_html(
    group: str,
    key: str,
    *,
    tag: str = "p",
    classes: str = "dp-note",
    attributes: str = "",
    path: Path | None = None,
    pages: Mapping[str, str | None] | None = None,
    fields: Mapping[str, str] | None = None,
) -> str:
    """One slot from the content file as an element, or ``""`` where blank.

    *fields* fill the ``{name}`` fields the slot declares, as escaped text.
    """
    value = load_site_copy(group, path).get(key, "")
    return render_slot(
        group,
        key,
        value,
        tag=tag,
        classes=classes,
        attributes=attributes,
        pages=pages,
        fields=fields,
    )


def copy_span(
    group: str,
    key: str,
    *,
    path: Path | None = None,
    fields: Mapping[str, str] | None = None,
) -> str:
    """One label slot as a bare ``<span>``, for text inside another element."""
    return copy_html(group, key, tag="span", classes="", path=path, fields=fields)


def copy_markdown(
    group: str,
    key: str,
    *,
    path: Path | None = None,
    fields: Mapping[str, str] | None = None,
) -> str:
    """One slot as the Markdown its author wrote, for a document that renders it.

    Links stay Markdown links; each field is filled with its value as written.
    """
    given = _checked(group, key, fields)
    value = load_site_copy(group, path).get(key, "")
    if value == PLACEHOLDER:
        return slot_marker(group, key)
    text, values = _tokenized(value, given)
    return _TOKEN.sub(lambda match: values[int(match.group(1))], text)


# Characters Markdown reads as syntax even inside a tag on an inline line.
_MARKDOWN_SYNTAX = str.maketrans({char: f"&#{ord(char)};" for char in "[]()*_`\\!"})


def copy_attribute(*groups: str, path: Path | None = None) -> str:
    """The ``data-site-copy`` attribute a page's scripts read *groups* from.

    Markdown's syntax characters are character references, so a page whose
    line opens inline makes no link or emphasis of the slots' own Markdown.
    """
    payload = {group: load_site_copy(group, path) for group in groups}
    escaped = html.escape(json.dumps(payload, ensure_ascii=False), quote=True)
    return ' data-site-copy="' + escaped.translate(_MARKDOWN_SYNTAX) + '"'


def on_env(env: Any, **_: Any) -> Any:
    """Give every page the ``site`` group, which the scripts every page runs read.

    ``env`` is the theme's ``jinja2.Environment``; ``main.html`` writes the
    attribute on a hidden element, and its own labels through
    ``site_copy_text`` (escaped, for an attribute) and ``site_copy_span``.
    """
    env.globals["site_copy_attribute"] = copy_attribute("site")
    env.globals["site_copy_text"] = lambda key: html.escape(
        copy_text("site", key), quote=True
    )
    env.globals["site_copy_span"] = lambda key: copy_span("site", key)
    return env
