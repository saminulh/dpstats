"""Native control templates used by every Explore view.

The page controller owns the evaluation and plotted selections. These
templates own their labels, identities and accessible structure; the shared
algorithm-picker script owns search, keyboard selection and dismissal.
"""

from __future__ import annotations

import html

from site_copy import copy_span, copy_text


def algorithm_picker(identifier: str, label: str) -> str:
    """One searchable algorithm choice opened by a nearby native trigger.

    *label* is the ``explore`` slot naming the search for a screen reader.
    """
    identifier = html.escape(identifier, quote=True)
    placeholder = html.escape(copy_text("explore", "picker_placeholder"), quote=True)
    return (
        '<div class="dp-picker dp-picker-popover"'
        f' id="{identifier}" data-algorithm-picker'
        f' role="group" aria-labelledby="{identifier}-label"'
        ' hidden popover="auto">'
        f'<label class="dp-chip-label dp-visually-hidden" id="{identifier}-label"'
        f' for="{identifier}-search">{copy_span("explore", label)}</label>'
        f'<input id="{identifier}-search" type="text" autofocus'
        ' class="dp-picker-input form-control form-control-sm" role="combobox"'
        f' placeholder="{placeholder}" aria-autocomplete="list" aria-expanded="false"'
        f' autocomplete="off" aria-controls="{identifier}-options">'
        f'<ul class="dp-picker-options list-group list-group-flush" id="{identifier}-options"'
        f' role="listbox" aria-labelledby="{identifier}-label"></ul>'
        '<p class="dp-picker-count dp-visually-hidden" role="status"></p></div>'
    )


def evaluation_settings(controls: str) -> str:
    """Every evaluation control in one always-visible bar above the views.

    The page script adds the privacy budget and the bar's actions (the
    instance filter and the notebook download) to the same bar.
    """
    return (
        '<div class="dp-evaluation-settings dp-control-bar" data-evaluation-settings>'
        + controls
        + "</div>"
    )


def comparison_selection(identifier: str) -> str:
    """The selected algorithms are the Error ratio chart's editable legend."""
    key = html.escape(identifier, quote=True)
    return (
        f'<section class="dp-comparison-selection" data-comparison-selection aria-labelledby="{key}-plotted">'
        '<div class="d-flex align-items-center flex-wrap gap-2 mb-1">'
        f'<h4 class="dp-plotted-heading m-0" id="{key}-plotted">'
        + copy_span("shared", "plotted_heading")
        + "</h4>"
        '<span class="text-secondary" data-comparison-selected-count></span>'
        '<button type="button" class="btn btn-sm ms-auto"'
        f' data-comparison-add popovertarget="{key}" aria-expanded="false">'
        + copy_span("explore", "add_algorithm")
        + "</button></div>"
        + algorithm_picker(identifier, "add_algorithm")
        + '<ul class="dp-plotted-rows list-group list-group-flush"'
        f' aria-labelledby="{key}-plotted" data-comparison-algorithms></ul>'
        "</section>"
    )


def outcome_slots(identifier: str) -> str:
    """The Outcomes chart's two choices: one algorithm, and one to compare with.

    Each slot is a trigger naming its current algorithm and the shared
    algorithm search; only the comparison can be cleared.
    """
    key = html.escape(identifier, quote=True)

    def slot(role: str, label: str, search: str, clearable: bool) -> str:
        own = f"{key}-{role}"
        named = copy_text("explore", label)
        clear = html.escape(
            copy_text("explore", "outcome_clear_aria", fields={"label": named}),
            quote=True,
        )
        return (
            f'<div class="mw-100" data-outcome-slot="{role}">'
            f'<label class="form-label" id="{own}-slot-label" for="{own}-trigger">'
            + copy_span("explore", label)
            + "</label>"
            '<div class="d-flex align-items-center flex-wrap gap-1">'
            f'<button type="button" id="{own}-trigger" class="btn btn-sm text-wrap text-start mw-100"'
            f' aria-labelledby="{own}-slot-label {own}-trigger"'
            f' data-outcome-trigger popovertarget="{own}" aria-expanded="false"></button>'
            + (
                '<button type="button" class="btn btn-sm" data-outcome-clear'
                f' aria-label="{clear}">'
                + copy_span("explore", "outcome_clear")
                + "</button>"
                if clearable
                else ""
            )
            + "</div>"
            + algorithm_picker(own, search)
            + "</div>"
        )

    return (
        '<div data-outcome-slots hidden><div class="d-flex flex-wrap gap-3 mb-2">'
        + slot(
            "algorithm", "outcome_algorithm_label", "outcome_algorithm_search", False
        )
        + slot("compare", "outcome_compare_label", "outcome_compare_search", True)
        + "</div></div>"
    )


def view_tabs(chart: str) -> str:
    """The three views of an evaluation, including two on the feature panel."""
    views = ("profiles", "feature", "outcomes")
    tabs = html.escape(copy_text("explore", "view_tabs_label"), quote=True)
    return (
        '<div class="dp-task-tabs nav nav-underline" role="tablist"'
        f' aria-label="{tabs}" data-view-tabs>'
        + "".join(
            f'<button type="button" id="playground-view-{key}" data-view="{key}"'
            f' role="tab" class="nav-link{" active" if key == "profiles" else ""}"'
            f' aria-label="{html.escape(copy_text("explore", f"view_{key}"), quote=True)}"'
            f' aria-selected="{str(key == "profiles").lower()}"'
            f' tabindex="{0 if key == "profiles" else -1}"'
            f' aria-controls="{html.escape(chart, quote=True)}-{"feature" if key == "outcomes" else key}">'
            '<span class="dp-tab-label-full">'
            + copy_span("explore", f"view_{key}")
            + '</span><span class="dp-tab-label-short" aria-hidden="true">'
            + copy_span("explore", f"view_{key}_short")
            + "</span></button>"
            for key in views
        )
        + "</div>"
    )
