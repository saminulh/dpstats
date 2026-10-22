/* One algorithm typeahead for a baseline choice and additions to a plotted set.
   Callers own the selection; this module owns suggestions, the native popover,
   focus and the selection-count announcement. GitHub's combobox handles option
   navigation, active-descendant tracking and keyboard/pointer commitment. */
import Combobox from "./vendor/combobox-nav-3.0.2.js";

(function () {
  "use strict";

  const CANDIDATE_CAP = 8;

  function candidates(roster, text, selected) {
    const needle = text.trim().toLocaleLowerCase();
    if (!needle) return [];
    return roster.filter(entry => !selected.has(entry.id)).map(entry => {
      const label = entry.label.toLocaleLowerCase();
      const core = (entry.core || "").toLocaleLowerCase();
      return {entry, rank: label === needle ? 0 : core.startsWith(needle) ? 1 : label.includes(needle) ? 2 : -1};
    }).filter(found => found.rank >= 0).sort((one, two) =>
      one.rank - two.rank || (one.entry.label < two.entry.label ? -1 : 1)
    ).map(found => found.entry);
  }

  function labelled(label, needle) {
    const holder = document.createElement("span");
    holder.className = "dp-picker-label";
    const at = label.toLocaleLowerCase().indexOf(needle.toLocaleLowerCase());
    if (at < 0) { holder.textContent = label; return holder; }
    const match = document.createElement("strong");
    match.textContent = label.slice(at, at + needle.length);
    holder.append(label.slice(0, at), match, label.slice(at + needle.length));
    return holder;
  }

  window.dpAlgorithmPicker = function (root, choice) {
    const search = root.querySelector('[role="combobox"]');
    const options = root.querySelector('[role="listbox"]');
    const status = root.querySelector('[role="status"]');
    const trigger = choice.trigger;
    const popup = root;
    const selected = () => choice.selected ? choice.selected() : new Set();
    const full = () => choice.limit && selected().size >= choice.limit;
    let typing = null;
    const navigation = new Combobox(search, options, {
      tabInsertsSuggestions: false,
      firstOptionSelectionMode: "active",
    });
    const anchor = "--" + root.id + "-anchor";
    trigger.style.anchorName = anchor;
    popup.style.positionAnchor = anchor;
    // A closed native popover supplies visibility; `hidden` is only the
    // server-rendered state before this controller is ready.
    popup.hidden = false;

    function cancelTyping() { clearTimeout(typing); typing = null; }
    function clearOptions() {
      cancelTyping();
      navigation.clearSelection();
      options.replaceChildren();
      search.setAttribute("aria-expanded", "false");
    }
    function close() {
      const hadFocus = root.contains(document.activeElement);
      if (popup.matches(":popover-open")) popup.hidePopover();
      navigation.stop();
      clearOptions();
      search.value = "";
      trigger.setAttribute("aria-expanded", "false");
      if (hadFocus) trigger.focus({preventScroll: true});
    }
    function announce(suggestions) {
      search.disabled = Boolean(full());
      search.placeholder = window.dpSiteCopy(full() ? "explore.picker_full_placeholder" : "explore.picker_placeholder");
      const parts = choice.limit
        ? [window.dpSiteCopy("explore.picker_count", {selected: selected().size, limit: choice.limit})]
        : [];
      if (full()) {
        const hint = window.dpSiteCopy("explore.picker_full");
        if (hint) parts.push(hint);
      } else if (typeof suggestions === "number") {
        parts.push(window.dpSiteCopy(suggestions === 1 ? "explore.picker_suggestions_one" : "explore.picker_suggestions",
          {count: suggestions}));
        if (suggestions) parts.push(window.dpSiteCopy("explore.picker_enter_hint"));
      }
      status.textContent = parts.join(" · ");
    }
    function commit(id) {
      if (full()) return;
      // Close before changing the caller's selection: it may synchronously
      // refresh every picker on the page or change the active panel.
      close();
      search.value = "";
      choice.choose(id);
      announce();
    }
    function renderOptions() {
      clearOptions();
      const needle = search.value.trim();
      if (!needle || full()) {
        announce();
        return;
      }
      const algorithms = choice.algorithms();
      const matches = candidates(algorithms, needle, selected());
      if (!matches.length) {
        const none = document.createElement("li");
        none.className = "dp-picker-none";
        const vocabulary = [...new Set(algorithms.flatMap(entry => entry.transformations || []))].sort();
        none.append(window.dpSiteCopySpan("explore.picker_no_match"));
        if (vocabulary.length) {
          none.append(" ", window.dpSiteCopySpan("explore.picker_transformations", {transformations: vocabulary.join(", ")}), " ");
        }
        const route = root.closest('[data-transformations-url]').dataset.transformationsUrl;
        if (route) {
          const link = document.createElement("a");
          link.href = route;
          link.append(window.dpSiteCopySpan("explore.picker_browse_transformations"));
          none.append(" ", link);
        }
        options.append(none);
      }
      matches.slice(0, CANDIDATE_CAP).forEach((algorithm, position) => {
        const row = document.createElement("li");
        row.id = options.id + "-" + position;
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", "false");
        row.className = "dp-picker-option list-group-item list-group-item-action d-flex align-items-center gap-2 py-2 px-2";
        if (choice.treatment) row.append(window.dpSeriesSwatch(choice.treatment(algorithm)));
        row.append(labelled(algorithm.label, needle), window.dpInterpretationBadge(algorithm.class));
        row.dataset.algorithm = algorithm.id;
        // Keep input focus during pointer/touch selection; clicks, not a
        // mouse-only event, commit the visible option.
        row.addEventListener("pointerdown", event => event.preventDefault());
        options.append(row);
      });
      if (matches.length > CANDIDATE_CAP) {
        const more = document.createElement("li");
        more.className = "dp-picker-more";
        more.append(window.dpSiteCopySpan("explore.picker_more", {count: (matches.length - CANDIDATE_CAP).toLocaleString()}));
        options.append(more);
      }
      if (!popup.matches(":popover-open")) popup.showPopover();
      navigation.start();
      announce(matches.length);
    }
    options.addEventListener("combobox-commit", event => commit(event.target.dataset.algorithm));
    trigger.setAttribute("popovertarget", popup.id);
    trigger.setAttribute("popovertargetaction", "toggle");
    popup.addEventListener("beforetoggle", event => {
      if (event.target !== popup) return;
      trigger.setAttribute("aria-expanded", String(event.newState === "open"));
      if (event.newState !== "closed") return;
      // Native `toggle` is queued. Clearing there could cancel a new query
      // typed after dismissal but before that old close event is delivered.
      navigation.stop();
      clearOptions();
      search.value = "";
    });
    search.addEventListener("input", () => {
      clearOptions();
      if (!search.value.trim()) { renderOptions(); return; }
      typing = setTimeout(renderOptions, 120);
    });
    search.addEventListener("keydown", event => {
      if (event.isComposing) return;
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        trigger.focus({preventScroll: true});
      } else if (["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) {
        // Flush a new query before the library's key handler sees it. No
        // navigation or commitment can use an old set of suggestions.
        if (typing !== null || search.getAttribute("aria-expanded") !== "true") renderOptions();
      } else if (event.key === "Backspace" && !search.value && choice.remove) {
        const last = [...selected()].pop();
        if (last) { event.preventDefault(); choice.remove(last); }
      }
    }, {capture: true});
    root.addEventListener("focusout", () => {
      setTimeout(() => {
        if (!root.contains(document.activeElement)) close();
      }, 0);
    });
    // A picker inside a disclosure closes with it; one in an open bar has
    // nothing to follow.
    const disclosure = root.closest("details");
    if (disclosure) {
      disclosure.addEventListener("toggle", event => {
        if (!event.target.open) close();
      });
    }
    return {
      refresh: function () {
        if (full()) close();
        if (popup.matches(":popover-open")) renderOptions();
        else announce();
      },
    };
  };
})();
