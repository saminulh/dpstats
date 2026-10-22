/* Feed build-computed profile-chart payloads to the vendored Plotly bundle.
   This file only draws them. Static pages embed their data; the playground
   controller supplies its selected metric/budget payload on demand.

   One algorithm keeps one colour: `series_style`'s checked-in table keys it on
   the core estimator, dash names the privacy interpretation, and marker shape
   separates the configurations of one core. Only emphasis follows the reading
   on screen. The rankings table and the plotted-set strip draw the same
   swatch, so the three surfaces cannot disagree about which curve is which. */
(function () {
  "use strict";

  /* A segmented radio group (a `<fieldset>` of native radios) reads and
     writes like the `<select>` it may fall back to instead, so a controller
     that reads `.value`/listens for `change` never has to know which one a
     page rendered. Native radio `change` events already bubble to the
     fieldset on their own; only `.value` needs the adapter, and only once
     per element. */
  function asValueControl(el) {
    if (el.tagName === 'SELECT' || el.dpValueControl) return el;
    el.dpValueControl = true;
    Object.defineProperty(el, 'value', {
      get: function () {
        var checked = el.querySelector('input[type="radio"]:checked');
        return checked ? checked.value : '';
      },
      set: function (next) {
        var radios = el.querySelectorAll('input[type="radio"]');
        for (var index = 0; index < radios.length; index += 1) {
          radios[index].checked = radios[index].value === next;
        }
      },
    });
    return el;
  }
  window.dpAsValueControl = asValueControl;

  /* Contextual help opens from the control's own label rather than from a
     separate icon beside it, so the label a reader already reads is the
     thing they point at or tab to. */
  function helpLabel(label, content) {
    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = label.className + ' dp-help-label';
    // A group named by this label keeps pointing at it.
    if (label.id) trigger.id = label.id;
    trigger.append.apply(trigger, Array.from(label.childNodes));
    label.replaceWith(trigger);
    window.dpContextualHelp(trigger, content);
    /* A label with nothing to say is not a trigger: it drops out of the tab
       order and out of the popover's reach rather than opening an empty box. */
    trigger.dpOffersHelp = function (available) {
      trigger.classList.toggle('dp-has-help', available);
      trigger.tabIndex = available ? 0 : -1;
      if (available) {
        trigger.setAttribute('popovertarget', content.id);
        trigger.setAttribute('aria-describedby', content.id);
      } else {
        trigger.removeAttribute('popovertarget');
        trigger.removeAttribute('aria-describedby');
      }
    };
    trigger.dpOffersHelp(Boolean(content.textContent.trim()));
    return trigger;
  }

  /* Shared comparison chrome. Page controllers retain their own selection
     rules; neither owns a second metric/budget/baseline layout. */
  window.dpResultsControls = function (page, sections, index) {
    var select = asValueControl(page.querySelector('[data-metric-select]'));
    var top = page.querySelector('[data-top-n]');
    var topLabel = top.closest('.dp-shared-baseline');
    if (page.closest('.dp-home')) topLabel.hidden = true;
    page.dpTopCustom = function () { top.value = 'custom'; };
    // A budget choice belongs to the page, not to one metric's cached panel.
    page.addEventListener('dp-budget-change', function (event) {
      sections.forEach(function (section) {
        if (section !== event.target && section.dpSelectBudget) {
          section.dpSelectBudget(event.detail.budget, false);
        }
      });
    });
    // Each setting box's Details link follows its choice, reading the exact
    // page-relative anchor the build wrote on that choice's own section —
    // this file never slugifies a label into a link.
    var metricDefinition = page.querySelector('[data-metric-summary-link]');
    var baselineDefinition = page.querySelector('[data-baseline-summary-link]');
    var termsHost = page.querySelector('[data-reader-terms]');
    var terms = {};
    try { terms = JSON.parse(termsHost ? termsHost.dataset.readerTerms : '{}'); } catch (error) { terms = {}; }
    // Each control's label opens its help: the maintainer's meaning of the
    // term alone. What the selected value means is its box's description,
    // never this help (W93).
    [
      ['.dp-metric-control .dp-chip-label', 'metric', 'dp-metric'],
      ['.dp-baseline-control .dp-chip-label', 'baseline', 'dp-baseline'],
    ].forEach(function (spec) {
      var label = page.querySelector(spec[0]);
      var text = terms[spec[1]] && terms[spec[1]].text;
      if (!label || !text) return;
      var help = document.createElement('div');
      help.id = (label.id || spec[2]) + '-help';
      help.append(text);
      helpLabel(label, help);
    });
    // The selected metric's one-sentence definition and its link, under the bar.
    var summary = page.querySelector('[data-metric-summary]');
    var summaries = {};
    try { summaries = JSON.parse(summary ? summary.dataset.summaries : '{}'); } catch (error) { summaries = {}; }
    function describeMetric(section) {
      if (!summary) return;
      // Inner HTML the build rendered from the library's own docstring.
      summary.querySelector('[data-metric-summary-text]').innerHTML = summaries[section.dataset.metric] || '';
      summary.querySelector('[data-metric-summary-link]').href = section.dataset.definitionHref || '';
    }
    // The Baseline box's description says what the chosen baseline is.
    var baselineDescription = page.querySelector('[data-baseline-summary-text]');
    function describeBaseline(metric, value) {
      if (!baselineDescription) return;
      var entry = index.baselines[metric].options.find(function (option) {
        return option.key === value;
      });
      if (entry) {
        // Escaped markup from the protocol page's own definition renderer.
        baselineDescription.innerHTML = entry.description || '';
        baselineDescription.classList.remove('dp-authored-copy', 'dp-copy-slot');
        delete baselineDescription.dataset.copySlot;
      } else {
        window.dpSiteCopyFill(baselineDescription, 'shared.algorithm_baseline');
      }
    }
    // The Trial aggregation box's description follows its choice.
    var trials = page.querySelector('select[data-trial-aggregation]');
    var trialsSummary = page.querySelector('[data-trials-summary]');
    function describeTrials() {
      if (!trials || !trialsSummary) return;
      window.dpSiteCopyFill(trialsSummary, trials.value === 'median'
        ? 'explore.median_per_instance_description'
        : 'explore.each_trial_description');
    }
    if (trials) trials.addEventListener('change', describeTrials);
    var shared = document.createElement('div');
    shared.className = 'dp-filters dp-shared-controls';
    var controls = page.querySelector('.dp-comparison-controls');
    // Explore gathers every control into one bar; elsewhere the budget row
    // follows the comparison controls.
    var bar = controls.closest('[data-evaluation-settings]');
    if (bar) bar.appendChild(shared); else controls.after(shared);
    var privacy = new Map();
    sections.forEach(function (section) {
      var controls = section.querySelector(':scope > .dp-filters');
      if (controls) {
        controls.classList.add('dp-privacy-filters');
        controls.hidden = section.hidden;
        shared.appendChild(controls);
        privacy.set(section, controls);
      }
    });
    shared.appendChild(page.querySelector('.dp-threshold-control'));
    // The notebook download exports the view, so it closes the control row
    // instead of splitting the comparison choices from the privacy budget.
    var notebook = page.querySelector('.dp-notebook-control');
    if (notebook) shared.appendChild(notebook);
    // A page's own trailing actions (the Leaderboard's Regime) close the bar,
    // where Explore puts its instance filter.
    var actions = page.querySelector('.dp-control-actions');
    if (bar && actions) bar.appendChild(actions);
    return {
      describeBaseline: describeBaseline,
      describeTrials: describeTrials,
      select: select,
      top: top,
      selectTop: function (section, ordered) {
        if (top.value !== 'custom') {
          section.dpSelectTop(ordered, top.value === 'all' ? Infinity : Number(top.value));
        }
      },
      shared: shared,
      show: function (section) {
        select.value = section.dataset.metric;
        privacy.forEach(function (controls, owner) { controls.hidden = owner !== section; });
        var baseline = page.querySelector('[data-official-baseline-label]');
        if (baseline) baseline.textContent = section.dataset.baselineLabel;
        if (metricDefinition) metricDefinition.href = section.dataset.definitionHref || '';
        describeMetric(section);
        // Explore's own baseline-change handler re-targets this link when the
        // reader picks an algorithm baseline; a metric switch always resets
        // it to that metric's own default reading first.
        if (baselineDefinition) {
          baselineDefinition.href =
            section.dataset.baselineDefinitionHref || section.dataset.definitionHref || '';
          // The Leaderboard's frozen baseline needs no second link to the
          // same Metric definition; Explore's selectable baseline keeps its own.
          if (page.querySelector('[data-official-baseline]') && metricDefinition) {
            baselineDefinition.hidden = baselineDefinition.href === metricDefinition.href;
          }
        }
        describeBaseline(section.dataset.metric, index.baselines[section.dataset.metric].default);
        // One "Show top" control, shared across every metric's success rate
        // view: it moves into whichever metric's plotted-strip head is now
        // shown (Explore), or beside that metric's own Rankings heading
        // (Home/leaderboard, whose strip carries no head), rather than each
        // metric owning a separate selection.
        if (topLabel && !section.contains(topLabel)) {
          var head = section.querySelector('.dp-plotted-head');
          if (head) head.appendChild(topLabel);
          else {
            var rankingsHeading = section.querySelector('.dp-rankings-heading');
            if (rankingsHeading) rankingsHeading.before(topLabel);
          }
        }
      },
    };
  };

  /* The budget families the benchmark's privacy policy produces: their help
     explains the policy, so the family name itself is not shown. */
  var BENCHMARK_BUDGET_FAMILIES = ["ApproxDPBudget", "SubinverseSampleSizeDelta"];
  var PRIVACY_INTERPRETATIONS = ["standard", "qualified", "theoretical", "informal", "diagnostic", "nonprivate"];
  /* Each class's reading, as the library's PrivacyInterpretation documents it;
     the page carries it as `data-privacy-interpretations`. */
  var interpretationReadings = null;
  window.dpInterpretationText = function (name) {
    if (!interpretationReadings) {
      var host = document.querySelector("[data-privacy-interpretations]");
      interpretationReadings = host ? JSON.parse(host.dataset.privacyInterpretations) : {};
    }
    return interpretationReadings[name] || "";
  };
  window.dpInterpretationBadge = function (value) {
    var name = PRIVACY_INTERPRETATIONS.indexOf(value) >= 0 ? value : "unknown";
    var badge = document.createElement("span");
    badge.className = "dp-badge dp-badge--" + name + " badge";
    badge.dataset.privacyInterpretation = name;
    badge.textContent = name;
    var reading = window.dpInterpretationText(name);
    if (reading) badge.title = reading;
    badge.setAttribute("aria-label", reading
      ? window.dpSiteCopy("shared.interpretation_aria_reading", { class: name, reading: reading })
      : window.dpSiteCopy("shared.interpretation_line", { class: name }));
    return badge;
  };
  window.dpQualifiedAlgorithmLabel = function (label, interpretation) {
    return label + " [" + (interpretation || "unknown") + "]";
  };

  /* One plotted-set row, so a profile chart and a pairwise chart cannot drift
     into naming the same curve two different ways. The row body is not a
     control: activating it highlights the series, and removal is its own
     target beside it. */
  window.dpPlottedRow = function (entry) {
    var row = document.createElement("li");
    row.className = "dp-plotted-row list-group-item d-flex align-items-center gap-2";
    row.dataset.uuid = entry.uuid;
    row.tabIndex = 0;
    if (entry.treatment) row.appendChild(window.dpSeriesSwatch(entry.treatment));
    /* Where a details dialog exists the name is its button; Enter opens it and
       only Delete or Backspace removes, so the row body is never destructive. */
    var name = document.createElement(entry.onDetails ? "button" : "span");
    name.className = "dp-plotted-label" + (entry.onDetails ? " dp-plotted-open" : "");
    name.textContent = entry.label;
    // Standard is the default class, so only another class is tagged.
    if (entry.interpretation && entry.interpretation !== "standard") {
      name.append(" ", window.dpInterpretationBadge(entry.interpretation));
    }
    if (entry.onDetails) {
      name.type = "button";
      name.setAttribute("aria-haspopup", "dialog");
      name.addEventListener("click", function (event) {
        event.stopPropagation();
        entry.onDetails(name);
      });
      /* The whole row is the target: a click anywhere but the remove control
         opens the details, and the legend itself stays as it was. */
      row.classList.add("dp-plotted-row-opens");
      row.addEventListener("click", function (event) {
        if (event.target.closest(".dp-plotted-remove")) return;
        entry.onDetails(name);
      });
    }
    row.appendChild(name);
    if (entry.onRemove) {
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "dp-plotted-remove btn btn-sm btn-icon btn-ghost-secondary";
      remove.setAttribute("aria-label", window.dpSiteCopy("shared.remove_aria", { algorithm: entry.label }));
      remove.textContent = "\u00d7";
      remove.addEventListener("click", function (event) {
        event.stopPropagation();
        entry.onRemove();
      });
      row.appendChild(remove);
      row.addEventListener("keydown", function (event) {
        if (event.target !== row) return;
        if (event.key === "Enter" && entry.onDetails) { event.preventDefault(); entry.onDetails(name); return; }
        if (["Delete", "Backspace"].indexOf(event.key) === -1) return;
        event.preventDefault();
        var next = row.nextElementSibling;
        entry.onRemove();
        if (next) next.focus();
      });
    }
    return row;
  };

  /* Fixed reader-facing order for window.dpPanel.outcomes' counts keys, each
     with the site-copy slot naming it. */
  var OUTCOME_LABELS = [
    ["success", "explore.outcome_released"],
    ["no_release", "explore.outcome_no_release"],
    ["invalid_regime", "explore.outcome_invalid_regime"],
    ["algorithm_error", "explore.outcome_algorithm_error"],
    ["timeout", "explore.outcome_timeout"],
    ["memory_limit_exceeded", "explore.outcome_memory_limit"],
  ];

  /* Explore's one extra dialog line: the non-zero outcomes over recorded
     trials, then, when a declared trial went unrecorded, how much of the
     declared matrix the shares are read over. */
  /* A non-zero count never reads 0.0%, the same floor the T21 Outcomes
     table applies. */
  function formatShare(count, total) {
    var pct = (count / total) * 100;
    return count > 0 && pct < 0.1 ? "<0.1%" : pct.toFixed(1) + "%";
  }

  function outcomesLine(outcomes) {
    var line = document.createElement("p");
    line.className = "dp-algorithm-dialog-facts dp-algorithm-dialog-outcomes";
    var lead = document.createElement("span");
    lead.className = "dp-summary-lead";
    lead.append(window.dpSiteCopySpan("explore.outcomes_lead"), " ");
    line.appendChild(lead);
    var shown = 0;
    OUTCOME_LABELS.forEach(function (pair) {
      var count = outcomes.counts[pair[0]];
      if (!count) return;
      if (shown) line.appendChild(document.createTextNode(" · "));
      shown += 1;
      var item = document.createElement("span");
      item.className = "dp-summary-item";
      item.title = window.dpSiteCopy("explore.outcome_count_title", { count: count, recorded: outcomes.recorded });
      var value = document.createElement("strong");
      value.textContent = formatShare(count, outcomes.recorded);
      item.append(window.dpSiteCopySpan(pair[1]), " ", value);
      line.appendChild(item);
    });
    if (outcomes.recorded < outcomes.declared) {
      if (shown) line.appendChild(document.createTextNode(" · "));
      line.appendChild(window.dpSiteCopySpan("explore.trials_recorded", {
        recorded: outcomes.recorded, declared: outcomes.declared,
      }));
    }
    return line;
  }

  /* One details dialog per page, in the top layer: the title and swatch, the
     standing under the current view as term/value pairs, then the full
     provenance the row popover carries, with room to read it. Every chart's
     legend opens the same dialog. */
  var algorithmDialog = null;
  window.dpAlgorithmDialog = function (options) {
    var dialog = algorithmDialog;
    if (!dialog) {
      dialog = algorithmDialog = document.createElement("dialog");
      dialog.className = "dp-algorithm-dialog rounded-3 border bg-surface p-4 shadow-lg";
      dialog.addEventListener("click", function (event) {
        if (event.target === dialog) dialog.close();
      });
      document.body.appendChild(dialog);
    }
    dialog.replaceChildren();
    var closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "dp-provenance-close btn btn-sm btn-icon btn-ghost-secondary";
    closeButton.setAttribute("aria-label", window.dpSiteCopy("shared.close_details"));
    closeButton.textContent = "\u00d7";
    closeButton.addEventListener("click", function () { dialog.close(); });
    var header = document.createElement("header");
    header.className = "dp-algorithm-dialog-header border-bottom pb-3 mb-3";
    if (options.treatment) header.appendChild(window.dpSeriesSwatch(options.treatment));
    var title = document.createElement("h2");
    title.className = "dp-algorithm-dialog-title";
    title.textContent = options.title;
    header.appendChild(title);
    dialog.setAttribute("aria-label", window.dpSiteCopy("component.details_aria", { algorithm: options.title }));
    dialog.append(closeButton, header);
    /* The standing as one line: the budget it is read at, then each reading
       with its number emphasised, separated by middle dots. */
    var facts = document.createElement("p");
    facts.className = "dp-algorithm-dialog-facts";
    if (options.budget && (options.summary || []).length) {
      var lead = document.createElement("span");
      lead.className = "dp-summary-lead";
      lead.append(window.dpSiteCopySpan("shared.dialog_budget_lead", { budget: options.budget }), " ");
      facts.appendChild(lead);
    }
    /* Each reading is a slot whose emphasised field is its number:
       [slot, field, value, other fields]. */
    (options.summary || []).forEach(function (part, index) {
      if (index) facts.appendChild(document.createTextNode(" \u00b7 "));
      var item = document.createElement("span");
      item.className = "dp-summary-item";
      var value = document.createElement("strong");
      value.textContent = part[2];
      var fields = Object.assign({}, part[3]);
      fields[part[1]] = value;
      var tokens = window.dpSiteCopyTokens(part[0], fields);
      item.append.apply(item, tokens);
      item.dataset.copySlot = part[0];
      item.classList.add("dp-authored-copy");
      item.classList.toggle("dp-copy-slot", tokens.length === 1 && tokens[0] === "\u27e8" + part[0] + "\u27e9");
      facts.appendChild(item);
    });
    if (facts.childElementCount) dialog.appendChild(facts);
    // The class line reads as it does on a component's page; its link
    // explains what each class means.
    if (options.interpretation) {
      var line = document.createElement("p");
      line.className = "dp-interpretation";
      line.append(window.dpSiteCopySpan("shared.privacy_interpretation"), " ", window.dpInterpretationBadge(options.interpretation));
      var href = options.host && options.host.dataset.interpretationHref;
      if (href) {
        var meaning = document.createElement("a");
        meaning.href = href;
        meaning.append(window.dpSiteCopySpan("component.semantics_link"));
        line.append(" \u00b7 ", meaning);
      }
      dialog.appendChild(line);
    }
    if (options.outcomes) dialog.appendChild(outcomesLine(options.outcomes));
    var body = document.createElement("div");
    body.className = "dp-algorithm-dialog-body";
    // The status lines are slots of their own, so the loaded body carries no
    // slot's marking.
    var waiting = document.createElement("p");
    window.dpSiteCopyFill(waiting, "component.provenance_loading");
    body.appendChild(waiting);
    dialog.appendChild(body);
    var host = options.host;
    if (window.dpProvenanceDetails && host && host.dataset.algorithmProvenance) {
      window.dpProvenanceDetails(host, options.uuid).then(function (fragment) {
        body.replaceChildren(fragment);
      }).catch(function (error) {
        var failed = document.createElement("p");
        window.dpSiteCopyFill(failed, "component.provenance_load_error", {error: error.message});
        body.replaceChildren(failed);
      });
    } else {
      window.dpSiteCopyFill(body, "component.no_provenance");
    }
    var opener = options.opener;
    dialog.addEventListener("close", function restore() {
      dialog.removeEventListener("close", restore);
      if (opener && opener.isConnected) opener.focus({ preventScroll: true });
    });
    dialog.showModal();
    closeButton.focus({ preventScroll: true });
  };

  function connectLegend(section, rows) {
    var chart = section.querySelector(".dp-chart");
    if (!rows || !chart || section.dataset.legendConnected === "true") return;
    section.dataset.legendConnected = "true";
    var algorithms = rows.rows.slice();
    /* Which cores and transformations a reader chooses belong to the page rather
       than to one metric's panel: the narrowing keeps when the metric or the
       budget moves. */
    var host = section.closest("[data-leaderboard], [data-playground]") || document;
    var firstSelection = !host.dpSelectedAlgorithms;
    var selected = host.dpSelectedAlgorithms || (host.dpSelectedAlgorithms = {});
    var transformationNames = [];
    algorithms.forEach(function (row) {
      (row.transformations || []).forEach(function (name) {
        if (transformationNames.indexOf(name) === -1) transformationNames.push(name);
      });
    });
    transformationNames.sort();
    var state = { hovered: null };

    var shell = document.createElement("div");
    shell.className = "dp-chart-shell";
    chart.before(shell);
    shell.appendChild(chart);
    var chrome = document.createElement("div");
    chrome.className = "dp-filters dp-algorithm-filters";
    // The page's controller adds the class chips here, wherever this box lives.
    section.dpAlgorithmFilters = chrome;
    var heading = document.createElement("strong");
    heading.className = "dp-filter-title";
    heading.append(window.dpSiteCopySpan("shared.filter_heading"));
    var filters = document.createElement("div");
    filters.className = "dp-roster-filters";
    chrome.append(heading, filters);
    var removable = !!section.closest("[data-playground]");
    if (!removable) {
      var profileHeading = section.querySelector(".dp-profile-heading");
      if (profileHeading) {
        var plotColumn = document.createElement("div");
        plotColumn.className = "dp-profile-column";
        shell.prepend(plotColumn);
        var formula = profileHeading.nextElementSibling;
        plotColumn.append(profileHeading, chart);
        if (formula && formula.classList.contains("dp-profile-formula")) profileHeading.after(formula);
      }
    }
    /* Both results pages open the same filter dialog from the plotted strip. */
    var filterDialog = document.createElement("dialog");
    filterDialog.className = "dp-algorithm-dialog dp-filter-dialog rounded-3 border bg-surface p-4 shadow-lg";
    filterDialog.setAttribute("aria-label", window.dpSiteCopy("shared.filter_dialog_aria"));
    var filterClose = document.createElement("button");
    filterClose.type = "button";
    filterClose.className = "dp-provenance-close btn btn-sm btn-icon btn-ghost-secondary";
    filterClose.setAttribute("aria-label", window.dpSiteCopy("shared.close"));
    filterClose.textContent = "\u00d7";
    filterClose.addEventListener("click", function () { filterDialog.close(); });
    filterDialog.addEventListener("click", function (event) {
      if (event.target === filterDialog) filterDialog.close();
    });
    filterDialog.append(filterClose, chrome);
    document.body.appendChild(filterDialog);

    /* Nothing checked is a reading, not a fault: the chart draws empty and
       this says which list emptied it and carries the control that undoes
       it. */
    var emptyState = document.createElement("div");
    emptyState.className = "dp-note dp-warn-note dp-empty-pool alert alert-warning";
    emptyState.setAttribute("role", "status");
    emptyState.hidden = true;
    var emptyText = document.createElement("span");
    var unplotted = false;
    var emptyRecovery = document.createElement("button");
    emptyRecovery.type = "button"; emptyRecovery.className = "btn btn-sm";
    emptyState.append(emptyText, emptyRecovery);
    shell.before(emptyState);

    /* The ranked legend enhances the static standings without changing the
       eligible pool or the rank when a curve is hidden. */
    /* Leaderboard and Explore share the plotted strip; only Explore removes rows. */
    var strip = section.querySelector("[data-plotted-strip]");
    /* The plotted set and the standings remain visible on both surfaces. */
    var stripHeading = strip.querySelector(".dp-plotted-heading");
    /* The list is named for what it is on this page: the plotted algorithms
       Explore removes from, or a Leaderboard's standings. */
    var stripHeadingSlot = removable ? "shared.plotted_heading" : "shared.standings_heading";
    stripHeading.replaceChildren(window.dpSiteCopySpan(stripHeadingSlot));
    var stripHead = stripHeading.parentElement;
    if (!stripHead.classList.contains("dp-plotted-head")) {
      stripHead = document.createElement("div");
      stripHead.className = "dp-plotted-head";
      stripHeading.before(stripHead);
      stripHead.appendChild(stripHeading);
    }
    var filterButton = document.createElement("button");
    filterButton.type = "button";
    // The filter sits beside the list it narrows, styled as every filter
    // button on these pages is: the funnel and its name.
    filterButton.className = "dp-plotted-filter btn btn-sm dp-control-action";
    filterButton.innerHTML = window.dpIcons.filter;
    filterButton.appendChild(window.dpSiteCopySpan("shared.filter_algorithms"));
    filterButton.setAttribute("aria-haspopup", "dialog");
    filterButton.addEventListener("click", function () {
      filterDialog.showModal();
    });
    stripHeading.after(filterButton);
    var host = section.closest("[data-leaderboard], [data-playground]") || document.body;
    section.dpOpenDetails = openDetails;
    function openDetails(uuid, opener) {
      // A row not currently plotted carries no chart trace; its title and
      // summary still come from the static standings table, which lists
      // every eligible row whether or not it is drawn.
      var trace = (chart.data || []).filter(function (t) { return t.legendgroup === uuid; })[0];
      /* The standing under the current view as one sentence, read from the
         row the table carries for this configuration:
         "At eps = 1.0: rank 1 of 12 · success rate 95.3% at tau = 10 ·
         released an estimate on 97.8% of trials". */
      var summary = [];
      var title = trace ? (trace.meta.hoverName || trace.name) : "";
      var row = section.querySelector('.dp-budget:not([hidden]) .dp-leaderboard tr[data-uuid="' + uuid + '"]');
      if (row) {
        var table = row.closest("table");
        var columns = (table.dataset.columns || "").split(",");
        var cellText = function (name) {
          var index = columns.indexOf(name);
          if (index < 0 || !row.cells[index]) return "";
          var clone = row.cells[index].cloneNode(true);
          [].forEach.call(clone.querySelectorAll(".dp-provenance-trigger, .dp-series-marker"), function (node) { node.remove(); });
          return clone.textContent.trim();
        };
        if (!title) title = cellText("algorithm");
        var rank = cellText("rank");
        if (rank && rank !== "\u2014") {
          summary.push(["shared.dialog_rank", "rank", rank, { count: table.tBodies[0].rows.length }]);
        }
        var headline = cellText("headline");
        if (headline) {
          summary.push(["shared.dialog_headline", "value", headline, {
            percent: window.dpPercent(currentQuantile(chart)),
          }]);
        }
      }
      if (!title) return;
      var panel = section.querySelector(".dp-budget:not([hidden]) .dp-panel-heading");
      var budget = panel ? panel.textContent.trim() : "";
      var algorithm = algorithms.find(function (entry) { return entry.uuid === uuid; });
      window.dpAlgorithmDialog({
        host: host,
        uuid: uuid,
        title: title,
        treatment: trace ? (chart.dpTreatments || {})[uuid] : null,
        budget: budget,
        summary: summary,
        interpretation: algorithm ? algorithm.interpretation : "",
        outcomes: section.dpOutcomes ? section.dpOutcomes(uuid) : null,
        opener: opener,
      });
    }
    // The page rendered the strip; this fills it and moves it beside the chart.
    var stripBox = strip.querySelector(".dp-plotted-box");
    var stripSummary = stripBox.querySelector("summary");
    var stripList = strip.querySelector(".dp-plotted-rows");
    var stripMore = strip.querySelector(".dp-plotted-more");
    {
      var columnHeadings = document.createElement("div");
      columnHeadings.className = "dp-standings-columns";
      /* The score heading names the quantile it is read at, which each
         draw writes. */
      ["leaderboard.column_rank", "shared.column_algorithm", "leaderboard.column_headline"].forEach(function (slot) {
        var column = document.createElement("span");
        if (slot === "leaderboard.column_headline") column.className = "dp-score-heading";
        else column.append(window.dpSiteCopySpan(slot));
        columnHeadings.appendChild(column);
      });
      stripBox.before(columnHeadings);
    }
    strip.hidden = false;
    shell.appendChild(strip);
    var narrow = window.matchMedia("(max-width: 54.999em)");

    /* One row per drawn curve. Activating a row highlights that series rather
       than removing it, so the only destructive control is the remove button,
       which is its own target. */
    /* The rows box gets what remains of the chart's height under the
       strip's own chrome (and the instance bar), and scrolls past that. */
    /* Sizing reads layout, so it runs once per frame after the frame's
       writes, never inside a repaint. */
    var fitting = null;
    function fitRows() {
      if (fitting === null) fitting = requestAnimationFrame(function () { fitting = null; fitNow(); });
    }
    function fitNow() {
      var chartHeight = chart.getBoundingClientRect().height;
      if (narrow.matches) {
        stripList.style.maxHeight = "";
      } else if (chartHeight > 0) {
        var room = chartHeight
          - (stripList.getBoundingClientRect().top - strip.getBoundingClientRect().top);
        stripList.style.maxHeight = Math.max(room, 120) + "px";
      }
    }
    /* One plotted-list row for one configured algorithm, built once for each
       plotted state and then kept: dpPaintStrip updates its rank and score. */
    function buildStripRow(entry, isPlotted, raised) {
      var uuid = entry.uuid;
      var row = window.dpPlottedRow({
        uuid: uuid,
        label: entry.trace.meta.hoverName || entry.trace.name,
        interpretation: entry.trace.meta.interpretation,
        treatment: isPlotted ? raised(uuid) : null,
        onDetails: function (opener) { openDetails(uuid, opener); },
      });
      var rank = document.createElement("span");
      row.classList.remove("d-flex", "align-items-center", "gap-2");
      row.classList.add("dp-ranked-row");
      rank.className = "dp-legend-rank";
      rank.textContent = entry.rank;
      rank.title = window.dpSiteCopy(entry.rank === "—" ? "shared.unranked_title" : "leaderboard.column_rank");
      row.prepend(rank);
      /* The legend swatch is the show/hide control: a filled line plots
         the curve, an empty outline means it is hidden. */
      var toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "dp-plotted-remove dp-swatch-toggle";
      var plainName = (entry.trace.meta.hoverName || entry.trace.name).replace(/<br>/g, " ");
      toggle.title = window.dpSiteCopy(isPlotted ? "shared.hide_curve_title" : "shared.plot_curve_title", { algorithm: plainName });
      toggle.setAttribute("aria-label", window.dpSiteCopy(isPlotted ? "shared.hide_curve_aria" : "shared.plot_curve_aria", { algorithm: plainName }));
      toggle.setAttribute("aria-pressed", String(isPlotted));
      var swatch = row.querySelector(".dp-series-swatch");
      if (!swatch) {
        swatch = document.createElement("span");
        swatch.className = "dp-series-swatch";
      }
      toggle.appendChild(swatch);
      rank.after(toggle);
      toggle.addEventListener("click", function (event) {
        event.stopPropagation();
        selected[uuid] = !row.dpPlotted;
        markCustom();
        update();
      });
      var score = document.createElement("span");
      score.className = "dp-legend-score";
      row.appendChild(score);
      ["pointerover", "focusin"].forEach(function (type) {
        row.addEventListener(type, function () { state.hovered = uuid; paint(true); });
      });
      ["pointerout", "focusout"].forEach(function (type) {
        row.addEventListener(type, function () { state.hovered = null; paint(true); });
      });
      row.dpPlotted = isPlotted;
      if (isPlotted && raised(uuid)) row.dpSwatchKey = JSON.stringify(raised(uuid));
      return row;
    }
    section.dpPaintStrip = function (plotted, raised, active, hovering) {
      if (hovering && stripList.dpRows) return paintStripHighlight(plotted, raised, active);
      var table = section.querySelector(".dp-budget:not([hidden]) .dp-leaderboard");
      if (!table) return;
      var columns = (table.dataset.columns || "").split(",");
      var scoreIndex = columns.indexOf("headline");
      var rankIndex = columns.indexOf("rank");
      var rows = [].slice.call(table.querySelectorAll("tbody tr[data-uuid]"));
      // Each row carries its reading; the list ranks through the one rule.
      function reading(row) { return window.dpPanel.shippedReading(JSON.parse(row.dataset.reading)); }
      rows.sort(function (a, b) { return window.dpPanel.compareReadings(reading(a), reading(b)); });
      var traces = {};
      var standard = {};
      (chart.data || []).forEach(function (trace) {
        if (!traces[trace.legendgroup]) traces[trace.legendgroup] = trace;
        if (trace.meta && trace.meta.interpretation === "standard") standard[trace.legendgroup] = true;
      });
      /* Rows are in rank order, so a standard row's rank is one more than the
         standard rows strictly ahead of it; ties share a rank. */
      var ahead = 0;
      var tied = 0;
      var last = null;
      var entries = rows.map(function (row) {
        var uuid = row.dataset.uuid;
        var rank = "\u2014";
        if (standard[uuid]) {
          var now = reading(row);
          if (!last || window.dpPanel.compareReadings(now, last) !== 0) { ahead += tied; tied = 0; last = now; }
          tied += 1;
          rank = rankIndex >= 0 ? row.cells[rankIndex].textContent : 1 + ahead;
        }
        return { uuid: uuid, trace: traces[uuid], score: row.cells[scoreIndex], rank: rank };
      }).filter(function (entry) { return entry.trace; });
      /* The filter narrows the list, never the ranking: a row it leaves out
         disappears, and every row it keeps shows its rank in the full pool. */
      var ranked = entries.length;
      var kept = {};
      eligible().forEach(function (row) { kept[row.uuid] = true; });
      entries = entries.filter(function (entry) { return kept[entry.uuid]; });
      strip.classList.toggle("dp-standings-rail", !removable);
      var counts = { shown: entries.length, ranked: ranked, plotted: Object.keys(plotted).length };
      var summarySlot = "shared.strip_summary" + (entries.length < ranked ? "_filtered" : "") + (ranked === 1 ? "_one" : "");
      if (stripSummary.dataset.copySlot !== summarySlot || stripSummary.dpCounts !== JSON.stringify(counts)) {
        stripSummary.dpCounts = JSON.stringify(counts);
        window.dpSiteCopyFill(stripSummary, summarySlot, counts);
      }
      stripMore.hidden = true;
      stripBox.open = true;
      /* Rows are kept per configured algorithm and updated in place: a new
         threshold or budget rewrites ranks and scores and reorders the rows,
         and only a row whose plotted state flipped is rebuilt, because
         rebuilding the list restyles the page on every move. */
      var kept = stripList.dpRows || (stripList.dpRows = {});
      var order = entries.map(function (entry) {
        var uuid = entry.uuid;
        var row = kept[uuid];
        if (!row || row.dpPlotted !== !!plotted[uuid]) {
          if (row) row.remove();
          row = kept[uuid] = buildStripRow(entry, !!plotted[uuid], raised);
        }
        var rank = row.querySelector(".dp-legend-rank");
        if (rank.textContent !== String(entry.rank)) {
          rank.textContent = entry.rank;
          rank.title = window.dpSiteCopy(entry.rank === "\u2014" ? "shared.unranked_title" : "leaderboard.column_rank");
        }
        var score = row.querySelector(".dp-legend-score");
        var scoreTitle = window.dpSiteCopy(
          section.dpState && section.dpState.aggregation === "median" ? "shared.headline_title_median" : "shared.headline_title",
          { value: entry.score.textContent, percent: window.dpPercent(currentQuantile(chart)) }
        );
        if (score.textContent !== entry.score.textContent) score.textContent = entry.score.textContent;
        if (score.title !== scoreTitle) score.title = scoreTitle;
        return row;
      });
      var listed = {};
      order.forEach(function (row, position) {
        listed[row.dataset.uuid] = true;
        if (stripList.children[position] !== row) stripList.insertBefore(row, stripList.children[position] || null);
      });
      Object.keys(kept).forEach(function (uuid) {
        if (!listed[uuid]) { kept[uuid].remove(); delete kept[uuid]; }
      });
      paintStripHighlight(plotted, raised, active);
      var fallback = section.querySelector(".dp-results-table");
      if (fallback) fallback.hidden = !!section.closest('[data-playground]') || entries.length > 0;
      fitRows();
    };
    function paintStripHighlight(plotted, raised, active) {
      [].forEach.call(stripList.children, function (row) {
        var highlighted = row.dataset.uuid === active;
        if (row.classList.contains("dp-row-highlight") !== highlighted) row.classList.toggle("dp-row-highlight", highlighted);
        var swatch = row.querySelector(".dp-series-swatch");
        var treatment = raised(row.dataset.uuid);
        if (!swatch || !plotted[row.dataset.uuid] || !treatment) return;
        // Replace a swatch only when its drawing changes, not on every repaint.
        var key = JSON.stringify(treatment);
        if (row.dpSwatchKey === key) return;
        row.dpSwatchKey = key;
        swatch.replaceWith(window.dpSeriesSwatch(treatment));
      });
    }
    narrow.addEventListener("change", function () { paint(); });

    var below = document.createElement("div");
    below.className = "dp-below-chart";
    shell.after(below);
    /* Two searchable checklists over the same roster, read the same way: a
       checked option is allowed, so unchecking one hides what uses it.
       Each option's count is its matches given the other list's
       current selection, so an option that would empty the chart is disabled
       before it can be reached. */
    /* `name` is the list's own key; `slots` its label and search box's names. */
    function checklist(name, slots, help) {
      var panel = document.createElement("div");
      panel.className = "dp-checklist";
      var label = document.createElement("span");
      label.className = "dp-chip-label";
      label.append(window.dpSiteCopySpan(slots.label));
      var head = document.createElement("div");
      head.className = "dp-checklist-head";
      head.appendChild(label);
      var tally = document.createElement("span");
      tally.className = "dp-checklist-count badge bg-secondary-lt";
      var all = document.createElement("button");
      all.type = "button"; all.className = "btn btn-sm";
      all.append(window.dpSiteCopySpan("shared.select_all"));
      var none = document.createElement("button");
      none.type = "button"; none.className = "btn btn-sm";
      none.append(window.dpSiteCopySpan("shared.deselect_all"));
      head.append(tally, all, none);
      var find = document.createElement("input");
      find.type = "search";
      find.className = "dp-checklist-search form-control form-control-sm";
      find.setAttribute("aria-label", window.dpSiteCopy(slots.find));
      find.placeholder = window.dpSiteCopy(slots.find) + "\u2026";
      var list = document.createElement("div");
      list.className = "dp-checklist-rows list-group list-group-flush";
      if (window.dpHintMore) window.dpHintMore(list);
      panel.append(head, find, list);
      filters.appendChild(panel);
      /* Contextual help opens from the label the reader already reads. */
      if (help && window.dpSiteCopy(help) && window.dpContextualHelp) {
        var content = document.createElement("div");
        content.id = chart.id + "-" + name + "-help";
        window.dpSiteCopyFill(content, help);
        helpLabel(label, content);
      } else if (help && window.dpSiteCopy(help)) {
        label.title = window.dpSiteCopy(help);
      }
      find.addEventListener("input", function () { renderChecklists(); });
      return { panel: panel, tally: tally, all: all, none: none, find: find, list: list };
    }
    var coreList = checklist("cores", { label: "shared.cores_label", find: "shared.find_core" }, "");
    var transformationList = checklist(
      "transformations", { label: "shared.transformations_label", find: "shared.find_transformation" }, ""
    );

    var coreNames = [];
    algorithms.forEach(function (row) {
      if (row.core && coreNames.indexOf(row.core) === -1) coreNames.push(row.core);
    });
    coreNames.sort();
    var cores = host.dpSelectedCores;
    if (!cores) {
      cores = host.dpSelectedCores = new Set(coreNames);
    }
    // A transformation this page has not offered before starts checked.
    var allowed = host.dpAllowedTransformations || (host.dpAllowedTransformations = new Set());
    var offered = host.dpOfferedTransformations || (host.dpOfferedTransformations = new Set());
    transformationNames.forEach(function (name) {
      if (!offered.has(name)) { offered.add(name); allowed.add(name); }
    });

    /* A row is offered when its core is checked and every transformation in
       its chain is checked; a row with no transformations needs none. Nothing
       checked in the cores list offers nothing: the previous selection is not
       silently kept. */
    function within(row, permitted) {
      return (row.transformations || []).every(function (name) { return permitted.has(name); });
    }
    section.dpRosterMatches = function (row) {
      return cores.has(row.core) && within(row, allowed);
    };
    // What checking this option would leave, so a zero can be disabled.
    function coreCount(core) {
      return algorithms.filter(function (row) {
        return row.core === core && within(row, allowed);
      }).length;
    }
    // The rows using this transformation that are offered once it is checked.
    function transformationCount(name) {
      var permitted = new Set(allowed);
      permitted.add(name);
      return algorithms.filter(function (row) {
        return cores.has(row.core) && (row.transformations || []).indexOf(name) !== -1 && within(row, permitted);
      }).length;
    }
    /* Show exactly the first n of the given competitors, in that order. Used by the
       page's Top N control, so it repaints without announcing a roster change
       (which would ask the page to render again). */
    section.dpSelectTop = function (ordered, n) {
      var offered = {};
      eligible().forEach(function (row) { offered[row.uuid] = true; });
      var wanted = {};
      ordered.filter(function (uuid) { return offered[uuid]; }).slice(0, n)
        .forEach(function (uuid) { wanted[uuid] = true; });
      algorithms.forEach(function (row) { selected[row.uuid] = !!wanted[row.uuid]; });
      section.dpRefreshSelection();
    };
    section.dpRefreshSelection = function () {
      var visible = {};
      eligible().forEach(function (row) { visible[row.uuid] = !!selected[row.uuid]; });
      chart.dpVisibility = visible;
      renderChecklists();
    };
    function markCustom() {
      if (host.dpTopCustom) host.dpTopCustom();
    }
    function eligible() { return algorithms.filter(section.dpRosterMatches); }

    /* One reading, three surfaces. The chart, the rankings table and the
       plotted-set strip all take their treatment from `chart.dpTreatments`,
       which the draw computed from the current ranking; hover or focus raises
       one series across all three and changes nothing else. */
    /* `hovering` marks a repaint that only moves the hover highlight: it is
       applied to the drawn curves in place (no Plotly redraw), and the next
       full repaint restyles from the data again. */
    function paint(hovering) {
      if (!chart.data) return;
      var plotted = {};
      chart.data.forEach(function (trace) {
        if (trace.visible !== false && trace.visible !== "legendonly") plotted[trace.legendgroup] = trace;
      });
      var nothingDrawn = !Object.keys(plotted).length && !section.classList.contains("dp-empty-selection");
      if (nothingDrawn !== unplotted) {
        unplotted = nothingDrawn;
        showPool();
      }
      var active = state.hovered;
      if (!plotted[active]) active = null;
      var treatments = chart.dpTreatments || {};
      function raised(uuid) {
        var treatment = treatments[uuid];
        if (!treatment) return null;
        if (!active) return treatment;
        var emphasis = window.dpSeriesEmphasis();
        if (uuid === active) {
          /* The hovered series stands out in every state: its own
             colour, wider than any regular curve, fully opaque. */
          var own = window.dpSeriesInk(plotted[uuid].meta.core, plotted[uuid].meta.interpretation);
          return Object.assign({}, treatment, {
            colour: own.colour,
            width: emphasis.regularWidth + 1.5,
            opacity: 1,
          });
        }
        /* Every other curve steps back while one is active, so the linked row
           and curve read as one thing. */
        return Object.assign({}, treatment, {
          opacity: Math.min(treatment.opacity, 0.35),
        });
      }
      /* A hover changes only the shown budget's table; the others are
         brought up to date by the full repaint a budget switch makes. A
         swatch is replaced only when its drawing changes. */
      var tables = hovering ? '.dp-budget:not([hidden]) .dp-leaderboard tr[data-uuid]' : '.dp-leaderboard tr[data-uuid]';
      section.querySelectorAll(tables).forEach(function (row) {
        var uuid = row.dataset.uuid;
        var treatment = plotted[uuid] ? raised(uuid) : null;
        var key = treatment ? JSON.stringify(treatment) : "";
        if (row.dpMarkerKey !== key) {
          row.dpMarkerKey = key;
          var marker = row.querySelector('.dp-series-marker');
          if (marker) marker.remove();
          if (treatment) {
            var swatch = window.dpSeriesSwatch(treatment);
            swatch.classList.add('dp-series-marker');
            row.querySelector('.dp-alg').prepend(swatch);
          }
        }
        var tab = treatment ? 0 : -1;
        if (row.tabIndex !== tab) row.tabIndex = tab;
        var highlighted = !!treatment && uuid === active;
        if (row.classList.contains('dp-row-highlight') !== highlighted) row.classList.toggle('dp-row-highlight', highlighted);
        if (highlighted) revealRow(row);
      });
      /* Restyle only the curves actually drawn, and only when their style
         changed: a hover must not redraw the hidden roster. */
      var drawnIndices = [];
      chart.data.forEach(function (trace, index) {
        if (trace.visible !== false && trace.visible !== "legendonly") drawnIndices.push(index);
      });
      var style = { opacity: [], "line.color": [], "line.width": [] };
      drawnIndices.forEach(function (index) {
        var trace = chart.data[index];
        var treatment = raised(trace.legendgroup);
        style.opacity.push(treatment ? treatment.opacity : 1);
        style["line.color"].push(treatment ? treatment.colour : trace.line.color);
        style["line.width"].push(treatment ? treatment.width : window.dpSeriesEmphasis().mutedWidth);
      });
      /* A hover, or a threshold moving under a held tab, styles the drawn
         curves in place; Plotly takes the style once the tab is let go. */
      var inPlace = hovering || chart.dpDragging;
      /* A draw has just given the curves this very style (styled() reads the
         same treatments), so a restyle would redraw the chart for nothing;
         unless curves were styled in place since Plotly last drew them. */
      var held = !chart.dpStyledInPlace && drawnIndices.every(function (index, at) {
        var trace = chart.data[index];
        return trace.opacity === style.opacity[at] && trace.line.color === style["line.color"][at] &&
          trace.line.width === style["line.width"][at];
      });
      if (inPlace && drawnIndices.length && window.dpStyleDrawn(chart, drawnIndices, style)) {
        chart.dpStyleKey = null;
        chart.dpStyledInPlace = true;
      } else if (drawnIndices.length && held) {
        chart.dpStyleKey = JSON.stringify([drawnIndices, style]);
      } else if (drawnIndices.length) {
        var styleKey = JSON.stringify([drawnIndices, style]);
        if (chart.dpStyleKey !== styleKey) {
          chart.dpStyleKey = styleKey;
          Plotly.restyle(chart, style, drawnIndices).then(function () {
            window.dpStackDrawn(chart);
          });
        }
      }
      /* Emphasis does not change on a hover, and moving drawn groups per
         hover costs frames, so only a full repaint restacks. */
      if (!hovering) window.dpStackDrawn(chart);
      if (section.dpPaintStrip) section.dpPaintStrip(plotted, raised, active, hovering);
    }
    section.addEventListener('dp-standings', paint);
    function retreat() {
      if (chart.data) {
        chart.dpTreatments = {};
        treatments(chart.data, currentQuantile(chart)).forEach(function (treatment) {
          chart.dpTreatments[treatment.uuid] = treatment;
        });
      }
      paint();
    }
    document.addEventListener("dp-quantile", function (event) {
      if (!(event.target && event.target.contains && event.target.contains(chart))) return;
      // A hidden chart (another Explore view is shown) restyles on its next
      // draw. Its hidden ancestor is read, not its layout: the threshold has
      // just rewritten the tables, and a layout read here would lay out the page.
      if (chart.closest("[hidden]")) return;
      retreat();
    });
    section.addEventListener("dp-restyle", retreat);
    // A let-go tab's threshold is settled by now (the page's own settle
    // listeners run first): give Plotly the style drawn in place meanwhile.
    document.addEventListener("dp-threshold-settle", function (event) {
      if (event.target === chart) paint();
    });
    ['pointerover', 'pointerout', 'focusin', 'focusout'].forEach(function (type) {
      section.addEventListener(type, function (event) {
        var row = event.target.closest('.dp-leaderboard tr[data-uuid]');
        if (!row || (event.relatedTarget && row.contains(event.relatedTarget))) return;
        state.hovered = type === 'pointerover' || type === 'focusin' ? row.dataset.uuid : null;
        paint(true);
      });
    });
    /* The empty reading names the list that emptied it, because a reader who
       has just clicked one of two lists cannot otherwise tell which. */
    function showPool() {
      var pool = eligible();
      section.classList.toggle("dp-empty-selection", !pool.length);
      emptyState.hidden = !!pool.length && !unplotted;
      if (pool.length) {
        if (unplotted) {
          window.dpSiteCopyFill(emptyText, "shared.empty_no_curves");
          emptyRecovery.hidden = true;
          delete emptyRecovery.dataset.recovery;
        }
        return;
      }
      emptyRecovery.hidden = false;
      if (!cores.size) {
        window.dpSiteCopyFill(emptyText, "shared.empty_no_cores");
        emptyRecovery.replaceChildren(window.dpSiteCopySpan("shared.select_all_cores"));
        emptyRecovery.dataset.recovery = "cores";
      } else {
        window.dpSiteCopyFill(emptyText, "shared.empty_no_transformation_match");
        emptyRecovery.replaceChildren(window.dpSiteCopySpan("shared.select_all_transformations"));
        emptyRecovery.dataset.recovery = "transformations";
      }
    }
    emptyRecovery.addEventListener("click", function () {
      if (emptyRecovery.dataset.recovery === "cores") {
        coreNames.forEach(function (core) { cores.add(core); });
      } else {
        transformationNames.forEach(function (name) { allowed.add(name); });
      }
      restoreSelection();
      update();
    });
    /* A pool that no longer holds anything a reader chose opens at its own
       leading N, the way the page first opened. */
    function restoreSelection() {
      var pool = eligible();
      if (pool.some(function (row) { return selected[row.uuid]; })) return;
      pool.slice(0, rows.leading).forEach(function (row) { selected[row.uuid] = true; });
    }
    function update() {
      renderChecklists();
      var visible = {};
      eligible().forEach(function (row) { visible[row.uuid] = !!selected[row.uuid]; });
      section.dispatchEvent(new CustomEvent("dp-visibility", {
        bubbles: true, detail: { chart: chart.id, visible: visible },
      }));
      section.dispatchEvent(new CustomEvent("dp-roster-change", { bubbles: true }));
      paint();
    }

    /* One row per option: the checkbox's accessible name is the option alone,
       and the count beside it is supporting text outside the label, so a
       screen reader does not read "AdaSSP 3" as the name of a thing. */
    function renderPanel(panel, names, chosen, countOf, noMatch) {
      var needle = panel.find.value.trim().toLowerCase();
      var shown = names.filter(function (name) {
        return name.toLowerCase().indexOf(needle) !== -1;
      });
      /* Rows are built once per option and kept; a re-render only updates
         each one's check, availability and count and the list's order. */
      var made = panel.rows || (panel.rows = {});
      var order = shown.map(function (name) {
        var count = countOf(name);
        var entry = made[name];
        if (!entry) {
          var row = document.createElement("div");
          row.className = "dp-checklist-row list-group-item d-flex align-items-center gap-2";
          var label = document.createElement("label");
          var box = document.createElement("input");
          box.type = "checkbox"; box.className = "form-check-input";
          var text = document.createElement("span");
          // A long class name wraps between its words, never mid-word.
          name.split(/(?<=[a-z0-9])(?=[A-Z])/).forEach(function (part, index) {
            if (index) text.appendChild(document.createElement("wbr"));
            text.appendChild(document.createTextNode(part));
          });
          label.className = "form-check mb-0"; text.className = "form-check-label"; label.append(box, text);
          var tally = document.createElement("span");
          tally.className = "dp-checklist-tally badge bg-secondary-lt";
          box.addEventListener("change", function () {
            if (box.checked) chosen.add(name); else chosen.delete(name);
            restoreSelection();
            update();
          });
          row.append(label, tally);
          entry = made[name] = { row: row, box: box, tally: tally };
        }
        var checked = chosen.has(name);
        if (entry.box.checked !== checked) entry.box.checked = checked;
        // An option that would leave nothing to draw cannot be reached.
        var disabled = count === 0 && !checked;
        if (entry.box.disabled !== disabled) entry.box.disabled = disabled;
        if (entry.tally.textContent !== String(count)) entry.tally.textContent = String(count);
        return entry.row;
      });
      var none = panel.list.querySelector(".dp-checklist-empty");
      if (!shown.length) {
        if (!none) {
          none = document.createElement("p");
          none.className = "dp-checklist-empty";
        }
        window.dpSiteCopyFill(none, noMatch);
        order.push(none);
      }
      order.forEach(function (node, position) {
        if (panel.list.children[position] !== node) panel.list.insertBefore(node, panel.list.children[position] || null);
      });
      while (panel.list.children.length > order.length) panel.list.lastElementChild.remove();
      var tally = window.dpSiteCopy("shared.checklist_count", { selected: chosen.size, total: names.length });
      if (panel.tally.textContent !== tally) panel.tally.textContent = tally;
    }
    function renderChecklists() {
      renderPanel(coreList, coreNames, cores, coreCount, "shared.checklist_no_core_match");
      renderPanel(transformationList, transformationNames, allowed, transformationCount, "shared.checklist_no_transformation_match");
      showPool();
    }
    /* Select all takes every option that still leaves something drawn. */
    function wireChecklist(panel, names, chosen, countOf) {
      panel.all.addEventListener("click", function () {
        names.forEach(function (name) {
          if (countOf(name) > 0) chosen.add(name);
        });
        restoreSelection();
        update();
      });
      panel.none.addEventListener("click", function () {
        chosen.clear();
        restoreSelection();
        update();
      });
    }
    wireChecklist(coreList, coreNames, cores, coreCount);
    wireChecklist(transformationList, transformationNames, allowed, transformationCount);

    var hoverConnected = false;
    chart.addEventListener("dp-chart-rendered", function () {
      if (!hoverConnected) {
        hoverConnected = true;
        /* Moving from one curve to the next unhovers then hovers; the reset
           waits a moment so the chart is not repainted in between, and
           nothing repaints while a pan or zoom is being dragged. */
        var leaving = 0;
        var panning = false;
        chart.on("plotly_relayouting", function () { panning = true; });
        // A redraw resets the curves' drawn style, so dpStyleDrawn writes every curve again.
        chart.on("plotly_afterplot", function () {
          chart.dpPlotGeneration = (chart.dpPlotGeneration || 0) + 1;
          chart.dpStyledInPlace = false;
        });
        chart.on("plotly_relayout", function () {
          if (!panning) return;
          panning = false;
          if (state.hovered !== null) { state.hovered = null; paint(true); }
        });
        // A click on a curve opens that configuration's details, as its row does.
        chart.on("plotly_click", function (event) {
          var point = event.points && event.points[0];
          if (!point || !section.dpOpenDetails) return;
          var uuid = point.data.legendgroup;
          section.dpOpenDetails(uuid, section.querySelector('.dp-ranked-row[data-uuid="' + uuid + '"] .dp-plotted-open'));
        });
        chart.on("plotly_hover", function (event) {
          clearTimeout(leaving);
          if (panning) return;
          var uuid = event.points[0].data.legendgroup;
          if (state.hovered !== uuid) {
            state.hovered = uuid;
            paint(true);
          }
        });
        chart.on("plotly_unhover", function () {
          clearTimeout(leaving);
          leaving = setTimeout(function () {
            if (state.hovered === null || panning) return;
            state.hovered = null;
            paint(true);
          }, 90);
        });
      }
      state.hovered = null;
      chart.dpStyleKey = null;
      renderChecklists();
      paint();
    });
    if (firstSelection) eligible().slice(0, rows.leading).forEach(function (row) { selected[row.uuid] = true; });
    update();
  }
  window.dpConnectLegend = connectLegend;

  var DIAGNOSTIC_INK = "--dp-chart-diagnostic";
  var UNMAPPED_INK = "--dp-chart-other";

  function token(style, name) {
    return style.getPropertyValue(name).trim();
  }

  /* Tabler 1.5 writes some colours as light-dark(), which Plotly cannot read;
     a probe element resolves one to plain rgb in the chart's own theme. */
  function resolvedColour(target, value) {
    var probe = document.createElement("span");
    probe.style.color = value;
    probe.hidden = true;
    target.parentElement.appendChild(probe);
    var colour = getComputedStyle(probe).color;
    probe.remove();
    return colour;
  }

  /* The one series rule, shipped with the page: which colour a core takes in
     each scheme, which dash a privacy interpretation takes, which shapes
     separate the configurations of one core, and how heavily the reading's
     leading curves are drawn. Every surface that draws a curve, a table
     marker or a plotted-set swatch reads it from here, so they cannot
     disagree. */
  var seriesRule = null;
  function rule() {
    if (seriesRule === null) {
      var tag = document.querySelector("script[data-series-style]");
      seriesRule = tag ? JSON.parse(tag.textContent) : false;
    }
    return seriesRule;
  }
  function scheme() {
    return document.body.dataset.bsTheme === "dark" ? "slate" : "default";
  }

  /* One algorithm's ink is a function of its core and its class alone: never
     of its rank, the threshold, the budget or the selection. A nonprivate
     reference is neutral because it is a reference and not a competitor, and
     a core the table does not name takes the unmapped ink rather than
     borrowing another core's colour. Curves outside the leading set draw in
     context ink. */
  var neutralShades = {};
  function neutral(name) {
    // Read once per theme: a computed-style read restyles the page.
    var key = (document.documentElement.dataset.bsTheme || "") + name;
    if (!(key in neutralShades)) {
      neutralShades[key] = token(getComputedStyle(document.documentElement), name);
    }
    var shade = neutralShades[key];
    return { colour: shade, muted: shade };
  }
  function ink(core, interpretation) {
    var table = rule();
    if (!table) return neutral(UNMAPPED_INK);
    if (interpretation === table.referenceClass) return neutral(DIAGNOSTIC_INK);
    var colour = table.colours[scheme()][core];
    if (!colour) return neutral(UNMAPPED_INK);
    return { colour: colour, muted: neutral(UNMAPPED_INK).colour };
  }
  window.dpSeriesInk = ink;
  /* Five-choice comparisons identify selected configurations, not cores.
     Their controller owns slot lifetime; the shipped rule owns its ink. */
  window.dpComparisonStyle = function (slot, interpretation) {
    var table = rule();
    return {
      colour: interpretation === table.referenceClass
        ? neutral(DIAGNOSTIC_INK).colour : table.comparisonColours[scheme()][slot],
      marker: table.markers[slot],
    };
  };
  /* How far past what it must show each profile axis reaches. */
  window.dpProfilePad = function () { return rule().pad; };
  window.dpSeriesDash = function (interpretation) {
    var table = rule();
    return (table && table.dash[interpretation]) || "solid";
  };
  /* The one set of weights every surface draws with. A page whose rule did not
     ship draws nothing, so there is no second set of numbers to keep in step. */
  window.dpSeriesEmphasis = function () {
    var table = rule();
    return table ? table.emphasis : EMPHASIS_WITHOUT_A_RULE;
  };
  var EMPHASIS_WITHOUT_A_RULE = {
    curves: 0,
    regularWidth: 2,
    mutedWidth: 2,
    regularOpacity: 1,
    mutedOpacity: 1,
  };

  /* A swatch drawn at CSS size, so the dash and the marker that separate two
     curves stay legible beside a chart scaled down to phone width. */
  window.dpSeriesSwatch = function (treatment) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "dp-series-swatch");
    svg.setAttribute("viewBox", "0 0 26 12");
    svg.setAttribute("aria-hidden", "true");
    var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", "1");
    line.setAttribute("x2", "25");
    line.setAttribute("y1", "6");
    line.setAttribute("y2", "6");
    line.setAttribute("stroke", treatment.colour);
    line.setAttribute("stroke-width", String(treatment.width));
    line.setAttribute("stroke-opacity", String(treatment.opacity));
    if (treatment.dash && treatment.dash !== "solid") {
      line.setAttribute("stroke-dasharray", treatment.dash);
    }
    svg.appendChild(line);
    var mark = document.createElementNS("http://www.w3.org/2000/svg", "path");
    mark.setAttribute("d", markerPath(treatment.marker));
    mark.setAttribute("fill", treatment.marker === "cross" ? "none" : treatment.colour);
    mark.setAttribute("stroke", treatment.colour);
    mark.setAttribute("stroke-width", "1.4");
    mark.setAttribute("opacity", String(treatment.opacity));
    svg.appendChild(mark);
    return svg;
  };

  /* The five shapes, centred on (13, 6), in the same order the build assigns
     them from a configuration's position among its core's labels. */
  function markerPath(shape) {
    if (shape === "square") return "M10 3h6v6h-6z";
    if (shape === "triangle-up") return "M13 2.5l3.2 6h-6.4z";
    if (shape === "diamond") return "M13 2.2l3.4 3.8-3.4 3.8-3.4-3.8z";
    if (shape === "cross") return "M10 3l6 6M16 3l-6 6";
    return "M13 2.8a3.2 3.2 0 1 0 0.01 0z";
  }

  /* The quantile a chart is read at: the last one the slider sent, else the
     one its page's control holds (in percent). */
  function currentQuantile(target) {
    var p = parseFloat(target.dataset.quantile);
    if (!isNaN(p)) return p;
    var host = target.closest("[data-leaderboard], [data-playground], .dp-home") || document;
    var range = host.querySelector('.dp-threshold-control input[type="range"]');
    return range ? Number(range.value) / 100 : NaN;
  }

  /* One drawn curve's reading at quantile p, through the one ranking rule. */
  function readingOf(trace, p) {
    var curve = { x: (trace.meta && trace.meta.tau) || trace.x || [], y: trace.y || [] };
    var reached = (trace.meta && trace.meta.reached) || window.dpPanel.reachedBy(curve);
    return window.dpPanel.readingAt(curve, p, reached);
  }

  /* The rank of the drawn curves at the quantile now on screen, whatever
     their privacy interpretation; ties keep the payload's order. */
  function rankedAt(data, p) {
    return data.map(function (trace, index) {
      return { trace: trace, index: index, reading: readingOf(trace, p) };
    })
      .sort(function (a, b) {
        return window.dpPanel.compareReadings(a.reading, b.reading) || a.index - b.index;
      })
      .map(function (entry) { return entry.trace; });
  }

  /* Emphasis follows the reading now on screen: the leading traces are the
     top N by share at the current threshold among the curves drawn now. They draw in their core's colour; everything
     else plotted is drawn thin in context ink. */
  function treatments(data, p) {
    var emphasis = window.dpSeriesEmphasis();
    var leading = {};
    var drawn = data.filter(function (trace) { return trace.visible !== false && trace.visible !== "legendonly"; });
    rankedAt(drawn, p).slice(0, emphasis.curves).forEach(function (trace) {
      leading[trace.legendgroup] = true;
    });
    return data.map(function (trace) {
      var meta = trace.meta || {};
      var shades = ink(meta.core, meta.interpretation);
      var raised = !!leading[trace.legendgroup];
      return {
        uuid: trace.legendgroup,
        emphasised: raised,
        colour: raised ? shades.colour : shades.muted,
        width: raised ? emphasis.regularWidth : emphasis.mutedWidth,
        opacity: raised ? emphasis.regularOpacity : emphasis.mutedOpacity,
        dash: window.dpSeriesDash(meta.interpretation),
        marker: meta.marker || "circle",
      };
    });
  }

  function styled(data, target) {
    var table = rule();
    var drawn = treatments(data, currentQuantile(target));
    target.dpTreatments = {};
    drawn.forEach(function (treatment) {
      target.dpTreatments[treatment.uuid] = treatment;
    });
    return data.map(function (trace, at) {
      var treatment = drawn[at];
      var line = Object.assign({}, trace.line, {
        color: treatment.colour,
        width: treatment.width,
      });
      if (treatment.dash !== "solid") line.dash = treatment.dash;
      else delete line.dash;
      return Object.assign({}, trace, {
        line: line,
        opacity: treatment.opacity,
        marker: {
          symbol: treatment.marker,
          size: table.markerSize,
          color: treatment.colour,
          maxdisplayed: table.markersShown,
        },
      });
    });
  }

  /* Slot text inside a Plotly hover template: escaped as markup, and a
     literal "%{" kept from being read as one of the template's fields. */
  function hoverText(text) {
    return text.replace(/[&<>]/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char];
    }).replace(/%\{/g, "%\u200b{");
  }
  /* The τ axis. On a log1p grid the chart draws log(1 + τ) on a linear axis:
     near 0 almost linear, far out logarithmic, with τ = 0 on it. No Plotly axis
     type does this, so each trace keeps its τ (meta.tau, and customdata for the
     hover) and the ticks are labelled in τ. */
  var TAU_TICKS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];
  var SUPERSCRIPT = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079";
  // A tick from a thousand up reads as its power of ten.
  function tauTick(tau) {
    if (tau < 1000) return window.dpTau(tau);
    return "10" + String(Math.round(Math.log10(tau))).split("").map(function (digit) {
      return SUPERSCRIPT[Number(digit)];
    }).join("");
  }
  function tauAxis(scale) {
    var log = scale === "log1p";
    return {
      to: log ? Math.log1p : function (tau) { return tau; },
      layout: log
        ? {
          tickmode: "array",
          tickvals: TAU_TICKS.map(Math.log1p),
          ticktext: TAU_TICKS.map(tauTick),
        }
        : {},
    };
  }
  function onTauAxis(data, axis) {
    return data.map(function (trace) {
      var tau = trace.x || [];
      return Object.assign({}, trace, {
        x: tau.map(axis.to),
        customdata: tau,
        meta: Object.assign({}, trace.meta, { tau: tau }),
      });
    });
  }

  function labelled(data) {
    // Name and the reading at the pointer; the identifier and the table's
    // own columns stay in the table and the details box. The template's own
    // fields fill each slot's.
    var template = "<b>%{meta.hoverName}</b><br>" +
      hoverText(window.dpSiteCopy("shared.interpretation_line", { class: "\u0000class\u0000" }))
        .replace("\u0000class\u0000", "%{meta.interpretation}") + "<br>" +
      hoverText(window.dpSiteCopy("shared.curve_hover_reading", { tau: "\u0000tau\u0000", share: "\u0000share\u0000" }))
        .replace("\u0000tau\u0000", "%{customdata:.3~g}").replace("\u0000share\u0000", "%{y:.1%}") +
      "<extra></extra>";
    return data.map(function (trace) {
      var original = String(trace.name);
      return Object.assign({}, trace, {
        name: window.dpQualifiedAlgorithmLabel(original, trace.meta.interpretation),
        meta: Object.assign({}, trace.meta, { hoverName: original, uuid: trace.legendgroup }),
        hovertemplate: template,
      });
    });
  }

  /* Every panel carries its own guide, and they all mark the one threshold the
     page's standings are being read at, which a control may have moved since
     the payload was written. Redrawing must not send them back to the
     payload's opening threshold. */
  function guided(layout, target) {
    var home = !!target.closest(".dp-home");
    if (home) {
      // Home's opening paragraph says what is measured, so its axis names the
      // threshold alone rather than the Metric and baseline identifiers.
      layout = Object.assign({}, layout, {
        xaxis: Object.assign({}, layout.xaxis, {
          title: Object.assign({}, layout.xaxis.title, { text: window.dpSiteCopy("shared.threshold_label") }),
        }),
      });
    }
    if (!layout.shapes || !layout.shapes.length) {
      return layout;
    }
    var at = target.dataset.quantile;
    var handled = seekable(target);
    var accent = getComputedStyle(target).getPropertyValue("--tblr-primary").trim();
    // The guide runs across the plot at the quantile the standings read.
    var shapes = layout.shapes.map(function (shape) {
      var y = at ? Number(at) : Number(shape.y0);
      var across = { xref: "paper", x0: 0, x1: 1, y0: y, y1: y };
      return handled
        ? Object.assign({}, shape, across, { line: { color: accent, width: rule().guideWidth, dash: "solid" } })
        : Object.assign({}, shape, across, { label: guideLabel(y, home) });
    });
    return Object.assign({}, layout, { shapes: shapes });
  }

  /* A threshold as a reader sees it: three significant figures, no trailing
     zeros. The grid value itself is what every reading uses. */
  window.dpTau = function (tau) {
    return String(Number(Number(tau).toPrecision(3)));
  };

  /* The standings' score heading under *root*: the headline column at p. */
  function scoreHeadings(root, p) {
    root.querySelectorAll(".dp-standings-columns .dp-score-heading").forEach(function (heading) {
      window.dpSiteCopyFill(heading, "leaderboard.column_headline", { percent: window.dpPercent(p) });
    });
  }

  /* A quantile as a reader sees it, in percent. */
  window.dpPercent = function (p) {
    return String(Number((Number(p) * 100).toPrecision(3)));
  };

  /* The guide names the quantile it marks, at its left end; on Home, whose
     standings are fixed at that quantile, it says it is the one ranked at. */
  function guideLabel(p, ranked) {
    return {
      text: window.dpSiteCopy(ranked ? "shared.ranked_at_quantile" : "shared.quantile_guide", {
        percent: window.dpPercent(p),
      }),
      textposition: "start",
      textangle: 0,
      // Just above the line at the plot's left edge.
      xanchor: "left",
      yanchor: "bottom",
      padding: 4,
      font: { size: 11 },
    };
  }

  /* On a Leaderboard a playhead riding the top of the guide sets the
     threshold: drag it and the page reads the grid threshold nearest the
     pointer. Dragging the plot itself pans, as on every other chart. The tab is also the keyboard control (role=slider): arrow keys
     step the page's own range input, which stays the single state holder. */
  /* How far the drawn curves must show for the standings to be read on the
     chart: where each meets the guide, or, for one that never reaches p,
     where it reaches its whole share, which ranks those. */
  function reach(data, p) {
    return (data || []).reduce(function (most, trace) {
      if (trace.visible !== true) return most;
      var reading = readingOf(trace, p);
      var tau = isFinite(reading[0]) ? reading[0] : reading[2];
      // Never past the curve's own last point: nothing is drawn beyond it.
      var drawn = (trace.meta && trace.meta.tau) || trace.x || [];
      if (drawn.length) tau = Math.min(tau, drawn[drawn.length - 1]);
      return isFinite(tau) && tau > most ? tau : most;
    }, 0);
  }

  /* A Leaderboard's axis always reaches where the drawn curves meet the
     guide, so the reading the standings show is on screen even when it lies
     past the shared presentation range. It extends, never shrinks: a redraw
     keeps the reader's pan or zoom, and a page resetting the view
     (dpViewRevision) starts over at the shared range. */
  function seekRange(target, data) {
    var to = (target.dpTauAxis || tauAxis()).to;
    var range = rule().range.map(to);
    if (!seekable(target)) return range;
    var revision = target.dpViewRevision || 0;
    var reset = revision !== (target.dpViewApplied || 0);
    target.dpViewApplied = revision;
    var shown = target._fullLayout && target._fullLayout.xaxis && target._fullLayout.xaxis.range;
    if (shown && !reset) range = shown.slice();
    var far = to(reach(data, currentQuantile(target))) * rule().pad;
    if (far > range[1]) range[1] = far;
    return range;
  }
  /* A hover highlight restyles the drawn SVG of each curve directly, which
     costs no Plotly redraw; it returns false when a curve's group cannot be
     found, and the caller then restyles through Plotly. */
  window.dpStyleDrawn = function (chart, indices, style) {
    var groups = {};
    [].forEach.call(chart.querySelectorAll(".overplot .trace.scatter"), function (group) {
      var datum = group.__data__ && group.__data__[0];
      if (datum && datum.trace) groups[datum.trace.index] = group;
    });
    if (!indices.every(function (index) { return groups[index]; })) return false;
    indices.forEach(function (index, at) {
      var group = groups[index];
      var key = (chart.dpPlotGeneration || 0) + "|" + style.opacity[at] + "|" + style["line.color"][at] + "|" + style["line.width"][at];
      if (group.dpDrawnStyle === key) return;
      group.dpDrawnStyle = key;
      group.style.opacity = String(style.opacity[at]);
      var line = group.querySelector(".js-line");
      if (line) {
        line.style.stroke = style["line.color"][at];
        line.style.strokeWidth = style["line.width"][at] + "px";
      }
      [].forEach.call(group.querySelectorAll(".point"), function (point) {
        point.style.fill = style["line.color"][at];
        point.style.stroke = style["line.color"][at];
      });
    });
    return true;
  };

  /* Emphasised curves draw above the muted context, whatever the payload's
     order: the drawn groups are reordered in place, which costs neither a
     Plotly redraw nor a second layer. Within a level the payload order holds. */
  window.dpStackDrawn = function (chart) {
    var treatments = chart.dpTreatments || {};
    function level(group) {
      var datum = group.__data__ && group.__data__[0];
      var uuid = datum && datum.trace && datum.trace.legendgroup;
      return treatments[uuid] && treatments[uuid].emphasised ? 1 : 0;
    }
    [].forEach.call(chart.querySelectorAll(".scatterlayer"), function (layer) {
      var groups = [].filter.call(layer.children, function (group) { return group.classList.contains("trace"); });
      var levels = groups.map(level);
      if (levels.every(function (value, at) { return at === 0 || levels[at - 1] <= value; })) return;
      groups.forEach(function (group, at) {
        if (levels[at] === 1) layer.appendChild(group);
      });
    });
  };

  /* A new top-N at a moved threshold changes only which curves show: flip
     those traces' visibility in place rather than rebuilding the figure. */
  window.dpApplyVisibility = function (target) {
    var visible = target.dpVisibility;
    if (!visible || !target.data || target.dataset.rendered !== "true") return false;
    var indices = [];
    var values = [];
    // A curve outside the shown set is off, not "legendonly": Plotly skips an
    // off trace entirely, which makes every flip about a third cheaper.
    target.data.forEach(function (trace, index) {
      var want = !!visible[trace.legendgroup];
      if (trace.visible !== want) { indices.push(index); values.push(want); }
    });
    if (indices.length) Plotly.restyle(target, { visible: values }, indices);
    return true;
  };

  /* A quantile move, the same on every results page. The curves do not
     depend on p, so it re-ranks them by their readings at p through the one
     ranking rule (ties by label) and flips only the traces entering or
     leaving the shown set. False when the chart is not drawn yet, so the page
     draws it instead. */
  window.dpRerank = function (section, controls, chart, readings, labels) {
    var order = Object.keys(readings).sort(function (left, right) {
      return window.dpPanel.compareReadings(readings[left], readings[right]) ||
        (labels[left] < labels[right] ? -1 : labels[left] > labels[right] ? 1 : 0);
    });
    controls.selectTop(section, order);
    return !!chart && window.dpApplyVisibility(chart);
  };

  /* A share is 0-100 % by definition and a threshold is never negative. A pan
     or zoom may overshoot those edges while the pointer is held; on release
     the view snaps back inside them, keeping the reader's zoom width. */
  function bindBounds(target) {
    if (target.dpBoundsBound) return;
    target.dpBoundsBound = true;
    var snapping = false;
    // The share axis's own frame, read once: a reader's zoom rewrites layout.yaxis.range.
    var top = (target.layout.yaxis.range || [0, rule().pad])[1];
    function inside(range, low, high) {
      var width = range[1] - range[0];
      var eps = 1e-9 * (high - low);
      if (range[0] >= low - eps && range[1] <= high + eps) return null;
      if (width >= high - low) return [low, high];
      if (range[0] < low) return [low, low + width];
      if (range[1] > high) return [high - width, high];
      return null;
    }
    target.on("plotly_relayout", function () {
      if (snapping) return;
      var full = target._fullLayout;
      if (!full || !full.xaxis || !full.yaxis) return;
      var x = full.xaxis.range.slice(), y = full.yaxis.range.slice();
      var nextX = x[0] < -1e-9 ? [0, x[1] - x[0]] : null;
      var nextY = inside(y, 0, top);
      if (!nextX && !nextY) return;
      // One immediate relayout: a stepped or animated ease costs a redraw per
      // frame (laggy), and Plotly's transitions stall in a background tab.
      snapping = true;
      var exact = {};
      if (nextX) exact["xaxis.range"] = nextX;
      if (nextY) exact["yaxis.range"] = nextY;
      Plotly.relayout(target, exact).then(function () { snapping = false; }, function () { snapping = false; });
    });
  }

  function seekable(target) {
    return !!target.closest("[data-leaderboard], [data-playground]") && !target.closest(".dp-home");
  }
  /* A threshold (or, with *name*, a quantile) request, paced: a drag or slider calls request() on every
     move. A resting pacer commits when the pointer pauses and on settle(), so
     a moving control never waits on the re-rank and repaint a threshold
     costs. A `live` pacer follows the control: it commits at most once a
     frame and then leaves the page idle for as long as that commit's work
     took (measured through the next frame, where pages defer it), so a slow
     machine updates less often instead of stalling the tab. */
  window.dpTauPacer = function (source, live, name) {
    var pending = null, pause = null, frame = null, idleUntil = 0;
    function commit() {
      clearTimeout(pause);
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      if (pending === null) return;
      var tau = pending;
      pending = null;
      var started = performance.now();
      // A threshold request names its τ; a named request (the quantile tab's
      // "dp-quantile-seek") carries its own value.
      source.dispatchEvent(new CustomEvent(name || "dp-threshold-seek", {
        bubbles: true, detail: name ? { value: tau } : { tau: tau },
      }));
      if (!live) return;
      var cost = performance.now() - started;
      idleUntil = Infinity;
      requestAnimationFrame(function (frameStart) {
        cost += Math.max(0, performance.now() - frameStart);
        idleUntil = performance.now() + cost;
        if (pending !== null) schedule();
      });
    }
    function schedule() {
      if (frame !== null || idleUntil === Infinity) return;
      var wait = idleUntil - performance.now();
      if (wait > 0) {
        clearTimeout(pause);
        pause = setTimeout(schedule, wait);
        return;
      }
      frame = requestAnimationFrame(commit);
    }
    return {
      request: function (tau) {
        pending = tau;
        if (live) return schedule();
        clearTimeout(pause);
        pause = setTimeout(commit, 80);
      },
      hold: function (tau) { pending = tau; clearTimeout(pause); },
      settle: function () {
        commit();
        source.dispatchEvent(new CustomEvent("dp-threshold-settle", { bubbles: true }));
      },
    };
  };

  /* The grid threshold nearest tau: every tau control reads the page's own
     grid. A log axis has no line at 0, so its controls pass `positive`. */
  window.dpSnapTau = function (page, tau, positive) {
    var grid = page && page.dpTaus;
    if (grid && positive) grid = grid.filter(function (each) { return each > 0; });
    if (!grid || !grid.length) return tau;
    return grid.reduce(function (best, each) {
      return Math.abs(each - tau) < Math.abs(best - tau) ? each : best;
    });
  };

  /* One draggable tau tab for every chart that draws the page's threshold,
     so each chart supplies only its geometry. While the tab is held, it and a
     plain line follow the pointer on the page's tau grid at no cost, and the
     chart's own guide hides; `live` charts request the threshold as the
     pointer moves (paced), the rest only on release. Keys step the page's
     threshold control; a tab whose tau is out of view pins to that edge and
     a press brings tau back into view.

     Geometry reads the chart's place in its host through window.dpOrigin,
     cached for the length of a drag.

     geometry: orient ("x": the tab above the plot, pointing down; "y": the
     tab at the plot's left edge, pointing right), name() its accessible name,
     label(value) its text, snap(value) the nearest value its control holds,
     bounds [min, max], event the request it sends, live (commit every frame)
     or rest (commit when the pointer pauses; neither: on release),
     tauAt(event), at() (the chart's settled value),
     anchor(tau) -> {left, top, side, orient} in the host's pixels (orient,
     when given, overrides for that spot: a line that leaves the plot through
     its top edge) or null to hide, segment(tau) -> [x0, y0, x1, y1] the
     preview line, guide(visible), emphasise(on), reveal(). */
  window.dpOrigin = function (target) {
    return target.dpOrigin || target.dpOriginAtRest ||
      (target.dpOriginAtRest = { left: target.offsetLeft, top: target.offsetTop });
  };
  window.dpTauTab = function (target, geometry) {
    var page = target.closest("[data-leaderboard], [data-playground]");
    var range = page.querySelector('.dp-threshold-control input[type="range"]');
    var pacer = window.dpTauPacer(target, geometry.live, geometry.event);
    var host = target.parentElement;
    /* The chart's place in its host is read once and kept: a read after each
       redraw would lay out the whole page. Anything that moves the chart in
       its host resizes one of them, which forgets the place. */
    if (window.ResizeObserver) {
      new ResizeObserver(function () { target.dpOriginAtRest = null; }).observe(host);
      new ResizeObserver(function () { target.dpOriginAtRest = null; }).observe(target);
    }
    host.classList.add("dp-tau-host");
    host.dataset.orient = geometry.orient;
    var handle = document.createElement("div");
    handle.className = "dp-tau-handle";
    handle.tabIndex = 0;
    handle.setAttribute("role", "slider");
    handle.setAttribute("aria-label", geometry.name());
    handle.innerHTML = '<span class="dp-tau-arrow" aria-hidden="true"></span>' +
      '<span class="dp-tau-value"></span><span class="dp-tau-arrow" aria-hidden="true"></span>';
    // A tab on a right edge moves up and down, one above the plot left and right.
    function orient(which) {
      if (handle.dataset.orient === which) return;
      handle.dataset.orient = which;
      var arrows = which === "y" ? ["\u25b4", "\u25be"] : ["\u2039", "\u203a"];
      handle.querySelectorAll(".dp-tau-arrow").forEach(function (arrow, index) { arrow.textContent = arrows[index]; });
      if (which === "y") handle.setAttribute("aria-orientation", "vertical");
      else handle.removeAttribute("aria-orientation");
    }
    orient(geometry.orient);
    host.appendChild(handle);
    var line = document.createElement("div");
    line.className = "dp-tau-line";
    line.setAttribute("aria-hidden", "true");
    host.appendChild(line);
    var dragging = false, moved = false;

    function label(tau) {
      var text = geometry.label(tau);
      handle.querySelector(".dp-tau-value").textContent = text;
      handle.setAttribute("aria-valuenow", String(tau));
      handle.setAttribute("aria-valuetext", text);
    }
    function put(tau) {
      var spot = geometry.anchor(tau);
      handle.hidden = !spot;
      if (!spot) return false;
      orient(spot.orient || geometry.orient);
      handle.style.left = spot.left + "px";
      handle.style.top = spot.top + "px";
      if (spot.side) handle.dataset.side = spot.side; else delete handle.dataset.side;
      handle.title = window.dpSiteCopy(spot.side ? "shared.quantile_handle_offscreen_hint" : "shared.quantile_handle_hint");
      label(tau);
      return true;
    }
    function drawLine(tau) {
      var ends = geometry.segment(tau);
      if (!ends) { line.hidden = true; return; }
      line.hidden = false;
      var dx = ends[2] - ends[0], dy = ends[3] - ends[1];
      // A 3px bar from the first end, turned about its own centre line.
      line.style.left = ends[0] + "px";
      line.style.top = (ends[1] - 1.5) + "px";
      line.style.width = Math.hypot(dx, dy) + "px";
      line.style.transform = "rotate(" + Math.atan2(dy, dx) + "rad)";
    }
    function move(event) {
      if (!dragging) return;
      var tau = geometry.tauAt(event);
      if (tau === null) return;
      if (!moved) {
        moved = true;
        // The chart's place in its host, read once: a read per pointer move,
        // after the tab's own writes, would lay out the whole page each time.
        target.dpOrigin = { left: target.offsetLeft, top: target.offsetTop };
        target.dpDragging = true;
        host.classList.add("is-seeking");
        geometry.guide(false);
      }
      tau = geometry.snap(tau);
      put(tau);
      delete handle.dataset.side;
      drawLine(tau);
      // A live chart follows every frame; a resting one commits when the
      // pointer pauses; the rest wait for release.
      if (geometry.live || geometry.rest) pacer.request(tau); else pacer.hold(tau);
    }
    function end() {
      if (dragging && moved) {
        target.dpDragging = false;
        target.dpOrigin = null;
        host.classList.remove("is-seeking");
        // A drag that moved the threshold settles it, as releasing a slider
        // does; the guide then shows at the settled threshold.
        pacer.settle();
        geometry.guide(true);
      } else if (dragging && handle.dataset.side) {
        geometry.reveal();
      }
      // A settled drag's guide redraw already takes the resting width.
      if (geometry.emphasise && !(dragging && moved)) geometry.emphasise(false);
      dragging = false;
      handle.classList.remove("is-dragging");
    }
    handle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      event.preventDefault();
      handle.classList.add("is-dragging");
      if (geometry.emphasise) geometry.emphasise(true);
      handle.focus();
      dragging = true;
      moved = false;
      try { handle.setPointerCapture(event.pointerId); } catch (_) { /* a synthetic pointer has no capture */ }
    });
    handle.addEventListener("pointerenter", function () { if (geometry.emphasise) geometry.emphasise(true); });
    handle.addEventListener("pointerleave", function () { if (!dragging && geometry.emphasise) geometry.emphasise(false); });
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
    handle.addEventListener("keydown", function (event) {
      if (!range) return;
      var steps = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 };
      var min = Number(range.min), max = Number(range.max), now = Number(range.value), next;
      if (event.key in steps) next = now + steps[event.key];
      else if (event.key === "Home") next = min;
      else if (event.key === "End") next = max;
      else return;
      event.preventDefault();
      range.value = String(Math.min(max, Math.max(min, next)));
      range.dispatchEvent(new Event("input", { bubbles: true }));
      range.dispatchEvent(new Event("change", { bubbles: true }));
    });
    return {
      // After every draw: the tab rests on the chart's settled threshold.
      place: function () {
        // Mid-drag the pointer places the tab; a repaint's redraw must not pull it back.
        if (target.dpDragging) return;
        var tau = geometry.at();
        if (isNaN(tau) || !put(tau)) return;
        handle.setAttribute("aria-valuemin", String(geometry.bounds[0]));
        handle.setAttribute("aria-valuemax", String(geometry.bounds[1]));
      },
    };
  };

  /* The profile chart's quantile: a guide across the plot, the tab at its
     left edge. Dragging it reads the quantile p, in whole percent, as the
     page's control holds it. */
  function bindSeek(target) {
    if (target.dpSeekBound || !seekable(target)) return;
    target.dpSeekBound = true;
    function axes() {
      var full = target._fullLayout;
      return full && full.xaxis && full.yaxis ? full : null;
    }
    function shapes(update) {
      var patch = {};
      (target.layout.shapes || []).forEach(function (_, index) {
        Object.keys(update).forEach(function (key) { patch["shapes[" + index + "]." + key] = update[key]; });
      });
      if (Object.keys(patch).length) Plotly.relayout(target, patch);
    }
    /* The guide's drawn line, styled in place: hiding or thickening it
       through Plotly.relayout redraws the whole chart, which made grabbing
       the tab freeze before it moved. A redraw while the tab is held draws
       the guide again from the layout, so it is hidden again after each. */
    function guideLines() { return target.querySelectorAll(".shapelayer path"); }
    function hideGuide() { guideLines().forEach(function (path) { path.style.visibility = "hidden"; }); }
    // Plotly writes the line's width on the element; a thickened line keeps it to restore.
    function thicken(on) {
      guideLines().forEach(function (path) {
        if (!path.dataset.width) path.dataset.width = path.style.strokeWidth;
        path.style.strokeWidth = on ? "3px" : path.dataset.width;
      });
    }
    target.on("plotly_afterplot", function () { if (target.dpDragging) hideGuide(); });
    var tab = window.dpTauTab(target, {
      orient: "y",
      event: "dp-quantile-seek",
      bounds: [0.01, 0.99],
      name: function () { return window.dpSiteCopy("shared.quantile_label"); },
      label: function (p) { return "p = " + window.dpPercent(p) + "%"; },
      snap: function (p) { return Math.min(0.99, Math.max(0.01, Math.round(p * 100) / 100)); },
      // The curves re-rank and recolour as the tab moves, once a frame.
      live: true,
      at: function () {
        var shape = (target.layout.shapes || [])[0];
        return Number(target.dataset.quantile || (shape ? shape.y0 : NaN));
      },
      tauAt: function (event) {
        var full = axes();
        if (!full) return null;
        var y = event.clientY - target.getBoundingClientRect().top - full.yaxis._offset;
        return full.yaxis.p2d(Math.min(Math.max(y, 0), full.yaxis._length));
      },
      anchor: function (p) {
        var full = axes();
        if (!full) return null;
        var visible = full.yaxis.range;
        var side = p < visible[0] ? "bottom" : p > visible[1] ? "top" : "";
        var py = side === "bottom" ? full.yaxis._length : side === "top" ? 0 : full.yaxis.d2p(p);
        var origin = window.dpOrigin(target);
        return { left: origin.left + full.xaxis._offset, top: origin.top + full.yaxis._offset + py, side: side };
      },
      segment: function (p) {
        var full = axes();
        if (!full) return null;
        var py = Math.min(Math.max(full.yaxis.d2p(p), 0), full.yaxis._length);
        var origin = window.dpOrigin(target);
        var y = origin.top + full.yaxis._offset + py, left = origin.left + full.xaxis._offset;
        return [left, y, left + full.xaxis._length, y];
      },
      // The guide hides while the tab's own line stands in, then takes the settled p once.
      guide: function (visible) {
        if (!visible) return hideGuide();
        var at = Number(target.dataset.quantile);
        var shape = (target.layout.shapes || [])[0];
        if (shape && !isNaN(at) && Number(shape.y0) !== at) shapes({ y0: at, y1: at });
        guideLines().forEach(function (path) { path.style.visibility = ""; });
        thicken(false);
      },
      // The guide thickens while the tab is hovered or held, so tab and line read as one object.
      emphasise: thicken,
      // A press on a tab pinned to the edge brings the share axis back to its whole range.
      reveal: function () {
        Plotly.relayout(target, { "yaxis.range": [0, rule().pad] });
      },
    });
    target.dpPlaceHandle = tab.place;
    target.on("plotly_afterplot", tab.place);
    tab.place();
  }

  /* A chart's colours and fonts, read once per theme: reading computed
     style, and resolving a colour through a probe element, each restyle the
     chart's whole host, which every redraw would otherwise pay. */
  function surface(target) {
    var theme = document.documentElement.dataset.bsTheme || "";
    if (target.dpSurface && target.dpSurface.theme === theme) return target.dpSurface;
    var style = getComputedStyle(target);
    target.dpSurface = {
      theme: theme,
      colour: style.color,
      family: style.fontFamily,
      size: parseFloat(style.fontSize),
      bodyFamily: getComputedStyle(document.body).fontFamily,
      grid: token(style, "--dp-chart-grid"),
      plot: token(style, "--dp-chart-plot"),
      tooltip: resolvedColour(target, "var(--tblr-bg-surface)"),
      tooltipBorder: resolvedColour(target, "var(--tblr-border-color)"),
    };
    return target.dpSurface;
  }

  function surfaced(layout, target) {
    var look = surface(target);
    var grid = look.grid;
    return Object.assign({}, layout, {
      font: { color: look.colour, family: look.family, size: look.size },
      // Tooltips use the site's popover surface, not the trace colour.
      hoverlabel: {
        bgcolor: look.tooltip,
        bordercolor: look.tooltipBorder,
        font: { color: look.colour, family: look.bodyFamily, size: 12 },
        align: "left",
      },
      modebar: { bgcolor: "rgba(0,0,0,0)" },
      plot_bgcolor: look.plot,
      xaxis: Object.assign({}, layout.xaxis, {
        gridcolor: grid,
        zerolinecolor: grid,
      }),
      yaxis: Object.assign({}, layout.yaxis, {
        gridcolor: grid,
        zerolinecolor: grid,
      }),
    });
  }

  /* Family controls replace the built-in legend on interactive result pages. */
  function profileLayout(layout, target, data) {
    return Object.assign({}, layout, {
      autosize: true,
      title: Object.assign({}, layout.title, { text: '' }),
      height: target.closest('.dp-home') ? parseFloat(getComputedStyle(target).getPropertyValue('--dp-home-plot-height')) + 44 : 600,
      hovermode: "closest",
      hoverdistance: 30,
      dragmode: "pan",
      // A reader's zoom survives redraws until a page resets the view
      // (dpViewRevision), as an emptied selection does.
      uirevision: "profile-chart-" + (target.dpViewRevision || 0),
      xaxis: Object.assign({}, layout.xaxis, {
        autorange: false,
        domain: [0, 1],
        fixedrange: false,
        range: seekRange(target, data),
        type: "linear",
      }, (target.dpTauAxis || tauAxis()).layout),
      yaxis: Object.assign({}, layout.yaxis, { title: { text: '' } }),
      legend: { visible: false },
      margin: Object.assign({}, layout.margin, {
        r: 16,
        b: 44,
        t: target.closest('.dp-home') ? 0 : (seekable(target) ? 34 : layout.margin.t),
        l: 48,
      }),
    });
  }

  function controlled(data, target) {
    var section = target.closest(".dp-metric");
    if (section && section.dpRefreshSelection) section.dpRefreshSelection();
    if (!target.dpVisibility) {
      return data;
    }
    return data.map(function (trace) {
      if (trace.visible === false) return trace;
      return Object.assign({}, trace, {
        visible: !!target.dpVisibility[trace.legendgroup],
      });
    });
  }

  /* A chart element carries the budget it is currently showing in its own
     `budget` data attribute — set at build time to the default, and moved by
     the budget selector below. Every budget's payload ships beside the
     element; this is the lookup from "which budget" to "which payload". */
  function payloadTag(target) {
    var selector = 'script[data-profile-chart="' + target.id + '"]';
    if (target.dataset.budget) {
      selector += '[data-budget="' + target.dataset.budget + '"]';
    }
    return document.querySelector(selector);
  }

  /* A reader term the maintainer's file defines is a dotted-underline trigger
     for its definition; one it does not define stays the plain word. */
  function termTrigger(profile, definition, word, id) {
    var trigger = document.createElement('button');
    trigger.className = 'dp-term' + (word === 'τ' ? ' dp-symbol' : '');
    trigger.textContent = word;
    var content = document.createElement('span');
    content.id = id;
    content.append(definition.text);
    if (definition.link) {
      var more = document.createElement('a');
      more.href = definition.link;
      more.append(window.dpSiteCopySpan('task.term_more'));
      content.append(' ', more);
    }
    window.dpContextualHelp(trigger, content);
    profile.append(content);
    return { trigger: trigger, content: content };
  }

  /* One name in the formula, opening a copy of the control bar help that
     holds *marker* (the Metric's or the Baseline's), or plain text where the
     page has no such help. */
  var definedNames = 0;
  function definedName(host, marker, word) {
    var source = host && host.querySelector(marker);
    if (!source || !source.textContent.trim()) return document.createTextNode(word);
    var trigger = document.createElement('button');
    trigger.className = 'dp-term';
    trigger.textContent = word;
    // A copy of the setting box's description, stripped of the hooks the
    // page scripts find the original by.
    var content = document.createElement('div');
    content.append.apply(content, Array.from(source.cloneNode(true).childNodes));
    definedNames += 1;
    content.id = 'dp-term-' + definedNames;
    content.querySelectorAll('*').forEach(function (node) {
      Array.from(node.attributes).forEach(function (attribute) {
        if (attribute.name === 'id' || attribute.name.indexOf('data-') === 0) node.removeAttribute(attribute.name);
      });
    });
    var holder = document.createDocumentFragment();
    holder.append(trigger);
    window.dpContextualHelp(trigger, content);
    holder.append(content);
    return holder;
  }

  /* One aggregation's heading and formula, built once so each term's help is
     bound once; drawing only swaps them in and fills the metric and baseline. */
  function readingFor(profile, aggregation, chartId) {
    var cache = profile.dpReadings || (profile.dpReadings = {});
    if (cache[aggregation]) return cache[aggregation];
    if (!profile.dpTerms) {
      try { profile.dpTerms = JSON.parse(profile.dataset.readerTerms || '{}'); } catch (error) { profile.dpTerms = {}; }
    }
    var counted = window.dpPanel.counted(aggregation);
    // Every {metric} and {baseline} the slots place, filled on each draw;
    // an unwritten slot places none.
    var reading = { heading: document.createElement('span'), formula: document.createElement('span'), mirrors: [], metrics: [], tails: [] };
    var count = 0;
    function fill(into, tokens) {
      tokens.forEach(function (token) {
        if (typeof token === 'string') return into.append(token);
        if (token.slot === 'metric') return into.append(reading.metrics[reading.metrics.push(document.createElement('span')) - 1]);
        if (token.slot === 'baseline') return into.append(reading.tails[reading.tails.push(document.createElement('span')) - 1]);
        var word = token.copy ? window.dpSiteCopy(token.copy) : token.word;
        var definition = profile.dpTerms[token.term];
        if (!definition || !definition.text) return into.append(word);
        count += 1;
        var made = termTrigger(profile, definition, word, chartId + '-term-' + aggregation + '-' + count);
        into.append(made.trigger);
        if (token.term === 'tau') {
          // The threshold's own reading stays with τ wherever τ is defined.
          var mirror = document.createElement('span');
          mirror.style.display = 'block';
          mirror.style.marginTop = '0.5em';
          made.content.append(mirror);
          reading.mirrors.push(mirror);
        }
      });
    }
    [[reading.heading, counted.question], [reading.formula, counted.formula]].forEach(function (pair) {
      var tokens = window.dpSiteCopyTokens(pair[1], window.dpPanel.readingFields);
      fill(pair[0], tokens);
      pair[0].dataset.copySlot = pair[1];
      pair[0].classList.add('dp-authored-copy');
      pair[0].classList.toggle('dp-copy-slot', tokens.length === 1 && tokens[0] === '\u27e8' + pair[1] + '\u27e9');
    });
    cache[aggregation] = reading;
    return reading;
  }

  function draw(target) {
    if (!target || target.offsetParent === null) {
      return;
    }
    var tag = payloadTag(target);
    if (!tag) {
      return;
    }
    /* A page's payloads are derived from its panel datasets on demand, so a
       tag with neither a derived payload nor inline text has nothing to draw
       yet. */
    if (!tag.dpPayload) {
      if (!tag.textContent.trim()) return;
      tag.dpPayload = JSON.parse(tag.textContent);
    }
    var loading = target.closest('.dp-metric');
    if (loading && loading.dataset.loading === 'true') return;
    var payload = tag.dpPayload;
    var host = target.closest('[data-leaderboard], [data-playground]');
    /* The heading states the reading exactly: with a baseline the share is of
       trials whose error is at most τ times that baseline. */
    var profile = target.closest('.dp-results-profile');
    var heading = profile && profile.querySelector('.dp-profile-heading');
    var view = null;
    if (heading) {
      view = readingFor(profile, payload.aggregation, target.id);
      // The metric and the baseline named here open what their setting
      // boxes say of the current choice, so the formula and the bar agree.
      // A draw that changes neither the name nor its description keeps the term as it is.
      function defineIn(slot, marker, word, prefix) {
        var source = host && host.querySelector(marker);
        var key = word ? word + '\u0000' + (source ? source.textContent : '') : '';
        if (slot.dpDefined === key) return;
        slot.dpDefined = key;
        slot.replaceChildren();
        if (word) slot.append(prefix, definedName(host, marker, word));
      }
      view.metrics.forEach(function (slot) {
        defineIn(slot, '[data-metric-summary]', payload.units.metric, '');
      });
      view.tails.forEach(function (slot) {
        defineIn(slot, '[data-baseline-summary]', payload.units.baseline, ' × ');
      });
      heading.replaceChildren(view.heading);
      profile.querySelector('.dp-profile-formula').replaceChildren(view.formula);
    }
    var thresholdLabel = host && host.querySelector('.dp-threshold-control .dp-chip-label');
    if (thresholdLabel) {
      if (!host.dpThresholdHelp) {
        host.dpThresholdHelp = document.createElement('div');
        host.dpThresholdHelp.id = target.id + '-threshold-help';
        window.dpSiteCopyFill(host.dpThresholdHelp, 'shared.threshold_help');
        // The standings' "Success rate" heading is where the question arises,
        // so it opens the help wherever the page shows standings.
        var scoreHeading = host.querySelector('.dp-standings-columns .dp-score-heading');
        helpLabel(scoreHeading || thresholdLabel, host.dpThresholdHelp);
      }
      if (view) view.mirrors.forEach(function (mirror) { window.dpSiteCopyFill(mirror, 'shared.threshold_help'); });
    }
    var axis = tauAxis(payload.tauScale);
    target.dpTauAxis = axis;
    var data = styled(controlled(labelled(onTauAxis(payload.data, axis)), target), target);
    var layout = profileLayout(
      surfaced(guided(payload.layout, target), target),
      target,
      data
    );
    // The standings' score heading names the quantile the chart is read at.
    var readAt = target.dataset.quantile || ((layout.shapes || [])[0] || {}).y0;
    var section = target.closest(".dp-metric");
    if (section && readAt !== undefined && readAt !== "") scoreHeadings(section, readAt);
    // Plotly's responsive redraw fills the chart box, so the box carries the drawing's height.
    target.style.height = layout.height ? layout.height + "px" : "";
    // A drawn section keeps its chart on screen while a later load runs.
    if (section && !section.hasAttribute("data-drawn")) section.setAttribute("data-drawn", "");
    if (target.dataset.rendered === "true") {
      Plotly.react(target, data, layout).then(function () {
        target.dispatchEvent(new CustomEvent("dp-chart-rendered"));
      });
      return;
    }
    Plotly.newPlot(target, data, layout, {
      responsive: true,
      displaylogo: false,
      // Home shows one fixed reading; zoom and pan belong to the full pages.
      displayModeBar: target.closest(".dp-home") ? false : "hover",
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    }).then(function () {
      bindBounds(target);
      bindSeek(target);
      target.dispatchEvent(new CustomEvent("dp-chart-rendered"));
    });
    target.dataset.rendered = "true";
  }

  /* A figure a tutorial notebook showed, in the reader's theme. Its JSON
     follows it, already in the site's chart template. */
  function drawNotebookFigure(target) {
    var figure = JSON.parse(target.nextElementSibling.textContent);
    Plotly.react(target, figure.data, surfaced(figure.layout, target));
  }

  function drawAll() {
    document.querySelectorAll(".dp-chart").forEach(draw);
    document.querySelectorAll(".dp-notebook-figure").forEach(drawNotebookFigure);
  }

  /* A budget selector shows one budget's chart and table at a time within a
     metric section: every budget's payload and table ship, and a click only
     rewrites which budget the section's shared chart element points at and
     which panel is visible. The element itself is never replaced, so a
     reader's legend choices carry forward exactly as they already do across
     a threshold move. */
  function connectBudgets(section) {
    var group = section.querySelector('[data-budget-select]');
    var panels = [].slice.call(section.querySelectorAll(".dp-budget"));
    var target = section.querySelector(".dp-chart");
    if (!group || !target) {
      return;
    }
    asValueControl(group);
    var known = group.tagName === 'SELECT'
      ? [].slice.call(group.options).map(function (option) { return option.value; })
      : [].slice.call(group.querySelectorAll('input[type="radio"]')).map(function (input) { return input.value; });
    function selectBudget(budget, notify) {
        // Different metrics may offer different budgets; retain a valid panel
        // where the newly chosen budget does not exist.
        if (known.indexOf(budget) === -1) return;
        target.dataset.budget = budget;
        group.value = budget;
        panels.forEach(function (panel) {
          panel.hidden = panel.dataset.budget !== budget;
        });
        if (!notify) return;
        /* A page that recomputes the budget's curves itself (Explore) cancels
           the event and draws once, from fresh data; drawing the stored payload
           first would draw the chart twice. */
        var change = new CustomEvent("dp-budget-change", {
          bubbles: true,
          cancelable: true,
          detail: { budget: budget },
        });
        if (!section.dispatchEvent(change)) return;
        document.dispatchEvent(
          new CustomEvent("dp-chart", { detail: { chart: target.id } })
        );
    }
    section.dpSelectBudget = selectBudget;
    var family = section.querySelector('.dp-budget-family');
    if (family && BENCHMARK_BUDGET_FAMILIES.indexOf(family.textContent) >= 0) {
      var label = group.querySelector('.dp-chip-label') || group.previousElementSibling;
      var explanation = document.createElement('div');
      explanation.id = target.id + '-delta-policy';
      window.dpSiteCopyFill(explanation, 'shared.privacy_budget_help');
      if (label && explanation.textContent.trim()) helpLabel(label, explanation);
      family.remove();
    }
    group.addEventListener("change", function () { selectBudget(group.value, true); });
  }

  /* A rankings table shows ten rows at a time. Every eligible row stays in the
     page and stays reachable through the pager; nothing is dropped, and a
     reader without scripting still sees the complete roster. */
  var PAGE_ROWS = 10;

  function pagerOf(panel) {
    if (panel.dpPager) return panel.dpPager;
    var scroll = panel.querySelector(".dp-table-scroll");
    if (!scroll) return null;
    var pager = document.createElement("div");
    pager.className = "dp-pager";
    pager.dpStatus = document.createElement("span");
    pager.dpStatus.className = "dp-pager-status";
    pager.dpStatus.setAttribute("role", "status");
    pager.dpPrevious = document.createElement("button");
    pager.dpPrevious.type = "button"; pager.dpPrevious.className = "btn btn-sm";
    pager.dpPrevious.append(window.dpSiteCopySpan("shared.pager_previous"));
    pager.dpPrevious.setAttribute("aria-label", window.dpSiteCopy("shared.pager_previous_aria"));
    pager.dpNext = document.createElement("button");
    pager.dpNext.type = "button"; pager.dpNext.className = "btn btn-sm";
    pager.dpNext.append(window.dpSiteCopySpan("shared.pager_next"));
    pager.dpNext.setAttribute("aria-label", window.dpSiteCopy("shared.pager_next_aria"));
    pager.append(pager.dpStatus, pager.dpPrevious, pager.dpNext);
    pager.dpPrevious.addEventListener("click", function () { turn(panel, -1); });
    pager.dpNext.addEventListener("click", function () { turn(panel, 1); });
    scroll.after(pager);
    panel.dpPager = pager;
    return pager;
  }

  function turn(panel, step) {
    panel.dpPage = (panel.dpPage || 0) + step;
    applyPage(panel);
  }

  /* Rows a filter or the chart selection already hid are not part of the pool;
     the pager reports what remains eligible at the current reading. */
  function eligibleRows(panel) {
    return [].slice
      .call(panel.querySelectorAll(".dp-leaderboard tbody tr"))
      .filter(function (row) {
        return !row.hidden && !row.querySelector(".dp-empty-row");
      });
  }

  function applyPage(panel) {
    var pager = pagerOf(panel);
    if (!pager) return;
    var rows = eligibleRows(panel);
    var pages = Math.max(1, Math.ceil(rows.length / PAGE_ROWS));
    var current = Math.min(Math.max(panel.dpPage || 0, 0), pages - 1);
    panel.dpPage = current;
    var first = current * PAGE_ROWS;
    rows.forEach(function (row, position) {
      row.classList.toggle(
        "dp-paged-out",
        position < first || position >= first + PAGE_ROWS
      );
    });
    // An empty panel already says so in its own row; the pager stays out of it.
    pager.hidden = !rows.length;
    pager.dpStatus.textContent = window.dpSiteCopy("shared.pager_status", {
      first: first + 1, last: Math.min(rows.length, first + PAGE_ROWS), total: rows.length,
    });
    pager.dpPrevious.disabled = current === 0;
    pager.dpNext.disabled = current >= pages - 1;
  }

  /* A curve highlighted from the chart brings its own row's page up, so the
     highlighted row is one the reader can actually see. */
  function revealRow(row) {
    if (!row.classList.contains("dp-paged-out")) return;
    var panel = row.closest(".dp-budget");
    if (!panel) return;
    var position = eligibleRows(panel).indexOf(row);
    if (position < 0) return;
    panel.dpPage = Math.floor(position / PAGE_ROWS);
    applyPage(panel);
  }

  function connectPager(section) {
    if (section.dataset.pagerConnected === "true") return;
    section.dataset.pagerConnected = "true";
    // A new reading is a new first page: the ten shown are the leading ten.
    section.addEventListener("dp-standings", function () {
      section.querySelectorAll(".dp-budget").forEach(function (panel) {
        panel.dpPage = 0;
        applyPage(panel);
      });
    });
    section.querySelectorAll(".dp-budget").forEach(applyPage);
  }

  function connectAllBudgets() {
    document.querySelectorAll(".dp-metric").forEach(connectBudgets);
    document.querySelectorAll(".dp-metric").forEach(connectPager);
  }

  function init() {
    if (typeof Plotly === "undefined") {
      return;
    }
    drawAll();
    connectAllBudgets();
    // A page that reveals or rewrites a chart names it, so only that one is
    // drawn and the reader's other charts are left alone.
    document.addEventListener("dp-chart", function (event) {
      var target = document.getElementById(event.detail.chart);
      if (target) {
        draw(target);
      }
    });
    document.addEventListener("dp-visibility", function (event) {
      var target = document.getElementById(event.detail.chart);
      if (!target) {
        return;
      }
      target.dpVisibility = event.detail.visible;
      // A drawn chart flips the affected traces in place and recomputes their
      // treatments; only a chart not yet drawn needs the full draw.
      if (window.dpApplyVisibility(target)) {
        var section = target.closest(".dp-metric");
        if (section) section.dispatchEvent(new CustomEvent("dp-restyle"));
        return;
      }
      draw(target);
    });
    new MutationObserver(drawAll).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-bs-theme"],
    });
    // A shape takes the quantile itself, across the plot.
    document.addEventListener("dp-quantile", function (event) {
      var target = event.target.querySelector(".dp-chart");
      if (!target) {
        return;
      }
      var at = event.detail.p;
      target.dataset.quantile = at;
      // The standings' score is the error ratio at this quantile; its heading says which.
      scoreHeadings(event.target, at);
      if (target.dataset.rendered !== "true" || !target.layout) {
        return;
      }
      // A held tab draws its own line; the guide takes the value on release.
      // A hidden chart takes the threshold on its next draw.
      if (target.dpDragging || !target.offsetParent) return;
      var moved = {};
      (target.layout.shapes || []).forEach(function (_, index) {
        moved["shapes[" + index + "].y0"] = at;
        moved["shapes[" + index + "].y1"] = at;
        if (!seekable(target)) moved["shapes[" + index + "].label"] = guideLabel(at, !!target.closest(".dp-home"));
      });
      // Extend, never shrink, so the curves the guide meets stay on screen.
      var shown = target._fullLayout && target._fullLayout.xaxis && target._fullLayout.xaxis.range;
      var far = (target.dpTauAxis || tauAxis()).to(reach(target.data, Number(at))) * rule().pad;
      if (seekable(target) && shown && far > shown[1]) moved["xaxis.range"] = [shown[0], far];
      Plotly.relayout(target, moved);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
