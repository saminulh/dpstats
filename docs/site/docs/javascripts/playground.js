/* Rebuild an Explore page's views over the instances and controls a reader picks.

   The build emits one index and, per panel, one median matrix and one per-trial
   matrix. Everything a control changes is derived here from those, through
   `panel-data.js`'s mirrors of the library's own analysis: the share within a
   threshold mirrors profile_curves, and the ratio a cost is read at mirrors
   compute_error_ratio. This file selects instances and algorithms, asks for
   those derivations, and hands the result to the shared chart runtime.

   The page ships correct without this file: every metric's panels are already
   tabulated at their opening threshold, as the library traced them. */
(function () {
  "use strict";

  var NOT_FOUND = -1;
  /* Each Instance property's range label, by the site-copy slot it is read
     from, in the order the ranges are shown. */
  var AXIS_LABELS = {
    n: "site.card_samples",
    d: "site.card_dimension",
    n_over_d: "explore.range_samples_per_dimension",
    design_condition_number: "site.fact_condition_number",
    vif: "explore.feature_vif",
  };
  /* The properties that span decades, so their ranges step in them. */
  var LOGARITHMIC = { design_condition_number: true, vif: true };
  /* Steps per decade on the condition-number range: fine enough to place a
     handle inside one decade, coarse enough to drag across all of them. */
  var CONDITION_STEPS = 20;

  function readPayload(tag) {
    if (!tag.dpPayload) tag.dpPayload = JSON.parse(tag.textContent);
    return tag.dpPayload;
  }

  /* One instance property as the index carries it. JSON has no infinities,
     so the build writes an infinite one as the numeric string restored here;
     an undefined one is null. */
  function featureValue(instance, axis) {
    var value = instance[axis];
    return typeof value === "string" ? Number(value) : value;
  }

  /* The integer cut points that bracket every observed value, so the full
     range retains fractional n/d values at both ends. */
  function integerCuts(observed) {
    return Array.from(
      new Set(
        observed.flatMap(function (value) {
          return [Math.floor(value), Math.ceil(value)];
        })
      )
    ).sort(function (left, right) { return left - right; });
  }

  /* Cut points evenly spaced in log10 across the whole decades that bracket
     every observed finite value, because condition numbers span decades rather
     than counts. Infinity is the last stop wherever a design is singular, so
     the full range still keeps every instance this axis can place. */
  function logarithmicCuts(observed) {
    var finite = observed.filter(function (value) {
      return Number.isFinite(value) && value > 0;
    });
    if (!finite.length) return [];
    var low = Math.floor(Math.log10(Math.min.apply(null, finite)));
    var high = Math.ceil(Math.log10(Math.max.apply(null, finite)));
    var values = [];
    for (var step = 0; step <= (high - low) * CONDITION_STEPS; step += 1) {
      values.push(Math.pow(10, low + step / CONDITION_STEPS));
    }
    if (observed.some(function (value) { return value === Infinity; })) {
      values.push(Infinity);
    }
    return values;
  }

  /* How a whole-number axis is written, read back and ordered. */
  var WHOLE_NUMBERS = {
    inputMode: "numeric",
    rank: function (value) { return value; },
    cuts: integerCuts,
    format: function (value) { return value.toLocaleString(); },
    parse: function (text) {
      return Number.isInteger(Number(text)) ? Number(text) : null;
    },
    complaint: "explore.range_invalid_whole",
    hint: "explore.range_hint_whole",
  };

  /* The same, for a value read on a logarithmic scale: any positive number is
     typable, an infinite endpoint is written and read as the infinity sign,
     and nearest means nearest in decades rather than in units. */
  var DECADES = {
    inputMode: "decimal",
    cuts: logarithmicCuts,
    rank: function (value) {
      return Number.isFinite(value) ? Math.log10(value) : Number.MAX_VALUE;
    },
    format: window.dpNumber,
    parse: function (text) {
      if (text === "∞" || text.toLowerCase() === "inf") return Infinity;
      var value = window.dpParseNumber(text);
      return value > 0 ? value : null;
    },
    complaint: "explore.range_invalid_positive",
    hint: "explore.range_hint_positive",
  };

  var RANGE_BINS = 10;

  /* An Instance's place on a range's slider, as the tenth of the track it
     falls in: between two cut points it sits in proportion to its rank (so a
     decade axis is read on its log scale). -1 where the property is undefined. */
  function rangeBin(value, cuts, policy) {
    if (value === null || value === undefined || Number.isNaN(value)) return -1;
    var last = cuts.length - 1;
    var position = last;
    if (value <= cuts[0]) position = 0;
    else if (value < cuts[last]) {
      var high = 1;
      while (cuts[high] < value) high += 1;
      var from = policy.rank(cuts[high - 1]), to = policy.rank(cuts[high]);
      position = high - 1 + (to > from ? (policy.rank(value) - from) / (to - from) : 0);
    }
    return Math.min(RANGE_BINS - 1, Math.floor(position / last * RANGE_BINS));
  }

  /* Coalesce a hot stream of events (a dragged slider fires one per pixel)
     into at most one `deferred` call per animation frame, always acting on
     the latest state rather than replaying every intermediate one. `immediate`
     runs on every call for cheap, always-live feedback (a readout, a count).
     `.flush()` runs `deferred` now and cancels any pending frame, for the
     point an interaction commits (release, a typed value, a reset). */
  function coalesced(immediate, deferred) {
    var frame = null;
    function fire() {
      frame = null;
      deferred();
    }
    function trigger() {
      if (immediate) immediate();
      if (frame === null) frame = requestAnimationFrame(fire);
    }
    trigger.flush = function () {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      deferred();
    };
    return trigger;
  }

  /* As `coalesced`, for work that redraws the chart: after each run the page
     stays idle for as long as that run took, so a dragged handle keeps moving
     while the chart catches up as often as the machine allows. */
  function paced(immediate, deferred) {
    var frame = null, wait = null, idleUntil = 0;
    function fire() {
      frame = null;
      var started = performance.now();
      deferred();
      var now = performance.now();
      idleUntil = now + (now - started);
    }
    function trigger() {
      if (immediate) immediate();
      if (frame !== null || wait !== null) return;
      var delay = idleUntil - performance.now();
      if (delay <= 0) {
        frame = requestAnimationFrame(fire);
        return;
      }
      wait = setTimeout(function () {
        wait = null;
        frame = requestAnimationFrame(fire);
      }, delay);
    }
    trigger.flush = function () {
      if (frame !== null) cancelAnimationFrame(frame);
      if (wait !== null) clearTimeout(wait);
      frame = wait = null;
      deferred();
    };
    return trigger;
  }

  /* A two-handled range over supplied cut points, written and read back by the
     policy the axis is measured under: a noUiSlider over the cut indices, with
     both handles and the band between them draggable, and two typed endpoints.
     `changed` is called on every move (typically a coalescing trigger from
     `coalesced`, above) and flushed wherever an interaction commits. */
  function rangeFilter(values, slot, changed, policy) {
    var label = window.dpSiteCopy(slot);
    var axis = { axis: label };
    var filter = { values: values, low: 0, high: values.length - 1 };
    var last = values.length - 1;
    var box = document.createElement('div');
    box.className = 'dp-numeric-range';
    box.setAttribute('role', 'group');
    box.setAttribute('aria-label', label);
    var head = document.createElement('div');
    head.className = 'dp-numeric-head';
    var name = document.createElement('span');
    name.className = 'form-label dp-numeric-label';
    name.append(window.dpSiteCopySpan(slot));
    var reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'dp-numeric-reset btn btn-sm btn-ghost-secondary';
    reset.append(window.dpSiteCopySpan('explore.range_reset'));
    reset.setAttribute('aria-label', window.dpSiteCopy('explore.range_reset_aria', axis));
    head.append(name, reset);
    /* Ten bars over the slider's track, one per tenth of it: how many
       Instances the other ranges keep there, and how many of those this range
       keeps too. Decorative; the selected count is read out beside the filter
       button. */
    var hist = document.createElement('div');
    hist.className = 'dp-numeric-hist';
    hist.setAttribute('aria-hidden', 'true');
    var bars = [];
    for (var bin = 0; bin < RANGE_BINS; bin += 1) {
      var bar = document.createElement('span');
      var kept = document.createElement('span');
      bar.appendChild(kept);
      hist.appendChild(bar);
      bars.push({ bar: bar, kept: kept, open: -1, held: -1 });
    }
    var slider = document.createElement('div');
    slider.className = 'dp-numeric-slider';
    window.noUiSlider.create(slider, {
      start: [0, last],
      connect: true,
      step: 1,
      range: { min: 0, max: last },
      behaviour: 'tap-drag',
      // A handle follows the pointer directly; a typed value or reset jumps.
      animate: false,
      handleAttributes: [
        { 'aria-label': window.dpSiteCopy('explore.range_from_aria', axis) },
        { 'aria-label': window.dpSiteCopy('explore.range_to_aria', axis) },
      ],
      ariaFormat: { to: function (index) { return policy.format(values[Math.round(index)]); }, from: Number },
    });
    var endpoints = document.createElement('div');
    endpoints.className = 'dp-numeric-endpoints';
    var editors = [];
    function place(low, high, fromSlider) {
      filter.low = low;
      filter.high = high;
      show(fromSlider);
      changed();
    }
    // `update` follows every move, by pointer, key or `set`; a move the slider
    // made itself is never written back to it.
    slider.noUiSlider.on('update', function (positions) {
      var low = Math.round(Number(positions[0]));
      var high = Math.round(Number(positions[1]));
      if (low !== filter.low || high !== filter.high) place(low, high, true);
    });
    slider.noUiSlider.on('change', function () { if (changed.flush) changed.flush(); });
    ['explore.range_from_value_aria', 'explore.range_to_value_aria'].forEach(function (caption, side) {
      var editor = document.createElement('input');
      editor.type = 'text'; editor.className = 'form-control form-control-sm';
      editor.inputMode = policy.inputMode;
      editor.setAttribute('aria-label', window.dpSiteCopy(caption, axis));
      var original;
      editor.addEventListener('input', function () { editor.setCustomValidity(''); });
      editor.addEventListener('focus', function () {
        original = policy.format(values[side ? filter.high : filter.low]);
        editor.value = original;
        editor.select();
      });
      editor.addEventListener('blur', function () {
        var typed = editor.value.trim();
        if (typed === original) { show(); return; }
        var wanted = typed === '' ? null : policy.parse(typed);
        if (wanted === null) {
          editor.setCustomValidity(window.dpSiteCopy(policy.complaint));
          editor.reportValidity();
          return;
        }
        editor.setCustomValidity('');
        var target = policy.rank(wanted);
        var position = values.reduce(function (best, value, index) {
          return Math.abs(policy.rank(value) - target) <
            Math.abs(policy.rank(values[best]) - target) ? index : best;
        }, 0);
        if (side === 0) place(Math.min(position, filter.high), filter.high);
        else place(filter.low, Math.max(position, filter.low));
        if (changed.flush) changed.flush();
      });
      editor.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') editor.blur();
        if (event.key === 'Escape') { editor.value = original; editor.blur(); }
      });
      endpoints.appendChild(editor);
      editors.push(editor);
    });
    reset.addEventListener('click', function () {
      place(0, last);
      if (changed.flush) changed.flush();
    });
    function show(fromSlider) {
      if (!fromSlider) slider.noUiSlider.set([filter.low, filter.high], false);
      [filter.low, filter.high].forEach(function (position, side) {
        var value = values[position];
        editors[side].setCustomValidity('');
        editors[side].value = policy.format(value);
        editors[side].title = [window.dpSiteCopy('explore.range_endpoint', { value: policy.format(value) }), window.dpSiteCopy(policy.hint)]
          .filter(Boolean).join(' ');
      });
      reset.disabled = filter.low === 0 && filter.high === last;
    }
    box.append(head, hist, slider, endpoints);
    show();
    /* Bar heights are shares of the tallest bar with no range narrowed, so a
       bar shrinks as other ranges narrow; only a changed height is written. */
    function paint(open, held, tallest) {
      bars.forEach(function (entry, bin) {
        if (entry.open !== open[bin]) {
          entry.open = open[bin];
          entry.bar.style.height = (tallest ? 100 * open[bin] / tallest : 0) + '%';
        }
        if (entry.held !== held[bin]) {
          entry.held = held[bin];
          entry.kept.style.height = (open[bin] ? 100 * held[bin] / open[bin] : 0) + '%';
        }
      });
    }
    return { filter: filter, element: box, paint: paint };
  }

  /* Set a select to one of its own options; false where it has none. */
  function choose(control, value) {
    var offered = value !== null && [].some.call(control.options, function (option) {
      return option.value === value;
    });
    if (offered) control.value = value;
    return offered;
  }

  function connect(page) {
    var status = page.querySelector('[data-explore-loading]');
    var message = status.querySelector('p');
    var retry = status.querySelector('[data-explore-retry]');
    retry.onclick = function () { connect(page); };
    message.replaceChildren(window.dpSiteCopySpan('explore.loading'));
    retry.hidden = true;
    Promise.resolve().then(function () { return window.dpPanel
      .loadIndex(page.dataset.index)
    }).then(function (index) {
        start(page, index);
        status.hidden = true;
      })
      .catch(function (error) {
        console.error("Explore could not load its index: " + error.message);
        message.replaceChildren(window.dpSiteCopySpan('explore.load_error'));
        retry.hidden = false;
      });
  }

  function start(page, index) {
    var sections = [].slice.call(page.querySelectorAll(".dp-metric"));
    var empty = page.querySelector(".dp-empty");
    var slider = page.querySelector('.dp-filters input[type="range"]');
    var readout = page.querySelector(".dp-filters output");
    if (!sections.length || !empty || !slider || !readout) return;
    var roster = window.dpPanel.roster(index);
    var taus = index.taus;
    // The chart's tau tab previews on the same grid the standings are read at.
    page.dpTaus = taus.map(Number);
    var currentPanel;
    var pendingPanel;
    var chips = {};
    page.querySelectorAll(".dp-chip[data-filter]").forEach(function (chip) {
      // Budget chips show one budget's chart and table at a time; they do
      // not slice the selection, so profile-charts.js wires them on its own.
      if (chip.dataset.filter === "budget") return;
      chips[chip.dataset.filter] = (chips[chip.dataset.filter] || []).concat([chip]);
    });
    var loadMessage = document.createElement('p');
    loadMessage.setAttribute('role', 'status');
    var retry = document.createElement('button');
    retry.type = 'button';
    retry.append(window.dpSiteCopySpan('explore.retry_results'));
    retry.hidden = true;
    retry.addEventListener('click', function () { load(); });
    empty.before(loadMessage, retry);
    var chrome = window.dpResultsControls(page, sections, index);
    /* The reading lives in the address, so a reload or a shared link reopens it. */
    var query = new URLSearchParams(window.location.search);
    var defaultTop = chrome.top.value;
    if (choose(chrome.top, query.get('top'))) {
      if (chrome.top.value === 'custom' && query.has('plotted')) {
        page.dpSelectedAlgorithms = {};
        query.get('plotted').split(',').forEach(function (uuid) {
          if (roster.some(function (entry) { return entry.uuid === uuid; })) {
            page.dpSelectedAlgorithms[uuid] = true;
          }
        });
      } else if (chrome.top.value === 'custom') {
        chrome.top.value = '5';
      }
    }
    /* Each comparison chart's own selection: Error ratio's list, and
       Outcomes' algorithm and the one it is compared with. A value the
       address does not name stays null, so the chart opens on its default;
       comparison-charts.js writes the reader's choices back here. */
    var known = function (uuid) {
      return roster.some(function (entry) { return entry.uuid === uuid; });
    };
    page.dpComparisonSelection = {
      feature: query.has('feature_plotted')
        ? query.get('feature_plotted').split(',').filter(known) : null,
      outcomes: known(query.get('outcomes')) ? query.get('outcomes') : null,
      compare: known(query.get('outcomes_compare')) && query.get('outcomes_compare') !== query.get('outcomes')
        ? query.get('outcomes_compare') : null,
    };
    page.addEventListener('dp-comparison-selection', function () { writeAddress(); });
    var instanceCount = document.createElement("span");
    instanceCount.setAttribute("aria-live", "polite");
    instanceCount.className = 'dp-control-count';
    var interpretation = chips["class"] && chips["class"][0].closest(".dp-chip-group");
    var baselineFieldset = page.querySelector('[data-shared-baseline]');
    var baselineAlgorithmOption = page.querySelector('[data-baseline-algorithm-option]');
    var baselineAlgorithmClear = page.querySelector('[data-baseline-algorithm-clear]');
    var baselineAlgorithmTrigger = page.querySelector('[data-baseline-algorithm-trigger]');
    var baselineStatus = page.querySelector('[data-baseline-status]');
    var baselineDefinitionLink = page.querySelector('[data-baseline-summary-link]');
    var algorithmBaselineHref = page.dataset.algorithmBaselineHref || '';
    var baselinesHref = page.dataset.baselinesHref || '';
    var trialAggregation = page.querySelector('[data-trial-aggregation]');
    var viewTabs = page.querySelector('[data-view-tabs]');
    var settings = page.querySelector('[data-evaluation-settings]');
    /* The instance filter applies to every view. Its button closes the
       control bar beside the notebook download, keeping the selected count in
       sight; its ranges open in a panel under the bar, above the view tabs. */
    var instanceToggle = document.createElement('button');
    instanceToggle.type = 'button';
    instanceToggle.className = 'btn btn-sm dp-control-action';
    instanceToggle.setAttribute('data-bs-toggle', 'collapse');
    instanceToggle.setAttribute('data-bs-target', '#instance-filters');
    instanceToggle.setAttribute('aria-expanded', 'false');
    instanceToggle.setAttribute('aria-controls', 'instance-filters');
    instanceToggle.innerHTML = window.dpIcons.filter;
    instanceToggle.append(window.dpSiteCopySpan('explore.filter_instances'), instanceCount);
    var instanceBlock = document.createElement('div');
    instanceBlock.id = 'instance-filters';
    instanceBlock.className = 'collapse dp-instance-filter';
    var instanceFilters = document.createElement('div');
    instanceFilters.className = 'card card-body dp-instance-ranges';
    instanceBlock.appendChild(instanceFilters);
    var actions = document.createElement('div');
    actions.className = 'dp-control-actions';
    actions.appendChild(instanceToggle);
    var notebookControl = page.querySelector('.dp-notebook-control');
    if (notebookControl) actions.appendChild(notebookControl);
    settings.appendChild(actions);
    function showInstanceCount(selected) {
      instanceCount.textContent = selected === null ? '…' : window.dpSiteCopy('explore.instance_count', {
        selected: selected.toLocaleString(), total: index.instances.length.toLocaleString(),
      });
    }
    showInstanceCount(null);
    /* Dragging a range handle fires one `input` event per pixel; re-tracing
       every algorithm's profile curve and redrawing the chart at that rate is
       what makes the filters feel laggy. The
       selected-instance count is cheap and stays live on every event; the
       curves/table/chart recompute (`render`) runs at most once per
       animation frame while dragging, paced by its own cost (`paced`), and is
       flushed wherever an interaction commits (release, a typed value, a
       reset). `render` is a hoisted
       function declaration, so it is safe to close over here. */
    function updateInstanceCount() {
      showInstanceCount(selection().instances.length);
      paintHistograms();
    }
    var scheduleFilterChange = paced(updateInstanceCount, render);
    /* One range per Instance property the task lists. */
    var ranges = {};
    var binsOf = {}, painters = {}, tallest = {};
    Object.keys(AXIS_LABELS).filter(function (axis) {
      return axis in index.features;
    }).forEach(function (axis) {
      var policy = LOGARITHMIC[axis] ? DECADES : WHOLE_NUMBERS;
      var cuts = policy.cuts(index.instances.map(function (instance) {
        return featureValue(instance, axis);
      }));
      if (cuts.length < 2) return;
      var control = rangeFilter(
        cuts, AXIS_LABELS[axis], scheduleFilterChange, policy
      );
      ranges[axis] = control.filter;
      painters[axis] = control.paint;
      binsOf[axis] = Int8Array.from(index.instances, function (instance) {
        return rangeBin(featureValue(instance, axis), cuts, policy);
      });
      var all = new Array(RANGE_BINS).fill(0);
      binsOf[axis].forEach(function (bin) { if (bin >= 0) all[bin] += 1; });
      tallest[axis] = Math.max.apply(null, all);
      instanceFilters.appendChild(control.element);
    });
    viewTabs.before(instanceBlock);
    paintHistograms();
    sections.forEach(function (section) {
      var controls = controlsOf(section);
      window.dpConnectLegend(section, { leading: Number(chrome.top.value), rows: roster });
      /* The card's Outcomes line, over the same instance columns the table
         and curves read. A released curves-only panel carries no run matrix
         (no `values`), so there is nothing to derive an outcome from. */
      section.dpOutcomes = function (uuid) {
        var panel = activePanel(section);
        var held = currentPanel && currentPanel.entry === panel ? currentPanel.payload : null;
        if (!held || !held.values) return null;
        var row = roster.findIndex(function (entry) { return entry.uuid === uuid; });
        if (row < 0) return null;
        return window.dpPanel.outcomes(held, row, selection().instances);
      };
      var profilePanel = section.querySelector('.dp-profile-view');
      profilePanel.setAttribute('role', 'tabpanel');
      profilePanel.setAttribute('aria-labelledby', 'playground-view-profiles');
    });
    /* "outcomes" shares the "feature" tab's panel and controls rather than
       opening a second copy: only which chart inside that panel is drawn and
       visible changes (updateViews, below). */
    var buttons = Array.from(viewTabs.querySelectorAll('[role="tab"]'));
    var VIEWS = buttons.map(function (button) { return button.dataset.view; });
    buttons.forEach(function (button, position) {
      var view = button.dataset.view;
      button.addEventListener('click', function () {
        state.view = view;
        render();
      });
      button.addEventListener('keydown', function (event) {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        var last = VIEWS.length - 1;
        var next = event.key === 'Home' ? 0 : event.key === 'End' ? last
          : (position + (event.key === 'ArrowRight' ? 1 : last)) % VIEWS.length;
        viewTabs.children[next].click();
        viewTabs.children[next].focus();
      });
    });

    function controlsOf(section) {
      return readPayload(section.querySelector("script[data-results-section]"));
    }

    function updateViews() {
      var featureVisible = state.view === 'feature' || state.view === 'outcomes';
      sections.forEach(function (section) {
        section.querySelector('.dp-profile-view').hidden = state.view !== 'profiles';
        var feature = section.querySelector('.dp-comparison');
        feature.hidden = !featureVisible;
        feature.open = featureVisible;
        feature.setAttribute(
          'aria-labelledby',
          'playground-view-' + (state.view === 'outcomes' ? 'outcomes' : 'feature')
        );
        // Only which chart inside the shared panel is drawn changes with the
        // subview; the panel and its controls (feature/scale/aggregation/
        // baseline/instance filter/threshold) are not duplicated. Dispatched
        // only on an actual change, so an unrelated render (a threshold or
        // filter update) never forces a second, redundant redraw.
        var nextSubview = state.view === 'outcomes' ? 'outcomes' : 'chart';
        if (feature.dataset.comparisonSubview !== nextSubview) {
          feature.dataset.comparisonSubview = nextSubview;
          feature.dispatchEvent(new Event('dp-comparison-subview'));
        }
      });
      Array.from(viewTabs.children).forEach(function (button) {
        var selected = button.dataset.view === state.view;
        button.setAttribute('aria-selected', String(selected)); button.classList.toggle('active', selected);
        button.tabIndex = selected ? 0 : -1;
        // "outcomes" draws inside the "feature" tab's own panel; it has no
        // second copy of that panel to control.
        var controlled = button.dataset.view === 'outcomes' ? 'feature' : button.dataset.view;
        button.setAttribute('aria-controls', controlsOf(activeSection()).chart + '-' + controlled);
      });
    }

    var visibleClasses = index.visible_classes;
    /* Comparisons always require an explicit choice, even without standard rows. */
    if (chips["class"]) {
      chips["class"].forEach(function (chip) {
        if (visibleClasses.indexOf(chip.dataset.value) === NOT_FOUND) {
          chip.setAttribute("aria-pressed", "false");
        }
      });
    }
    // The outcomes view and the notebook read at a τ, opening at the index's;
    // the standings read at the quantile p, in whole percent.
    var opening = taus.findIndex(function (tau) { return Number(tau) >= index.opening_tau; });
    var state = {
      view: 'profiles',
      metric: chrome.select.value,
      baseline: baselineFieldset.value,
      aggregation: trialAggregation.value,
      threshold: opening < 0 ? 0 : opening,
      quantile: Math.round(Number(index.headline_quantile) * 100),
      classes: chips.class ? new Set(chips.class.filter(function (chip) {
        return chip.getAttribute('aria-pressed') === 'true';
      }).map(function (chip) { return chip.dataset.value; })) : new Set(visibleClasses),
    };
    var opened = {
      metric: state.metric,
      aggregation: state.aggregation,
      threshold: state.threshold,
      quantile: state.quantile,
    };
    var openedMetric = query.get('metric');
    if (sections.some(function (section) { return section.dataset.metric === openedMetric; })) {
      state.metric = openedMetric;
      chrome.select.value = openedMetric;
      syncBaselines();
    }
    var openedBaseline = query.get('baseline');
    var openedAlgorithm = window.dpPanel.baselineAlgorithm(openedBaseline || '');
    if (openedAlgorithm && roster.some(function (entry) { return entry.uuid === openedAlgorithm; })) {
      showAlgorithmBaseline(openedAlgorithm);
      state.baseline = baselineFieldset.value;
    } else if (openedBaseline && !openedAlgorithm &&
        [].some.call(baselineFieldset.options, function (option) {
          return option.value === openedBaseline && !option.hasAttribute('data-baseline-algorithm-trigger');
        })) {
      state.baseline = openedBaseline;
      baselineFieldset.value = openedBaseline;
    }
    if (choose(trialAggregation, query.get('aggregation'))) state.aggregation = trialAggregation.value;
    chrome.describeTrials();
    var openedTau = taus.findIndex(function (tau) { return Number(tau) === Number(query.get('tau')); });
    if (query.has('tau') && openedTau >= 0) state.threshold = openedTau;
    var openedQuantile = Number(query.get('p'));
    if (query.has('p') && openedQuantile >= 1 && openedQuantile <= 99) state.quantile = Math.round(openedQuantile);
    if (VIEWS.indexOf(query.get('view')) !== NOT_FOUND) state.view = query.get('view');
    if (query.get('budget')) {
      sections.forEach(function (section) {
        if (section.dpSelectBudget) section.dpSelectBudget(query.get('budget'), false);
      });
    }
    /* The reading is written as query parameters, replacing the entry rather
       than adding one, so Back still leaves the page. */
    function writeAddress() {
      var section = activeSection();
      var chart = section.querySelector('.dp-chart');
      var next = new URLSearchParams();
      if (state.metric !== opened.metric) next.set('metric', state.metric);
      if (page.querySelector('[data-budget-select]')) next.set('budget', chart.dataset.budget);
      if (state.baseline !== index.baselines[state.metric].default) next.set('baseline', state.baseline);
      if (state.aggregation !== opened.aggregation) next.set('aggregation', state.aggregation);
      if (state.threshold !== opened.threshold) next.set('tau', String(taus[state.threshold]));
      if (state.quantile !== opened.quantile) next.set('p', String(state.quantile));
      if (state.view !== 'profiles') next.set('view', state.view);
      if (chrome.top.value !== defaultTop) next.set('top', chrome.top.value);
      if (chrome.top.value === 'custom') {
        var picked = page.dpSelectedAlgorithms || {};
        next.set('plotted', Object.keys(picked).filter(function (uuid) { return picked[uuid]; }).join(','));
      }
      var compared = page.dpComparisonSelection;
      if (compared.feature) next.set('feature_plotted', compared.feature.join(','));
      if (compared.outcomes) next.set('outcomes', compared.outcomes);
      if (compared.compare) next.set('outcomes_compare', compared.compare);
      var text = next.toString();
      var address = window.location.pathname + (text ? '?' + text : '') + window.location.hash;
      try { window.history.replaceState(window.history.state, '', address); } catch (error) { /* sandboxed */ }
    }
    /* activeSection is declared further below (hoisted); the initial link
       target still needs setting before the first render. */
    /* The named baselines belong to the Metric they divide, so a Metric
       change rebuilds them; the algorithm option and the Algorithm… search
       option are the roster and never move. The reader's own choice is kept
       wherever the new Metric offers it too, and only an absent one falls
       back to that Metric's protocol baseline. */
    function namedBaselines() {
      return [].filter.call(baselineFieldset.options, function (option) {
        return !option.hasAttribute('data-baseline-algorithm-option') &&
          !option.hasAttribute('data-baseline-algorithm-trigger');
      });
    }
    function syncBaselines() {
      var catalogue = index.baselines[state.metric];
      if (!baselineFieldset || !catalogue) return;
      namedBaselines().forEach(function (option) { option.remove(); });
      var offered = false;
      catalogue.options.forEach(function (entry) {
        baselineAlgorithmOption.before(new Option(entry.label, entry.key));
        if (entry.key === state.baseline) offered = true;
      });
      if (!offered && !window.dpPanel.baselineAlgorithm(state.baseline)) {
        state.baseline = catalogue.default;
      }
      baselineFieldset.value = state.baseline;
      syncBaselineDefinition();
    }

    /* The Definition link follows the reader's own baseline: a named one
       reads at its row in the task's baselines table, while reading against
       none or a chosen algorithm reads the library's baseline API instead. */
    function syncBaselineDefinition() {
      if (!baselineDefinitionLink) return;
      var algorithmic = state.baseline === 'none' || window.dpPanel.baselineAlgorithm(state.baseline);
      baselineDefinitionLink.href = algorithmic
        ? algorithmBaselineHref
        : baselinesHref + '#baseline-' + state.baseline;
      chrome.describeBaseline(state.metric, state.baseline);
    }

    chrome.select.addEventListener('change', function () {
      state.metric = chrome.select.value;
      syncBaselines();
      load();
    });

    function activeSection() {
      for (var position = 0; position < sections.length; position += 1) {
        if (sections[position].dataset.metric === state.metric) return sections[position];
      }
      return sections[0];
    }

    /* The index owns every data path; a section names only its own budgets. */
    function activePanel(section) {
      var budget = section.querySelector(".dp-chart").dataset.budget;
      return index.panels.filter(function (panel) {
        return panel.metric === section.dataset.metric && panel.budget === budget;
      })[0];
    }

    /* The other budgets of the metric on screen, fetched one at a time once
       the page is idle, so a reader's first click on a budget does not wait on
       the network. A connection that asks to save data, or a slow one, is
       left alone. */
    var warmed = {};
    var warming = false;
    function warmBudgets() {
      var connection = navigator.connection || {};
      if (warming || connection.saveData || /(^|-)2g$|^3g$/.test(connection.effectiveType || "")) return;
      var idle = window.requestIdleCallback || function (fn) { return setTimeout(fn, 200); };
      // One fetch at a time, always for the metric on screen when it starts:
      // a reader who moves on stops the old metric's queue.
      function next() {
        var metric = activeSection().dataset.metric;
        var panel = index.panels.filter(function (candidate) {
          return candidate.metric === metric && candidate.runs && !warmed[candidate.runs];
        })[0];
        if (!panel) { warming = false; return; }
        warming = true;
        warmed[panel.runs] = true;
        idle(function () {
          window.dpPanel.panelOf(index, panel).then(next, next);
        });
      }
      next();
    }

    /* An open range keeps every instance, including one whose property is
       undefined; a narrowed one keeps only the values inside it. */
    function keeps(instance, axis) {
      var range = ranges[axis];
      if (range.low === 0 && range.high === range.values.length - 1) return true;
      var value = featureValue(instance, axis);
      return value !== null && value >= range.values[range.low] && value <= range.values[range.high];
    }
    function matchesFilters(instance) {
      return Object.keys(ranges).every(function (axis) { return keeps(instance, axis); });
    }

    /* Each range's bars: an Instance counts on an axis when every other range
       keeps it, and is held there when this range keeps it too. One pass over
       the Instances per call, so it runs on every move of a handle. */
    function paintHistograms() {
      var axes = Object.keys(ranges);
      var open = {}, held = {};
      axes.forEach(function (axis) {
        open[axis] = new Array(RANGE_BINS).fill(0);
        held[axis] = new Array(RANGE_BINS).fill(0);
      });
      index.instances.forEach(function (instance, column) {
        var missed = null, misses = 0;
        for (var at = 0; at < axes.length && misses < 2; at += 1) {
          if (!keeps(instance, axes[at])) { missed = axes[at]; misses += 1; }
        }
        if (misses > 1) return;
        axes.forEach(function (axis) {
          if (misses && axis !== missed) return;
          var bin = binsOf[axis][column];
          if (bin < 0) return;
          open[axis][bin] += 1;
          if (!misses) held[axis][bin] += 1;
        });
      });
      axes.forEach(function (axis) { painters[axis](open[axis], held[axis], tallest[axis]); });
    }

    function selection() {
      var columns = [];
      index.instances.forEach(function (instance, column) {
        if (matchesFilters(instance)) columns.push(column);
      });
      var rows = [];
      var against = window.dpPanel.baselineAlgorithm(state.baseline);
      roster.forEach(function (algorithm, row) {
        if (algorithm.uuid === against) return;
        if (!state.classes || state.classes.has(algorithm.interpretation)) {
          rows.push(row);
        }
      });
      return { instances: columns, rows: rows };
    }

    /* The columns the table declares, so a rebuilt row matches its header. */
    function declaredColumns(body) {
      var table = body.closest ? body.closest("table") : body.parentNode;
      var declared = table && table.getAttribute("data-columns");
      return declared ? declared.split(",") : ["algorithm", "interpretation", "family", "headline"];
    }

    /* The rows are updated in place, keyed by configured algorithm: a
       re-render rewrites the cells that changed and moves only the rows whose
       place changed, because rebuilding the table restyles and lays out the
       whole page on every threshold, budget or selection move. */
    function fillTable(body, rows, readings) {
      var columns = declaredColumns(body);
      var existing = {};
      [].slice.call(body.children).forEach(function (line) {
        // A row this script did not write (the build's fallback table) goes.
        if (line.dpCurve && line.dataset.uuid) existing[line.dataset.uuid] = line;
        else line.remove();
      });
      rows.forEach(function (row, position) {
        var algorithm = roster[row];
        var reading = readings[algorithm.uuid];
        var line = existing[algorithm.uuid];
        delete existing[algorithm.uuid];
        if (!line) {
          line = document.createElement("tr");
          line.dataset.uuid = algorithm.uuid;
        }
        line.dpCurve = true;
        line.dataset.reading = JSON.stringify(reading.map(String));
        /* The table says which columns it has; the row follows that list so a
           re-render can never add a cell the header does not name. */
        var specs = {
          algorithm: ["dp-alg", algorithm.algorithm, null],
          interpretation: ["dp-value", algorithm.interpretation, null],
          family: ["dp-value", algorithm.family || "", null],
          headline: ["dp-value", window.dpHeadlineText(reading), String(reading[0])],
        };
        var wanted = columns.map(function (name) { return specs[name]; }).filter(Boolean);
        if (line.children.length !== wanted.length) {
          line.textContent = "";
          wanted.forEach(function () { line.appendChild(document.createElement("td")); });
        }
        wanted.forEach(function (spec, index) {
          var cell = line.children[index];
          if (cell.className !== spec[0]) cell.className = spec[0];
          /* The first cell may carry a plotted curve's swatch before its text
             and a class badge after it: the text is the cell's one text node,
             wherever the swatch put it. */
          var text = null;
          for (var node = cell.firstChild; node; node = node.nextSibling) {
            if (node.nodeType === 3) { text = node; break; }
          }
          if (!text) {
            text = document.createTextNode("");
            var swatch = cell.querySelector(".dp-series-marker");
            cell.insertBefore(text, swatch ? swatch.nextSibling : cell.firstChild);
          }
          if (text.data !== spec[1]) text.data = spec[1];
          if (spec[2]) { if (cell.title !== spec[2]) cell.title = spec[2]; }
          else if (cell.hasAttribute("title")) cell.removeAttribute("title");
        });
        /* A class the roster does not show unasked carries it on the row a
           reader revealed, so a weaker claim is never read as an ordinary one. */
        var first = line.firstElementChild;
        var badge = first.querySelector(".dp-badge");
        if (visibleClasses.indexOf(algorithm.interpretation) === NOT_FOUND) {
          if (!badge) {
            badge = document.createElement("span");
            badge.className = "dp-badge dp-badge--" + algorithm.interpretation;
            badge.textContent = algorithm.interpretation;
            var reading = window.dpInterpretationText(algorithm.interpretation);
            if (reading) badge.title = reading;
            first.appendChild(badge);
          }
        } else if (badge) {
          badge.remove();
        }
        if (body.children[position] !== line) body.insertBefore(line, body.children[position] || null);
      });
      Object.keys(existing).forEach(function (uuid) { existing[uuid].remove(); });
    }

    function syncTable(section) {
      var chart = section.querySelector(".dp-chart");
      if (!chart.data) return;
      var visible = {};
      chart.data.forEach(function (trace) {
        if (trace.visible !== false && trace.visible !== "legendonly") {
          visible[trace.legendgroup] = trace;
        }
      });
      section.querySelectorAll(".dp-leaderboard tbody").forEach(function (body) {
        var rows = [].slice.call(body.children);
        rows.forEach(function (row) {
          var hidden = !visible[row.dataset.uuid];
          if (row.hidden !== hidden) row.hidden = hidden;
        });
        rows.sort(function (left, right) {
          return Number(right.lastElementChild.title) - Number(left.lastElementChild.title);
        }).forEach(function (row, position) {
          if (body.children[position] !== row) body.insertBefore(row, body.children[position] || null);
        });
      });
      section.dispatchEvent(new CustomEvent('dp-standings'));
    }

    function updateQuantileReadout() {
      readout.textContent = state.quantile + "%";
      readout.title = String(state.quantile / 100);
    }

    /* Each drawn algorithm's reading at the quantile now chosen, from its
       exact curve over every trial, through the one ranking rule. */
    function readingsNow(section) {
      var readings = {};
      Object.keys(section.dpExact || {}).forEach(function (uuid) {
        var exact = section.dpExact[uuid];
        readings[uuid] = window.dpPanel.readingAt(exact.curve, state.quantile / 100, exact.reached);
      });
      return readings;
    }

    /* The outcomes view and the notebook read at the τ the chart carries. */
    function showThreshold() {
      activeSection().querySelector(".dp-chart").dataset.threshold = String(taus[state.threshold]);
    }

    function showQuantile() {
      updateQuantileReadout();
      var section = activeSection();
      section.querySelector(".dp-chart").dataset.quantile = String(state.quantile / 100);
      // The standings belong to the Share view; another view leaves them for
      // the render that shows it again.
      if (state.view !== "profiles") return;
      var readings = readingsNow(section);
      section.querySelectorAll(".dp-leaderboard tbody tr").forEach(function (row) {
        // Until the matrices arrive, the server-rendered table stays intact.
        var reading = readings[row.dataset.uuid];
        if (row.dpCurve && reading) {
          row.dataset.reading = JSON.stringify(reading.map(String));
          row.lastElementChild.textContent = window.dpHeadlineText(reading);
          row.lastElementChild.title = String(reading[0]);
        }
      });
      syncTable(section);
    }

    function render() {
      writeAddress();
      var section = activeSection();
      // Set again below once this render draws the profiles; until then a
      // threshold move has no curves of this reading to re-rank.
      section.dpCurves = null;
      var controls = controlsOf(section);
      /* The algorithm box may live in a modal off the page (Explore), so the
         section names it; before the legend connects there is no box yet. */
      var algorithmBox = section.dpAlgorithmFilters || section.querySelector(".dp-algorithm-filters");
      if (interpretation && algorithmBox) {
        algorithmBox.insertBefore(interpretation, algorithmBox.querySelector(".dp-roster-filters"));
        interpretation.hidden = false;
      }
      sections.forEach(function (each) {
        each.hidden = each !== section;
      });
      if (baselineStatus) { baselineStatus.hidden = true; baselineStatus.textContent = ''; }
      chrome.show(section);
      // chrome.show() just reset the baseline Definition link to this
      // metric's own default; re-apply the reader's actual choice, which may
      // read the library's baseline API instead.
      syncBaselineDefinition();
      updateViews();
      showThreshold();
      showQuantile();
      var panel = activePanel(section);
      var held = currentPanel && currentPanel.entry === panel ? currentPanel.payload : null;
      section.dataset.loading = String(!held);
      if (!held) {
        showInstanceCount(null);
        empty.hidden = true;
        loadMessage.replaceChildren(window.dpSiteCopySpan('explore.results_loading'));
        load();
        return;
      }
      loadMessage.textContent = '';
      retry.hidden = true;
      var picked = selection();
      section.dpSelectedInstances = new Set(picked.instances.map(function (column) {
        return index.instances[column].uuid;
      }));
      /* What every other view of this panel reads: the loaded matrices, the
         reader's selection, and the controls that decide how a cost is read. */
      section.dpState = {
        index: index,
        roster: roster,
        metric: section.dataset.metric,
        budget: panel.budget,
        panel: held,
        aggregation: state.aggregation,
        baseline: state.baseline,
        instances: picked.instances,
        rows: picked.rows,
        threshold: taus[state.threshold],
        quantile: state.quantile / 100,
      };
      showInstanceCount(picked.instances.length);
      paintHistograms();
      var blank = !picked.instances.length || !picked.rows.length;
      window.dpSiteCopyFill(empty, !picked.instances.length
        ? 'explore.empty_no_instances'
        : 'explore.empty_no_algorithms');
      empty.hidden = !blank || state.view !== 'profiles';
      section.classList.toggle("dp-empty-selection", blank && state.view === 'profiles');
      var vector = window.dpPanel.resolveBaseline(held, state.baseline, index);
      var baselineAlgorithmId = window.dpPanel.baselineAlgorithm(state.baseline);
      var unavailableBaseline = !!(baselineAlgorithmId && vector &&
        vector.every(function (denominator) { return denominator === null; }));
      // A missing denominator makes every view unavailable, not a view of
      // the previous baseline. Keep its controls available for recovery.
      section.classList.toggle('dp-unavailable-baseline', unavailableBaseline);
      if (unavailableBaseline) {
        baselineStatus.hidden = false;
        window.dpSiteCopyFill(baselineStatus, 'explore.baseline_no_trials');
        return;
      }
      page.dispatchEvent(new CustomEvent('dp-panel-ready', { detail: { section: section } }));
      if (blank && state.view === 'profiles') {
        publishEmpty(section, controls, panel);
        return;
      }
      section.dpShowingEmpty = false;
      if (state.view !== 'profiles') {
        document.dispatchEvent(new CustomEvent("dp-chart", { detail: { chart: controls.chart } }));
        return;
      }
      var traced = {};
      section.dpCurves = {};
      section.dpLabels = {};
      section.dpExact = {};
      picked.rows.forEach(function (row) {
        var uuid = roster[row].uuid;
        traced[row] = window.dpPanel.curve(
          held, row, picked.instances, vector, state.aggregation, taus, index.weights
        ).fractions;
        // The standings read every trial: each curve exactly, at every ratio.
        var exact = window.dpPanel.exactCurve(
          held, row, picked.instances, vector, state.aggregation, index.weights
        );
        section.dpExact[uuid] = { curve: exact, reached: window.dpPanel.reachedBy(exact) };
        section.dpCurves[uuid] = traced[row];
        section.dpLabels[uuid] = roster[row].algorithm;
      });
      var readings = readingsNow(section);
      var headline = Number(index.headline_quantile);
      var bodies = [].slice.call(
        section.querySelectorAll(".dp-budget:not([hidden]) .dp-leaderboard tbody")
      );
      bodies.forEach(function (body) { fillTable(body, picked.rows, readings); });
      /* Keep filtered algorithms available when a later selection restores them. */
      var series = roster.map(function (algorithm, row) {
        var exact = section.dpExact[algorithm.uuid];
        return {
          key: algorithm.uuid,
          label: algorithm.algorithm,
          slot: algorithm.slot,
          diagnostic: algorithm.diagnostic,
          core: algorithm.core,
          interpretation: algorithm.interpretation,
          marker: algorithm.marker,
          hidden: !(row in traced),
          fractions: traced[row] || new Float64Array(taus.length),
          headline: exact ? window.dpPanel.readingAt(exact.curve, headline, exact.reached) : null,
          reached: exact ? exact.reached : null,
        };
      });
      series.sort(function (left, right) {
        if (left.hidden !== right.hidden) return left.hidden ? 1 : -1;
        var order = left.hidden ? 0 : window.dpPanel.compareReadings(readings[left.key], readings[right.key]);
        if (order) return order;
        return left.label < right.label ? -1 : left.label > right.label ? 1 : 0;
      });
      /* Top N: the best-standing competitors at this threshold among those the
         filters keep, the baseline already excluded from the rows. */
      chrome.selectTop(section, series.filter(function (entry) {
        return !entry.hidden;
      }).map(function (entry) { return entry.key; }));
      publish(section, controls, panel, series);
    }

    /* The chart payload for one budget, written beside its chart and drawn. */
    function publish(section, controls, panel, series) {
      var tag = section.querySelector(
        'script[data-profile-chart="' + controls.chart + '"][data-budget="' + panel.budget + '"]'
      );
      if (!tag) {
        tag = document.createElement("script");
        tag.type = "application/json";
        tag.dataset.profileChart = controls.chart;
        tag.dataset.budget = panel.budget;
        section.appendChild(tag);
      }
      tag.dpPayload = window.dpPanel.figure({
        title: panel.budget,
        series: series,
        taus: taus,
        aggregation: state.aggregation,
        leading: roster.length, // Visibility comes from the shared selection.
        units: window.dpPanel.units(index, section.dataset.metric, state.baseline),
        guide: state.quantile / 100,
        quantile: Number(index.headline_quantile),
        tauScale: index.tau_scale,
      });
      document.dispatchEvent(
        new CustomEvent("dp-chart", { detail: { chart: controls.chart } })
      );
    }

    /* Nothing to read: every curve hidden and the standings empty, so the
       chart stays on screen at its own scale instead of keeping the last
       reading or vanishing. */
    function publishEmpty(section, controls, panel) {
      // An emptied chart returns to its own scale rather than a reader's zoom.
      var chart = document.getElementById(controls.chart);
      if (chart && !section.dpShowingEmpty) chart.dpViewRevision = (chart.dpViewRevision || 0) + 1;
      section.dpShowingEmpty = true;
      section.dpCurves = {};
      section.dpLabels = {};
      section.dpExact = {};
      section.querySelectorAll(".dp-budget:not([hidden]) .dp-leaderboard tbody")
        .forEach(function (body) { fillTable(body, [], {}); });
      publish(section, controls, panel, roster.map(function (algorithm) {
        return {
          key: algorithm.uuid,
          label: algorithm.algorithm,
          slot: algorithm.slot,
          diagnostic: algorithm.diagnostic,
          core: algorithm.core,
          interpretation: algorithm.interpretation,
          marker: algorithm.marker,
          hidden: true,
          fractions: new Float64Array(taus.length),
        };
      }));
    }

    function load() {
      var section = activeSection();
      var panel = activePanel(section);
      var available = panel.baselines || {};
      // A named radio this budget carries no denominator for is disabled; the
      // algorithm radio is left alone here — its own budget carries no denial
      // to check, only whether it was ever measured, which render() reads
      // from the run matrix itself once loaded.
      namedBaselines().forEach(function (option) {
        option.disabled = option.value !== 'none' &&
          !Object.prototype.hasOwnProperty.call(available, option.value);
      });
      var chosen = window.dpPanel.baselineAlgorithm(state.baseline);
      var known = chosen
        ? roster.some(function (algorithm) { return algorithm.uuid === chosen; })
        : state.baseline === 'none' || Object.prototype.hasOwnProperty.call(available, state.baseline);
      if (!known) {
        state.baseline = index.baselines[state.metric].default;
        baselineFieldset.value = state.baseline;
        syncBaselineDefinition();
      }
      slider.value = String(state.quantile);
      if (currentPanel && currentPanel.entry === panel) {
        render();
        return;
      }
      section.dataset.loading = 'true';
      retry.hidden = true;
      if (pendingPanel === panel) return;
      pendingPanel = panel;
      // Another Metric's section still on screen would read as this one's
      // result while its panel downloads: show this section, loading, now.
      // render() finds the panel pending and starts no second fetch.
      if (sections.some(function (each) { return each !== section && !each.hidden; })) render();
      window.dpPanel.panelOf(index, panel).then(function (payload) {
        if (section === activeSection() && panel === activePanel(activeSection())) {
          currentPanel = { entry: panel, payload: payload };
          render();
          warmBudgets();
        }
      }).catch(function (error) {
        console.error('Explore could not load ' + panel.runs + ': ' + error.message);
        if (section === activeSection() && panel === activePanel(section)) {
          loadMessage.replaceChildren(window.dpSiteCopySpan('explore.results_load_error', { error: error.message }));
          retry.hidden = false;
        }
      }).finally(function () { if (pendingPanel === panel) pendingPanel = null; });
    }

    page.querySelectorAll('.dp-chip[data-filter="class"]').forEach(function (chip) {
      chip.addEventListener("click", function () {
        if (state.classes.has(chip.dataset.value)) state.classes.delete(chip.dataset.value);
        else state.classes.add(chip.dataset.value);
        chip.setAttribute("aria-pressed", String(state.classes.has(chip.dataset.value)));
        render();
      });
    });
    baselineFieldset.addEventListener('change', function () {
      // "Algorithm…" opens the search and leaves the current baseline chosen
      // until the reader picks an algorithm.
      if (baselineFieldset.value === baselineAlgorithmTrigger.value) {
        baselineFieldset.value = state.baseline;
        document.getElementById('baseline-picker').showPopover();
        return;
      }
      state.baseline = baselineFieldset.value;
      baselineAlgorithmClear.hidden = !window.dpPanel.baselineAlgorithm(state.baseline);
      syncBaselineDefinition();
      render();
    });

    /* "Algorithm…" opens the same typeahead ranking as the page's own Add…
       search (exact label, then a core prefix, then anywhere in the label),
       in a choose-baseline role: picking one never touches the plotted set
       or its five-pick cap, and the reader's roster is the whole roster, not
       only what happens to be plotted. */
    function showAlgorithmBaseline(uuid) {
      var algorithm = roster.filter(function (entry) { return entry.uuid === uuid; })[0];
      baselineAlgorithmOption.hidden = false;
      baselineAlgorithmOption.value = 'algorithm:' + uuid;
      var label = window.dpQualifiedAlgorithmLabel(algorithm.algorithm, algorithm.interpretation);
      baselineAlgorithmOption.textContent = window.dpSiteCopy('explore.baseline_algorithm_option', { algorithm: label });
      baselineFieldset.value = baselineAlgorithmOption.value;
      baselineAlgorithmClear.hidden = false;
      state.baseline = baselineFieldset.value;
      syncBaselineDefinition();
    }
    function commitBaseline(uuid) {
      if (!roster.some(function (entry) { return entry.uuid === uuid; })) return;
      showAlgorithmBaseline(uuid);
      render();
    }
    window.dpAlgorithmPicker(document.getElementById('baseline-picker'), {
      // The select anchors the search and takes focus back when it closes.
      trigger: baselineFieldset,
      algorithms: function () {
        return roster.map(function (entry) {
          return {
            id: entry.uuid, label: entry.algorithm, core: entry.core,
            class: entry.interpretation, transformations: entry.transformations,
          };
        });
      },
      choose: commitBaseline,
    });
    if (baselineAlgorithmClear) {
      baselineAlgorithmClear.addEventListener('click', function () {
        baselineAlgorithmOption.hidden = true;
        baselineAlgorithmOption.value = '';
        baselineAlgorithmClear.hidden = true;
        state.baseline = index.baselines[state.metric].default;
        baselineFieldset.value = state.baseline;
        syncBaselineDefinition();
        render();
      });
    }
    chrome.top.addEventListener('change', render);
    trialAggregation.addEventListener('change', function () {
      state.aggregation = trialAggregation.value;
      load();
    });
    /* showThreshold's per-row rewrite, table re-sort and the chart's
       guide-line move share the range filters' per-drag cost; the readout
       stays live on every event and the rest is coalesced the same way,
       flushing on release (`change`). "Show top N" means the top N at this
       threshold: the curves do not depend on it, so the move re-ranks them
       and flips visibility, as the Leaderboard does, and redraws only a
       chart not drawn yet. */
    var scheduleQuantileMove = coalesced(updateQuantileReadout, function () {
      writeAddress();
      showQuantile();
      var section = activeSection();
      // The curves entering the shown set are drawn first, so the recolour
      // the quantile announces reaches them too.
      var redraw = state.view === "profiles" && section.dpCurves && !window.dpRerank(
        section, chrome, document.getElementById(controlsOf(section).chart),
        readingsNow(section), section.dpLabels
      );
      section.dispatchEvent(
        new CustomEvent("dp-quantile", {
          bubbles: true,
          detail: { p: state.quantile / 100 },
        })
      );
      if (redraw) render();
    });
    slider.addEventListener("input", function () {
      state.quantile = Number(slider.value);
      scheduleQuantileMove();
    });
    slider.addEventListener("change", function () {
      if (scheduleQuantileMove.flush) scheduleQuantileMove.flush();
    });
    // The chart's tab asks for a quantile; the page reads the whole percent
    // nearest it through the same path as the slider.
    page.addEventListener("dp-quantile-seek", function (event) {
      var percent = Math.round(event.detail.value * 100);
      if (percent === state.quantile) return;
      slider.value = String(percent);
      state.quantile = percent;
      scheduleQuantileMove();
    });
    // The outcomes view's own slider asks for a τ on the page's grid.
    var scheduleThresholdMove = coalesced(null, function () {
      writeAddress();
      showThreshold();
      activeSection().dispatchEvent(
        new CustomEvent("dp-threshold", {
          bubbles: true,
          detail: { tau: taus[state.threshold] },
        })
      );
    });
    page.addEventListener("dp-threshold-seek", function (event) {
      var want = event.detail.tau;
      var best = 0;
      taus.forEach(function (tau, index) {
        if (Math.abs(Number(tau) - want) < Math.abs(Number(taus[best]) - want)) best = index;
      });
      if (best === state.threshold) return;
      state.threshold = best;
      scheduleThresholdMove();
    });
    page.addEventListener("dp-threshold-settle", function () {
      if (scheduleThresholdMove.flush) scheduleThresholdMove.flush();
      if (scheduleQuantileMove.flush) scheduleQuantileMove.flush();
    });
    // Explore draws the new budget once, from its own recomputed curves.
    page.addEventListener("dp-budget-change", function (event) {
      event.preventDefault();
      load();
    });
    page.addEventListener("dp-roster-change", function () { render(); });
    sections.forEach(function (section) {
      section.querySelector(".dp-chart").addEventListener("dp-chart-rendered", function () {
        syncTable(section);
      });
    });

    /* One range as the notebook takes it: inclusive endpoints, with null for
       an end resting on the range's own limit, since JSON carries no infinity. */
    function bounds(filter) {
      var low = filter.values[filter.low];
      var high = filter.values[filter.high];
      return [
        Number.isFinite(low) && filter.low > 0 ? low : null,
        Number.isFinite(high) && filter.high < filter.values.length - 1 ? high : null,
      ];
    }

    /* The reading a reader's own notebook opens at: the controls above, the
       algorithms they keep, the curves shown and the instance filters. */
    if (window.dpNotebookDownload) window.dpNotebookDownload(page, function () {
      var picked = selection();
      var filters = {};
      Object.keys(ranges).forEach(function (axis) { filters[axis] = bounds(ranges[axis]); });
      var reading = {
        TASK: index.task,
        METRIC: state.metric,
        BUDGET: activePanel(activeSection()).budget,
        TAU: Number(taus[state.threshold]),
        P: state.quantile / 100,
        BASELINE: state.baseline,
        PROTOCOL_BASELINE: index.baselines[state.metric].default,
        AGGREGATION: state.aggregation,
        ALGORITHMS: picked.rows.map(function (row) { return roster[row].uuid; }),
        PLOTTED_ALGORITHMS: picked.rows.filter(function (row) {
          return page.dpSelectedAlgorithms[roster[row].uuid] &&
            activeSection().dpRosterMatches(roster[row]);
        }).map(function (row) { return roster[row].uuid; }),
        INSTANCE_FILTERS: filters,
        // The open view decides which notebook is handed over; a property
        // view adds what its own chart draws.
        VIEW: state.view,
      };
      var comparison = state.view === 'profiles' ? null : activeSection().querySelector('.dp-comparison');
      return comparison && comparison.dpNotebookSelection
        ? Object.assign(reading, comparison.dpNotebookSelection())
        : reading;
    });

    syncBaselineDefinition();
    page.classList.add("is-interactive");
    load();
  }

  function init() {
    document.querySelectorAll("[data-playground]").forEach(connect);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
